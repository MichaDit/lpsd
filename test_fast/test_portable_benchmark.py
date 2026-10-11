"""Resource, deadline, public-payload and tiny numerical integration contracts."""
from __future__ import annotations

import base64
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from lpsd_benchmark import SCHEMA
from lpsd_benchmark import privacy, resources, runner
from lpsd_benchmark.worker import generate_white, isolated_tone_metrics, noise_line_mask, resolve_point_plan


def report_fixture():
    return {
        "schema": SCHEMA, "state": "completed", "source_commit": "a" * 40,
        "environment": {"os": "Linux", "process_arch": "aarch64", "host_arch": "aarch64",
                        "ram_total_mib": 4096, "logical_cpus_available": 4,
                        "versions": {"python": "3.12.14", "numpy": "2.3.5"}},
        "settings": {"sizes": [100_000], "points": [], "point_factors": [1., 2.],
                     "base_points": 1000, "max_seconds": 86400., "storage": "auto"},
        "elapsed_seconds": 3.0,
        "cases": [{"n": 100_000, "target_points": 2425, "actual_points": 938,
                   "point_factor": 2., "base_actual_points": 469, "requested_output_points": 938,
                   "point_count_resolved": True, "method": "fftw", "signal": "white",
                   "storage": "ram", "status": "completed", "reason": "",
                   "compute_wall_s": .2, "process_wall_s": 1.2, "compute_cpu_s": .3,
                   "psd_sha256": "b" * 64, "fftw_version": "3.3.5",
                   "noise": {"normalized_nsd_std": .02}}],
    }


def test_allowlist_reconstructs_payload_and_rejects_identity_values():
    report = report_fixture()
    report.update(hostname="midi-laptop", token="ghp_secret", spectra=[{"user_data": "private"}])
    report["environment"].update(username="owner", os="midi-laptop", kernel="owner-built", mac="aa:bb:cc:dd:ee:ff")
    report["environment"]["versions"]["numpy"] = "2.3.5+owner@example.com"
    row = report["cases"][0]
    row.update(error="/home/owner/private.csv", token="ghp_secret", output_sha256="owner@example.com",
               native_version="1.2.3+owner", compute_cpu_s=float("nan"))
    row["noise"]["username"] = "owner"
    clean = privacy.sanitize(report)
    serialized = json.dumps(clean)
    assert clean["environment"]["os"] == "Other"
    assert "numpy" not in clean["environment"]["versions"]
    assert "compute_cpu_s" not in clean["cases"][0]
    assert "owner" not in serialized and "ghp_" not in serialized and "midi-laptop" not in serialized
    assert "spectra" not in clean
    assert clean["cases"][0]["actual_points"] == 938
    assert clean["cases"][0]["target_points"] == 2425
    assert privacy.sanitize(clean) == clean


@pytest.mark.parametrize("bad", [None, [], {}, {"schema": "other"}, {"schema": SCHEMA, "cases": [None] * 513}])
def test_unknown_or_oversized_report_rejected(bad):
    with pytest.raises(ValueError):
        privacy.sanitize(bad)


def test_issue_body_retains_all_cases_with_bounded_compression():
    report = report_fixture()
    report["cases"] *= 300
    body = privacy.issue_body(report)
    assert len(body.encode()) <= privacy.MAX_ISSUE_BYTES
    assert "gzip + base64" in body
    encoded = body.split("```text\n")[1].split("\n```")[0]
    decoded = json.loads(gzip.decompress(base64.b64decode(encoded)))
    assert len(decoded["cases"]) == 300
    assert decoded == privacy.sanitize(report)


def test_submit_prepares_without_transport_until_explicit_acceptance(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(report_fixture()))
    def forbidden(*args, **kwargs):
        pytest.fail("No transport is permitted without public-upload opt in")
    result = privacy.submit(path, accepted=False, transport=forbidden, find_gh=forbidden)
    assert result["status"] == "prepared"
    assert result["body_path"].is_file()


def test_submit_only_sends_prepared_allowlisted_body_and_uses_issue_not_push(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TELEMETRY", "log")
    monkeypatch.setenv("GH_DEBUG", "api")
    monkeypatch.setenv("GH_FORCE_TTY", "100%")
    path = tmp_path / "result.json"
    report = report_fixture()
    report["hostname"] = "not-for-upload"
    path.write_text(json.dumps(report))
    calls = []
    def transport(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["auth", "status"]:
            return SimpleNamespace(returncode=0, stdout="account information stays local", stderr="")
        body = Path(command[command.index("--body-file") + 1]).read_text()
        assert "not-for-upload" not in body
        return SimpleNamespace(returncode=0, stdout="https://github.com/MichaDit/lpsd/issues/123\n", stderr="private log")
    result = privacy.submit(path, True, transport=transport, find_gh=lambda _: "gh-test")
    assert result["status"] == "submitted"
    command, settings = calls[1]
    assert calls[0][0][1:3] == ["auth", "status"]
    assert command[:3] == ["gh-test", "issue", "create"]
    assert "push" not in command and "pr" not in command
    assert settings["env"]["GH_HOST"] == "github.com"
    assert settings["env"]["GH_TELEMETRY"] == "0"
    assert settings["env"]["DO_NOT_TRACK"] == "1"
    assert settings["env"]["GH_NO_UPDATE_NOTIFIER"] == "1"
    assert settings["env"]["GH_NO_EXTENSION_UPDATE_NOTIFIER"] == "1"
    assert "GH_DEBUG" not in settings["env"] and "GH_FORCE_TTY" not in settings["env"]
    assert "private log" not in json.dumps({k: str(v) for k, v in result.items()})


def test_submit_never_echoes_failed_transport_output(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(report_fixture()))
    response = SimpleNamespace(returncode=1, stdout="owner@example.com", stderr="ghp_secret /home/owner")
    result = privacy.submit(path, True, transport=lambda *a, **k: response, find_gh=lambda _: "gh")
    assert result["reason"] == "github_login_unavailable"
    assert "owner" not in str(result) and "ghp_secret" not in str(result)


def test_lost_issue_receipt_is_uncertain_and_not_retried(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(report_fixture()))
    calls = []
    def transport(command, **kwargs):
        calls.append(command)
        if command[1] == "auth":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    result = privacy.submit(path, True, transport=transport, find_gh=lambda _: "gh")
    assert result["status"] == "uncertain"
    assert len(calls) == 2


def test_upload_preserves_actual_disk_phase_names():
    report = report_fixture()
    report["cases"][0]["phases"] = {
        "blocked_fftw_s": 100., "ordered_one_sided_power_s": 20.,
        "window_and_normalization_s": 3., "debug_path": "/home/owner/private"}
    result = privacy.sanitize(report)["cases"][0]["phases"]
    assert result == {"blocked_fftw_s": 100., "ordered_one_sided_power_s": 20., "window_and_normalization_s": 3.}


def test_deadline_is_monotonic_and_includes_bootstrap_setup():
    clock = [100.]
    deadline = resources.Deadline(86400., external_epoch=10030.,
                                  monotonic=lambda: clock[0], wallclock=lambda: 10000.)
    assert deadline.remaining() == 30.
    clock[0] += 31.
    assert deadline.remaining() == 0
    with pytest.raises(ValueError):
        resources.Deadline(86400.1)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "", "ram"])
def test_bad_ram_disk_budget_rejected(value):
    with pytest.raises(ValueError):
        resources.budget_bytes(value)


def test_explicit_32gib_cannot_turn_8gib_machine_into_32gib(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "system_memory", lambda: (8 * resources.GIB, 6 * resources.GIB))
    monkeypatch.setattr(resources, "cpu_capacity", lambda: (4, 4.))
    detected = resources.detect_resources(tmp_path, ram="32", disk="1", threads=16)
    assert detected.ram_budget <= 6 * resources.GIB - int(8 * resources.GIB * .1)
    assert detected.ram_budget < 8 * resources.GIB
    assert detected.threads == 4
    assert detected.disk_budget <= resources.GIB


def test_explicit_budgets_can_use_available_space_beyond_auto_but_keep_reserves(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "system_memory", lambda: (32 * resources.GIB, 31 * resources.GIB))
    monkeypatch.setattr(resources, "cpu_capacity", lambda: (8, 8.))
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=25_760_000_000))
    automatic = resources.detect_resources(tmp_path)
    explicit = resources.detect_resources(tmp_path, ram="32", disk="24")
    assert explicit.ram_budget == 29 * resources.GIB
    assert automatic.ram_budget < explicit.ram_budget
    assert automatic.disk_budget < resources.required_disk(1_000_000_000, "fftw") < explicit.disk_budget
    assert explicit.disk_budget <= 25_760_000_000 - 512 * resources.MIB
    tiny = resources.detect_resources(tmp_path, ram="1e-20", disk="1e-20")
    assert tiny.ram_budget == tiny.disk_budget == 0


def test_small_arm_budget_prevents_giant_files_and_keeps_small_ram_case():
    limit = resources.Resources(1024 * resources.MIB, 800 * resources.MIB,
                                4 * resources.GIB, 600 * resources.MIB,
                                2 * resources.GIB, 4, 4.)
    small = resources.select_storage(100_000, "matched", "white", "auto", limit)
    huge = resources.select_storage(1_000_000_000, "fftw", "white", "auto", limit)
    assert small[0] == "ram" and small[3] is None
    assert huge[3] == "disk_budget"


def test_in_place_auto_can_admit_1b_fftw_with_32gib_allocation_budget():
    n = 1_000_000_000
    # A deliberately tight 26-GiB budget fits the 24N layout, not 32N.
    chosen = resources.choose_fft_layout(n, "fftw", "white", "auto", 26 * resources.GIB)
    assert chosen == "in-place"
    assert resources.required_ram(n, "fftw", fft_layout="out-of-place") > 26 * resources.GIB
    assert resources.required_ram(n, "fftw", fft_layout=chosen) < 26 * resources.GIB
    assert resources.choose_fft_layout(100_000, "fftw", "white", "auto", 32 * resources.GIB) == "out-of-place"


def test_disk_native_requires_only_input_file_and_fftw_checks_workspace_minimum():
    assert resources.required_disk(1_000_000_000, "lpsd") == 8_000_000_000 + 16 * resources.MIB
    assert resources.required_disk(1_000_000_000, "fftw") == 24_000_000_000 + 16 * resources.MIB


def test_forecast_accounts_for_points_storage_and_no_giant_uncalibrated_start():
    previous = {"status": "completed", "method": "matched", "signal": "white",
                "storage": "ram", "n": 100_000, "target_points": 1000, "point_factor": 1., "process_wall_s": 2.}
    case = {"method": "matched", "signal": "white", "storage": "disk",
            "n": 1_000_000, "target_points": 1000, "point_factor": 2.}
    assert resources.predict_seconds(case, []) is None
    predicted = resources.predict_seconds(case, [previous])
    assert predicted > 2 * 10 * 2 * 8
    same = dict(case, storage="ram", point_factor=1.)
    assert predicted > resources.predict_seconds(same, [previous])


def test_default_matrix_includes_historical_sizes_signals_and_output_factors():
    args = runner.parser().parse_args(["run"])
    matrix = runner._matrix(args)
    assert len(matrix) == 132
    assert {c["n"] for c in matrix} == set(runner.DEFAULT_SIZES)
    assert {c["point_factor"] for c in matrix} == {1., 2.}
    assert {c["signal"] for c in matrix if c["n"] == 100_000} == set(runner.SIGNALS)
    assert {c["signal"] for c in matrix if c["n"] == 1_000_000} == set(runner.SIGNALS)
    assert {c["signal"] for c in matrix if c["n"] == 1_000_003} == set(runner.SIGNALS)
    assert all(c["signal"] == "white" for c in matrix if c["n"] > 1_000_003)


def test_quality_only_1m_gets_each_signal_calibrated_at_100k():
    args = runner.parser().parse_args(["run", "--sizes", "1000000", "--methods", "lpsd"])
    matrix = runner._matrix(args)
    calibrations = [row for row in matrix if row["case_role"] == "calibration"]
    assert len(calibrations) == 6
    assert {row["signal"] for row in calibrations} == set(runner.SIGNALS)
    assert all(row["n"] == 100_000 for row in calibrations)


def test_large_only_request_gets_small_calibration_first():
    args = runner.parser().parse_args(["run", "--sizes", "1000000000", "--methods", "lpsd"])
    matrix = runner._matrix(args)
    assert matrix[0]["n"] == 100_000 and matrix[0]["case_role"] == "calibration"
    assert all(c["n"] == 1_000_000_000 for c in matrix[1:])


def test_original_planner_resolves_exact_doubled_output_without_changing_base_grid():
    from lpsd_fast import api
    base, base_info = resolve_point_plan(api, 100_000, 50., 1000, 1., 1000, 100)
    double, double_info = resolve_point_plan(api, 100_000, 50., 1000, 2., 1000, 100)
    assert base_info["target_points"] == 1000
    assert len(base[0]) == 469
    assert len(double[0]) == 2 * len(base[0])
    assert double_info["target_points"] == 2425
    assert double_info["requested_output_points"] == double_info["actual_points"]


def test_bounded_white_generation_matches_historical_signal_across_chunk_boundary():
    import numpy as np
    from benchmarks.fftw_comparison_signals import signal
    expected, _ = signal("white", 262145, 50., 20261010)
    actual = generate_white(262145, 50., 20261010)
    np.testing.assert_array_equal(actual, expected)


def test_noise_line_mask_matches_native_enbw_and_removes_all_mixed_lines():
    import numpy as np
    from benchmarks.bench_fftw import KaiserWindow
    from benchmarks.fftw_comparison_metrics import summarize_noise
    from lpsd_fast import api
    plan, _ = resolve_point_plan(api, 65536, 50., 1000, 1., 1000, 100)
    reference = api.lpsd(np.zeros(65536), 50., n_frequencies=1000, n_averages=100,
                         psll=200., outputs=("enbw",), workers=1, kernel="fast", window_cache_mb=0)
    frequencies = reference.index.to_numpy()
    tones = [{"frequency_hz": frequency} for frequency in (.12345, 3.123456, 12.34567)]
    expected = np.zeros(len(frequencies), dtype=bool)
    for tone in tones:
        expected |= np.abs(frequencies - tone["frequency_hz"]) <= 3 * reference.enbw.to_numpy().real
    actual = noise_line_mask(frequencies, plan[3], 50., tones,
                             KaiserWindow("native", Path(api._LIB._name)), api._kaiser_alpha(200.) * np.pi)
    np.testing.assert_array_equal(actual, expected)
    assert actual.any() and (~actual).any()
    nsd = np.where(actual, 1000., 1.)
    masked = summarize_noise(frequencies, nsd, np.ones(len(frequencies)), exclude=actual)["overall"]
    unmasked = summarize_noise(frequencies, nsd, np.ones(len(frequencies)))["overall"]
    assert masked["normalized_nsd_std"] == 0.
    assert masked["mean_nsd_over_source"] == 1.
    assert unmasked["normalized_nsd_std"] > 100.


def test_peak_width_is_reserved_for_isolated_tone_and_sparse_grid_is_undefined():
    import numpy as np
    frequencies = np.array([.01, .10, .12, .14, .16, .20, .30, 1.])
    psd = np.array([1000., 0., 1., 4., 1., 0., 0., 0.])
    tones = [{"frequency_hz": .14, "rms_v": 1.}]
    assert isolated_tone_metrics("mixed_tones", frequencies, psd, tones) == {}
    isolated = isolated_tone_metrics("offbin_tone", frequencies[1:], psd[1:], tones)
    assert isolated["peak_label_hz"] == .14
    assert isolated["psd_fwhm_hz"] > 0 and isolated["nsd_fwhm_hz"] > 0
    assert isolated_tone_metrics("offbin_tone", np.array([2., 3.]), np.ones(2), tones) == {}


def test_timeout_kills_owned_process_and_is_not_a_completed_timing(tmp_path):
    case = {"storage": "ram", "ram_budget_bytes": 1024 * resources.MIB,
            "disk_budget_bytes": 1024 * resources.MIB}
    deadline = resources.Deadline(10.)
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    started = time.monotonic()
    result = runner.execute_case(case, tmp_path, deadline, .15, command=command, interval=.02)
    assert time.monotonic() - started < 4.
    assert result["status"] == "timeout"
    assert result["reason"] == "time_limit"
    assert "compute_wall_s" not in result


@pytest.mark.parametrize("children_file", [True, False])
def test_linux_memory_resolves_proc_namespace_and_rejects_unrelated_pid_trees(tmp_path, children_file):
    def status(pid, parent, nested, rss, anonymous):
        directory = tmp_path / str(pid)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "status").write_text(
            f"Pid:\t{pid}\nPPid:\t{parent}\nNSpid:\t{' '.join(map(str, nested))}\n"
            f"VmRSS:\t{rss} kB\nRssAnon:\t{anonymous} kB\n")
    def children(pid, values):
        task = tmp_path / str(pid) / "task" / str(pid)
        task.mkdir(parents=True, exist_ok=True)
        (task / "children").write_text(" ".join(map(str, values)))
    status("self", 1, (2000, 5), 0, 0)
    (tmp_path / "self" / "status").write_text("Pid:\t2000\nPPid:\t1\nNSpid:\t2000 5\n")
    status(6, 5000, (6, 6), 9000, 9000)  # /proc/Popen.pid would hit unrelated memory.
    status(3000, 2000, (3000, 6), 1000, 600)
    status(3100, 3000, (3100, 7), 400, 300)
    status(4000, 5000, (4000, 6), 9000, 9000)  # Same inner PID, different controller.
    status(5000, 2000, (5000, 8, 6), 9000, 9000)  # Match at the wrong namespace depth.
    if children_file:
        children(2000, (6, 3000, 4000, 5000))
    children(3000, (3100, 4000))
    reader = runner._LinuxMemoryReader(6, proc_root=tmp_path)
    assert reader.read() == (1400 * 1024, 900 * 1024)
    assert reader.visible_pid == 3000


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux procfs/PID namespace guard")
def test_linux_memory_guard_stops_real_child_allocation(tmp_path):
    case = {"storage": "ram", "ram_budget_bytes": 25 * resources.MIB,
            "disk_budget_bytes": resources.MIB}
    script = "import time; memory=bytearray(64 * 1024 ** 2); time.sleep(10)"
    result = runner.execute_case(case, tmp_path, resources.Deadline(5.), 4.,
                                 command=[sys.executable, "-c", script], interval=.02)
    assert result["status"] == "failed"
    assert result["reason"] == "memory_limit"
    assert result["peak_rss_bytes"] > case["ram_budget_bytes"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group contract; Windows uses taskkill /T")
def test_timeout_also_stops_grandchild(tmp_path):
    pid_path = tmp_path / "grandchild.pid"
    script = ("import subprocess,sys,time,pathlib; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
              f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(30)")
    case = {"storage": "ram", "ram_budget_bytes": 1024 * resources.MIB,
            "disk_budget_bytes": 1024 * resources.MIB}
    result = runner.execute_case(case, tmp_path, resources.Deadline(10.), .5,
                                 command=[sys.executable, "-c", script], interval=.02)
    assert result["status"] == "timeout"
    assert pid_path.exists()
    pid = int(pid_path.read_text())
    try:
        import psutil
        child = psutil.Process(pid)
        assert child.status() == psutil.STATUS_ZOMBIE or not child.is_running()
    except ImportError:
        state_path = Path(f"/proc/{pid}/stat")
        assert not state_path.exists() or state_path.read_text().split()[2] == "Z"
    except Exception as exc:
        if type(exc).__name__ != "NoSuchProcess":
            raise


def test_expired_bootstrap_deadline_starts_no_numerical_case(tmp_path, monkeypatch):
    monkeypatch.setenv("LPSD_BENCHMARK_DEADLINE_EPOCH", str(time.time() - 1))
    def forbidden(*args, **kwargs):
        pytest.fail("An expired setup-inclusive deadline must not start a case")
    monkeypatch.setattr(runner, "execute_case", forbidden)
    output = tmp_path / "results.json"
    assert runner.main(["run", "--sizes", "1000", "--methods", "lpsd", "--profile", "performance",
                        "--work-dir", str(tmp_path / "work"), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["state"] == "deadline"
    assert all(row["reason"] == "time_budget" for row in report["cases"])


def test_help_does_not_need_numpy_in_controller():
    # -S disables site packages. argparse help must remain usable on a machine
    # whose numerical installation is missing or unsupported.
    result = subprocess.run([sys.executable, "-S", "-m", "lpsd_benchmark", "run", "--help"],
                            cwd=runner.ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "--point-factors" in result.stdout
    assert "--accept-public-upload" in result.stdout


def test_native_worker_tiny_complete_case_matches_output_contract(tmp_path):
    case = {"n": 4096, "method": "lpsd", "signal": "white", "storage": "ram",
            "target_points": 100, "base_target_points": 100, "point_factor": 1.,
            "requested_threads": 1, "ram_budget_bytes": 512 * resources.MIB,
            "disk_budget_bytes": 512 * resources.MIB, "sample_rate_hz": 50.,
            "seed": 20261010, "n_averages": 100, "fft_memory_mb": 16., "low_working_mb": 32.}
    result = runner.execute_case(case, tmp_path, resources.Deadline(30.), 25.)
    assert result["status"] == "completed", result
    assert result["actual_points"] == len(result["_spectrum"]["frequency_hz"])
    assert result["nsd_matches_sqrt_psd"] and result["input_unchanged"]
    assert result["compute_wall_s"] < result["process_wall_s"]
    assert result["native_backend"] == "bounded"
    assert list(tmp_path.glob("*.f64")) == []
