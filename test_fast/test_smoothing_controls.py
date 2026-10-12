"""Portable smoothing selection, admission, provenance and complete-call gates.

The large lengths in scalar tests never allocate sample arrays. The final
integration cases create only tiny synthetic inputs in isolated processes.
"""
from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lpsd_benchmark import SCHEMA
from lpsd_benchmark import privacy, resources, runner, worker
from lpsd_benchmark.resources import GIB, MIB, smoothing_controls


def _case(*, method="matched", storage="ram", threads=8, request=None, ram=32 * GIB, n=1_000_000_000):
    return {"method": method, "storage": storage, "requested_threads": threads,
            "requested_smoothing_workers": request, "ram_budget_bytes": ram,
            "low_working_mb": 2048., "n": n}


@pytest.mark.parametrize("threads,smoothing_request,limit,extra_mib", [
    (1, None, 1, 0), (2, None, 2, 0), (8, None, 4, 0),
    (8, 1, 1, 0), (8, 4, 4, 0), (8, 8, 8, 36),
    (6, 8, 6, 18), (2, 8, 2, 0), (8, 100, 8, 36),
])
def test_request_is_capped_without_changing_the_default(threads, smoothing_request, limit, extra_mib):
    value = smoothing_controls("matched", threads, smoothing_request)
    assert value["requested_smoothing_workers"] == smoothing_request
    assert value["smoothing_workers_limit"] == limit
    assert value["smoothing_runtime_reserve_bytes"] == extra_mib * MIB
    assert value["smoothing_worker_policy"] == ("default_cap4" if smoothing_request is None else "explicit")
    assert value["smoothing_runtime_reserve_model"] == "incremental_9mib_above_cap4_v1"


@pytest.mark.parametrize("bad", [0, -1, True, False, 4.0, "8", 2 ** 63])
def test_invalid_json_request_is_rejected_before_numerical_imports(bad, monkeypatch, tmp_path):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "numpy" or name.startswith("benchmarks"):
            pytest.fail("An invalid scalar option must fail before numerical imports")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="positive integer"):
        worker.run(_case(request=bad), tmp_path)


@pytest.mark.parametrize("bad", [None, 0, -1, True, 2.0, "8"])
def test_thread_admission_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="positive integer"):
        smoothing_controls("matched", bad, 8)


@pytest.mark.parametrize("method", ["fftw", "lpsd"])
def test_other_methods_have_no_smoothing_work_or_reserve(method):
    case = _case(method=method, request=8)
    controls = smoothing_controls(method, 8, 8)
    assert controls["smoothing_worker_policy"] == "not_applicable"
    assert controls["requested_smoothing_workers"] is None
    assert controls["smoothing_workers_limit"] == controls["smoothing_runtime_reserve_bytes"] == 0
    assert worker._smoothing_result_fields(case) == {
        "smoothing_pool_capacity": 0, "smoothing_task_ranges": 0,
        "smoothed_frequency_points": 0, "smoothing_backend": "not_applicable"}
    assert resources.required_ram(100_000, method, smoothing_runtime_reserve_bytes=36 * MIB) == resources.required_ram(100_000, method)


@pytest.mark.parametrize("value", ["0", "-1", "4.5", "true", "nan", str(2 ** 63)])
def test_cli_rejects_invalid_smoothing_count(value):
    with pytest.raises(SystemExit) as error:
        runner.parser().parse_args(["run", "--smoothing-workers", value])
    assert error.value.code == 2


@pytest.mark.parametrize("changes", [
    {"smoothing_workers_limit": 16}, {"smoothing_runtime_reserve_bytes": 0},
    {"smoothing_workers_limit": True}, {"smoothing_worker_policy": "default_cap4"},
])
def test_worker_rejects_forged_limit_or_missing_reserve(changes):
    case = _case(request=8)
    case.update(smoothing_controls("matched", 8, 8))
    with pytest.raises(ValueError, match="Inconsistent"):
        worker._case_smoothing_controls({**case, **changes})


def test_ram_layout_and_window_admission_include_the_same_reserve():
    n, extra = 100_000_000, 36 * MIB
    out = resources.required_ram(n, "matched", fft_layout="out-of-place")
    full = resources.required_ram(n, "matched", fft_layout="in-place")
    bounded = resources.required_ram(n, "matched", fft_layout="in-place", window_storage="fft_input_then_blocked")
    for layout, window, previous in (("out-of-place", "full", out), ("in-place", "full", full),
                                      ("in-place", "fft_input_then_blocked", bounded)):
        assert resources.required_ram(n, "matched", fft_layout=layout, window_storage=window,
                                      smoothing_runtime_reserve_bytes=extra) == previous + extra
    assert resources.choose_fft_layout(n, "matched", "white", "auto", out) == "out-of-place"
    assert resources.choose_fft_layout(n, "matched", "white", "auto", out,
                                       smoothing_runtime_reserve_bytes=extra) == "in-place"
    assert resources.choose_window_storage(n, "matched", "white", "auto", "in-place", full) == "full"
    assert resources.choose_window_storage(n, "matched", "white", "auto", "in-place", full,
                                           smoothing_runtime_reserve_bytes=extra) == "fft_input_then_blocked"
    limit = bounded + extra - 1
    admitted = resources.Resources(32 * GIB, 32 * GIB, 32 * GIB, limit, 16 * GIB, 8, 8.)
    chosen = resources.select_storage(n, "matched", "white", "ram", admitted, "in-place",
                                      "fft_input_then_blocked", smoothing_runtime_reserve_bytes=extra)
    assert chosen[1] == bounded + extra and chosen[3] == "memory_budget"


def test_billion_samples_32gib_low_headroom_is_scalar_and_keeps_the_original_ceiling():
    original = _case()
    selected = _case(request=8)
    old_mb, old_headroom = worker._native_working_budget(original)
    new_mb, new_headroom = worker._native_working_budget(selected)
    assert old_headroom == 32 * GIB - 8_000_000_000 - 280 * MIB
    assert old_headroom - new_headroom == 36 * MIB
    assert old_mb - new_mb == 36
    assert new_mb * MIB == new_headroom
    assert selected["ram_budget_bytes"] == original["ram_budget_bytes"] == 32 * GIB
    needed = resources.required_ram(1_000_000_000, "matched", fft_layout="in-place",
                                    window_storage="fft_input_then_blocked", smoothing_runtime_reserve_bytes=36 * MIB)
    assert needed <= 32 * GIB


def test_disk_and_memory_limited_arm_preflight_do_not_spend_the_extra_headroom():
    extra = 36 * MIB
    admitted = resources.Resources(GIB, GIB, 32 * GIB, 384 * MIB + extra - 1, 32 * GIB, 8, 8.)
    assert resources.select_storage(100_000_000, "matched", "white", "disk", admitted)[3] is None
    selected = resources.select_storage(100_000_000, "matched", "white", "disk", admitted,
                                        smoothing_runtime_reserve_bytes=extra)
    assert selected[1] == 384 * MIB + extra and selected[3] == "memory_budget"
    case = _case(storage="disk", request=8, ram=(256 + 64 + 36) * MIB - 1)
    assert worker._native_working_budget(case) == (None, 64 * MIB - 1)
    case = _case(storage="ram", threads=2, request=8, n=100_000, ram=400 * MIB)
    assert worker._native_working_budget(case) == worker._native_working_budget({**case, "requested_smoothing_workers": None})


def test_linux_address_guard_does_not_grow_for_explicit_smoothing(monkeypatch):
    applied = []
    fake_resource = SimpleNamespace(RLIMIT_AS=9, RLIM_INFINITY=-1,
                                    getrlimit=lambda _: (-1, -1),
                                    setrlimit=lambda kind, value: applied.append((kind, value)))
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(worker.sys, "platform", "linux")
    def status(path, *args, **kwargs):
        assert path.as_posix() == "/proc/self/status"
        return "VmSize: 1000000 kB\nVmRSS: 100000 kB\n"
    monkeypatch.setattr(Path, "read_text", status)
    for request in (None, 8):
        worker._address_guard(_case(storage="disk", request=request, ram=2 * GIB, n=100_000))
    assert applied[0] == applied[1] == (9, (900000 * 1024 + 28 * 100_000 + 16 + 2 * GIB, -1))


@pytest.mark.parametrize("storage", ["ram", "disk"])
def test_returned_capacity_and_ranges_are_not_an_active_thread_measurement(storage):
    case = _case(storage=storage, request=8)
    metadata = {"requested_smoothing_workers": 8, "smoothing_pool_capacity": 8,
                "smoothing_task_ranges": 16, "smoothed_frequency_points": 40, "smoothing_backend": "native"}
    disk = {"smoothing": {"workers": 8, "pool_capacity": 8, "task_ranges": 16,
                           "high_points": 40, "backend": "native"}}
    result = worker._smoothing_result_fields(case, metadata, disk)
    assert result == {"smoothing_pool_capacity": 8, "smoothing_task_ranges": 16,
                      "smoothed_frequency_points": 40, "smoothing_backend": "native"}
    metadata["requested_smoothing_workers"] = 4
    disk["smoothing"]["workers"] = 4
    with pytest.raises(ValueError, match="differs from admission"):
        worker._smoothing_result_fields(case, metadata, disk)


def test_serial_and_empty_ram_smoothing_have_no_executor_capacity():
    case = _case(request=1)
    for points, ranges in ((0, 0), (1, 1)):
        metadata = {"requested_smoothing_workers": 1, "smoothing_pool_capacity": 0,
                    "smoothing_task_ranges": ranges, "smoothed_frequency_points": points,
                    "smoothing_backend": "native"}
        result = worker._smoothing_result_fields(case, metadata)
        assert result["smoothing_pool_capacity"] == 0 and result["smoothing_task_ranges"] == ranges


def test_public_schema_preserves_controls_and_does_not_invent_old_provenance():
    controls = smoothing_controls("matched", 8, 8)
    row = {"n": 100_000, "target_points": 1000, "method": "matched", "signal": "white",
           "storage": "ram", "status": "completed", **controls,
           "smoothing_pool_capacity": 8, "smoothing_task_ranges": 16,
           "smoothed_frequency_points": 200, "smoothing_backend": "native"}
    report = {"schema": SCHEMA, "state": "completed", "settings": controls, "cases": [row]}
    clean = privacy.sanitize(report)
    for key, value in row.items():
        if "smoothing" in key or key == "smoothed_frequency_points":
            assert clean["cases"][0][key] == value
    assert clean["settings"]["requested_smoothing_workers"] == 8
    assert privacy.sanitize(clean) == clean
    old = privacy.sanitize({"schema": SCHEMA, "cases": [{k:v for k,v in row.items()
                           if "smoothing" not in k and k != "smoothed_frequency_points"}]})
    assert not any("smoothing" in key for key in old["cases"][0])
    assert not any("smoothing" in key for key in old["settings"])


@pytest.mark.parametrize("bad", [-1, True, 1.5, "owner@example.com"])
def test_public_schema_rejects_non_integer_control_fields(bad):
    fields = {name: bad for name in ("requested_smoothing_workers", "smoothing_workers_limit",
              "smoothing_pool_capacity", "smoothing_task_ranges", "smoothed_frequency_points",
              "smoothing_runtime_reserve_bytes")}
    assert privacy._smoothing_fields(fields) == {}
    assert privacy._smoothing_fields({"smoothing_worker_policy": "owner@example.com",
                                     "smoothing_runtime_reserve_model": "private-host",
                                     "smoothing_backend": "token"}) == {}


@pytest.mark.parametrize("threads,limit,extra", [(2, 2, 0), (8, 8, 36 * MIB)])
def test_controller_forwards_cap_and_reserve_and_preserves_resource_limits(tmp_path, monkeypatch, threads, limit, extra):
    monkeypatch.delenv("LPSD_BENCHMARK_DEADLINE_EPOCH", raising=False)
    fixed = resources.Resources(4 * GIB, 3 * GIB, 8 * GIB, 2 * GIB, 4 * GIB, threads, float(threads))
    monkeypatch.setattr(runner, "detect_resources", lambda *a, **k: fixed)
    monkeypatch.setattr(runner, "environment", lambda _: {})
    monkeypatch.setattr(runner, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_source_dirty", lambda: False)
    seen = []
    def fake_case(case, directory, deadline, timeout):
        seen.append(case)
        assert case["requested_smoothing_workers"] == 8
        assert case["smoothing_workers_limit"] == limit
        assert case["smoothing_runtime_reserve_bytes"] == extra
        assert case["ram_budget_bytes"] == fixed.ram_budget
        assert case["disk_budget_bytes"] == fixed.disk_budget
        assert case["low_working_mb"] == (fixed.ram_budget - extra) / MIB / 3
        assert case["ram_required_bytes"] == resources.required_ram(100_000, "matched", smoothing_runtime_reserve_bytes=extra)
        return {"status": "completed", "compute_wall_s": .01, "process_wall_s": .1}
    monkeypatch.setattr(runner, "execute_case", fake_case)
    args = runner.parser().parse_args(["run", "--sizes", "100000", "--methods", "matched",
        "--point-factors", "1", "--profile", "performance", "--smoothing-workers", "8",
        "--threads", "8", "--work-dir", str(tmp_path / "work"), "--output", str(tmp_path / "report.json")])
    assert runner.run(args) == 0 and len(seen) == 1
    report = json.loads(args.output.read_text())
    assert report["settings"]["requested_smoothing_workers"] == 8
    assert report["cases"][0]["smoothing_workers_limit"] == limit


def test_prediction_does_not_reuse_a_different_smoothing_limit():
    case = {"n": 10_000_000, "method": "matched", "signal": "white", "target_points": 1000,
            "storage": "ram", "smoothing_workers_limit": 4}
    faster = {**case, "n": 100_000, "status": "completed", "process_wall_s": 1., "smoothing_workers_limit": 8}
    assert resources.predict_seconds(case, [faster]) is None
    assert resources.predict_seconds(case, [{**faster, "smoothing_workers_limit": 4}]) is not None


@pytest.mark.parametrize("storage,n,window,factor,signal", [
    ("ram", 4096, "full", 1., "white"), ("ram", 4093, "fft_input_then_blocked", 2., "dc_nanovolt"),
    ("disk", 4096, "blocked", 1., "white"), ("ram", 257, "full", 1., "white"),
])
def test_tiny_complete_matched_calls_forward_the_option_and_keep_spectra(tmp_path, storage, n, window, factor, signal):
    """Actual worker/pipeline integration; only a few thousand samples."""
    case = {"n": n, "method": "matched", "signal": signal, "storage": storage,
            "target_points": 100, "base_target_points": 100, "point_factor": factor,
            "requested_threads": 8, "ram_budget_bytes": 768 * MIB, "disk_budget_bytes": 256 * MIB,
            "sample_rate_hz": 50., "seed": 20261010, "n_averages": 100,
            "fft_memory_mb": 1., "low_working_mb": 64., "fft_layout": "in-place",
            "window_storage": window}
    results = []
    for request in (None, 8):
        chosen = {**case, **smoothing_controls("matched", 8, request)}
        directory = tmp_path / ("default" if request is None else "explicit")
        directory.mkdir()
        result = runner.execute_case(chosen, directory, resources.Deadline(45.), 40.)
        if result.get("reason") == "dependency_unavailable":
            pytest.skip("Optional FFTW/native runtime unavailable")
        assert result["status"] == "completed", result
        assert result["smoothing_workers_limit"] == (4 if request is None else 8)
        assert result["smoothing_runtime_reserve_bytes"] == (0 if request is None else 36 * MIB)
        assert result["smoothing_backend"] == "native"
        if n == 257:
            # N/16 < 32: the preserved low-mask makes every point native.
            assert result["smoothed_frequency_points"] == result["smoothing_pool_capacity"] == result["smoothing_task_ranges"] == 0
        else:
            assert 0 < result["smoothing_pool_capacity"] <= result["smoothing_workers_limit"]
            assert 0 < result["smoothing_task_ranges"] <= min(result["smoothed_frequency_points"], 2 * result["smoothing_workers_limit"])
        assert result["nsd_matches_sqrt_psd"] and result["input_unchanged"]
        assert not list(directory.glob("*.f64"))
        results.append(result)
    for key in ("input_sha256", "frequency_sha256", "psd_sha256", "nsd_sha256", "source_sha256"):
        assert results[0][key] == results[1][key]
    assert results[0]["_spectrum"] == results[1]["_spectrum"]
