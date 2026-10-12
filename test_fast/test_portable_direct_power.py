"""Direct disk selection, admission, measured readback and public boundaries.

Large-N checks are scalar only. Numerical routing uses at most 4096 inputs.
"""
import builtins
import json
from types import SimpleNamespace

import pytest

from lpsd_benchmark import SCHEMA, privacy, resources, runner, worker
from test_fast.test_native_working_controls import _case


def _direct_case(**changes):
    case = {**_case(method="fftw", n=4096, threads=1),
            "disk_fft_algorithm": "full", "disk_power_reduction": "direct-log",
            "fft_memory_mb": 128.}
    case.update(changes)
    return case


def _details(case, points=4):
    n = case["n"]
    return {"disk_power_reduction": "direct-log", "power_vector_created": False,
            "fft": {"n": n, "power_storage": "direct_log_groups"},
            "power_storage": {"mode": "direct_log_groups", "extra_power_file_bytes": 0,
                              "maximum_large_file_bytes": 24 * n, "workspace_power_offset_bytes": None},
            "direct_group_reduction": {"n": n, "groups": points, "abi_version": 3,
                "energy_bins": n // 2 + 1, "dc_bins": 1, "nyquist_bins": int(n % 2 == 0),
                "array_state_bytes": 48 * points + 48, "additional_workers": 0,
                "power_vector_created": False, "matrix_written": False, "ordinary_result_arrays": True},
            "phases": {"direct_log_groups_and_energy_s": .125}}


def test_cli_default_and_explicit_selection():
    assert runner.parser().parse_args(["run"]).disk_power_reduction == "materialized"
    assert runner.parser().parse_args(["run", "--disk-power-reduction", "direct-log"]).disk_power_reduction == "direct-log"
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["run", "--disk-power-reduction", "private-host"])


@pytest.mark.parametrize("bad", [True, None, [], {}, "unknown", "not_applicable"])
def test_invalid_selection_rejected_before_numerical_imports(tmp_path, monkeypatch, bad):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "numpy" or name.startswith("benchmarks"):
            pytest.fail("Invalid scalar selection must be rejected before numerical imports")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(ValueError, match="power reduction"):
        worker.run(_direct_case(disk_power_reduction=bad), tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changes", [
    {"n": 1_000_000_001}, {"disk_fft_algorithm": "packed-real"},
    {"sample_rate_hz": 51.}, {"n_averages": 99}, {"psll_db": 199.},
    {"fft_memory_mb": 127.},
])
def test_unqualified_profile_skips_before_imports_or_allocation(tmp_path, monkeypatch, changes):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "numpy" or name.startswith("benchmarks"):
            pytest.fail("Unsupported scalar profile must skip before numerical imports")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    result = worker.run(_direct_case(**changes), tmp_path)
    assert result["status"] == "skipped" and result["reason"] == "unsupported"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method,storage", [("fftw", "ram"), ("matched", "ram"),
                                              ("matched", "disk"), ("lpsd", "ram"), ("lpsd", "disk")])
def test_disk_option_is_not_applied_to_other_routes(method, storage):
    case = _direct_case(method=method, storage=storage)
    assert worker._case_disk_power_reduction(case) == "not_applicable"
    assert worker._disk_power_result_fields(case, None) == {"disk_power_reduction": "not_applicable"}


def test_actual_readback_checks_counts_abi_storage_and_strict_integer_types():
    case = _direct_case(actual_points=4)
    expected = {"disk_power_reduction": "direct-log", "direct_log_power_abi": 3,
                "direct_log_power_state_bytes": 240, "power_vector_created": False}
    assert worker._disk_power_result_fields(case, _details(case)) == expected
    for key, value in (("n", case["n"] + 1), ("groups", 3), ("abi_version", 2),
                       ("abi_version", 3.0), ("dc_bins", True), ("energy_bins", case["n"] // 2),
                       ("nyquist_bins", 0), ("array_state_bytes", 239),
                       ("additional_workers", 1), ("matrix_written", True)):
        details = _details(case)
        details["direct_group_reduction"][key] = value
        with pytest.raises(ValueError, match="metadata"):
            worker._disk_power_result_fields(case, details)
    for details in ({}, {"disk_power_reduction": "materialized"},
                    {**_details(case), "power_vector_created": True},
                    {**_details(case), "fft": {"n": case["n"], "power_storage": "workspace"}},
                    {**_details(case), "power_storage": {"mode": "direct_log_groups"}}):
        with pytest.raises(ValueError):
            worker._disk_power_result_fields(case, details)


def test_direct_forecasts_require_actual_direct_disk_calibration():
    target = _direct_case(n=1_000_000)
    previous = {**_direct_case(n=100_000), "status": "completed", "process_wall_s": .01,
                "direct_log_power_abi": 3, "power_vector_created": False}
    assert resources.calibration_matches(target, previous)
    for changes in ({"disk_power_reduction": "materialized"}, {"storage": "ram"},
                    {"disk_fft_algorithm": "packed-real"}, {"fft_memory_mb": 64.},
                    {"direct_log_power_abi": True}, {"direct_log_power_abi": None},
                    {"power_vector_created": True}):
        assert not resources.calibration_matches(target, {**previous, **changes})
        assert resources.predict_seconds(target, [{**previous, **changes}]) is None
    legacy_target = {key: value for key, value in target.items() if key != "disk_power_reduction"}
    legacy_row = {key: value for key, value in previous.items()
                  if key not in ("disk_power_reduction", "direct_log_power_abi", "power_vector_created")}
    assert resources.calibration_matches(legacy_target, {**legacy_row, "storage": "ram"})
    assert not resources.calibration_matches(legacy_target, previous)


def test_public_fields_roundtrip_without_free_text_or_invented_observations(tmp_path):
    case = _direct_case(actual_points=4)
    fields = worker._disk_power_result_fields(case, _details(case))
    report = {"schema": SCHEMA, "settings": {"disk_power_reduction": "direct-log"},
              "cases": [{**case, **fields, "status": "completed", "phases": _details(case)["phases"]}]}
    clean = privacy.sanitize(report)
    row = clean["cases"][0]
    for key, value in fields.items():
        assert row[key] == value
    assert row["phases"] == {"direct_log_groups_and_energy_s": .125}
    assert privacy.sanitize(clean) == clean
    path = tmp_path / "result.json"
    assert runner.write_report(path, report) == path
    assert json.loads(path.read_text()) == clean
    body = privacy.issue_body(clean)
    assert '"disk_power_reduction":"direct-log"' in body
    report["settings"]["disk_power_reduction"] = "private-host"
    report["cases"][0].update(direct_log_power_abi=True, direct_log_power_state_bytes=240.,
                                power_vector_created="private-file", direct_group_values=["private-sum"])
    report["cases"][0]["phases"]["direct_log_groups_and_energy_s"] = float("nan")
    serialized = json.dumps(privacy.sanitize(report))
    assert "private" not in serialized
    for key in ("direct_log_power_abi", "direct_log_power_state_bytes", "power_vector_created"):
        assert key not in privacy.sanitize(report)["cases"][0]
    old = privacy.sanitize({"schema": SCHEMA, "cases": [{**_case(method="fftw"), "status": "completed"}]})["cases"][0]
    assert old["disk_power_reduction"] == "materialized"
    assert "direct_log_power_abi" not in old and "power_vector_created" not in old


def _fake_worker(monkeypatch, *, workspace=True):
    import numpy as np
    import pandas as pd
    from benchmarks import bench_fftw as bf, bench_fftw_disk as disk, _fftw_direct_power as direct
    from lpsd_fast import api

    f = np.array([.25, 1., 5., 20.])
    plan = (f, np.ones(4), np.ones(4), np.full(4, 32), np.full(4, 2))
    resolved = {"target_points": 100, "base_target_points": 100, "base_actual_points": 4,
                "requested_output_points": 4, "actual_points": 4, "point_count_resolved": True}
    monkeypatch.setattr(worker, "resolve_point_plan", lambda *args: (plan, dict(resolved)))
    monkeypatch.setattr(worker, "_address_guard", lambda case: None)
    monkeypatch.setattr(worker, "_file_hash", lambda path: "a" * 64)
    monkeypatch.setattr(api, "_native", lambda: None)
    monkeypatch.setattr(api, "_LIB", SimpleNamespace(_name=worker.__file__))
    monkeypatch.setattr(direct, "DirectLibrary", lambda library=None: SimpleNamespace(abi_version=3))
    monkeypatch.setattr(disk.DiskFFT, "memory_estimate", staticmethod(
        lambda *args, **kwargs: {"workspace_power_supported": workspace}))
    backend = SimpleNamespace(evidence={}, threads_evidence=None, version="fftw-3.3.10", threads=1)
    monkeypatch.setattr(worker, "_load_fftw", lambda *args: backend)
    clock = {"wall": 0., "cpu": 0., "executions": 0}
    monkeypatch.setattr(worker, "time", SimpleNamespace(
        perf_counter=lambda: clock["wall"], process_time=lambda: clock["cpu"]))

    def pipeline(method, values, backend, directory, options):
        assert method == "fftw" and options.disk_fft_algorithm == "full"
        assert options.disk_power_reduction == "direct-log" and options.power_storage == "workspace"
        assert options.fft_memory_mb == 128. and options.n_averages == 100
        clock.update(wall=11., cpu=7., executions=clock["executions"] + 1)
        psd = np.full(4, 1e-16, dtype=np.float32)
        frame = pd.DataFrame({"psd": psd, "nsd": np.sqrt(psd.astype(np.complex64)).real}, index=f)
        return frame, _details({"n": len(values)})

    monkeypatch.setattr(disk, "disk_pipeline", pipeline)
    return clock, direct


def test_worker_actual_readback_is_outside_both_compute_timers(tmp_path, monkeypatch):
    clock, _ = _fake_worker(monkeypatch)
    readback = worker._disk_power_result_fields

    def delayed_readback(*args, **kwargs):
        assert clock["executions"] == 1
        values = readback(*args, **kwargs)
        clock.update(wall=111., cpu=107.)
        return values

    monkeypatch.setattr(worker, "_disk_power_result_fields", delayed_readback)
    result = worker.run(_direct_case(), tmp_path)
    assert result["status"] == "completed"
    assert result["compute_wall_s"] == 11. and result["compute_cpu_s"] == 7.
    assert result["direct_log_power_abi"] == 3 and result["direct_log_power_state_bytes"] == 240
    assert result["actual_points"] == 4 and result["disk_fft_inner_samples"] == 4096
    assert result["input_unchanged"] and not list(tmp_path.iterdir())


def test_workspace_ineligibility_and_missing_abi_stop_before_input(tmp_path, monkeypatch):
    clock, direct = _fake_worker(monkeypatch, workspace=False)
    result = worker.run(_direct_case(), tmp_path)
    assert result["status"] == "skipped" and result["reason"] == "unsupported"
    assert clock["executions"] == 0 and not list(tmp_path.iterdir())

    def absent(library=None):
        raise RuntimeError("ABI unavailable")

    monkeypatch.setattr(direct, "DirectLibrary", absent)
    with pytest.raises(ImportError, match="ABI3"):
        worker.run(_direct_case(), tmp_path)
    assert not list(tmp_path.iterdir())


def _fake_controller(monkeypatch, tmp_path, matrix):
    monkeypatch.delenv("LPSD_BENCHMARK_DEADLINE_EPOCH", raising=False)
    fixed = resources.Resources(4 * resources.GIB, 3 * resources.GIB, 4 * resources.GIB,
                                2 * resources.GIB, resources.GIB, 1, 1.)
    monkeypatch.setattr(runner, "detect_resources", lambda *args: fixed)
    monkeypatch.setattr(runner, "environment", lambda values: {})
    monkeypatch.setattr(runner, "_matrix", lambda args: matrix)
    monkeypatch.setattr(runner, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_source_dirty", lambda: False)
    return runner.parser().parse_args(["run", "--methods", "fftw", "--disk-power-reduction", "direct-log",
                                      "--work-dir", str(tmp_path / "work"), "--output", str(tmp_path / "result.json")])


def _spec(n, storage):
    return {"n": n, "target_points": 1000, "point_factor": 1., "method": "fftw", "signal": "white",
            "requested_storage": storage, "case_role": "comparison"}


def test_direct_does_not_override_global_deadline(tmp_path, monkeypatch):
    args = _fake_controller(monkeypatch, tmp_path, [_spec(100_000, "disk"), _spec(1_000_000, "disk")])
    clock = [0.]
    deadline = resources.Deadline(86400., external_epoch=10004.,
                                  monotonic=lambda: clock[0], wallclock=lambda: 10000.)
    monkeypatch.setattr(runner, "Deadline", lambda *args: deadline)
    observed = []

    def execute(case, directory, active, timeout):
        assert active is deadline and 0 < timeout <= 4.
        observed.append(case["n"])
        clock[0] = 3.95
        return {"status": "completed", "actual_points": 4, "point_count_resolved": True,
                "compute_wall_s": .01, "process_wall_s": .02, "direct_log_power_abi": 3,
                "power_vector_created": False}

    monkeypatch.setattr(runner, "execute_case", execute)
    assert runner.run(args) == 0
    report = json.loads(args.output.read_text())
    assert observed == [100_000]
    assert report["state"] == "deadline"
    assert report["cases"][1]["status"] == "skipped" and report["cases"][1]["reason"] == "time_budget"
    assert report["settings"]["bootstrap_remaining_seconds"] == 4.


def test_missing_direct_dependency_does_not_disable_ram_route(tmp_path, monkeypatch):
    args = _fake_controller(monkeypatch, tmp_path, [_spec(100_000, "disk"), _spec(100_000, "ram")])
    observed = []

    def execute(case, directory, deadline, timeout):
        observed.append((case["storage"], case["disk_power_reduction"]))
        if case["storage"] == "disk":
            return {"status": "skipped", "reason": "dependency_unavailable"}
        return {"status": "completed", "actual_points": 4, "point_count_resolved": True,
                "compute_wall_s": .01, "process_wall_s": .02}

    monkeypatch.setattr(runner, "execute_case", execute)
    assert runner.run(args) == 0
    assert observed == [("disk", "direct-log"), ("ram", "not_applicable")]
