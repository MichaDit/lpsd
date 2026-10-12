"""Native budget caps: scalar admission, actual call wiring and provenance.

Billion-sample cases below are scalar only. Numerical integration uses at
most 4096 samples; no large dataset is part of this test suite.
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lpsd_benchmark import SCHEMA, privacy, resources, runner, worker
from lpsd_benchmark.resources import GIB, MIB


def _case(*, method="lpsd", storage="disk", request=None, ram=4 * GIB, n=100_000, threads=8):
    return {"method": method, "storage": storage, "requested_threads": threads,
            "requested_native_working_mib": request, "ram_budget_bytes": ram,
            "low_working_mb": 2048., "n": n, "signal": "white", "target_points": 100,
            "base_target_points": 100, "point_factor": 1., "sample_rate_hz": 50.,
            "seed": 20261010, "n_averages": 100, "fft_memory_mb": 1.,
            "disk_budget_bytes": 256 * MIB, "fft_layout": "in-place", "window_storage": "full"}


@pytest.mark.parametrize("bad", [0, -1, 63, True, False, 64.0, "384", float("nan"),
                                 float("inf"), resources.MAX_NATIVE_WORKING_MIB + 1])
def test_invalid_cap_fails_before_numerical_imports(bad, monkeypatch, tmp_path):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "numpy" or name.startswith("benchmarks"):
            pytest.fail("Invalid scalar options must be rejected before numerical imports")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="native-working-mib"):
        worker.run(_case(request=bad), tmp_path)


@pytest.mark.parametrize("bad", ["0", "63", "64.5", "true", "nan", "inf",
                                 str(resources.MAX_NATIVE_WORKING_MIB + 1)])
def test_invalid_cli_cap_is_rejected(bad):
    with pytest.raises(SystemExit) as error:
        runner.parser().parse_args(["run", "--native-working-mib", bad])
    assert error.value.code == 2


def test_default_headroom_and_explicit_cap_are_separate_without_extra_spending():
    case = _case(method="matched", storage="ram", n=1_000_000_000, ram=32 * GIB)
    original, headroom, fields = resources.native_working_budget(case)
    assert headroom == 32 * GIB - 8_000_000_000 - 280 * MIB
    assert worker._native_working_budget(case) == (original, headroom)
    assert original * MIB == fields["native_working_limit_bytes"] == headroom
    assert fields["native_working_policy"] == "auto"
    chosen = {**case, "requested_native_working_mib": 384, "requested_smoothing_workers": 8}
    before = chosen.copy()
    actual, remaining, limited = resources.native_working_budget(chosen)
    assert chosen == before and actual == 384
    assert remaining == headroom - 36 * MIB
    assert limited["native_working_auto_limit_bytes"] == remaining
    assert limited["native_working_limit_bytes"] == 384 * MIB
    assert limited["requested_native_working_mib"] == 384
    assert limited["native_working_policy"] == "explicit_cap"
    # A huge request never grows the existing allowance or changes reserves.
    large = {**chosen, "requested_native_working_mib": resources.MAX_NATIVE_WORKING_MIB}
    assert resources.native_working_budget(large)[0] * MIB <= remaining


def test_disk_cap_can_be_reduced_again_or_skipped_without_adapter_minimum_overshoot():
    mb, headroom, fields = resources.native_working_budget(_case(request=384, ram=512 * MIB))
    assert mb == 256 and headroom == fields["native_working_limit_bytes"] == 256 * MIB
    assert fields["native_working_auto_limit_bytes"] == 2048 * MIB
    mb, headroom, fields = resources.native_working_budget(_case(request=384, ram=320 * MIB - 1))
    assert mb is None and headroom == fields["native_working_limit_bytes"] == 64 * MIB - 1
    # Existing private automatic cases keep their historical MiB argument.
    legacy = {**_case(), "low_working_mb": 12.5}
    assert worker._native_working_budget(legacy)[0] == 12.5
    assert resources.native_working_budget(legacy)[2]["native_working_limit_bytes"] == 64 * MIB
    assert resources.native_working_budget({**legacy, "requested_native_working_mib": 384})[0] == 64


@pytest.mark.parametrize("ram", [2 ** 53 - 1, 2 ** 53 + 1, 2 ** 63 - 1])
def test_large_byte_headroom_roundtrip_never_rounds_the_explicit_cap_up(ram):
    # Scalar only: exercise bytes that binary64 cannot all represent exactly.
    case = _case(storage="ram", request=resources.MAX_NATIVE_WORKING_MIB,
                 ram=ram, n=1_000_000_000)
    mb, headroom, fields = resources.native_working_budget(case)
    limit = fields["native_working_limit_bytes"]
    assert int(mb * MIB) == limit
    assert 64 * MIB <= limit <= min(headroom, fields["native_working_auto_limit_bytes"],
                                    resources.MAX_NATIVE_WORKING_MIB * MIB)


def test_fftw_has_no_native_capacity():
    assert resources.native_working_budget(_case(method="fftw", request=384)) == (
        None, None, {"requested_native_working_mib": None, "native_working_policy": "not_applicable"})
    assert worker._native_working_result_fields(_case(method="fftw", request=384), {}) == {}


@pytest.mark.parametrize("changed", [
    {"native_working_policy": "auto"}, {"native_working_limit_bytes": 384 * MIB + 1},
    {"native_working_auto_limit_bytes": 2048 * MIB + 1}, {"native_working_limit_bytes": True},
])
def test_worker_rejects_forged_admitted_fields(changed):
    case = _case(request=384)
    case.update(resources.native_working_budget(case)[2])
    with pytest.raises(ValueError, match="Inconsistent"):
        worker._case_native_working_controls({**case, **changed})


@pytest.mark.parametrize("method,storage", [(m, s) for m in ("lpsd", "matched") for s in ("ram", "disk")])
def test_applied_capacity_comes_from_the_returned_route(method, storage):
    case = _case(method=method, storage=storage, request=384)
    source = {"concurrency_budget_bytes": 384 * MIB, "frequencies": 4}
    attrs = {"lpsd_disk": source, "matched_smoothing_once": {"low_frequency_preparation": source}}
    disk = {"lpsd": source, "low_lpsd": source}
    assert worker._native_working_result_fields(case, attrs, disk) == {"native_working_applied_bytes": 384 * MIB}
    source["concurrency_budget_bytes"] = 256 * MIB
    with pytest.raises(ValueError, match="differs from admission"):
        worker._native_working_result_fields(case, attrs, disk)
    source["frequencies"] = 0
    assert worker._native_working_result_fields(case, attrs, disk) == {}


def test_missing_or_empty_native_metadata_is_not_reported_as_executed():
    case = _case(method="matched", storage="ram", request=384)
    assert worker._native_working_result_fields(case, {
        "matched_smoothing_once": {"exact_low_frequency_points": 0}}) == {}
    with pytest.raises(ValueError, match="differs from admission"):
        worker._native_working_result_fields(case, {})
    assert worker._native_working_result_fields({**case, "requested_native_working_mib": None}, {}) == {}


def test_forecasts_and_calibration_require_the_same_explicit_profile():
    target = _case(request=384, n=10_000_000)
    target.update(resources.native_working_budget(target)[2])
    previous = {**target, "n": 100_000, "status": "completed", "process_wall_s": .5}
    assert resources.calibration_matches(target, previous)
    assert resources.predict_seconds(target, [previous]) is not None
    for changes in ({"native_working_policy": "auto"}, {"native_working_limit_bytes": 256 * MIB},
                    {"requested_native_working_mib": 512}, {"native_working_limit_bytes": None}):
        other = {**previous, **changes}
        assert not resources.calibration_matches(target, other)
        assert resources.predict_seconds(target, [other]) is None
    legacy_target = {k: v for k, v in target.items() if "native_working" not in k}
    legacy_previous = {k: v for k, v in previous.items() if "native_working" not in k}
    assert resources.calibration_matches(legacy_target, legacy_previous)
    assert resources.predict_seconds(legacy_target, [legacy_previous]) is not None
    assert not resources.calibration_matches(legacy_target, previous)


@pytest.mark.parametrize("method", ["lpsd", "matched"])
def test_calibration_keeps_corrected_and_legacy_statistics_separate(method):
    base = {"method": method, "signal": "white", "storage": "ram"}
    old = {**base, "status": "completed"}
    corrected = {**base, "statistics": "corrected"}
    assert resources.calibration_matches(base, old)
    assert resources.calibration_matches({**base, "statistics": "legacy"}, old)
    assert not resources.calibration_matches(corrected, old)
    assert not resources.calibration_matches(base, {**old, "statistics": "corrected"})
    assert resources.calibration_matches(corrected, {**old, "statistics": "corrected"})


def test_public_cap_provenance_is_closed_idempotent_and_optional():
    case = _case(request=384)
    fields = {**resources.native_working_budget(case)[2], "native_working_applied_bytes": 384 * MIB}
    clean = privacy.sanitize({"schema": SCHEMA, "state": "completed", "settings": fields,
        "cases": [{**case, **fields, "status": "completed"}]})
    for key, value in fields.items():
        assert clean["cases"][0][key] == value
    assert clean["settings"]["requested_native_working_mib"] == 384
    assert privacy.sanitize(clean) == clean
    for bad in (True, -1, 1.5, "private/path", 2 ** 63):
        assert privacy._native_working_fields({key: bad for key in fields}) == {}
    old = privacy.sanitize({"schema": SCHEMA, "cases": [{k:v for k,v in case.items()
        if "native_working" not in k}]})
    assert not any("native_working" in key for key in old["settings"])
    assert not any("native_working" in key for key in old["cases"][0])


def test_controller_forwards_native_cap_without_changing_other_budgets(tmp_path, monkeypatch):
    monkeypatch.delenv("LPSD_BENCHMARK_DEADLINE_EPOCH", raising=False)
    fixed = resources.Resources(4 * GIB, 3 * GIB, 8 * GIB, 2 * GIB, 4 * GIB, 8, 8.)
    monkeypatch.setattr(runner, "detect_resources", lambda *args, **kwargs: fixed)
    monkeypatch.setattr(runner, "environment", lambda _: {})
    monkeypatch.setattr(runner, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_source_dirty", lambda: False)
    seen = []
    def fake_case(case, directory, deadline, timeout):
        seen.append(case)
        assert case["ram_budget_bytes"] == fixed.ram_budget and case["disk_budget_bytes"] == fixed.disk_budget
        assert case["fft_memory_mb"] == 128. and case["low_working_mb"] == fixed.ram_budget / MIB / 3
        if case["method"] == "fftw":
            assert case["native_working_policy"] == "not_applicable"
            assert "native_working_limit_bytes" not in case
        else:
            assert case["requested_native_working_mib"] == 384
            assert case["native_working_limit_bytes"] == 384 * MIB
        return {"status": "completed", "compute_wall_s": .01, "process_wall_s": .1}
    monkeypatch.setattr(runner, "execute_case", fake_case)
    args = runner.parser().parse_args(["run", "--sizes", "100000", "--point-factors", "1",
        "--profile", "performance", "--storage", "disk", "--native-working-mib", "384",
        "--work-dir", str(tmp_path / "work"), "--output", str(tmp_path / "report.json")])
    assert runner.run(args) == 0 and len(seen) == 3
    report = json.loads(args.output.read_text())
    assert report["settings"]["requested_native_working_mib"] == 384


@pytest.mark.parametrize("method,storage", [(m, s) for m in ("lpsd", "matched") for s in ("ram", "disk")])
@pytest.mark.parametrize("too_large", [False, True])
def test_preview_and_actual_routes_share_cap_and_skip_before_allocation(tmp_path, monkeypatch, method, storage, too_large):
    """A fake adapter observes actual argument flow, independent of its math."""
    import numpy as np
    import pandas as pd
    from benchmarks import bench_fftw as bf, bench_fftw_disk as disk, matched_smoothing, _lpsd_disk
    from lpsd_fast import api

    case = _case(method=method, storage=storage, request=384, n=4096, threads=1)
    f = np.array([.25, 1., 5., 20.])
    plan = (f, np.ones(4), np.ones(4), np.full(4, 32), np.full(4, 2))
    resolved = {"target_points": 100, "base_target_points": 100, "base_actual_points": 4,
                "requested_output_points": 4, "actual_points": 4, "point_count_resolved": True}
    monkeypatch.setattr(worker, "resolve_point_plan", lambda *args: (plan, dict(resolved)))
    monkeypatch.setattr(worker, "_address_guard", lambda _: None)
    monkeypatch.setattr(worker, "_file_hash", lambda _: "a" * 64)
    monkeypatch.setattr(api, "_native", lambda: None)
    monkeypatch.setattr(api, "_LIB", SimpleNamespace(_name=worker.__file__))
    monkeypatch.setattr(bf, "KaiserWindow", lambda *args: object())
    monkeypatch.setattr(disk.DiskFFT, "memory_estimate", staticmethod(lambda *args, **kwargs: {"workspace_power_supported": True}))
    backend = SimpleNamespace(evidence={}, threads_evidence=None, version="fftw-3.3.10", threads=1, forget_wisdom=lambda: None)
    monkeypatch.setattr(worker, "_load_fftw", lambda *args: backend)
    clock = {"wall": 0., "cpu": 0.}
    monkeypatch.setattr(worker, "time", SimpleNamespace(perf_counter=lambda: clock["wall"], process_time=lambda: clock["cpu"]))
    allowances, executions = [], []

    class Estimator:
        def __init__(self, *args, **kwargs):
            self.budget = int(kwargs["max_working_mb"] * MIB)
            allowances.append(self.budget)
        @property
        def metadata(self):
            return {"concurrency_budget_bytes": self.budget, "frequencies": 4,
                    "bounded_memory_model": {"scratch_bytes": 4096, "streaming_bytes_upper_bound": 0,
                    "working_bytes_upper_bound": 385 * MIB if too_large else 8192,
                    "fits_working_budget": True}, "streaming": {"eligible_points": 0}}
        def compute(self, values, **kwargs):
            assert not too_large
            executions.append(len(values))
            clock.update(wall=11., cpu=7.)
            psd = np.full(4, 1e-16, dtype=np.float32)
            frame = pd.DataFrame({"psd": psd, "nsd": np.sqrt(psd.astype(np.complex64)).real}, index=f)
            frame.attrs["lpsd_disk"] = self.metadata
            return frame

    monkeypatch.setattr(_lpsd_disk, "DiskLPSDSubset", Estimator)
    def disk_pipeline(kind, values, backend, directory, args, **kwargs):
        estimator = Estimator(max_working_mb=args.low_working_mb)
        frame = estimator.compute(values)
        details = {"phases": {}, "lpsd" if kind == "lpsd" else "low_lpsd": estimator.metadata}
        if kind == "matched":
            details["smoothing"] = {"workers": 1, "pool_capacity": 1, "task_ranges": 2, "high_points": 4, "backend": "native"}
        return frame, details
    def estimate_once(backend, values, **kwargs):
        estimator = Estimator(max_working_mb=kwargs["max_working_mb"])
        frame = estimator.compute(values)
        frame.attrs["matched_smoothing_once"] = {
            "low_frequency_preparation": estimator.metadata, "window_storage": "full",
            "window_buffer_bytes": 0, "window_scratch_bytes": 0, "one_shot_window": True,
            "requested_smoothing_workers": 1, "smoothing_pool_capacity": 0,
            "smoothing_task_ranges": 1, "smoothed_frequency_points": 4, "smoothing_backend": "native"}
        return frame
    monkeypatch.setattr(disk, "disk_pipeline", disk_pipeline)
    monkeypatch.setattr(matched_smoothing, "estimate_once", estimate_once)
    if too_large:
        def forbidden(*args, **kwargs):
            pytest.fail("Over-cap preflight must skip before sample allocation")
        monkeypatch.setattr(worker, "generate_white", forbidden)
        monkeypatch.setattr(disk, "input_file", forbidden)
    readback = worker._native_working_result_fields
    def observed(*args, **kwargs):
        assert executions
        result = readback(*args, **kwargs)
        clock.update(wall=111., cpu=107.)
        return result
    monkeypatch.setattr(worker, "_native_working_result_fields", observed)
    result = worker.run(case, tmp_path)
    if too_large:
        assert result["status"] == "skipped" and result["reason"] == "memory_budget"
        assert allowances == [384 * MIB] and not executions
    else:
        assert result["status"] == "completed", result
        assert allowances == [384 * MIB, 384 * MIB] and executions == [4096]
        assert result["native_working_applied_bytes"] == 384 * MIB
        assert result["compute_wall_s"] == 11. and result["compute_cpu_s"] == 7.


@pytest.mark.parametrize("method,storage,signal", [
    ("lpsd", "ram", "white"), ("matched", "ram", "offbin_tone"),
    ("lpsd", "disk", "dc_nanovolt"), ("matched", "disk", "white"),
])
def test_tiny_complete_calls_apply_cap_and_preserve_input_and_spectra(tmp_path, method, storage, signal):
    results = []
    for request in (None, 128):
        case = {**_case(method=method, storage=storage, request=request, n=4096, ram=768 * MIB, threads=2),
                "signal": signal}
        directory = tmp_path / ("auto" if request is None else "capped")
        directory.mkdir()
        result = runner.execute_case(case, directory, resources.Deadline(45.), 40.)
        if result.get("reason") == "dependency_unavailable":
            pytest.skip("Optional FFTW/native runtime unavailable")
        assert result["status"] == "completed", result
        assert result["native_working_applied_bytes"] == result["native_working_limit_bytes"]
        if request is not None:
            assert result["native_working_applied_bytes"] == request * MIB
        assert result["input_unchanged"] and result["nsd_matches_sqrt_psd"]
        assert not list(directory.glob("*.f64"))
        results.append(result)
    for key in ("input_sha256", "frequency_sha256", "psd_sha256", "nsd_sha256", "source_sha256"):
        assert results[0][key] == results[1][key]
    assert results[0]["_spectrum"] == results[1]["_spectrum"]
