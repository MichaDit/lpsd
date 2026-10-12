"""Portable pool-lifecycle reporting; no large arrays or FFTW runtime needed."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lpsd_benchmark import SCHEMA, privacy, resources, worker


CAPACITIES = {
    "native_scratch_initial_bytes": 4096,
    "native_scratch_final_bytes": 256,
    "native_scratch_peak_bytes": 4096,
}
INDICES = {"native_scratch_retirement_index": 1,
           "native_scratch_retirement_source_index": 3}
OBSERVED = {**CAPACITIES, **INDICES, "native_scratch_retired": True}


def source(**changes):
    lifecycle = {"initial_bytes": 4096, "final_bytes": 256,
                 "peak_allocated_bytes": 4096, "state": "retired",
                 "retirement_index": 1, "retirement_source_index": 3,
                 "planned_retirement_index": 1,
                 "scope": "PRIVATE_LOCAL_CANARY", "allocation_note": "PRIVATE_LOCAL_CANARY"}
    lifecycle.update(changes)
    return {"scratch_pool_lifecycle": lifecycle}


def report(**fields):
    return {"schema": SCHEMA, "state": "completed",
            "cases": [{"n": 4096, "target_points": 8, "method": "lpsd",
                       "signal": "white", "storage": "disk", "status": "completed",
                       "native_scratch_bytes": 4096, **fields}]}


@pytest.mark.parametrize("method,storage,initial", [
    ("lpsd", "ram", 4096), ("matched", "ram", 8192),
    ("lpsd", "disk", 16384), ("matched", "disk", 32768),
    ("fftw", "ram", None), ("fftw", "disk", None),
])
def test_worker_routes_only_post_call_native_metadata(method, storage, initial):
    attrs = {"lpsd_disk": source(), "matched_smoothing_once": {
        "low_frequency_preparation": source(initial_bytes=8192, peak_allocated_bytes=8192)}}
    details = {"lpsd": source(initial_bytes=16384, peak_allocated_bytes=16384),
               "low_lpsd": source(initial_bytes=32768, peak_allocated_bytes=32768)}
    fields = worker._native_scratch_fields(method, storage, attrs, details)
    expected = ({} if initial is None else {
        **OBSERVED, "native_scratch_initial_bytes": initial,
        "native_scratch_peak_bytes": initial})
    assert fields == expected
    assert "PRIVATE_LOCAL_CANARY" not in json.dumps(fields)


@pytest.mark.parametrize("state", ["unchanged", "scheduled", "replacement_failed"])
def test_planned_or_failed_retirement_never_exports_indices(state):
    fields = worker._native_scratch_fields("lpsd", "ram", {"lpsd_disk": source(state=state)})
    assert fields == {**CAPACITIES, "native_scratch_retired": False}
    # Even explicitly supplied stale indices cannot survive the public schema.
    clean = privacy.sanitize(report(**fields, **INDICES))["cases"][0]
    assert clean["native_scratch_retired"] is False
    assert not set(INDICES).intersection(clean)


@pytest.mark.parametrize("metadata", [None, {}])
def test_absent_lifecycle_preserves_old_payload(metadata):
    assert worker._native_scratch_fields("lpsd", "ram", {"lpsd_disk": metadata}) == {}
    clean = privacy.sanitize(report())["cases"][0]
    assert clean["native_scratch_bytes"] == 4096
    assert not set(OBSERVED).intersection(clean)


def test_public_scratch_fields_round_trip_without_extra_payload():
    raw = report(**OBSERVED, native_scratch_state="PRIVATE_LOCAL_CANARY",
                 native_scratch_planned_retirement_index=1,
                 native_scratch_rss_saved_bytes=3840, **source())
    clean = privacy.sanitize(raw)
    row = clean["cases"][0]
    assert {key: row[key] for key in OBSERVED} == OBSERVED
    assert row["native_scratch_bytes"] == 4096
    assert "native_scratch_planned_retirement_index" not in row
    assert "native_scratch_rss_saved_bytes" not in row
    assert "scratch_pool_lifecycle" not in row
    assert "PRIVATE_LOCAL_CANARY" not in json.dumps(clean, allow_nan=False)
    assert privacy.sanitize(clean) == clean


@pytest.mark.parametrize("bad", [-1, True, 1.5, "PRIVATE_LOCAL_CANARY", float("nan"), 2**63])
def test_lifecycle_scalars_reject_non_integer_or_out_of_range_values(bad):
    original_names = ("initial_bytes", "final_bytes", "peak_allocated_bytes",
                      "retirement_index", "retirement_source_index")
    metadata = source(**dict.fromkeys(original_names, bad))
    fields = worker._native_scratch_fields("lpsd", "ram", {"lpsd_disk": metadata})
    assert fields == {"native_scratch_retired": True}
    supplied = dict.fromkeys((*CAPACITIES, *INDICES), bad)
    row = privacy.sanitize(report(**supplied, native_scratch_retired=True))["cases"][0]
    assert row["native_scratch_retired"] is True
    assert not set(supplied).intersection(row)


@pytest.mark.parametrize("bad", [None, 1, "retired"])
def test_public_retirement_indices_require_boolean_true(bad):
    row = privacy.sanitize(report(**CAPACITIES, **INDICES,
                                  native_scratch_retired=bad))["cases"][0]
    assert "native_scratch_retired" not in row
    assert not set(INDICES).intersection(row)


@pytest.mark.parametrize("method,storage", [
    ("lpsd", "ram"), ("lpsd", "disk"), ("matched", "ram"), ("matched", "disk"),
])
def test_worker_records_actual_metadata_after_timing(tmp_path, monkeypatch, method, storage):
    """Exercise run's four routes with tiny, fake metadata producers.

    This is a dataflow/timing-boundary test, not a numerical retirement test.
    A fake clock makes any inclusion of the reporting work observable.
    """
    import numpy as np
    import pandas as pd
    from benchmarks import bench_fftw as bf
    from benchmarks import bench_fftw_disk as disk
    from benchmarks import matched_smoothing
    from benchmarks import _lpsd_disk as native

    f = np.array([.25, 1., 5., 20.], dtype=np.float64)
    plan = (f, np.ones(4), np.ones(4), np.full(4, 32), np.full(4, 2))
    resolved = {"target_points": 8, "base_target_points": 8, "base_actual_points": 4,
                "requested_output_points": 4, "actual_points": 4, "point_count_resolved": True}
    clock = {"wall": 0., "cpu": 0., "completed": False, "exports": 0}
    monkeypatch.setattr(worker, "time", SimpleNamespace(
        perf_counter=lambda: clock["wall"], process_time=lambda: clock["cpu"]))
    monkeypatch.setattr(worker, "resolve_point_plan", lambda *args: (plan, dict(resolved)))
    monkeypatch.setattr(worker, "_address_guard", lambda case: None)
    monkeypatch.setattr(worker, "_file_hash", lambda path: "a" * 64)
    monkeypatch.setattr(bf, "KaiserWindow", lambda *args: object())
    monkeypatch.setattr(disk.DiskFFT, "memory_estimate", staticmethod(
        lambda *args: {"workspace_power_supported": True}))
    backend = SimpleNamespace(evidence={"binary_sha256": "b" * 64},
                              threads_evidence=None, version="fftw-3.3.11", threads=1,
                              forget_wisdom=lambda: None)
    monkeypatch.setattr(worker, "_load_fftw", lambda *args: backend)

    class Estimator:
        def __init__(self, *args, **kwargs):
            self.completed = False

        @property
        def metadata(self):
            value = {"bounded_memory_model": {"scratch_bytes": 4096,
                     "streaming_bytes_upper_bound": 0, "working_bytes_upper_bound": 8192,
                     "fits_working_budget": True}, "streaming": {"eligible_points": 0}}
            if self.completed:
                value.update(source())
            return value

        def compute(self, values, **kwargs):
            self.completed = True
            clock.update(wall=11., cpu=7., completed=True)
            psd = np.full(4, 1e-16, dtype=np.float32)
            frame = pd.DataFrame({"psd": psd, "nsd": np.sqrt(psd.astype(np.complex64)).real}, index=f)
            frame.attrs["lpsd_disk"] = self.metadata
            return frame

    monkeypatch.setattr(native, "DiskLPSDSubset", Estimator)

    def disk_pipeline(kind, values, backend, directory, args):
        estimator = Estimator()
        frame = estimator.compute(values)
        # This deliberately stale frame entry must not replace disk details.
        frame.attrs["lpsd_disk"] = source(state="scheduled", final_bytes=4096)
        details = {"phases": {}, "lpsd" if kind == "lpsd" else "low_lpsd": estimator.metadata}
        if kind == "matched":
            assert args.smoothing_workers == 1
            details["smoothing"] = {"workers": 1, "pool_capacity": 1, "task_ranges": 2,
                                    "high_points": 4, "backend": "native"}
        return frame, details

    def estimate_once(backend, values, **kwargs):
        assert kwargs["smoothing_workers"] == 1
        estimator = Estimator()
        frame = estimator.compute(values)
        frame.attrs.clear()
        frame.attrs["matched_smoothing_once"] = {
            "low_frequency_preparation": estimator.metadata, "window_storage": "full",
            "window_buffer_bytes": 0, "window_scratch_bytes": 0, "one_shot_window": True,
            "requested_smoothing_workers": 1, "smoothing_pool_capacity": 0,
            "smoothing_task_ranges": 1, "smoothed_frequency_points": 4,
            "smoothing_backend": "native"}
        return frame

    monkeypatch.setattr(disk, "disk_pipeline", disk_pipeline)
    monkeypatch.setattr(matched_smoothing, "estimate_once", estimate_once)
    original = worker._native_scratch_fields

    def export(*args, **kwargs):
        assert clock["completed"]
        clock["exports"] += 1
        fields = original(*args, **kwargs)
        clock["wall"] += 100.
        clock["cpu"] += 100.
        return fields

    monkeypatch.setattr(worker, "_native_scratch_fields", export)
    case = {"n": 4096, "method": method, "signal": "white", "storage": storage,
            "target_points": 8, "base_target_points": 8, "point_factor": 1.,
            "requested_threads": 1, "ram_budget_bytes": 512 * resources.MIB,
            "disk_budget_bytes": 512 * resources.MIB, "sample_rate_hz": 50.,
            "seed": 20261010, "n_averages": 100, "fft_memory_mb": 16., "low_working_mb": 64.}
    result = worker.run(case, tmp_path)
    assert result["status"] == "completed", result
    assert clock["exports"] == 1
    assert result["compute_wall_s"] == 11. and result["compute_cpu_s"] == 7.
    assert result["native_scratch_bytes"] == 4096
    assert {key: result[key] for key in OBSERVED} == OBSERVED
    assert result["input_unchanged"] and result["nsd_matches_sqrt_psd"]
    assert not list(tmp_path.glob("*.f64"))
    public = privacy.sanitize({"schema": SCHEMA, "cases": [{**case, **result}]})["cases"][0]
    assert {key: public[key] for key in OBSERVED} == OBSERVED
