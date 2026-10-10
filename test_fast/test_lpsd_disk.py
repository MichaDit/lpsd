"""Small numerical/storage gates for the separate file-backed LPSD adapter.

These tests use the unchanged public native API and prepared native subsets as
references. They do not benchmark, relax memory limits, or allocate large maps.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes as ct
import json
import threading

import numpy as np
import pandas as pd
import pytest

from benchmarks import _lpsd_disk as disk_module
from benchmarks._lpsd_disk import DiskLPSDSubset, compute_disk_lpsd
from lpsd_fast import api
from lpsd_fast.planning import _kaiser_alpha, _kaiser_rov
from lpsd_fast.prepared import PreparedLPSDSubset
from test_fast._support import DEFAULTS


FS = DEFAULTS["sample_rate"]
OVERLAP = _kaiser_rov(_kaiser_alpha(200))
SETTINGS = {"n_frequencies": DEFAULTS["n_frequencies"],
            "n_averages": DEFAULTS["n_averages"]}


def _signal(n, kind):
    rng = np.random.default_rng(81729 + n)
    white = rng.standard_normal(n)
    if kind == "dc_nanovolt":
        return 10.0 + 1e-9 * white
    if kind == "offbin_tone":
        return np.sqrt(2) * 1e-6 * np.sin(2 * np.pi * 3.123456 * np.arange(n) / FS)
    return white * 5e-8


def _plan(n):
    return api._frequency_plan(n, FS, OVERLAP, 1, 0,
                               SETTINGS["n_frequencies"], SETTINGS["n_averages"])


def _reference(values):
    return api.lpsd(values, sample_rate=FS, psll=200, overlap=OVERLAP,
                    detrending_order=0, workers=8, kernel="fast", window_cache_mb=0,
                    max_working_mb=64, outputs=("psd", "nsd"), **SETTINGS)


@contextmanager
def _mapped_arrays(tmp_path, values):
    n = len(values)
    data = np.memmap(tmp_path / "input.bin", mode="w+", dtype=np.float64, shape=(n,))
    workspace = np.memmap(tmp_path / "workspace.bin", mode="w+", dtype=np.complex128,
                          shape=(n,))
    data[:] = values
    try:
        yield data, workspace
    finally:
        workspace._mmap.close()
        data._mmap.close()


def _assert_same_spectrum(actual, expected):
    """Prefer bit equality, retaining the existing 3e-6 cross-target gate."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    exact = actual.tobytes() == expected.tobytes()
    if not exact:
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=0)
    return exact


@pytest.mark.parametrize("n", (4096, 10000))
@pytest.mark.parametrize("kind", ("white", "dc_nanovolt", "offbin_tone"))
@pytest.mark.parametrize("workers", (1, 8))
def test_mapped_psd_nsd_match_public_eight_worker_path(tmp_path, n, kind, workers,
                                                    record_property):
    values = _signal(n, kind)
    expected = _reference(values)
    with _mapped_arrays(tmp_path, values) as (data, workspace):
        with DiskLPSDSubset(n, FS, _plan(n), workers=workers,
                            max_working_mb=64) as method:
            actual = method.compute(data, workspace, outputs=("psd", "nsd"), profile=True)
            pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
            assert tuple(actual.columns) == ("psd", "nsd")
            for output in actual:
                exact = _assert_same_spectrum(actual[output], expected[output])
                record_property(f"bitwise_{output}", exact)
            nsd_from_rounded_psd = np.sqrt(actual.psd.to_numpy().astype(np.complex64)).real
            assert actual.nsd.to_numpy().tobytes() == nsd_from_rounded_psd.tobytes()
            assert data.tobytes() == values.tobytes()
            assert not np.shares_memory(actual.to_numpy(), data)
            assert not np.shares_memory(actual.to_numpy(), workspace)
            profile = method.last_profile
            assert profile["peak_active_scratch_bytes"] <= 16 * n
            assert profile["peak_reserved_bytes"] <= profile["concurrency_budget_bytes"]
            assert len(profile["frequencies"]) == len(actual)
            assert [row["j"] for row in profile["frequencies"]] == list(range(len(actual)))
            assert profile["sample_iterations"] == sum(
                row["L"] * row["K"] for row in profile["frequencies"])
            assert profile["validation_wall_s"] >= 0
            assert all(row["c_preparation_s"] >= 0 and row["c_segments_s"] >= 0
                       for row in profile["frequencies"])
            assert not method.metadata["bounded_segment_fma_enabled"]
            json.dumps(method.metadata, allow_nan=False)
            json.dumps(profile, allow_nan=False)
            saved = actual.copy(deep=True)
        # Closing the adapter does not close either caller-owned mapping, and
        # subsequent scratch/input writes cannot change the returned spectrum.
        assert not data._mmap.closed and not workspace._mmap.closed
        workspace[:] = -321 + 17j
        data[:] = 0
        pd.testing.assert_frame_equal(actual, saved, check_exact=True)


@pytest.mark.parametrize("n", (4096, 10000))
@pytest.mark.parametrize("kind", ("white", "dc_nanovolt", "offbin_tone"))
def test_low_plan_mask_matches_prepared_and_public_points(tmp_path, n, kind, record_property):
    values = _signal(n, kind)
    full = _plan(n)
    lengths = np.asarray(full[3])
    mask = (lengths > n / 16) | (lengths < 32)
    subset = tuple(np.asarray(part)[mask] for part in full)
    expected = _reference(values).iloc[np.flatnonzero(mask)]
    assert len(subset[0]) > 1  # Prepared workers=8 must also use ordinary dots.
    with _mapped_arrays(tmp_path, values) as (data, workspace):
        with PreparedLPSDSubset(n, FS, subset, psll=200, overlap=OVERLAP, workers=8,
                                max_cache_mb=4, max_working_mb=64) as prepared:
            prepared_psd = prepared.compute(data)
        with DiskLPSDSubset(n, FS, full, plan_mask=mask, workers=8,
                            max_working_mb=64) as method:
            actual = method.compute(data, workspace, outputs=("psd", "nsd"))
            pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
            record_property("bitwise_prepared", _assert_same_spectrum(actual.psd, prepared_psd))
            for output in actual:
                record_property(f"bitwise_public_{output}",
                                _assert_same_spectrum(actual[output], expected[output]))
            assert method.metadata["source_plan_indices"] == np.flatnonzero(mask).tolist()


@pytest.mark.parametrize("n", (17, 31, 63))
@pytest.mark.parametrize("workers", (1, 8))
def test_short_input_window_and_coefficient_fallbacks(tmp_path, n, workers):
    values = _signal(n, "dc_nanovolt")
    expected = _reference(values)
    with _mapped_arrays(tmp_path, values) as (data, workspace):
        result = compute_disk_lpsd(data, workspace, FS, plan=_plan(n), workers=workers,
                                   outputs=("psd", "nsd"), profile=False)
        for output in result:
            _assert_same_spectrum(result[output], expected[output])
        assert not result.attrs["lpsd_disk_profile"]["stage_profiling_enabled"]
        assert result.attrs["lpsd_disk_profile"]["constructor_compute_close_wall_s"] >= 0


@pytest.mark.parametrize("generator_name", ("generate_coefficients", "generate_coefficients_blocked"))
@pytest.mark.parametrize("length,frequency_bin", ((31, 1.37), (63, 2.75), (257, 2.37),
                                                 (1024, 2000.0), (4096, 17.37)))
def test_native_window_cr_alias_matches_separate_arrays(tmp_path, generator_name,
                                                       length, frequency_bin):
    api._native()
    lib = api._LIB
    beta = _kaiser_alpha(200) * np.pi
    window = np.empty(length, dtype=np.float64)
    expected_cr, expected_ci = np.empty_like(window), np.empty_like(window)
    assert lib.generate_kaiser_series(api._pointer(window), length, beta) == 0
    generator = getattr(lib, generator_name)
    assert generator(api._pointer(expected_cr), api._pointer(expected_ci),
                     api._pointer(window), length, frequency_bin) == 0
    workspace = np.memmap(tmp_path / "alias.bin", mode="w+", dtype=np.complex128,
                          shape=(length,))
    try:
        doubles = workspace.view(np.float64)
        cr, ci = doubles[:length], doubles[length:]
        assert lib.generate_kaiser_series(api._pointer(cr), length, beta) == 0
        sums = []
        for candidate in (window, cr):
            s1, s2 = ct.c_double(), ct.c_double()
            assert lib.window_sums(api._pointer(candidate), length, ct.byref(s1), ct.byref(s2)) == 0
            sums.append((s1.value, s2.value))
        assert sums[0] == sums[1]
        assert generator(api._pointer(cr), api._pointer(ci), api._pointer(cr),
                         length, frequency_bin) == 0
        assert cr.tobytes() == expected_cr.tobytes()
        assert ci.tobytes() == expected_ci.tobytes()
    finally:
        workspace._mmap.close()


@pytest.mark.parametrize("workers", (1, 8))
def test_nsd_only_recomputes_and_never_uses_optional_fma(tmp_path, monkeypatch, workers):
    values = _signal(1024, "white")
    changed = _signal(1024, "dc_nanovolt")
    expected = _reference(changed)
    with _mapped_arrays(tmp_path, values) as (data, workspace):
        with DiskLPSDSubset(len(data), FS, _plan(len(data)), workers=workers) as method:
            def forbidden(*args):
                raise AssertionError("The disk path must retain ordinary segment arithmetic.")

            monkeypatch.setattr(method._lib, "fast_dft_selected_bounded", forbidden)
            first = method.compute(data, workspace, outputs="psd")
            saved = first.psd.to_numpy().copy()
            data[:] = changed
            second = method.compute(data, workspace, outputs="nsd", profile=False)
            assert tuple(second.columns) == ("nsd",)
            _assert_same_spectrum(second.nsd, expected.nsd)
            assert first.psd.to_numpy().tobytes() == saved.tobytes()


def test_exception_returns_every_gate_and_scratch_reservation(tmp_path, monkeypatch):
    pools, gates = [], []
    original_pool, original_gate = disk_module._ScratchPool, api._MemoryGate

    class TrackingPool(original_pool):
        def __init__(self, *args):
            super().__init__(*args)
            pools.append(self)

    class TrackingGate(original_gate):
        def __init__(self, *args):
            super().__init__(*args)
            gates.append(self)

    monkeypatch.setattr(disk_module, "_ScratchPool", TrackingPool)
    monkeypatch.setattr(api, "_MemoryGate", TrackingGate)
    values = _signal(257, "white")
    with _mapped_arrays(tmp_path, values) as (data, workspace):
        with DiskLPSDSubset(len(data), FS, _plan(len(data)), workers=8) as method:
            original_generate = method._lib.generate_coefficients_blocked
            guard = threading.Lock()
            failed = False

            def fail_once(*args):
                nonlocal failed
                with guard:
                    if not failed:
                        failed = True
                        return 3
                return original_generate(*args)

            with monkeypatch.context() as local:
                local.setattr(method._lib, "generate_coefficients_blocked", fail_once)
                with pytest.raises(RuntimeError, match="native allocation failed"):
                    method.compute(data, workspace)
            assert pools and gates
            assert all(pool.used_slots == 0 and pool._free == [(0, len(data))] for pool in pools)
            assert all(gate.used == 0 for gate in gates)
            result = method.compute(data, workspace)
            _assert_same_spectrum(result.psd, _reference(values).psd)


def test_distinct_mappings_of_overlapping_file_are_rejected(tmp_path):
    n = 1024
    path = tmp_path / "shared.bin"
    workspace = np.memmap(path, mode="w+", dtype=np.complex128, shape=(n,))
    data = np.memmap(path, mode="r+", dtype=np.float64, shape=(n,))
    data[:] = _signal(n, "white")
    saved = data.copy()
    try:
        assert not np.shares_memory(data, workspace)
        with DiskLPSDSubset(n, FS, _plan(n)) as method:
            with pytest.raises(ValueError, match="must not overlap"):
                method.compute(data, workspace)
        assert data.tobytes() == saved.tobytes()
    finally:
        data._mmap.close()
        workspace._mmap.close()


def test_readonly_workspace_is_rejected_before_native_write(tmp_path):
    values = _signal(257, "white")
    with _mapped_arrays(tmp_path, values) as (data, workspace):
        readonly = np.memmap(workspace.filename, mode="r", dtype=np.complex128,
                             shape=workspace.shape)
        try:
            with DiskLPSDSubset(len(data), FS, _plan(len(data))) as method:
                with pytest.raises(ValueError, match="must be writable"):
                    method.compute(data, readonly)
            assert data.tobytes() == values.tobytes()
        finally:
            readonly._mmap.close()


def test_finite_validation_is_chunked_and_checks_final_block(tmp_path, monkeypatch):
    values = _signal(1024, "white")
    with _mapped_arrays(tmp_path, values) as (data, workspace):
        with DiskLPSDSubset(len(data), FS, _plan(len(data)),
                            validation_chunk_size=31) as method:
            original = np.isfinite
            checked = []

            def track(array, *args, **kwargs):
                if isinstance(array, np.ndarray):
                    checked.append(array.size)
                return original(array, *args, **kwargs)

            data[-1] = np.nan
            monkeypatch.setattr(disk_module.np, "isfinite", track)
            with pytest.raises(ValueError, match="must be finite"):
                method.compute(data, workspace)
            assert checked and max(checked) <= 31
            assert sum(checked) == len(data)
