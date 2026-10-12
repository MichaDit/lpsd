"""Focused correctness/resource gates for the exact-length Bluestein adapter.

All transforms are small. These tests intentionally contain no speed threshold
or repeat-timing loop; the separate benchmark chooses useful convolution sizes.
"""
import ctypes as ct
import json
import os

import numpy as np
import pandas as pd
import pytest

from benchmarks import bench_fftw as adapter
from benchmarks._fftw_bluestein import BluesteinRealFFT
from lpsd_fast import api


@pytest.fixture(scope="module")
def fftw():
    try:
        return adapter.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as exc:
        pytest.skip(f"Optional FFTW shared library unavailable: {exc}")


def _naive_r2c(values):
    """Independent direct DFT, used only for N <= 8 (no chirp identity)."""
    n = len(values)
    phase = -2j * np.pi * np.arange(n // 2 + 1)[:, None] * np.arange(n) / n
    return np.sum(values * np.exp(phase), axis=1)


@pytest.mark.parametrize("n,m,mode", [
    (2, 3, "numpy"),       # Smallest input; minimum cyclic convolution.
    (3, 4, "native"),      # Odd original N; last r2c value remains complex.
    (5, 7, "native"),      # Both original N and minimum M are prime.
    (8, 13, "numpy"),      # Even N and deliberately awkward prime M.
    (31, 47, "native"),
    (32, None, "numpy"),   # Preserve the existing default power-of-two M.
    (127, 190, "numpy"),   # Minimum M with an odd prime original N.
    (128, 193, "native"),  # Prime M, beyond its minimum of 192.
])
def test_original_dft_for_minimal_unusual_and_default_convolution(fftw, n, m, mode):
    rng = np.random.default_rng(20261010 + n)
    impulse = np.zeros(n)
    impulse[-1] = 1.0
    values_to_test = [rng.normal(size=n), impulse,
                      np.cos(2 * np.pi * (n // 2) * np.arange(n) / n)]
    with BluesteinRealFFT(fftw, n, convolution_length=m, native_ops=mode) as fft:
        assert fft.native_ops == mode
        assert fft.m == (m if m is not None else 1 << (2 * n - 2).bit_length())
        assert fft.y.shape == (n // 2 + 1,)
        for values in values_to_test:
            fft.x[:] = values
            saved = fft.x.copy()
            fft.execute()
            scale = max(float(np.sum(np.abs(values))), 1.0)
            # Absolute error protects bins near nulls without dividing by
            # an arbitrarily small reference coefficient.
            np.testing.assert_allclose(fft.y, np.fft.rfft(values),
                                       rtol=5e-13, atol=5e-14 * scale)
            if n <= 8:
                np.testing.assert_allclose(fft.y, _naive_r2c(values),
                                           rtol=5e-13, atol=5e-14 * scale)
            np.testing.assert_array_equal(fft.x, saved)
            assert fft.y[0].imag == 0.0
            if n % 2 == 0:
                assert fft.y[-1].imag == 0.0
        # A reused inverse-transform workspace must have its tail cleared.
        fft.x.fill(0.0)
        fft.execute()
        np.testing.assert_array_equal(fft.y, np.zeros(n // 2 + 1))


def test_native_fusion_and_numpy_keep_the_same_dft(fftw):
    n, m = 127, 191
    values = np.random.default_rng(911).normal(size=n)
    with BluesteinRealFFT(fftw, n, convolution_length=m, native_ops="numpy") as baseline, \
         BluesteinRealFFT(fftw, n, convolution_length=m, native_ops="native") as native:
        baseline.x[:] = native.x[:] = values
        baseline.execute()
        native.execute()
        # Complex multiplication and FFTW plans can differ in final rounding.
        np.testing.assert_allclose(native.y, baseline.y, rtol=3e-13,
                                   atol=3e-14 * np.sum(np.abs(values)))
        assert native.describe()["native_operations"]["abi_version"] == 1


@pytest.mark.parametrize("n,m", [(127, 190), (128, 193), (8191, 12286)])
def test_full_psd_nsd_dc_noise_and_offbin_tone(fftw, n, m):
    rng = np.random.default_rng(20261010 + n)
    signals = [
        (10.0 + 1e-9 * rng.normal(size=n), 1e-23),
        (np.cos(2 * np.pi * .12345 * np.arange(n) + .17), 3e-14),
    ]
    kwargs = dict(sample_rate=50.0, psll=200.0, n_frequencies=32,
                  n_averages=8, window_provider=adapter.KaiserWindow("native"),
                  operations="native", output_dtype="float64", density="both")
    with adapter.Periodogram(fftw, n, **kwargs) as ordinary, \
         adapter.Periodogram(fftw, n, fft_algorithm="bluestein",
                             fft_convolution_length=m, **kwargs) as bluestein:
        assert isinstance(bluestein.fft, BluesteinRealFFT)
        assert bluestein.fft.n == n
        np.testing.assert_array_equal(bluestein.frequencies, ordinary.frequencies)
        for signal, nsd_absolute_limit in signals:
            values = signal[::-1]  # Also exercise the owning pipeline's copy.
            original = values.copy()
            expected = ordinary.compute(values)
            actual = bluestein.compute(values)
            assert list(actual.columns) == ["psd", "nsd"]
            np.testing.assert_array_equal(actual.index, expected.index)
            assert np.max(np.abs(actual.nsd - expected.nsd)) <= nsd_absolute_limit
            # Check PSD itself, allowing the matching absolute NSD bound at
            # tiny leakage/null bins; relative-only assertions are unsuitable.
            psd_error_bound = (actual.nsd + expected.nsd) * nsd_absolute_limit + nsd_absolute_limit ** 2
            assert np.all(np.abs(actual.psd - expected.psd) <= psd_error_bound)
            np.testing.assert_allclose(actual.nsd ** 2, actual.psd, rtol=4e-15, atol=0)
            check = bluestein.validation()
            assert check["finite"]
            assert check["parseval_absolute_error"] <= 3e-13 * check["weighted_detrended_time_power"]
            np.testing.assert_array_equal(values, original)
            saved = actual.copy(deep=True)
            bluestein.compute(np.zeros(n))
            pd.testing.assert_frame_equal(actual, saved, check_exact=True)


def test_float32_joint_output_keeps_complex64_sqrt_route(fftw):
    values = 10.0 + 1e-9 * np.random.default_rng(556).normal(size=127)
    with adapter.Periodogram(
        fftw, 127, sample_rate=50.0, n_frequencies=12, n_averages=4,
        window_provider=adapter.KaiserWindow("native"), operations="native_grouped",
        fft_algorithm="bluestein", fft_convolution_length=191,
        density="both", output_dtype="float32",
    ) as pipeline:
        result = pipeline.compute(values)
        assert result.psd.dtype == result.nsd.dtype == np.dtype("float32")
        np.testing.assert_array_equal(
            result.nsd, np.sqrt(result.psd.to_numpy().astype(np.complex64)).real)
        assert pipeline.validation()["finite"]


def test_measure_plans_are_initialized_before_kernel_preparation(fftw):
    with BluesteinRealFFT(fftw, 31, convolution_length=47, planner="measure",
                          time_limit=.01, native_ops="numpy") as fft:
        fft.x[:] = np.random.default_rng(88).normal(size=31)
        expected = np.fft.rfft(fft.x)
        fft.execute()
        np.testing.assert_allclose(fft.y, expected, rtol=3e-13, atol=3e-13)
        description = fft.describe()
        assert description["fft_executions_per_call"] == 2
        assert description["kernel_fft_executions_during_setup"] == 1
        assert description["time_limit_scope"] == "per C2C plan"
        json.dumps(description, allow_nan=False)


def test_native_optional_fallback_and_required_mode(fftw):
    api._native()
    with BluesteinRealFFT(fftw, 31, native_ops="auto", operations_library=api._LIB) as fft:
        assert fft.native_ops == "native"
    # FFTW is an actual shared library but deliberately lacks our additive ABI.
    with BluesteinRealFFT(fftw, 31, native_ops="auto", operations_library=fftw.lib) as fft:
        assert fft.native_ops == "numpy"
        assert fft.native_ops_metadata["fallback_reason"]
        fft.x[:] = np.arange(31)
        fft.execute()
        np.testing.assert_allclose(fft.y, np.fft.rfft(np.arange(31)), rtol=3e-13, atol=3e-13)
    with pytest.raises(RuntimeError, match="Native Bluestein operations are unavailable"):
        BluesteinRealFFT(fftw, 31, native_ops="native", operations_library=fftw.lib)


@pytest.mark.parametrize("options,error", [
    ({"n": 1}, ValueError),
    ({"n": True}, TypeError),
    ({"n": 3.5}, TypeError),
    ({"convolution_length": 45}, ValueError),
    ({"convolution_length": 47.0}, TypeError),
    ({"convolution_length": 1 << 31}, ValueError),
    ({"n": 1 << 30}, ValueError),
    ({"n": 1_000_003, "max_working_mb": 1.0}, MemoryError),
    ({"max_working_mb": float("inf")}, ValueError),
    ({"chirp_block_size": 0}, ValueError),
    ({"planner": "unknown"}, ValueError),
    ({"time_limit": -1.0}, ValueError),
    ({"native_ops": "unknown"}, ValueError),
])
def test_constructor_guards_precede_backend_access_and_allocation(options, error):
    arguments = {"n": 31, "native_ops": "numpy"}
    arguments.update(options)
    # None would fail on any backend/FFTW access: every rejection must happen
    # before such access, including the large-memory and integer-limit cases.
    with pytest.raises(error):
        BluesteinRealFFT(None, **arguments)


def test_memory_bound_and_close_invalidate_bound_pointers(fftw):
    n, m, block = 31, 47, 7
    estimate = BluesteinRealFFT.memory_estimate(n, block, m)
    with pytest.raises(MemoryError):
        BluesteinRealFFT(None, n, convolution_length=m, chirp_block_size=block,
                         max_working_mb=(estimate["known_peak_buffer_bytes"] - 1) / 1024 ** 2)
    fft = BluesteinRealFFT(fftw, n, convolution_length=m, chirp_block_size=block,
                           max_working_mb=estimate["known_peak_buffer_bytes"] / 1024 ** 2,
                           native_ops="native")
    try:
        actual_arrays = (fft.x, fft.y, fft._chirp, fft._work, fft._kernel_fft)
        assert sum(array.nbytes for array in actual_arrays) == estimate["persistent_buffer_bytes"]
        assert estimate["chirp_setup_scratch_bytes"] == 17 * block
        assert estimate["minimum_convolution_length"] == n + n // 2
        assert fft._native_prepare_args is not None
        # Do not retain views into the FFTW allocations through close().
        del actual_arrays
    finally:
        fft.close()
    fft.close()
    assert fft.plan is fft.inverse_plan is None
    assert fft.x is fft.y is None
    assert fft._native_prepare_args is fft._native_finish_args is None
    assert all(getattr(fft, name) is None for name in ("ip", "op", "wp", "kp"))
    with pytest.raises(RuntimeError, match="closed"):
        fft.execute()
    with pytest.raises(RuntimeError, match="closed"):
        fft.describe()


def test_partial_plan_failure_frees_existing_plan_and_allocations(fftw, monkeypatch):
    lib = fftw.lib
    real_malloc, real_free = lib.fftw_malloc, lib.fftw_free
    real_plan, real_destroy = lib.fftw_plan_dft_1d, lib.fftw_destroy_plan
    real_plan.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_uint]
    real_plan.restype = ct.c_void_p
    allocations, freed, plans, destroyed = [], [], [], []

    def malloc(size):
        pointer = real_malloc(size)
        allocations.append(pointer)
        return pointer

    def free(pointer):
        freed.append(pointer)
        real_free(pointer)

    def plan(n, input_pointer, output_pointer, sign, flags):
        if sign == +1:
            return None
        pointer = real_plan(n, input_pointer, output_pointer, sign, flags)
        plans.append(pointer)
        return pointer

    def destroy(pointer):
        destroyed.append(pointer)
        real_destroy(pointer)

    monkeypatch.setattr(lib, "fftw_malloc", malloc)
    monkeypatch.setattr(lib, "fftw_free", free)
    monkeypatch.setattr(lib, "fftw_plan_dft_1d", plan)
    monkeypatch.setattr(lib, "fftw_destroy_plan", destroy)
    with pytest.raises(RuntimeError, match="backward plan creation failed"):
        BluesteinRealFFT(fftw, 31, native_ops="numpy")
    assert len(allocations) == 4 and all(allocations)
    assert sorted(allocations) == sorted(freed)
    assert len(plans) == 1 and plans == destroyed


def test_factory_selection_is_explicit_and_preserves_default(fftw):
    with adapter.create_real_fft(fftw, 31) as fft:
        assert isinstance(fft, adapter.RealFFT)
    with adapter.create_real_fft(fftw, 31, algorithm="bluestein",
                                 convolution_length=46) as fft:
        assert isinstance(fft, BluesteinRealFFT)
        assert fft.n == 31 and fft.m == 46 and len(fft.y) == 16
    with pytest.raises(ValueError, match="requires"):
        adapter.create_real_fft(None, 31, convolution_length=46)
    with pytest.raises(ValueError, match="native or bluestein"):
        adapter.create_real_fft(None, 31, algorithm="automatic")
