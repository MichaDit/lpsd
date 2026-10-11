"""Numerical and memory contracts for the unchanged FFTW-hybrid estimator.

The frozen pre-optimization adapter is the independent end-to-end reference.
Small deterministic inputs cover spectral shapes, odd/even FFT endpoints,
cache exhaustion, ownership, and the public float32-PSD to complex64-NSD route.
These checks do not benchmark either implementation.
"""
from __future__ import annotations

import gc
import os
from types import SimpleNamespace
import weakref

import numpy as np
import pandas as pd
import pytest

from benchmarks import bench_fftw
from benchmarks import matched_smoothing as smoothing_module
from benchmarks.matched_smoothing import (
    MatchedSmoothing, _Kernel, _NativeSmoothing, estimate_once, kaiser_power_response,
)
from benchmarks.reference.matched_smoothing_20261010 import (
    MatchedSmoothing as OriginalSmoothing,
)
from lpsd_fast import api


@pytest.fixture(scope="module")
def fftw():
    try:
        return bench_fftw.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as error:
        pytest.skip(f"Optional FFTW shared library unavailable: {error}")


@pytest.fixture(scope="module")
def native_smoothing():
    api._native()
    # A rebuilt native library is required for this additive implementation.
    return _NativeSmoothing(api._LIB, backend="native")


def _signals(n):
    rng = np.random.default_rng(611 + n)
    white = rng.normal(size=n) * 5e-8
    frequency = np.fft.rfftfreq(n, d=1 / 50.0)
    z = np.fft.rfft(rng.normal(size=n))
    z[0] = 0.0
    scale = np.maximum(frequency, 50 / n)
    pink = np.fft.irfft(z / np.sqrt(scale), n=n) * 1e-8
    brown = np.fft.irfft(z / scale, n=n) * 1e-9
    t = np.arange(n) / 50.0
    tone = np.sqrt(2) * 1e-6 * np.sin(2 * np.pi * (int(.06246 * n) + .37) * np.arange(n) / n)
    mixture = (white + pink + brown + np.sqrt(2) * 5e-8 * np.sin(2 * np.pi * 3.123456 * t)
               + np.sqrt(2) * 2e-8 * np.sin(2 * np.pi * 12.34567 * t))
    return {
        "white": white, "pink": pink, "brown": brown,
        "mixed_tones": mixture, "offbin_tone": tone,
        "dc_nanovolt": 10.0 + 1e-9 * rng.normal(size=n),
    }


def test_measured_bluestein_hybrid_retains_original_density(fftw, native_smoothing):
    """Planning may overwrite buffers; the complete hybrid must initialize them again."""
    n = 2049
    common = dict(sample_rate=50., psll=200., n_frequencies=64, n_averages=16,
                  workers=2, max_kernel_cache_mb=4, window_cache_mb=4,
                  max_working_mb=32)
    with OriginalSmoothing(fftw, n, **common) as before, MatchedSmoothing(
            fftw, n, fft_algorithm="bluestein", fft_convolution_length=4096,
            planner="measure", time_limit=.02, **common) as after:
        assert after.metadata["planner"] == "measure"
        assert after.metadata["planner_time_limit_s"] == .02
        for case in ("white", "offbin_tone", "dc_nanovolt"):
            x = _signals(n)[case]
            expected = before.compute(x, outputs=("psd", "nsd"))
            actual = after.compute(x, outputs=("psd", "nsd"))
            pd.testing.assert_index_equal(actual.index, expected.index)
            for output in ("psd", "nsd"):
                peak = np.max(expected[output])
                np.testing.assert_allclose(actual[output], expected[output], rtol=3e-7,
                                           atol=2e-14 * peak)


@pytest.mark.parametrize("n", [256, 257])
@pytest.mark.parametrize("beta", [0.0, 25.402566024390598])
@pytest.mark.parametrize("frequency", [0.0, 0.125, 24.875, 25.0])
def test_native_full_kernel_support_reflection_and_endpoint_normalization(
        native_smoothing, n, beta, frequency):
    fs, length, halfwidth = 50.0, 64, 9.0
    first, stop = 0, n // 2 + 1
    owner = SimpleNamespace(n=n, sample_rate=fs, beta=beta, halfwidth=halfwidth)
    cached = _Kernel(0, first, stop, length, frequency, 0.0,
                     np.empty(stop - first, dtype=np.float64))
    uncached = _Kernel(0, first, stop, length, frequency, 0.0, None)
    native_smoothing.prepare(owner, cached)
    native_smoothing.prepare(owner, uncached)
    f = np.arange(first, stop) * (fs / n)
    direct = (f - frequency) * (length / fs)
    reflected = np.minimum(f + frequency, fs - (f + frequency)) * (length / fs)
    weights = (kaiser_power_response(direct, beta, halfwidth)
               + kaiser_power_response(reflected, beta, halfwidth))
    denominator = float(weights.sum() - .5 * weights[0])
    if n % 2 == 0:
        denominator -= .5 * float(weights[-1])
    np.testing.assert_allclose(cached.weights, weights / denominator, rtol=5e-13, atol=1e-29)
    assert cached.denominator == pytest.approx(denominator, rel=3e-14)
    assert uncached.denominator == pytest.approx(cached.denominator, rel=3e-14)
    np.testing.assert_array_equal(cached.weights[weights == 0], 0)

    # Interior unit density plus correctly halved real-transform endpoints
    # must remain unit density even when reflected support reaches an edge.
    power = np.ones(n // 2 + 1)
    power[0] *= .5
    if n % 2 == 0:
        power[-1] *= .5
    owner.powers = power
    for kernel in (cached, uncached):
        native_smoothing.bind([kernel])
        result = np.empty(1)
        native_smoothing.apply(owner, 0, 1, result)
        assert result[0] == pytest.approx(1.0, rel=3e-14)


@pytest.mark.parametrize("n", [2048, 2049])
@pytest.mark.parametrize("cache_mb", [0.0, .003, 4.0])
@pytest.mark.parametrize("weight_dtype", ["float64", "float32"])
@pytest.mark.parametrize("defer_normalization", [False, True])
def test_all_six_signals_match_frozen_estimator(
        fftw, native_smoothing, n, cache_mb, weight_dtype, defer_normalization):
    common = dict(sample_rate=50.0, psll=200, n_frequencies=64, n_averages=16,
                  workers=2, max_kernel_cache_mb=cache_mb, window_cache_mb=4,
                  max_working_mb=32, block_bins=127)
    with OriginalSmoothing(fftw, n, **common) as before, MatchedSmoothing(
            fftw, n, operations_backend="native", smoothing_backend="native",
            weight_dtype=weight_dtype, defer_normalization=defer_normalization, **common) as after:
        assert after.metadata["smoothing_backend"] == "native"
        assert after._last_low_metadata is not None
        np.testing.assert_array_equal(after.frequencies, before.frequencies)
        np.testing.assert_array_equal(after.low_mask, before.low_mask)
        np.testing.assert_array_equal(after.lengths, before.lengths)
        np.testing.assert_array_equal(after.enbw, before.enbw)
        np.testing.assert_array_equal(after.window, before.window)
        pending = after.metadata["initially_deferred_frequency_kernels"]
        assert pending == (after.metadata["streamed_frequency_kernels"] if defer_normalization else 0)
        assert sum(np.isnan(k.denominator) for k in after._kernels) == pending
        for name, values in _signals(n).items():
            original_input = values.copy()
            expected = before.compute(values, outputs=("psd", "nsd"))
            actual = after.compute(values, outputs=("psd", "nsd"))
            np.testing.assert_array_equal(actual.index, expected.index)
            for column in ("psd", "nsd"):
                assert actual[column].dtype == np.float32
                # This is an FP32-rounding-sized numerical gate, not a relaxed
                # 1%-spectral-shape test. Absolute leakage is retained throughout.
                np.testing.assert_allclose(actual[column], expected[column],
                                           rtol=3 * np.finfo(np.float32).eps, atol=0,
                                           err_msg=f"{name}, {column}, cache={cache_mb}")
            np.testing.assert_array_equal(
                actual.nsd, np.sqrt(actual.psd.to_numpy().astype(np.complex64)).real)
            np.testing.assert_array_equal(values, original_input)
            assert not after._pending_normalizations
            assert all(np.isfinite(k.denominator) and k.denominator > 0 for k in after._kernels)
        used = sum(kernel.weights.nbytes for kernel in after._kernels if kernel.weights is not None)
        assert used == after.metadata["cached_kernel_bytes"]
        assert used <= after.metadata["kernel_cache_budget_bytes"]
        assert all(kernel.weights.dtype == np.dtype(weight_dtype)
                   for kernel in after._kernels if kernel.weights is not None)
        if cache_mb == 0:
            assert used == 0
            assert all(kernel.weights is None for kernel in after._kernels)
        if cache_mb == .003:
            assert 0 < after.metadata["cached_frequency_kernels"] < len(after._kernels)


@pytest.mark.parametrize("full_window", ["rectangular", "kaiser", "tukey"])
def test_window_options_and_numpy_fallback_keep_same_estimator(fftw, native_smoothing, full_window):
    values = _signals(2049)["offbin_tone"]
    common = dict(n_frequencies=64, n_averages=16, workers=1,
                  max_kernel_cache_mb=.004, window_cache_mb=4, max_working_mb=32,
                  full_window=full_window, taper_fraction=.13, block_bins=83)
    with MatchedSmoothing(fftw, len(values), smoothing_backend="numpy",
                          operations_backend="numpy", **common) as numpy_path:
        expected = numpy_path.compute(values, outputs=("psd", "nsd"))
    with MatchedSmoothing(fftw, len(values), smoothing_backend="native",
                          operations_backend="native", **common) as native_path:
        actual = native_path.compute(values, outputs=("psd", "nsd"))
        assert native_path.scratch is None
        np.testing.assert_allclose(actual, expected, rtol=3 * np.finfo(np.float32).eps, atol=0)


def test_uncached_native_compute_does_not_regenerate_numpy_arrays(fftw, native_smoothing, monkeypatch):
    values = _signals(2048)["mixed_tones"]
    with MatchedSmoothing(fftw, len(values), n_frequencies=64, n_averages=16,
                          workers=2, max_kernel_cache_mb=0, window_cache_mb=4,
                          max_working_mb=32, operations_backend="native",
                          smoothing_backend="native") as pipeline:
        def forbidden(*args):
            pytest.fail("Native streamed smoothing must not allocate NumPy response arrays")
        monkeypatch.setattr(pipeline, "_raw_weights", forbidden)
        result = pipeline.compute(values)
        assert np.isfinite(result.psd).all()
        assert pipeline.metadata["streamed_kernel_temporary_bytes"] == 0


def test_repeated_outputs_own_data_and_close_releases_caches(fftw, native_smoothing):
    source = _signals(2049)["dc_nanovolt"]
    series = pd.Series(source[::-1], copy=False)
    saved_input = series.copy(deep=True)
    pipeline = MatchedSmoothing(fftw, len(source), n_frequencies=64, n_averages=16,
                               workers=2, max_kernel_cache_mb=4, window_cache_mb=4,
                               max_working_mb=32, operations_backend="native",
                               smoothing_backend="native")
    result = pipeline.compute(series, outputs=("psd", "nsd"))
    saved = result.copy(deep=True)
    nsd_only = pipeline.compute(series, outputs="nsd")
    np.testing.assert_array_equal(result.nsd, nsd_only.nsd)
    pipeline.compute(np.ones(len(source)))
    pd.testing.assert_frame_equal(result, saved, check_exact=True)
    pd.testing.assert_series_equal(series, saved_input, check_exact=True)
    references = [weakref.ref(k.weights) for k in pipeline._kernels if k.weights is not None]
    pipeline.close()
    pipeline.close()
    gc.collect()
    assert all(reference() is None for reference in references)
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.compute(series)
    pd.testing.assert_frame_equal(result, saved, check_exact=True)


def test_optional_profile_is_separate_from_untimed_compute(fftw, native_smoothing, monkeypatch):
    values = _signals(2048)["white"]
    with MatchedSmoothing(fftw, len(values), n_frequencies=64, n_averages=16,
                          workers=1, max_kernel_cache_mb=4, window_cache_mb=4,
                          max_working_mb=32, smoothing_backend="native",
                          operations_backend="native") as pipeline:
        expected = pipeline.compute(values, outputs="nsd")
        assert pipeline.last_profile is None
        actual = pipeline.profile(values, outputs="nsd")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert pipeline.last_profile["total_s"] > 0
        assert all(value >= 0 for value in pipeline.last_profile["phases"].values())
        saved_profile = pipeline.last_profile
        def forbidden():
            pytest.fail("Normal compute must not read profiling timers")
        monkeypatch.setattr(smoothing_module, "time", SimpleNamespace(perf_counter=forbidden))
        pd.testing.assert_frame_equal(pipeline.compute(values, outputs="nsd"), expected, check_exact=True)
        assert pipeline.last_profile is saved_profile


@pytest.mark.parametrize("kwargs", [
    {"max_kernel_cache_mb": -1}, {"window_cache_mb": -1},
    {"max_working_mb": 0}, {"block_bins": 0}, {"block_bins": 1.5},
    {"smoothing_backend": "invalid"}, {"operations_backend": "invalid"},
    {"defer_normalization": "yes"},
    {"total_cache_mb": -1}, {"total_cache_mb": np.nan},
    {"total_cache_mb": np.inf}, {"window_cache_mb": None},
])
def test_invalid_resource_options_fail_before_planning(fftw, kwargs):
    with pytest.raises(ValueError):
        MatchedSmoothing(fftw, 2048, n_frequencies=64, n_averages=16, **kwargs)


@pytest.mark.parametrize("n", [256, 257])
@pytest.mark.parametrize("beta", [0.0, 25.402566024390598])
@pytest.mark.parametrize("frequency", [0.0, .125, 24.875, 25.0])
@pytest.mark.parametrize("weight_dtype", ["float64", "float32"])
def test_deferred_native_reduction_keeps_endpoints_and_reuses_denominator(
        native_smoothing, n, beta, frequency, weight_dtype):
    owner = SimpleNamespace(n=n, sample_rate=50., beta=beta, halfwidth=9.)
    eager = _Kernel(0, 0, n // 2 + 1, 64, frequency, 0., None)
    native_smoothing.prepare(owner, eager)
    native_smoothing.bind([eager])
    deferred = _NativeSmoothing(api._LIB, "native", weight_dtype,
                                defer_normalization=True)
    deferred.bind([_Kernel(0, eager.first, eager.stop, eager.length,
                          frequency, np.nan, None)])
    assert np.isnan(deferred.denominators[0])
    owner.powers = np.linspace(.1, 2., n // 2 + 1)
    owner.powers[0] *= .5
    if n % 2 == 0:
        owner.powers[-1] *= .5
    expected, actual = np.empty(1), np.empty(1)
    native_smoothing.apply(owner, 0, 1, expected)
    deferred.apply(owner, 0, 1, actual)
    np.testing.assert_allclose(actual, expected, rtol=4e-14, atol=0)
    assert deferred.denominators[0] == pytest.approx(eager.denominator, rel=4e-14)
    saved = deferred.denominators.tobytes()
    owner.powers.fill(1.)
    owner.powers[0] *= .5
    if n % 2 == 0:
        owner.powers[-1] *= .5
    deferred.apply(owner, 0, 1, actual)
    assert actual[0] == pytest.approx(1., rel=4e-14)
    assert deferred.denominators.tobytes() == saved


def test_deferred_constructor_skips_uncached_formula_preparation(fftw, native_smoothing, monkeypatch):
    original_prepare = _NativeSmoothing.prepare
    calls = []

    def cached_only(binding, owner, kernel):
        assert kernel.weights is not None, "Uncached normalization must wait for compute"
        calls.append(kernel.output_index)
        return original_prepare(binding, owner, kernel)

    monkeypatch.setattr(_NativeSmoothing, "prepare", cached_only)
    with MatchedSmoothing(fftw, 2048, n_frequencies=64, n_averages=16,
                          workers=2, smoothing_workers=1, max_kernel_cache_mb=.003,
                          window_cache_mb=0, max_working_mb=32,
                          operations_backend="native", smoothing_backend="native",
                          defer_normalization=True) as pipeline:
        assert len(calls) == pipeline.metadata["cached_frequency_kernels"]
        assert pipeline.metadata["initially_deferred_frequency_kernels"] > 0
        assert pipeline.metadata["requested_smoothing_workers"] == 1
        assert pipeline.metadata["smoothing_workers"] == 1
        result = pipeline.compute(_signals(2048)["brown"])
        assert np.isfinite(result.psd).all()
        assert len(calls) == pipeline.metadata["cached_frequency_kernels"]


@pytest.mark.parametrize("outputs", ["psd", "nsd", ("psd", "nsd")])
def test_estimate_once_avoids_persistent_caches_and_closes(fftw, native_smoothing, monkeypatch, outputs):
    values = _signals(2049)["mixed_tones"]
    common = dict(n_frequencies=64, n_averages=16, workers=2,
                  operations_backend="native", smoothing_backend="native", max_working_mb=32)
    with MatchedSmoothing(fftw, len(values), max_kernel_cache_mb=0,
                          window_cache_mb=0, **common) as eager:
        expected = eager.compute(values, outputs=outputs)
    instances = []

    class ObservedSmoothing(MatchedSmoothing):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(smoothing_module, "MatchedSmoothing", ObservedSmoothing)
    actual = estimate_once(fftw, values, outputs=outputs,
                           max_kernel_cache_mb=8, window_cache_mb=8,
                           total_cache_mb=.1, **common)
    np.testing.assert_allclose(actual, expected, rtol=3 * np.finfo(np.float32).eps, atol=0)
    metadata = actual.attrs["matched_smoothing_once"]
    assert metadata["cached_kernel_bytes"] == metadata["kernel_cache_budget_bytes"] == 0
    assert metadata["low_frequency_preparation"]["cached_coefficient_bytes"] == 0
    assert metadata["low_frequency_cache_budget_bytes"] == 0
    assert metadata["persistent_cache_bytes"] == 0
    assert metadata["total_cache_budget_bytes"] == int(.1 * 1024**2)
    assert metadata["deferred_normalization"]
    assert instances[0]._closed
    assert instances[0]._executor is instances[0]._low_prepared is instances[0]._native_smoothing is None
    with pytest.raises(ValueError, match="finite"):
        estimate_once(fftw, np.full(len(values), np.nan), **common)
    assert instances[-1]._closed


def test_numpy_fallback_reports_eager_normalization_when_defer_requested(fftw, native_smoothing):
    values = _signals(2048)["white"]
    result = estimate_once(fftw, values, n_frequencies=64, n_averages=16,
                           workers=1, smoothing_backend="numpy", operations_backend="numpy",
                           max_working_mb=32)
    metadata = result.attrs["matched_smoothing_once"]
    assert metadata["deferred_normalization_requested"]
    assert not metadata["deferred_normalization"]
    assert metadata["initially_deferred_frequency_kernels"] == 0
    assert np.isfinite(result.psd).all()


@pytest.mark.parametrize("weight_dtype", ["float64", "float32"])
@pytest.mark.parametrize("total_mb,kernel_mb,low_mb", [
    (.031231, .021173, None),  # Allocate low coefficients from the actual remainder.
    (.011231, .051173, None),  # Clip the kernel cap to the combined cap first.
    (.031231, .021173, .001),  # Respect an explicitly smaller low cap.
    (0., 1., None),
    (.031231, 1., 0.),
])
def test_total_cache_budget_uses_whole_bytes_and_preserves_the_estimator(
        fftw, native_smoothing, weight_dtype, total_mb, kernel_mb, low_mb):
    values = _signals(2048)["mixed_tones"]
    common = dict(n_frequencies=64, n_averages=16, workers=2,
                  operations_backend="native", smoothing_backend="native",
                  max_working_mb=32, weight_dtype=weight_dtype,
                  defer_normalization=True)
    with MatchedSmoothing(fftw, len(values), total_cache_mb=total_mb,
                          max_kernel_cache_mb=kernel_mb, window_cache_mb=low_mb,
                          **common) as combined:
        metadata = combined.metadata
        total_bytes = int(total_mb * 1024**2)
        kernel_limit = min(int(kernel_mb * 1024**2), total_bytes)
        actual_weight_bytes = sum(k.weights.nbytes for k in combined._kernels if k.weights is not None)
        remaining = total_bytes - actual_weight_bytes
        low_limit = remaining if low_mb is None else min(int(low_mb * 1024**2), remaining)
        assert metadata["cache_budget_mode"] == "total"
        assert metadata["total_cache_budget_bytes"] == total_bytes
        assert metadata["kernel_cache_budget_bytes"] == kernel_limit
        assert metadata["cached_kernel_bytes"] == actual_weight_bytes <= kernel_limit
        assert metadata["remaining_cache_budget_after_weights_bytes"] == remaining
        assert metadata["low_frequency_cache_budget_bytes"] == low_limit
        low = metadata["low_frequency_preparation"]
        assert low["cache_limit_bytes"] == low_limit
        assert low["cached_coefficient_bytes"] <= low_limit
        assert metadata["persistent_cache_bytes"] == actual_weight_bytes + low["cached_coefficient_bytes"]
        assert metadata["persistent_cache_bytes"] <= total_bytes
        actual = combined.compute(values, outputs=("psd", "nsd"))
        # Match the effective caps through the original independent mode.
        # The combined policy must only choose storage, never spectral work.
        with MatchedSmoothing(fftw, len(values),
                              max_kernel_cache_mb=kernel_limit / 1024**2,
                              window_cache_mb=low_limit / 1024**2, **common) as independent:
            expected = independent.compute(values, outputs=("psd", "nsd"))
            assert independent.metadata["cache_budget_mode"] == "independent"
            assert independent.metadata["total_cache_budget_bytes"] is None
            assert actual.to_numpy().tobytes() == expected.to_numpy().tobytes()
            np.testing.assert_array_equal(actual.index, expected.index)
