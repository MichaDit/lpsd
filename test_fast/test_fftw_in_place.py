"""Numerical, ownership and memory gates for the optional shared FFTW buffer."""
import os

import numpy as np
import pytest

from benchmarks import bench_fftw as bf
from benchmarks._fftw_native import FFTWOperations
from benchmarks.bench_fftw_large_n import bounded_parseval, memory_lower_bound
from benchmarks.matched_smoothing import MatchedSmoothing, estimate_once
from lpsd_fast import api


@pytest.fixture(scope="module")
def fftw():
    try:
        return bf.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as error:
        pytest.skip(f"Optional FFTW shared library unavailable: {error}")


@pytest.mark.parametrize("n", [2, 3, 31, 32, 257, 4096])
def test_in_place_original_dft_endpoints_refill_and_close(fftw, n):
    original = np.random.default_rng(410 + n).normal(size=n)
    impulse = np.zeros(n)
    impulse[-1] = 1.
    transform = bf.RealFFT(fftw, n, in_place=True)
    assert np.shares_memory(transform.x, transform.y)
    assert transform.x.ctypes.data == transform.y.ctypes.data
    assert transform.allocated_bytes == 16 * (n // 2 + 1)
    try:
        for source in (original, impulse, original):
            saved = source.copy()
            transform.x[:] = source
            transform.execute()
            np.testing.assert_allclose(transform.y, np.fft.rfft(source),
                                       rtol=2e-12, atol=1e-12 * np.linalg.norm(source))
            np.testing.assert_array_equal(source, saved)
        assert transform.describe()["input_preserved_after_execute"] is False
    finally:
        transform.close()
        transform.close()
    assert transform.ip is transform.op is transform.plan is None
    assert transform.x is transform.y is None


@pytest.mark.parametrize("kind", ["fftw", "matched"])
@pytest.mark.parametrize("signal", ["white", "dc_nanovolt", "offbin", "constant"])
def test_complete_shared_buffer_estimator_and_input_ownership(fftw, kind, signal):
    api._native()
    n = 8192
    rng = np.random.default_rng(88421)
    x = rng.normal(size=n)
    if signal == "dc_nanovolt":
        x = 10. + 1e-9 * x
    elif signal == "offbin":
        x = np.sin(2 * np.pi * 571.37 * np.arange(n) / n) + 1e-8 * x
    elif signal == "constant":
        x.fill(10.)
    saved = x.copy()
    common = dict(sample_rate=50., n_frequencies=80, n_averages=10,
                  window_provider=bf.KaiserWindow("native", api._LIB._name))
    if kind == "fftw":
        constructor = bf.Periodogram
        common.update(operations="native_grouped", operations_library=api._LIB)
    else:
        constructor = MatchedSmoothing
        common.update(workers=1, smoothing_workers=1, operations_backend="native",
                      smoothing_backend="native", weight_dtype="float32",
                      max_kernel_cache_mb=0, total_cache_mb=0)
    with constructor(fftw, n, **common) as reference:
        expected = reference.compute(x, outputs=("psd", "nsd"))
        original_bytes = (reference.metadata["principal_buffer_bytes"] if kind == "fftw"
                          else reference.metadata["fftw_array_bytes"])
    with constructor(fftw, n, **common, fft_in_place=True) as compact:
        for source in (x, x[::-1], x):
            actual = compact.compute(source, outputs=("psd", "nsd"))
            check = compact.validation(source)
            err = check.get("parseval_absolute_error", check.get("parseval_absolute_error_v2"))
            power = check.get("weighted_detrended_time_power", check.get("global_windowed_time_power_v2"))
            assert err <= 3e-12 * power
        np.testing.assert_array_equal(actual.index, expected.index)
        for column in ("psd", "nsd"):
            np.testing.assert_allclose(actual[column], expected[column], rtol=3e-6, atol=0.)
        np.testing.assert_array_equal(actual.nsd, np.sqrt(actual.psd.to_numpy().astype(np.complex64)).real)
        reduced_bytes = (compact.metadata["principal_buffer_bytes"] if kind == "fftw"
                         else compact.metadata["fftw_array_bytes"])
        assert original_bytes - reduced_bytes == 8 * n
        with pytest.raises(ValueError, match="unchanged input"):
            compact.validation()
        if signal != "constant":
            bounded_parseval(compact, samples=x)
    np.testing.assert_array_equal(x, saved)


def test_in_place_binding_only_allows_exact_fft_alias(fftw):
    api._native()
    ops = FFTWOperations("native", api._LIB)
    n = 32
    with bf.RealFFT(fftw, n, in_place=True) as fft:
        args = dict(time_buffer=fft.x, window=np.ones(n), transform=fft.y,
                    n=n, sample_rate=50., window_square_sum=n,
                    powers=np.empty(n // 2 + 1))
        with pytest.raises(ValueError, match="must not overlap"):
            ops.bind(**args)
        bound = ops.bind(**args, allow_in_place_fft=True)
        bound.close()
        with pytest.raises(ValueError, match="must not overlap"):
            ops.bind(**{**args, "window": fft.x}, allow_in_place_fft=True)
        # Shifted views are not FFTW's padded, same-origin r2c layout.
        with pytest.raises(ValueError, match="must not overlap"):
            ops.bind(**{**args, "time_buffer": fft.y.view(np.float64)[1:n + 1]},
                     allow_in_place_fft=True)
    with pytest.raises(ValueError, match="requires fft_algorithm"):
        bf.create_real_fft(fftw, 31, algorithm="bluestein", in_place=True)


def test_in_place_billion_sample_memory_model_without_allocation():
    n = 1_000_000_000
    for method in ("fftw", "matched", "matched_once"):
        old, _ = memory_lower_bound(method, n)
        new, _ = memory_lower_bound(method, n, fft_in_place=True)
        assert old - new == 8 * n
    assert memory_lower_bound("lpsd", n, True) == memory_lower_bound("lpsd", n)


@pytest.mark.parametrize("in_place", [False, True])
def test_bounded_one_shot_releases_fft_before_native_subset(fftw, monkeypatch, in_place):
    api._native()
    from benchmarks import matched_smoothing as module
    n = 16384
    values = 10. + 1e-9 * np.random.default_rng(414).normal(size=n)
    saved = values.copy()
    options = dict(n_frequencies=80, n_averages=10, workers=2,
                   smoothing_workers=1, total_cache_mb=0, max_working_mb=.5,
                   fft_in_place=in_place, operations_backend="native",
                   smoothing_backend="native", weight_dtype="float32")
    expected = estimate_once(fftw, values, outputs=("psd", "nsd"),
                             **options, release_fft_before_low=False)
    observed = []
    original = module.MatchedSmoothing

    class Observed(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            compute = self._low_prepared.compute

            def checked(samples, **kw):
                assert self.fft is self._bound_operations is self._native_smoothing is None
                assert self.window is self.powers is self.scratch is None
                assert kw["workspace"] is None
                observed.append(True)
                return compute(samples, **kw)

            self._low_prepared.compute = checked

    monkeypatch.setattr(module, "MatchedSmoothing", Observed)
    actual = module.estimate_once(fftw, values, outputs=("psd", "nsd"),
                                  **options, low_frequency_backend="bounded")
    assert observed == [True]
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=0.)
    np.testing.assert_array_equal(values, saved)
    metadata = actual.attrs["matched_smoothing_once"]
    assert metadata["fft_released_before_low_lpsd"]
    assert metadata["low_frequency_preparation"]["benchmark_category"] == "bounded_memory_complete_call"
    assert metadata["low_frequency_preparation"]["total_mapped_bytes"] == 0
