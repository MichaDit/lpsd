"""Numerical and ownership gates for the opt-in one-shot FFT window storage."""
import os

import numpy as np
import pytest

from benchmarks import bench_fftw as bf
from benchmarks._fftw_native import FFTWOperations
from benchmarks.matched_smoothing import MatchedSmoothing, estimate_once
from lpsd_fast import api


@pytest.fixture(scope="module")
def fftw():
    try:
        return bf.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as error:
        pytest.skip(f"Optional FFTW shared library unavailable: {error}")


@pytest.fixture(scope="module")
def provider():
    api._native()
    return bf.KaiserWindow("native", api._LIB._name)


@pytest.fixture
def small_window_blocks(monkeypatch):
    # Exercise many full blocks and the final partial block with small signals.
    from benchmarks._bounded_window import BoundedWindow
    original = BoundedWindow.__init__

    def initialize(self, *args, **kwargs):
        kwargs["block_bins"] = 256
        original(self, *args, **kwargs)

    monkeypatch.setattr(BoundedWindow, "__init__", initialize)


def _signal(n, kind):
    values = np.random.default_rng(819 + n).normal(size=n)
    if kind == "dc_nanovolt":
        values = 10. + 1e-9 * values
    elif kind == "offbin":
        values = np.sin(2 * np.pi * 137.37 * np.arange(n) / n) + 1e-8 * values
    elif kind == "constant":
        values.fill(10.)
    # Strides must not change anchoring or result in an input mutation.
    return values[::-1]


def _options(provider, method, window, in_place):
    common = dict(sample_rate=50., n_frequencies=80, n_averages=10,
                  window_provider=provider, fft_in_place=in_place)
    if method == "fftw":
        common.update(operations="native_grouped", operations_library=api._LIB)
    else:
        common.update(full_window=window, workers=2, smoothing_workers=1,
                      operations_backend="native", smoothing_backend="native",
                      weight_dtype="float32", max_kernel_cache_mb=0,
                      window_cache_mb=0, total_cache_mb=0, max_working_mb=64,
                      low_frequency_backend="bounded", defer_normalization=True)
    return common


@pytest.mark.parametrize("n", [4096, 4099])  # even and odd prime
@pytest.mark.parametrize("method,window", [("fftw", "kaiser"), ("matched", "tukey"),
                                           ("matched", "kaiser"), ("matched", "rectangular")])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("signal", ["white", "dc_nanovolt", "offbin", "constant"])
def test_bounded_complete_spectra_normalization_and_close(
        fftw, provider, small_window_blocks, n, method, window, in_place, signal, monkeypatch):
    values = _signal(n, signal)
    saved = values.copy()
    constructor = bf.Periodogram if method == "fftw" else MatchedSmoothing
    options = _options(provider, method, window, in_place)
    with constructor(fftw, n, **options) as reference:
        expected_window = reference.window.copy()
        expected_sums = (reference.s1, reference.s2)
        expected = reference.compute(values, outputs=("psd", "nsd"))
        full_buffers = reference.metadata["principal_buffer_bytes" if method == "fftw" else "fftw_array_bytes"]
        if method == "matched":
            expected_fourth = reference.metadata["full_record_time_weight_factor_n_sum_w4_over_sum_w2_squared"]
        # The unchanged full-window path retains its existing validation API.
        check = reference.validation(values)
        error = check.get("parseval_absolute_error", check.get("parseval_absolute_error_v2"))
        power = check.get("weighted_detrended_time_power", check.get("global_windowed_time_power_v2"))
        assert error <= 5e-12 * power
    compact = constructor(fftw, n, **options, window_storage="fft_input_then_blocked")
    binding = compact._bound_ops if method == "fftw" else compact._bound_operations
    fft = compact.fft
    try:
        assert compact.window is None
        assert compact._bounded_window is not None
        assert binding.window is binding._center_args is None
        assert binding.metadata["preparation_enabled"] is False
        assert (compact.s1, compact.s2) == expected_sums
        np.testing.assert_array_equal(compact.fft.x, expected_window)
        if method == "matched":
            assert compact.metadata["full_record_time_weight_factor_n_sum_w4_over_sum_w2_squared"] == expected_fourth
        metadata = compact.metadata
        assert metadata["window_storage"] == "fft_input_then_blocked"
        assert metadata["window_buffer_bytes"] == 0
        assert metadata["one_shot_window"] is True
        bounded_buffers = metadata["principal_buffer_bytes" if method == "fftw" else "fftw_array_bytes"]
        assert full_buffers - bounded_buffers == 8 * n - metadata["window_scratch_bytes"]
        for name in ("prepare", "_prepare_validated"):
            with pytest.raises(RuntimeError, match="output-only"):
                getattr(binding, name)(values)
        actual = (compact.compute(values, outputs=("psd", "nsd"), profile=True)
                  if method == "fftw" else compact.profile(values, outputs=("psd", "nsd")))
        assert compact._bounded_window is None
        np.testing.assert_array_equal(actual.index, expected.index)
        for column in ("psd", "nsd"):
            np.testing.assert_allclose(actual[column], expected[column], rtol=3e-6, atol=0.)
        np.testing.assert_array_equal(actual.nsd, np.sqrt(actual.psd.to_numpy().astype(np.complex64)).real)
        with pytest.raises(RuntimeError, match="one-shot"):
            compact.compute(values, outputs=("psd", "nsd"))
        with pytest.raises(RuntimeError, match="opt-in one-shot"):
            compact.validation(values)
        original_close = fft.close

        def checked_close():
            # ctypes pointers and NumPy references die before FFTW-owned memory.
            assert binding._closed
            assert binding.time_buffer is binding.transform is binding._power_args is None
            original_close()

        monkeypatch.setattr(fft, "close", checked_close)
    finally:
        compact.close()
        compact.close()
    assert binding.time_buffer is binding.transform is None
    assert fft.ip is fft.op is fft.plan is None
    np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize("in_place", [False, True])
def test_output_only_binding_rejects_all_ordinary_overlaps(fftw, provider, in_place):
    operations = FFTWOperations("native", api._LIB)
    n = 32
    with bf.RealFFT(fftw, n, in_place=in_place) as fft:
        options = dict(time_buffer=fft.x, transform=fft.y, n=n, sample_rate=50.,
                       window_square_sum=n, powers=np.empty(n // 2 + 1),
                       allow_in_place_fft=in_place)
        bound = operations.bind_output_only(**options)
        try:
            assert bound.window is bound._center is bound._center_args is None
            fft.x[:] = np.arange(n)
            fft.execute()
            np.testing.assert_array_equal(bound.power(), operations.power(fft.y, n, 50., n))
            for buffer in (fft.x[:n // 2 + 1], fft.y.view(np.float64)[:n // 2 + 1]):
                with pytest.raises(ValueError, match="must not overlap"):
                    operations.bind_output_only(**{**options, "powers": buffer})
            if in_place:
                with pytest.raises(ValueError, match="must not overlap"):
                    operations.bind_output_only(**{**options, "allow_in_place_fft": False})
                with pytest.raises(ValueError, match="must not overlap"):
                    operations.bind_output_only(**{**options, "time_buffer": fft.y.view(np.float64)[1:n + 1]})
            with pytest.raises(ValueError, match="Bound window"):
                operations.bind(window=None, **options)
        finally:
            bound.close()
        with pytest.raises(RuntimeError, match="closed"):
            bound.prepare(np.ones(n))


@pytest.mark.parametrize("method", ["fftw", "matched"])
def test_unsupported_bounded_window_fails_before_fft_allocation(fftw, provider, monkeypatch, method):
    def allocated(*args, **kwargs):
        pytest.fail("Unsupported bounded configuration allocated an FFT")

    monkeypatch.setattr(bf, "create_real_fft", allocated)
    constructor = bf.Periodogram if method == "fftw" else MatchedSmoothing
    kwargs = _options(bf.KaiserWindow("numpy"), method, "kaiser", False)
    with pytest.raises(ValueError, match="[Nn]ative"):
        constructor(fftw, 4096, **kwargs, window_storage="fft_input_then_blocked")


@pytest.mark.parametrize("method", ["fftw", "matched"])
def test_bounded_prepare_failure_closes_scratch_and_owner(fftw, provider, monkeypatch, method):
    n = 4096
    constructor = bf.Periodogram if method == "fftw" else MatchedSmoothing
    compact = constructor(fftw, n, **_options(provider, method, "kaiser", True),
                          window_storage="fft_input_then_blocked")
    fft = compact.fft
    helper = compact._bounded_window

    def failed(*args, **kwargs):
        raise ArithmeticError("deliberate preparation failure")

    monkeypatch.setattr(helper, "prepare_into", failed)
    try:
        with pytest.raises(ArithmeticError, match="deliberate"):
            compact.compute(_signal(n, "dc_nanovolt"), outputs=("psd", "nsd"))
        assert compact._bounded_window is None
        with pytest.raises(RuntimeError, match="one-shot"):
            compact.compute(np.zeros(n))
    finally:
        compact.close()
    assert fft.ip is fft.op is fft.plan is None


@pytest.mark.parametrize("in_place", [False, True])
def test_estimate_once_bounded_window_frees_fft_before_low(
        fftw, provider, monkeypatch, small_window_blocks, in_place):
    from benchmarks import matched_smoothing as module
    n = 8192
    values = _signal(n, "dc_nanovolt")
    saved = values.copy()
    options = _options(provider, "matched", "tukey", in_place)
    expected = estimate_once(fftw, values, outputs=("psd", "nsd"), **options)
    observed = []
    original = module.MatchedSmoothing

    class Checked(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            calculate = self._low_prepared.compute

            def compute(samples, **kw):
                assert self.fft is self._bound_operations is self._native_smoothing is None
                assert self._bounded_window is self.window is self.powers is self.scratch is None
                assert kw["workspace"] is None
                observed.append(True)
                return calculate(samples, **kw)

            self._low_prepared.compute = compute

    monkeypatch.setattr(module, "MatchedSmoothing", Checked)
    actual = module.estimate_once(fftw, values, outputs=("psd", "nsd"),
                                  **options, window_storage="fft_input_then_blocked")
    assert observed == [True]
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=0.)
    np.testing.assert_array_equal(values, saved)
    assert actual.attrs["matched_smoothing_once"]["window_storage"] == "fft_input_then_blocked"
    assert actual.attrs["matched_smoothing_once"]["fft_released_before_low_lpsd"]
