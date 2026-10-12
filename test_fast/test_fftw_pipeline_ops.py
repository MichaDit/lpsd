"""Numerical and ownership gates for additive FFTW pipeline fusion."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from benchmarks import bench_fftw as adapter
from benchmarks._fftw_native import FFTWOperations
from lpsd_fast import api


@pytest.fixture(scope="module")
def native_ops():
    api._native()
    # This suite requires the new ABI rather than silently testing fallback.
    return FFTWOperations("native", api._LIB)


@pytest.fixture(scope="module")
def fftw():
    try:
        return adapter.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as exc:
        pytest.skip(f"Optional FFTW shared library unavailable: {exc}")


def signals(n):
    rng = np.random.default_rng(20261010 + n)
    index = np.arange(n)
    return (rng.normal(size=n),
            10.0 + 1e-9 * rng.normal(size=n),
            np.sin(2 * np.pi * 7.37 * index / n) +
            1e-8 * np.sin(2 * np.pi * 19.71 * index / n),
            np.full(n, 10.0))


@pytest.mark.parametrize("n", [31, 32, 127, 128, 8192])
def test_native_prepare_preserves_baseline_operations_and_strides(native_ops, n):
    window = np.kaiser(n + 1, 25.402566024390598)[:-1]
    for source in signals(n):
        values = source[::-1]
        saved = values.copy()
        expected = values - values[0]
        expected_mean = expected.mean()
        expected -= expected_mean
        expected *= window
        actual = np.empty(n)
        mean = native_ops.prepare(values, window, actual)
        assert mean == expected_mean
        assert actual.tobytes() == expected.tobytes()
        np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize("n", [31, 32, 127, 128, 8192])
def test_native_power_matches_baseline_bits_and_endpoints(native_ops, n):
    rng = np.random.default_rng(881 + n)
    scale = np.geomspace(1e-145, 1e145, n // 2 + 1)
    transform = (rng.normal(size=len(scale)) + 1j * rng.normal(size=len(scale))) * scale
    expected = adapter.one_sided_psd(transform, n, 50.0, 0.23 * n)
    actual = native_ops.power(transform, n, 50.0, 0.23 * n)
    assert actual.tobytes() == expected.tobytes()
    # DC is always halved; the last bin is only halved when N is even.
    ones = np.ones(n // 2 + 1, dtype=np.complex128)
    powers = native_ops.power(ones, n, 1.0, 1.0)
    assert powers[0] == 1.0
    assert powers[-1] == (1.0 if n % 2 == 0 else 2.0)


@pytest.mark.parametrize("n", [127, 128, 8192])
def test_direct_groups_preserve_exact_membership_and_power(native_ops, n):
    transform = np.fft.rfft(signals(n)[0])
    cuts = np.array([1, 3, 7, n // 2 + 1], dtype=np.int64)
    expected_power = adapter.one_sided_psd(transform, n, 50.0, n)
    expected, expected_bands = adapter.aggregate_power(expected_power, cuts, 50.0, n)
    actual, bands = native_ops.log_power(transform, n, 50.0, n, cuts)
    np.testing.assert_allclose(actual, expected, rtol=8e-15, atol=0)
    np.testing.assert_allclose(bands, expected_bands, rtol=8e-15, atol=0)
    np.testing.assert_allclose(bands.sum(), expected_power[1:].sum() * 50.0 / n,
                               rtol=8e-15, atol=0)
    for bad in ([0, 3, n // 2 + 1], [1, 3, 3, n // 2 + 1], [1, n // 2]):
        with pytest.raises(ValueError, match="partition"):
            native_ops.log_power(transform, n, 50.0, n, bad)


@pytest.mark.parametrize("n", [127, 128, 8192])
@pytest.mark.parametrize("aggregation", ["log", "none"])
def test_native_pipeline_matches_baseline_and_joint_output(native_ops, fftw, n, aggregation):
    kwargs = dict(sample_rate=50.0, n_frequencies=32, n_averages=8,
                  window_provider=adapter.KaiserWindow("numpy"),
                  operations_library=native_ops.lib, aggregation=aggregation)
    with adapter.Periodogram(fftw, n, operations="numpy", **kwargs) as baseline, \
         adapter.Periodogram(fftw, n, operations="native", **kwargs) as optimized:
        assert optimized.scratch is None
        assert baseline.scratch is not None
        for values in signals(n):
            original = values.copy()
            expected = baseline.compute(values, outputs=("psd", "nsd"))
            actual = optimized.compute(values, outputs=("psd", "nsd"), profile=True)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            np.testing.assert_array_equal(actual.nsd, np.sqrt(actual.psd.to_numpy().astype(np.complex64)).real)
            np.testing.assert_array_equal(values, original)
            saved = actual.copy(deep=True)
            optimized.compute(np.zeros(n))
            pd.testing.assert_frame_equal(actual, saved, check_exact=True)
            optimized.compute(values)
            check = optimized.validation()
            assert check["finite"]
            reference_power = check["weighted_detrended_time_power"]
            assert check["parseval_absolute_error"] <= 5e-13 * reference_power


@pytest.mark.parametrize("n", [127, 128, 8192])
def test_direct_pipeline_diagnostics_do_not_retain_linear_buffers(native_ops, fftw, n):
    kwargs = dict(sample_rate=50.0, n_frequencies=32, n_averages=8,
                  window_provider=adapter.KaiserWindow("numpy"),
                  operations_library=native_ops.lib, density="both")
    with adapter.Periodogram(fftw, n, operations="numpy", **kwargs) as baseline, \
         adapter.Periodogram(fftw, n, operations="native_grouped", **kwargs) as grouped:
        for values in signals(n):
            expected = baseline.compute(values)
            actual = grouped.compute(values, profile=True)
            pd.testing.assert_index_equal(actual.index, expected.index)
            np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)
            np.testing.assert_array_equal(actual.nsd, np.sqrt(actual.psd.to_numpy().astype(np.complex64)).real)
            assert grouped.powers is None and grouped.scratch is None
            check = grouped.validation()
            assert grouped.powers is None and grouped.scratch is None
            assert check["finite"]
            assert check["aggregation_absolute_error"] <= 5e-13 * check["positive_bin_power"]
            assert grouped.last_profile["log_aggregation_s"] == 0.0
            assert grouped.last_profile["window_multiply_s"] == 0.0


def test_operations_backend_fallback_is_explicit(tmp_path):
    missing = tmp_path / "deliberately_missing_ops.so"
    automatic = FFTWOperations("auto", missing)
    assert automatic.backend == "numpy"
    assert automatic.metadata["fallback_reason"]
    with pytest.raises(RuntimeError, match="rebuild"):
        FFTWOperations("native", missing)
    baseline = FFTWOperations("numpy", missing)
    assert baseline.backend == "numpy" and baseline.metadata["fallback_reason"] is None
    values = signals(32)[1]
    expected = values - values[0]
    expected -= expected.mean()
    expected *= np.hanning(32)
    actual = np.empty(32)
    automatic.prepare(values, np.hanning(32), actual)
    np.testing.assert_array_equal(actual, expected)


def test_tiny_cli_records_operation_backend_and_both_outputs(native_ops, fftw, tmp_path):
    output = tmp_path / "native-operations.json"
    adapter.main(["--n", "128", "--n-frequencies", "12", "--n-averages", "4",
                  "--repeats", "1", "--warmups", "0", "--raw-batch-seconds", "0",
                  "--window-backend", "numpy", "--fftw-library", fftw.lib._name,
                  "--lpsd-fast-library", native_ops.lib._name,
                  "--operations", "native_grouped", "--density", "both",
                  "--profile", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["parameters"]["operations"] == "native_grouped"
    assert report["pipeline_operations"]["validation_reconstructs_powers"]
    assert report["prepared_pipeline"]["repetitions"][0]["outputs"] == ["psd", "nsd"]
    assert report["additional_profiled_prepared_call"]["phases"]["log_aggregation_s"] == 0.0


@pytest.mark.parametrize("operations", ["native", "native_grouped"])
def test_bound_pipeline_validates_new_inputs_and_invalidates_before_close(native_ops, fftw, monkeypatch, operations):
    """Prepared computation no longer routes private buffers through public APIs."""
    n = 128
    kwargs = dict(sample_rate=50.0, n_frequencies=12, n_averages=4,
                  window_provider=adapter.KaiserWindow("numpy"),
                  operations_library=native_ops.lib, density="both")
    with adapter.Periodogram(fftw, n, operations="numpy", **kwargs) as baseline:
        expected = baseline.compute(signals(n)[1][::-1])
    pipeline = adapter.Periodogram(fftw, n, operations=operations, **kwargs)
    binding = pipeline._bound_ops
    assert pipeline.metadata["operation_binding"]["cached_native_pointers"]
    try:
        def public_path_called(*args, **kwargs):
            pytest.fail("A fixed private buffer was routed through the per-call wrapper")
        with monkeypatch.context() as patch:
            patch.setattr(FFTWOperations, "prepare", public_path_called)
            patch.setattr(FFTWOperations, "power", public_path_called)
            patch.setattr(FFTWOperations, "log_power", public_path_called)
            actual = pipeline.compute(signals(n)[1][::-1])
            np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)
            for bad in (np.zeros(n + 1), np.zeros((n, 1))):
                with pytest.raises(ValueError, match="matching the prepared length"):
                    pipeline.compute(bad)
        assert pipeline.validation()["finite"]
    finally:
        pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        binding.prepare(np.ones(n))
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.compute(np.ones(n))
    assert binding.time_buffer is None and binding.transform is None


def test_binding_rejects_unsafe_private_buffers(native_ops):
    n = 32
    kwargs = dict(time_buffer=np.empty(n), window=np.ones(n),
                  transform=np.empty(n // 2 + 1, dtype=np.complex128),
                  n=n, sample_rate=50.0, window_square_sum=n,
                  powers=np.empty(n // 2 + 1))
    with pytest.raises(ValueError, match="contiguous"):
        native_ops.bind(**dict(kwargs, window=np.ones(n * 2)[::2]))
    with pytest.raises(ValueError, match="overlap"):
        native_ops.bind(**dict(kwargs, window=kwargs["time_buffer"]))
    with pytest.raises(ValueError, match="partition"):
        native_ops.bind(**dict(kwargs, cuts=np.array([1, 3, 3, n // 2 + 1])))
    binding = native_ops.bind(**kwargs)
    try:
        with pytest.raises(ValueError, match="matching the bound length"):
            binding.prepare(np.ones(n + 1))
        source = signals(n)[1]
        saved = source.copy()
        expected = source - source[0]
        expected -= expected.mean()
        binding.prepare(source)
        np.testing.assert_array_equal(kwargs["time_buffer"], expected)
        np.testing.assert_array_equal(source, saved)
    finally:
        binding.close()
