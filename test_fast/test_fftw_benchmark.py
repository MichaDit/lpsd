"""Small numerical gates for the optional FFTW comparison instrument."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace


@pytest.fixture(scope="module")
def adapter():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_fftw.py"
    spec = importlib.util.spec_from_file_location("fftw_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fftw(adapter):
    try:
        return adapter.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as exc:
        pytest.skip(f"Optional FFTW shared library unavailable: {exc}")


@pytest.mark.parametrize("n", [31, 32, 63, 64])
@pytest.mark.parametrize("sample_rate", [1.0, 123.5])
def test_psd_parseval_and_all_positive_bins(adapter, fftw, n, sample_rate):
    rng = np.random.default_rng(2013 + n)
    signals = [rng.normal(size=n), np.eye(1, n, n // 3).ravel(),
               np.cos(2 * np.pi * (n // 2) * np.arange(n) / n)]
    windows = [np.ones(n), np.hanning(n), np.kaiser(n + 1, 25.402566024390598)[:-1]]
    cuts = np.array([1, 3, 7, n // 2 + 1])
    with adapter.RealFFT(fftw, n) as transform:
        for signal in signals:
            original = signal.copy()
            for window in windows:
                centered = signal - signal[0]
                centered -= centered.mean()
                transform.x[:] = centered * window
                transform.execute()
                np.testing.assert_allclose(transform.y, np.fft.rfft(centered * window), rtol=5e-13, atol=5e-13)
                s2 = np.sum(window * window)
                psd = adapter.one_sided_psd(transform.y, n, sample_rate, s2)
                np.testing.assert_allclose(np.sum(psd) * sample_rate / n,
                                           np.sum((centered * window) ** 2) / s2,
                                           rtol=5e-13, atol=2e-15)
                means, powers = adapter.aggregate_power(psd, cuts, sample_rate, n)
                np.testing.assert_allclose(powers, means * np.diff(cuts) * sample_rate / n, rtol=5e-15)
                np.testing.assert_allclose(powers.sum(), psd[1:].sum() * sample_rate / n, rtol=5e-15)
            np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("n", [31, 32, 63, 64])
def test_dc_and_highest_bin_have_correct_one_sided_weights(adapter, fftw, n):
    # No detrending here: isolate normalization, DC, and odd/even endpoints.
    fs = 17.0
    with adapter.RealFFT(fftw, n) as transform:
        transform.x[:] = 1.0
        transform.execute()
        psd = adapter.one_sided_psd(transform.y, n, fs, n)
        assert psd[0] == pytest.approx(n / fs)
        assert psd.sum() * fs / n == pytest.approx(1.0)
        transform.x[:] = np.cos(2 * np.pi * (n // 2) * np.arange(n) / n)
        transform.execute()
        psd = adapter.one_sided_psd(transform.y, n, fs, n)
        expected = 1.0 if n % 2 == 0 else 0.5
        assert psd[-1] * fs / n == pytest.approx(expected, rel=5e-13)
        assert psd.sum() * fs / n == pytest.approx(expected, rel=5e-13)


@pytest.mark.parametrize("n", [127, 128])
@pytest.mark.parametrize("sample_rate", [1.0, 123.5])
def test_actual_lpsd_grid_coverage(adapter, n, sample_rate):
    grid = adapter.lpsd_grid(n, sample_rate, 200.0, 12, 4)
    cuts = grid["cuts"]
    covered = np.concatenate([np.arange(a, b) for a, b in zip(cuts[:-1], cuts[1:])])
    np.testing.assert_array_equal(covered, np.arange(1, n // 2 + 1))
    assert len(grid["frequency_labels_hz"]) == len(grid["counts"])
    assert np.all(grid["counts"] > 0)
    np.testing.assert_array_equal(grid["discrete_bandwidth_hz"], grid["counts"] * (sample_rate / n))
    # The nominal interval widths must not silently replace count*df.
    assert not np.array_equal(np.diff(grid["nominal_midpoint_boundaries_hz"]), grid["discrete_bandwidth_hz"])


@pytest.mark.parametrize("planner", ["estimate", "measure"])
@pytest.mark.parametrize("aggregation", ["log", "none"])
def test_prepared_pipeline_preserves_input_and_owns_outputs(adapter, fftw, planner, aggregation):
    n = 127
    values = 10.0 + 1e-9 * np.random.default_rng(399).normal(size=n)
    series = pd.Series(values[::-1], copy=False)
    original = series.to_numpy().copy()
    provider = adapter.KaiserWindow("numpy")
    fftw.forget_wisdom()
    with adapter.Periodogram(fftw, n, sample_rate=123.5, n_frequencies=12,
                            n_averages=4, planner=planner, time_limit=0.01,
                            window_provider=provider, aggregation=aggregation) as pipeline:
        first = pipeline.compute(series)
        saved = first.copy(deep=True)
        second = pipeline.compute(series)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        pd.testing.assert_frame_equal(first, saved, check_exact=True)
        expected = original - original[0]
        expected -= expected.mean()
        reference = adapter.one_sided_psd(np.fft.rfft(expected * pipeline.window),
                                         n, 123.5, pipeline.s2)
        if aggregation == "log":
            reference, _ = adapter.aggregate_power(reference, pipeline.grid["cuts"], 123.5, n)
        np.testing.assert_allclose(first.psd, reference.astype(np.float32), rtol=2e-6, atol=0)
        validation = pipeline.validation()
        assert validation["finite"]
        assert validation["parseval_absolute_error"] <= 1e-30
        # Run different data through the same scratch buffers: old frames must stay intact.
        pipeline.compute(np.ones(n))
        pd.testing.assert_frame_equal(first, saved, check_exact=True)
    np.testing.assert_array_equal(series.to_numpy(), original)


def test_nsd_uses_rounded_psd_and_preserves_dc_diagnostic(adapter, fftw):
    values = np.random.default_rng(115).normal(size=128)
    kwargs = dict(n_frequencies=12, n_averages=4, window_provider=adapter.KaiserWindow("numpy"))
    with adapter.Periodogram(fftw, len(values), density="psd", **kwargs) as psd:
        power = psd.compute(values)
        assert psd.validation()["dc_bin_power"] > 0
    with adapter.Periodogram(fftw, len(values), density="nsd", **kwargs) as nsd:
        noise = nsd.compute(values)
    np.testing.assert_array_equal(noise.nsd, np.sqrt(power.psd.to_numpy().astype(np.complex64)).real)


def test_empty_bin_partition_is_rejected(adapter):
    with pytest.raises(ValueError, match="empty or invalid"):
        adapter.partitions(32, 1.0, [0.01, 0.02, 0.03])


def test_cold_pipeline_records_planning_and_free(adapter, fftw):
    data = pd.Series(np.arange(128, dtype=float))
    kwargs = dict(n_frequencies=12, n_averages=4, planner="measure", time_limit=0.01,
                  window_provider=adapter.KaiserWindow("numpy"))
    frame, detail = adapter.cold_call(fftw, data, kwargs, profile=True)
    assert len(frame) > 0
    assert detail["planning_s"] >= 0
    assert detail["free_s"] >= 0
    assert "copy_and_detrend_s" in detail["execution_phases"]
    assert "window_generation_s" in detail["setup_phases"]


def test_cli_rejects_invalid_counts_before_loading_fftw(adapter, tmp_path):
    output = tmp_path / "unused.json"
    with pytest.raises(SystemExit) as error:
        adapter.main(["--n", "0", "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_windows_plan_description_avoids_incompatible_free(adapter, monkeypatch):
    def forbidden(*args):
        pytest.fail('Windows must not allocate a malloc string for fftw_free')
    transform=object.__new__(adapter.RealFFT)
    transform.backend=SimpleNamespace(lib=SimpleNamespace(fftw_flops=lambda *args:None,
                                                         fftw_sprint_plan=forbidden,
                                                         fftw_free=forbidden))
    transform.plan=1
    transform.planner='estimate'
    transform.time_limit=None
    transform.planning_s=transform.allocation_s=0.
    monkeypatch.setattr(adapter,'os',SimpleNamespace(name='nt'))
    description=transform.describe()
    assert description['plan_text_available'] is False
    assert description['plan_text']==''


def test_tiny_cli_report_contains_separate_totals_and_provenance(adapter, fftw, tmp_path):
    output = tmp_path / "tiny-schema-smoke.json"
    adapter.main(["--n", "128", "--n-frequencies", "12", "--n-averages", "4",
                  "--repeats", "1", "--warmups", "0", "--raw-batch-seconds", "0",
                  "--window-backend", "numpy", "--fftw-library", fftw.lib._name,
                  "--profile", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["parameters"]["padding_samples"] == 0
    assert report["parameters"]["input_dtype"] == "float64"
    assert report["fftw"]["version"] == fftw.version
    for category in ("raw_fft", "prepared_pipeline", "cold_pipeline"):
        assert len(report[category]["repetitions"]) == 1
        assert report[category]["median_wall_s"] >= 0
    assert report["raw_fft"]["repetitions"][0]["batch_executions"] == 1
    assert report["cold_pipeline"]["repetitions"][0]["planning_s"] >= 0
    assert report["validation"]["positive_bins_covered"] == 64
    assert report["additional_profiled_prepared_call"]["phases"]["fftw_execute_s"] >= 0
    assert report["additional_profiled_cold_call"]["setup_phases"]["grid_s"] >= 0
    assert report["input"]["unchanged"]


def test_optional_native_kaiser_matches_periodic_numpy(adapter):
    provider = adapter.KaiserWindow("auto", os.environ.get("LPSD_TEST_FAST_LIBRARY"))
    if provider.lib is None:
        pytest.skip("Optional fast.2 native Kaiser generator unavailable")
    window = provider.generate(128, 25.402566024390598)
    np.testing.assert_allclose(window, np.kaiser(129, 25.402566024390598)[:-1], rtol=8e-14, atol=4e-15)
    s1, s2 = provider.sums(window)
    assert s1 == sum(window)
    assert s2 == sum(window * window)


def test_optional_threaded_plan_matches_serial(adapter, fftw):
    try:
        threaded = adapter.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"),
                                       os.environ.get("LPSD_TEST_FFTW_THREADS_LIBRARY"), threads=2)
    except OSError as exc:
        pytest.skip(f"Optional FFTW threads library unavailable: {exc}")
    signal = np.random.default_rng(775).normal(size=64)
    with adapter.RealFFT(threaded, len(signal)) as transform:
        transform.x[:] = signal
        transform.execute()
        np.testing.assert_allclose(transform.y, np.fft.rfft(signal), rtol=5e-13, atol=5e-13)
    # The process-global planner thread count must return to one for this plan.
    with adapter.RealFFT(fftw, len(signal)) as transform:
        transform.x[:] = signal
        transform.execute()
        np.testing.assert_allclose(transform.y, np.fft.rfft(signal), rtol=5e-13, atol=5e-13)
