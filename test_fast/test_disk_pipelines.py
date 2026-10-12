"""End-to-end gates for the separately reported out-of-core estimators."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bench_fftw as bf
from benchmarks import bench_fftw_disk as disk_benchmark
from benchmarks.bench_fftw_disk import disk_pipeline, fill_tukey, input_file
from benchmarks.bench_fftw_optimized import assert_equivalent
from benchmarks.fftw_comparison_signals import signal
from benchmarks.matched_smoothing import _periodic_tukey, estimate_once
from lpsd_fast import api


@pytest.fixture(scope="module")
def fftw():
    try:
        return bf.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as exc:
        pytest.skip(f"Optional FFTW shared library unavailable: {exc}")


@pytest.mark.parametrize("n,power_storage", [
    (4095, "external"), (10000, "external"), (4096, "workspace"), (10000, "workspace"),
])
@pytest.mark.parametrize("case", ["white", "offbin_tone", "dc_nanovolt"])
@pytest.mark.parametrize("method", ["fftw", "matched", "lpsd"])
def test_complete_disk_estimator_matches_existing_pipeline(fftw, tmp_path, n, power_storage,
                                                          case, method):
    samples, _ = signal(case, n, 50.0, 20261010)
    path = tmp_path / "input.f64"
    mapped = np.memmap(path, mode="w+", dtype=np.float64, shape=(n,))
    mapped[:] = samples
    fingerprint = bf.digest_array(mapped)
    common = dict(sample_rate=50.0, psll=200.0, n_frequencies=1000, n_averages=100)
    if method == "fftw":
        with bf.Periodogram(fftw, n, **common, operations="native_grouped") as reference:
            expected = reference.compute(mapped, outputs=("psd", "nsd"))
    elif method == "matched":
        expected = estimate_once(fftw, mapped, **common, workers=8,
                                 smoothing_workers=4, operations_backend="native",
                                 smoothing_backend="native", weight_dtype="float32",
                                 outputs=("psd", "nsd"))
    else:
        expected = api.lpsd(mapped, **common, workers=8, kernel="fast", outputs=("psd", "nsd"))
    args = SimpleNamespace(sample_rate=50.0, workers=8, smoothing_workers=4,
                           low_working_mb=64, fft_memory_mb=.1, power_storage=power_storage)
    try:
        actual, details = disk_pipeline(method, mapped, fftw, tmp_path, args)
        np.testing.assert_array_equal(actual.index.to_numpy(), expected.index.to_numpy())
        for name in ("psd", "nsd"):
            assert_equivalent(actual[name], expected[name])
        np.testing.assert_array_equal(np.sqrt(actual.psd.to_numpy().astype(np.complex64)).real,
                                      actual.nsd.to_numpy())
        assert bf.digest_array(mapped) == fingerprint
        assert not (tmp_path / "workspace.c128").exists()
        assert not (tmp_path / "powers.f64").exists()
        assert details["phases"]["workspace_cleanup_s"] >= 0
        if power_storage == "workspace" or method == "lpsd":
            assert details["power_storage"]["maximum_large_file_bytes"] == (8 if method == "lpsd" else 24) * n
            assert details["power_storage"]["extra_power_file_bytes"] == 0
        if power_storage == "workspace" and method != "lpsd":
            assert details["fft"]["power_storage"] == "workspace"
            assert details["power_storage"]["workspace_power_offset_bytes"] == 8 * n
    finally:
        mapped._mmap.close()


@pytest.mark.parametrize("n", [1, 31, 4095, 10000, 600001])
def test_tukey_blocks_preserve_exact_original_window(n):
    window = np.empty(n, dtype=np.float64)
    fill_tukey(window, .05)
    np.testing.assert_array_equal(window, _periodic_tukey(n, .05))


def test_chunked_input_matches_original_source(tmp_path):
    n = 600001
    source = input_file(tmp_path / "input.f64", n, 50.0, 20261010)
    expected, _ = signal("white", n, 50.0, 20261010)
    mapped = np.memmap(tmp_path / "input.f64", dtype=np.float64, mode="r", shape=(n,))
    try:
        np.testing.assert_array_equal(mapped, expected)
        assert source["sha256"] == bf.digest_array(expected)
    finally:
        mapped._mmap.close()


def _small_options(algorithm="full", storage="workspace"):
    return SimpleNamespace(sample_rate=50., n_frequencies=1000, n_averages=100,
                           workers=1, smoothing_workers=1, low_working_mb=8.,
                           fft_memory_mb=.1, power_storage=storage,
                           disk_fft_algorithm=algorithm)


@pytest.mark.parametrize("algorithm", ["full", "packed-real"])
@pytest.mark.parametrize("method", ["fftw", "matched"])
@pytest.mark.parametrize("level", [0., 7.])
def test_zero_detrended_energy_has_zero_spectra_and_valid_parseval(fftw, tmp_path,
                                                                algorithm, method, level):
    frame, details = disk_pipeline(method, np.full(1000, level), fftw, tmp_path,
                                   _small_options(algorithm))
    np.testing.assert_array_equal(frame.to_numpy(), np.zeros_like(frame.to_numpy()))
    assert details["parseval"]["time_power_v2"] == 0.
    assert details["parseval"]["fft_power_v2"] == 0.
    assert details["parseval"]["relative_error"] == 0.
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("algorithm", ["full", "packed-real"])
@pytest.mark.parametrize("storage", ["workspace", "external"])
@pytest.mark.parametrize("failure", [None, "planning", "preprocessing"])
def test_pipeline_preserves_caller_files_and_cleans_only_owned_work(
        fftw, tmp_path, monkeypatch, algorithm, storage, failure):
    sentinels = {tmp_path / "workspace.c128": b"Caller workspace",
                 tmp_path / "powers.f64": b"Caller powers"}
    for path, content in sentinels.items():
        path.write_bytes(content)
    if failure == "planning":
        # Fail after a real workspace and anonymous buffers were created.
        monkeypatch.setattr(fftw.lib, "fftw_plan_many_dft", lambda *args: None)
    elif failure == "preprocessing":
        def fail_prepare(*args, **kwargs):
            raise RuntimeError("Controlled preprocessing failure")
        monkeypatch.setattr(disk_benchmark.FFTWOperations, "prepare", fail_prepare)
    args = _small_options(algorithm, storage)
    values = np.random.default_rng(95).normal(size=1000)
    if failure is None:
        frame, _ = disk_pipeline("fftw", values, fftw, tmp_path, args)
        assert np.isfinite(frame.to_numpy()).all()
    else:
        with pytest.raises((MemoryError, RuntimeError), match="batch-plan|Controlled preprocessing"):
            disk_pipeline("fftw", values, fftw, tmp_path, args)
    assert set(tmp_path.iterdir()) == set(sentinels)
    for path, content in sentinels.items():
        assert path.read_bytes() == content
