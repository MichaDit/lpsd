"""End-to-end gates for the separately reported out-of-core estimators."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bench_fftw as bf
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
            assert details["power_storage"]["maximum_large_file_bytes"] == 24 * n
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
