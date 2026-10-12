"""Original-N numerics, bounded buffers and ownership for the real FFT route."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bench_fftw as bf
from benchmarks._fftw_disk import DiskFFT
from benchmarks._fftw_packed_real import PackedRealDiskFFT
from benchmarks.bench_fftw_disk import disk_pipeline
from benchmarks.fftw_comparison_signals import signal


@pytest.fixture(scope="module")
def fftw():
    try:
        return bf.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as error:
        pytest.skip(f"Optional FFTW shared library unavailable: {error}")


@pytest.mark.parametrize("n", [2, 3, 4, 6, 10, 12, 32, 130, 202, 1023, 4096])
@pytest.mark.parametrize("kind", ["white", "impulse", "dc", "nyquist", "zero"])
def test_original_n_power_parseval_and_refill(fftw, tmp_path, n, kind):
    rng = np.random.default_rng(20261012 + n)
    values = rng.normal(size=n)
    if kind == "impulse":
        values.fill(0.)
        values[-1] = 1.
    elif kind == "dc":
        values.fill(3.)
    elif kind == "nyquist":
        values = (-1.) ** np.arange(n)
    elif kind == "zero":
        values.fill(0.)
    before = values.copy()
    path = tmp_path / "packed.c128"
    with PackedRealDiskFFT(fftw, n, path, memory_mb=.1,
                           reconstruction_elements=7) as transform:
        assert transform.x.shape == transform.window.shape == (n,)
        assert path.stat().st_size == 16 * n
        assert transform.memory["workspace_bytes"] == 16 * n
        assert not np.shares_memory(transform.x, transform.window)
        for source in (values, rng.normal(size=n)):
            transform.window.fill(np.nan)
            transform.x[:] = source
            transform.execute()
            expected = bf.one_sided_psd(np.fft.rfft(source), n, 50., n)
            actual = np.empty(n // 2 + 1)
            assert transform.power_into(actual, 50., n) is actual
            np.testing.assert_allclose(actual, expected, rtol=1e-11,
                                       atol=float(expected.max()) * 1e-24)
            np.testing.assert_allclose(actual.sum() * 50. / n,
                                       np.dot(source, source) / n,
                                       rtol=3e-12, atol=0.)
            if transform.memory["workspace_power_supported"]:
                mapped = transform.power_in_workspace(50., n)
                try:
                    np.testing.assert_array_equal(mapped, actual)
                finally:
                    mapped._mmap.close()
                with pytest.raises(RuntimeError, match="Execute"):
                    transform.power_into(actual, 50., n)
        assert transform.execution_count == 2
        if n % 2 == 0 and n > 2:
            assert transform.inner_n == n // 2
            assert transform.last_phases["real_expansion_s"] == 0.
            assert transform.memory["reconstruction_additional_array_bytes"] == 0
    np.testing.assert_array_equal(values, before)
    assert path.exists()
    transform.close()
    with pytest.raises(RuntimeError, match="closed"):
        transform.execute()


@pytest.mark.parametrize("factors", [(1, 60), (60, 1), (6, 10), (5, 12)])
def test_inner_factors_and_partner_order(fftw, tmp_path, factors):
    n = 120
    values = np.random.default_rng(1280).normal(size=n)
    with PackedRealDiskFFT(fftw, n, tmp_path / "factor.c128", memory_mb=.03,
                           packed_factors=factors, reconstruction_elements=3) as transform:
        transform.x[:] = values
        transform.execute()
        actual = np.empty(n // 2 + 1)
        transform.power_into(actual, 50., n)
        expected = bf.one_sided_psd(np.fft.rfft(values), n, 50., n)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0.)
        assert transform.packed_matrix.shape == factors


@pytest.mark.parametrize("strategy", ["strided", "transpose"])
def test_progress_and_transpose(fftw, tmp_path, strategy):
    events = []
    values = np.random.default_rng(61).normal(size=256)
    with PackedRealDiskFFT(fftw, 256, tmp_path / "progress.c128", memory_mb=.03,
                           packed_factors=(8, 16), column_strategy=strategy,
                           reconstruction_elements=5, progress=events.append) as transform:
        transform.x[:] = values
        transform.execute()
        actual = np.empty(129)
        transform.power_into(actual, 50., 256)
        expected = bf.one_sided_psd(np.fft.rfft(values), 256, 50., 256)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0.)
        records = [item for item in events if item["phase"] == "packed_power"]
        assert records[0]["completed"] == 0
        assert records[-1]["completed"] == records[-1]["total"]
        assert all(set(item) == {"phase", "completed", "total"} for item in events)


def test_external_alias_and_owned_power_mapping(fftw, tmp_path):
    n = 1024
    path = tmp_path / "workspace.c128"
    with PackedRealDiskFFT(fftw, n, path, memory_mb=1) as transform:
        transform.x[:] = np.random.default_rng(51).normal(size=n)
        transform.execute()
        separate = np.memmap(path, dtype=np.float64, mode="r+", shape=(n // 2 + 1,))
        try:
            with pytest.raises(ValueError, match="overlap|different file"):
                transform.power_into(separate.view(np.ndarray), 50., n)
            with pytest.raises(ValueError, match="scratch"):
                transform.power_into(transform._core._work.view(np.float64)[:n // 2 + 1], 50., n)
        finally:
            separate._mmap.close()
        powers = transform.power_in_workspace(50., n)
        saved = powers.copy()
    # Closing the adapter leaves the caller's separately owned mapping valid.
    try:
        np.testing.assert_array_equal(powers, saved)
    finally:
        powers._mmap.close()
    path.unlink()


@pytest.mark.parametrize("adapter", [DiskFFT, PackedRealDiskFFT])
@pytest.mark.parametrize("as_view", [False, True])
def test_hardlink_alias_rejected_before_transform_write(fftw, tmp_path, adapter, as_view):
    path = tmp_path / "transform.c128"
    alias = tmp_path / "alias.c128"
    with adapter(fftw, 1024, path, memory_mb=1) as transform:
        values = np.random.default_rng(130).normal(size=1024)
        transform.x[:] = values
        transform.execute()
        os.link(path, alias)
        powers = np.memmap(alias, dtype=np.float64, mode="r+", shape=(513,))
        try:
            with pytest.raises(ValueError, match="separate file|different file"):
                transform.power_into(powers.view(np.ndarray) if as_view else powers, 50., 1024)
            actual = np.empty(513)
            transform.power_into(actual, 50., 1024)
            expected = bf.one_sided_psd(np.fft.rfft(values), 1024, 50., 1024)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0.)
        finally:
            powers._mmap.close()


def test_billion_sample_layout_without_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A scalar memory estimate must not allocate mapped arrays")
    monkeypatch.setattr(np, "memmap", forbidden)
    result = PackedRealDiskFFT.memory_estimate(1_000_000_000, 128)
    assert result["inner_n"] == 500_000_000
    assert result["packed_complex_region_bytes"] == 8_000_000_000
    assert result["workspace_bytes"] == 16_000_000_000
    assert result["maximum_large_file_bytes_with_input"] == 24_000_000_000
    assert result["known_ram_buffer_bytes"] <= 128 * 1024**2
    assert result["reconstruction_additional_array_bytes"] == 0
    assert result["workspace_power_supported"]


@pytest.mark.parametrize("method", ["fftw", "matched"])
@pytest.mark.parametrize("name", ["white", "pink", "brown", "mixed_tones", "offbin_tone", "dc_nanovolt"])
@pytest.mark.parametrize("points", [1000, 2240])
def test_complete_spectra_user_accuracy_contract(fftw, tmp_path, method, name, points):
    # The original strict full-C2C tests remain separate and unchanged.
    # Packing is compared at every point under the user's <1% allowance;
    # endpoints and transform algebra have much tighter float64 gates above.
    values, _ = signal(name, 10_000, 50., 20261010)
    before = values.tobytes()
    frames = []
    for algorithm in ("full", "packed-real"):
        directory = tmp_path / algorithm
        directory.mkdir()
        args = SimpleNamespace(sample_rate=50., n_frequencies=points, n_averages=100,
                               fft_memory_mb=1., low_working_mb=8., workers=1,
                               smoothing_workers=1, power_storage="workspace",
                               disk_fft_algorithm=algorithm)
        frame, details = disk_pipeline(method, values, fftw, directory, args)
        frames.append(frame)
        assert details["disk_fft_algorithm"] == algorithm
        assert np.isfinite(frame.to_numpy()).all()
        assert (frame.to_numpy() >= 0).all()
    np.testing.assert_array_equal(frames[0].index, frames[1].index)
    for column in ("psd", "nsd"):
        a, b = frames[0][column].to_numpy(), frames[1][column].to_numpy()
        assert np.all(np.abs(a - b) <= .01 * np.abs(a))
    assert values.tobytes() == before


def test_invalid_inputs_and_no_file_overwrite(fftw, tmp_path):
    path = tmp_path / "existing.c128"
    path.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        PackedRealDiskFFT(fftw, 128, path)
    assert path.read_bytes() == b"preserve"
    for n in (True, 1, 0, -1, 1.5, 2**54):
        with pytest.raises((ValueError, TypeError)):
            PackedRealDiskFFT.memory_estimate(n)
    with pytest.raises(ValueError, match="packed_factors"):
        PackedRealDiskFFT.memory_estimate(128, factors=(8, 16))
    with pytest.raises(ValueError, match="packed_factors"):
        PackedRealDiskFFT.memory_estimate(127, packed_factors=(1, 127))
    for limit in (0, 65537, True, 1.5):
        with pytest.raises((ValueError, TypeError)):
            PackedRealDiskFFT.memory_estimate(128, reconstruction_elements=limit)
