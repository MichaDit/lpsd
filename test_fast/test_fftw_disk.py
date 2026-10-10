"""Small numerical/lifecycle gates for the file-backed exact-N FFT adapter.

No large transform or performance threshold is exercised. The billion-sample
case checks scalar layout arithmetic only and forbids array/file allocation.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import _fftw_disk as disk_module
from benchmarks import bench_fftw as adapter
from benchmarks._fftw_disk import DiskFFT
from benchmarks._fftw_native import FFTWOperations
from lpsd_fast import api
from test_fast._support import CASES


@pytest.fixture(scope="module")
def fftw():
    try:
        return adapter.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as error:
        pytest.skip(f"Optional FFTW shared library unavailable: {error}")


def _ordered_copy(transform):
    # Only test-sized matrices are reordered in RAM. Production gathers
    # bounded column tiles and never allocates this full transpose.
    return np.array(transform.matrix.T, order="C", copy=True).reshape(-1)


def _assert_dft_and_power(transform, values, powers, sample_rate=50.0, s2=None):
    n = len(values)
    s2 = float(n) if s2 is None else s2
    expected_complex = np.fft.fft(values)
    actual_complex = _ordered_copy(transform)
    amplitude = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    np.testing.assert_allclose(actual_complex, expected_complex,
                               rtol=3e-12, atol=3e-13 * amplitude)
    expected_positive = np.fft.rfft(values)
    expected_power = adapter.one_sided_psd(expected_positive, n, sample_rate, s2)
    returned = transform.power_into(powers, sample_rate, s2)
    assert returned is powers
    np.testing.assert_allclose(powers, expected_power, rtol=8e-12,
                               atol=float(expected_power.max()) * 1e-24)
    np.testing.assert_allclose(np.sum(powers) * sample_rate / n,
                               np.dot(values, values) / s2, rtol=3e-12, atol=0)
    return actual_complex.copy()


@pytest.mark.parametrize("factors", [(125, 80), (63, 65), (1, 1009), (1009, 1)])
def test_full_original_dft_powers_and_refill(fftw, tmp_path, factors):
    n = factors[0] * factors[1]
    values = np.random.default_rng(20261010 + n).normal(size=n)
    impulse = np.zeros(n)
    impulse[-1] = 1.0
    workspace_path = tmp_path / "workspace.c128"
    powers = np.memmap(tmp_path / "powers.f64", mode="w+", dtype=np.float64,
                       shape=(n // 2 + 1,))
    try:
        with DiskFFT(fftw, n, workspace_path, memory_mb=1, factors=factors) as transform:
            assert transform.matrix.shape == factors
            assert transform.x.shape == transform.window.shape == (n,)
            assert not np.shares_memory(transform.x, transform.window)
            assert np.shares_memory(transform.matrix, transform.x)
            assert workspace_path.stat().st_size == 16 * n
            # A nonzero window staging region catches accidentally reading
            # the second half as complex input instead of expanding x.
            transform.window.fill(12345.0)
            transform.x[:] = values
            assert transform.execute() is transform.matrix
            first_result = _assert_dft_and_power(transform, values, powers)
            original_values = values.copy()
            transform.window.fill(-7654.0)
            transform.x[:] = impulse
            transform.execute()
            _assert_dft_and_power(transform, impulse, powers)
            np.testing.assert_array_equal(values, original_values)
            # The owning test result survives reuse of the mapped workspace.
            np.testing.assert_allclose(first_result, np.fft.fft(values),
                                       rtol=3e-12, atol=3e-13 * np.linalg.norm(values))
            assert transform.metadata["execution_count"] == 2
            assert all(plan["length"] <= max(factors) for plan in transform.metadata["plans"])
            assert transform.last_phases["execute_total_s"] >= 0
            assert transform.last_phases["power_total_s"] >= 0
    finally:
        powers._mmap.close()
    assert workspace_path.exists()


def test_partial_batches_and_overlapping_reverse_expansion(fftw, tmp_path):
    n, factors = 10000, (125, 80)
    # This fits individual subtransforms but forces tails in both FFT stages
    # and many real-expansion chunks. No full-size hidden scratch is needed.
    with DiskFFT(fftw, n, tmp_path / "tail.c128", memory_mb=.03,
                 factors=factors) as transform:
        assert factors[1] % transform.column_batch
        assert factors[0] % transform.row_batch
        assert transform.memory["real_scratch_elements"] < n
        plan_sizes = {(entry["length"], entry["howmany"])
                      for entry in transform.metadata["plans"]}
        assert (factors[0], factors[1] % transform.column_batch) in plan_sizes
        assert (factors[1], factors[0] % transform.row_batch) in plan_sizes
        # An aperiodic source makes overwritten/reused chunk boundaries visible.
        values = (np.arange(n, dtype=np.float64) % 37 - 18) / 19
        values += np.random.default_rng(704).normal(scale=.03, size=n)
        transform.window.fill(np.nan)
        transform.x[:] = values
        transform.execute()
        _assert_dft_and_power(transform, values, np.empty(n // 2 + 1))


def test_column_staging_bounds_file_views_and_preserves_complex_values(fftw, tmp_path, monkeypatch):
    with DiskFFT(fftw, 10000, tmp_path / "bounded_io.c128", memory_mb=.03,
                 factors=(125, 80), io_advice=False) as transform:
        original = np.arange(10000, dtype=np.float64).reshape(125, 80).astype(np.complex128)
        original.imag = -original.real - .25
        transform.matrix[:] = original
        first, count = 5, transform.column_batch
        tile = transform._work[:count * transform.a].reshape(count, transform.a)
        expected = original[:, first:first + count].T.copy()
        observed_rows = []
        copyto = np.copyto

        def bounded_copy(destination, source, *args, **kwargs):
            for array in (destination, source):
                if isinstance(array, np.memmap):
                    assert array.ndim == 2
                    assert array.strides[-1] == array.itemsize
                    assert array.shape[0] <= transform.io_row_block
                    assert array.shape[1] <= transform.io_column_block
                    assert array.nbytes <= transform._twiddle.nbytes
                    observed_rows.append(array.shape[0])
            return copyto(destination, source, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(disk_module.np, "copyto", bounded_copy)
            transform._copy_column_tile(first, tile)
            np.testing.assert_array_equal(tile, expected)
            tile += 1 + 2j
            transform._copy_column_tile(first, tile, scatter=True)
        original[:, first:first + count] += 1 + 2j
        np.testing.assert_array_equal(transform.matrix, original)
        assert transform.io_row_block in observed_rows
        assert transform.a % transform.io_row_block in observed_rows
        assert transform.memory["io_staging_reuses_twiddle"] is True


def test_advice_disabled_and_scalar_progress_preserve_transform(fftw, tmp_path):
    events = []
    values = np.random.default_rng(810).normal(size=80)
    with DiskFFT(fftw, 80, tmp_path / "progress.c128", memory_mb=.1,
                 factors=(8, 10), io_advice=False, progress=events.append) as transform:
        transform.x[:] = values
        transform.execute()
        _assert_dft_and_power(transform, values, np.empty(41))
        assert transform.metadata["io_advice"]["enabled"] is False
        assert transform.metadata["io_advice"]["outcomes"] == {}
        for phase in ("real_expansion", "column_fft", "row_fft", "power_reorder"):
            records = [event for event in events if event["phase"] == phase]
            assert records[0]["completed"] == 0
            assert records[-1]["completed"] == records[-1]["total"]
            assert [event["completed"] for event in records] == sorted(event["completed"] for event in records)
        assert all(set(event) == {"phase", "completed", "total"} for event in events)
        assert all(isinstance(event["completed"], int) and isinstance(event["total"], int) for event in events)


def test_optional_madvise_failure_or_absence_does_not_raise(monkeypatch):
    class UnsupportedAdvice:
        def madvise(self, advice):
            raise OSError("advice unsupported by this mapping")

    transform = DiskFFT.__new__(DiskFFT)
    transform.io_advice = True
    transform._advice_outcomes = {}
    transform._mapping = SimpleNamespace(_mmap=UnsupportedAdvice())
    monkeypatch.setattr(disk_module.mmap, "MADV_RANDOM", 1, raising=False)
    transform._advise("random")
    assert transform._advice_outcomes["random"].startswith("unavailable: OSError:")
    transform._mapping = SimpleNamespace(_mmap=object())
    transform._advise("random")
    assert transform._advice_outcomes["random"] == "unavailable"


@pytest.mark.parametrize("factors", [(8, 10), (7, 9)])
def test_dc_and_highest_frequency_endpoint_weights(fftw, tmp_path, factors):
    n, fs = factors[0] * factors[1], 17.0
    powers = np.empty(n // 2 + 1)
    with DiskFFT(fftw, n, tmp_path / "endpoints.c128", memory_mb=.1,
                 factors=factors) as transform:
        transform.x.fill(1.0)
        transform.execute()
        transform.power_into(powers, fs, n)
        assert powers[0] == pytest.approx(n / fs, rel=2e-13)
        assert np.sum(powers) * fs / n == pytest.approx(1.0, rel=3e-13)
        values = np.cos(2 * np.pi * (n // 2) * np.arange(n) / n)
        transform.x[:] = values
        transform.execute()
        transform.power_in_frequency_order_into(powers, fs, n)
        expected = 1.0 if n % 2 == 0 else .5
        assert powers[-1] * fs / n == pytest.approx(expected, rel=3e-12)
        assert np.sum(powers) * fs / n == pytest.approx(expected, rel=3e-12)


@pytest.mark.parametrize("case_name,nsd_absolute_limit", [
    ("dc_nanovolt_order0", 3e-23),
    ("offbin_tone", 3e-14),
])
def test_original_preprocessing_dc_noise_and_offbin_tone(
        fftw, tmp_path, case_name, nsd_absolute_limit):
    values = CASES[case_name].data.to_numpy().copy()
    original = values.copy()
    n, fs = len(values), 50.0
    provider = adapter.KaiserWindow("native", api._LIB._name)
    operations = FFTWOperations("native", api._LIB)
    with DiskFFT(fftw, n, tmp_path / "windowed.c128", memory_mb=1) as transform:
        transform.window[:] = provider.generate(n, 25.402566024390598)
        _, s2 = provider.sums(transform.window)
        operations.prepare(values, transform.window, transform.x)
        windowed = transform.x.copy()
        expected = adapter.one_sided_psd(np.fft.rfft(windowed), n, fs, s2)
        transform.execute()
        actual = transform.power_into(np.empty(n // 2 + 1), fs, s2)
        error = np.max(np.abs(np.sqrt(actual) - np.sqrt(expected)))
        assert error <= nsd_absolute_limit
        actual_psd = actual.astype(np.float32)
        actual_nsd = np.sqrt(actual_psd.astype(np.complex64)).real
        np.testing.assert_allclose(actual_nsd, np.sqrt(expected),
                                   rtol=3e-7, atol=nsd_absolute_limit)
        np.testing.assert_array_equal(values, original)


def test_close_preserves_workspace_and_rejects_use(fftw, tmp_path):
    path = tmp_path / "owned_by_caller.c128"
    transform = DiskFFT(fftw, 80, path, memory_mb=.1, factors=(8, 10))
    transform.x[:] = np.arange(80)
    transform.execute()
    saved = _ordered_copy(transform)
    transform.close()
    transform.close()
    assert transform.plan is None and transform.x is None and transform.matrix is None
    assert path.exists() and path.stat().st_size == 16 * 80
    # Another adapter may reuse exactly this file after the FFT releases it.
    remapped = np.memmap(path, mode="r+", dtype=np.complex128, shape=(8, 10))
    try:
        np.testing.assert_array_equal(np.array(remapped.T, order="C").reshape(-1), saved)
    finally:
        remapped._mmap.close()
    with pytest.raises(RuntimeError, match="closed"):
        transform.execute()
    with pytest.raises(RuntimeError, match="closed"):
        transform.power_into(np.empty(41), 50, 80)
    path.unlink()
    assert not path.exists()


def test_constructor_guards_before_creating_files(fftw, tmp_path):
    path = tmp_path / "must_not_exist.c128"
    with pytest.raises(ValueError, match=r"A\*B"):
        DiskFFT(fftw, 10000, path, factors=(100, 101))
    with pytest.raises(MemoryError, match="subtransform"):
        DiskFFT(fftw, 10000, path, factors=(125, 80), memory_mb=.001)
    with pytest.raises(TypeError, match="io_advice"):
        DiskFFT(fftw, 80, path, io_advice="random")
    with pytest.raises(TypeError, match="progress"):
        DiskFFT(fftw, 80, path, progress=1)
    assert not path.exists()
    path.write_bytes(b"owned existing content")
    with pytest.raises(FileExistsError):
        DiskFFT(fftw, 80, path, factors=(8, 10), memory_mb=.1)
    assert path.read_bytes() == b"owned existing content"


def test_power_destination_and_state_guards(fftw, tmp_path):
    with DiskFFT(fftw, 80, tmp_path / "state.c128", factors=(8, 10),
                 memory_mb=.1) as transform:
        with pytest.raises(RuntimeError, match="before"):
            transform.power_into(np.empty(41), 50, 80)
        transform.x.fill(0)
        transform.execute()
        with pytest.raises(ValueError, match="float64"):
            transform.power_into(np.empty(41, dtype=np.float32), 50, 80)
        with pytest.raises(ValueError, match="overlap"):
            transform.power_into(transform.x[:41], 50, 80)
        with pytest.raises(ValueError, match="Positive finite"):
            transform.power_into(np.empty(41), 0, 80)
        np.testing.assert_array_equal(transform.power_into(np.empty(41), 50, 80), np.zeros(41))


def test_billion_sample_memory_estimate_has_no_large_allocations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Scalar memory estimation must not allocate arrays or map files")
    with monkeypatch.context() as patch:
        patch.setattr(disk_module.np, "memmap", forbidden)
        patch.setattr(disk_module.np, "empty", forbidden)
        patch.setattr(disk_module.np, "arange", forbidden)
        memory = DiskFFT.memory_estimate(1_000_000_000, memory_mb=256)
        memory128 = DiskFFT.memory_estimate(1_000_000_000, memory_mb=128)
    assert memory["factors"] == [31250, 32000]
    assert memory["workspace_bytes"] == 16_000_000_000
    assert memory["known_ram_buffer_bytes"] <= 256 * 1024**2
    assert memory["ram_tile_bytes"] <= 128 * 1024**2
    assert memory["maximum_fftw_transform_length"] == 32000
    assert memory128["known_ram_buffer_bytes"] <= 128 * 1024**2
    assert memory128["column_batch"] == 134 and memory128["row_batch"] == 131
    assert memory128["io_row_block"] == 256 and memory128["io_column_block"] == 134
    assert 16 * memory128["io_row_block"] * memory128["io_column_block"] <= memory128["twiddle_scratch_bytes"]
    assert DiskFFT.memory_estimate(100_000_000)["factors"] == [10000, 10000]
