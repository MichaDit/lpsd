"""Small numerical, ABI and ownership gates for ordinary disk group power.

No performance threshold, external checkout, recovered artifact or large
transform is required. All numerical comparisons use absolute tolerance zero.
The complete external R3 qualification remains separate evidence.
"""
import ctypes as ct
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import _fftw_direct_power as direct
from benchmarks import bench_fftw
from benchmarks._fftw_disk import DiskFFT
from lpsd_fast import api


@pytest.fixture(scope="module")
def native():
    # Native availability is mandatory through the ordinary suite fixture.
    # Never skip a missing new ABI by testing a Python substitute.
    return direct.DirectLibrary()


@pytest.fixture(scope="module")
def fftw():
    try:
        return bench_fftw.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as error:
        pytest.skip(f"Optional FFTW shared library unavailable: {error}")


def _cuts(n):
    half = n // 2
    return np.asarray(sorted({1, max(1, half // 3), half, half + 1}), dtype=np.int64)


def _matrix(ordered, shape):
    a, b = shape
    return np.ascontiguousarray(np.asarray(ordered, dtype=np.complex128).reshape(b, a).T)


def _oracle(matrix, cuts, fs, s2):
    """Independent frequency-order bin operations, then math.fsum groups."""
    a, b = matrix.shape
    n = a * b
    positive = np.asarray([matrix[k % a, k // a] for k in range(n // 2 + 1)],
                          dtype=np.complex128)
    powers = positive.real * positive.real
    powers += positive.imag * positive.imag
    powers *= 2.0 / (fs * s2)
    powers[0] *= .5
    if n % 2 == 0:
        powers[-1] *= .5
    groups = [math.fsum(map(float, powers[first:stop]))
              for first, stop in zip(cuts[:-1], cuts[1:])]
    return np.asarray(groups), math.fsum(map(float, powers))


def test_integrated_abi_binds_the_selected_native_library(native):
    assert native.lib is api._LIB
    assert native.abi_version == native.lib.dlp_r3_abi_version() == 3
    assert direct.DirectLibrary(api._LIB).lib is api._LIB
    assert len(native.reduce.argtypes) == 12
    assert native.reduce.restype is ct.c_int


def test_missing_or_wrong_abi_does_not_fall_back():
    class Version:
        def __call__(self):
            return 2

    with pytest.raises(RuntimeError, match="unavailable"):
        direct.DirectLibrary(SimpleNamespace())
    with pytest.raises(RuntimeError, match="version 3"):
        direct.DirectLibrary(SimpleNamespace(dlp_r3_abi_version=Version(),
                                            dlp_r3_reduce_rows=object()))


@pytest.mark.parametrize("shape", [(1, 2), (3, 4), (7, 9), (3, 1027)])
def test_original_bin_mapping_parity_row_and_256_bin_tails(native, shape):
    a, b = shape
    n = a * b
    k = np.arange(n, dtype=np.float64)
    matrix = _matrix((k + .125) + 1j * ((k % 7) + .375), shape)
    before = matrix.tobytes()
    # The one-group choice also forces >256-bin C partials in the final case.
    for cuts in (_cuts(n), np.asarray([1, n // 2 + 1], dtype=np.int64)):
        cuts_before = cuts.tobytes()
        expected, energy = _oracle(matrix, cuts, 7., 3.25)
        for rows in dict.fromkeys((None, 1, min(a, 2))):
            result = direct.reduce_matrix(matrix, cuts, 7., 3.25, native,
                                          rows_per_block=rows)
            np.testing.assert_allclose(result["sums"], expected, rtol=1e-12, atol=0)
            np.testing.assert_allclose(result["energy_power_sum"], energy, rtol=1e-12, atol=0)
            np.testing.assert_array_equal(result["counts"], np.diff(cuts))
            metadata = result["metadata"]
            assert metadata["energy_bins"] == n // 2 + 1
            assert metadata["dc_bins"] == 1
            assert metadata["nyquist_bins"] == int(n % 2 == 0)
            assert metadata["matrix_stride_complex"] == b
            assert metadata["native_blocks"] == math.ceil(a / metadata["rows_per_block"])
        assert cuts.tobytes() == cuts_before
    assert matrix.tobytes() == before


def test_singleton_powers_keep_tiny_values_and_exact_original_operations(native):
    # Positive bins include an exact zero and powers spanning about 580 decades.
    # Singleton sums expose bin arithmetic without a group-order tolerance.
    ordered = np.asarray([0, 1e-145, 2e-120, 0, 1, 3e40, 1e145,
                          2, 3, 4, 5, 6], dtype=np.complex128)
    ordered.imag = ordered.real * .375
    matrix = _matrix(ordered, (3, 4))
    cuts = np.arange(1, 8, dtype=np.int64)
    expected, energy = _oracle(matrix, cuts, 50., 12.)
    result = direct.reduce_matrix(matrix, cuts, 50., 12., native)
    np.testing.assert_array_equal(result["sums"], expected)
    assert result["sums"][0] > 0
    assert result["sums"][2] == 0
    np.testing.assert_allclose(result["energy_power_sum"], energy, rtol=1e-12, atol=0)


@pytest.mark.parametrize("shape", [(3, 4), (3, 5)])
def test_zero_dc_and_highest_bin_include_original_imaginary_endpoint(native, shape):
    n = math.prod(shape)
    cuts = np.asarray([1, n // 2 + 1], dtype=np.int64)
    matrix = np.zeros(shape, dtype=np.complex128)
    zero = direct.reduce_matrix(matrix, cuts, 2., 1., native)
    np.testing.assert_array_equal(zero["sums"], [0.])
    assert zero["energy_power_sum"] == 0.
    # The reducer must preserve the actual C2C endpoint imaginary component.
    matrix[0, 0] = 3. + 4.j
    dc = direct.reduce_matrix(matrix, cuts, 2., 1., native)
    np.testing.assert_array_equal(dc["sums"], [0.])
    assert dc["energy_power_sum"] == 12.5
    half = n // 2
    matrix[half % shape[0], half // shape[0]] = 5. + 12.j
    result = direct.reduce_matrix(matrix, cuts, 2., 1., native)
    high_power = 84.5 if n % 2 == 0 else 169.
    np.testing.assert_array_equal(result["sums"], [high_power])
    assert result["energy_power_sum"] == 12.5 + high_power


def test_readonly_borrowed_arrays_produce_owned_bounded_results(native):
    matrix = np.full((6, 11), 2. + 3.j, dtype=np.complex128)
    cuts = _cuts(matrix.size)
    matrix.flags.writeable = cuts.flags.writeable = False
    result = direct.reduce_matrix(matrix, cuts, 50., 8., native)
    assert isinstance(result, dict)
    for name in ("sums", "counts", "cuts"):
        assert type(result[name]) is np.ndarray
        assert not np.shares_memory(result[name], matrix)
        assert not np.shares_memory(result[name], cuts)
    assert not result["cuts"].flags.writeable
    np.testing.assert_array_equal(matrix, np.full(matrix.shape, 2. + 3.j))
    metadata = result["metadata"]
    assert metadata["abi_version"] == 3
    assert metadata["array_state_bytes"] == 48 * len(result["sums"]) + 48
    assert metadata["additional_workers"] == 0
    assert metadata["power_vector_created"] is False
    assert metadata["matrix_written"] is False
    assert metadata["ordinary_result_arrays"] is True


def test_invalid_python_shapes_alignment_cuts_and_scalars_are_rejected(native):
    matrix = np.ones((3, 4), dtype=np.complex128)
    cuts = np.asarray([1, 3, 7], dtype=np.int64)
    unaligned = np.ndarray(matrix.shape, dtype=np.complex128,
                           buffer=bytearray(matrix.nbytes + 1), offset=1)
    unaligned_cuts = np.ndarray(cuts.shape, dtype=np.int64,
                                buffer=bytearray(cuts.nbytes + 1), offset=1)
    unaligned_cuts[:] = cuts
    for invalid in (matrix.real, matrix.T, matrix.reshape(-1), unaligned,
                    np.empty((0, 4), dtype=np.complex128)):
        with pytest.raises(ValueError):
            direct.reduce_matrix(invalid, cuts, 50., 1., native)
    for invalid in (cuts.tolist(), cuts.astype(np.int32), unaligned_cuts,
                    np.asarray([1, 1, 7], dtype=np.int64),
                    np.asarray([0, 3, 7], dtype=np.int64),
                    np.asarray([1, 3, 6], dtype=np.int64)):
        with pytest.raises(ValueError):
            direct.reduce_matrix(matrix, invalid, 50., 1., native)
    for invalid in (0, -1, True, 1.5, 4):
        with pytest.raises(ValueError):
            direct.reduce_matrix(matrix, cuts, 50., 1., native, rows_per_block=invalid)
    for fs, s2 in ((0., 1.), (50., -1.), (math.nan, 1.), (50., math.inf),
                   (1e-300, 1e-300), (1e300, 1e300)):
        with pytest.raises(ValueError):
            direct.reduce_matrix(matrix, cuts, fs, s2, native)


@pytest.mark.parametrize("invalid", [complex(math.nan, 0), complex(0, math.inf),
                                    complex(1e300, 0)])
def test_nonfinite_or_overflowed_native_power_is_rejected(native, invalid):
    matrix = np.ones((3, 4), dtype=np.complex128)
    matrix[0, 0] = invalid
    with pytest.raises(RuntimeError, match="status 3"):
        direct.reduce_matrix(matrix, _cuts(matrix.size), 50., 1., native)


def _native_arguments():
    matrix = np.ones((3, 4), dtype=np.complex128)
    cuts = np.asarray([1, 3, 7], dtype=np.int64)
    state = np.zeros(6, dtype=np.float64)
    counts = np.zeros(5, dtype=np.uint64)
    double, integer, unsigned = ct.POINTER(ct.c_double), ct.POINTER(ct.c_int64), ct.POINTER(ct.c_uint64)
    args = [matrix.ctypes.data_as(double), 12, 3, 4, 4, 0, 3,
            cuts.ctypes.data_as(integer), 2, .25,
            state.ctypes.data_as(double), counts.ctypes.data_as(unsigned)]
    return matrix, cuts, state, counts, args


def test_native_scalar_pointer_overlap_and_previous_count_guards(native):
    matrix, cuts, state, counts, base = _native_arguments()
    original = matrix.tobytes()
    double, integer, unsigned = ct.POINTER(ct.c_double), ct.POINTER(ct.c_int64), ct.POINTER(ct.c_uint64)
    for index, value in ((0, double()), (1, 1), (1, 1_000_000_001), (2, 0),
                         (4, 2), (5, 3), (6, 0), (6, 4), (8, 0), (9, 0.)):
        args = base.copy()
        args[index] = value
        assert native.reduce(*args) == 1, (index, value)
    for index, value in ((7, ct.cast(base[0], integer)),
                         (10, base[0]), (11, ct.cast(base[0], unsigned))):
        args = base.copy()
        args[index] = value
        assert native.reduce(*args) == 2, index
    args = base.copy()
    args[0] = ct.cast(matrix.ctypes.data + 1, double)
    assert native.reduce(*args) == 1
    cuts[:] = [1, 1, 7]
    assert native.reduce(*base) == 1
    cuts[:] = [1, 3, 7]
    counts[0] = 3  # More than the first group's two-bin capacity.
    assert native.reduce(*base) == 4
    counts[:] = 0
    state[0] = math.inf
    assert native.reduce(*base) == 3
    assert matrix.tobytes() == original


def test_native_rejects_non_nearest_rounding_and_restores_environment(native):
    # fenv symbols are not exported by every Windows DLL loader. The test is
    # explicit about that narrow absence; the native ABI tests never skip.
    try:
        get_round, set_round = native.lib.fegetround, native.lib.fesetround
    except AttributeError:
        pytest.skip("Selected platform library does not expose C fenv control symbols")
    get_round.argtypes, get_round.restype = [], ct.c_int
    set_round.argtypes, set_round.restype = [ct.c_int], ct.c_int
    matrix, cuts, state, counts, args = _native_arguments()
    original = get_round()
    if original != 0:
        pytest.skip("This fenv contract needs an initially nearest-rounding environment")
    status = None
    try:
        # Each candidate must be accepted and read back by the platform fenv
        # API; no unsupported constant is assumed valid across architectures.
        for mode in (0x400, 0x800000, 0x100, 2):
            if set_round(mode) == 0 and get_round() == mode:
                status = native.reduce(*args)
                break
    finally:
        assert set_round(original) == 0
    assert get_round() == original
    if status is None:
        pytest.skip("No checked non-nearest fenv mode is exposed on this platform")
    assert status == 5
    np.testing.assert_array_equal(state, np.zeros_like(state))
    np.testing.assert_array_equal(counts, np.zeros_like(counts))
    np.testing.assert_array_equal(matrix, np.ones_like(matrix))


def test_wrong_complete_energy_count_is_not_accepted(native):
    class CountFault:
        abi_version = 3

        def reduce(self, *args):
            status = native.reduce(*args)
            if status == 0:
                args[-1][int(args[8])] -= 1
            return status

    matrix = np.ones((1, 12), dtype=np.complex128)
    with pytest.raises(AssertionError, match="coverage"):
        direct.reduce_matrix(matrix, _cuts(matrix.size), 50., 1., CountFault())


@pytest.mark.parametrize("factors", [(7, 9), (8, 8)])
def test_disk_group_power_refill_parseval_and_result_survives_close(native, fftw, tmp_path, factors):
    n = math.prod(factors)
    path = tmp_path / "direct_groups.c128"
    cuts = _cuts(n)
    transform = DiskFFT(fftw, n, path, memory_mb=.1, factors=factors)
    saved, results = [], []
    try:
        with pytest.raises(RuntimeError):
            transform.group_power(cuts, 50., float(n), native)
        for seed in (71, 83):
            values = np.random.default_rng(seed).standard_normal(n)
            transform.x[:] = values
            transform.execute()
            before = transform.matrix.tobytes()
            # Exercise explicit binding and the ordinary selected-library default.
            result = transform.group_power(cuts, 50., float(n), native)
            repeated = transform.group_power(cuts, 50., float(n))
            np.testing.assert_array_equal(result["sums"], repeated["sums"])
            assert transform.matrix.tobytes() == before
            expected, energy = _oracle(transform.matrix, cuts, 50., float(n))
            np.testing.assert_allclose(result["sums"], expected, rtol=1e-12, atol=0)
            np.testing.assert_allclose(result["energy_power_sum"], energy, rtol=1e-12, atol=0)
            ordered = _matrix(np.fft.fft(values), factors)
            reference, reference_energy = _oracle(ordered, cuts, 50., float(n))
            np.testing.assert_allclose(result["sums"], reference, rtol=1e-10, atol=0)
            np.testing.assert_allclose(result["energy_power_sum"], reference_energy, rtol=1e-10, atol=0)
            time_energy = math.fsum(float(value) * float(value) for value in values) / n
            np.testing.assert_allclose(result["energy_power_sum"] * 50. / n,
                                       time_energy, rtol=1e-10, atol=0)
            for name in ("sums", "counts", "cuts"):
                assert not np.shares_memory(result[name], transform.matrix)
            saved.append(result["sums"].copy())
            results.append(result)
        assert not np.array_equal(saved[0], saved[1])
    finally:
        transform.close()
        transform.close()
    for expected, result in zip(saved, results):
        np.testing.assert_array_equal(result["sums"], expected)
    assert transform._closed and transform.matrix is None
    assert path.exists()  # DiskFFT closes its mapping; the caller owns the file.
    with pytest.raises(RuntimeError):
        transform.group_power(cuts, 50., float(n), native)
