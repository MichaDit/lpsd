"""Numerical gates for bounded coefficient generation and long-segment LPSD.

All numerical cases are small. The billion-length case generates only a few
bounded windows, not a billion-element vector or a performance measurement.
"""
import ctypes as ct

import numpy as np
import pytest

from lpsd_fast import api


@pytest.fixture(scope="module")
def native():
    api._native()
    lib = api._LIB
    dp = ct.POINTER(ct.c_double)
    lp = ct.POINTER(ct.c_long)
    lib.generate_kaiser_coefficients_range.argtypes = [
        dp, dp, ct.c_long, ct.c_long, ct.c_long, ct.c_double, ct.c_double, ct.c_bool]
    lib.generate_kaiser_coefficients_range.restype = ct.c_int
    lib.fast_dft_kaiser_streaming.argtypes = [
        dp, dp, lp, dp, ct.c_long, ct.c_long,
        ct.c_double, ct.c_double, ct.c_double, ct.c_int,
        ct.c_long, ct.c_long, dp, dp, dp, dp]
    lib.fast_dft_kaiser_streaming.restype = ct.c_int
    return lib


@pytest.mark.parametrize("length", [2048, 2051, 65539])
@pytest.mark.parametrize("beta,frequency_bin", [(25.402566024390598, 1.17), (32.0, 1305.0)])
def test_streaming_window_and_phase_ranges_match_existing_bits(native, length, beta, frequency_bin):
    window = np.empty(length)
    assert native.generate_kaiser_series(api._pointer(window), length, beta) == 0
    real, imag = np.empty(length), np.empty(length)
    assert native.generate_coefficients_blocked(
        api._pointer(real), api._pointer(imag), api._pointer(window), length, frequency_bin) == 0
    for first in (0, (length // 2 // 128) * 128, ((length - 1) // 128) * 128):
        count = min(389, length - first)
        actual_r, actual_i = np.empty(count + 2), np.empty(count + 2)
        actual_r[-2:] = [771.0, -883.0]
        actual_i[-2:] = [665.0, -997.0]
        assert native.generate_kaiser_coefficients_range(
            api._pointer(actual_r), api._pointer(actual_i), length, first, count,
            beta, frequency_bin, False) == 0
        np.testing.assert_array_equal(actual_r[:count], window[first:first + count])
        assert native.generate_kaiser_coefficients_range(
            api._pointer(actual_r), api._pointer(actual_i), length, first, count,
            beta, frequency_bin, True) == 0
        np.testing.assert_array_equal(actual_r[:count], real[first:first + count])
        np.testing.assert_array_equal(actual_i[:count], imag[first:first + count])
        np.testing.assert_array_equal(actual_r[-2:], [771.0, -883.0])
        np.testing.assert_array_equal(actual_i[-2:], [665.0, -997.0])


def _reference(native, samples, length, beta, frequency_bin, overlap, mode):
    real, imag = np.empty(length), np.empty(length)
    assert native.generate_kaiser_series(api._pointer(real), length, beta) == 0
    s1, s2 = ct.c_double(), ct.c_double()
    assert native.window_sums(api._pointer(real), length, ct.byref(s1), ct.byref(s2)) == 0
    assert native.generate_coefficients_blocked(
        api._pointer(real), api._pointer(imag), api._pointer(real), length, frequency_bin) == 0
    pr, pi, vr, vi, k = ct.c_double(), ct.c_double(), ct.c_double(), ct.c_double(), ct.c_long()
    status = native.fast_dft_selected(
        ct.byref(pr), ct.byref(pi), ct.byref(vr), ct.byref(vi), ct.byref(k),
        api._pointer(samples), api._pointer(samples), len(samples), length,
        api._pointer(real), api._pointer(imag), overlap, 0, False, mode, False, True)
    assert status == 0
    return complex(pr.value, pi.value), k.value, s1.value, s2.value


def _streaming(native, samples, length, beta, frequency_bin, overlap, mode, block):
    pr, pi, k = ct.c_double(), ct.c_double(), ct.c_long()
    s1, s2, prep, segments = ct.c_double(), ct.c_double(), ct.c_double(), ct.c_double()
    status = native.fast_dft_kaiser_streaming(
        ct.byref(pr), ct.byref(pi), ct.byref(k), api._pointer(samples), len(samples), length,
        beta, frequency_bin, overlap, mode, block, 256,
        ct.byref(s1), ct.byref(s2), ct.byref(prep), ct.byref(segments))
    assert status == 0
    assert prep.value >= 0 and segments.value >= 0
    return complex(pr.value, pi.value), k.value, s1.value, s2.value


@pytest.mark.parametrize("length", [2048, 4099, 32769])
@pytest.mark.parametrize("mode", [2, 3])
@pytest.mark.parametrize("signal", ["white", "dc_nanovolt", "tone", "constant"])
def test_streaming_retains_native_plan_normalization_and_spectrum(native, length, mode, signal):
    n = 3 * length + 51
    rng = np.random.default_rng(716 + length)
    if signal == "white":
        samples = rng.normal(size=n)
    elif signal == "dc_nanovolt":
        samples = 7.0 + rng.normal(scale=1e-9, size=n)
    elif signal == "tone":
        samples = .23 * np.sin(2 * np.pi * 7.39 * np.arange(n) / length) + rng.normal(scale=1e-7, size=n)
    else:
        samples = np.full(n, 7.0)
    samples = np.ascontiguousarray(samples)
    beta, frequency_bin, overlap = 25.402566024390598, 4.79, 76.58465180619339
    reference = _reference(native, samples, length, beta, frequency_bin, overlap, mode)
    for block in (128, 768, 32768):
        actual = _streaming(native, samples, length, beta, frequency_bin, overlap, mode, block)
        assert actual[1:] == reference[1:]
        np.testing.assert_allclose(actual[0], reference[0], rtol=2e-10, atol=1e-30)
        # The production output retains its existing complex64 rounding and
        # sqrt location. Exact equality is a stronger useful small-case gate.
        reference_psd = np.complex64(2 * reference[0] / 50 / reference[3])
        actual_psd = np.complex64(2 * actual[0] / 50 / actual[3])
        assert actual_psd == reference_psd


@pytest.mark.parametrize("length", [2048, 65539])
def test_streaming_single_segment_and_full_record(native, length):
    samples = np.random.default_rng(length).normal(size=length)
    args = (samples, length, 25.402566024390598, 1.0, 76.58465180619339, 2)
    reference = _reference(native, *args)
    actual = _streaming(native, *args, 128)
    assert actual[1] == reference[1] == 1
    assert actual[2:] == reference[2:]
    np.testing.assert_allclose(actual[0], reference[0], rtol=2e-10, atol=0)


def test_billion_length_window_range_requires_only_bounded_arrays(native):
    real, imag = np.empty(256), np.empty(256)
    for first in (0, 499_999_872, 999_999_744):
        assert native.generate_kaiser_coefficients_range(
            api._pointer(real), api._pointer(imag), 1_000_000_000,
            first, 256, 25.402566024390598, 1.0, True) == 0
        assert np.isfinite(real).all() and np.isfinite(imag).all()
        assert np.max(np.abs(real + 1j * imag)) <= 1 + 1e-14


def test_streaming_unsupported_request_does_not_modify_input(native):
    samples = np.arange(8192, dtype=np.float64)
    original = samples.copy()
    pr, pi, k = ct.c_double(), ct.c_double(), ct.c_long()
    values = [ct.c_double() for _ in range(4)]
    for length, beta, mode, block, max_segments, expected in (
            (1024, 25., 2, 128, 256, 2), (2048, 40., 2, 128, 256, 4),
            (2048, 25., 0, 128, 256, 4), (2048, 25., 2, 129, 256, 1),
            (2048, 25., 2, 128, 1, 4)):
        status = native.fast_dft_kaiser_streaming(
            ct.byref(pr), ct.byref(pi), ct.byref(k), api._pointer(samples), len(samples), length,
            beta, 1., 75., mode, block, max_segments, *(ct.byref(value) for value in values))
        assert status == expected
    np.testing.assert_array_equal(samples, original)
