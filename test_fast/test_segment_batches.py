"""Batched projections retain starts, auto-spectrum semantics and legacy M2."""
import ctypes as ct

import numpy as np
import pytest

from lpsd_fast import api


def native_result(samples, real, imag, mode, order, *, selected, statistics=True):
    api._native()
    result = [ct.c_double() for _ in range(4)]
    count = ct.c_long()
    cr, ci = real.copy(), imag.copy()
    arguments = (
        *(ct.byref(value) for value in result), ct.byref(count),
        api._pointer(samples), None, len(samples), len(cr),
        api._pointer(cr), api._pointer(ci), 75.0, order, False, mode,
    )
    if selected:
        status = api._LIB.fast_dft_selected(*arguments, statistics, True)
    else:
        status = api._LIB.fast_dft(*arguments)
    assert status == 0
    return np.asarray([value.value for value in result]), count.value


@pytest.mark.parametrize("length,segments", (
    (127, 8), (128, 8), (129, 9), (255, 15), (256, 16),
    (511, 17), (512, 31), (1023, 7), (1023, 9), (1024, 9),
    (2047, 8), (2048, 7), (2049, 13), (16384, 17),
))
@pytest.mark.parametrize("mode,order", ((2, 0), (2, 1), (3, 0)))
@pytest.mark.parametrize("signal", ("white", "dc_noise", "tone", "constant"))
def test_native_batches_match_independent_segment_projections(length, segments, mode, order, signal):
    # A non-integral hop exercises the inherited repeated floating-point
    # start update and its rounding, including 8-way/4-way/single remainders.
    n = length + round((segments - 1) * (length / 4 + 0.123))
    rng = np.random.default_rng(4819 + length)
    samples = rng.normal(size=n)
    if signal == "dc_noise":
        samples = 10.0 + 1e-9 * samples
    elif signal == "tone":
        samples = np.sin(2 * np.pi * 4.371 * np.arange(n) / length)
    elif signal == "constant":
        samples.fill(10.0)
    phase = 2 * np.pi * 4.371 * np.arange(length) / length
    window = np.kaiser(length + 1, 23.7)[:-1]
    cr, ci = window * np.cos(phase), window * np.sin(phase)
    expected, expected_count = native_result(samples, cr, ci, mode, order, selected=False)
    actual, actual_count = native_result(samples, cr, ci, mode, order, selected=True)
    selected, selected_count = native_result(samples, cr, ci, mode, order, selected=True, statistics=False)
    assert actual_count == expected_count == selected_count == segments
    # SIMD lane reductions may round differently, but the underlying power,
    # final legacy variance and exactly real auto-spectrum must be retained.
    scale = max(float(abs(expected[0])), np.finfo(float).tiny)
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=2e-12, atol=scale * 2e-14)
    np.testing.assert_allclose(actual[2:], expected[2:], rtol=2e-9, atol=scale * scale * 2e-13)
    assert actual[1] == actual[3] == 0.0
    assert selected[:2].tobytes() == actual[:2].tobytes()
    assert np.isnan(selected[2:]).all()
    if signal == "constant":
        assert np.all(actual == 0.0)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, 1e155, 1e308))
def test_native_batch_exceptional_arithmetic_is_not_simplified(value):
    # The public API rejects nonfinite input, but the native entry point must
    # still retain its prior NaN/overflow behavior and avoid forcing PSD real.
    length = 129
    samples = np.random.default_rng(4142).normal(size=length * 8)
    samples[::11] = value
    phase = 2 * np.pi * 3.71 * np.arange(length) / length
    cr, ci = np.cos(phase), np.sin(phase)
    expected, expected_count = native_result(samples, cr, ci, 2, 0, selected=False)
    actual, actual_count = native_result(samples, cr, ci, 2, 0, selected=True)
    assert actual_count == expected_count
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=2e-12, atol=0)


@pytest.mark.parametrize("length", (128, 132, 256, 260, 512, 516, 1024, 1028, 2048, 2052, 4096, 4100))
@pytest.mark.parametrize("segments", (9, 17, 33))
def test_first_periodogram_cancellation_keeps_the_legacy_reset(length, segments):
    # The unique ii=1 update is P0 + (P1-P0). An O(eps*P0) dot-product
    # difference can become a material final PSD error when P0/P1~1/eps.
    # Keep the first four starts exactly equal to the old <=4-segment path;
    # the fractional eighth-sample hop still exercises later start rounding.
    hop = length / 4 + 0.125
    n = int(length + (segments - 1) * hop)
    samples = np.random.default_rng(9191 + length).normal(size=n)
    samples[0] = 0.0
    phase = 2 * np.pi * 4.371 * np.arange(length) / length
    window = np.kaiser(length + 1, 23.7)[:-1]
    cr, ci = window * np.cos(phase), window * np.sin(phase)
    first_hop = int(np.floor(hop + 0.5))
    p1 = native_result(samples[first_hop:first_hop + length], cr, ci, 2, 0, selected=False)[0][0]
    impulse = np.zeros(length)
    impulse[1] = 1.0
    impulse_power = native_result(impulse, cr, ci, 2, 0, selected=False)[0][0]
    assert p1 > 0 and impulse_power > 0
    tail_powers = []
    start = 0.0
    for index in range(segments):
        offset = int(np.floor(start + 0.5))
        start += hop
        if index >= 4:
            power = native_result(samples[offset:offset + length], cr, ci, 2, 0, selected=False)[0][0]
            tail_powers.append(power)
    for exponent in (50, 53, 56):
        amplitude = np.sqrt(p1 * 2.0**exponent / impulse_power)
        for scale in (np.nextafter(1.0, 0.0), 1.0, np.nextafter(1.0, 2.0)):
            samples[1] = amplitude * scale
            prefix = length + 3 * (length // 4)
            expected, count = native_result(samples[:prefix], cr, ci, 2, 0, selected=True)
            assert count == 4
            mean = float(expected[0])
            for index, power in enumerate(tail_powers, 4):
                mean += (float(power) - mean) / index
            actual, count = native_result(samples, cr, ci, 2, 0, selected=True)
            assert count == segments
            assert actual[1] == 0.0
            np.testing.assert_allclose(actual[0], mean, rtol=0.01, atol=0.0)
