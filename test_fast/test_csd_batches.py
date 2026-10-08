"""CSD segment reuse preserves independent channels and legacy statistics."""
import ctypes as ct

import numpy as np
import pandas as pd
import pytest

from lpsd_fast import api


def native_csd(first, second, real, imag, mode=2, order=0, *,
               selected=True, statistics=True):
    api._native()
    values = [ct.c_double() for _ in range(4)]
    count = ct.c_long()
    cr, ci = real.copy(), imag.copy()
    arguments = (
        *(ct.byref(value) for value in values), ct.byref(count),
        api._pointer(first), api._pointer(second), len(first), len(cr),
        api._pointer(cr), api._pointer(ci), 75.0, order, True, mode,
    )
    if selected:
        status = api._LIB.fast_dft_selected(*arguments, statistics, True)
    else:
        status = api._LIB.fast_dft(*arguments)
    assert status == 0
    return np.asarray([value.value for value in values]), count.value


def coefficients(length):
    phase = 2 * np.pi * 4.371 * np.arange(length) / length
    window = np.kaiser(length + 1, 23.7)[:-1]
    return window * np.cos(phase), window * np.sin(phase)


@pytest.mark.parametrize("length,segments", (
    (127, 11), (128, 6), (129, 7), (255, 8), (256, 9),
    (1023, 11), (1024, 9), (2047, 7), (2048, 4), (2049, 5),
    (4095, 6), (4096, 9), (16385, 17),
))
@pytest.mark.parametrize("mode,order", ((2, 0), (2, 1), (3, 0)))
@pytest.mark.parametrize("signal", ("white", "dc_noise", "tone", "constant"))
def test_csd_batches_match_independent_segments(length, segments, mode, order, signal):
    # Nonintegral hops exercise repeated start addition and all remainders.
    n = length + round((segments - 1) * (length / 4 + 0.123))
    rng = np.random.default_rng(91017 + length)
    first = rng.normal(size=n)
    second = 0.7 * first + rng.normal(size=n)
    if signal == "dc_noise":
        first = 10.0 + 1e-9 * first
        second = -7.0 + 2e-9 * second
    elif signal == "tone":
        phase = 2 * np.pi * 4.371 * np.arange(n) / length
        first = np.sin(phase)
        second = 0.7 * np.sin(phase + 0.61)
    elif signal == "constant":
        first.fill(10.0)
        second.fill(-7.0)
    original = first.copy(), second.copy()
    cr, ci = coefficients(length)
    expected, expected_count = native_csd(
        first, second, cr, ci, mode, order, selected=False)
    actual, actual_count = native_csd(first, second, cr, ci, mode, order)
    spectrum, spectrum_count = native_csd(
        first, second, cr, ci, mode, order, statistics=False)
    assert actual_count == expected_count == spectrum_count == segments
    scale = max(float(np.linalg.norm(expected[:2])), np.finfo(float).tiny)
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=2e-12, atol=scale * 2e-14)
    np.testing.assert_allclose(actual[2:], expected[2:], rtol=2e-9, atol=scale**2 * 2e-13)
    assert spectrum[:2].tobytes() == actual[:2].tobytes()
    assert np.isnan(spectrum[2:]).all()
    np.testing.assert_array_equal(first, original[0])
    np.testing.assert_array_equal(second, original[1])
    if signal == "constant":
        assert np.all(actual == 0.0)
    if signal == "tone":
        # Preserve the CSD phase; exchanging channels conjugates the output.
        swapped, _ = native_csd(second, first, cr, ci, mode, order)
        assert actual[1] != 0.0
        np.testing.assert_allclose(swapped, actual * [1, -1, 1, -1],
                                   rtol=2e-12, atol=scale**2 * 2e-14)


@pytest.mark.parametrize("length", (129, 2049))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, 1e155, 1e308))
def test_csd_batches_preserve_native_exceptional_arithmetic(length, value):
    rng = np.random.default_rng(79171)
    first, second = rng.normal(size=(2, length * 5))
    first[::11] = value
    second[::13] = value
    cr, ci = coefficients(length)
    expected, expected_count = native_csd(first, second, cr, ci, selected=False)
    actual, actual_count = native_csd(first, second, cr, ci)
    assert actual_count == expected_count
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=2e-12, atol=0)


@pytest.mark.parametrize("relation", ("same", "negated", "zero", "unequal_scale"))
def test_csd_batches_preserve_channel_relations(relation):
    length = 2049
    first = np.random.default_rng(19424).normal(size=length * 5)
    second = first.copy()
    if relation == "negated":
        second = -second
    elif relation == "zero":
        second.fill(0.)
    elif relation == "unequal_scale":
        first *= 1e100
        second *= 1e-100
    cr, ci = coefficients(length)
    expected, count = native_csd(first, second, cr, ci, selected=False)
    actual, actual_count = native_csd(first, second, cr, ci)
    assert count == actual_count
    scale = max(float(np.linalg.norm(expected[:2])), np.finfo(float).tiny)
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=2e-12, atol=scale * 2e-14)
    np.testing.assert_allclose(actual[2:], expected[2:], rtol=2e-9, atol=scale**2 * 2e-13)
    if relation in ("same", "negated", "zero"):
        assert actual[1] == actual[3] == 0.
    if relation == "zero":
        assert np.all(actual == 0.)


@pytest.mark.parametrize("length", (128, 1024, 2048, 4096))
def test_csd_batches_keep_first_periodogram_reset(length):
    # Reconstruct the recurrence from independent projections when the first
    # periodogram is so large that the inherited ii=1 reset loses precision.
    segments = 11
    hop = length / 4 + 0.125
    n = int(length + (segments - 1) * hop)
    rng = np.random.default_rng(73291 + length)
    first, second = rng.normal(size=(2, n))
    first[0] = second[0] = 0.0
    cr, ci = coefficients(length)
    first_hop = int(np.floor(hop + 0.5))
    second_power, _ = native_csd(
        first[first_hop:first_hop + length],
        second[first_hop:first_hop + length], cr, ci, selected=False)
    impulse = np.zeros(length)
    impulse[1] = 1.0
    impulse_power, _ = native_csd(impulse, impulse, cr, ci, selected=False)
    amplitude = np.sqrt(np.linalg.norm(second_power[:2]) * 2.0**56 / impulse_power[0])
    first[1] = amplitude
    second[1] = -amplitude
    # The actual hop is reconstructed exactly as in the original entry point.
    shift = (n - length) / (segments - 1)
    start = 0.0
    mean = None
    for index in range(segments):
        offset = int(np.floor(start + 0.5))
        start += shift
        power, _ = native_csd(first[offset:offset + length],
                              second[offset:offset + length], cr, ci, selected=False)
        if mean is None:
            mean = power[:2].copy()
        else:
            mean += (power[:2] - mean) / index
    actual, count = native_csd(first, second, cr, ci)
    assert count == segments
    np.testing.assert_allclose(actual[:2], mean, rtol=0.01, atol=0.0)


@pytest.mark.parametrize("order", (0, 1))
def test_public_csd_batches_preserve_columns_dtypes_and_phase(order, monkeypatch):
    rng = np.random.default_rng(82624)
    t = np.arange(32769) / 50.0
    first = rng.normal(size=len(t)) + 0.2 * np.sin(2*np.pi * 3.21 * t)
    second = 0.7 * first + 0.4 * np.sin(2*np.pi * 3.21 * t + 0.61)
    data = pd.DataFrame({"left": first, "right": second})
    kwargs = dict(sample_rate=50.0, n_frequencies=64, n_averages=12,
                  detrending_order=order, csd=True, workers=1)
    api._native()
    # Use the public projection mode for both calls, replacing only its
    # selected entry point with the retained independent-segment C export.
    # Comparing projected versus scalar residuals would also measure their
    # preexisting order-1 rounding differences in tiny legacy deviations.
    with monkeypatch.context() as patch:
        patch.setattr(api._LIB, "fast_dft_selected",
                      lambda *args: api._LIB.fast_dft(*args[:-2]))
        expected = api.lpsd(data, kernel="projected", **kwargs)
    actual = api.lpsd(data, kernel="projected", **kwargs)
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    assert actual.columns.equals(expected.columns)
    assert actual.dtypes.equals(expected.dtypes)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=2e-6, atol=0)
    spectrum = api.lpsd(data, kernel="projected", outputs="psd", **kwargs)
    assert spectrum["psd"].to_numpy().tobytes() == actual["psd"].to_numpy().tobytes()
