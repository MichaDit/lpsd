"""Bounded FP64 FMA keeps the estimator and its exceptional-value fallback."""
import ctypes as ct

import numpy as np
import pytest

from lpsd_fast import api


def _native_result(samples, real, imag, *, bounded, mode=2, order=0,
                   statistics=True, input_peak=None, profile=False):
    api._native()
    output = [ct.c_double() for _ in range(4)]
    count, batches = ct.c_long(), ct.c_long(-1)
    preparation, segments = ct.c_double(), ct.c_double()
    cr, ci = real.copy(), imag.copy()
    arguments = (
        *(ct.byref(value) for value in output), ct.byref(count),
        api._pointer(samples), None, len(samples), len(cr),
        api._pointer(cr), api._pointer(ci), 75.0, order, False, mode,
        statistics, True,
    )
    if bounded:
        if input_peak is None:
            input_peak = float(np.max(np.abs(samples)))
        status = api._LIB.fast_dft_selected_bounded(
            *arguments, input_peak,
            ct.byref(preparation) if profile else None,
            ct.byref(segments) if profile else None, ct.byref(batches))
    else:
        status = api._LIB.fast_dft_selected(*arguments)
    assert status == 0
    return (np.asarray([value.value for value in output]), count.value,
            batches.value, preparation.value, segments.value)


def _problem(length, segments, signal):
    # Repeated fractional starts exercise every batch/remainder boundary.
    n = length + round((segments - 1) * (length / 4 + .125))
    rng = np.random.default_rng(10391 + length)
    samples = rng.normal(size=n)
    if signal == "dc_noise":
        samples = 10.0 + 1e-9 * samples
    elif signal == "tone":
        samples = np.sin(2 * np.pi * 4.371 * np.arange(n) / length)
    elif signal == "constant":
        samples.fill(10.0)
    phase = 2 * np.pi * 4.371 * np.arange(length) / length
    window = np.kaiser(length + 1, 23.7)[:-1]
    return samples, window * np.cos(phase), window * np.sin(phase)


@pytest.mark.parametrize("length,segments", (
    (127, 33), (128, 9), (128, 16), (129, 17), (255, 31),
    (256, 10), (259, 19), (511, 33), (512, 31), (1023, 18),
    (1024, 16), (1027, 17), (2048, 17), (2049, 33), (65537, 17),
))
@pytest.mark.parametrize("mode,order", ((2, 0), (2, 1), (3, 0)))
@pytest.mark.parametrize("signal", ("white", "dc_noise", "tone", "constant"))
def test_bounded_batches_preserve_native_power_and_legacy_statistics(
        length, segments, mode, order, signal):
    samples, cr, ci = _problem(length, segments, signal)
    expected, expected_count, *_ = _native_result(
        samples, cr, ci, bounded=False, mode=mode, order=order)
    actual, count, batches, preparation_s, segment_s = _native_result(
        samples, cr, ci, bounded=True, mode=mode, order=order, profile=True)
    selected, selected_count, selected_batches, *_ = _native_result(
        samples, cr, ci, bounded=True, mode=mode, order=order, statistics=False)
    assert count == expected_count == selected_count == segments
    assert batches == selected_batches
    assert preparation_s >= 0 and segment_s >= 0
    scale = max(float(abs(expected[0])), np.finfo(float).tiny)
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=2e-12, atol=scale * 2e-14)
    np.testing.assert_allclose(actual[2:], expected[2:], rtol=2e-9, atol=scale**2 * 2e-13)
    assert actual[1] == actual[3] == 0
    assert selected[:2].tobytes() == actual[:2].tobytes()
    assert np.isnan(selected[2:]).all()
    if signal == "constant":
        assert np.all(actual == 0)
    eligible = length >= 128 and segments >= (10 if 256 <= length < 1024 else 16)
    if api._LIB.native_segment_fma_supported() and eligible:
        assert batches > 0
    else:
        assert batches == 0
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("target", ("input", "coefficient"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, 1e40, 1e155, 1e308))
def test_extreme_ranges_use_the_unchanged_selected_kernel(target, value):
    samples, cr, ci = _problem(259, 33, "white")
    if target == "input":
        samples[::11] = value
        # Infinity explicitly disables FMA and retains native NaN behavior.
        peak = float(np.max(np.abs(samples))) if np.isfinite(value) else np.inf
    else:
        cr[::11] = value
        peak = float(np.max(np.abs(samples)))
    expected, expected_count, *_ = _native_result(samples, cr, ci, bounded=False)
    actual, count, batches, *_ = _native_result(
        samples, cr, ci, bounded=True, input_peak=peak)
    assert count == expected_count and batches == 0
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert actual[finite].tobytes() == expected[finite].tobytes()


@pytest.mark.parametrize("remainder", range(8))
def test_every_vector_tail_with_fifteen_to_seventeen_segments(remainder):
    samples, cr, ci = _problem(256 + remainder, 15 + remainder % 3, "white")
    expected, expected_count, *_ = _native_result(samples, cr, ci, bounded=False)
    actual, count, batches, *_ = _native_result(samples, cr, ci, bounded=True)
    assert count == expected_count
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-24)
    if api._LIB.native_segment_fma_supported():
        assert batches > 0


@pytest.mark.parametrize("peak", (2.0**100, np.nextafter(2.0**100, np.inf), np.inf))
def test_input_upper_bound_threshold_and_explicit_disable(peak):
    samples, cr, ci = _problem(264, 33, "white")
    expected, expected_count, *_ = _native_result(samples, cr, ci, bounded=False)
    actual, count, batches, *_ = _native_result(
        samples, cr, ci, bounded=True, input_peak=peak)
    assert count == expected_count
    if peak > 2.0**100 or not api._LIB.native_segment_fma_supported():
        assert batches == 0
        assert actual.tobytes() == expected.tobytes()
    else:
        assert batches > 0
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-24)


@pytest.mark.parametrize("coefficient", (
    np.nextafter(2.0**100, 0.0), 2.0**100, np.nextafter(2.0**100, np.inf),
))
def test_projected_coefficient_threshold_neighbors(coefficient):
    samples, cr, ci = _problem(264, 33, "white")
    # Even alternating powers of two have an exact zero mean, so projection
    # leaves the three neighboring boundary values unchanged.
    cr[:] = coefficient
    cr[1::2] *= -1
    ci.fill(0)
    expected, expected_count, *_ = _native_result(samples, cr, ci, bounded=False)
    actual, count, batches, *_ = _native_result(samples, cr, ci, bounded=True)
    assert count == expected_count
    if coefficient > 2.0**100 or not api._LIB.native_segment_fma_supported():
        assert batches == 0
        assert actual.tobytes() == expected.tobytes()
    else:
        assert batches > 0
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=0)


@pytest.mark.parametrize("length", (128, 132, 256, 260, 512, 516, 1024, 1028, 2048, 2052))
@pytest.mark.parametrize("segments", (17, 33))
def test_fused_tail_keeps_the_cancellation_sensitive_initial_mean(length, segments):
    samples, cr, ci = _problem(length, segments, "white")
    # Only the first segment sees this large impulse. Its later removal makes
    # the inherited first mean reset sensitive to an ulp of the first dot.
    first_hop = int(np.floor(length / 4 + .625))
    next_power = _native_result(samples[first_hop:first_hop + length], cr, ci,
                                bounded=False)[0][0]
    impulse = np.zeros(length)
    impulse[1] = 1.0
    impulse_power = _native_result(impulse, cr, ci, bounded=False)[0][0]
    for exponent in (50, 53, 56):
        amplitude = np.sqrt(next_power * 2.0**exponent / impulse_power)
        for rounding in (np.nextafter(1.0, 0.0), 1.0, np.nextafter(1.0, 2.0)):
            samples[1] = amplitude * rounding
            expected, *_ = _native_result(samples, cr, ci, bounded=False)
            actual, count, batches, *_ = _native_result(samples, cr, ci, bounded=True)
            assert count == segments
            np.testing.assert_allclose(actual[0], expected[0], rtol=.01, atol=1e-24)
            assert actual[1] == 0
            if api._LIB.native_segment_fma_supported():
                assert batches > 0


def test_api_reports_fused_work_and_preserves_the_scalar_frequency_grid():
    samples = 10 + 1e-9 * np.random.default_rng(841).normal(size=32769)
    kwargs = dict(sample_rate=1, n_frequencies=80, n_averages=40,
                  workers=1, outputs=("psd", "nsd"))
    result = api.lpsd(samples, kernel="fast", profile=True, **kwargs)
    reference = api.lpsd(samples, kernel="scalar", **kwargs)
    assert result.index.equals(reference.index)
    assert result.dtypes.equals(reference.dtypes)
    np.testing.assert_allclose(result, reference, rtol=.01, atol=1e-24)
    counts = [row["fused_segment_batches"] for row in result.attrs["lpsd_profile"]["frequencies"]]
    assert sum(counts) == result.attrs["lpsd_fast"]["fused_segment_batches"]
    if api._LIB.native_segment_fma_supported():
        assert sum(counts) > 0
    assert result.attrs["lpsd_profile"]["input_bound_s"] >= 0


@pytest.mark.parametrize("workers", (2, None))
def test_parallel_api_keeps_the_ordinary_selected_path(workers, monkeypatch):
    samples = np.random.default_rng(7103).normal(size=16385)
    kwargs = dict(sample_rate=1, n_frequencies=64, n_averages=32,
                  kernel="fast", outputs=("psd", "nsd"))
    api._native()
    with monkeypatch.context() as ordinary:
        ordinary.setattr(api._LIB, "native_segment_fma_supported", lambda: 0)
        reference = api.lpsd(samples, workers=1, **kwargs)

    def reject_bounded_call(*args):
        pytest.fail("Parallel PSD must retain the ordinary selected entry point")

    # Model a capable CPU even on a portable CI target, and exercise both
    # explicit concurrency and the default resolved worker count.
    monkeypatch.setattr(api, "available_workers", lambda: 2)
    monkeypatch.setattr(api._LIB, "native_segment_fma_supported", lambda: 1)
    monkeypatch.setattr(api._LIB, "fast_dft_selected_bounded", reject_bounded_call)
    result = api.lpsd(samples, workers=workers, profile=True, **kwargs)
    assert result.attrs["lpsd_fast"]["workers"] == 2
    assert result.attrs["lpsd_fast"]["fused_segment_batches"] == 0
    assert all(row["segment_method"] == "direct"
               for row in result.attrs["lpsd_profile"]["frequencies"])
    assert result.index.equals(reference.index)
    assert result.dtypes.equals(reference.dtypes)
    assert result.to_numpy().tobytes() == reference.to_numpy().tobytes()
