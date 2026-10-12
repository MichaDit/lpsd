"""Independent all-segment mean and nominal-SEM contracts.

The corrected estimator is checked against individual segment periodograms,
not against the known-defective legacy recurrence. Historical compatibility
tests elsewhere continue to exercise explicitly selected legacy statistics.
"""
import ctypes as ct
import math

import numpy as np
import pandas as pd
import pytest

import lpsd
import lpsd_fast
from lpsd_fast import api
from lpsd_fast.planning import _kaiser_alpha
from lpsd_fast.prepared import PreparedLPSDSubset, _native_prepared


def _impulse_window(length):
    result = np.zeros(length)
    result[0] = 1.0
    return result


def _window(kind, length, psll=80):
    if kind == 'kaiser':
        return np.kaiser(length + 1, np.pi * _kaiser_alpha(psll))[:-1]
    if kind == 'hann':
        return np.hanning(length)
    if kind == 'boxcar':
        return np.ones(length)
    return kind(length)


def _residual(values, order):
    """Independent least-squares residual, retaining enough reference precision."""
    values = np.asarray(values, dtype=np.longdouble)
    if order is None:
        return values
    centered = values - np.mean(values, dtype=np.longdouble)
    if order == 0:
        return centered
    axis = np.arange(len(values), dtype=np.longdouble) - (len(values) - 1) / 2
    return centered - axis * np.sum(axis * centered) / np.sum(axis * axis)


def _periodograms(x, y, length, frequency_bin, overlap, window, order):
    """Form every segment independently; use a separate two-pass variance."""
    n = len(x)
    native_overlap = (overlap * 100.0) / 100.0
    count = math.floor((n - length) * (1.0 / (1.0 - native_overlap)) / length + 1.5)
    shift = 1.0 if count == 1 else (n - length) / (count - 1)
    phase = np.exp((2j * np.pi * frequency_bin / length) * np.arange(length))
    coefficients = np.asarray(window * phase, dtype=np.clongdouble)
    start = 0.0
    result = []
    for k in range(count):
        first = math.floor(start + .5)
        start += shift
        if k == count - 1:
            first = min(first, n - length)
        sx = _residual(x[first:first + length], order)
        sy = sx if y is None else _residual(y[first:first + length], order)
        zx = np.sum(coefficients * sx, dtype=np.clongdouble)
        zy = zx if y is None else np.sum(coefficients * sy, dtype=np.clongdouble)
        result.append(zx * np.conj(zy))
    return np.asarray(result, dtype=np.clongdouble)


def _independent_outputs(x, y, plan, fs, overlap, kind, order, psll=80):
    expected = {name: [] for name in ('ps', 'psd', 'ps_std', 'psd_std')}
    actual_counts = []
    for frequency_bin, length in zip(plan[2], plan[3]):
        length = int(length)
        window = _window(kind, length, psll)
        powers = _periodograms(x, y, length, frequency_bin, overlap, window, order)
        count = len(powers)
        mean = np.mean(powers, dtype=np.clongdouble)
        sem = (np.sqrt(np.sum(np.abs(powers - mean) ** 2) / (count * (count - 1)))
               if count > 1 else np.longdouble(np.nan))
        s1, s2 = sum(window), sum(window * window)
        expected['ps'].append(2 * mean / s1**2)
        expected['psd'].append(2 * mean / fs / s2)
        expected['ps_std'].append(2 * sem / s1**2)
        expected['psd_std'].append(2 * sem / fs / s2)
        actual_counts.append(count)
    return {name: np.asarray(values, dtype=np.float32 if name.endswith('_std') else np.complex64)
            for name, values in expected.items()}, actual_counts


def _native_values(powers, *, cross=False, statistics=True, legacy=False, mode=0):
    """Exact quarter-cycle coefficient fixture; no Fourier implementation oracle."""
    powers = np.asarray(powers, dtype=np.complex128)
    length = 4
    x = np.zeros(length * len(powers), dtype=np.float64)
    y = np.zeros_like(x)
    if cross:
        x[::length], x[1::length] = powers.real, powers.imag
        y[::length] = 1.0
    else:
        x[::length] = np.sqrt(powers.real)
        y[:] = x
    real = np.array([1.0, 0.0, -1.0, 0.0])
    imag = np.array([0.0, 1.0, 0.0, -1.0])
    mean_r, mean_i, variance_r, variance_i = (ct.c_double() for _ in range(4))
    count = ct.c_long()
    args = (ct.byref(mean_r), ct.byref(mean_i),
            ct.byref(variance_r) if statistics else None,
            ct.byref(variance_i) if statistics else None, ct.byref(count),
            api._pointer(x), api._pointer(y), len(x), length,
            api._pointer(real), api._pointer(imag), 0.0, -1, cross, mode,
            statistics, False)
    if legacy:
        status = api._LIB.fast_dft_selected(*args)
    else:
        status = api._LIB.fast_dft_selected_corrected(*args, math.inf, None, None, None)
    assert status == 0
    return complex(mean_r.value, mean_i.value), complex(variance_r.value, variance_i.value), count.value


@pytest.mark.parametrize('powers,mean,sem', [([1, 4], 2.5, 1.5),
                                            ([1, 4, 9], 14 / 3, 7 / 3),
                                            ([9, 4, 1], 14 / 3, 7 / 3)])
def test_exact_all_segment_mean_and_accumulated_variance(powers, mean, sem):
    actual_mean, variance, count = _native_values(powers)
    assert count == len(powers)
    assert actual_mean == pytest.approx(mean, rel=2e-15, abs=0)
    assert variance.imag == 0
    assert math.sqrt(variance.real / count) == pytest.approx(sem, rel=2e-15, abs=0)


@pytest.mark.parametrize('powers,mean,sem', [([1j, -1j], 0j, 1.0),
                                            ([1, 1j, -1], 1j / 3, 2 / 3)])
def test_complex_variance_uses_squared_magnitude_not_pseudovariance(powers, mean, sem):
    actual_mean, variance, count = _native_values(powers, cross=True)
    assert actual_mean == pytest.approx(mean, rel=2e-15, abs=2e-16)
    assert variance.imag == 0
    assert variance.real >= 0
    assert math.sqrt(variance.real / count) == pytest.approx(sem, rel=2e-15, abs=0)


def test_public_two_segment_fixture_is_corrected_by_default_and_legacy_is_explicit():
    data = pd.Series([1.0, 0, 0, 0, 2.0, 0, 0, 0])
    options = dict(sample_rate=1.0, n_frequencies=1, n_averages=1, overlap=0,
                   detrending_order=None, window_function=_impulse_window)
    corrected = lpsd_fast.lpsd(data, workers=1, kernel='scalar', **options)
    legacy = lpsd_fast.lpsd(data, workers=1, kernel='scalar', statistics='legacy', **options)
    original = lpsd.lpsd(data, **options)
    assert corrected.index[-1] == .25
    assert corrected.psd.iloc[-1] == 5.0
    assert corrected.psd_std.iloc[-1] == 3.0
    assert legacy.psd.iloc[-1] == 8.0
    assert legacy.psd_std.iloc[-1] == 0.0
    pd.testing.assert_frame_equal(legacy, original, check_exact=True)
    assert corrected.attrs['lpsd_fast']['statistics'] == 'corrected'
    assert corrected.attrs['lpsd_fast']['legacy_statistics'] is False


@pytest.mark.parametrize('kernel', ['scalar', 'auto', 'fast'])
@pytest.mark.parametrize('kind', ['kaiser', 'hann', 'boxcar'])
@pytest.mark.parametrize('order', [None, 0, 1])
@pytest.mark.parametrize('cross', [False, True])
def test_corrected_public_outputs_against_independent_segment_periodograms(kernel, kind, order, cross):
    n, fs, overlap, psll = 513, 50.0, .4, 80
    rng = np.random.default_rng(624511)
    x = rng.normal(size=n) + .3 * np.arange(n) / n
    y = .2 * x + rng.normal(size=n) if cross else None
    values = pd.DataFrame({'x': x, 'y': y}) if cross else x
    plan = api._frequency_plan(n, fs, overlap, 1, 16, 16, 6)
    expected, counts = _independent_outputs(x, y, plan, fs, overlap, kind, order, psll)
    actual = lpsd_fast.lpsd(values, sample_rate=fs, overlap=overlap,
                            window_function=kind, detrending_order=order,
                            min_segment_length=16, n_frequencies=16, n_averages=6,
                            psll=psll, kernel=kernel, workers=2, csd=cross, profile=True)
    np.testing.assert_array_equal(actual.index, plan[0])
    for name, reference in expected.items():
        np.testing.assert_allclose(actual[name], reference, rtol=2e-6, atol=0, equal_nan=True,
                                   err_msg=f'{kernel}/{kind}/{order}/{cross}/{name}')
    assert [row['K'] for row in actual.attrs['lpsd_profile']['frequencies']] == counts
    assert actual.psd_std.dtype == np.dtype('float32')
    assert actual.ps_std.dtype == np.dtype('float32')
    assert actual.attrs['lpsd_fast']['sem_kind'] == 'nominal_independent_segments'
    assert actual.attrs['lpsd_fast']['sem_correlation_adjusted'] is False


@pytest.mark.parametrize('kernel', ['scalar', 'auto', 'fast'])
@pytest.mark.parametrize('cross', [False, True])
def test_selected_full_and_parallel_corrected_outputs_are_identical(kernel, cross):
    rng = np.random.default_rng(40865)
    x = rng.normal(size=1025)
    values = pd.DataFrame({'x': x, 'y': np.roll(x, 2) + rng.normal(size=len(x))}) if cross else x
    options = dict(sample_rate=50.0, n_frequencies=24, n_averages=8,
                   kernel=kernel, csd=cross, statistics='corrected')
    full = lpsd_fast.lpsd(values, workers=1, **options)
    parallel = lpsd_fast.lpsd(values, workers=2, **options)
    pd.testing.assert_frame_equal(full, parallel, check_exact=True)
    for outputs in [('psd',), ('ps', 'psd_std'), ('ps_std', 'psd_std')]:
        selected = lpsd_fast.lpsd(values, workers=2, outputs=outputs, **options)
        pd.testing.assert_frame_equal(selected, full[list(outputs)], check_exact=True)
    if not cross:
        nsd = lpsd_fast.lnsd(values, sample_rate=50.0, n_frequencies=24,
                            n_averages=8, kernel=kernel, workers=1)
        np.testing.assert_array_equal(nsd.nsd, full.asd)


@pytest.mark.parametrize('kernel', ['scalar', 'auto', 'fast'])
def test_single_segment_has_valid_psd_and_undefined_sem(kernel):
    values = np.random.default_rng(736).normal(size=257)
    options = dict(sample_rate=50, n_frequencies=8, n_averages=1, kernel=kernel, workers=1,
                   min_segment_length=16, outputs=('psd', 'psd_std'), profile=True)
    full = lpsd_fast.lpsd(values, **options)
    counts = np.array([row['K'] for row in full.attrs['lpsd_profile']['frequencies']])
    assert (counts == 1).any()
    assert np.isfinite(full.psd).all()
    assert np.isnan(full.psd_std[counts == 1]).all()
    assert np.isfinite(full.psd_std[counts > 1]).all()
    only = lpsd_fast.lpsd(values, **(options | {'outputs': 'psd'}))
    np.testing.assert_array_equal(only.psd, full.psd)
    assert only.attrs['lpsd_fast']['variance_computed'] is False


def test_variance_work_is_selected_separately_from_mean(monkeypatch):
    recorded = []
    native = api._LIB.fast_dft_selected_corrected

    def capture(*args):
        recorded.append(bool(args[15]))
        return native(*args)

    monkeypatch.setattr(api._LIB, 'fast_dft_selected_corrected', capture)
    x = np.random.default_rng(553).normal(size=257)
    options = dict(sample_rate=50, n_frequencies=8, n_averages=4,
                   min_segment_length=16, kernel='scalar', workers=1)
    for outputs, expected in [('psd', False), ('nsd', False), ('ps', False),
                              ('psd_std', True), ('ps_std', True)]:
        recorded.clear()
        result = lpsd_fast.lpsd(x, outputs=outputs, **options)
        assert recorded and all(flag is expected for flag in recorded)
        assert result.attrs['lpsd_fast']['variance_computed'] is expected


def test_psd_only_does_not_need_representable_variance():
    mean, _, count = _native_values([1e200, 4e200, 9e200], statistics=False)
    assert count == 3 and np.isfinite(mean)
    assert mean.real == pytest.approx(14e200 / 3, rel=2e-15)


@pytest.mark.parametrize('scale', [1e-50, 1.0, 1e50])
def test_real_sample_variance_never_becomes_negative(scale):
    # Repeated, nearly equal powers exercise cancellation without a spectral null.
    values = scale * np.array([1.0, 1.0 + 4e-15, 1.0 - 4e-15, 1.0, 1.0])
    mean, variance, count = _native_values(values)
    assert count == len(values) and np.isfinite(mean)
    assert np.isfinite(variance) and variance.real >= 0 and variance.imag == 0


def test_complex_opposite_finite_values_keep_finite_mean_when_difference_overflows():
    mean, _, count = _native_values([1e308, -1e308], cross=True, statistics=False)
    assert count == 2
    assert mean == 0j


def test_csd_conjugation_scaling_and_auto_equivalence():
    rng = np.random.default_rng(707)
    x, y = rng.normal(size=1025), rng.normal(size=1025)
    options = dict(sample_rate=50, n_frequencies=24, n_averages=8, workers=2,
                   outputs=('psd', 'psd_std'))
    xy = lpsd_fast.lcsd(pd.DataFrame({'x': x, 'y': y}), **options)
    yx = lpsd_fast.lcsd(pd.DataFrame({'y': y, 'x': x}), **options)
    scaled = lpsd_fast.lcsd(pd.DataFrame({'x': -2 * x, 'y': 3 * y}), **options)
    xx = lpsd_fast.lcsd(pd.DataFrame({'x': x, 'same': x}), **options)
    auto = lpsd_fast.lpsd(x, **options)
    np.testing.assert_array_equal(xy.psd, np.conj(yx.psd))
    np.testing.assert_allclose(xy.psd_std, yx.psd_std, rtol=2e-6, atol=0, equal_nan=True)
    np.testing.assert_allclose(scaled.psd, -6 * xy.psd, rtol=2e-6, atol=0)
    np.testing.assert_allclose(scaled.psd_std, 6 * xy.psd_std, rtol=2e-6, atol=0, equal_nan=True)
    pd.testing.assert_frame_equal(xx, auto, check_exact=True)
    assert xy.psd.dtype == np.dtype('complex64')
    assert xy.psd_std.dtype == np.dtype('float32')


def test_rolling_boxcar_corrected_mean_and_sem_against_segment_reference():
    n, fs, overlap = 16385, 50.0, .9
    x = np.random.default_rng(951).normal(size=n)
    plan = api._frequency_plan(n, fs, overlap, 1, 256, 32, 64)
    expected, counts = _independent_outputs(x, None, plan, fs, overlap, 'boxcar', 0)
    actual = lpsd_fast.lpsd(x, sample_rate=fs, overlap=overlap, window_function='boxcar',
                            min_segment_length=256, n_frequencies=32, n_averages=64,
                            kernel='fast', workers=1, profile=True)
    assert actual.attrs['lpsd_fast']['rolling_boxcar_frequencies'] > 0
    for name, reference in expected.items():
        np.testing.assert_allclose(actual[name], reference, rtol=2e-6, atol=0, equal_nan=True)
    assert [row['K'] for row in actual.attrs['lpsd_profile']['frequencies']] == counts


@pytest.mark.parametrize('cache_mb,workers', [(0, 1), (1, 2)])
def test_prepared_corrected_points_have_independent_all_segment_mean(cache_mb, workers):
    n, fs, overlap, psll = 8193, 50.0, .5, 80
    x = np.random.default_rng(157).normal(size=n)
    x[:1000] *= 20  # A first-segment contribution must not disappear.
    plan = api._frequency_plan(n, fs, overlap, 1, 32, 12, 8)
    expected, _ = _independent_outputs(x, None, plan, fs, overlap, 'kaiser', 0, psll)
    with PreparedLPSDSubset(n, fs, plan, psll=psll, overlap=overlap,
                            workers=workers, max_cache_mb=cache_mb) as prepared:
        actual = prepared.compute(x)
        np.testing.assert_allclose(actual, expected['psd'].real, rtol=2e-6, atol=0)
        assert prepared.metadata['statistics'] == 'corrected'
        assert prepared.metadata['variance_computed'] is False
        assert prepared.metadata['legacy_first_segment_recurrence'] is False


@pytest.mark.parametrize('kernel', ['auto', 'fast'])
@pytest.mark.parametrize('channels', ['auto', 'cross', 'unequal_cross', 'zero_cross'])
def test_long_segment_batches_and_csd_fallback_use_corrected_statistics(kernel, channels):
    n, fs, overlap, psll = 32769, 50.0, .5, 80
    rng = np.random.default_rng(37011)
    x = rng.normal(size=n)
    y = None if channels == 'auto' else rng.normal(size=n)
    if channels == 'unequal_cross':
        # Huge projections force the native CSD range fallback while all
        # cross periodograms and their sample variance stay representable.
        x *= 1e150
        y *= 1e-150
    elif channels == 'zero_cross':
        x *= 1e150
        y[:] = 0
    plan = api._frequency_plan(n, fs, overlap, 1, 2048, 16, 16)
    expected, counts = _independent_outputs(x, y, plan, fs, overlap, 'kaiser', 0, psll)
    values = x if y is None else pd.DataFrame({'x': x, 'y': y})
    actual = lpsd_fast.lpsd(values, sample_rate=fs, overlap=overlap,
                            psll=psll, n_frequencies=16, n_averages=16,
                            min_segment_length=2048, kernel=kernel, workers=1,
                            csd=y is not None, outputs=('psd', 'psd_std'), profile=True)
    assert max(counts) >= 8 and min(plan[3]) >= 2048
    for name in ('psd', 'psd_std'):
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=0, equal_nan=True)
    assert np.isfinite(actual.psd).all()
    assert (actual.psd_std.dropna() >= 0).all()


@pytest.mark.parametrize('streaming', ['off', 'on'])
def test_disk_corrected_mean_with_owned_mappings_and_independent_reference(tmp_path, streaming):
    from benchmarks._lpsd_disk import DiskLPSDSubset

    n, fs, overlap, psll = 12291, 50.0, .5, 80
    x = np.random.default_rng(85621).normal(size=n)
    x[:2048] *= 11
    plan = api._frequency_plan(n, fs, overlap, 1, 2048, 12, 12)
    expected, _ = _independent_outputs(x, None, plan, fs, overlap, 'kaiser', 0, psll)
    data = np.memmap(tmp_path / 'statistics-input.bin', mode='w+', dtype=np.float64, shape=(n,))
    scratch = np.memmap(tmp_path / 'statistics-scratch.bin', mode='w+',
                        dtype=np.complex128, shape=(n,))
    data[:] = x
    try:
        with DiskLPSDSubset(n, fs, plan, psll=psll, overlap=overlap,
                            workers=2, max_working_mb=64, streaming=streaming,
                            streaming_block_samples=128) as method:
            result = method.compute(data, scratch, outputs=('psd', 'nsd'), profile=True)
            np.testing.assert_allclose(result.psd, expected['psd'].real, rtol=2e-6, atol=0)
            np.testing.assert_array_equal(result.nsd, np.sqrt(result.psd.to_numpy(dtype=np.complex64)).real)
            assert method.metadata['statistics'] == 'corrected'
            assert method.metadata['legacy_first_segment_recurrence'] is False
            assert bool(method.last_profile['streaming_points']) is (streaming == 'on')
        np.testing.assert_array_equal(data, x)
        assert not data._mmap.closed and not scratch._mmap.closed
    finally:
        data._mmap.close()
        scratch._mmap.close()


@pytest.mark.parametrize('mode', [2, 3])
@pytest.mark.parametrize('length,block', [(2048, 128), (4099, 768)])
def test_streaming_corrected_mean_against_independent_segments(mode, length, block):
    n, beta, frequency_bin, overlap = 3 * length + 51, 25.402566024390598, 4.79, .7658465180619339
    x = np.random.default_rng(716 + length).normal(size=n)
    x[:length] *= 7
    window = np.kaiser(length + 1, beta)[:-1]
    powers = _periodograms(x, None, length, frequency_bin, overlap, window, 0)
    expected = complex(np.mean(powers, dtype=np.clongdouble))
    dp, lp = ct.POINTER(ct.c_double), ct.POINTER(ct.c_long)
    fn = api._LIB.fast_dft_kaiser_streaming_corrected
    fn.argtypes = [dp, dp, lp, dp, ct.c_long, ct.c_long,
                   ct.c_double, ct.c_double, ct.c_double, ct.c_int,
                   ct.c_long, ct.c_long, dp, dp, dp, dp]
    fn.restype = ct.c_int
    pr, pi, s1, s2, prep, segments = (ct.c_double() for _ in range(6))
    count = ct.c_long()
    status = fn(ct.byref(pr), ct.byref(pi), ct.byref(count), api._pointer(x), n, length,
                beta, frequency_bin, overlap * 100, mode, block, 256,
                ct.byref(s1), ct.byref(s2), ct.byref(prep), ct.byref(segments))
    assert status == 0 and count.value == len(powers)
    np.testing.assert_allclose(complex(pr.value, pi.value), expected, rtol=2e-10, atol=0)


@pytest.mark.parametrize('bad', [None, False, True, 1, '', 'auto', 'CORRECTED', [], np.array([1])])
def test_invalid_statistics_modes_are_rejected(bad):
    with pytest.raises(ValueError, match='statistics'):
        lpsd_fast.lpsd(np.ones(17), sample_rate=1, statistics=bad)
    with pytest.raises(ValueError, match='statistics'):
        PreparedLPSDSubset(17, 1, (), statistics=bad)


def test_original_prepared_native_version_is_retained():
    assert _native_prepared().fast_dft_prepared_version() == 1
