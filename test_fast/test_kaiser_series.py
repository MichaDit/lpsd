"""Exercise the optional positive-series generator and its exact fallbacks."""
import numpy as np
import pytest

from lpsd_fast import api, lpsd


@pytest.mark.parametrize('length', (1, 2047, 2048, 2049, 8193))
@pytest.mark.parametrize('beta', (0., 1e-4, 8., 25.402566024390598, 32., 40.))
def test_kaiser_series_preserves_periodic_window(length, beta):
    api._native()
    direct = np.empty(length)
    series = np.empty(length)
    assert api._LIB.generate_kaiser(api._pointer(direct), length, beta) == 0
    assert api._LIB.generate_kaiser_series(api._pointer(series), length, beta) == 0
    if length < 2048 or beta == 0 or beta > 32:
        np.testing.assert_array_equal(series, direct)
    else:
        np.testing.assert_allclose(series, direct, rtol=5e-14, atol=0)
        np.testing.assert_allclose(series, np.kaiser(length + 1, beta)[:-1], rtol=5e-14, atol=0)
    np.testing.assert_array_equal(series[1:], series[:0:-1])
    if length % 2 == 0:
        assert series[length // 2] == 1.


def test_fast_kaiser_density_keeps_weak_component_accuracy():
    n, fs = 32769, 50.
    t = np.arange(n) / fs
    values = np.sin(2*np.pi*3.01*t) + 1e-9*np.sin(2*np.pi*8.04*t)
    kwargs = dict(sample_rate=fs, n_frequencies=96, n_averages=16,
                  detrending_order=None, workers=1)
    reference = lpsd(values, kernel='scalar', outputs=('psd', 'nsd'), **kwargs)
    actual = lpsd(values, kernel='fast', outputs=('psd', 'nsd'), **kwargs)
    np.testing.assert_array_equal(actual.index, reference.index)
    for name, absolute_limit in (('psd', 1e-24), ('nsd', 1e-12)):
        expected = reference[name].to_numpy().astype(np.float64)
        error = np.abs(actual[name].to_numpy().astype(np.float64) - expected)
        assert np.all((error < .01 * np.abs(expected)) | (error <= absolute_limit))
    assert actual.attrs['lpsd_fast']['kaiser_method'] == 'series_with_fallback'
