"""Deterministic, small inputs and comparison helpers; no external baseline."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

import lpsd
import lpsd_fast

COLUMNS = ("ps", "psd", "ps_std", "psd_std", "enbw", "asd", "asdrms")
PRIMARY = ("ps", "psd", "enbw", "asd", "asdrms")
RTOL = 2e-6  # Final spectra are float32/complex64; the native work uses double.
DEFAULTS = dict(sample_rate=50.0, n_frequencies=64, n_averages=8)


@dataclass
class Case:
    data: object
    kwargs: dict


def cases():
    rng = np.random.default_rng(20261008)
    result = {}

    def add(name, values, **kwargs):
        if not isinstance(values, (pd.Series, pd.DataFrame)):
            values = pd.Series(values, name="signal")
        result[name] = Case(values, DEFAULTS | kwargs)

    add("white_odd", rng.standard_normal(511))
    spectrum = np.fft.rfft(rng.standard_normal(2049))
    spectrum[0] = 0
    spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
    pink = np.fft.irfft(spectrum, n=2049)
    add("pink", pink / pink.std())
    time = np.arange(1025) / 50.0
    mixed = 0.07 * rng.standard_normal(1025) + 0.003 * time + np.sin(2*np.pi*3.123*time)
    add("drift_tone_order0", mixed)
    add("drift_tone_order1", mixed, detrending_order=1)
    nano = 10 + 1e-9 * rng.standard_normal(2049)
    add("dc_nanovolt_order0", nano)
    add("dc_nanovolt_order1", nano, detrending_order=1)
    ramp_rng = np.random.default_rng(20261009)
    add("ramp_nanovolt_order1", np.linspace(-10., 10., 1025) + 1e-9*ramp_rng.standard_normal(1025), detrending_order=1)
    add("no_detrending", 10 + rng.standard_normal(1025), detrending_order=None)
    add("short_5", rng.standard_normal(5))
    add("short_9", rng.standard_normal(9))
    add("hanning", rng.standard_normal(1025), window_function=np.hanning, overlap=0.5)
    add("kaiser80_no_overlap", rng.standard_normal(1025), psll=80, overlap=0.)
    add("min_segment", rng.standard_normal(1025), min_segment_length=128)
    add("one_average", rng.standard_normal(1025), n_averages=1)
    add("float32", rng.standard_normal(513).astype(np.float32))
    add("integer", rng.integers(-100, 100, size=513))
    add("order2", rng.standard_normal(513), detrending_order=2)
    add("one_column", pd.DataFrame({"signal": rng.standard_normal(513)}))
    add("two_columns", pd.DataFrame(rng.standard_normal((513, 2)), columns=["left", "right"]))
    dates = pd.date_range("2026-01-01", periods=513, freq="20ms")
    add("datetime_rate", pd.Series(rng.standard_normal(513), index=dates), sample_rate=None)
    shared = rng.standard_normal(1025)
    cross = pd.DataFrame({"a": shared + np.sin(2*np.pi*4.32*time),
                          "b": .7*shared + np.sin(2*np.pi*4.32*time+.61)})
    add("csd", cross, csd=True)
    add("offbin_tone", np.sin(2*np.pi*3.123*time), detrending_order=None)
    add("onbin_tone", np.sin(2*np.pi*17*np.arange(1024)/1024), detrending_order=None)
    time = np.arange(2049) / 50.0
    add("weak_tone", np.sin(2*np.pi*3.01*time) + 1e-8*np.sin(2*np.pi*8.04*time))
    add("constant", np.full(257, 10.0))
    add("zero", np.zeros(257))
    return result


CASES = cases()
TONES = ("offbin_tone", "onbin_tone", "weak_tone")
ORDINARY = tuple(name for name in CASES if name not in TONES + ("constant", "zero"))


@lru_cache(maxsize=None)
def run(name, kernel=None, workers=1):
    case = CASES[name]
    # Give upstream its own copy: its API can mutate inputs when removing NaNs.
    data = case.data.copy(deep=True)
    if kernel is None:
        return lpsd.lpsd(data, use_c_core=True, **case.kwargs)
    return lpsd_fast.lpsd(data, kernel=kernel, workers=workers, **case.kwargs)


def compare(actual, expected, *, exact=False, columns=COLUMNS):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert tuple(actual) == tuple(expected)
        for key in expected:
            compare(actual[key], expected[key], exact=exact, columns=columns)
        return
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    assert tuple(actual.columns) == tuple(expected.columns) == COLUMNS
    assert actual.dtypes.equals(expected.dtypes)
    for column in columns:
        left, right = actual[column].to_numpy(), expected[column].to_numpy()
        assert np.isfinite(left).all(), column
        if exact:
            # Include signed zero and every component of complex outputs.
            assert left.tobytes() == right.tobytes(), column
        else:
            np.testing.assert_allclose(left, right, rtol=RTOL, atol=0, err_msg=column)
