"""Periodic flat-top definitions retain their upstream coefficients and phase."""
import numpy as np
import pandas as pd
import pytest

import lpsd
import lpsd_fast
from lpsd_fast import api
from lpsd_fast._windows import COSINE_WINDOWS

from ._support import PRIMARY, compare


@pytest.mark.parametrize("window,coefficients", COSINE_WINDOWS,
                         ids=[window.__name__ for window, _ in COSINE_WINDOWS])
def test_native_flattop_window_values(window, coefficients):
    api._native()
    coefficients = np.asarray(coefficients, dtype=np.float64)
    for length in (1, 2, 3, 17, 256, 4097):
        result = np.empty(length, dtype=np.float64)
        assert api._LIB.generate_cosine_window(
            api._pointer(result), length, api._pointer(coefficients), len(coefficients)
        ) == 0
        expected = window(length)
        np.testing.assert_allclose(result, expected, rtol=2e-13, atol=4e-14)
        # These are periodic length-L windows, not symmetric length-(L-1).
        if length > 2:
            assert result[1] == result[-1]


@pytest.mark.parametrize("window,coefficients", COSINE_WINDOWS,
                         ids=[window.__name__ for window, _ in COSINE_WINDOWS])
def test_flattop_public_spectra(window, coefficients):
    data = pd.Series(np.random.default_rng(10353).normal(size=513))
    kwargs = dict(sample_rate=50., n_frequencies=32, n_averages=4,
                  min_segment_length=16, overlap=.5)
    original = lpsd.lpsd(data, window_function=window, **kwargs)
    scalar = lpsd_fast.lpsd(data, window_function=window.__name__,
                            kernel="scalar", workers=1, **kwargs)
    fast = lpsd_fast.lpsd(data, window_function=window,
                          kernel="auto", workers=2, **kwargs)
    compare(scalar, original, exact=True)
    compare(fast, original, columns=PRIMARY)
    assert fast.attrs["lpsd_fast"]["native_window"]
