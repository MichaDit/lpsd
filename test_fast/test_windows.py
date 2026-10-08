"""Window acceleration keeps the original window and normalization semantics."""
import numpy as np
import pandas as pd
import pytest

import lpsd
import lpsd_fast

from ._support import PRIMARY, compare

WINDOWS = (
    ("kaiser", np.kaiser), ("hann", np.hanning), ("hanning", np.hanning),
    ("hamming", np.hamming), ("blackman", np.blackman),
    ("bartlett", np.bartlett), ("boxcar", np.ones),
)
KWARGS = dict(sample_rate=50., n_frequencies=32, n_averages=4,
              min_segment_length=16, overlap=.5)


def signal():
    return pd.Series(np.random.default_rng(23214).standard_normal(513))


@pytest.mark.parametrize("name,window", WINDOWS, ids=[item[0] for item in WINDOWS])
@pytest.mark.parametrize("kernel", ("scalar", "auto"))
def test_window_name_matches_numpy_callable(name, window, kernel):
    data = signal()
    named = lpsd_fast.lpsd(data, window_function=name, kernel=kernel, workers=1, **KWARGS)
    callback = lpsd_fast.lpsd(data, window_function=window, kernel=kernel, workers=1, **KWARGS)
    compare(named, callback, exact=True)


@pytest.mark.parametrize("name,window", WINDOWS, ids=[item[0] for item in WINDOWS])
@pytest.mark.parametrize("kernel", ("scalar", "auto"))
def test_standard_windows_preserve_upstream_spectra(name, window, kernel):
    data = signal()
    reference = lpsd.lpsd(data, window_function=window, use_c_core=True, **KWARGS)
    actual = lpsd_fast.lpsd(data, window_function=name, kernel=kernel, workers=1, **KWARGS)
    compare(actual, reference, exact=kernel == "scalar",
            columns=tuple(reference.columns) if kernel == "scalar" else PRIMARY)


@pytest.mark.parametrize("name", ("hann", "hamming", "blackman", "bartlett", "boxcar"))
def test_nonkaiser_window_requires_explicit_overlap(name):
    kwargs = {key: value for key, value in KWARGS.items() if key != "overlap"}
    with pytest.raises(ValueError):
        lpsd_fast.lpsd(signal(), window_function=name, workers=1, **kwargs)


def test_unknown_window_name_is_rejected():
    with pytest.raises(ValueError):
        lpsd_fast.lpsd(signal(), window_function="not-a-window", workers=1, **KWARGS)


def test_opaque_callback_is_not_recognized_by_its_name():
    def hanning(length):
        return np.hamming(length)

    data = signal()
    actual = lpsd_fast.lpsd(data, window_function=hanning, workers=2, **KWARGS)
    expected = lpsd.lpsd(data, window_function=hanning, **KWARGS)
    compare(actual, expected, columns=PRIMARY)


def test_callback_reused_noncontiguous_buffer_is_copied_before_next_call():
    data = signal()
    buffer = np.empty(2*len(data), dtype=np.float64)

    def reusable(length):
        view = buffer[:2*length:2]
        view[:] = np.hamming(length)
        return view

    parallel = lpsd_fast.lpsd(data, window_function=reusable,
                              kernel="scalar", workers=4, **KWARGS)
    expected = lpsd.lpsd(data, window_function=np.hamming, **KWARGS)
    compare(parallel, expected, exact=True)
