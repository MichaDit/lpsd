"""Mandatory error bounds plus two narrow, visible, strict known limitations."""
import numpy as np
import pandas as pd
import pytest

import lpsd
import lpsd_fast
from lpsd._helpers import _kaiser_alpha, _kaiser_rov, _ltf_plan

from ._support import CASES, RTOL, TONES, compare, run


@pytest.mark.parametrize("name", TONES)
def test_tone_absolute_roundoff_bound(name):
    """A near-zero relative discrepancy cannot excuse a large amplitude error."""
    case = CASES[name]
    original, actual = run(name), run(name, "auto")
    compare(actual, original, columns=("enbw", "asdrms"))
    kwargs = case.kwargs
    beta = np.pi * _kaiser_alpha(kwargs.get("psll", 200))
    overlap = _kaiser_rov(beta / np.pi)
    _, _, _, lengths, _ = _ltf_plan(len(case.data), kwargs["sample_rate"], overlap,
                                    1, 0, kwargs["n_frequencies"], kwargs["n_averages"])
    eps = np.finfo(np.float64).eps
    max_input = np.abs(case.data.to_numpy()).max()
    bounds = []
    for length in lengths:
        w = np.kaiser(int(length) + 1, beta)[:-1]
        gamma = (int(length) + 4) * (eps / 2)
        gamma /= 1 - gamma
        # Two evaluations, centering/projection and complex dot products.
        # This is a forward roundoff allowance, not statistical uncertainty.
        amplitude = (8*gamma + 64*eps) * max_input * np.abs(w).sum()
        bounds.append(amplitude * np.sqrt(2 / (kwargs["sample_rate"] * np.square(w).sum())))
    bounds = np.asarray(bounds) + 4*np.finfo(np.float32).eps*np.abs(original.asd.to_numpy())
    error = np.abs(actual.asd.to_numpy() - original.asd.to_numpy())
    assert np.isfinite(error).all()
    assert np.all(error <= bounds), f"ASD exceeds roundoff bound by {np.max(error / bounds)}"
    assert (actual.psd >= 0).all()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Off-bin pure-tone sidelobes approach a spectral null: reordered dot products "
    "do not provide a uniform 2e-6 relative PSD guarantee. Absolute ASD is gated separately."
))
def test_known_offbin_tone_uniform_relative_accuracy():
    np.testing.assert_allclose(run("offbin_tone", "auto").psd,
                               run("offbin_tone").psd, rtol=RTOL, atol=0)


def impulse_window(length):
    window = np.zeros(length)
    window[0] = 1.0
    return window


@pytest.fixture(scope="module")
def two_segment_result():
    # At f=.25, L=4, overlap=0, the two windowed segment powers are 1 and 4.
    data = pd.Series([1., 0, 0, 0, 2., 0, 0, 0])
    kwargs = dict(sample_rate=1., n_frequencies=1, n_averages=1, overlap=0.,
                  detrending_order=None, window_function=impulse_window)
    original = lpsd.lpsd(data, **kwargs)
    actual = lpsd_fast.lpsd(data, kernel="auto", workers=1, **kwargs)
    return original, actual


def test_legacy_two_segment_statistics_are_preserved(two_segment_result):
    original, actual = two_segment_result
    compare(actual, original, exact=True)
    assert actual.index[-1] == .25
    assert actual.psd.iloc[-1] == 8.0  # Upstream drops the first periodogram.
    assert actual.psd_std.iloc[-1] == 0.0


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Preserved upstream C estimator defect: K=2 gives zero variance instead of "
    "the standard error of the two periodograms. This optimization does not correct statistics."
))
def test_known_legacy_standard_error_is_not_statistically_correct(two_segment_result):
    _, actual = two_segment_result
    # One-sided periodograms are [2, 8]; sample std / sqrt(2) equals 3.
    expected = np.std([2., 8.], ddof=1) / np.sqrt(2)
    assert actual.psd_std.iloc[-1] == pytest.approx(expected)
