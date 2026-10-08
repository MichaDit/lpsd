"""High-overlap Boxcar reuses boundaries without changing the LPSD grid."""
import numpy as np
import pandas as pd
import pytest

from lpsd_fast import lpsd


PARAMETERS = dict(sample_rate=50., n_frequencies=96, n_averages=100,
                  min_segment_length=256, overlap=.9, kernel="fast", workers=1)


def direct_boxcar(length):
    """An opaque but identical window deliberately retains direct segments."""
    return np.ones(length)


def make_signal(kind, n=32769):
    rng = np.random.default_rng(20261008)
    noise = rng.standard_normal(n)
    t = np.arange(n) / 50.
    if kind == "white":
        return noise
    if kind == "dc_nanovolt":
        return 10. + 1e-9 * noise
    if kind == "offbin":
        return np.sin(2*np.pi*3.123*t)
    if kind == "weak_tone":
        return np.sin(2*np.pi*3.01*t) + 1e-8*np.sin(2*np.pi*8.04*t)
    if kind == "onbin":
        return np.sin(2*np.pi*17*np.arange(n)/n)
    if kind == "ramp":
        return np.linspace(-1e6, 1e6, n) + 1e-4*noise
    if kind == "steps":
        return np.where(np.arange(n) % 3001 < 500, 1., -1.) + 1e-8*noise
    if kind == "alternating":
        return np.where(np.arange(n) % 2, 1., -1.)
    if kind == "impulses":
        values = np.zeros(n)
        values[[0, 133, 10004, n-1]] = [1., -7., 3., 9.]
        return values
    if kind == "constant":
        return np.full(n, 10.)
    if kind == "zero":
        return np.zeros(n)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ("white", "dc_nanovolt", "offbin", "weak_tone",
                                   "onbin", "ramp", "steps", "alternating",
                                   "impulses", "constant", "zero"))
def test_rolling_boxcar_preserves_outputs(kind):
    values = make_signal(kind)
    expected = lpsd(values, window_function=direct_boxcar, **PARAMETERS)
    actual = lpsd(values, window_function="boxcar", **PARAMETERS)
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    assert actual.dtypes.equals(expected.dtypes)
    if kind == "impulses":
        assert actual.attrs["lpsd_fast"]["rolling_boxcar_frequencies"] == 0
    else:
        assert actual.attrs["lpsd_fast"]["rolling_boxcar_frequencies"] > 0
    assert expected.attrs["lpsd_fast"]["rolling_boxcar_frequencies"] == 0
    for column in actual:
        assert np.isfinite(actual[column]).all(), column
        # A scale-aware diagnostic floor admits only roundoff-level powers
        # near mathematical nulls, not a unit-independent physical tolerance.
        reference = expected[column].to_numpy()
        scale_column = {"ps_std": "ps", "psd_std": "psd"}.get(column, column)
        absolute_floor = max(float(np.max(np.abs(expected[scale_column]))) * 1e-12, 1e-35)
        np.testing.assert_allclose(actual[column], reference, rtol=2e-5,
                                   atol=absolute_floor, err_msg=f"{kind}/{column}")
    if kind in ("constant", "zero"):
        assert not actual.psd.to_numpy().any()


@pytest.mark.parametrize("n,overlap", ((4096, .8), (4097, .81), (16385, .9), (32771, .95)))
def test_segment_rounding_and_rebases_match_direct(n, overlap):
    values = make_signal("white", n)
    kwargs = PARAMETERS | dict(overlap=overlap, outputs=("psd", "nsd"), profile=True)
    expected = lpsd(values, window_function=direct_boxcar, **kwargs)
    actual = lpsd(values, window_function="boxcar", **kwargs)
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)
    actual_rows = actual.attrs["lpsd_profile"]["frequencies"]
    expected_rows = expected.attrs["lpsd_profile"]["frequencies"]
    assert [row["K"] for row in actual_rows] == [row["K"] for row in expected_rows]
    assert [row["L"] for row in actual_rows] == [row["L"] for row in expected_rows]
    used = [row for row in actual_rows if row["segment_method"] == "rolling_boxcar"]
    assert used
    assert all(row["rolling_rebases"] == (row["K"]+31)//32 for row in used)
    assert all(row["c_preparation_s"] >= 0 and row["c_segments_s"] >= 0 for row in actual_rows)


def test_selection_workers_and_nsd_do_not_change_rolling_spectrum():
    values = make_signal("white", 16385)
    full = lpsd(values, window_function=np.ones, **PARAMETERS)
    selected = lpsd(values, window_function="boxcar",
                    **(PARAMETERS | dict(workers=4, outputs=("nsd", "psd"))))
    assert full.psd.to_numpy().tobytes() == selected.psd.to_numpy().tobytes()
    assert full.asd.to_numpy().tobytes() == selected.nsd.to_numpy().tobytes()
    assert full.attrs["lpsd_fast"]["rolling_boxcar_frequencies"] == selected.attrs["lpsd_fast"]["rolling_boxcar_frequencies"]


@pytest.mark.parametrize("override", (
    dict(overlap=.5), dict(detrending_order=1), dict(detrending_order=None),
    dict(kernel="auto"), dict(kernel="scalar"), dict(outputs="enbw"),
))
def test_ineligible_calls_keep_direct_segments(override):
    actual = lpsd(make_signal("white", 4097), window_function="boxcar",
                  **(PARAMETERS | override))
    assert actual.attrs["lpsd_fast"]["rolling_boxcar_frequencies"] == 0


def test_cross_spectrum_keeps_direct_segments():
    values = make_signal("white", 4097)
    data = pd.DataFrame({"a": values, "b": .5*values+np.roll(values, 1)})
    kwargs = PARAMETERS | dict(csd=True)
    actual = lpsd(data, window_function="boxcar", **kwargs)
    expected = lpsd(data, window_function=direct_boxcar, **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs["lpsd_fast"]["rolling_boxcar_frequencies"] == 0


@pytest.mark.parametrize("amplitude", (1., 1e9, 1e15))
def test_departed_leading_transient_does_not_create_a_noise_floor(amplitude):
    values = np.zeros(32769)
    values[0] = amplitude
    kwargs = PARAMETERS | dict(outputs=("psd", "nsd"), profile=True)
    expected = lpsd(values, window_function=direct_boxcar, **kwargs)
    actual = lpsd(values, window_function="boxcar", **kwargs)
    # Upstream's mean recurrence discards the first segment. Every later
    # segment is exactly zero for this input; a state residual is not noise.
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs["lpsd_fast"]["rolling_boxcar_frequencies"] == 0


@pytest.mark.parametrize("amplitude,outputs", ((1e77, None), (1e155, "psd")))
def test_extreme_power_intermediates_retain_direct_semantics(amplitude, outputs):
    values = make_signal("white", 4097) * amplitude
    kwargs = PARAMETERS | dict(outputs=outputs)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = lpsd(values, window_function=direct_boxcar, **kwargs)
        actual = lpsd(values, window_function="boxcar", **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs["lpsd_fast"]["rolling_boxcar_frequencies"] == 0


@pytest.mark.parametrize("seed", (2, 17, 99))
@pytest.mark.parametrize("amplitude", (1e9, 1e10, 1e11, 1e12))
def test_first_legacy_mean_reset_with_large_transient_and_quiet_noise(seed, amplitude):
    values = np.random.default_rng(seed).standard_normal(32769)
    # This transient lies inside the first hop but is not the anchor. It
    # disappears from segment 1 while the quiet noise continues throughout.
    values[1] += amplitude
    kwargs = PARAMETERS | dict(outputs=("psd", "nsd"))
    expected = lpsd(values, window_function=direct_boxcar, **kwargs)
    actual = lpsd(values, window_function="boxcar", **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs["lpsd_fast"]["rolling_boxcar_frequencies"] == 0


@pytest.mark.parametrize("kind", ("white", "dc_nanovolt", "offbin", "weak_tone",
                                   "onbin", "constant"))
def test_longer_boxcar_psd_and_nsd_against_scalar_reference(kind):
    values = make_signal(kind, 131073)
    kwargs = PARAMETERS | dict(outputs=("psd", "nsd"))
    expected = lpsd(values, window_function="boxcar", **(kwargs | dict(kernel="scalar")))
    actual = lpsd(values, window_function="boxcar", **kwargs)
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    assert actual.dtypes.equals(expected.dtypes)
    for column, floor in (("psd", 1e-24), ("nsd", 1e-12)):
        left = expected[column].to_numpy().astype(np.float64)
        right = actual[column].to_numpy().astype(np.float64)
        difference = np.abs(left-right)
        assert np.isfinite(left).all() and np.isfinite(right).all()
        # These explicit absolute floors describe only this synthetic input
        # family; callers must choose a floor in their physical signal units.
        assert np.all((difference < .01*np.abs(left)) | (difference <= floor)), column
