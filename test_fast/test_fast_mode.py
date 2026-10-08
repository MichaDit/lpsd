"""The faster coefficient path is opt-in and checked against the scalar path."""
import numpy as np
import pandas as pd
import pytest

from lpsd_fast import lpsd

from ._support import CASES, PRIMARY, run


@pytest.mark.parametrize("name", (
    "white_odd", "pink", "dc_nanovolt_order0", "dc_nanovolt_order1",
    "ramp_nanovolt_order1", "offbin_tone", "onbin_tone", "weak_tone",
    "constant", "zero", "hanning", "csd",
))
def test_fast_mode_primary_error_below_one_percent(name):
    reference = run(name, "scalar")
    actual = run(name, "fast")
    pd.testing.assert_index_equal(actual.index, reference.index, exact=True)
    for column in PRIMARY:
        expected = reference[column].to_numpy().astype(np.complex128)
        computed = actual[column].to_numpy().astype(np.complex128)
        nonzero = expected != 0
        assert np.isfinite(computed).all()
        assert np.all(computed[~nonzero] == 0), column
        assert np.all(np.abs((computed[nonzero] - expected[nonzero]) / expected[nonzero]) < .01), column


def test_fast_selection_profile_and_workers_are_consistent():
    values = np.random.default_rng(577).normal(size=16385)
    kwargs = dict(sample_rate=50., n_frequencies=96, n_averages=16, kernel="fast")
    full = lpsd(values, workers=1, **kwargs)
    selected = lpsd(values, outputs=("nsd", "psd"), workers=4, profile=True, **kwargs)
    assert selected.psd.to_numpy().tobytes() == full.psd.to_numpy().tobytes()
    assert selected.nsd.to_numpy().tobytes() == full.asd.to_numpy().tobytes()
    metadata = selected.attrs["lpsd_fast"]
    assert metadata["coefficient_method"] == "blocked"
    assert metadata["native_mode"] == (3 if metadata["native_long_double_mantissa_bits"] > 64 else 2)
    assert not metadata["variance_computed"]


def test_fast_linear_detrending_keeps_the_stable_residual_path():
    case = CASES["ramp_nanovolt_order1"]
    result = lpsd(case.data, kernel="fast", workers=1, outputs="psd", **case.kwargs)
    assert result.attrs["lpsd_fast"]["native_mode"] == 1
