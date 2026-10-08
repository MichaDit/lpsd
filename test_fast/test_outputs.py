"""Selected outputs preserve the same kernel's full-result rounding path."""
import numpy as np
import pandas as pd
import pytest

import lpsd_fast
from lpsd_fast import api

from ._support import CASES, COLUMNS, DEFAULTS, run


def assert_selected(actual, full, names):
    if isinstance(full, dict):
        assert tuple(actual) == tuple(full)
        for key in full:
            assert_selected(actual[key], full[key], names)
        return
    assert isinstance(actual, pd.DataFrame)
    assert tuple(actual.columns) == tuple(names)
    pd.testing.assert_index_equal(actual.index, full.index, exact=True)
    for name in names:
        reference = full["asd" if name == "nsd" else name].to_numpy()
        values = actual[name].to_numpy()
        assert values.dtype == reference.dtype, name
        assert values.tobytes() == reference.tobytes(), name


def selected(name, outputs, kernel="auto", workers=1):
    case = CASES[name]
    return lpsd_fast.lpsd(case.data, outputs=outputs, kernel=kernel,
                          workers=workers, **case.kwargs)


@pytest.mark.parametrize("kernel", ("scalar", "auto"))
@pytest.mark.parametrize("outputs", (None, "all"))
def test_explicit_full_output_keeps_default(kernel, outputs):
    assert_selected(selected("white_odd", outputs, kernel),
                    run("white_odd", kernel), COLUMNS)


@pytest.mark.parametrize("kernel", ("scalar", "auto"))
@pytest.mark.parametrize("outputs", (
    "ps", "psd", "ps_std", "psd_std", "enbw", "asd", "asdrms", "nsd",
    ("nsd", "asd", "psd"), ("psd_std", "ps", "asd"),
))
def test_selected_outputs_match_full_result(kernel, outputs):
    names = (outputs,) if isinstance(outputs, str) else outputs
    assert_selected(selected("white_odd", outputs, kernel),
                    run("white_odd", kernel), names)


@pytest.mark.parametrize("name", (
    "dc_nanovolt_order0", "ramp_nanovolt_order1", "offbin_tone",
    "short_5", "constant", "zero", "two_columns",
))
def test_selected_nsd_preserves_adversarial_rounding(name):
    assert_selected(selected(name, ("nsd", "psd")),
                    run(name, "auto"), ("nsd", "psd"))


@pytest.mark.parametrize("outputs", ("psd", "nsd", ("enbw", "psd_std", "asdrms")))
def test_selected_parallel_is_deterministic(outputs):
    names = (outputs,) if isinstance(outputs, str) else outputs
    single = selected("white_odd", outputs)
    parallel = selected("white_odd", outputs, workers=4)
    # No alias remapping is needed when both sides already use selected names.
    for name in names:
        assert parallel[name].dtype == single[name].dtype
        assert parallel[name].to_numpy().tobytes() == single[name].to_numpy().tobytes()
    pd.testing.assert_index_equal(parallel.index, single.index, exact=True)


@pytest.mark.parametrize("anticorrelated", (False, True))
def test_selected_csd_retains_legacy_complex_and_real_rules(anticorrelated):
    case = CASES["csd"]
    data = case.data
    if anticorrelated:
        values = np.random.default_rng(414).standard_normal(513)
        data = pd.DataFrame({"a": values, "b": -values})
    full = lpsd_fast.lcsd(data, workers=1, **DEFAULTS)
    actual = lpsd_fast.lcsd(data, outputs=("asd", "psd", "psd_std"),
                            workers=1, **DEFAULTS)
    assert_selected(actual, full, ("asd", "psd", "psd_std"))
    # In particular, do not sqrt a real negative CSD before complex64 casting.
    assert np.isfinite(actual.asd).all()


@pytest.mark.parametrize("outputs", ("nsd", ("psd", "nsd")))
def test_nsd_is_rejected_for_csd(outputs):
    with pytest.raises(ValueError):
        lpsd_fast.lcsd(CASES["csd"].data, outputs=outputs, workers=1, **DEFAULTS)


def test_lnsd_is_direct_nsd_entry_point():
    values = CASES["dc_nanovolt_order0"].data.to_numpy()
    actual = lpsd_fast.lnsd(values, workers=2, **DEFAULTS)
    full = lpsd_fast.lpsd(values, workers=2, **DEFAULTS)
    assert_selected(actual, full, ("nsd",))
    assert "lnsd" in lpsd_fast.__all__


@pytest.mark.parametrize("outputs", (None, "psd", "nsd"))
def test_lnsd_rejects_conflicting_output_argument(outputs):
    with pytest.raises(TypeError):
        lpsd_fast.lnsd(CASES["white_odd"].data, outputs=outputs,
                       workers=1, **DEFAULTS)


def test_lnsd_rejects_csd():
    with pytest.raises(ValueError):
        lpsd_fast.lnsd(CASES["csd"].data, csd=True, workers=1, **DEFAULTS)


@pytest.mark.parametrize("outputs", ("", "unknown", (), [], ["all"],
                                     ["psd", "psd"], ["psd", "unknown"]))
def test_invalid_output_selections_are_rejected(outputs):
    with pytest.raises(ValueError):
        selected("white_odd", outputs)


@pytest.mark.parametrize("outputs", (7, {"psd"}, {"psd": True}, [None]))
def test_output_selection_requires_ordered_names(outputs):
    with pytest.raises((TypeError, ValueError)):
        selected("white_odd", outputs)


@pytest.mark.parametrize("outputs", ("psd", "nsd"))
def test_density_only_does_not_compute_integrated_rms(monkeypatch, outputs):
    def forbidden(*args, **kwargs):
        raise AssertionError("An unrequested cumulative RMS was evaluated")

    monkeypatch.setattr(api, "_asdrms", forbidden)
    assert tuple(selected("white_odd", outputs).columns) == (outputs,)


def test_enbw_only_does_not_evaluate_a_dft(monkeypatch):
    full = run("white_odd", "auto")
    api._native()

    def forbidden(*args, **kwargs):
        raise AssertionError("ENBW depends on the window, not on a signal DFT")

    monkeypatch.setattr(api, "_coefficients", forbidden)
    for name in ("fast_dft", "fast_dft_profile", "fast_dft_selected", "fast_dft_selected_profile"):
        monkeypatch.setattr(api._LIB, name, forbidden, raising=False)
    assert_selected(selected("white_odd", "enbw"), full, ("enbw",))
