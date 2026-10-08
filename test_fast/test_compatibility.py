import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import lpsd
import lpsd_fast

from ._support import CASES, ORDINARY, PRIMARY, compare, run


def test_reference_sources_match_pinned_upstream():
    manifest = json.loads(Path(__file__).with_name("reference_sources.json").read_text())
    root = Path(lpsd.__file__).resolve().parent
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, (
            f"Reference {relative} differs from upstream {manifest['commit']}"
        )


@pytest.mark.parametrize("name", tuple(CASES))
def test_scalar_preserves_every_output_bit(name):
    compare(run(name, "scalar"), run(name), exact=True)


@pytest.mark.parametrize("name", ORDINARY)
def test_auto_primary_outputs(name):
    compare(run(name, "auto"), run(name), columns=PRIMARY)


@pytest.mark.parametrize("name", ("constant", "zero"))
def test_auto_zero_residual_is_exact_zero(name):
    actual = run(name, "auto")
    for column in ("ps", "psd", "ps_std", "psd_std", "asd", "asdrms"):
        np.testing.assert_array_equal(actual[column], 0)
    np.testing.assert_allclose(actual.enbw, run(name).enbw, rtol=2e-6, atol=0)


@pytest.mark.parametrize("name", ("white_odd", "dc_nanovolt_order0", "ramp_nanovolt_order1", "csd", "offbin_tone"))
@pytest.mark.parametrize("kernel", ("scalar", "auto"))
def test_parallel_is_deterministic(name, kernel):
    compare(run(name, kernel, workers=4), run(name, kernel, workers=1), exact=True)


def test_auto_uses_stable_linear_detrending_for_large_ramp():
    actual = run("ramp_nanovolt_order1", "auto")
    assert actual.attrs["lpsd_fast"]["native_mode"] == 1
    compare(actual, run("ramp_nanovolt_order1", "simd"), exact=True)
    assert run("dc_nanovolt_order0", "auto").attrs["lpsd_fast"]["native_mode"] == 2


def test_lcsd_alias_and_version_metadata():
    case = CASES["csd"]
    kwargs = {key: value for key, value in case.kwargs.items() if key != "csd"}
    actual = lpsd_fast.lcsd(case.data, workers=1, kernel="scalar", **kwargs)
    compare(actual, lpsd.lcsd(case.data, **kwargs), exact=True)
    assert actual.attrs["lpsd_fast"]["version"] == lpsd_fast.__version__
    assert actual.attrs["lpsd_fast"]["legacy_statistics"] is True
