"""Run with an isolated installed interpreter, outside the source checkout."""
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

import lpsd
import lpsd_fast
from lpsd._helpers import c_core_available


def main():
    checkout = Path(__file__).resolve().parents[1]
    for module in (lpsd, lpsd_fast):
        assert not Path(module.__file__).resolve().is_relative_to(checkout), (
            f"{module.__name__} was imported from the checkout, not the installed wheel"
        )
    assert c_core_available(), "The wheel is missing the original native library"
    assert version("lpsd") == lpsd_fast.__version__
    data = pd.Series(np.random.default_rng(20261008).standard_normal(257))
    kwargs = dict(sample_rate=50., n_frequencies=24, n_averages=4)
    original = lpsd.lpsd(data, **kwargs)
    scalar = lpsd_fast.lpsd(data, workers=1, kernel="scalar", **kwargs)
    auto = lpsd_fast.lpsd(data.to_numpy(), workers=2, **kwargs)
    pd.testing.assert_frame_equal(scalar, original, check_exact=True)
    for column in ("ps", "psd", "enbw", "asd", "asdrms"):
        np.testing.assert_allclose(auto[column], original[column], rtol=2e-6, atol=0)
    print(f"Installed wheel {version('lpsd')}: both native backends, scalar equality and auto passed")


if __name__ == "__main__":
    main()
