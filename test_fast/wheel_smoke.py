"""Run with an isolated installed interpreter, outside the source checkout."""
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import platform

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
    reports = (
        Path(lpsd.__file__).parent / "ltpda_dft.build.json",
        Path(lpsd_fast.__file__).parent / "_native" / "liblpsd_fast.build.json",
    )
    for path in reports:
        report = json.loads(path.read_text())
        assert report["native"] is False, "A redistributable wheel used native-only ISA flags"
        assert "-ffp-contract=off" in report["flags"]
        assert report["floating_point"]["long_double_mant_dig"] == np.finfo(np.longdouble).nmant + 1
        library = path.parent / report["library"]
        assert hashlib.sha256(library.read_bytes()).hexdigest() == report["binary_sha256"]
        print(json.dumps(report, sort_keys=True))
    machine = platform.machine().lower()
    wheel_metadata = distribution("lpsd").read_text("WHEEL")
    assert "Tag: py3-none-" in wheel_metadata
    assert "_" + machine in wheel_metadata, (machine, wheel_metadata)
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
