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
    compiler_mantissas = []
    for path in reports:
        report = json.loads(path.read_text())
        assert report["native"] is False, "A redistributable wheel used native-only ISA flags"
        assert "-ffp-contract=off" in report["flags"]
        compiler_mantissas.append(report["floating_point"]["long_double_mant_dig"])
        library = path.parent / report["library"]
        assert hashlib.sha256(library.read_bytes()).hexdigest() == report["binary_sha256"]
        print(json.dumps(report, sort_keys=True))
    # A Windows NumPy wheel can use MSVC FP64 long double while the plain C
    # libraries use MinGW x87. Check the actually loaded compiler ABI.
    from lpsd_fast import api
    api._native()
    assert compiler_mantissas == [api._LIB.native_long_double_mantissa_bits()] * 2
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
    selected = lpsd_fast.lpsd(data, workers=2, kernel="fast", outputs=("psd", "nsd"), **kwargs)
    density = lpsd_fast.lnsd(data, workers=1, kernel="fast", **kwargs)
    np.testing.assert_allclose(selected.psd, original.psd, rtol=.01, atol=0)
    np.testing.assert_array_equal(selected.nsd, np.sqrt(selected.psd))
    np.testing.assert_array_equal(density.nsd, selected.nsd)
    bits = selected.attrs["lpsd_fast"]["native_long_double_mantissa_bits"]
    assert selected.attrs["lpsd_fast"]["native_mode"] == (3 if bits > 64 else 2)
    print(f"Installed wheel {version('lpsd')}: both native backends, scalar equality auto, fast and selected PSD/NSD passed")


if __name__ == "__main__":
    main()
