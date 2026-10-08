import numpy as np
import pandas as pd
import pytest

import lpsd
import lpsd_fast

from ._support import DEFAULTS, PRIMARY, compare


@pytest.mark.parametrize("layout", ("reversed", "strided", "unaligned", "readonly"))
@pytest.mark.parametrize("kernel", ("scalar", "auto"))
def test_array_layout_and_no_mutation(layout, kernel):
    rng = np.random.default_rng(20261008)
    if layout == "reversed":
        data = rng.standard_normal(1025)[::-1]
        assert data.strides == (-8,)
    elif layout == "strided":
        data = rng.standard_normal(2050)[::2]
        assert data.strides == (16,)
    elif layout == "unaligned":
        data = np.ndarray((1025,), dtype=np.float64, buffer=bytearray(8*1025+1), offset=1)
        data[:] = rng.standard_normal(1025)
        assert not data.flags.aligned
    else:
        data = rng.standard_normal(1025)
        data.flags.writeable = False
    original = data.copy()
    flags = (data.flags.writeable, data.strides, data.ctypes.data)
    expected = lpsd.lpsd(pd.Series(original.copy()), **DEFAULTS)
    actual = lpsd_fast.lpsd(data, workers=2, kernel=kernel, **DEFAULTS)
    np.testing.assert_array_equal(data, original)
    assert flags == (data.flags.writeable, data.strides, data.ctypes.data)
    compare(actual, expected, exact=kernel == "scalar", columns=PRIMARY if kernel == "auto" else tuple(expected.columns))


def test_pandas_is_not_mutated():
    data = pd.DataFrame({"a": np.arange(129.), "b": np.arange(129.)**2})
    before = data.copy(deep=True)
    lpsd_fast.lpsd(data, workers=2, **DEFAULTS)
    pd.testing.assert_frame_equal(data, before, check_exact=True)
