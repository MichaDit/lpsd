"""Input validation must not allocate an N-byte boolean for a large record."""
import numpy as np
import pytest

from lpsd_fast import api


@pytest.mark.parametrize("bad_index", [None, 0, 262144, 524304])
def test_finite_check_has_bounded_temporary_and_preserves_input(monkeypatch, bad_index):
    values = np.ones(524305, dtype=np.float64)
    if bad_index is not None:
        values[bad_index] = np.nan
    saved = values.copy()
    finite = np.isfinite
    visited = []

    def bounded(array, *args, **kwargs):
        assert array.size <= 262144
        visited.append(array.size)
        return finite(array, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(api.np, "isfinite", bounded)
        if bad_index is None:
            returned = api._array(values)
            assert returned is values
            assert sum(visited) == len(values)
        else:
            with pytest.raises(ValueError, match="finite"):
                api._array(values)
    np.testing.assert_array_equal(values, saved)
