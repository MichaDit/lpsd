import pytest

from lpsd._helpers import c_core_available
from lpsd_fast.api import _native


@pytest.fixture(scope="session", autouse=True)
def require_both_native_libraries():
    """A missing reference C backend must fail, never fall back to Python."""
    assert c_core_available(), "Build the original C backend with: make compile"
    _native()
