"""Parallel native implementation of the LPSD 1.0.6 estimator."""
__version__ = "1.0.6+fast.3"

from .api import available_workers, lcsd, lnsd, lpsd

__all__ = ["lpsd", "lcsd", "lnsd", "available_workers"]
