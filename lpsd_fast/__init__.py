"""Parallel native implementation of the LPSD 1.0.6 estimator."""
__version__ = "1.0.6+fast.11"

__all__ = ["lpsd", "lcsd", "lnsd", "available_workers"]


def __getattr__(name):
    # Build/provenance tools only need this package's lightweight metadata.
    # Public imports still return the actual functions, with their signatures.
    if name in __all__:
        from importlib import import_module
        value = getattr(import_module('.api', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(__all__))
