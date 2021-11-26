import logging as log
import sys

from .wrapper import lpsd, lpsd_trad

if sys.version_info >= (3, 8):
    import importlib.metadata as metadata  # Python 3.8 pylint: disable=import-error,no-name-in-module
else:
    import importlib_metadata as metadata  # <= Python 3.7 pylint: disable=import-error

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "unknown"
    log.warning("Version not known, importlib.metadata is not working correctly.")
