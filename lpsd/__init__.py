import logging as log
import sys

from .wrapper import lpsd, lcsd, lpsd_trad

if sys.version_info >= (3, 8):
    from importlib import metadata  # >= Python 3.8
else:
    import importlib_metadata as metadata  # <= Python 3.7 pylint: disable=import-error

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "unknown"
    log.warning("Version not known, importlib.metadata is not working correctly.")
