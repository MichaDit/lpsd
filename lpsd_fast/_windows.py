"""Original periodic flat-top coefficients, with their signed harmonics.

Copied from lpsd 1.0.6 flattop.py; see that module for window references.
Only exact function identities use the native path. Arbitrary callbacks
retain their own definitions. GPL-3.0-or-later.
"""
from functools import lru_cache


_COSINE_COEFFICIENTS = (
    ("SFT3F", (0.26526, -0.5, 0.23474)),
    ("SFT4F", (0.21706, -0.42103, 0.28294, -0.07897)),
    ("SFT5F", (0.1881, -0.36923, 0.28702, -0.13077, 0.02488)),
    ("SFT3M", (0.28235, -0.52105, 0.19659)),
    ("SFT4M", (0.241906, -0.460841, 0.255381, -0.041872)),
    ("SFT5M", (0.209671, -0.407331, 0.281225, -0.092669, 0.0091036)),
    ("FTNI", (0.2810639, -0.5208972, 0.1980399)),
    ("FTHP", (1.0, -1.912510941, 1.079173272, -0.1832630879)),
    ("FTSR", (1.0, -1.93, 1.29, -0.388, 0.028)),
    ("Matlab", (0.21557895, -0.41663158, 0.277263158, -0.083578947, 0.006947368)),
    ("HFT70", (1.0, -1.90796, 1.07349, -0.18199)),
    ("HFT95", (1.0, -1.9383379, 1.3045202, -0.402827, 0.0350665)),
    ("HFT90D", (1.0, -1.942604, 1.340318, -0.440811, 0.043097)),
    ("HFT116D", (1.0, -1.9575375, 1.4780705, -0.6367431, 0.1228389, -0.0066288)),
    ("HFT144D", (1.0, -1.96760033, 1.57983607, -0.81123644, 0.22583558, -0.02773848, 0.0009036)),
    ("HFT169D", (1.0, -1.97441842, 1.65409888, -0.95788186, 0.3367342, -0.06364621, 0.00521942, -0.00010599)),
    ("HFT196D", (1.0, -1.97928042, 1.710288951, -1.081629853, 0.448734314, -0.112376628, 0.015122992, -0.000871252, 1.1896e-05)),
    ("HFT223D", (1.0, -1.98298997309, 1.75556083063, -1.19037717712, 0.56155440797, -0.17296769663, 0.03233247087, -0.00324954578, 0.0001380104, -1.32725e-06)),
    ("HFT248D", (1.0, -1.985844164102, 1.791176438506, -1.282075284005, 0.667777530266, -0.240160796576, 0.056656381764, -0.008134974479, 0.00062454465, -1.9808998e-05, 1.32974e-07)),
)

COSINE_NAMES = tuple(name.lower() for name, _ in _COSINE_COEFFICIENTS)


@lru_cache(maxsize=1)
def _cosine_windows():
    # Common NumPy/Kaiser calls do not need the reference package (or SciPy).
    # Flat-top callbacks remain the actual original function objects.
    from lpsd import flattop
    return tuple((getattr(flattop, name), coefficients)
                 for name, coefficients in _COSINE_COEFFICIENTS)


def resolve_cosine_window(name):
    index = COSINE_NAMES.index(name)
    return _cosine_windows()[index][0]


def get_cosine_coefficients(window):
    return next((coefficients for original, coefficients in _cosine_windows()
                 if window is original), None)


def __getattr__(name):
    # Preserve the existing private registries for tests and diagnostic tools.
    if name == "COSINE_WINDOWS":
        return _cosine_windows()
    if name == "COSINE_WINDOW_NAMES":
        return {window.__name__.lower(): window for window, _ in _cosine_windows()}
    raise AttributeError(name)
