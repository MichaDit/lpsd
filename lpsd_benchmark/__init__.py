"""Portable, resource-aware benchmark controller (run from the source checkout).

SPDX-License-Identifier: GPL-3.0-or-later
"""

SCHEMA = "lpsd-portable-benchmark-v1"
MAX_SECONDS = 24 * 60 * 60
DEFAULT_SIZES = (100_000, 1_000_000, 1_000_003, 10_000_000, 30_000_000,
                 100_000_000, 1_000_000_000)
METHODS = ("fftw", "matched", "lpsd")
SIGNALS = ("white", "pink", "brown", "mixed_tones", "offbin_tone", "dc_nanovolt")
