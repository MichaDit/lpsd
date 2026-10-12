"""Explicit R3 group reduction on an existing original-N complex FFT matrix.

Returns ordinary owned O(J) arrays and independent one-sided energy. The
reviewed row/interval order, 256-bin partials and compensated merges are
preserved. This helper does not allocate a power vector, run an FFT or own
the matrix. Full-pipeline profile admission belongs to bench_fftw_disk.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import ctypes as ct
import math

import numpy as np

MAX_N, MAX_GROUPS, BLOCK_COMPLEX = 1_000_000_000, 65536, 262144


class DirectLibrary:
    """Bind ABI3 from the selected native library; never build a sidecar."""

    def __init__(self, library=None):
        if library is None:
            from lpsd_fast import api
            api._native()
            library = api._LIB
        self.lib = library
        try:
            version = self.lib.dlp_r3_abi_version
            self.reduce = self.lib.dlp_r3_reduce_rows
        except AttributeError as error:
            raise RuntimeError("Direct log-power ABI3 is unavailable in the selected native library") from error
        version.argtypes = []
        version.restype = ct.c_uint64
        self.abi_version = int(version())
        if self.abi_version != 3:
            raise RuntimeError("Direct log-power requires native ABI version 3")
        self.reduce.argtypes = [ct.POINTER(ct.c_double)] + [ct.c_uint64] * 6 + [
            ct.POINTER(ct.c_int64), ct.c_uint64, ct.c_double,
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_uint64)]
        self.reduce.restype = ct.c_int


def integer(value, name, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(name + " outside the explicit supported range")
    return value


def reduce_matrix(matrix, cuts, sample_rate, window_square_sum, library, *, rows_per_block=None):
    """Read aligned Y[r,c], k=r+A*c, into owned sums/counts/cuts and energy."""
    if (not isinstance(matrix, np.ndarray) or matrix.dtype != np.dtype("complex128")
            or matrix.ndim != 2 or not matrix.flags.c_contiguous or not matrix.flags.aligned):
        raise ValueError("Require an existing aligned contiguous complex128 matrix")
    a, b = matrix.shape
    n = integer(a * b, "n", 2, MAX_N)
    if (not isinstance(cuts, np.ndarray) or cuts.dtype != np.dtype("int64")
            or cuts.ndim != 1 or not cuts.flags.c_contiguous or not cuts.flags.aligned):
        raise ValueError("Require the existing aligned contiguous original int64 cuts")
    groups = integer(len(cuts) - 1, "groups", 1, min(MAX_GROUPS, n // 2))
    if cuts[0] != 1 or cuts[-1] != n // 2 + 1 or np.any(np.diff(cuts) <= 0):
        raise ValueError("Original cuts must partition every positive bin exactly")
    fs, s2 = float(sample_rate), float(window_square_sum)
    if not math.isfinite(fs) or fs <= 0 or not math.isfinite(s2) or s2 <= 0:
        raise ValueError("Require original finite positive fs and S2")
    denominator = fs * s2
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Nonrepresentable original bin-power denominator")
    scale = 2.0 / denominator
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Nonrepresentable original bin-power scale")
    rows = (max(1, min(a, 16, BLOCK_COMPLEX // b)) if rows_per_block is None
            else integer(rows_per_block, "rows_per_block", 1, a))
    owned_cuts = np.array(cuts, dtype=np.int64, copy=True)
    owned_cuts.flags.writeable = False
    state = np.zeros(2 * (groups + 1), dtype=np.float64)
    counts = np.zeros(groups + 3, dtype=np.uint64)
    source = matrix.ctypes.data_as(ct.POINTER(ct.c_double))
    cut_pointer = owned_cuts.ctypes.data_as(ct.POINTER(ct.c_int64))
    state_pointer = state.ctypes.data_as(ct.POINTER(ct.c_double))
    count_pointer = counts.ctypes.data_as(ct.POINTER(ct.c_uint64))
    blocks = 0
    for first in range(0, a, rows):
        count = min(rows, a - first)
        status = library.reduce(source, n, a, b, b, first, count,
                                cut_pointer, groups, scale, state_pointer, count_pointer)
        if status:
            raise RuntimeError(f"R3 native group reduction failed with status {status}")
        blocks += 1
    original_counts = np.diff(owned_cuts)
    if (not np.array_equal(counts[:groups], original_counts)
            or int(counts[groups]) != n // 2 + 1 or int(counts[groups + 1]) != 1
            or int(counts[groups + 2]) != int(n % 2 == 0)):
        raise AssertionError("Actual group/energy/endpoint coverage differs from the original bins")
    sums = state[0:2 * groups:2] + state[1:2 * groups:2]
    energy = float(state[-2] + state[-1])
    if not np.isfinite(sums).all() or np.any(sums < 0) or not math.isfinite(energy) or energy < 0:
        raise AssertionError("Invalid completed group or complete-energy sum")
    return {"sums": sums, "energy_power_sum": energy, "cuts": owned_cuts, "counts": original_counts,
            "metadata": {"revision": "R3_new_source", "abi_version": library.abi_version,
                "n": n, "groups": groups,
                "matrix_shape": [a, b], "matrix_stride_complex": b, "rows_per_block": rows,
                "native_blocks": blocks, "bounded_partial_bins": 256,
                "energy_bins": int(counts[groups]), "dc_bins": int(counts[groups + 1]),
                "nyquist_bins": int(counts[groups + 2]), "scale": scale,
                "array_state_bytes": int(owned_cuts.nbytes + state.nbytes + counts.nbytes
                                         + original_counts.nbytes + sums.nbytes),
                "array_state_scope": "O(J) reduction arrays only; Python metadata and bounded C stack are additional",
                "summation": "Sequential256-bin partials within each row/interval and Neumaier merge",
                "energy_definition": "Independent complete nonnegative-bin traversal including original endpoint scaling",
                "additional_workers": 0, "power_vector_created": False,
                "matrix_written": False, "ordinary_result_arrays": True}}
