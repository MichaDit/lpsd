# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded full-record windows for a fresh FFTW pipeline.

The owner temporarily uses its already allocated FFT input as the full window
and evaluates its original native sums/fourth moment on that vector. This
helper retains only a bounded block. Preparation then overwrites the FFT input
with anchored samples, takes the unchanged full-vector NumPy mean and applies
regenerated windows through the existing center/window operation.
"""
from __future__ import annotations

import ctypes as ct
import math
import operator

import numpy as np


_DP = ct.POINTER(ct.c_double)
_INT_MAX = 2_147_483_647
_MAX_BLOCK = 1_048_576


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


class BoundedWindow:
    """Own one small float64 window block; never retain caller/FFT arrays.

    Native window sums are required for every kind: the NumPy provider's
    square-sum fallback would otherwise silently allocate another N-vector.
    Kaiser regeneration supports the existing series ABI's range, with block
    starts aligned to 128 samples. Unsupported requests fail before scratch
    allocation and must be constructed before the owner's large FFT buffers.
    """

    def __init__(self, n, beta, provider, kind="kaiser", taper_fraction=.05,
                 block_bins=262144):
        n, block = _integer(n, "n"), _integer(block_bins, "block_bins")
        if not 2 <= n <= _INT_MAX:
            raise ValueError("Bounded windows require 2 <= n <= INT_MAX")
        if kind not in ("kaiser", "tukey", "rectangular"):
            raise ValueError("Unknown bounded window kind")
        if not 1 <= block <= _MAX_BLOCK:
            raise ValueError("block_bins must lie in [1, 1048576]")
        original_taper = taper_fraction
        try:
            beta, taper_fraction = float(beta), float(taper_fraction)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Window parameters must be finite numbers") from error
        if not math.isfinite(beta) or not math.isfinite(taper_fraction) or not 0 <= taper_fraction <= 1:
            raise ValueError("Window parameters must be finite; taper_fraction must lie in [0, 1]")
        half_ramp = n * taper_fraction / 2
        if kind == "tukey" and isinstance(original_taper, np.floating):
            if original_taper.dtype.itemsize > 8:
                raise ValueError("Bounded Tukey requires a taper scalar with at most float64 precision")
            # Legacy Tukey multiplies the original scalar before constructing
            # its float64 ramp. NumPy float16/32 may round that scalar product
            # differently from Python float; retain the same denominator.
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    half_ramp = float(n * original_taper / 2)
            except (TypeError, OverflowError, FloatingPointError) as error:
                raise ValueError("Tukey half_ramp must be finite") from error
        if not math.isfinite(half_ramp):
            raise ValueError("Tukey half_ramp must be finite")
        lib = getattr(provider, "lib", None)
        metadata = getattr(provider, "metadata", {})
        if (lib is None or not isinstance(metadata, dict)
                or metadata.get("backend") != "lpsd_fast_native_series"
                or not callable(getattr(lib, "window_sums", None))
                or not callable(getattr(provider, "sums", None))):
            raise ValueError("Bounded windows require the native Kaiser provider and native window_sums")
        self._generate_full = self._generate_range = None
        if kind == "kaiser":
            if n < 2048 or not 0 < beta <= 32 or block % 128:
                raise ValueError("Bounded Kaiser requires n >= 2048, 0 < beta <= 32 and block_bins divisible by 128")
            full = getattr(lib, "generate_kaiser_series", None)
            ranged = getattr(lib, "generate_kaiser_coefficients_range", None)
            if not callable(full) or not callable(ranged):
                raise ValueError("Native full/range Kaiser generation is unavailable")
            full.argtypes, full.restype = [_DP, ct.c_long, ct.c_double], ct.c_int
            ranged.argtypes = [_DP, _DP, ct.c_long, ct.c_long, ct.c_long,
                              ct.c_double, ct.c_double, ct.c_bool]
            ranged.restype = ct.c_int
            self._generate_full, self._generate_range = full, ranged
        self.n, self.beta, self.kind = n, beta, kind
        self.taper_fraction = taper_fraction
        self.block_samples = min(n, block)
        self._half_ramp = half_ramp
        self._ramp_count = min(n, int(math.ceil(self._half_ramp))) if taper_fraction else 0
        # Low-precision scalar rounding can make the two ramps overlap by a
        # few samples at taper=1. Legacy writes the right mirror last; reserve
        # that overlap exclusively for the right side in both helper passes.
        self._left_stop = min(self._ramp_count, n - self._ramp_count + 1)
        self._closed = False
        # All validation above precedes this sole persistent array allocation.
        self._scratch = np.empty(self.block_samples, dtype=np.float64)
        persistent = self._scratch.nbytes
        # Tukey's int64 indices and reversed float64 block copy are bounded.
        # Add another block for a conservative ufunc/casting allowance; ufunc
        # arithmetic below explicitly writes into the existing float64 block.
        temporary = 3 * persistent if kind == "tukey" else 0
        self.metadata = {
            "kind": kind, "block_samples": self.block_samples,
            "window_buffer_bytes": 0,
            "persistent_scratch_bytes": persistent,
            "temporary_scratch_bytes_upper_bound": temporary,
            "scratch_bytes": persistent + temporary,
            "scratch_scope": "Helper array bound: persistent float64 block plus conservative Tukey index/reversal/ufunc blocks; excludes owner's FFT buffers and native call stack",
            "window_storage": "fft_input_then_blocked",
            "window_sums": "Owner's unchanged full-vector native window_sums",
            "mean_reduction": "numpy.mean of the complete anchored FFT input",
            "window_generation": "native series with integer-mirrored range indices" if kind == "kaiser" else "original NumPy Tukey operation order and copied mirror" if kind == "tukey" else "ones",
            "closed": False,
        }

    def _output(self, out):
        if self._closed:
            raise RuntimeError("BoundedWindow is closed")
        if (not isinstance(out, np.ndarray) or out.dtype != np.dtype("float64")
                or out.ndim != 1 or len(out) != self.n
                or not out.flags.c_contiguous or not out.flags.aligned
                or not out.flags.writeable):
            raise ValueError("Output must be a writable aligned contiguous float64 vector of length n")
        return out

    def _tukey_left(self, first, count):
        """Retain pi*j / half_ramp, cos, *.5, .5- in that exact order."""
        values = self._scratch[:count]
        indices = np.arange(first, first + count, dtype=np.int64)
        np.multiply(np.pi, indices, out=values)
        np.divide(values, self._half_ramp, out=values)
        np.cos(values, out=values)
        np.multiply(.5, values, out=values)
        np.subtract(.5, values, out=values)
        return values

    def fill_into(self, out):
        """Write the original full window into the owner's existing FFT input."""
        out = self._output(out)
        if self.kind == "kaiser":
            if self._generate_full(out.ctypes.data_as(_DP), self.n, self.beta):
                raise RuntimeError("Native bounded-window setup generation failed")
            return out
        out.fill(1.)
        if self.kind == "tukey" and self._ramp_count:
            for first in range(0, self._ramp_count, self.block_samples):
                stop = min(first + self.block_samples, self._ramp_count)
                values = self._tukey_left(first, stop - first)
                left_stop = min(stop, self._left_stop)
                if first < left_stop:
                    out[first:left_stop] = values[:left_stop - first]
                # The periodic endpoint N is omitted, so index zero has no
                # stored mirror. Every other right value copies its left bit.
                begin = max(first, 1)
                if begin < stop:
                    out[self.n - stop + 1:self.n - begin + 1] = values[begin - first:][::-1]
        return out

    def _apply_ones(self, out, first, stop, mean, operations):
        self._scratch.fill(1.)
        for begin in range(first, stop, self.block_samples):
            end = min(begin + self.block_samples, stop)
            operations.center_window(out[begin:end], self._scratch[:end - begin], mean)

    def prepare_into(self, values, out, operations):
        """Anchor/copy, take one full-vector mean, then center/window in blocks.

        Input dtype/shape are checked without converting or copying an N-sized
        vector. Strided float64 caller samples remain supported and unchanged.
        No input, output or operation object is retained after this call.
        """
        out = self._output(out)
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("float64")
                or values.ndim != 1 or len(values) != self.n):
            raise ValueError("Samples must be a float64 vector of length n")
        if np.shares_memory(values, out):
            raise ValueError("Caller samples must not overlap the FFT-owned output")
        if not callable(getattr(operations, "center_window", None)):
            raise ValueError("Operations must provide center_window")
        np.subtract(values, values[0], out=out)
        mean = float(np.mean(out))
        if self.kind == "kaiser":
            for first in range(0, self.n, self.block_samples):
                count = min(self.block_samples, self.n - first)
                window = self._scratch[:count]
                if self._generate_range(window.ctypes.data_as(_DP), None, self.n,
                                        first, count, self.beta, 0., False):
                    raise RuntimeError("Native bounded Kaiser range generation failed")
                operations.center_window(out[first:first + count], window, mean)
        elif self.kind == "rectangular" or self._ramp_count == 0:
            self._apply_ones(out, 0, self.n, mean, operations)
        else:
            # With the full mean already fixed, elementwise application order
            # is independent. Generate each left ramp once and reuse its exact
            # mirrored bits for the right side, including nonintegral ramps.
            self._apply_ones(out, self._left_stop, self.n - self._ramp_count + 1,
                             mean, operations)
            for first in range(0, self._ramp_count, self.block_samples):
                stop = min(first + self.block_samples, self._ramp_count)
                window = self._tukey_left(first, stop - first)
                left_stop = min(stop, self._left_stop)
                if first < left_stop:
                    operations.center_window(out[first:left_stop], window[:left_stop - first], mean)
                begin = max(first, 1)
                if begin < stop:
                    operations.center_window(
                        out[self.n - stop + 1:self.n - begin + 1],
                        window[begin - first:][::-1], mean)
        return mean

    def close(self):
        """Idempotently release the bounded buffer; capacity metadata remains."""
        self._scratch = None
        self._generate_full = self._generate_range = None
        self._closed = True
        self.metadata["closed"] = True
