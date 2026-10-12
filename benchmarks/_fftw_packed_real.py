# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded real-input FFT using an N/2-point complex FFT for even N.

The original N samples, frequency grid, window and density normalization
are preserved. Adjacent real samples are viewed as complex pairs without a
packing copy. The shared DiskFFT core transforms N/2 complex values; this
adapter reconstructs the original positive-frequency powers in bounded
tiles. Odd N and N=2 retain the full-C2C route.

The file retains its full 16N-byte size for original-N input/window staging.
Reconstruction writes a disjoint power region at byte offset 8N and adds no
anonymous sample-sized array. Caller-owned input and returned power mappings
keep their existing lifetime. Floating-point rounding can differ from the
full-C2C route, especially at tiny spectral nulls. This is an explicit
algorithm choice; it does not change the native LPSD estimator.

``matrix`` exposes the complete staging workspace; it is not the original-N
DFT. ``packed_matrix`` and execute() return the permuted inner-M transform.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import numpy as np

from benchmarks import _fftw_disk as disk

MAX_RECONSTRUCTION_ELEMENTS = 65536


def _checked_n(n):
    n = disk._integer(n, "n")
    if not 2 <= n <= min(2**53, np.iinfo(np.intp).max // 16):
        raise ValueError("Require 2 <= n <= 2**53 and a platform-addressable workspace")
    return n


def _reconstruction_limit(value):
    value = disk._integer(value, "reconstruction_elements")
    if not 1 <= value <= MAX_RECONSTRUCTION_ELEMENTS:
        raise ValueError("reconstruction_elements must lie in [1, 65536]")
    return value


class _ComplexInputDiskFFT(disk.DiskFFT):
    """Share all FFT plans and execution stages, with original-N staging."""

    def __init__(self, backend, real_n, workspace_path, memory_mb=256, factors=None,
                 **kwargs):
        self.real_n = real_n
        super().__init__(backend, real_n // 2, workspace_path, memory_mb,
                         factors, **kwargs)

    def _staging_length(self):
        return self.real_n

    def _expand_real_input(self):
        # N real doubles already represent N/2 interleaved complex values.
        return 0.0


class PackedRealDiskFFT:
    """Original-N window staging with bounded even-N real reconstruction.

    ``packed_factors`` describes inner M only. The legacy ``factors`` argument
    is accepted for the unchanged odd/N2 fallback, and rejected for packed N
    so its original-N meaning cannot silently change. Returned workspace
    power maps are independently caller-owned. close() never unlinks files.
    """

    @staticmethod
    def memory_estimate(n, memory_mb=256, factors=None, *, column_strategy="auto",
                        resident_memory_mb=None, packed_factors=None,
                        reconstruction_elements=MAX_RECONSTRUCTION_ELEMENTS):
        n = _checked_n(n)
        limit = _reconstruction_limit(reconstruction_elements)
        if n % 2 or n == 2:
            if packed_factors is not None:
                raise ValueError("packed_factors do not apply to the full-C2C fallback")
            result = dict(disk.DiskFFT.memory_estimate(
                n, memory_mb, factors, column_strategy=column_strategy,
                resident_memory_mb=resident_memory_mb))
            result.update(route="full_c2c_fallback", original_n=n, inner_n=n)
            return result
        if factors is not None:
            raise ValueError("Use packed_factors for explicit inner-M factors")
        inner = disk.DiskFFT.memory_estimate(
            n // 2, memory_mb, packed_factors, column_strategy=column_strategy,
            resident_memory_mb=resident_memory_mb)
        result = dict(inner)
        capacity = min(limit, inner["ram_tile_bytes"] // 32,
                       inner["twiddle_scratch_bytes"] // 16)
        if capacity < 1:
            raise MemoryError("Existing FFT work/twiddle buffers cannot hold one partner tile")
        result.update(
            n=n, original_n=n, inner_n=n // 2, route="packed_even",
            workspace_bytes=16 * n, workspace_power_supported=True,
            workspace_power_offset_bytes=8 * n, workspace_power_bytes=4 * n + 8,
            workspace_power_extra_file_bytes=0,
            workspace_power_extra_virtual_mapping_bytes=4 * n + 8,
            packed_complex_region_bytes=8 * n, reconstruction_elements=capacity,
            reconstruction_additional_array_bytes=0,
            reconstruction_scratch="Two disjoint complex views of the FFTW tile, plus the existing twiddle buffer; vectors use the existing k2 array.",
            maximum_large_file_bytes_with_input=24 * n,
            inner_layout=dict(inner),
        )
        return result

    def __init__(self, backend, n, workspace_path, memory_mb=256, factors=None,
                 *, io_advice=True, progress=None, column_strategy="auto",
                 resident_memory_mb=None, packed_factors=None,
                 reconstruction_elements=MAX_RECONSTRUCTION_ELEMENTS):
        self.n = _checked_n(n)
        self.memory = self.memory_estimate(
            n, memory_mb, factors, column_strategy=column_strategy,
            resident_memory_mb=resident_memory_mb, packed_factors=packed_factors,
            reconstruction_elements=reconstruction_elements)
        self.route = self.memory["route"]
        self.inner_n = self.memory["inner_n"]
        self.fallback_reason = "odd_n" if n % 2 else "packed_length_below_two" if n == 2 else None
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.backend = backend
        self._closed = False
        self._has_transform = False
        self._power_storage = None
        self._core = None
        self.matrix = self.packed_matrix = self.x = self.window = self.y = None
        self.plan = None
        self._reconstruction = {}
        try:
            if self.route == "full_c2c_fallback":
                core = disk.DiskFFT(backend, self.n, workspace_path, memory_mb, factors,
                                      io_advice=io_advice, progress=progress,
                                      column_strategy=column_strategy,
                                      resident_memory_mb=resident_memory_mb)
            else:
                core = _ComplexInputDiskFFT(backend, self.n, workspace_path, memory_mb,
                                           packed_factors, io_advice=io_advice, progress=progress,
                                           column_strategy=column_strategy,
                                           resident_memory_mb=resident_memory_mb)
            self._core = core
            self.a, self.b = core.a, core.b
            self.factors = core.factors
            self.column_strategy = core.column_strategy
            self.x, self.window = core.x, core.window
            self.matrix = core._mapping
            self.packed_matrix = core.matrix if self.route == "packed_even" else None
            self.y = core.matrix
            self.plan = core.plan
            if self.workspace_path.stat().st_size != 16 * self.n:
                raise AssertionError("The initial full 16N-byte workspace was not retained")
        except BaseException:
            self.close()
            raise

    @property
    def execution_count(self):
        return 0 if self._core is None else self._core.execution_count

    @property
    def last_profile(self):
        return {} if self._core is None else self._core.last_profile

    @property
    def last_phases(self):
        return self.last_profile

    def _require_open(self):
        if self._closed:
            raise RuntimeError("PackedRealDiskFFT is closed")

    def execute(self):
        self._require_open()
        self._has_transform = False
        self._power_storage = None
        result = self._core.execute()
        self._has_transform = True
        self._reconstruction = {}
        return result

    def _validate_power(self, powers, sample_rate, window_square_sum):
        self._require_open()
        if not self._has_transform:
            raise RuntimeError("Execute the transform before extracting power")
        if (not isinstance(powers, np.ndarray) or powers.dtype != np.dtype("float64")
                or powers.ndim != 1 or len(powers) != self.n // 2 + 1
                or not powers.flags.c_contiguous or not powers.flags.writeable):
            raise ValueError("powers must be writable contiguous float64 with N//2+1 values")
        fs, s2 = float(sample_rate), float(window_square_sum)
        if not math.isfinite(fs) or fs <= 0 or not math.isfinite(s2) or s2 <= 0:
            raise ValueError("Positive finite sample_rate and window_square_sum are required")
        return 2.0 / (fs * s2)

    def _reject_external_alias(self, powers):
        if np.shares_memory(powers, self.matrix):
            raise ValueError("External powers must not overlap the transform workspace")
        if (np.shares_memory(powers, self._core._work)
                or np.shares_memory(powers, self._core._twiddle)):
            raise ValueError("External powers must not overlap reconstruction scratch")
        if disk._maps_same_file(powers, self.workspace_path):
            raise ValueError("External powers must use a different file/inode")

    def _tile(self, output, first_row, row_count, first_column, column_count, scale):
        """Read both source rectangles before modifying any arithmetic view."""
        core = self._core
        capacity = self.memory["reconstruction_elements"]
        count = row_count * column_count
        if not 1 <= count <= capacity:
            raise AssertionError("Partner tile exceeds the admitted reused scratch")
        left = core._work[:count].reshape(row_count, column_count)
        right = core._work[capacity:capacity + count].reshape(row_count, column_count)
        difference = core._twiddle.reshape(-1)[:count].reshape(row_count, column_count)
        matrix = core.matrix
        np.copyto(left, matrix[first_row:first_row + row_count,
                               first_column:first_column + column_count])
        if first_row == 0:
            if row_count != 1 or first_column < 1:
                raise AssertionError("DC is handled separately; row-zero partner wraps at DC")
            partner = matrix[0:1, self.b - first_column - column_count + 1:
                                     self.b - first_column + 1]
            np.copyto(right, partner[:, ::-1])
        else:
            partner = matrix[self.a - first_row - row_count + 1:self.a - first_row + 1,
                             self.b - first_column - column_count:self.b - first_column]
            np.copyto(right, partner[::-1, ::-1])
        np.conjugate(right, out=right)
        np.subtract(left, right, out=difference)
        np.add(left, right, out=left)
        # Reuse the now-dead partner buffer for W_N^k. Integers <= M are
        # represented exactly; no M-based frequency or power normalization.
        for column in range(column_count):
            np.add(core._k2[first_row:first_row + row_count],
                   float(self.a * (first_column + column)), out=right.imag[:, column])
        right.imag *= -2.0 * np.pi / self.n
        right.real.fill(0.0)
        np.exp(right, out=right)
        np.multiply(difference, right, out=difference)
        difference *= -1j
        np.add(left, difference, out=left)
        left *= 0.5
        # W is dead; its two real halves hold powers and imag-square scratch.
        doubles = right.reshape(-1).view(np.float64)
        power = doubles[:count].reshape(row_count, column_count)
        scratch = doubles[count:2 * count].reshape(row_count, column_count)
        np.square(left.real, out=power)
        np.square(left.imag, out=scratch)
        np.add(power, scratch, out=power)
        np.multiply(power, scale, out=power)
        np.copyto(output[first_column:first_column + column_count,
                         first_row:first_row + row_count], power.T)

    def _reconstruct_into(self, powers, scale):
        core = self._core
        started = time.perf_counter()
        m = self.inner_n
        capacity = self.memory["reconstruction_elements"]
        rows = min(self.a, max(1, math.isqrt(capacity)))
        columns = min(self.b, max(1, capacity // rows))
        output = powers[:-1].reshape(self.b, self.a)
        # Z stays entirely in [0,8N); workspace powers are in [8N,12N+8).
        # External outputs were separately alias-checked before this call.
        z0 = complex(core.matrix[0, 0])
        dc, nyquist = z0.real + z0.imag, z0.real - z0.imag
        powers[0] = np.multiply(np.square(dc), scale) * 0.5
        powers[-1] = np.multiply(np.square(nyquist), scale) * 0.5
        tiles = 0
        total_tiles = ((self.b - 1 + columns - 1) // columns +
                       ((self.a - 1 + rows - 1) // rows) *
                       ((self.b + columns - 1) // columns))
        core._advise("random")
        core._notify_progress("packed_power", 0, total_tiles)
        for first in range(1, self.b, columns):
            self._tile(output, 0, 1, first, min(columns, self.b - first), scale)
            tiles += 1
            core._notify_progress("packed_power", tiles, total_tiles)
        for first_row in range(1, self.a, rows):
            count = min(rows, self.a - first_row)
            for first in range(0, self.b, columns):
                self._tile(output, first_row, count, first, min(columns, self.b - first), scale)
                tiles += 1
                core._notify_progress("packed_power", tiles, total_tiles)
        if m % 2 == 0:
            # Exact W=-i identity: X[M/2]=conj(Z[M/2]); no residual cos(pi/2).
            k = m // 2
            value = complex(core.matrix[k % self.a, k // self.a])
            powers[k] = np.multiply(np.add(np.square(value.real), np.square(value.imag)), scale)
        core._advise("normal")
        self._reconstruction = {
            "tiles": tiles, "maximum_tile_rows": rows, "maximum_tile_columns": columns,
            "capacity_complex_values_per_buffer": capacity,
            "partner_reads": "Both rectangles copied before arithmetic; partner pairs are visited twice in bounded tiles.",
            "source_region_bytes": [0, 8 * self.n],
            "workspace_power_region_bytes": [8 * self.n, 12 * self.n + 8],
            "self_partner_handled_without_trigonometric_residue": m % 2 == 0,
        }
        core.last_profile.update(power_total_s=time.perf_counter() - started)
        return powers

    def power_into(self, powers, sample_rate, window_square_sum):
        self._require_open()
        if self.route == "full_c2c_fallback":
            result = self._core.power_into(powers, sample_rate, window_square_sum)
        else:
            scale = self._validate_power(powers, sample_rate, window_square_sum)
            self._reject_external_alias(powers)
            result = self._reconstruct_into(powers, scale)
        self._power_storage = "external"
        return result

    def power_in_workspace(self, sample_rate, window_square_sum):
        self._require_open()
        if self.route == "full_c2c_fallback":
            result = self._core.power_in_workspace(sample_rate, window_square_sum)
            self._has_transform = False
            self._power_storage = "workspace"
            return result
        if not self._has_transform:
            raise RuntimeError("Execute the transform before extracting power")
        powers = np.memmap(self.workspace_path, mode="r+", dtype=np.float64,
                           offset=8 * self.n, shape=(self.n // 2 + 1,))
        try:
            scale = self._validate_power(powers, sample_rate, window_square_sum)
            self._has_transform = False
            self._core._has_transform = False
            self._reconstruct_into(powers, scale)
        except BaseException:
            powers._mmap.close()
            raise
        self._power_storage = "workspace"
        return powers

    def power_in_frequency_order_into(self, powers, sample_rate, window_square_sum):
        return self.power_into(powers, sample_rate, window_square_sum)

    def describe(self):
        core = {} if self._core is None else self._core.describe()
        return {
            "algorithm": "exact N-point packed-real reconstruction over bounded N/2 C2C" if self.route == "packed_even" else "full-C2C fallback",
            "n": self.n, "original_n": self.n, "inner_n": self.inner_n,
            "route": self.route, "fallback_reason": self.fallback_reason,
            "factors": list(getattr(self, "factors", ())),
            "column_strategy": getattr(self, "column_strategy", None),
            "workspace_bytes": 16 * self.n, "maximum_large_file_bytes_with_input": 24 * self.n,
            "memory": dict(self.memory), "inner_core": core,
            "matrix_contract": "Full staging workspace; packed_matrix/execute return inner-M Z[k2,k1], k=k2+A*k1. matrix is not the N-point DFT.",
            "normalization": "Original-N unnormalized DFT via real packing; PSD=2*abs(X)^2/(fs*S2), half DC and even Nyquist; no M/N rescaling.",
            "power_storage": self._power_storage, "reconstruction": dict(self._reconstruction),
            "execution_count": self.execution_count, "last_phases": dict(self.last_profile),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "disk_core_source_sha256": hashlib.sha256(Path(disk.__file__).read_bytes()).hexdigest(),
            "input_refill_required_before_each_execute": True,
            "close_unlinks_workspace": False, "caller_owns_returned_power_mapping": True,
        }

    @property
    def metadata(self):
        return self.describe()

    def close(self):
        if self._closed:
            return
        self._has_transform = False
        self.matrix = self.packed_matrix = self.x = self.window = self.y = None
        self.plan = None
        if self._core is not None:
            self._core.close()
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()
