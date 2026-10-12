# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact N-point FFT through a bounded-RAM, file-backed Cooley--Tukey DFT.

For N=A*B, the time samples have layout X[n2,n1] with n=n1+B*n2.
First transform the A-point columns, multiply exp(-2*pi*i*k2*n1/N),
then transform the B-point rows. The final matrix is Y[k2,k1], whose
original N-point frequency index is k=k2+A*k1. The sample count is unchanged.
The strided path needs no full matrix permutation; the optional transposed
path permutes the existing file in bounded tiles without a second full array.

The sole workspace file holds N complex128 values (16N bytes). Before
execute(), its first N doubles are the caller's windowed real input x;
its second N doubles may temporarily hold the window. Real-to-complex
expansion consumes x and overwrites that window. The input must be filled
again before another execute(). The caller's separate input file is never
read or modified by this adapter.

The RAM budget covers this adapter's explicit anonymous arrays. FFTW's
internal plan/work storage and OS file-cache pages are additional. All
FFTW plans are constructed serially, use ESTIMATE and have length A or B;
execution uses backend.threads without another outer worker pool.
Column gathers/scatters use small row blocks and transpose only RAM-resident
staging data. That staging reuses the existing twiddle buffer. Optional mmap
advice disables speculative readahead during the strided file phases.
For suitable large N=S*S*R, the default instead swaps the equal outer axes
of [S,R,S] in place and executes the first FFT stage contiguously. Both
transpose scratch tiles reuse the existing aligned FFTW allocation; no
additional file or N-sized RAM allocation is introduced. Other lengths
retain the strided path. ``column_strategy`` can select either explicitly.

close() destroys the plans, releases RAM and closes the mapping. It does
not unlink the workspace file. The caller owns file cleanup and durability;
there is no forced fsync/flush in the timed transform or close operation.
Mapped arrays/views must not be used after close(). One object is not
reentrant. Mathematical DFT values are unchanged; floating-point rounding
differs from a single full-record FFTW r2c plan.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import math
import mmap
import operator
import os
from pathlib import Path
import time

import numpy as np

from benchmarks._fftw_bluestein import _plan_description


_MIB = 1024**2
_INT_MAX = np.iinfo(np.int32).max
_ESTIMATE, _FORWARD = 64, -1
_DOUBLE_PTR = ct.POINTER(ct.c_double)
_INT_PTR = ct.POINTER(ct.c_int)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _maps_same_file(array, path):
    """Recognize independent mappings, ndarray views and hard-link aliases."""
    seen = set()
    while array is not None and id(array) not in seen:
        seen.add(id(array))
        filename = getattr(array, "filename", None)
        if filename is not None:
            try:
                if os.path.samefile(filename, path):
                    return True
            except OSError:
                if Path(filename).resolve() == path:
                    return True
        array = getattr(array, "base", None)
    return False


def _factors(n, factors):
    n = _integer(n, "n")
    if n < 2:
        raise ValueError("Require n >= 2")
    # All products k2*n1 used in the twiddle are then exact float64 integers.
    if n > 2**53:
        raise ValueError("This float64 twiddle implementation requires n <= 2**53")
    if n > np.iinfo(np.intp).max // 16:
        raise ValueError("Workspace byte count exceeds the platform index range")
    if factors is None:
        a = math.isqrt(n)
        while n % a:
            a -= 1
        b = n // a
    else:
        try:
            a, b = factors
        except (ValueError, TypeError) as error:
            raise ValueError("factors must be the pair (A, B)") from error
        a, b = _integer(a, "A"), _integer(b, "B")
    if a < 1 or b < 1 or a * b != n:
        raise ValueError("Positive factors must satisfy A*B == n")
    if max(a, b) > _INT_MAX:
        raise ValueError("Each FFTW subtransform length must fit its int interface")
    return n, a, b


def _memory_ceiling():
    """Best-effort physical/cgroup ceiling for a performance choice only."""
    limits = []
    try:
        limits.append(int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE")))
    except (AttributeError, OSError, ValueError):
        pass
    for filename in ("/sys/fs/cgroup/memory.max",
                     "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = int(Path(filename).read_text().strip())
            if 0 < value < 2**60:
                limits.append(value)
        except (OSError, ValueError):
            pass
    if os.name == "nt":
        class MemoryStatus(ct.Structure):
            _fields_ = [("length", ct.c_uint32), ("load", ct.c_uint32)] + [
                (name, ct.c_uint64) for name in (
                    "total_physical", "available_physical", "total_pagefile",
                    "available_pagefile", "total_virtual", "available_virtual", "extended")]
        status = MemoryStatus()
        status.length = ct.sizeof(status)
        try:
            if ct.windll.kernel32.GlobalMemoryStatusEx(ct.byref(status)):
                limits.append(int(status.total_physical))
        except (AttributeError, OSError):
            pass
    if not limits:
        # macOS/BSD expose physical memory through sysctl, without an
        # external command or package dependency.
        try:
            libc = ct.CDLL(None)
            value, size = ct.c_uint64(), ct.c_size_t(8)
            if libc.sysctlbyname(b"hw.memsize", ct.byref(value), ct.byref(size), None, 0) == 0:
                limits.append(int(value.value))
        except (AttributeError, OSError):
            pass
    return min((value for value in limits if value > 0), default=2 * 1024**3)


def _layout(n, memory_mb, factors, column_strategy="auto", resident_memory_mb=None):
    n, a, b = _factors(n, factors)
    if column_strategy not in ("auto", "strided", "transpose"):
        raise ValueError("column_strategy must be 'auto', 'strided' or 'transpose'")
    requested_strategy = column_strategy
    if resident_memory_mb is None:
        resident_limit = _memory_ceiling()
    else:
        if isinstance(resident_memory_mb, (bool, np.bool_)):
            raise TypeError("resident_memory_mb must be a positive finite number")
        resident_memory_mb = float(resident_memory_mb)
        if not math.isfinite(resident_memory_mb) or resident_memory_mb <= 0:
            raise ValueError("resident_memory_mb must be positive and finite")
        resident_limit = int(resident_memory_mb * _MIB)
    prefer_transpose = (column_strategy == "transpose" or
                        column_strategy == "auto" and n >= 10_000_000 and
                        24 * n > resident_limit * 0.8)
    # N=S*S*R can exchange its equal outer axes in-place. This replaces
    # repeated thin stripes through a large file by two bounded tile
    # transposes and contiguous FFT batches; it needs no second large file.
    # Retain the old factor choice for small or explicitly strided calls.
    if factors is None and prefer_transpose:
        square = math.isqrt(n)
        while square > 1 and n % (square * square):
            square -= 1
        if square > 1 and n // (square * square) <= 64:
            a, b = square, n // square
    transposable = a > 1 and b % a == 0
    if column_strategy == "transpose" and not transposable:
        raise ValueError("In-place transpose requires B to be an integer multiple of A > 1")
    if column_strategy == "auto":
        column_strategy = "transpose" if transposable and prefer_transpose else "strided"
    if isinstance(memory_mb, (bool, np.bool_)):
        raise TypeError("memory_mb must be a positive finite number")
    memory_mb = float(memory_mb)
    if not math.isfinite(memory_mb) or memory_mb <= 0:
        raise ValueError("memory_mb must be positive and finite")
    budget = int(memory_mb * _MIB)
    # Keep room for phase generation, expansion/power scratch and FFTW internals.
    # The latter are not promised to fit a particular RSS ceiling.
    tile_cap = min(128 * _MIB, budget // 2)
    tile_elements = tile_cap // 16
    if tile_elements < max(a, b):
        raise MemoryError("RAM tile cannot hold one subtransform; raise memory_mb or choose more balanced factors")
    column_batch = min(b, 512, tile_elements // a)
    row_batch = min(a, 512, tile_elements // b)
    work_elements = max(column_batch * a, row_batch * b)
    transpose_block = min(a, max(1, math.isqrt(work_elements // 2)))
    twiddle_target = min(32 * _MIB, budget // 8)
    twiddle_rows = min(column_batch, max(1, twiddle_target // (16 * a)))
    # Reuse the twiddle storage between arithmetic phases. File copies have
    # contiguous columns; only the bounded staging-to-FFT copy is transposed.
    io_column_block = min(column_batch, twiddle_rows * a)
    io_row_block = min(a, 256, (twiddle_rows * a) // io_column_block)
    real_scratch_elements = min(n, 1 << 20, max(1, min(8 * _MIB, budget // 16) // 8))
    known = (16 * work_elements + 16 * twiddle_rows * a
             + 8 * real_scratch_elements + 8 * a + 16 * twiddle_rows)
    if known > budget:
        raise MemoryError(f"Known RAM arrays require {known} bytes, exceeding the {budget}-byte budget")
    return {
        "n": n, "factors": [a, b], "memory_budget_bytes": budget,
        "column_strategy": column_strategy,
        "requested_column_strategy": requested_strategy,
        "resident_memory_decision_bytes": resident_limit,
        "resident_memory_decision_note": "Explicit budget or detected physical/cgroup ceiling; "
                                          "performance selection only, not enforcement. Auto "
                                          "transpose when 24N exceeds 80% of this ceiling.",
        "transpose_repeat": b // a if transposable else None,
        "transpose_block": transpose_block,
        "transpose_extra_file_bytes": 0,
        "transpose_reuses_fftw_tile": True,
        "workspace_bytes": 16 * n,
        "workspace_power_supported": b % 2 == 0,
        "workspace_power_offset_bytes": 8 * n if b % 2 == 0 else None,
        "workspace_power_bytes": 4 * n + 8 if b % 2 == 0 else None,
        "workspace_power_extra_file_bytes": 0 if b % 2 == 0 else None,
        "workspace_power_extra_virtual_mapping_bytes": 4 * n + 8 if b % 2 == 0 else None,
        "ram_tile_bytes": 16 * work_elements,
        "twiddle_scratch_bytes": 16 * twiddle_rows * a,
        "real_scratch_bytes": 8 * real_scratch_elements,
        "index_vector_bytes": 8 * a + 16 * twiddle_rows,
        "known_ram_buffer_bytes": known,
        "column_batch": column_batch, "row_batch": row_batch,
        "twiddle_rows": twiddle_rows,
        "io_row_block": io_row_block, "io_column_block": io_column_block,
        "io_staging_reuses_twiddle": True,
        "real_scratch_elements": real_scratch_elements,
        "maximum_fftw_transform_length": max(a, b),
        "excludes": "FFTW internal plan/work storage, caller mappings and OS file-cache pages",
    }


class DiskFFT:
    """File-backed exact N-point DFT with small aligned FFTW batch buffers.

    Construct with a new workspace pathname; existing files are not replaced.
    Fill ``x`` with the N real, already detrended/windowed samples. ``window``
    exposes the second half of the same file for the caller's preprocessing.
    ``execute()`` destroys both staging views' contents and returns ``matrix``
    in the explicitly documented [k2,k1] order. ``power_into`` writes the
    original positive-bin order into a separate writable float64 array or
    memmap, applying the standard one-sided density normalization.

    For even B, ``power_in_workspace`` instead consumes the transform and
    stores those powers inside the same 16N-byte file. It returns a separate,
    caller-owned memmap at byte offset 8N; close that mapping independently.
    No additional power file is created. Refill x and execute again before
    either power-extraction method can consume another transform.
    """

    @staticmethod
    def memory_estimate(n, memory_mb=256, factors=None, *, column_strategy="auto",
                        resident_memory_mb=None):
        return _layout(n, memory_mb, factors, column_strategy, resident_memory_mb)

    def __init__(self, backend, n, workspace_path, memory_mb=256, factors=None,
                 *, io_advice=True, progress=None, column_strategy="auto", resident_memory_mb=None):
        started = time.perf_counter()
        layout = _layout(n, memory_mb, factors, column_strategy, resident_memory_mb)
        if not isinstance(io_advice, (bool, np.bool_)):
            raise TypeError("io_advice must be boolean")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        self.backend, self.n = backend, layout["n"]
        staging_n = self._staging_length()
        if staging_n != self.n:
            # The packed-real adapter retains original-N input/window
            # staging while this core transforms N/2 complex samples.
            layout["workspace_bytes"] = 16 * staging_n
        self.a, self.b = layout["factors"]
        self.factors = (self.a, self.b)
        self.column_strategy = layout["column_strategy"]
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.memory_mb, self.memory = float(memory_mb), layout
        self.io_advice = bool(io_advice)
        self._advice_outcomes = {}
        self._progress = progress
        self._last_progress_time = 0.0
        self.io_row_block = layout["io_row_block"]
        self.io_column_block = layout["io_column_block"]
        self._mapping = self.matrix = self.x = self.y = self.window = self._flat = None
        self._work = self._twiddle = self._real_scratch = self._k2 = None
        self._column_offsets = self._column_numbers = None
        self._work_pointer = None
        self._plans = {}
        self._plans_metadata = []
        self.plan = None
        self._closed = False
        self._has_transform = False
        self._power_storage = None
        self.execution_count = 0
        self.allocation_s = self.planning_s = 0.0
        self.last_phases = {}
        self.last_profile = self.last_phases
        self.setup_phases = {}
        self.column_batch, self.row_batch = layout["column_batch"], layout["row_batch"]
        self._source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        lib = self.backend.lib
        self._execute_plan = lib.fftw_execute
        plan_many = lib.fftw_plan_many_dft
        plan_many.argtypes = [ct.c_int, _INT_PTR, ct.c_int,
                              ct.c_void_p, _INT_PTR, ct.c_int, ct.c_int,
                              ct.c_void_p, _INT_PTR, ct.c_int, ct.c_int,
                              ct.c_int, ct.c_uint]
        plan_many.restype = ct.c_void_p
        try:
            tick = time.perf_counter()
            self.workspace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.workspace_path.open("xb") as file:
                file.truncate(layout["workspace_bytes"])
            self._mapping = np.memmap(self.workspace_path, mode="r+", dtype=np.complex128,
                                       shape=(staging_n,))
            self._flat = self._mapping[:self.n]
            self.matrix = self._flat.reshape(self.a, self.b)
            self.y = self.matrix
            staging = self._mapping.view(np.float64)
            self.x, self.window = staging[:staging_n], staging[staging_n:]
            self.setup_phases["workspace_file_and_mapping_s"] = time.perf_counter() - tick

            tick = time.perf_counter()
            self._work_pointer = lib.fftw_malloc(layout["ram_tile_bytes"])
            if not self._work_pointer:
                raise MemoryError("FFTW aligned tile allocation failed")
            doubles = np.ctypeslib.as_array(
                ct.cast(self._work_pointer, _DOUBLE_PTR),
                shape=(layout["ram_tile_bytes"] // 8,))
            self._work = doubles.view(np.complex128)
            self._twiddle = np.empty((layout["twiddle_rows"], self.a), dtype=np.complex128)
            self._real_scratch = np.empty(layout["real_scratch_elements"], dtype=np.float64)
            self._k2 = np.arange(self.a, dtype=np.float64)
            self._column_offsets = np.arange(layout["twiddle_rows"], dtype=np.float64)
            self._column_numbers = np.empty(layout["twiddle_rows"], dtype=np.float64)
            self.allocation_s = time.perf_counter() - tick
            self.setup_phases["anonymous_buffers_s"] = self.allocation_s

            tick = time.perf_counter()
            thread_runtime = (getattr(backend, "threadlib", None)
                              or getattr(type(backend), "_thread_runtime", None))
            if thread_runtime is not None:
                thread_runtime.fftw_plan_with_nthreads(backend.threads)
            lib.fftw_set_timelimit(-1.0)
            for length, count, batch in ((self.a, self.b, self.column_batch),
                                         (self.b, self.a, self.row_batch)):
                if length == 1:
                    continue
                sizes = [batch]
                if count % batch:
                    sizes.append(count % batch)
                for howmany in sizes:
                    key = (length, howmany)
                    if key in self._plans:
                        continue
                    dimensions = (ct.c_int * 1)(length)
                    handle = plan_many(1, dimensions, howmany,
                                       self._work_pointer, None, 1, length,
                                       self._work_pointer, None, 1, length,
                                       _FORWARD, _ESTIMATE)
                    if not handle:
                        raise MemoryError(f"FFTW batch-plan creation failed: length={length}, howmany={howmany}")
                    self._plans[key] = handle
                    if self.plan is None:
                        self.plan = handle
                    self._plans_metadata.append({
                        "length": length, "howmany": howmany, "stride": 1,
                        "distance": length, "in_place": True,
                        **_plan_description(lib, handle),
                    })
            self.planning_s = time.perf_counter() - tick
            self.setup_phases["fftw_batch_planning_s"] = self.planning_s
            self.setup_phases["total_s"] = time.perf_counter() - started
        except BaseException:
            self.close()
            raise

    def _require_open(self):
        if self._closed:
            raise RuntimeError("DiskFFT is closed")

    def _advise(self, mode):
        if not self.io_advice:
            return
        operation = getattr(self._mapping._mmap, "madvise", None)
        advice = getattr(mmap, "MADV_" + mode.upper(), None)
        if operation is None or advice is None:
            self._advice_outcomes[mode] = "unavailable"
            return
        try:
            operation(advice)
        except (OSError, ValueError) as error:
            # Advice is an optional I/O hint, never a correctness requirement.
            self._advice_outcomes[mode] = f"unavailable: {type(error).__name__}: {error}"
        else:
            self._advice_outcomes[mode] = "applied"

    def _notify_progress(self, phase, completed, total):
        if self._progress is None:
            return
        now = time.perf_counter()
        if completed in (0, total) or now - self._last_progress_time >= 15.0:
            self._last_progress_time = now
            self._progress({"phase": phase, "completed": completed, "total": total})

    def _copy_column_tile(self, first_column, tile, *, scatter=False):
        """Copy file rows contiguously; transpose only within bounded RAM."""
        scratch = self._twiddle.reshape(-1)
        for first_row in range(0, self.a, self.io_row_block):
            row_count = min(self.io_row_block, self.a - first_row)
            for column in range(0, len(tile), self.io_column_block):
                column_count = min(self.io_column_block, len(tile) - column)
                staging = scratch[:row_count * column_count].reshape(row_count, column_count)
                mapped = self.matrix[first_row:first_row + row_count,
                                     first_column + column:first_column + column + column_count]
                target = tile[column:column + column_count, first_row:first_row + row_count]
                if scatter:
                    np.copyto(staging, target.T)
                    np.copyto(mapped, staging)
                else:
                    np.copyto(staging, mapped)
                    np.copyto(target, staging.T)

    def _transpose_columns(self, phase):
        """Swap equal outer axes of [A, B/A, A], without a second file.

        Both off-diagonal source rectangles reach disjoint RAM buffers before
        either file rectangle is overwritten. Each file element is copied
        exactly once in each direction. The operation is its own inverse.
        """
        repeat = self.b // self.a
        matrix = self._flat.reshape(self.a, repeat, self.a)
        block = self.memory["transpose_block"]
        scratch_size = block * block
        first_scratch = self._work[:scratch_size]
        second_scratch = self._work[scratch_size:2 * scratch_size]
        blocks = (self.a + block - 1) // block
        total = repeat * blocks * (blocks + 1) // 2
        completed = 0
        self._notify_progress(phase, 0, total)
        for first in range(0, self.a, block):
            rows = min(block, self.a - first)
            for second in range(first, self.a, block):
                columns = min(block, self.a - second)
                left = first_scratch[:rows * columns].reshape(rows, columns)
                right = second_scratch[:rows * columns].reshape(columns, rows)
                for middle in range(repeat):
                    source = matrix[first:first + rows, middle, second:second + columns]
                    np.copyto(left, source)
                    if first == second:
                        np.copyto(source, left.T)
                    else:
                        opposite = matrix[second:second + columns, middle, first:first + rows]
                        np.copyto(right, opposite)
                        np.copyto(source, right.T)
                        np.copyto(opposite, left.T)
                    completed += 1
                    self._notify_progress(phase, completed, total)

    def _multiply_twiddle(self, tile, first_column, *, transposed=False):
        rows = len(self._twiddle)
        scale = -2.0 * np.pi / self.n
        for first in range(0, len(tile), rows):
            count = min(rows, len(tile) - first)
            twiddle = self._twiddle[:count]
            if transposed:
                repeat = self.b // self.a
                # After the outer-axis swap, consecutive rows correspond to
                # original n1 = outer + middle*A, not to consecutive n1.
                for offset in range(count):
                    row = first_column + first + offset
                    self._column_numbers[offset] = row // repeat + (row % repeat) * self.a
            else:
                np.add(self._column_offsets[:count], float(first_column + first),
                       out=self._column_numbers[:count])
            np.multiply(self._column_numbers[:count, None], self._k2[None, :],
                        out=twiddle.imag)
            twiddle.imag *= scale
            twiddle.real.fill(0.0)
            np.exp(twiddle, out=twiddle)
            np.multiply(tile[first:first + count], twiddle, out=tile[first:first + count])

    def _staging_length(self):
        """Real input/window length; overridden only by the packed adapter."""
        return self.n

    def _expand_real_input(self):
        """Expand real staging without overwriting still-needed samples."""
        self._advise("normal")
        self._notify_progress("real_expansion", 0, self.n)
        tick = time.perf_counter()
        stop = self.n
        while stop:
            first = max(0, stop - len(self._real_scratch))
            count = stop - first
            # Complete the source copy before any overlapping destination write.
            np.copyto(self._real_scratch[:count], self.x[first:stop])
            destination = self._flat[first:stop]
            np.copyto(destination.real, self._real_scratch[:count])
            destination.imag.fill(0.0)
            stop = first
            self._notify_progress("real_expansion", self.n - stop, self.n)
        return time.perf_counter() - tick

    def execute(self):
        """Consume windowed real x, returning Y[k2,k1], k=k2+A*k1.

        The caller must refill x before every call, because expansion and the
        subsequent in-place disk stages overwrite both x and window.
        """
        self._require_open()
        self._has_transform = False
        self._power_storage = None
        started = time.perf_counter()
        phases = {name: 0.0 for name in (
            "real_expansion_s", "column_gather_s", "column_fftw_s", "twiddle_s",
            "column_scatter_s", "row_gather_s", "row_fftw_s", "row_scatter_s",
            "column_transpose_before_s", "column_transpose_after_s")}
        phases["real_expansion_s"] = self._expand_real_input()

        if self.a > 1:
            self._advise("random")
            transposed = self.column_strategy == "transpose"
            if transposed:
                tick = time.perf_counter()
                self._transpose_columns("column_transpose_before")
                phases["column_transpose_before_s"] = time.perf_counter() - tick
                self._advise("sequential")
                contiguous_columns = self._flat.reshape(self.b, self.a)
            batches = (self.b + self.column_batch - 1) // self.column_batch
            self._notify_progress("column_fft", 0, batches)
            for first in range(0, self.b, self.column_batch):
                count = min(self.column_batch, self.b - first)
                tile = self._work[:count * self.a].reshape(count, self.a)
                tick = time.perf_counter()
                if transposed:
                    np.copyto(tile, contiguous_columns[first:first + count])
                else:
                    self._copy_column_tile(first, tile)
                phases["column_gather_s"] += time.perf_counter() - tick
                tick = time.perf_counter()
                self._execute_plan(self._plans[(self.a, count)])
                phases["column_fftw_s"] += time.perf_counter() - tick
                if self.b > 1:
                    tick = time.perf_counter()
                    self._multiply_twiddle(tile, first, transposed=transposed)
                    phases["twiddle_s"] += time.perf_counter() - tick
                tick = time.perf_counter()
                if transposed:
                    np.copyto(contiguous_columns[first:first + count], tile)
                else:
                    self._copy_column_tile(first, tile, scatter=True)
                phases["column_scatter_s"] += time.perf_counter() - tick
                self._notify_progress("column_fft", first // self.column_batch + 1, batches)
            if transposed:
                self._advise("random")
                tick = time.perf_counter()
                self._transpose_columns("column_transpose_after")
                phases["column_transpose_after_s"] = time.perf_counter() - tick
        # For A=1 the first FFT and all its twiddles are the identity.
        if self.b > 1:
            self._advise("sequential")
            batches = (self.a + self.row_batch - 1) // self.row_batch
            self._notify_progress("row_fft", 0, batches)
            for first in range(0, self.a, self.row_batch):
                count = min(self.row_batch, self.a - first)
                tile = self._work[:count * self.b].reshape(count, self.b)
                tick = time.perf_counter()
                np.copyto(tile, self.matrix[first:first + count, :])
                phases["row_gather_s"] += time.perf_counter() - tick
                tick = time.perf_counter()
                self._execute_plan(self._plans[(self.b, count)])
                phases["row_fftw_s"] += time.perf_counter() - tick
                tick = time.perf_counter()
                np.copyto(self.matrix[first:first + count, :], tile)
                phases["row_scatter_s"] += time.perf_counter() - tick
                self._notify_progress("row_fft", first // self.row_batch + 1, batches)
        self._advise("normal")
        phases["execute_total_s"] = time.perf_counter() - started
        self.last_phases = phases
        self.last_profile = self.last_phases
        self.execution_count += 1
        self._has_transform = True
        return self.matrix

    def power_into(self, powers, sample_rate, window_square_sum):
        """Write positive original-bin PSD into a separate float64 array/memmap."""
        self._require_open()
        if not self._has_transform:
            raise RuntimeError("Execute the transform before extracting power")
        bins = self.n // 2 + 1
        if (not isinstance(powers, np.ndarray) or powers.dtype != np.dtype("float64")
                or powers.ndim != 1 or len(powers) != bins
                or not powers.flags.c_contiguous or not powers.flags.writeable):
            raise ValueError("powers must be a writable contiguous float64 vector of N//2+1 values")
        if np.shares_memory(powers, self.matrix):
            raise ValueError("powers must not overlap the transform workspace")
        if _maps_same_file(powers, self.workspace_path):
            raise ValueError("powers must use a separate file from the transform workspace")
        if (np.shares_memory(powers, self._work) or
                np.shares_memory(powers, self._twiddle)):
            raise ValueError("powers must not overlap transform scratch")
        sample_rate, window_square_sum = float(sample_rate), float(window_square_sum)
        if (not math.isfinite(sample_rate) or sample_rate <= 0
                or not math.isfinite(window_square_sum) or window_square_sum <= 0):
            raise ValueError("Positive finite sample_rate and window_square_sum are required")
        scale = 2.0 / (sample_rate * window_square_sum)
        started = time.perf_counter()
        gather_s = arithmetic_s = 0.0
        columns = (bins + self.a - 1) // self.a
        self._advise("random")
        batches = (columns + self.column_batch - 1) // self.column_batch
        self._notify_progress("power_reorder", 0, batches)
        for first in range(0, columns, self.column_batch):
            count = min(self.column_batch, columns - first)
            tile = self._work[:count * self.a].reshape(count, self.a)
            tick = time.perf_counter()
            self._copy_column_tile(first, tile)
            gather_s += time.perf_counter() - tick
            flat = tile.reshape(-1)
            frequency_start = first * self.a
            valid = min(len(flat), bins - frequency_start)
            tick = time.perf_counter()
            for offset in range(0, valid, len(self._real_scratch)):
                size = min(len(self._real_scratch), valid - offset)
                values = flat[offset:offset + size]
                target = powers[frequency_start + offset:frequency_start + offset + size]
                scratch = self._real_scratch[:size]
                np.square(values.real, out=target)
                np.square(values.imag, out=scratch)
                np.add(target, scratch, out=target)
                np.multiply(target, scale, out=target)
            arithmetic_s += time.perf_counter() - tick
            self._notify_progress("power_reorder", first // self.column_batch + 1, batches)
        powers[0] *= 0.5
        if self.n % 2 == 0:
            powers[-1] *= 0.5
        self._advise("normal")
        self.last_phases.update(power_gather_s=gather_s, power_arithmetic_s=arithmetic_s,
                                power_total_s=time.perf_counter() - started)
        self._power_storage = "external"
        return powers

    def power_in_workspace(self, sample_rate, window_square_sum):
        """Consume an even-B transform into an independent map of this file.

        The first phase copies every positive-bin source of a row batch into
        the existing RAM tile *before* writing any compact powers. After r
        rows, packed output ends at byte 4*B*r, whereas the next unread FFT
        row starts at byte 16*B*r. These writes cannot overtake unread rows.
        A bounded transpose then reads [0,4N) and writes [8N,12N+8), disjoint
        file regions. Nyquist is saved before the first destructive write.

        The returned float64 memmap has its own mapping of the existing file;
        the caller must close it, even after closing this adapter. Its 4N+8
        virtual bytes alias existing file storage, not an additional file.
        """
        self._require_open()
        if not self._has_transform:
            raise RuntimeError("Execute the transform before extracting power")
        if self.b % 2:
            raise ValueError("Workspace power storage requires an even B factor; use external power storage")
        sample_rate, window_square_sum = float(sample_rate), float(window_square_sum)
        if (not math.isfinite(sample_rate) or sample_rate <= 0
                or not math.isfinite(window_square_sum) or window_square_sum <= 0):
            raise ValueError("Positive finite sample_rate and window_square_sum are required")
        scale = 2.0 / (sample_rate * window_square_sum)
        started = time.perf_counter()
        half = self.b // 2
        bins = self.n // 2 + 1
        nyquist = self.matrix[0, half]  # Scalar copy, before its source is overwritten.
        nyquist_power = np.multiply(np.add(np.square(nyquist.real), np.square(nyquist.imag)), scale)
        doubles = self._mapping.view(np.float64)
        packed = doubles[:self.n // 2].reshape(self.a, half)
        ordered = doubles[self.n:self.n + bins]
        # A failure after the first write must never expose the partly consumed
        # matrix as a valid FFT to a later extraction call.
        self._has_transform = False
        gather_s = arithmetic_s = store_s = 0.0
        self._advise("sequential")
        batches = (self.a + self.row_batch - 1) // self.row_batch
        self._notify_progress("power_workspace_pack", 0, batches)
        pack_started = time.perf_counter()
        for first in range(0, self.a, self.row_batch):
            count = min(self.row_batch, self.a - first)
            tile = self._work[:count * half].reshape(count, half)
            tick = time.perf_counter()
            np.copyto(tile, self.matrix[first:first + count, :half])
            gather_s += time.perf_counter() - tick
            flat = tile.reshape(-1)
            destination = packed[first:first + count].reshape(-1)
            tick = time.perf_counter()
            for offset in range(0, len(flat), len(self._real_scratch)):
                size = min(len(self._real_scratch), len(flat) - offset)
                values = flat[offset:offset + size]
                target = destination[offset:offset + size]
                scratch = self._real_scratch[:size]
                np.square(values.real, out=target)
                np.square(values.imag, out=scratch)
                np.add(target, scratch, out=target)
                np.multiply(target, scale, out=target)
            arithmetic_s += time.perf_counter() - tick
            self._notify_progress("power_workspace_pack", first // self.row_batch + 1, batches)
        pack_s = time.perf_counter() - pack_started

        self._advise("random")
        # The compact source [0,4N) and ordered target [8N,12N) are
        # disjoint. A square tile therefore reads and writes each element
        # once, with useful contiguous spans in *both* file orientations.
        # Thin full-height stripes otherwise amplify page faults at large N.
        work_real = self._work.view(np.float64)
        edge = min(self.a, half, max(1, math.isqrt(len(work_real) // 2)))
        row_blocks = (self.a + edge - 1) // edge
        column_blocks = (half + edge - 1) // edge
        batches = row_blocks * column_blocks
        self._notify_progress("power_workspace_transpose", 0, batches)
        transpose_started = time.perf_counter()
        output_matrix = ordered[:-1].reshape(half, self.a)
        scratch_size = edge * edge
        completed = 0
        for first_row in range(0, self.a, edge):
            row_count = min(edge, self.a - first_row)
            for first in range(0, half, edge):
                count = min(edge, half - first)
                source_tile = work_real[:row_count * count].reshape(row_count, count)
                target_tile = work_real[scratch_size:scratch_size + row_count * count].reshape(count, row_count)
                tick = time.perf_counter()
                np.copyto(source_tile, packed[first_row:first_row + row_count, first:first + count])
                np.copyto(target_tile, source_tile.T)
                gather_s += time.perf_counter() - tick
                tick = time.perf_counter()
                np.copyto(output_matrix[first:first + count, first_row:first_row + row_count], target_tile)
                store_s += time.perf_counter() - tick
                completed += 1
                self._notify_progress("power_workspace_transpose", completed, batches)
        ordered[0] *= 0.5
        ordered[-1] = nyquist_power * 0.5
        transpose_s = time.perf_counter() - transpose_started
        self._advise("normal")
        powers = np.memmap(self.workspace_path, mode="r+", dtype=np.float64,
                           offset=8 * self.n, shape=(bins,))
        self._power_storage = "workspace"
        self.last_phases.update(power_gather_s=gather_s, power_arithmetic_s=arithmetic_s,
                                power_store_s=store_s, power_workspace_pack_s=pack_s,
                                power_workspace_transpose_s=transpose_s,
                                power_workspace_transpose_tile_edge=edge,
                                power_total_s=time.perf_counter() - started)
        return powers

    def power_in_frequency_order_into(self, powers, sample_rate, window_square_sum):
        return self.power_into(powers, sample_rate, window_square_sum)

    @property
    def metadata(self):
        return self.describe()

    def describe(self):
        return {
            "algorithm": "file-backed exact-N two-factor Cooley-Tukey C2C",
            "n": self.n, "factors": list(self.factors),
            "column_strategy": self.column_strategy,
            "matrix_shape": [self.a, self.b],
            "time_layout": "X[n2,n1], n=n1+B*n2",
            "frequency_layout": "Y[k2,k1], k=k2+A*k1",
            "twiddle": "exp(-2*pi*i*k2*n1/N), float64 phase and complex128 values",
            "normalization": "unnormalized N-point forward DFT; power is |Y|^2/(fs*S2), doubled except DC/even Nyquist",
            "workspace_path": str(self.workspace_path), "memory": dict(self.memory),
            "planner": "estimate", "planner_flags": _ESTIMATE,
            "fftw_threads": self.backend.threads,
            "fftw_version": getattr(self.backend, "version", None),
            "plans": [dict(plan) for plan in self._plans_metadata],
            "setup_phases": dict(self.setup_phases),
            "last_phases": dict(self.last_phases), "execution_count": self.execution_count,
            "input_refill_required_before_each_execute": True,
            "window_staging_overwritten_by_execute": True,
            "close_unlinks_workspace": False,
            "power_storage": self._power_storage,
            "workspace_power_layout": "For even B: packed [k2,k1] powers in [0,4N), then "
                                      "original-bin powers in [8N,12N+8). Consumes the FFT; "
                                      "returned map is caller-owned and aliases this file.",
            "io_layout": ("two bounded in-place swaps of equal [A,B/A,A] outer axes; "
                          "contiguous FFT batches; transpose scratch reuses FFTW tile"
                          if self.column_strategy == "transpose" else
                          "bounded contiguous file-row blocks; transpose in reused RAM twiddle scratch"),
            "io_advice": {"enabled": self.io_advice, "outcomes": dict(self._advice_outcomes),
                          "policy": "normal for reverse expansion; random for FFT columns/external "
                                    "power gather/workspace power transpose; sequential for FFT "
                                    "rows/workspace power packing; normal on completion"},
            "progress_callback": self._progress is not None,
            "progress_interval_s": 15.0,
            "flush_policy": "No forced fsync/flush; temporary workspace durability and file-cache eviction are caller-controlled",
            "source_sha256": self._source_sha256,
            "rounding": "same mathematical N-point DFT, different operation order from a full-length FFTW r2c plan",
        }

    def close(self):
        if self._closed:
            return
        self._has_transform = False
        for handle in self._plans.values():
            self.backend.lib.fftw_destroy_plan(handle)
        self._plans.clear()
        self.plan = None
        self._work = self._twiddle = self._real_scratch = self._k2 = None
        self._column_offsets = self._column_numbers = None
        if self._work_pointer:
            self.backend.lib.fftw_free(self._work_pointer)
            self._work_pointer = None
        mapping = self._mapping
        self.matrix = self.x = self.y = self.window = self._flat = self._mapping = None
        if mapping is not None:
            mapping._mmap.close()
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()
