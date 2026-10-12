"""Bounded-workspace execution of the fast Kaiser LPSD estimator.

This is a separate benchmark adapter, not a new storage option in the public
LPSD API. The caller supplies and owns an N-element float64 input. It can
also supply a complex128 scratch memmap, or omit the workspace to allocate
the recommended bounded scratch pool in RAM. Each stored-coefficient point
of length L reserves L complex slots, then uses the region's first L real
doubles for the window and q_r and its next L doubles for q_i. A bounded
frequency thread pool shares that pool; it creates no separate L-sized
heap array per point. Explicit mapped files remain open, unflushed and
owned by the caller.

Large eligible points can regenerate window/Fourier coefficients in bounded
RAM blocks, avoiding full coefficient files and releasing the old length-
based concurrency constraint. The estimator and segment positions remain
unchanged; blockwise dot summation is separately recorded in metadata.
``compute(values, workspace=None)`` accepts a contiguous float64 RAM input and
allocates only the recommended bounded scratch pool for remaining points.

Input validation is blockwise and is included in compute's complete-call
time. File creation, input generation, mapping and durable flushing belong to
the caller; they are not silently included in native stage times. Ordinary
file-backed pages still count against the process's existing memory limits.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import ctypes as ct
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

from lpsd_fast import __version__
from lpsd_fast import api
from lpsd_fast.prepared import _copy_plan, _finite_number, _positive_integer


_ERRORS = {1: "invalid native arguments", 2: "unsafe segment bounds",
           3: "native allocation failed", 4: "kernel/order combination unsupported"}
_STAGES = ("window_generation_s", "window_sums_s", "coefficients_s",
           "c_kernel_s", "c_preparation_s", "c_segments_s", "normalization_s")


class _PointProgress:
    """Serialize callbacks and retain every finished point until a report.

    Callback exceptions propagate after resources have been released. After
    the first callback failure, remaining workers do not call it again. Point
    records contain scalars and are copied before the callback owns them.
    """

    def __init__(self, callback, total, interval_s, started, *, clock=None):
        self._callback = callback
        self._total = int(total)
        self._interval_s = float(interval_s)
        self._started = started
        self._clock = time.perf_counter if clock is None else clock
        self._lock = threading.Lock()
        self._pending = []
        self._completed = 0
        self._last_report = started
        self._failed = False
        self._callback_count = 0
        self._callback_wall_s = 0.0

    def _emit_locked(self, phase):
        if self._failed:
            return
        before = self._clock()
        event = {"phase": phase, "completed": self._completed, "total": self._total,
                 "elapsed_wall_s": before - self._started, "points": self._pending}
        self._pending = []
        self._callback_count += 1
        try:
            self._callback(event)
        except BaseException:
            self._failed = True
            raise
        finally:
            after = self._clock()
            self._callback_wall_s += after - before
            self._last_report = after

    def begin(self):
        with self._lock:
            self._emit_locked("lpsd_points_started")

    def record(self, point):
        with self._lock:
            self._completed += 1
            self._pending.append(dict(point))
            if (self._completed == 1 or
                    self._clock() - self._last_report >= self._interval_s):
                self._emit_locked("lpsd_points")

    def finish(self, status):
        with self._lock:
            self._emit_locked("lpsd_points_" + status)

    def summary(self):
        with self._lock:
            return {"completed_points": self._completed, "total_points": self._total,
                    "interval_s": self._interval_s, "callback_count": self._callback_count,
                    "callback_wall_s": self._callback_wall_s,
                    "callback_failed": self._failed,
                    "scope": "Callbacks run after point scratch/gate release. Their elapsed "
                             "time is included in the complete call, can overlap other "
                             "workers, and is not a correction to subtract from call time."}


class _FairMemoryGate:
    """Local FIFO byte admission; an oversized mapped point runs alone.

    A new short point cannot repeatedly pass an older large reservation.
    This adapter needs that guarantee because its workers immediately take
    another point after releasing resources. The public API's gate is not
    changed. As before, the gate is a concurrency target, not an RSS limit.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("Memory gate capacity must be positive.")
        self.used = 0
        self.peak = 0
        self.condition = threading.Condition()
        self._waiters = deque()

    def acquire(self, amount):
        amount = min(int(amount), self.capacity)
        if amount < 1:
            raise ValueError("Memory gate reservation must be positive.")
        ticket = object()
        with self.condition:
            self._waiters.append(ticket)
            try:
                while self._waiters[0] is not ticket or self.used + amount > self.capacity:
                    self.condition.wait()
            finally:
                self._waiters.remove(ticket)
                # The next FIFO request may already fit alongside this one;
                # it need not wait for an unrelated release to be notified.
                # Notify before committing the reservation: a failed wait
                # or notification has not handed out any resource yet. The
                # condition remains locked until after the state update.
                self.condition.notify_all()
            self.used += amount
            self.peak = max(self.peak, self.used)
            return amount

    def release(self, amount):
        with self.condition:
            self.used -= int(amount)
            self.condition.notify_all()


class _ScratchPool:
    """Allocate disjoint contiguous slots in FIFO order; coalesce releases.

    Preserve arrival order even when a newer, smaller request already fits.
    Otherwise an L=N waiter can lose the whole pool to every subsequent
    short task on the worker that most recently released it.
    """

    def __init__(self, slots):
        self.capacity_slots = int(slots)
        if self.capacity_slots < 1:
            raise ValueError("Scratch pool capacity must be positive.")
        self._condition = threading.Condition()
        self._free = [(0, self.capacity_slots)]
        self._waiters = deque()
        self.used_slots = 0
        self.peak_slots = 0

    def acquire(self, slots):
        slots = int(slots)
        if not 1 <= slots <= self.capacity_slots:
            raise ValueError("Scratch reservation must fit within the pool.")
        ticket = object()
        with self._condition:
            self._waiters.append(ticket)
            choice = None
            try:
                while choice is None:
                    if self._waiters[0] is ticket:
                        for j, (start, available) in enumerate(self._free):
                            if slots <= available:
                                choice = (j, start, available)
                                break
                    if choice is None:
                        self._condition.wait()
            finally:
                self._waiters.remove(ticket)
                # Waiters cannot inspect the pool until this condition is
                # unlocked, so notify before committing the chosen region.
                self._condition.notify_all()
            j, start, available = choice
            if slots == available:
                del self._free[j]
            else:
                self._free[j] = (start + slots, available - slots)
            self.used_slots += slots
            self.peak_slots = max(self.peak_slots, self.used_slots)
            return start

    def release(self, start, slots):
        with self._condition:
            self._free.append((int(start), int(slots)))
            self._free.sort()
            merged = []
            for offset, count in self._free:
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    previous, size = merged[-1]
                    merged[-1] = (previous, size + count)
                else:
                    merged.append((offset, count))
            self._free = merged
            self.used_slots -= int(slots)
            self._condition.notify_all()


def _check_status(status, stage):
    if status:
        raise RuntimeError(f"{stage}: {_ERRORS.get(status, f'native error {status}')}")


def _mapping_region(array, name, dtype, length, writable):
    """Validate storage metadata without copying or examining sample values.

    Compare file regions as well as virtual addresses: two distinct mappings
    of the same file can alias without np.shares_memory detecting it.
    """
    if not isinstance(array, np.memmap):
        raise TypeError(f"{name} must be a caller-owned numpy.memmap.")
    if array.dtype != np.dtype(dtype) or not array.dtype.isnative:
        raise TypeError(f"{name} must have native-endian dtype {np.dtype(dtype).name}.")
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must contain exactly {length} one-dimensional elements.")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError(f"{name} must be contiguous and naturally aligned.")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writable.")
    if getattr(array, "mode", None) == "c":
        raise ValueError(f"{name} must use shared file storage, not copy-on-write mode 'c'.")
    mapping = getattr(array, "_mmap", None)
    if mapping is None or mapping.closed:
        raise ValueError(f"{name} must have a live file mapping.")

    # A sliced/viewed memmap retains its parent's .offset. Walk to the last
    # memmap base and add the actual view displacement to that file offset.
    root = array
    base = array
    while getattr(base, "base", None) is not None:
        base = base.base
        if isinstance(base, np.memmap):
            root = base
    filename = getattr(root, "filename", None)
    if filename is None:
        raise ValueError(f"{name} needs a named mapped file to verify non-overlap.")
    try:
        stat = os.stat(filename)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}'s mapped file must remain available during the call.") from exc
    start = int(root.offset) + int(array.ctypes.data) - int(root.ctypes.data)
    end = start + int(array.nbytes)
    if start < 0 or end > stat.st_size:
        raise ValueError(f"{name}'s mapped region extends beyond its current file.")
    return {"device": int(stat.st_dev), "inode": int(stat.st_ino),
            "offset_bytes": start, "end_bytes": end,
            "mapped_bytes": int(array.nbytes), "mode": str(array.mode)}


def _memory_input_region(values, n):
    if (not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float64)
            or not values.dtype.isnative or values.ndim != 1 or len(values) != n
            or not values.flags.c_contiguous or not values.flags.aligned):
        raise ValueError("RAM input must be an aligned contiguous native float64 vector of exactly n samples.")
    return {"storage": "caller_array", "bytes": int(values.nbytes)}


class DiskLPSDSubset:
    """Evaluate original Kaiser/order-zero PSD points with mapped q scratch.

    ``plan`` is the original five-array tuple ``(f, r, m, lengths, counts)``;
    when omitted, the existing planner receives the standard LPSD parameters.
    ``plan_mask`` is an optional boolean mask applied uniformly, preserving
    the supplied plan's order. At most ``workers`` frequencies are evaluated
    concurrently, without persistent windows or coefficients. Each point
    reserves ``24*L+8192`` nominal working bytes through a local FIFO gate,
    then a disjoint L-complex-slot region through the FIFO scratch pool.
    An explicitly mapped point larger than ``max_working_mb`` runs alone;
    anonymous buffers must first pass the strict budget preflight. The gate
    itself is not a hard RSS limit.

    Unlike the public API's optional single-worker FMA optimization, this
    adapter deliberately calls the ordinary selected kernel used by the
    eight-frequency-worker comparison. Local anchors, native segment starts,
    projection, legacy first-segment recurrence and float32 PSD rounding are
    unchanged. NSD uses the existing complex64 square root after that rounding.

    ``compute(values, workspace, ...)`` validates the entire input in bounded
    blocks and returns an independent DataFrame. No input/workspace reference
    is retained. ``close()`` only closes this adapter; the caller must unmap,
    flush when desired, and remove its own files. Keep the input and mappings
    fixed during a call. Reusing the workspace with another adapter requires
    caller synchronization; calls on this one instance are serialized.

    ``streaming='auto'`` selects bounded Kaiser regeneration only when two
    ordinary point reservations would exceed the working budget. ``'on'``
    selects every supported point; ``'off'`` retains stored coefficients.
    Only L>=2048, 0<beta<=32 and K<=streaming_max_segments are supported by
    the streaming kernel. Other points use the original selected kernel.
    ``recommended_scratch_slots`` permits a smaller file than N slots; no
    file shorter than ``minimum_scratch_slots`` is accepted. Omitting the
    workspace allocates ``recommended_ram_scratch_slots`` anonymously for
    the call. That separate recommendation may exceed N when the working
    budget can accommodate several long points concurrently. No temporary
    files are created. Input, results and OS/runtime memory are extra.

    Optional ``progress(event)`` callbacks are serialized across workers and
    run only after a completed point has released both resource reservations.
    The first point, start and end are reported immediately; other finished
    point profiles are batched at ``progress_interval_s`` (zero reports every
    point). Events contain only scalar metadata, never arrays. Callback time
    belongs to the complete call. A callback must not re-enter compute/close
    or access the shared mappings; its first exception aborts the call after
    worker resources are returned. Progress is disabled when omitted.
    """

    def __init__(self, n, sample_rate, plan=None, *, psll=200.0, overlap=None,
                 plan_mask=None, n_frequencies=1000, n_averages=100,
                 n_min_bins=1, min_segment_length=0, workers=8, max_working_mb=2048,
                 validation_chunk_size=262144, progress=None, progress_interval_s=15.0,
                 streaming="auto", streaming_block_samples=131072,
                 streaming_max_segments=256):
        started = time.perf_counter()
        self._lock = threading.RLock()
        self._closed = False
        self.last_profile = None
        self._last_storage_metadata = None
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None.")
        self._progress = progress
        self._progress_interval_s = _finite_number(progress_interval_s, "progress_interval_s")
        if self._progress_interval_s < 0:
            raise ValueError("progress_interval_s must be nonnegative.")
        self.n = _positive_integer(n, "n", minimum=5)
        self.sample_rate = _finite_number(sample_rate, "sample_rate")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        self.psll = _finite_number(psll, "psll")
        alpha = api._kaiser_alpha(self.psll)
        self.beta = float(alpha * np.pi)
        if not math.isfinite(self.beta):
            raise ValueError("psll produces a nonfinite Kaiser parameter.")
        self.overlap = _finite_number(
            api._kaiser_rov(alpha) if overlap is None else overlap, "overlap")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1).")
        self.workers = _positive_integer(workers, "workers")
        max_working_mb = _finite_number(max_working_mb, "max_working_mb")
        if max_working_mb <= 0:
            raise ValueError("max_working_mb must be positive.")
        self._working_capacity = max(64 * 1024**2, int(max_working_mb * 1024**2))
        self.validation_chunk_size = _positive_integer(
            validation_chunk_size, "validation_chunk_size")
        supplied_plan = plan is not None
        if plan is None:
            plan = api._frequency_plan(self.n, self.sample_rate, self.overlap,
                                       n_min_bins, min_segment_length,
                                       n_frequencies, n_averages)
        original = _copy_plan(plan, self.n, self.sample_rate, self.overlap)
        indices = np.arange(len(original[0]), dtype=np.int64)
        if plan_mask is not None:
            mask = np.asarray(plan_mask)
            if mask.dtype != np.dtype(bool) or mask.ndim != 1 or mask.shape != indices.shape:
                raise ValueError("plan_mask must be a boolean vector matching the original plan.")
            indices = indices[mask]
            self.plan = tuple(part[mask] for part in original)
            for part in self.plan:
                part.setflags(write=False)
        else:
            self.plan = original
        self.frequencies = self.plan[0]
        self.effective_workers = min(self.workers, len(self.frequencies))
        self.enbw = np.full(len(self.frequencies), np.nan, dtype=np.float64)
        self._source_indices = tuple(int(j) for j in indices)
        api._native()
        self._lib = api._LIB
        self._mantissa_bits = int(self._lib.native_long_double_mantissa_bits())
        self._mode = 3 if self._mantissa_bits > 64 else 2
        if streaming not in ("auto", "off", "on"):
            raise ValueError("streaming must be 'auto', 'off' or 'on'.")
        self.streaming_block_samples = _positive_integer(
            streaming_block_samples, "streaming_block_samples", minimum=128)
        if self.streaming_block_samples % 128 or self.streaming_block_samples > 1048576:
            raise ValueError("streaming_block_samples must be a multiple of 128, at most 1048576.")
        self.streaming_max_segments = _positive_integer(streaming_max_segments, "streaming_max_segments")
        if self.streaming_max_segments > 256:
            raise ValueError("streaming_max_segments must be at most 256.")
        self._streaming_fn = getattr(self._lib, "fast_dft_kaiser_streaming", None)
        if streaming == "on" and self._streaming_fn is None:
            raise RuntimeError("Rebuild the native library to enable bounded Kaiser streaming.")
        if self._streaming_fn is not None:
            dp, lp = ct.POINTER(ct.c_double), ct.POINTER(ct.c_long)
            self._streaming_fn.argtypes = [dp, dp, lp, dp, ct.c_long, ct.c_long,
                ct.c_double, ct.c_double, ct.c_double, ct.c_int, ct.c_long, ct.c_long,
                dp, dp, dp, dp]
            self._streaming_fn.restype = ct.c_int
        self._streaming_mask = tuple(
            self._streaming_fn is not None and streaming != "off" and 0 < self.beta <= 32
            and int(length) >= 2048 and int(count) <= self.streaming_max_segments
            and (streaming == "on" or 2 * (24 * int(length) + 8192) > self._working_capacity)
            for length, count in zip(self.plan[3], self.plan[4]))
        lengths = sorted((int(length) for length, streamed in zip(self.plan[3], self._streaming_mask)
                          if not streamed), reverse=True)
        self.minimum_scratch_slots = max(lengths, default=1)
        self.recommended_ram_scratch_slots = max(self.minimum_scratch_slots,
            min(sum(lengths[:self.effective_workers]), self._working_capacity // 24))
        # Existing caller-owned mappings retain the <=N contract. An
        # anonymous pool need not serialize long points to fit that file
        # layout when a larger pool is permitted by the explicit budget.
        self.recommended_scratch_slots = min(self.n, self.recommended_ram_scratch_slots)
        self._streaming_working_bytes = 16 * self.streaming_block_samples + 32768
        self._streaming_workers_bound = min(self.effective_workers, sum(self._streaming_mask),
            max(1, self._working_capacity // self._streaming_working_bytes))
        streaming_bytes = self._streaming_workers_bound * self._streaming_working_bytes
        bounded_scratch_bytes = 16 * self.recommended_ram_scratch_slots
        validation_bytes = min(self.n, self.validation_chunk_size)
        self._ram_working_upper_bound = bounded_scratch_bytes + streaming_bytes + validation_bytes
        self._metadata = {
            "adapter": "benchmarks._lpsd_disk.DiskLPSDSubset",
            "benchmark_category": "file_backed_complete_call",
            "lpsd_fast_version": __version__, "estimator": "lpsd-1.0.6",
            "kernel": "fast", "window": "kaiser", "psll": self.psll,
            "kaiser_beta": self.beta, "overlap": self.overlap,
            "detrending_order": 0, "statistics_computed": False,
            "native_mode": self._mode,
            "native_long_double_mantissa_bits": self._mantissa_bits,
            "bounded_segment_fma_enabled": False,
            "legacy_first_segment_recurrence": True,
            "n": self.n, "sample_rate": self.sample_rate,
            "workers": self.workers, "effective_workers": self.effective_workers,
            "frequencies": len(self.frequencies),
            "supplied_plan": supplied_plan, "original_frequency_count": len(original[0]),
            "source_plan_indices": list(self._source_indices),
            "input_bytes": 8 * self.n, "scratch_bytes": 16 * self.n,
            "total_mapped_bytes": 24 * self.n,
            "maximum_point_scratch_bytes": 16 * max(lengths, default=0),
            "original_maximum_point_coefficient_bytes": max((16 * int(v) for v in self.plan[3]), default=0),
            "concurrency_budget_bytes": self._working_capacity,
            "reservation_policy": "FIFO memory gate, then FIFO contiguous scratch pool; "
                                  "new requests cannot overtake existing waiters.",
            "working_budget_note": "Stored-coefficient points reserve 24*L+8192 bytes; streaming "
                                   "points reserve their bounded block buffers. Actual stored q "
                                   "uses 16*L bytes from the shared mapped or anonymous pool. "
                                   "With explicit file mappings one oversized point may run alone; "
                                   "workspace=None first checks the full RAM buffer bound. "
                                   "The scheduling gate is not an RSS limit.",
            "cached_coefficient_bytes": 0, "cached_window_bytes": 0,
            "streaming": {
                "policy": streaming, "native_available": self._streaming_fn is not None,
                "eligible_points": sum(self._streaming_mask),
                "source_plan_indices": [j for j, enabled in zip(self._source_indices, self._streaming_mask) if enabled],
                "block_samples": self.streaming_block_samples,
                "maximum_segments": self.streaming_max_segments,
                "working_bytes_per_point_upper_bound": self._streaming_working_bytes,
                "auto_rule": "Use bounded regeneration when two ordinary 24L+8192 reservations "
                             "would exceed the concurrency budget, L>=2048, K<=limit, 0<beta<=32.",
                "rounding": "Window/coefficient generation and projector moments retain their arithmetic; "
                            "dot products use block reductions. No decimation, altered grid or segment starts.",
            },
            "minimum_scratch_bytes": 16 * self.minimum_scratch_slots,
            "recommended_scratch_bytes": 16 * self.recommended_scratch_slots,
            "recommended_mapped_scratch_bytes": 16 * self.recommended_scratch_slots,
            "recommended_ram_scratch_bytes": bounded_scratch_bytes,
            "bounded_memory_model": {
                "input_bytes": 8 * self.n,
                "input_copy_bytes": 0,
                "scratch_bytes": bounded_scratch_bytes,
                "maximum_concurrent_streaming_points": self._streaming_workers_bound,
                "streaming_bytes_per_point_upper_bound": self._streaming_working_bytes,
                "streaming_bytes_upper_bound": streaming_bytes,
                "validation_temporary_bytes_upper_bound": validation_bytes,
                "input_scratch_streaming_validation_bytes_upper_bound": (
                    8 * self.n + bounded_scratch_bytes + streaming_bytes + validation_bytes),
                "working_bytes_upper_bound": self._ram_working_upper_bound,
                "working_budget_bytes": self._working_capacity,
                "fits_working_budget": self._ram_working_upper_bound <= self._working_capacity,
                "ram_preflight": "workspace=None raises MemoryError before allocating scratch when "
                                 "the conservative scratch+streaming+validation bound exceeds the "
                                 "working budget. Explicit file mappings retain the documented soft gate.",
                "scratch_formula": "16*max(max_nonstream_L,min(sum_largest_worker_L,capacity//24)) bytes; "
                                   "16 bytes if every point streams or the subset is empty.",
                "mapped_scratch_formula": "min(N,recommended_ram_scratch_slots) complex128 slots; "
                                          "the existing explicit-file recommendation stays capped at N.",
                "streaming_formula": "At most min(workers,eligible_points,capacity//per_point) active "
                                     "streaming points; per_point=16*block_samples+32768 covers the "
                                     "two native block vectors and bounded native scalar/stack state.",
                "excludes": "O(J) plan/result/profile objects, Python/pandas/native runtime, thread "
                            "runtime stacks outside the kernel, allocator overhead and OS file-cache pages. "
                            "This is a buffer model, not a process RSS promise.",
            },
            "progress_enabled": progress is not None,
            "progress_interval_s": self._progress_interval_s,
            "validation_chunk_samples": self.validation_chunk_size,
            "maximum_validation_boolean_bytes": min(self.n, self.validation_chunk_size),
            "scratch_layout": "A point reserves L complex slots at pool offset s: q_r = "
                              "doubles[2*s:2*s+L], q_i = doubles[2*s+L:2*s+2*L]. The window "
                              "aliases q_r before coefficient generation; returned regions coalesce.",
            "ownership": "Caller owns all mapped files; adapter never flushes, unmaps or deletes them.",
            "memory_note": "Mapped file pages remain subject to the existing memory limit. "
                           "The adapter allocates O(J) results, bounded validation temporaries "
                           "and active streaming block buffers. With workspace=None it also "
                           "allocates the shared recommended scratch pool after a budget check. "
                           "There is no separate L-sized heap work array per point.",
            "compute_scope": "Storage checks, blockwise finite validation, every selected window, "
                             "normalization, coefficient generation, native projection and original "
                             "segment evaluation, PSD rounding and requested DataFrame output.",
            "excluded_scope": "Caller file creation, input generation, mapping, unmapping and "
                              "any requested durable file flush; constructor/planning is separate.",
            "normalization": "complex64(2.0 * complex(Pr_r, Pr_i) / sample_rate / S2)",
            "nsd": "complex64 square root of the already rounded PSD",
            "native_library": str(Path(self._lib._name).resolve()),
            "native_source": "lpsd_fast/_native/fast_dft.c",
            "constructor_wall_s": time.perf_counter() - started,
        }

    @property
    def metadata(self):
        """JSON-serializable configuration; this is not a measured RAM bound."""
        result = copy.deepcopy(self._metadata)
        if self._last_storage_metadata is not None:
            result.update(self._last_storage_metadata)
        return result

    def _validate(self, values, workspace, mapped=True):
        if mapped:
            input_region = _mapping_region(values, "values", np.float64, self.n, False)
            if not isinstance(workspace, np.memmap):
                raise TypeError("workspace must be a caller-owned numpy.memmap.")
            if not self.minimum_scratch_slots <= len(workspace) <= self.n:
                raise ValueError(f"workspace must contain between {self.minimum_scratch_slots} and {self.n} complex slots.")
            scratch_region = _mapping_region(workspace, "workspace", np.complex128, len(workspace), True)
            same_file = (input_region["device"], input_region["inode"]) == (
                scratch_region["device"], scratch_region["inode"])
            overlaps_file = (same_file and input_region["offset_bytes"] < scratch_region["end_bytes"]
                             and scratch_region["offset_bytes"] < input_region["end_bytes"])
        else:
            input_region = _memory_input_region(values, self.n)
            scratch_region = {"storage": "bounded_anonymous", "bytes": int(workspace.nbytes)}
            overlaps_file = False
        if overlaps_file or np.shares_memory(values, workspace):
            raise ValueError("Input and scratch mapped regions must not overlap.")
        for start in range(0, self.n, self.validation_chunk_size):
            if not np.isfinite(values[start:min(self.n, start + self.validation_chunk_size)]).all():
                raise ValueError("Input samples must be finite; clean missing values before calling LPSD.")
        return input_region, scratch_region

    def compute(self, values, workspace=None, outputs="psd", profile=True):
        """Return PSD/NSD in plan order, including bounded input validation.

        ``outputs`` is ``'psd'``, ``'nsd'`` or an ordered sequence of these
        names. Every call recalculates its selected spectrum from the samples;
        an NSD-only call does not reuse a previous PSD. With ``profile=True``,
        per-point native preparation/segment timers and stage wall times are
        attached as ``result.attrs['lpsd_disk_profile']`` and ``last_profile``.
        L*K is logical segment coverage, not measured storage traffic.

        With no workspace, ``values`` may be an aligned contiguous native
        float64 ndarray. The call allocates ``16*recommended_ram_scratch_slots``
        anonymous bytes plus bounded streaming buffers under the same gate.
        No input copy, full-length window, or persistent cache is created.
        """
        started = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("DiskLPSDSubset is closed.")
            requested = api._requested_outputs(outputs, False)
            if any(name not in ("psd", "nsd") for name in requested):
                raise ValueError("DiskLPSDSubset supports only 'psd' and 'nsd' outputs.")
            if not isinstance(profile, (bool, np.bool_)):
                raise TypeError("profile must be a boolean.")
            self.last_profile = None
            self._last_storage_metadata = None
            validation_started = time.perf_counter()
            mapped = workspace is not None
            if not mapped:
                _memory_input_region(values, self.n)
                if self._ram_working_upper_bound > self._working_capacity:
                    raise MemoryError(
                        f"Bounded RAM buffers require {self._ram_working_upper_bound} bytes, exceeding "
                        f"the {self._working_capacity}-byte working budget. Raise max_working_mb, "
                        "reduce the dataset, or provide an explicitly file-backed workspace.")
                workspace = np.empty(self.recommended_ram_scratch_slots, dtype=np.complex128)
            input_region, scratch_region = self._validate(values, workspace, mapped=mapped)
            self._last_storage_metadata = {
                "storage_mode": "mapped" if mapped else "bounded_memory",
                "scratch_bytes": int(workspace.nbytes),
                "actual_scratch_bytes": int(workspace.nbytes),
                "anonymous_scratch_bytes": 0 if mapped else int(workspace.nbytes),
                "total_mapped_bytes": ((int(values.nbytes) if isinstance(values, np.memmap) else 0)
                                       + (int(workspace.nbytes) if mapped else 0)),
                "benchmark_category": "file_backed_complete_call" if mapped else "bounded_memory_complete_call",
            }
            validation_s = time.perf_counter() - validation_started
            doubles = workspace.view(np.float64)
            xp = api._pointer(values)
            fn = self._lib.fast_dft_selected_profile if profile else self._lib.fast_dft_selected
            density = np.empty(len(self.frequencies), dtype=np.complex64)
            rows = [None] * len(self.frequencies) if profile else []
            totals = {name: 0.0 for name in _STAGES}
            gate = _FairMemoryGate(self._working_capacity)
            scratch_pool = _ScratchPool(len(workspace))
            streaming_count_lock = threading.Lock()
            streaming_counts = [0, 0]
            point_progress = (_PointProgress(self._progress, len(self.frequencies),
                                             self._progress_interval_s, started)
                              if self._progress is not None else None)
            if point_progress is not None:
                point_progress.begin()

            def one(j):
                length = int(self.plan[3][j])
                streamed = self._streaming_mask[j]
                waiting_started = time.perf_counter()
                reserved = gate.acquire(self._streaming_working_bytes if streamed else 24 * length + 8192)
                gate_acquired = time.perf_counter()
                scratch_start = None
                streaming_active = False
                try:
                    # Always acquire the concurrency gate before the scratch
                    # pool. Holders of scratch never wait on a second gate;
                    # every active allocation is returned even on exceptions.
                    if not streamed:
                        scratch_start = scratch_pool.acquire(length)
                    else:
                        with streaming_count_lock:
                            streaming_counts[0] += 1
                            streaming_counts[1] = max(streaming_counts)
                        streaming_active = True
                    point_started = time.perf_counter()
                    s1_native, s2_native = ct.c_double(), ct.c_double()
                    pr, pi, vr, vi = ct.c_double(), ct.c_double(), ct.c_double(), ct.c_double()
                    navg = ct.c_long()
                    prep_s, segments_s = ct.c_double(), ct.c_double()
                    if streamed:
                        t3 = time.perf_counter()
                        _check_status(self._streaming_fn(
                            ct.byref(pr), ct.byref(pi), ct.byref(navg), xp, self.n, length,
                            self.beta, float(self.plan[2][j]), self.overlap * 100, self._mode,
                            self.streaming_block_samples, self.streaming_max_segments,
                            ct.byref(s1_native), ct.byref(s2_native),
                            ct.byref(prep_s), ct.byref(segments_s)), "Bounded LPSD segment evaluation")
                        t4 = time.perf_counter()
                        if profile:
                            # These components are fused inside the two C
                            # passes. None means unmeasured, never zero work.
                            stages = {"window_generation_s": None, "window_sums_s": None,
                                      "coefficients_s": None, "c_kernel_s": t4 - t3,
                                      "c_preparation_s": prep_s.value,
                                      "c_segments_s": segments_s.value}
                    else:
                        offset = 2 * scratch_start
                        cr = doubles[offset:offset + length]
                        ci = doubles[offset + length:offset + 2 * length]
                        crp, cip = api._pointer(cr), api._pointer(ci)
                        t0 = time.perf_counter()
                        _check_status(self._lib.generate_kaiser_series(crp, length, self.beta),
                                      "Kaiser window generation")
                        t1 = time.perf_counter()
                        _check_status(self._lib.window_sums(crp, length, ct.byref(s1_native),
                                                           ct.byref(s2_native)), "Window normalization")
                        if (not math.isfinite(s1_native.value) or not math.isfinite(s2_native.value)
                                or s1_native.value == 0 or s2_native.value == 0):
                            raise ValueError("The Kaiser window has nonfinite or zero normalization.")
                        t2 = time.perf_counter()
                        # Window aliases q_r only after its sums are saved.
                        _check_status(self._lib.generate_coefficients_blocked(
                            crp, cip, crp, length, float(self.plan[2][j])), "Fourier coefficient generation")
                        t3 = time.perf_counter()
                        extra = (ct.byref(prep_s), ct.byref(segments_s)) if profile else ()
                        # The ordinary mapped path retains the previous
                        # non-FMA selected kernel and its arithmetic.
                        _check_status(fn(ct.byref(pr), ct.byref(pi), ct.byref(vr), ct.byref(vi),
                                         ct.byref(navg), xp, xp, self.n, length, crp, cip,
                                         self.overlap * 100, 0, False, self._mode, False, True,
                                         *extra), "LPSD segment evaluation")
                        t4 = time.perf_counter()
                        if profile:
                            stages = {"window_generation_s": t1 - t0, "window_sums_s": t2 - t1,
                                      "coefficients_s": t3 - t2, "c_kernel_s": t4 - t3,
                                      "c_preparation_s": prep_s.value,
                                      "c_segments_s": segments_s.value}
                    s1, s2 = s1_native.value, s2_native.value
                    if not math.isfinite(s1) or not math.isfinite(s2) or s1 == 0 or s2 == 0:
                        raise ValueError("The Kaiser window has nonfinite or zero normalization.")
                    # Preserve the public API's operation order and its complex64
                    # rounding point before any NSD square root is performed.
                    density[j] = 2.0 * complex(pr.value, pi.value) / self.sample_rate / s2
                    self.enbw[j] = self.sample_rate * s2 / (s1 * s1)
                    point_end = time.perf_counter()
                    work = length * int(navg.value)
                    point_wall = point_end - point_started
                    if profile:
                        stages["normalization_s"] = point_end - t4
                        rows[j] = {"j": j, "source_j": self._source_indices[j],
                                   "frequency": float(self.frequencies[j]), "L": length,
                                   "K": int(navg.value), "planned_K": int(self.plan[4][j]),
                                   "sample_iterations": work, "scratch_bytes": 0 if streamed else 16 * length,
                                   "scratch_offset_bytes": None if streamed else 16 * scratch_start,
                                   "execution_method": "streaming_kaiser_blocks" if streamed else "stored_coefficients",
                                   "anonymous_working_bytes_upper_bound": self._streaming_working_bytes if streamed else 0,
                                   "window_s1": s1, "window_s2": s2,
                                   "enbw_hz": float(self.enbw[j]), "point_wall_s": point_wall,
                                   "memory_gate_wait_s": gate_acquired - waiting_started,
                                   "scratch_pool_wait_s": point_started - gate_acquired,
                                   **stages}
                finally:
                    if streaming_active:
                        with streaming_count_lock:
                            streaming_counts[0] -= 1
                    if scratch_start is not None:
                        scratch_pool.release(scratch_start, length)
                    gate.release(reserved)
                # Logging must not retain either reservation or expose a view
                # into scratch that another worker can now overwrite.
                if point_progress is not None:
                    point = rows[j] if profile else {
                        "j": j, "source_j": self._source_indices[j],
                        "frequency": float(self.frequencies[j]), "L": length,
                        "K": int(navg.value), "planned_K": int(self.plan[4][j]),
                        "sample_iterations": work, "point_wall_s": point_wall,
                        "memory_gate_wait_s": gate_acquired - waiting_started,
                        "scratch_pool_wait_s": point_started - gate_acquired}
                    point_progress.record(point)
                return work, point_wall

            try:
                if self.effective_workers <= 1:
                    completed = list(map(one, range(len(self.frequencies))))
                else:
                    with ThreadPoolExecutor(max_workers=self.effective_workers,
                                            thread_name_prefix="lpsd-disk") as pool:
                        completed = list(pool.map(one, range(len(self.frequencies))))
            except BaseException:
                if point_progress is not None:
                    try:
                        point_progress.finish("failed")
                    except BaseException:
                        pass  # Preserve the original compute/callback error.
                raise
            if point_progress is not None:
                point_progress.finish("completed")
            logical_work = sum(item[0] for item in completed)
            point_wall_sum = sum(item[1] for item in completed)
            if profile:
                totals = {name: sum(row[name] for row in rows if row[name] is not None) for name in _STAGES}
            assembly_started = time.perf_counter()
            columns = {}
            if "psd" in requested:
                columns["psd"] = density
            if "nsd" in requested:
                columns["nsd"] = np.sqrt(density)
            if not np.iscomplex(density).any():
                columns = {name: column.real for name, column in columns.items()}
            result = pd.DataFrame({name: columns[name] for name in requested},
                                  index=self.frequencies.copy(), copy=True)
            result.index.name = "frequency"
            result.attrs["lpsd_disk"] = self.metadata
            result.attrs["lpsd_disk"]["outputs"] = list(requested)
            result.attrs["lpsd_disk"]["input_mapping"] = input_region
            result.attrs["lpsd_disk"]["scratch_mapping"] = scratch_region
            result.attrs["lpsd_disk"]["storage_mode"] = "mapped" if mapped else "bounded_memory"
            result.attrs["lpsd_disk"]["actual_scratch_bytes"] = int(workspace.nbytes)
            assembly_s = time.perf_counter() - assembly_started
            details = {"call_wall_s": time.perf_counter() - started,
                       "validation_wall_s": validation_s, "point_wall_sum_s": point_wall_sum,
                       "output_assembly_s": assembly_s, "sample_iterations": logical_work,
                       "workers": self.workers, "effective_workers": self.effective_workers,
                       "native_mode": self._mode,
                       "concurrency_budget_bytes": self._working_capacity,
                       "peak_reserved_bytes": gate.peak,
                       "peak_active_scratch_bytes": 16 * scratch_pool.peak_slots,
                       "scratch_buffer_bytes": int(workspace.nbytes),
                       "storage_mode": "mapped" if mapped else "bounded_memory",
                       "streaming_points": sum(self._streaming_mask),
                       "peak_active_streaming_points": streaming_counts[1],
                       "observed_concurrency_streaming_bytes_upper_bound": streaming_counts[1] * self._streaming_working_bytes,
                       "anonymous_streaming_upper_bound_bytes": self._streaming_workers_bound * self._streaming_working_bytes,
                       "stage_profiling_enabled": bool(profile),
                       "progress": point_progress.summary() if point_progress is not None else None,
                       "frequencies": rows, "stage_totals_s": totals if profile else None,
                       "sample_iterations_note": "L*K is logical segment coverage/direct reference "
                                                 "work, not bytes read from RAM or disk.",
                       "stage_note": "Per-frequency elapsed times overlap across workers, so "
                                     "point_wall_sum_s and stage totals are not call wall time. Native "
                                     "preparation/segments are subdivisions of c_kernel_s; do not add "
                                     "them to c_kernel_s. File paging incurred inside these calls is "
                                     "included in their elapsed times. Streaming points regenerate "
                                     "window and coefficients inside both C passes; their separately "
                                     "unmeasured window/normalization/coefficient stages are null and "
                                     "excluded from the corresponding partial stage totals."}
            self.last_profile = details
            result.attrs["lpsd_disk_profile"] = copy.deepcopy(details)
            return result

    def close(self):
        """Disable further calls; caller-owned mappings are never closed here."""
        with self._lock:
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("DiskLPSDSubset is closed.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def compute_disk_lpsd(values, workspace, sample_rate, *, outputs="psd", profile=True,
                      plan=None, **kwargs):
    """One complete adapter construction/evaluation; mappings remain caller-owned."""
    started = time.perf_counter()
    with DiskLPSDSubset(len(values), sample_rate, plan=plan, **kwargs) as method:
        result = method.compute(values, workspace, outputs=outputs, profile=profile)
    result.attrs["lpsd_disk_profile"]["constructor_compute_close_wall_s"] = (
        time.perf_counter() - started)
    return result
