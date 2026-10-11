# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit, experimental Bluestein adapter for an exact N-point real DFT.

``BluesteinRealFFT`` has the x/y/execute/close/describe interface of the
benchmark's ``RealFFT``.  It does not change the sample count, output bins,
normalization, window, or density estimator.  Only the algorithm used to
calculate the original unnormalized N-point DFT changes.  Floating-point
rounding consequently differs from FFTW's direct r2c plan.

The N-point DFT becomes a complex convolution.  Computing only the
``K = N//2 + 1`` real-input output bins requires ``M >= N + K - 1``;
``convolution_length`` can select any such FFTW-supported length.  The default
remains ``M = next_power_of_two(2*N - 1)``.  One kernel transform is prepared
once; every execution uses one forward and one backward in-place C2C FFTW
plan.  There are no automatic selection rules in this module.

The constructor is deliberately serial: FFTW's planner and its global thread
and time-limit settings are not thread safe.  Execution uses the thread count
of the supplied FFTWLibrary backend, without another layer of worker threads.
Separate instances may execute independently; one instance is not reentrant.
The exposed FFTW-owned x/y arrays must not be resized or rebound, and their
views must not be used after close().
"""

import ctypes as ct
import hashlib
import math
import operator
import os
from pathlib import Path
import time

import numpy as np


_DOUBLE_PTR = ct.POINTER(ct.c_double)
_PLANNERS = {"estimate": 64, "measure": 0, "patient": 32}
_FORWARD = -1
_BACKWARD = +1
_DEFAULT_CHIRP_BLOCK = 1 << 18


def _dimensions(n, chirp_block_size, convolution_length=None):
    if isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be an integer sample count")
    n = operator.index(n)
    chirp_block_size = operator.index(chirp_block_size)
    if n < 2:
        raise ValueError("Require n >= 2")
    if chirp_block_size < 1:
        raise ValueError("chirp_block_size must be positive")
    minimum = n + n // 2
    if convolution_length is None:
        m = 1 << (2 * n - 2).bit_length()
    else:
        if isinstance(convolution_length, (bool, np.bool_)):
            raise TypeError("convolution_length must be an integer")
        m = operator.index(convolution_length)
        if m < minimum:
            raise ValueError(
                f"convolution_length must be at least N + (N//2+1) - 1 = {minimum}"
            )
    if m > np.iinfo(np.int32).max:
        raise ValueError("Bluestein convolution exceeds FFTW's INT_MAX 1D interface")
    return n, m, min(n, chirp_block_size)


def _plan_description(lib, plan):
    add, mul, fma = ct.c_double(), ct.c_double(), ct.c_double()
    lib.fftw_flops(plan, ct.byref(add), ct.byref(mul), ct.byref(fma))
    # On Windows the DLL's ordinary malloc and fftw_free's aligned allocator
    # need not pair.  Match the existing RealFFT diagnostic's safe omission.
    raw = b""
    if os.name != "nt":
        pointer = lib.fftw_sprint_plan(plan)
        try:
            raw = ct.string_at(pointer) if pointer else b""
        finally:
            if pointer:
                lib.fftw_free(pointer)
    return {
        "flops": {"add": add.value, "mul": mul.value, "fma": fma.value,
                  "total_fma_as_two": add.value + mul.value + 2 * fma.value},
        "plan_text": raw[:8192].decode(errors="replace"),
        "plan_text_available": os.name != "nt",
        "plan_text_truncated": len(raw) > 8192,
        "plan_text_sha256": hashlib.sha256(raw).hexdigest(),
    }


class BluesteinRealFFT:
    """Prepared exact-length DFT through FFTW complex Bluestein convolution.

    ``max_working_mb`` caps this adapter's known arrays plus its bounded chirp
    construction scratch, in MiB.  FFTW's internal plan/planning allocations
    and the caller's samples/window/power arrays are additional; this is not
    a process-RSS cap.  ``None`` disables the adapter-buffer cap.  Planning
    flags and the optional time limit apply to each of the two C2C plans.
    Caller initializes x after construction, just as for RealFFT.

    ``convolution_length=None`` retains the original power-of-two length
    ``next_power_of_two(2*N-1)``.  An explicit length may be shorter, down to
    ``N + (N//2+1) - 1``: only the first ``N//2+1`` convolution outputs are
    needed.  This changes neither the N-point DFT nor its frequency grid.

    ``native_ops='auto'`` uses the additive lpsd_fast ABI to fuse the input
    chirp/zero-tail and output chirp/inverse-scale passes when available.
    ``'numpy'`` retains the explicit NumPy implementation; ``'native'``
    requires that ABI.  The supplied operations_library may be a CDLL or a
    path.  These helpers do not center samples, change FFT normalization,
    allocate work buffers, or create their own worker threads.
    """

    @staticmethod
    def memory_estimate(n, chirp_block_size=_DEFAULT_CHIRP_BLOCK,
                        convolution_length=None):
        """Known owned arrays and worst simultaneous chirp scratch, no FFTW plan."""
        n, m, block = _dimensions(n, chirp_block_size, convolution_length)
        bins = n // 2 + 1
        # x=float64[N], y=complex128[bins], chirp=complex128[N], and
        # complex128[M] for both work and the prepared kernel spectrum.
        persistent = 8 * n + 16 * bins + 16 * n + 32 * m
        # uint64 residues, float64 angles, and one boolean reduction mask.
        scratch = 17 * block
        return {
            "n": n, "convolution_length": m, "r2c_bins": bins,
            "minimum_convolution_length": n + bins - 1,
            "persistent_buffer_bytes": persistent,
            "chirp_setup_scratch_bytes": scratch,
            "known_peak_buffer_bytes": persistent + scratch,
            "excludes": "FFTW internal plans/planning allocations and caller arrays",
        }

    def __init__(self, backend, n, planner="estimate", time_limit=None,
                 max_working_mb=512.0, chirp_block_size=_DEFAULT_CHIRP_BLOCK,
                 convolution_length=None, native_ops="auto",
                 operations_library=None):
        n, m, block = _dimensions(n, chirp_block_size, convolution_length)
        if planner not in _PLANNERS:
            raise ValueError("Unknown planner")
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
            raise ValueError("Planner time limit must be finite and nonnegative")
        if max_working_mb is not None and (not math.isfinite(max_working_mb) or max_working_mb <= 0):
            raise ValueError("max_working_mb must be finite and positive, or None")
        if native_ops not in ("auto", "native", "numpy"):
            raise ValueError("native_ops must be auto, native, or numpy")
        memory = self.memory_estimate(n, block, m)
        if max_working_mb is not None and memory["known_peak_buffer_bytes"] > max_working_mb * 1024 ** 2:
            raise MemoryError(
                f"Bluestein N={n}, M={m} needs at least "
                f"{memory['known_peak_buffer_bytes'] / 1024 ** 2:.2f} MiB of "
                f"known arrays/setup scratch; max_working_mb={max_working_mb}. "
                "FFTW's internal plan memory is additional."
            )
        self.backend, self.n, self.m = backend, n, m
        self.convolution_length = m
        self.convolution_length_requested = None if convolution_length is None else m
        self.planner, self.time_limit = planner, time_limit
        self.max_working_mb, self.memory = max_working_mb, memory
        self.chirp_block_size = block
        self.ip = self.op = self.wp = self.kp = None
        # .plan is the forward plan and preserves RealFFT's closed-state
        # contract.  The backward plan has separate ownership.
        self.plan = self.inverse_plan = None
        self.x = self.y = self._work = self._kernel_fft = self._chirp = None
        self._input = self._output = None
        self._work_real = self._work_imag = self._work_tail = None
        self._result_head = self._chirp_real = self._chirp_imag = self._chirp_output = None
        self._native_lib = self._native_prepare = self._native_finish = None
        self._native_prepare_args = self._native_finish_args = None
        self.native_ops = "numpy"
        self.native_ops_metadata = {
            "requested": native_ops, "backend": "numpy", "abi_version": None,
            "library": None, "fallback_reason": None, "native_worker_threads": 0,
        }
        self.allocation_s = self.planning_s = self.kernel_preparation_s = 0.0
        self.native_setup_s = 0.0
        self._inverse_m = 1.0 / m
        lib = backend.lib
        plan_c2c = lib.fftw_plan_dft_1d
        plan_c2c.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_uint]
        plan_c2c.restype = ct.c_void_p
        self._execute = lib.fftw_execute
        started = time.perf_counter()
        self._load_native_operations(native_ops, operations_library)
        self.native_setup_s = time.perf_counter() - started
        started = time.perf_counter()
        try:
            self.ip = lib.fftw_malloc(n * 8)
            self.op = lib.fftw_malloc((n // 2 + 1) * 16)
            self.wp = lib.fftw_malloc(m * 16)
            self.kp = lib.fftw_malloc(m * 16)
            if not all((self.ip, self.op, self.wp, self.kp)):
                raise MemoryError("FFTW aligned allocation failed")
            self.x = np.ctypeslib.as_array(ct.cast(self.ip, _DOUBLE_PTR), shape=(n,))
            self.y = self._complex_view(self.op, n // 2 + 1)
            self._work = self._complex_view(self.wp, m)
            self._kernel_fft = self._complex_view(self.kp, m)
            self._chirp = np.empty(n, dtype=np.complex128)
            self.allocation_s = time.perf_counter() - started
            thread_runtime = backend.threadlib or getattr(type(backend), "_thread_runtime", None)
            if thread_runtime:
                thread_runtime.fftw_plan_with_nthreads(backend.threads)
            lib.fftw_set_timelimit(-1.0 if time_limit is None else time_limit)
            started = time.perf_counter()
            self.plan = plan_c2c(m, self.wp, self.wp, _FORWARD, _PLANNERS[planner])
            if not self.plan:
                raise RuntimeError("FFTW Bluestein forward plan creation failed")
            self.inverse_plan = plan_c2c(m, self.wp, self.wp, _BACKWARD, _PLANNERS[planner])
            self.planning_s = time.perf_counter() - started
            if not self.inverse_plan:
                raise RuntimeError("FFTW Bluestein backward plan creation failed")
            # Measuring planners may overwrite their buffers: initialize the
            # convolution only after BOTH plans have been made.
            started = time.perf_counter()
            self._prepare_kernel()
            self.kernel_preparation_s = time.perf_counter() - started
            # Views and scalar normalization are immutable plan-owned state.
            self._input, self._output = self.x, self.y
            self._work_real = self._work[:n].real
            self._work_imag = self._work[:n].imag
            self._work_tail = self._work[n:]
            self._result_head = self._work[:n // 2 + 1]
            self._chirp_real, self._chirp_imag = self._chirp.real, self._chirp.imag
            self._chirp_output = self._chirp[:n // 2 + 1]
            if self._native_lib is not None:
                started = time.perf_counter()
                # The adapter exclusively owns these buffers. Bind their
                # pointers once; no public/caller pointers are retained.
                chirp_pointer = self._chirp.ctypes.data_as(_DOUBLE_PTR)
                self._native_prepare_args = (
                    ct.cast(self.ip, _DOUBLE_PTR), chirp_pointer,
                    ct.c_int64(n), ct.c_int64(m), ct.cast(self.wp, _DOUBLE_PTR))
                self._native_finish_args = (
                    ct.cast(self.wp, _DOUBLE_PTR), chirp_pointer, ct.c_int64(n),
                    ct.c_double(self._inverse_m), ct.cast(self.op, _DOUBLE_PTR))
                self.native_setup_s += time.perf_counter() - started
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _complex_view(pointer, length):
        raw = np.ctypeslib.as_array(ct.cast(pointer, _DOUBLE_PTR), shape=(2 * length,))
        return raw.view(np.complex128)

    def _load_native_operations(self, requested, library):
        if requested == "numpy":
            return
        try:
            if library is None:
                from lpsd_fast import api
                api._native()
                lib = api._LIB
            elif isinstance(library, ct.CDLL):
                lib = library
            else:
                lib = ct.CDLL(str(Path(library)))
            version = lib.fftw_bluestein_ops_version
            version.argtypes, version.restype = [], ct.c_int
            if version() != 1:
                raise RuntimeError("Unsupported Bluestein-operations native ABI")
            prepare = lib.fftw_bluestein_prepare
            prepare.argtypes = [_DOUBLE_PTR, _DOUBLE_PTR, ct.c_int64, ct.c_int64,
                                _DOUBLE_PTR]
            prepare.restype = ct.c_int
            finish = lib.fftw_bluestein_finish
            finish.argtypes = [_DOUBLE_PTR, _DOUBLE_PTR, ct.c_int64, ct.c_double,
                               _DOUBLE_PTR]
            finish.restype = ct.c_int
            self._native_lib, self._native_prepare, self._native_finish = lib, prepare, finish
            self.native_ops = "native"
            self.native_ops_metadata.update(
                backend="native", abi_version=1, library=str(lib._name))
        except (ImportError, OSError, AttributeError, RuntimeError) as exc:
            if requested == "native":
                raise RuntimeError("Native Bluestein operations are unavailable; rebuild lpsd_fast") from exc
            self.native_ops_metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"

    def _prepare_kernel(self):
        n, m, block = self.n, self.m, self.chirp_block_size
        # Since N <= M <= INT_MAX, uint64 j*j cannot overflow.
        # Reduce the integer square BEFORE conversion to floating point; no
        # trigonometric function sees a phase growing quadratically with j.
        for start in range(0, n, block):
            stop = min(start + block, n)
            residues = np.arange(start, stop, dtype=np.uint64)
            np.square(residues, out=residues)
            np.remainder(residues, np.uint64(2 * n), out=residues)
            angles = residues.astype(np.float64)
            negative = angles > n
            np.subtract(angles, 2.0 * n, out=angles, where=negative)
            angles *= -np.pi / n
            np.cos(angles, out=self._chirp[start:stop].real)
            np.sin(angles, out=self._chirp[start:stop].imag)
            # Release the block before the next block allocation, so the
            # advertised setup scratch is a simultaneous-allocation bound.
            del residues, angles, negative
        self._chirp[0] = 1.0
        self._work.fill(0.0)
        # c[j] = exp(-i*pi*j*j/N).  For the requested outputs k=0..K-1,
        # a[j] has j=0..N-1, so only kernel lags -(N-1)..K-1 occur.
        # Embed b[lag] = conj(c[abs(lag)]) modulo M.  With M >= N+K-1
        # the two written ranges cannot collide, even at the minimum M.
        # The unused convolution outputs k>=K are intentionally irrelevant.
        bins = n // 2 + 1
        np.conjugate(self._chirp[:bins], out=self._work[:bins])
        np.conjugate(self._chirp[:0:-1], out=self._work[m - n + 1:])
        self._execute(self.plan)
        np.copyto(self._kernel_fft, self._work)
        self._kernel_fft.flags.writeable = False
        self._chirp.flags.writeable = False

    def execute(self):
        if not self.plan or not self.inverse_plan:
            raise RuntimeError("Plan is closed")
        if self._native_prepare is not None:
            if self._native_prepare(*self._native_prepare_args):
                raise RuntimeError("Invalid bound native Bluestein input arguments")
        else:
            np.multiply(self._input, self._chirp_real, out=self._work_real)
            np.multiply(self._input, self._chirp_imag, out=self._work_imag)
            self._work_tail.fill(0.0)
        self._execute(self.plan)
        np.multiply(self._work, self._kernel_fft, out=self._work)
        self._execute(self.inverse_plan)
        if self._native_finish is not None:
            if self._native_finish(*self._native_finish_args):
                raise RuntimeError("Invalid bound native Bluestein output arguments")
        else:
            np.multiply(self._result_head, self._chirp_output, out=self._output)
            self._output *= self._inverse_m
            # The real-input DC and (only for even N) Nyquist coefficients are
            # mathematically real; enforce the same r2c representation as FFTW.
            self._output[0] = self._output[0].real
            if self.n % 2 == 0:
                self._output[-1] = self._output[-1].real

    def describe(self):
        if not self.plan or not self.inverse_plan:
            raise RuntimeError("Plan is closed")
        forward = _plan_description(self.backend.lib, self.plan)
        backward = _plan_description(self.backend.lib, self.inverse_plan)
        flops = {key: forward["flops"][key] + backward["flops"][key]
                 for key in forward["flops"]}
        plan_text = "Forward convolution plan:\n" + forward["plan_text"] + "\nBackward convolution plan:\n" + backward["plan_text"]
        return {
            "algorithm": "exact_n_point_bluestein_via_fftw_c2c",
            "experimental": True, "n": self.n, "r2c_bins": self.n // 2 + 1,
            "convolution_length": self.m,
            "convolution_length_requested": self.convolution_length_requested,
            "minimum_convolution_length": self.n + self.n // 2,
            "kernel_lags": {"minimum": -(self.n - 1), "maximum": self.n // 2},
            "planner": self.planner, "flags": _PLANNERS[self.planner],
            "time_limit_s": self.time_limit, "time_limit_scope": "per C2C plan",
            "threads": self.backend.threads,
            "thread_policy": "FFTW internal threads; no outer workers; serial planning",
            "planning_s": self.planning_s, "aligned_allocation_s": self.allocation_s,
            "kernel_preparation_s": self.kernel_preparation_s,
            "native_operations": dict(self.native_ops_metadata),
            "native_setup_s": self.native_setup_s,
            "setup_components_s": self.allocation_s + self.planning_s + self.kernel_preparation_s + self.native_setup_s,
            "fft_executions_per_call": 2, "kernel_fft_executions_during_setup": 1,
            "normalization": "unnormalized original N-point DFT; inverse convolution scaled by 1/M",
            "chirp_phase": "uint64 (j*j) % (2*N), mapped into [-N,N] before float64 trig",
            "memory": dict(self.memory), "max_working_mb": self.max_working_mb,
            "flops": flops, "flops_scope": "two C2C transforms only; chirp/scale/kernel element operations excluded",
            "forward_plan": forward, "backward_plan": backward,
            "plan_text": plan_text,
            "plan_text_available": forward["plan_text_available"] and backward["plan_text_available"],
            "plan_text_truncated": forward["plan_text_truncated"] or backward["plan_text_truncated"],
            "plan_text_sha256": hashlib.sha256(plan_text.encode()).hexdigest(),
        }

    def close(self):
        # Drop the cached pointer tuples before any underlying array/malloc.
        self._native_prepare_args = self._native_finish_args = None
        self._native_prepare = self._native_finish = self._native_lib = None
        self.x = self.y = self._input = self._output = None
        self._work_real = self._work_imag = self._work_tail = None
        self._result_head = self._chirp_real = self._chirp_imag = self._chirp_output = None
        self._work = self._kernel_fft = self._chirp = None
        lib = self.backend.lib
        if self.plan:
            lib.fftw_destroy_plan(self.plan)
            self.plan = None
        if self.inverse_plan:
            lib.fftw_destroy_plan(self.inverse_plan)
            self.inverse_plan = None
        for name in ("ip", "op", "wp", "kp"):
            pointer = getattr(self, name)
            if pointer:
                lib.fftw_free(pointer)
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
