"""A deliberately approximate FFTW/Kaiser-power smoother for comparison.

This is not an alternative implementation of the exact finite-record LPSD
estimator.  It uses a nearly flat full-record time window, then samples the
analytic *power response* of each target Kaiser window on the FFT grid.
The resulting approximation is closest to uniformly averaged circular shifts
of that target window.  Finite segment boundaries, detrending and phase cross
terms generally prevent equality with LPSD on an individual record.

The planned low-frequency subset uses the same native LPSD estimator with
reusable projected coefficients.  No coefficient is fitted to the input or a
reference spectrum.  A full-record Kaiser option is retained as a diagnostic:
it has the same smoothing shape but different time weighting and noise variance.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from benchmarks import bench_fftw as _fftw
from benchmarks._fftw_native import FFTWOperations
from lpsd_fast import api as _lpsd_api
from lpsd_fast.prepared import PreparedLPSDSubset


_SOURCE_METADATA = None


def _source_metadata():
    """Load provenance once, outside repeated prepared computations."""
    global _SOURCE_METADATA
    if _SOURCE_METADATA is None:
        checkout = Path(_lpsd_api.__file__).resolve().parent.parent
        files = [Path(__file__).resolve(), Path(_fftw.__file__).resolve(),
                 Path(_lpsd_api.__file__).resolve(),
                 checkout / "lpsd_fast" / "prepared.py",
                 checkout / "benchmarks" / "_fftw_native.py",
                 checkout / "benchmarks" / "_fftw_bluestein.py",
                 checkout / "lpsd_fast" / "_native" / "fast_dft.c",
                 checkout / "lpsd_fast" / "_native" / "fast_dft.h",
                 checkout / "lpsd_fast" / "_native" / "fftw_ops.h",
                 checkout / "lpsd_fast" / "_native" / "fftw_bluestein_ops.h",
                 checkout / "lpsd_fast" / "_native" / "fftw_smoothing.h"]
        _SOURCE_METADATA = {
            "repo": "https://github.com/MichaDit/lpsd",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in files},
            "native_binary_sha256": hashlib.sha256(Path(_lpsd_api._LIB._name).read_bytes()).hexdigest(),
        }
    return dict(_SOURCE_METADATA)


def _periodic_tukey(n, fraction):
    """Periodic Tukey window; fraction is the total fraction occupied by ramps."""
    window = np.ones(n, dtype=np.float64)
    if fraction == 0:
        return window
    half_ramp = fraction * n / 2
    count = min(n, int(math.ceil(half_ramp)))
    left = .5 - .5 * np.cos(np.pi * np.arange(count, dtype=np.float64) / half_ramp)
    window[:count] = left
    if count > 1:
        window[-(count - 1):] = left[1:][::-1]
    return window


def kaiser_power_response(offset_bins, beta, halfwidth):
    """Continuous Kaiser-window power response, with peak one.

    q = frequency_offset * segment_length / sample_rate.
    H(q) = sinhc(sqrt(beta**2 - (pi*q)**2)) / sinhc(beta),
    with analytic continuation sinhc(i*z) = sinc(z/pi).
    The returned kernel is |H|**2, explicitly truncated at |q| <= halfwidth.
    This is a continuous-window approximation, not a claim of an exact
    discrete, locally detrended LPSD response.
    """
    q = np.asarray(offset_bins, dtype=np.float64)
    result = np.zeros_like(q)
    mask = np.abs(q) <= halfwidth
    if not np.any(mask):
        return result
    if beta == 0:
        result[mask] = np.sinc(q[mask]) ** 2
        return result
    value = beta * beta - (np.pi * q[mask]) ** 2
    argument = np.sqrt(np.abs(value))
    scaled = np.ones_like(argument)
    positive = value > 0
    nonzero = argument > 1e-12
    use = positive & nonzero
    scaled[use] = np.sinh(argument[use]) / argument[use]
    use = ~positive & nonzero
    scaled[use] = np.sin(argument[use]) / argument[use]
    scaled *= beta / math.sinh(beta)
    result[mask] = scaled * scaled
    return result


@dataclass
class _Kernel:
    output_index: int
    first: int
    stop: int
    length: int
    frequency: float
    denominator: float
    weights: np.ndarray | None


class _NativeSmoothing:
    """Small private ctypes binding; all arrays remain Python-owned."""

    def __init__(self, library, backend="auto", weight_dtype="float64",
                 defer_normalization=False):
        if backend not in ("auto", "native", "numpy"):
            raise ValueError("smoothing_backend must be auto, native or numpy")
        self.backend, self.unavailable_reason = "numpy", None
        self.weight_dtype = np.dtype(weight_dtype)
        self.defer_normalization = False
        self._power_pointer = self._output_pointer = None
        if backend == "numpy":
            return
        try:
            self._prepare = library.lpsd_fftw_smoothing_prepare
            name = ("lpsd_fftw_smoothing_apply_deferred" if defer_normalization
                    else "lpsd_fftw_smoothing_apply")
            if self.weight_dtype == np.dtype("float32"):
                name += "_f32"
            self._apply = getattr(library, name)
        except AttributeError as error:
            self.unavailable_reason = str(error)
            if backend == "native":
                raise RuntimeError("Native FFTW smoothing unavailable; rebuild lpsd_fast") from error
            return
        self.dp, self.ip = ct.POINTER(ct.c_double), ct.POINTER(ct.c_int64)
        self.wp = ct.POINTER(ct.c_float) if self.weight_dtype == np.dtype("float32") else self.dp
        self._prepare.argtypes = [ct.c_int64, ct.c_double, ct.c_double, ct.c_double,
                                  ct.c_int64, ct.c_int64, ct.c_int64, ct.c_double,
                                  self.dp, self.dp]
        self._prepare.restype = ct.c_int
        self._apply.argtypes = [self.dp, ct.c_int64, ct.c_double, ct.c_double,
                                ct.c_double, ct.c_int64, ct.c_int64,
                                self.ip, self.ip, self.ip, self.ip, self.dp,
                                self.dp, ct.POINTER(self.wp), self.dp]
        self._apply.restype = ct.c_int
        self.backend = "native"
        self.defer_normalization = bool(defer_normalization)

    def prepare(self, owner, kernel):
        denominator = ct.c_double()
        # Compute and normalize in double before any one-time cache rounding.
        temporary = (np.empty(len(kernel.weights), dtype=np.float64)
                     if kernel.weights is not None and kernel.weights.dtype == np.float32
                     else kernel.weights)
        weights = None if temporary is None else temporary.ctypes.data_as(self.dp)
        status = self._prepare(
            owner.n, owner.sample_rate, owner.beta, owner.halfwidth,
            kernel.first, kernel.stop, kernel.length, kernel.frequency,
            weights, ct.byref(denominator))
        if status:
            raise RuntimeError(f"Native Kaiser-power preparation failed ({status})")
        kernel.denominator = denominator.value
        if temporary is not kernel.weights:
            kernel.weights[:] = temporary

    def bind(self, kernels):
        """Keep kernel arrays and their address table alive for repeated calls."""
        self.indices = np.asarray([k.output_index for k in kernels], dtype=np.int64)
        self.firsts = np.asarray([k.first for k in kernels], dtype=np.int64)
        self.stops = np.asarray([k.stop for k in kernels], dtype=np.int64)
        self.lengths = np.asarray([k.length for k in kernels], dtype=np.int64)
        self.frequencies = np.asarray([k.frequency for k in kernels], dtype=np.float64)
        self.denominators = np.asarray([k.denominator for k in kernels], dtype=np.float64)
        self.cached = (self.wp * len(kernels))(*[
            None if k.weights is None else k.weights.ctypes.data_as(self.wp)
            for k in kernels])
        self._arguments = [array.ctypes.data_as(self.ip) for array in
                           (self.indices, self.firsts, self.stops, self.lengths)]
        self._arguments.extend(array.ctypes.data_as(self.dp) for array in
                               (self.frequencies, self.denominators))
        self._arguments.append(self.cached)

    def apply(self, owner, begin, end, output):
        status = self._apply(
            self._power_pointer if self._power_pointer is not None else owner.powers.ctypes.data_as(self.dp), owner.n, owner.sample_rate,
            owner.beta, owner.halfwidth, begin, end, *self._arguments,
            self._output_pointer if self._output_pointer is not None else output.ctypes.data_as(self.dp))
        if status:
            raise RuntimeError(f"Native Kaiser-power smoothing failed ({status})")

    def bind_buffers(self, powers, output):
        """Bind the owning pipeline's private buffers, never caller samples."""
        self._power_pointer = powers.ctypes.data_as(self.dp)
        self._output_pointer = output.ctypes.data_as(self.dp)


class MatchedSmoothing:
    """Prepared FFTW power smoothing plus an explicit exact low-f subset.

    ``compute(x, outputs='psd'|'nsd')`` returns an owning float32 DataFrame.
    Construction includes FFT planning, the full-record window, target-grid
    planning, normalization, frequency weights and the reusable native
    low-frequency LPSD subset.  Its projected coefficients are cached within
    ``window_cache_mb``; any uncached subset work is explicitly bounded.

    The weight cache is capped by ``max_kernel_cache_mb``.  Uncached kernels
    are evaluated directly in native code without intermediate arrays.  The
    independent coefficient cache is capped by ``window_cache_mb`` and its
    working space by ``max_working_mb``.  Metadata reports actual bytes and
    each separate budget.  Memory is O(N+J), plus explicitly capped caches;
    no dense J-by-N matrix is constructed.  One instance is not reentrant.

    With ``total_cache_mb`` set, both persistent caches share that total cap.
    The weight cap is first limited by the total; only actual allocated weight
    bytes are deducted before limiting the low-frequency coefficient cache.
    ``window_cache_mb=None`` uses the entire remaining budget and requires an
    explicit total. A numeric low cap, including zero, is still respected.
    Defaults retain the original independent caps. The combined cache cap
    excludes FFTW/input buffers and temporary working storage; it is not an
    overall process-memory limit.

    ``weight_dtype='float32'`` is an explicit narrower-cache alternative. Only
    normalized, nonnegative cached smoothing weights are rounded; the input,
    transform, powers, accumulation and low-frequency coefficients remain
    float64. The float64 cache is retained as the numerical reference.

    ``defer_normalization=True`` omits the constructor's formula pass for
    uncached native kernels. Their first computation accumulates numerator
    and denominator together and retains the denominator for later calls.
    Cached kernels are prepared as before. The NumPy fallback retains eager
    normalization; metadata distinguishes the requested and actual behavior.
    """

    def __init__(self, backend, n, sample_rate=50.0, psll=200.0,
                 n_frequencies=1000, n_averages=100, *, workers=None,
                 window_provider=None,
                 full_window="tukey", taper_fraction=.05,
                 low_frequency_max_segment_fraction=1 / 16,
                 kernel_halfwidth_bins=None, max_kernel_cache_mb=1536,
                 block_bins=262144, max_working_mb=2048,
                 window_cache_mb=512, planner="estimate",
                 operations_backend="auto", smoothing_backend="auto",
                 weight_dtype="float64", smoothing_workers=None,
                 defer_normalization=False, total_cache_mb=None,
                 fft_algorithm="native", fft_convolution_length=None,
                 time_limit=None):
        started = time.perf_counter()
        self.fft = None
        self._low_prepared = None
        self._executor = None
        self._native_smoothing = None
        self._bound_operations = None
        self._kernels = []
        self._pending_normalizations = ()
        self.last_profile = None
        self._closed = False
        if weight_dtype not in ("float32", "float64"):
            raise ValueError("weight_dtype must be float32 or float64")
        self.weight_dtype = np.dtype(weight_dtype)
        self.backend, self.n = backend, int(n)
        self.sample_rate = float(sample_rate)
        if self.n != n or self.n < 2 or not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("Require integer n >= 2 and a positive finite sample rate")
        if full_window not in ("tukey", "rectangular", "kaiser"):
            raise ValueError("full_window must be tukey, rectangular or kaiser")
        if not math.isfinite(taper_fraction) or not 0 <= taper_fraction <= 1:
            raise ValueError("taper_fraction must lie in [0, 1]")
        if not math.isfinite(low_frequency_max_segment_fraction) or not 0 < low_frequency_max_segment_fraction <= 1:
            raise ValueError("low_frequency_max_segment_fraction must lie in (0, 1]")
        if (not math.isfinite(max_kernel_cache_mb) or max_kernel_cache_mb < 0
                or not isinstance(block_bins, (int, np.integer)) or block_bins < 1):
            raise ValueError("Require a nonnegative cache budget and positive block_bins")
        if window_cache_mb is not None and (not math.isfinite(window_cache_mb) or window_cache_mb < 0):
            raise ValueError("Require a nonnegative LPSD coefficient cache budget")
        if total_cache_mb is not None and (not math.isfinite(total_cache_mb) or total_cache_mb < 0):
            raise ValueError("Require a nonnegative total cache budget")
        if window_cache_mb is None and total_cache_mb is None:
            raise ValueError("window_cache_mb=None requires total_cache_mb")
        if not math.isfinite(max_working_mb) or max_working_mb <= 0:
            raise ValueError("Require a positive LPSD working-space budget")
        if not isinstance(defer_normalization, (bool, np.bool_)):
            raise ValueError("defer_normalization must be a boolean")
        self.workers = backend.threads if workers is None else int(workers)
        if self.workers < 1:
            raise ValueError("workers must be positive")
        self.smoothing_workers = self.workers if smoothing_workers is None else int(smoothing_workers)
        if self.smoothing_workers < 1 or (smoothing_workers is not None and self.smoothing_workers != smoothing_workers):
            raise ValueError("smoothing_workers must be a positive integer")
        self.psll = float(psll)
        self.beta = float(_lpsd_api._kaiser_alpha(self.psll) * np.pi)
        if not math.isfinite(self.beta) or not 0 <= self.beta <= 100:
            raise ValueError("This analytic comparison kernel supports 0 <= beta <= 100")
        self.overlap = float(_lpsd_api._kaiser_rov(self.beta / np.pi))
        self.halfwidth = (max(8.5, self.beta / np.pi + .5)
                          if kernel_halfwidth_bins is None else float(kernel_halfwidth_bins))
        if not math.isfinite(self.halfwidth) or self.halfwidth <= self.beta / np.pi:
            raise ValueError("Kernel halfwidth must exceed beta/pi for the stated tail bound")
        self.block_bins = int(block_bins)
        self.max_working_mb, self.window_cache_mb = max_working_mb, window_cache_mb
        # All budget arithmetic is in whole bytes. Dividing by a power of two
        # only at the existing PreparedLPSDSubset MiB interface preserves the
        # selected byte cap exactly for supported practical array sizes.
        requested_kernel_budget = int(max_kernel_cache_mb * 1024**2)
        requested_low_budget = (None if window_cache_mb is None else
                                int(window_cache_mb * 1024**2))
        total_budget = None if total_cache_mb is None else int(total_cache_mb * 1024**2)
        _lpsd_api._native()
        self.operations = FFTWOperations(backend=operations_backend, library=_lpsd_api._LIB)
        self._native_smoothing = _NativeSmoothing(
            _lpsd_api._LIB, smoothing_backend, weight_dtype, defer_normalization)
        self.defer_normalization = self._native_smoothing.defer_normalization
        self.provider = (window_provider if window_provider is not None else
                         _fftw.KaiserWindow("native", Path(_lpsd_api._LIB._name)))
        source_metadata = _source_metadata()
        self.df = self.sample_rate / self.n
        self.bin_count = self.n // 2 + 1
        raw_plan = _lpsd_api._frequency_plan(
            self.n, self.sample_rate, self.overlap, 1, 0, n_frequencies, n_averages)
        self._plan = tuple(np.asarray(values) for values in raw_plan)
        self.frequencies = self._plan[0].astype(np.float64, copy=True)
        self.lengths = self._plan[3].astype(np.int64, copy=True)
        # Very short windows are also excluded from the continuous-window
        # approximation, independently of the input values.
        self.low_mask = ((self.lengths > self.n * low_frequency_max_segment_fraction)
                         | (self.lengths < 32))
        self._low_plan = tuple(values[self.low_mask].tolist() for values in self._plan)
        self._high_indices = np.flatnonzero(~self.low_mask)
        self.window = self.powers = self.scratch = None
        self._last_low_metadata = None
        phases = {"grid_and_sources_s": time.perf_counter() - started}
        try:
            tick = time.perf_counter()
            self.fft = _fftw.create_real_fft(
                backend, self.n, planner=planner, time_limit=time_limit, algorithm=fft_algorithm,
                convolution_length=fft_convolution_length,
                operations_library=_lpsd_api._LIB)
            phases["fftw_allocation_and_plan_s"] = time.perf_counter() - tick
            tick = time.perf_counter()
            if full_window == "kaiser":
                self.window = self.provider.generate(self.n, self.beta)
            elif full_window == "rectangular":
                self.window = np.ones(self.n, dtype=np.float64)
            else:
                self.window = _periodic_tukey(self.n, taper_fraction)
            self.s1, self.s2 = self.provider.sums(self.window)
            fourth = 0.0
            for first in range(0, self.n, self.block_bins):
                squared = np.square(self.window[first:first + self.block_bins])
                fourth += float(np.dot(squared, squared))
            time_weight_factor = self.n * fourth / self.s2**2
            # At beta≈25.4 the endpoints are tiny.  Scaling the native
            # periodic-window constant gives a close target ENBW without
            # allocating every individual LPSD window merely for metadata.
            reference_length = max(4096, min(self.n, 65536))
            reference_window = self.provider.generate(reference_length, self.beta)
            reference_s1, reference_s2 = self.provider.sums(reference_window)
            enbw_bins = reference_length * reference_s2 / reference_s1**2
            self.enbw = enbw_bins * self.sample_rate / self.lengths
            del reference_window
            self.powers = np.empty(self.bin_count, dtype=np.float64)
            # The native pass consumes the interleaved FFT output directly.
            # Only the NumPy fallback needs a second powers-sized workspace.
            self.scratch = (np.empty_like(self.powers)
                            if self.operations.backend == "numpy" else None)
            self._bound_operations = self.operations.bind(
                time_buffer=self.fft.x, window=self.window, transform=self.fft.y,
                n=self.n, sample_rate=self.sample_rate, window_square_sum=self.s2,
                powers=self.powers, scratch=self.scratch)
            phases["windows_normalization_and_buffers_s"] = time.perf_counter() - tick
            tick = time.perf_counter()
            budget = (requested_kernel_budget if total_budget is None else
                      min(requested_kernel_budget, total_budget))
            used, total_weights, cached_count = 0, 0, 0
            for j in self._high_indices:
                frequency, length = float(self.frequencies[j]), int(self.lengths[j])
                radius = self.halfwidth * self.sample_rate / length
                first = max(0, int(math.ceil((frequency - radius) / self.df)))
                stop = min(self.bin_count, int(math.floor((frequency + radius) / self.df)) + 1)
                count = stop - first
                total_weights += count
                cache = (np.empty(count, dtype=self.weight_dtype)
                         if used + count * self.weight_dtype.itemsize <= budget else None)
                if cache is not None:
                    used += cache.nbytes
                    cached_count += 1
                self._kernels.append(_Kernel(int(j), first, stop, length,
                                             frequency, math.nan, cache))
            self._pending_normalizations = tuple(
                j for j, kernel in enumerate(self._kernels)
                if self.defer_normalization and kernel.weights is None)
            preparation_kernels = [kernel for kernel in self._kernels
                                   if not (self.defer_normalization and kernel.weights is None)]
            # Native functions release the GIL and process independent kernels.
            # Keep this small pool for repeated calls; no BLAS/OpenMP thread pool
            # is created inside the native helpers.
            if self._native_smoothing.backend == "native" and len(self._kernels) > 1 and self.smoothing_workers > 1:
                self._executor = ThreadPoolExecutor(
                    max_workers=min(self.smoothing_workers, len(self._kernels)),
                    thread_name_prefix="lpsd-smoothing")
                list(self._executor.map(self._prepare_kernel, preparation_kernels))
            else:
                for kernel in preparation_kernels:
                    self._prepare_kernel(kernel)
            if self._native_smoothing.backend == "native":
                self._native_smoothing.bind(self._kernels)
            self._apply_ranges = self._kernel_ranges()
            phases["frequency_kernel_preparation_s"] = time.perf_counter() - tick
            tick = time.perf_counter()
            remaining_budget = None if total_budget is None else max(0, total_budget - used)
            low_budget = (remaining_budget if requested_low_budget is None else
                          requested_low_budget if remaining_budget is None else
                          min(requested_low_budget, remaining_budget))
            self.window_cache_mb = low_budget / 1024**2
            if np.any(self.low_mask):
                self._low_prepared = PreparedLPSDSubset(
                    self.n, self.sample_rate, self._low_plan, psll=self.psll,
                    overlap=self.overlap, workers=min(self.workers, int(self.low_mask.sum())),
                    max_cache_mb=self.window_cache_mb, max_working_mb=self.max_working_mb)
                self._last_low_metadata = self._low_prepared.metadata
            phases["low_frequency_lpsd_preparation_s"] = time.perf_counter() - tick
            if self.beta:
                tail = (self.beta / (math.pi * math.sinh(self.beta)**2)
                        * math.log((math.pi*self.halfwidth+self.beta)
                                   / (math.pi*self.halfwidth-self.beta)) / enbw_bins)
            else:
                tail = 2 / (math.pi**2 * self.halfwidth * enbw_bins)
            self.metadata = {
                "algorithm": "FFTW analytic Kaiser-power smoothing with explicit low-frequency LPSD subset",
                "same_estimator_as_lpsd": False,
                "source": source_metadata,
                "fftw_version": backend.version,
                "n": self.n, "sample_rate_hz": self.sample_rate,
                "output_points": len(self.frequencies), "workers": self.workers,
                "fftw_threads": backend.threads, "planner": planner,
                "planner_time_limit_s": time_limit,
                "fft_algorithm": fft_algorithm,
                "fft_convolution_length": fft_convolution_length,
                "fft": self.fft.describe(),
                "requested_smoothing_workers": self.smoothing_workers,
                "operation_binding": self._bound_operations.metadata,
                "full_record_window": full_window,
                "full_record_taper_fraction": float(taper_fraction) if full_window == "tukey" else None,
                "full_record_kaiser_psll_db": self.psll if full_window == "kaiser" else None,
                "full_record_window_enbw_hz": self.sample_rate * self.s2 / self.s1**2,
                "full_record_time_weight_factor_n_sum_w4_over_sum_w2_squared": time_weight_factor,
                "target_kaiser_psll_db": self.psll, "target_kaiser_beta": self.beta,
                "target_lpsd_overlap": self.overlap,
                "target_enbw_method": "native periodic Kaiser reference constant scaled by fs/Lj",
                "target_enbw_reference_length": reference_length,
                "target_enbw_reference_bins": enbw_bins,
                "kernel": "squared analytic continuous Kaiser Fourier response, no data fit",
                "kernel_halfwidth_segment_bins": self.halfwidth,
                "continuous_kernel_omitted_mass_upper_bound": tail,
                "tail_bound_scope": "continuous Kaiser kernel only; not a bound on input leakage or LPSD error",
                "negative_frequency_handling": "reflected power response with correct r2c endpoint weights",
                "low_frequency_max_segment_fraction": float(low_frequency_max_segment_fraction),
                "exact_low_frequency_points": int(self.low_mask.sum()),
                "exact_low_frequency_max_hz": float(self.frequencies[self.low_mask].max()) if np.any(self.low_mask) else None,
                "exact_low_frequency_method": "PreparedLPSDSubset on the unchanged plan subset; reuses bounded native projected coefficients",
                "low_frequency_preparation": self._last_low_metadata,
                "low_frequency_cache_budget_bytes": low_budget,
                "low_frequency_working_budget_bytes": int(self.max_working_mb * 1024**2),
                "smoothed_frequency_points": len(self._kernels),
                "smoothing_backend": self._native_smoothing.backend,
                "smoothing_backend_unavailable_reason": self._native_smoothing.unavailable_reason,
                "fftw_operations": self.operations.metadata,
                "smoothing_workers": min(self.smoothing_workers, len(self._kernels)) if self._executor else 1,
                "smoothing_task_ranges": len(self._apply_ranges),
                "deferred_normalization_requested": bool(defer_normalization),
                "deferred_normalization": self.defer_normalization,
                "initially_deferred_frequency_kernels": len(self._pending_normalizations),
                "normalization_timing": ("Cached kernels at construction; uncached native kernels "
                                         "in their first compute, with denominator retained."
                                         if self.defer_normalization else
                                         "All kernels at construction."),
                "cached_frequency_kernels": cached_count,
                "streamed_frequency_kernels": len(self._kernels) - cached_count,
                "kernel_weight_count": total_weights, "cached_kernel_bytes": used,
                "kernel_cache_budget_bytes": budget, "weight_block_bins": self.block_bins,
                "cache_budget_mode": "independent" if total_budget is None else "total",
                "requested_kernel_cache_budget_bytes": requested_kernel_budget,
                "requested_low_frequency_cache_budget_bytes": requested_low_budget,
                "total_cache_budget_bytes": total_budget,
                "remaining_cache_budget_after_weights_bytes": remaining_budget,
                "persistent_cache_bytes": used + (self._last_low_metadata["cached_coefficient_bytes"]
                                                  if self._last_low_metadata else 0),
                "cache_budget_scope": "Persistent smoothing weights and projected low-frequency "
                                      "coefficients only; excludes input, FFTW/array buffers, "
                                      "temporary preparation and worker storage.",
                "kernel_weight_dtype": self.weight_dtype.name,
                "kernel_cache_rounding": "once after double normalization" if self.weight_dtype == np.float32 else "none beyond double arithmetic",
                "weight_preparation_temporary_bytes_per_worker_upper_bound":
                    max((8 * len(k.weights) for k in self._kernels if k.weights is not None), default=0)
                    if self.weight_dtype == np.float32 else 0,
                "fftw_array_bytes": int(self.fft.x.nbytes + self.fft.y.nbytes
                                        + self.window.nbytes + self.powers.nbytes
                                        + (0 if self.scratch is None else self.scratch.nbytes)),
                "streamed_kernel_temporary_bytes": (0 if self._native_smoothing.backend == "native"
                                                     else "bounded NumPy arrays of weight_block_bins"),
                "output_dtype": "float32", "power_arithmetic": "float64",
                "nsd": "complex64 square root after float32 PSD rounding",
                "setup_phases": phases,
                "setup_wall_s": time.perf_counter() - started,
                "limitations": [
                    "The Tukey/rectangular full-record window does not inherit the target Kaiser's 200-dB sidelobe specification.",
                    "Off-bin tones can leave a different leakage floor; it must be plotted and checked, not hidden.",
                    "Matching the spectral kernel does not guarantee identical finite-record time weighting, variance or point values.",
                    "Low-frequency fallback retains the current LPSD first-segment recurrence behavior.",
                    "No reference-spectrum fitting, per-realization normalization, or hidden accuracy-triggered rerun is used.",
                ],
            }
            self.frequencies.setflags(write=False)
            self.enbw.setflags(write=False)
            self._density = np.empty(len(self.frequencies), dtype=np.float64)
            if self._native_smoothing.backend == "native":
                self._native_smoothing.bind_buffers(self.powers, self._density)
        except BaseException:
            self.close()
            raise

    def _prepare_kernel(self, kernel):
        if self._native_smoothing.backend == "native":
            self._native_smoothing.prepare(self, kernel)
            return
        # Independently retained NumPy implementation of the original formula.
        # This explicit fallback also provides a useful numerical cross-check.
        denominator = 0.0
        # Match the native mixed-cache route: normalize a double temporary,
        # then round the stored weights once, never before normalization.
        temporary = (np.empty(len(kernel.weights), dtype=np.float64)
                     if kernel.weights is not None and kernel.weights.dtype == np.float32
                     else kernel.weights)
        for begin in range(kernel.first, kernel.stop, self.block_bins):
            end = min(kernel.stop, begin + self.block_bins)
            weights = self._raw_weights(begin, end, kernel.frequency, kernel.length)
            denominator += float(np.sum(weights))
            if begin == 0:
                denominator -= .5 * float(weights[0])
            if self.n % 2 == 0 and end == self.bin_count:
                denominator -= .5 * float(weights[-1])
            if temporary is not None:
                temporary[begin - kernel.first:end - kernel.first] = weights
        if not math.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("Invalid sampled Kaiser-power normalization")
        kernel.denominator = denominator
        if temporary is not None:
            temporary /= denominator
            if temporary is not kernel.weights:
                kernel.weights[:] = temporary

    def _kernel_ranges(self):
        if not self._kernels:
            return []
        if self._executor is None:
            return [(0, len(self._kernels))]
        # An uncached sample evaluates libm functions in addition to a multiply;
        # its larger scheduling weight changes only work assignment, not values.
        costs = np.asarray([(k.stop - k.first) * (32 if k.weights is None else 1)
                            for k in self._kernels], dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(costs)))
        tasks = min(len(self._kernels), self.smoothing_workers * 2)
        cuts = np.searchsorted(cumulative, np.linspace(0, cumulative[-1], tasks + 1))
        cuts = np.unique(np.clip(cuts, 0, len(self._kernels)))
        return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]

    def _raw_weights(self, first, stop, frequency, length):
        frequencies = np.arange(first, stop, dtype=np.float64) * self.df
        scale = length / self.sample_rate
        direct = (frequencies - frequency) * scale
        summed = frequencies + frequency
        reflected = np.minimum(summed, self.sample_rate - summed) * scale
        return (kaiser_power_response(direct, self.beta, self.halfwidth)
                + kaiser_power_response(reflected, self.beta, self.halfwidth))

    def compute(self, samples, outputs="psd"):
        """Compute the complete spectrum without profiling timer reads."""
        return self._compute(samples, outputs, None)

    def profile(self, samples, outputs="psd"):
        """Return the same spectrum and save a separately instrumented profile.

        Run this outside performance repetitions: the ordinary ``compute``
        path contains no timer reads and does not update ``last_profile``.
        """
        phases = {}
        started = time.perf_counter()
        result = self._compute(samples, outputs, phases)
        self.last_profile = {"phases": phases, "total_s": time.perf_counter() - started,
                             "smoothing_backend": self.metadata["smoothing_backend"],
                             "cached_kernel_bytes": self.metadata["cached_kernel_bytes"],
                             "streamed_frequency_kernels": self.metadata["streamed_frequency_kernels"]}
        return result

    def _compute(self, samples, outputs, phases):
        tick = time.perf_counter() if phases is not None else 0.0
        if self._closed:
            raise RuntimeError("MatchedSmoothing is closed")
        names = (outputs,) if isinstance(outputs, str) else tuple(outputs)
        if not names or len(set(names)) != len(names) or any(name not in ("psd", "nsd") for name in names):
            raise ValueError("outputs must select psd, nsd, or both")
        values = _lpsd_api._array(samples)
        if len(values) != self.n:
            raise ValueError("Input length differs from the prepared length")
        if phases is not None:
            now = time.perf_counter(); phases["input_validation_s"] = now - tick; tick = now
        self._bound_operations._prepare_validated(values)
        if phases is not None:
            now = time.perf_counter(); phases["copy_mean_and_window_s"] = now - tick; tick = now
        self.fft.execute()
        if phases is not None:
            now = time.perf_counter(); phases["fftw_execute_s"] = now - tick; tick = now
        self._bound_operations.power()
        if phases is not None:
            now = time.perf_counter(); phases["one_sided_power_s"] = now - tick; tick = now
        density = self._density
        if self._native_smoothing.backend == "native":
            if self._executor is not None:
                # Each task writes a disjoint subset of the owning density array.
                list(self._executor.map(
                    lambda bounds: self._native_smoothing.apply(self, *bounds, density),
                    self._apply_ranges))
            else:
                for begin, end in self._apply_ranges:
                    self._native_smoothing.apply(self, begin, end, density)
            if self._pending_normalizations:
                # Native ranges own disjoint denominator entries. Synchronize
                # their scalar Python descriptions once after all ranges end.
                for j in self._pending_normalizations:
                    denominator = float(self._native_smoothing.denominators[j])
                    if not math.isfinite(denominator) or denominator <= 0:
                        raise RuntimeError("Deferred Kaiser-power normalization was not completed")
                    self._kernels[j].denominator = denominator
                self._pending_normalizations = ()
        else:
            for kernel in self._kernels:
                if kernel.weights is not None:
                    density[kernel.output_index] = np.dot(
                        kernel.weights, self.powers[kernel.first:kernel.stop])
                else:
                    power = 0.0
                    for first in range(kernel.first, kernel.stop, self.block_bins):
                        stop = min(kernel.stop, first + self.block_bins)
                        weights = self._raw_weights(first, stop, kernel.frequency, kernel.length)
                        power += float(np.dot(weights, self.powers[first:stop]))
                    density[kernel.output_index] = power / kernel.denominator
        if phases is not None:
            now = time.perf_counter(); phases["frequency_smoothing_s"] = now - tick; tick = now
        if self._low_prepared is not None:
            density[self.low_mask] = self._low_prepared._compute_validated(values)
        if phases is not None:
            now = time.perf_counter(); phases["low_frequency_lpsd_s"] = now - tick; tick = now
        psd = density.astype(np.float32)
        columns = {}
        for name in names:
            columns[name] = psd if name == "psd" else np.sqrt(psd.astype(np.complex64)).real.copy()
        result = pd.DataFrame(columns, index=pd.Index(self.frequencies, name="frequency"), copy=False)
        result.attrs["matched_smoothing"] = {
            "full_record_window": self.metadata["full_record_window"],
            "exact_low_frequency_points": self.metadata["exact_low_frequency_points"],
            "same_estimator_as_lpsd": False,
        }
        if phases is not None:
            phases["output_conversion_and_dataframe_s"] = time.perf_counter() - tick
        return result

    def validation(self):
        """Global FFT normalization check; call outside compute timing."""
        if self._closed:
            raise RuntimeError("MatchedSmoothing is closed")
        expected = float(np.dot(self.fft.x, self.fft.x) / self.s2)
        obtained = float(np.sum(self.powers) * self.df)
        return {"global_windowed_time_power_v2": expected,
                "global_fftw_integrated_power_v2": obtained,
                "parseval_absolute_error_v2": abs(obtained - expected),
                "finite_global_psd": bool(np.isfinite(self.powers).all()),
                "note": "Smoothed LPSD-label values are point estimates, not a disjoint power partition."}

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        if self._low_prepared is not None:
            self._low_prepared.close()
            self._low_prepared = None
        if self._bound_operations is not None:
            self._bound_operations.close()
            self._bound_operations = None
        if self.fft is not None:
            self.fft.close()
        self._native_smoothing = None
        self._kernels = []
        self._pending_normalizations = ()
        self.window = self.powers = self.scratch = self._density = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def estimate_once(backend, samples, outputs="psd", **kwargs):
    """Construct, compute and close one complete FFTW-hybrid estimate.

    Both ``max_kernel_cache_mb`` and ``window_cache_mb`` are forced to zero:
    no persistent weight or low-frequency coefficient cache is built for a
    pipeline that will not be reused. Other constructor options pass through.
    Native normalization is deferred by default here, unless explicitly
    disabled; the prepared class itself keeps its existing eager default.

    Measure around this function to include planning, windows, computation
    and resource release. Preparation metadata is retained in the returned
    DataFrame's ``matched_smoothing_once`` attribute, without input or cache
    arrays. Public input validation and PSD/NSD rounding are unchanged.
    """
    options = dict(kwargs)
    options["max_kernel_cache_mb"] = 0.0
    options["window_cache_mb"] = 0.0
    options.setdefault("defer_normalization", True)
    with MatchedSmoothing(backend, len(samples), **options) as pipeline:
        result = pipeline.compute(samples, outputs=outputs)
        result.attrs["matched_smoothing_once"] = pipeline.metadata.copy()
    return result
