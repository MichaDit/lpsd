"""A deliberately approximate FFTW/Kaiser-power smoother for comparison.

This is not an alternative implementation of the exact finite-record LPSD
estimator.  It uses a nearly flat full-record time window, then samples the
analytic *power response* of each target Kaiser window on the FFT grid.
The resulting approximation is closest to uniformly averaged circular shifts
of that target window.  Finite segment boundaries, detrending and phase cross
terms generally prevent equality with LPSD on an individual record.

The planned low-frequency subset is evaluated by the unchanged, pinned native
LPSD channel routine.  No coefficient is fitted to the input or a reference
spectrum.  A full-record Kaiser option is retained as a diagnostic: it has the
same smoothing shape but a different time weighting and noise variance.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

try:
    from benchmarks import bench_fftw as _fftw
    from lpsd_fast import api as _lpsd_api
except ModuleNotFoundError:
    # The comparison directory and the pinned checkout are sibling directories.
    _checkout = Path(__file__).resolve().parent.parent / "lpsd"
    if not (_checkout / "benchmarks" / "bench_fftw.py").is_file():
        raise
    sys.path.insert(0, str(_checkout))
    from benchmarks import bench_fftw as _fftw
    from lpsd_fast import api as _lpsd_api


_SOURCE_METADATA = None


def _source_metadata():
    """Load provenance once, outside repeated prepared computations."""
    global _SOURCE_METADATA
    if _SOURCE_METADATA is None:
        checkout = Path(_lpsd_api.__file__).resolve().parent.parent
        files = [Path(__file__).resolve(), Path(_fftw.__file__).resolve(),
                 Path(_lpsd_api.__file__).resolve(),
                 checkout / "lpsd_fast" / "_native" / "fast_dft.c"]
        _SOURCE_METADATA = {
            "repo": "https://github.com/MichaDit/lpsd",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in files},
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


class MatchedSmoothing:
    """Prepared FFTW power smoothing plus an explicit exact low-f subset.

    ``compute(x, outputs='psd'|'nsd')`` returns an owning float32 DataFrame.
    Construction includes FFT planning, the full-record window, target-grid
    planning, normalization and reusable frequency weights.  Low-f LPSD
    windows/coefficients are prepared again inside every compute call.

    The weight cache is capped by ``max_kernel_cache_mb``.  Uncached kernels
    are regenerated in bounded blocks during compute; metadata records this.
    Memory is O(N+J), with a fixed cache cap and no J-by-N dense matrix.
    """

    def __init__(self, backend, n, sample_rate=50.0, psll=200.0,
                 n_frequencies=1000, n_averages=100, *, workers=None,
                 window_provider=None,
                 full_window="tukey", taper_fraction=.05,
                 low_frequency_max_segment_fraction=1 / 16,
                 kernel_halfwidth_bins=None, max_kernel_cache_mb=1024,
                 block_bins=262144, max_working_mb=4096,
                 window_cache_mb=128, planner="estimate"):
        started = time.perf_counter()
        self.fft = None
        self._closed = False
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
        if not math.isfinite(max_kernel_cache_mb) or max_kernel_cache_mb < 0 or block_bins < 1:
            raise ValueError("Require a nonnegative cache budget and positive block_bins")
        self.workers = backend.threads if workers is None else int(workers)
        if self.workers < 1:
            raise ValueError("workers must be positive")
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
        _lpsd_api._native()
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
        self._kernels = []
        self.window = self.powers = self.scratch = None
        self._last_low_metadata = None
        phases = {"grid_and_sources_s": time.perf_counter() - started}
        try:
            tick = time.perf_counter()
            self.fft = _fftw.RealFFT(backend, self.n, planner=planner)
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
            self.scratch = np.empty_like(self.powers)
            phases["windows_normalization_and_buffers_s"] = time.perf_counter() - tick
            tick = time.perf_counter()
            budget = int(max_kernel_cache_mb * 1024**2)
            used, total_weights, cached_count = 0, 0, 0
            for j in self._high_indices:
                frequency, length = float(self.frequencies[j]), int(self.lengths[j])
                radius = self.halfwidth * self.sample_rate / length
                first = max(0, int(math.ceil((frequency - radius) / self.df)))
                stop = min(self.bin_count, int(math.floor((frequency + radius) / self.df)) + 1)
                count = stop - first
                total_weights += count
                cache = np.empty(count, dtype=np.float64) if used + count * 8 <= budget else None
                denominator = 0.0
                for begin in range(first, stop, self.block_bins):
                    end = min(stop, begin + self.block_bins)
                    weights = self._raw_weights(begin, end, frequency, length)
                    denominator += float(np.sum(weights))
                    # One-sided DC/even-N Nyquist PSD has half the white-noise
                    # density of an interior bin.  Normalize the convolution
                    # with the matching half quadrature weights, while the
                    # numerator keeps both reflected response contributions.
                    if begin == 0:
                        denominator -= .5 * float(weights[0])
                    if self.n % 2 == 0 and end == self.bin_count:
                        denominator -= .5 * float(weights[-1])
                    if cache is not None:
                        cache[begin-first:end-first] = weights
                if not math.isfinite(denominator) or denominator <= 0:
                    raise RuntimeError("Invalid sampled Kaiser-power normalization")
                if cache is not None:
                    cache /= denominator
                    used += cache.nbytes
                    cached_count += 1
                self._kernels.append(_Kernel(int(j), first, stop, length,
                                             frequency, denominator, cache))
            phases["frequency_kernel_preparation_s"] = time.perf_counter() - tick
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
                "exact_low_frequency_method": "unchanged private lpsd_fast.api._run_channel on subset of pinned plan; prepares each call",
                "smoothed_frequency_points": len(self._kernels),
                "cached_frequency_kernels": cached_count,
                "streamed_frequency_kernels": len(self._kernels) - cached_count,
                "kernel_weight_count": total_weights, "cached_kernel_bytes": used,
                "kernel_cache_budget_bytes": budget, "weight_block_bins": self.block_bins,
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
        except BaseException:
            self.close()
            raise

    def _raw_weights(self, first, stop, frequency, length):
        frequencies = np.arange(first, stop, dtype=np.float64) * self.df
        scale = length / self.sample_rate
        direct = (frequencies - frequency) * scale
        summed = frequencies + frequency
        reflected = np.minimum(summed, self.sample_rate - summed) * scale
        return (kaiser_power_response(direct, self.beta, self.halfwidth)
                + kaiser_power_response(reflected, self.beta, self.halfwidth))

    def compute(self, samples, outputs="psd"):
        if self._closed:
            raise RuntimeError("MatchedSmoothing is closed")
        names = (outputs,) if isinstance(outputs, str) else tuple(outputs)
        if not names or len(set(names)) != len(names) or any(name not in ("psd", "nsd") for name in names):
            raise ValueError("outputs must select psd, nsd, or both")
        values = _lpsd_api._array(samples)
        if len(values) != self.n:
            raise ValueError("Input length differs from the prepared length")
        np.subtract(values, values[0], out=self.fft.x)
        self.fft.x -= np.mean(self.fft.x)
        self.fft.x *= self.window
        self.fft.execute()
        _fftw.one_sided_psd(self.fft.y, self.n, self.sample_rate, self.s2,
                            self.powers, self.scratch)
        density = np.empty(len(self.frequencies), dtype=np.float64)
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
        if np.any(self.low_mask):
            low = _lpsd_api._run_channel(
                values, values, self.sample_rate, self._low_plan, np.kaiser,
                self.psll, self.overlap, 0, min(self.workers, int(self.low_mask.sum())),
                "fast", self.max_working_mb, self.window_cache_mb, False, ("psd",), False)
            density[self.low_mask] = low.psd.to_numpy()
            self._last_low_metadata = low.attrs.get("lpsd_fast")
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
        if self.fft is not None:
            self.fft.close()
        self._kernels = []
        self.window = self.powers = self.scratch = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
