# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared optional native operations for the two FFTW spectral pipelines.

The native implementation is part of the ordinary ``lpsd_fast`` shared library.
No FFTW symbols, extra build, or native thread pool are needed. ``auto`` falls
back to the explicit NumPy implementation when an older/missing library lacks
the additive ABI; requesting ``native`` instead reports that condition.

The ordinary native path retains NumPy's anchored mean and its existing output
reduction. Only independent elementwise operations are fused, with contraction
disabled by the package build. ``log_power`` is an optional further reduction:
it preserves the estimator but may differ in summation rounding.
"""
from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np


_DP = ct.POINTER(ct.c_double)
_I64P = ct.POINTER(ct.c_int64)


def _output(array, size, name="output"):
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype("float64"):
        raise ValueError(f"{name} must be a float64 ndarray")
    if array.ndim != 1 or len(array) != size or not array.flags.c_contiguous or not array.flags.writeable:
        raise ValueError(f"{name} must be a writable contiguous vector of length {size}")
    return array


class FFTWOperations:
    """Reusable dispatcher; ``backend`` records the actual selected backend.

    ``library`` may be the already loaded ``lpsd_fast.api._LIB`` or its path.
    The operation methods neither keep references to input data nor create
    worker threads. Native power evaluation needs no scratch buffer; NumPy
    accepts an optional reusable one.
    """

    def __init__(self, backend="auto", library=None):
        if backend not in ("auto", "native", "numpy"):
            raise ValueError("Operation backend must be auto, native, or numpy")
        self.requested_backend = backend
        self.backend, self.lib = "numpy", None
        self.metadata = {"requested_backend": backend, "backend": "numpy",
                         "abi_version": None, "library": None,
                         "fallback_reason": None, "native_threads": 0,
                         "mean_reduction": "numpy.mean of anchored float64 samples",
                         "elementwise_fp_contract": "off",
                         "grouped_reduction": "optional SIMD; summation rounding may differ"}
        if backend == "numpy":
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
            version = lib.fftw_ops_version
            version.argtypes, version.restype = [], ct.c_int
            if version() != 1:
                raise RuntimeError("Unsupported FFTW-operations native ABI")
            signatures = {
                "fftw_ops_center_window": ([_DP, _DP, ct.c_int64, ct.c_double], ct.c_int),
                "fftw_ops_power": ([_DP, ct.c_int64, ct.c_double, _DP], ct.c_int),
                "fftw_ops_log_power": ([_DP, ct.c_int64, ct.c_double, _I64P,
                                        ct.c_int64, ct.c_double, _DP, _DP], ct.c_int),
            }
            for name, (arguments, result) in signatures.items():
                function = getattr(lib, name)
                function.argtypes, function.restype = arguments, result
            self.lib, self.backend = lib, "native"
            self.metadata.update(backend="native", abi_version=1, library=str(lib._name))
        except (ImportError, OSError, AttributeError, RuntimeError) as exc:
            if backend == "native":
                raise RuntimeError("Native FFTW operations are unavailable; rebuild lpsd_fast") from exc
            self.metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"

    @property
    def native_available(self):
        return self.backend == "native"

    def bind(self, *, time_buffer, window, transform, n, sample_rate,
             window_square_sum, powers=None, scratch=None, cuts=None,
             density=None, band_powers=None):
        """Validate and bind private, fixed buffers once for a prepared plan.

        The plan owner must not resize/rebind these arrays or free an external
        FFTW allocation while the binding is active. Close the binding before
        destroying that allocation. The binding keeps NumPy references alive.
        Window values, normalization and grouping must remain fixed; sample,
        transform, power and output buffer contents are updated normally.
        ``prepare`` validates a new input; an owning pipeline that has already
        checked its input can use the private ``_prepare_validated`` entry.
        All existing unbound methods retain their per-call validation.
        """
        return _BoundFFTWOperations(
            self, time_buffer=time_buffer, window=window, transform=transform,
            n=n, sample_rate=sample_rate, window_square_sum=window_square_sum,
            powers=powers, scratch=scratch, cuts=cuts, density=density,
            band_powers=band_powers)

    def center_window(self, data, window, mean):
        """Subtract a supplied mean and apply a window to an anchored buffer."""
        data = _output(data, len(data), "data")
        window = np.ascontiguousarray(window, dtype=np.float64)
        if window.ndim != 1 or window.shape != data.shape:
            raise ValueError("Window must match the data vector")
        if self.lib is None:
            data -= mean
            data *= window
        elif self.lib.fftw_ops_center_window(data.ctypes.data_as(_DP),
                                             window.ctypes.data_as(_DP), len(data), float(mean)):
            raise ValueError("Invalid native center/window arguments")
        return data

    def prepare(self, samples, window, out):
        """Copy/anchor samples, retain np.mean, then fuse subtraction/window.

        Caller samples may be strided and are never modified. ``out`` must be
        a private contiguous float64 buffer. Returns the anchored sample mean.
        """
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 1 or not len(samples):
            raise ValueError("Expected a nonempty real sample vector")
        out = _output(out, len(samples))
        np.subtract(samples, samples[0], out=out)
        mean = float(np.mean(out))
        self.center_window(out, window, mean)
        return mean

    def power(self, transform, n, sample_rate, window_square_sum, out=None, scratch=None):
        """One-sided float64 density, including the odd/even endpoint rules."""
        transform = np.ascontiguousarray(transform, dtype=np.complex128)
        bins = n // 2 + 1
        if transform.ndim != 1 or len(transform) != bins or n < 2 or sample_rate <= 0 or window_square_sum <= 0:
            raise ValueError("Invalid transform shape or normalization")
        powers = np.empty(bins) if out is None else _output(out, bins)
        scale = 2.0 / (sample_rate * window_square_sum)
        if self.lib is not None:
            if self.lib.fftw_ops_power(transform.ctypes.data_as(_DP), n, scale,
                                       powers.ctypes.data_as(_DP)):
                raise ValueError("Invalid native power arguments")
        else:
            temp = np.empty(bins) if scratch is None else _output(scratch, bins, "scratch")
            np.square(transform.real, out=powers)
            np.square(transform.imag, out=temp)
            powers += temp
            powers *= scale
            powers[0] *= 0.5
            if n % 2 == 0:
                powers[-1] *= 0.5
        return powers

    def log_power(self, transform, n, sample_rate, window_square_sum, cuts,
                  density=None, band_powers=None):
        """Power directly into the original groups; optional rounding variant.

        Native execution does not materialize N/2 powers. The NumPy fallback
        performs exactly the baseline power calculation and reduceat grouping.
        """
        transform = np.ascontiguousarray(transform, dtype=np.complex128)
        cuts = np.ascontiguousarray(cuts, dtype=np.int64)
        bins, groups = n // 2 + 1, len(cuts) - 1
        if (transform.ndim != 1 or len(transform) != bins or n < 2 or
                sample_rate <= 0 or window_square_sum <= 0 or
                cuts.ndim != 1 or groups < 1 or cuts[0] != 1 or cuts[-1] != bins or
                np.any(np.diff(cuts) <= 0)):
            raise ValueError("Cuts must partition all positive FFT bins exactly once")
        density = np.empty(groups) if density is None else _output(density, groups, "density")
        band_powers = np.empty(groups) if band_powers is None else _output(band_powers, groups, "band_powers")
        if self.lib is None:
            powers = self.power(transform, n, sample_rate, window_square_sum)
            sums = np.add.reduceat(powers, cuts[:-1])
            np.divide(sums, np.diff(cuts), out=density)
            np.multiply(sums, sample_rate / n, out=band_powers)
        elif self.lib.fftw_ops_log_power(transform.ctypes.data_as(_DP), n,
                                         2.0 / (sample_rate * window_square_sum),
                                         cuts.ctypes.data_as(_I64P), groups, sample_rate / n,
                                         density.ctypes.data_as(_DP), band_powers.ctypes.data_as(_DP)):
            raise ValueError("Invalid native grouped power arguments")
        return density, band_powers


class _BoundFFTWOperations:
    """Private fixed-buffer dispatcher; inputs to the plan remain per-call.

    Binding retains the same NumPy mean and (when selected) reduceat sums.
    Cached pointers address only arrays owned/controlled by the prepared plan,
    never a previous caller input. It does not execute or own an FFTW plan.
    """

    def __init__(self, operations, *, time_buffer, window, transform, n,
                 sample_rate, window_square_sum, powers, scratch, cuts,
                 density, band_powers):
        if (not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
                or n < 2 or sample_rate <= 0 or window_square_sum <= 0):
            raise ValueError("Invalid bound transform shape or normalization")
        self.n, self.bins = int(n), int(n) // 2 + 1
        self.time_buffer = _output(time_buffer, self.n, "time_buffer")
        # No silent copies here: these are precisely the owning plan's arrays.
        for array, size, dtype, name in (
                (window, self.n, "float64", "window"),
                (transform, self.bins, "complex128", "transform")):
            if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype)
                    or array.ndim != 1 or len(array) != size
                    or not array.flags.c_contiguous or not array.flags.aligned):
                raise ValueError(f"Bound {name} must be a contiguous aligned {dtype} vector of length {size}")
        self.window, self.transform = window, transform
        self.powers = None if powers is None else _output(powers, self.bins, "powers")
        self.scratch = None if scratch is None else _output(scratch, self.bins, "scratch")
        self.cuts = self._starts = self._counts = None
        self.density = self.band_powers = None
        if cuts is not None:
            if (not isinstance(cuts, np.ndarray) or cuts.dtype != np.dtype("int64")
                    or cuts.ndim != 1 or len(cuts) < 2 or not cuts.flags.c_contiguous
                    or not cuts.flags.aligned or cuts[0] != 1 or cuts[-1] != self.bins):
                raise ValueError("Bound cuts must partition all positive FFT bins exactly once")
            counts = np.diff(cuts)
            if np.any(counts <= 0):
                raise ValueError("Bound cuts must partition all positive FFT bins exactly once")
            self.cuts, self._starts, self._counts = cuts, cuts[:-1], counts
            groups = len(counts)
            if density is not None or band_powers is not None:
                self.density = _output(density, groups, "density")
                self.band_powers = _output(band_powers, groups, "band_powers")
        elif density is not None or band_powers is not None:
            raise ValueError("Bound grouped outputs require cuts")
        buffers = [array for array in (self.time_buffer, self.window, self.transform,
                                        self.powers, self.scratch, self.density,
                                        self.band_powers) if array is not None]
        if any(not array.flags.aligned for array in buffers):
            raise ValueError("Bound buffers must be aligned for their dtype")
        for j, array in enumerate(buffers):
            if any(np.shares_memory(array, other) for other in buffers[:j]):
                raise ValueError("Bound work buffers must not overlap")
        self._operations = operations
        self.backend = operations.backend
        self._closed = False
        self._scale = 2.0 / (sample_rate * window_square_sum)
        self._df = sample_rate / n
        self._center = self._power = self._log_power = None
        self._center_args = self._power_args = self._log_args = None
        if operations.lib is not None:
            lib = operations.lib
            self._center = lib.fftw_ops_center_window
            self._power = lib.fftw_ops_power
            self._log_power = lib.fftw_ops_log_power
            self._center_args = (time_buffer.ctypes.data_as(_DP),
                                 window.ctypes.data_as(_DP), self.n)
            transform_pointer = transform.ctypes.data_as(_DP)
            if self.powers is not None:
                self._power_args = (transform_pointer, self.n, self._scale,
                                    self.powers.ctypes.data_as(_DP))
            if self.density is not None:
                self._log_args = (transform_pointer, self.n, self._scale,
                                  cuts.ctypes.data_as(_I64P), len(self._counts), self._df,
                                  self.density.ctypes.data_as(_DP),
                                  self.band_powers.ctypes.data_as(_DP))
        elif self.powers is not None and self.scratch is None:
            self.scratch = np.empty_like(self.powers)
        self.metadata = {"private_buffers_bound": True,
                         "buffer_validation": "construction",
                         "input_validation": "per-call by owning pipeline",
                         "cached_native_pointers": operations.lib is not None,
                         "cached_group_counts": cuts is not None}

    def prepare(self, samples):
        """Checked public convenience entry for a new input."""
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 1 or len(values) != self.n:
            raise ValueError("Expected one real signal matching the bound length")
        return self._prepare_validated(values)

    def _prepare_validated(self, values):
        """The owning pipeline has already checked dtype, dimensionality, N."""
        if self._closed:
            raise RuntimeError("FFTW operation binding is closed")
        out = self.time_buffer
        np.subtract(values, values[0], out=out)
        mean = float(np.mean(out))
        if self._center is None:
            out -= mean
            out *= self.window
        elif self._center(*self._center_args, mean):
            raise RuntimeError("Bound native center/window operation failed")
        return mean

    def power(self):
        if self._closed or self.powers is None:
            raise RuntimeError("No active power output is bound")
        if self._power is not None:
            if self._power(*self._power_args):
                raise RuntimeError("Bound native power operation failed")
        else:
            np.square(self.transform.real, out=self.powers)
            np.square(self.transform.imag, out=self.scratch)
            self.powers += self.scratch
            self.powers *= self._scale
            self.powers[0] *= 0.5
            if self.n % 2 == 0:
                self.powers[-1] *= 0.5
        return self.powers

    def aggregate_power(self):
        """Unchanged reduceat arithmetic, with fixed grouping checked once."""
        if self._closed or self.powers is None or self.cuts is None:
            raise RuntimeError("No active power grouping is bound")
        sums = np.add.reduceat(self.powers, self._starts)
        return sums / self._counts, sums * self._df

    def log_power(self):
        if self._closed or self.density is None:
            raise RuntimeError("No active direct grouped output is bound")
        if self._log_power is not None:
            if self._log_power(*self._log_args):
                raise RuntimeError("Bound native grouped power operation failed")
        else:
            self.power()
            density, bands = self.aggregate_power()
            self.density[:] = density
            self.band_powers[:] = bands
        return self.density, self.band_powers

    def close(self):
        """Invalidate pointers before the owning FFT plan frees its buffers."""
        self._closed = True
        self._center_args = self._power_args = self._log_args = None
        self.time_buffer = self.window = self.transform = None
        self.powers = self.scratch = self.cuts = self._starts = self._counts = None
        self.density = self.band_powers = None
