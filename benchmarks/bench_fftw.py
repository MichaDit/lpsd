# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark FFTW, a full-record periodogram, and logarithmic power aggregation.

This is a comparison instrument, not an alternative LPSD implementation. The
logarithmic labels come from the actual LPSD plan, but each result averages
whole-record FFT-bin powers. Its filter, resolution and statistical smoothing
differ from LPSD's frequency-dependent, overlapping segments.
"""

import argparse
import ctypes as ct
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd

from lpsd._helpers import _kaiser_alpha, _kaiser_rov, _ltf_plan

try:
    import resource
except ImportError:
    resource = None


DOUBLE_PTR = ct.POINTER(ct.c_double)
PLANNERS = {"estimate": 64, "measure": 0, "patient": 32}


def digest_array(array):
    """Fingerprint without an extra full-size bytes allocation."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(array.dtype.str.encode("ascii") + b"\0")
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def max_rss_bytes():
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value * 1024) if sys.platform.startswith("linux") else None


def library_evidence(library, symbol):
    """Find the binary behind the loaded symbol, including a system soname."""
    filename = Path(str(library._name))
    address = ct.cast(getattr(library, symbol), ct.c_void_p).value
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            lo, hi = (int(x, 16) for x in fields[0].split("-"))
            if lo <= address < hi and len(fields) == 6:
                filename = Path(fields[5])
                break
    resolved = filename.resolve() if filename.is_file() else None
    return {
        "requested": str(library._name),
        "resolved": str(resolved) if resolved else None,
        "binary_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest() if resolved else None,
    }


class FFTWLibrary:
    """The actual double-precision FFTW C API, optionally with POSIX threads."""

    _thread_runtime = None

    def __init__(self, library=None, threads_library=None, threads=1):
        if threads < 1:
            raise ValueError("threads must be positive")
        name = str(library) if library else ctypes.util.find_library("fftw3")
        if not name:
            raise OSError("FFTW double library not found; use --fftw-library")
        self.lib = ct.CDLL(name, mode=getattr(ct, "RTLD_GLOBAL", 0))
        self.threads = threads
        self.threadlib = None
        if threads > 1 or threads_library:
            tname = str(threads_library) if threads_library else ctypes.util.find_library("fftw3_threads")
            if not tname:
                raise OSError("FFTW threads library not found; use --threads-library")
            self.threadlib = ct.CDLL(tname, mode=getattr(ct, "RTLD_GLOBAL", 0))
            self.threadlib.fftw_init_threads.argtypes = []
            self.threadlib.fftw_init_threads.restype = ct.c_int
            self.threadlib.fftw_plan_with_nthreads.argtypes = [ct.c_int]
            self.threadlib.fftw_plan_with_nthreads.restype = None
            if not self.threadlib.fftw_init_threads():
                raise RuntimeError("fftw_init_threads failed")
            FFTWLibrary._thread_runtime = self.threadlib
        signatures = {
            "fftw_malloc": ([ct.c_size_t], ct.c_void_p),
            "fftw_free": ([ct.c_void_p], None),
            "fftw_plan_dft_r2c_1d": ([ct.c_int, DOUBLE_PTR, ct.c_void_p, ct.c_uint], ct.c_void_p),
            "fftw_execute": ([ct.c_void_p], None),
            "fftw_destroy_plan": ([ct.c_void_p], None),
            "fftw_forget_wisdom": ([], None),
            "fftw_set_timelimit": ([ct.c_double], None),
            "fftw_flops": ([ct.c_void_p, DOUBLE_PTR, DOUBLE_PTR, DOUBLE_PTR], None),
            "fftw_sprint_plan": ([ct.c_void_p], ct.c_void_p),
        }
        for symbol, (arguments, returns) in signatures.items():
            function = getattr(self.lib, symbol)
            function.argtypes, function.restype = arguments, returns
        self.version = ct.string_at(ct.addressof(ct.c_char.in_dll(self.lib, "fftw_version"))).decode()
        self.evidence = library_evidence(self.lib, "fftw_execute")
        self.threads_evidence = library_evidence(self.threadlib, "fftw_init_threads") if self.threadlib else None

    def forget_wisdom(self):
        self.lib.fftw_forget_wisdom()


class RealFFT:
    """An aligned, out-of-place r2c plan; caller initializes input after planning."""

    def __init__(self, backend, n, planner="estimate", time_limit=None):
        if n < 2 or n > np.iinfo(np.int32).max:
            raise ValueError("Require 2 <= n <= INT_MAX for FFTW's 1D interface")
        if planner not in PLANNERS:
            raise ValueError("Unknown planner")
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
            raise ValueError("Planner time limit must be finite and nonnegative")
        self.backend, self.n = backend, n
        self.ip = self.op = self.plan = None
        self.x = self.y = None
        self.planner, self.time_limit = planner, time_limit
        lib = backend.lib
        started = time.perf_counter()
        try:
            self.ip = lib.fftw_malloc(n * 8)
            self.op = lib.fftw_malloc((n // 2 + 1) * 16)
            if not self.ip or not self.op:
                raise MemoryError("FFTW aligned allocation failed")
            self.x = np.ctypeslib.as_array(ct.cast(self.ip, DOUBLE_PTR), shape=(n,))
            raw = np.ctypeslib.as_array(ct.cast(self.op, DOUBLE_PTR), shape=(2 * (n // 2 + 1),))
            self.y = raw.view(np.complex128)
            self.allocation_s = time.perf_counter() - started
            thread_runtime = backend.threadlib or FFTWLibrary._thread_runtime
            if thread_runtime:
                thread_runtime.fftw_plan_with_nthreads(backend.threads)
            lib.fftw_set_timelimit(-1.0 if time_limit is None else time_limit)
            started = time.perf_counter()
            self.plan = lib.fftw_plan_dft_r2c_1d(n, ct.cast(self.ip, DOUBLE_PTR), self.op, PLANNERS[planner])
            self.planning_s = time.perf_counter() - started
            if not self.plan:
                raise RuntimeError("FFTW plan creation failed")
        except BaseException:
            self.close()
            raise

    def execute(self):
        if not self.plan:
            raise RuntimeError("Plan is closed")
        self.backend.lib.fftw_execute(self.plan)

    def describe(self):
        add, mul, fma = ct.c_double(), ct.c_double(), ct.c_double()
        self.backend.lib.fftw_flops(self.plan, ct.byref(add), ct.byref(mul), ct.byref(fma))
        # sprint_plan uses ordinary malloc, whereas Windows FFTW builds can
        # implement fftw_free with _aligned_free. That pairing corrupts the
        # heap. An arbitrary DLL's CRT cannot be assumed to match Python's.
        # Omit this optional diagnostic on Windows; numerical/timing work and
        # flop counts still run. Never call sprint_plan and leak its buffer.
        text = b""
        if os.name != "nt":
            pointer = self.backend.lib.fftw_sprint_plan(self.plan)
            try:
                text = ct.string_at(pointer) if pointer else b""
            finally:
                if pointer:
                    self.backend.lib.fftw_free(pointer)
        return {
            "planner": self.planner, "flags": PLANNERS[self.planner],
            "time_limit_s": self.time_limit, "planning_s": self.planning_s,
            "aligned_allocation_s": self.allocation_s,
            "flops": {"add": add.value, "mul": mul.value, "fma": fma.value,
                      "total_fma_as_two": add.value + mul.value + 2 * fma.value},
            "plan_text": text[:8192].decode(errors="replace"),
            "plan_text_available": os.name != "nt",
            "plan_text_truncated": len(text) > 8192,
            "plan_text_sha256": hashlib.sha256(text).hexdigest(),
        }

    def close(self):
        # Views must not be accessed after releasing their underlying C memory.
        self.x = self.y = None
        if self.plan:
            self.backend.lib.fftw_destroy_plan(self.plan)
            self.plan = None
        if self.ip:
            self.backend.lib.fftw_free(self.ip)
            self.ip = None
        if self.op:
            self.backend.lib.fftw_free(self.op)
            self.op = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class KaiserWindow:
    """Use fast.2's generator and sequential sums, with an explicit fallback."""

    def __init__(self, backend="auto", library=None):
        self.lib = None
        self.metadata = {"requested_backend": backend, "backend": "numpy", "fallback_reason": None}
        if backend not in ("auto", "native", "numpy"):
            raise ValueError("Unknown window backend")
        if backend != "numpy":
            try:
                import lpsd_fast
                path = Path(library) if library else Path(lpsd_fast.__file__).parent / "_native" / ("liblpsd_fast.dll" if os.name == "nt" else "liblpsd_fast.so")
                lib = ct.CDLL(str(path))
                lib.generate_kaiser_series.argtypes = [DOUBLE_PTR, ct.c_long, ct.c_double]
                lib.generate_kaiser_series.restype = ct.c_int
                lib.window_sums.argtypes = [DOUBLE_PTR, ct.c_long, DOUBLE_PTR, DOUBLE_PTR]
                lib.window_sums.restype = ct.c_int
                self.lib = lib
                self.metadata.update(backend="lpsd_fast_native_series", library=library_evidence(lib, "generate_kaiser_series"))
            except (ImportError, OSError, AttributeError) as exc:
                if backend == "native":
                    raise RuntimeError("Requested fast.2 native Kaiser generator is unavailable") from exc
                self.metadata["fallback_reason"] = str(exc)

    def generate(self, n, beta):
        if self.lib:
            window = np.empty(n, dtype=np.float64)
            if self.lib.generate_kaiser_series(window.ctypes.data_as(DOUBLE_PTR), n, beta):
                raise ValueError("Native Kaiser generation failed")
            return window
        return np.kaiser(n + 1, beta)[:-1]

    def sums(self, window):
        if self.lib:
            s1, s2 = ct.c_double(), ct.c_double()
            if self.lib.window_sums(window.ctypes.data_as(DOUBLE_PTR), len(window), ct.byref(s1), ct.byref(s2)):
                raise ValueError("Native window sums failed")
            return s1.value, s2.value
        return float(np.sum(window)), float(np.sum(window * window))


def partitions(n, sample_rate, frequencies):
    """Whole positive FFT bins assigned using arithmetic label midpoints.

    Density is the mean of bin densities. Its integration widths are counts*df,
    not differences of the nominal midpoint boundaries. DC stays separate.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.ndim != 1 or not len(frequencies) or not np.all(np.isfinite(frequencies)):
        raise ValueError("Require a nonempty finite frequency vector")
    if np.any(np.diff(frequencies) <= 0) or frequencies[0] <= 0 or frequencies[-1] > sample_rate / 2:
        raise ValueError("Frequency labels must increase within (0, fs/2]")
    df = sample_rate / n
    middle = 0.5 * (frequencies[:-1] + frequencies[1:])
    cuts = np.concatenate(([1], np.ceil(middle / df).astype(np.int64), [n // 2 + 1]))
    counts = np.diff(cuts)
    if np.any(counts <= 0) or counts.sum() != n // 2:
        raise ValueError("Frequency grid produces empty or invalid FFT-bin groups")
    return {
        "frequency_labels_hz": frequencies,
        "nominal_midpoint_boundaries_hz": np.concatenate(([0.0], middle, [sample_rate / 2])),
        "cuts": cuts, "counts": counts, "discrete_bandwidth_hz": counts * df,
    }


def lpsd_grid(n, sample_rate, psll, n_frequencies, n_averages):
    alpha = _kaiser_alpha(psll)
    overlap = _kaiser_rov(alpha)
    f, _, _, _, _ = _ltf_plan(n, sample_rate, overlap, 1, 0, n_frequencies, n_averages)
    result = partitions(n, sample_rate, f)
    result["lpsd_overlap"] = overlap
    return result


def one_sided_psd(transform, n, sample_rate, window_square_sum, out=None, scratch=None):
    """Density from FFTW's unnormalized double r2c result, without padding."""
    if len(transform) != n // 2 + 1 or sample_rate <= 0 or window_square_sum <= 0:
        raise ValueError("Invalid transform shape or normalization")
    powers = np.empty(len(transform)) if out is None else out
    temp = np.empty(len(transform)) if scratch is None else scratch
    np.square(transform.real, out=powers)
    np.square(transform.imag, out=temp)
    powers += temp
    powers *= 2.0 / (sample_rate * window_square_sum)
    powers[0] *= 0.5
    if n % 2 == 0:
        powers[-1] *= 0.5
    return powers


def aggregate_power(psd, cuts, sample_rate, n):
    counts = np.diff(cuts)
    if cuts[0] != 1 or cuts[-1] != len(psd) or np.any(counts <= 0):
        raise ValueError("Cuts must partition all positive FFT bins exactly once")
    sums = np.add.reduceat(psd, cuts[:-1])
    return sums / counts, sums * (sample_rate / n)


class Periodogram:
    """Prepared full-record pipeline. Returned DataFrames own their values."""

    def __init__(self, backend, n, sample_rate=1.0, psll=200.0,
                 n_frequencies=1000, n_averages=100, planner="estimate",
                 time_limit=None, window_provider=None, aggregation="log",
                 density="psd", output_dtype="float32"):
        if aggregation not in ("log", "none") or density not in ("psd", "nsd"):
            raise ValueError("Unknown aggregation or density")
        if output_dtype not in ("float32", "float64"):
            raise ValueError("Output dtype must be float32 or float64")
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Sample rate must be positive and finite")
        self.n, self.sample_rate = n, sample_rate
        self.aggregation, self.density, self.output_dtype = aggregation, density, output_dtype
        self.fft = RealFFT(backend, n, planner, time_limit)
        try:
            provider = window_provider or KaiserWindow()
            self.window_metadata = dict(provider.metadata)
            self.beta = _kaiser_alpha(psll) * np.pi
            t = time.perf_counter()
            self.window = provider.generate(n, self.beta)
            generation_s = time.perf_counter() - t
            t = time.perf_counter()
            self.s1, self.s2 = provider.sums(self.window)
            sums_s = time.perf_counter() - t
            if not math.isfinite(self.s2) or self.s2 <= 0:
                raise ValueError("Invalid window normalization")
            t = time.perf_counter()
            self.grid = lpsd_grid(n, sample_rate, psll, n_frequencies, n_averages) if aggregation == "log" else None
            self.frequencies = self.grid["frequency_labels_hz"] if self.grid else np.fft.rfftfreq(n, 1.0 / sample_rate)
            grid_s = time.perf_counter() - t
            self.powers = np.empty(n // 2 + 1)
            self.scratch = np.empty_like(self.powers)
            self.setup_phases = {"aligned_allocation_s": self.fft.allocation_s,
                                 "fftw_planning_s": self.fft.planning_s,
                                 "window_generation_s": generation_s,
                                 "window_sums_s": sums_s, "grid_s": grid_s}
            self.last_profile = None
            self.last_band_powers = None
        except BaseException:
            self.close()
            raise

    def compute(self, samples, profile=False):
        phases = {}
        tick = time.perf_counter() if profile else 0.0

        def mark(name):
            nonlocal tick
            if profile:
                now = time.perf_counter()
                phases[name], tick = now - tick, now

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 1 or len(values) != self.n:
            raise ValueError("Expected one real signal matching the prepared length")
        mark("input_view_s")
        # Anchor subtraction preserves small signals sitting on a large DC level.
        # All three passes belong to the API timer; caller data are unchanged.
        np.subtract(values, values[0], out=self.fft.x)
        self.fft.x -= np.mean(self.fft.x)
        mark("copy_and_detrend_s")
        self.fft.x *= self.window
        mark("window_multiply_s")
        self.fft.execute()
        mark("fftw_execute_s")
        one_sided_psd(self.fft.y, self.n, self.sample_rate, self.s2, self.powers, self.scratch)
        mark("power_and_normalization_s")
        if self.grid:
            density, self.last_band_powers = aggregate_power(self.powers, self.grid["cuts"], self.sample_rate, self.n)
        else:
            density = self.powers
        mark("log_aggregation_s")
        # Keep double arithmetic until output conversion. For float32 NSD use
        # the same complex64 square-root route as the legacy LPSD output.
        if self.density == "nsd" and self.output_dtype == "float32":
            result = np.sqrt(np.asarray(density, dtype=np.complex64)).real.copy()
        else:
            result = np.array(density, dtype=self.output_dtype, copy=True)
            if self.density == "nsd":
                np.sqrt(result, out=result)
        frame = pd.DataFrame({self.density: result}, index=pd.Index(self.frequencies, name="frequency"), copy=False)
        mark("conversion_and_dataframe_s")
        self.last_profile = phases if profile else None
        return frame

    def validation(self):
        """Checks on the last execution, deliberately outside API timers."""
        df = self.sample_rate / self.n
        expected = float(np.sum(self.fft.x * self.fft.x) / self.s2)
        full = float(np.sum(self.powers) * df)
        dc = float(self.powers[0] * df)
        positive = float(np.sum(self.powers[1:]) * df)
        result = {"weighted_detrended_time_power": expected,
                  "full_periodogram_integrated_power": full,
                  "dc_bin_power": dc, "positive_bin_power": positive,
                  "parseval_absolute_error": abs(full - expected),
                  "finite": bool(np.isfinite(self.powers).all())}
        if self.grid:
            total = float(np.sum(self.last_band_powers))
            result.update(aggregated_positive_power=total,
                          aggregation_absolute_error=abs(total - positive),
                          positive_bins_covered=int(np.sum(self.grid["counts"])),
                          smallest_group_bins=int(np.min(self.grid["counts"])))
        return result

    def close(self):
        self.fft.close()
        self.window = self.powers = self.scratch = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def timed_call(function):
    cpu, wall = time.process_time(), time.perf_counter()
    result = function()
    return result, {"wall_s": time.perf_counter() - wall,
                    "cpu_s": time.process_time() - cpu,
                    "process_max_rss_bytes": max_rss_bytes()}


def frame_fingerprint(frame):
    return {"frequency_sha256": digest_array(frame.index.to_numpy()),
            "output_sha256": digest_array(frame.iloc[:, 0].to_numpy()),
            "frequencies": len(frame), "dtype": str(frame.iloc[:, 0].dtype)}


def cold_call(backend, samples, kwargs, profile=False):
    """Includes wisdom reset, allocation, planning, preprocessing and freeing."""
    backend.forget_wisdom()
    pipeline = Periodogram(backend, len(samples), **kwargs)
    details = {"planning_s": pipeline.fft.planning_s}
    try:
        frame = pipeline.compute(samples, profile=profile)
        if profile:
            details.update(setup_phases=pipeline.setup_phases, execution_phases=pipeline.last_profile)
    finally:
        t = time.perf_counter()
        pipeline.close()
        if profile:
            details["free_s"] = time.perf_counter() - t
    return frame, details


def summarize(rows):
    return {"repetitions": rows,
            "median_wall_s": statistics.median(r["wall_s"] for r in rows),
            "min_wall_s": min(r["wall_s"] for r in rows),
            "median_cpu_s": statistics.median(r["cpu_s"] for r in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cold-repeats", type=int, help="Default: same as --repeats")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--planner", choices=tuple(PLANNERS), default="estimate")
    parser.add_argument("--planner-time-limit", type=float)
    parser.add_argument("--fftw-library", type=Path)
    parser.add_argument("--threads-library", type=Path)
    parser.add_argument("--window-backend", choices=("auto", "native", "numpy"), default="auto")
    parser.add_argument("--lpsd-fast-library", type=Path)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--n-frequencies", type=int, default=1000)
    parser.add_argument("--n-averages", type=int, default=100)
    parser.add_argument("--psll", type=float, default=200.0)
    parser.add_argument("--order", type=int, choices=(0,), default=0)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--aggregation", choices=("log", "none"), default="log")
    parser.add_argument("--density", choices=("psd", "nsd"), default="psd")
    parser.add_argument("--output-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--raw-batch-seconds", type=float, default=0.06)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.n < 16 or args.n > np.iinfo(np.int32).max or args.repeats < 1 or
            args.threads < 1 or args.warmups < 0 or args.n_frequencies < 2 or args.n_averages < 1 or
            (args.cold_repeats is not None and args.cold_repeats < 1)):
        parser.error("Invalid sample/repetition/thread/frequency/average count")
    for name, value, strictly_positive in (("sample rate", args.sample_rate, True),
                                           ("raw batch seconds", args.raw_batch_seconds, False),
                                           ("planner time limit", args.planner_time_limit, False)):
        if value is not None and (not math.isfinite(value) or value < 0 or (strictly_positive and value == 0)):
            parser.error(f"Invalid {name}")
    if not math.isfinite(args.psll):
        parser.error("PSLL must be finite")

    backend = FFTWLibrary(args.fftw_library, args.threads_library, args.threads)
    provider = KaiserWindow(args.window_backend, args.lpsd_fast_library)
    t = time.perf_counter()
    data = np.random.default_rng(args.seed).normal(size=args.n)
    series = pd.Series(data, copy=False)
    input_setup_s = time.perf_counter() - t
    input_hash = digest_array(data)
    kwargs = dict(sample_rate=args.sample_rate, psll=args.psll,
                  n_frequencies=args.n_frequencies, n_averages=args.n_averages,
                  planner=args.planner, time_limit=args.planner_time_limit,
                  window_provider=provider, aggregation=args.aggregation,
                  density=args.density, output_dtype=args.output_dtype)

    backend.forget_wisdom()
    pipeline, setup = timed_call(lambda: Periodogram(backend, args.n, **kwargs))
    try:
        plan = pipeline.fft.describe()
        # Plan creation may overwrite the aligned buffers: initialize afterwards.
        pipeline.fft.x[:] = data
        for _ in range(max(1, args.warmups)):
            pipeline.fft.execute()
        _, probe = timed_call(pipeline.fft.execute)
        batch = max(1, min(20000, int(args.raw_batch_seconds / max(probe["wall_s"], 1e-9))))
        raw_rows = []
        for repeat in range(args.repeats):
            def execute_batch():
                for _ in range(batch):
                    pipeline.fft.execute()
            _, row = timed_call(execute_batch)
            row.update(repeat=repeat, batch_executions=batch,
                       batch_wall_s=row["wall_s"], batch_cpu_s=row["cpu_s"])
            row["wall_s"] /= batch
            row["cpu_s"] /= batch
            raw_rows.append(row)
        raw_hash = digest_array(pipeline.fft.y)
        t = time.perf_counter()
        for _ in range(args.warmups):
            pipeline.compute(series)
        warmup_s = time.perf_counter() - t
        prepared_rows = []
        for repeat in range(args.repeats):
            frame, row = timed_call(lambda: pipeline.compute(series))
            row.update(repeat=repeat, **frame_fingerprint(frame))
            prepared_rows.append(row)
            del frame
        validation = pipeline.validation()
        grid = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pipeline.grid.items()} if pipeline.grid else None
        band_powers = pipeline.last_band_powers.tolist() if pipeline.grid else None
        profile = None
        if args.profile:
            profiled_frame, row = timed_call(lambda: pipeline.compute(series, profile=True))
            profile = {**row, "phases": pipeline.last_profile}
            del profiled_frame
    finally:
        pipeline.close()

    cold_rows = []
    for repeat in range(args.cold_repeats or args.repeats):
        (frame, details), row = timed_call(lambda: cold_call(backend, series, kwargs))
        row.update(repeat=repeat, **details, **frame_fingerprint(frame))
        cold_rows.append(row)
        del frame
    cold_profile = None
    if args.profile:
        (profiled_frame, details), row = timed_call(lambda: cold_call(backend, series, kwargs, profile=True))
        cold_profile = {**row, **details}
        del profiled_frame
    if digest_array(data) != input_hash:
        raise RuntimeError("Benchmark mutated caller input")
    if not validation["finite"] or not math.isclose(validation["weighted_detrended_time_power"], validation["full_periodogram_integrated_power"], rel_tol=2e-12, abs_tol=1e-30):
        raise RuntimeError("Periodogram failed finite/Parseval validation")
    if args.aggregation == "log" and not math.isclose(validation["positive_bin_power"], validation["aggregated_positive_power"], rel_tol=2e-12, abs_tol=1e-30):
        raise RuntimeError("Log aggregation did not preserve positive-bin power")

    report = {
        "schema_version": 1, "backend": "fftw_double_r2c", "parameters": {
            "n": args.n, "sample_rate": args.sample_rate, "seed": args.seed,
            "n_frequencies": args.n_frequencies, "n_averages": args.n_averages,
            "detrending_order": 0, "detrending": "subtract first sample, then mean of centered samples",
            "window": "periodic Kaiser: np.kaiser(n+1, beta)[:-1] semantics",
            "psll": args.psll, "beta": float(_kaiser_alpha(args.psll) * np.pi),
            "threads": args.threads, "planner": args.planner,
            "planner_time_limit_s": args.planner_time_limit,
            "aggregation": args.aggregation, "density": args.density,
            "input_dtype": "float64", "transform_dtype": "complex128",
            "output_dtype": args.output_dtype, "padding_samples": 0,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform(),
                        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "fftw": {"version": backend.version, "library": backend.evidence,
                 "threads_library": backend.threads_evidence, "plan": plan,
                 "plan_scope": "prepared/raw reusable plan; cold calls are separately replanned"},
        "window_generation": provider.metadata,
        "source_evidence": {
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lpsd_planning_source_sha256": hashlib.sha256(Path(_ltf_plan.__code__.co_filename).read_bytes()).hexdigest(),
        },
        "input": {"description": "pandas.Series of seeded float64 standard-normal samples", "sha256": input_hash, "setup_s": input_setup_s, "unchanged": True},
        "prepared_setup": setup,
        "warmup": {"full_length_pipeline_calls": args.warmups, "pipeline_wall_s": warmup_s, "raw_fft_calls": max(1, args.warmups) + 1},
        "raw_fft": {**summarize(raw_rows), "output_sha256": raw_hash},
        "prepared_pipeline": summarize(prepared_rows), "cold_pipeline": summarize(cold_rows),
        "grid": grid, "last_prepared_band_powers": band_powers, "validation": validation,
        "additional_profiled_prepared_call": profile, "additional_profiled_cold_call": cold_profile,
        "notes": [
            "Raw FFT reuses an aligned plan and initialized input/output buffers; it excludes copy, detrending, window, power conversion, grid, DataFrame and allocation. Raw row wall_s/cpu_s are per execution; batch totals and loop counts are also retained.",
            "Prepared pipeline reuses FFTW plan, aligned buffers, window, normalization and grid. All input anchoring, global detrending, window multiplication, FFT, powers, aggregation and output allocation/DataFrame creation are timed.",
            "Cold pipeline includes wisdom reset, malloc, planning, window generation, normalization, grid, pipeline, DataFrame and free. Library loading, input generation and hashes are excluded. Both planner setup and complete cold totals are reported for the chosen planner.",
            "Cold denotes a fresh plan and work buffers, not cold CPU caches, a new process, or unloaded shared libraries.",
            "MEASURE/PATIENT planning can overwrite buffers; input is initialized only after planning. FFTW time limits are approximate, not hard deadlines. ESTIMATE is not implicitly a cached MEASURE plan: wisdom is cleared before each cold setup.",
            "FFTW threads is a maximum set before plan creation. NumPy preprocessing and output conversion remain separate from FFTW parallel execution; record thread environment to detect unrelated BLAS/OpenMP oversubscription.",
            "Density is |DFT|^2/(fs*sum(window^2)); positive non-Nyquist bins have factor 2. DC and the even-N Nyquist bin have factor 1. For odd N the last bin is doubled. Full density integrates using df=fs/N.",
            "Log groups partition all positive FFT bins exactly once and exclude DC. They average power densities before an optional square root. Integrated band power is PSD_mean*count*df. Use discrete_bandwidth_hz, not nominal midpoint widths, for integration.",
            "LPSD frequency labels identify the groups; the full-record FFT window and bin averaging have different spectral transfer, resolution and variance from frequency-dependent segmented LPSD. No pointwise 1% equivalence is asserted.",
            "The nominal first boundary is 0 Hz, but DC is reported separately and excluded from log groups. Global mean removal before windowing does not in general make the windowed DC bin zero.",
            "Validation and saved band powers use internal float64 densities before final output rounding. For float32 NSD, sqrt follows complex64 PSD rounding as in legacy LPSD.",
            "RSS is a process-lifetime high-water mark. Categories in this process share its earlier high-water marks; compare separate invocations for independent memory measurements.",
            "Additional profiles are separate calls. Setup phase timers cover the principal subphases and need not sum exactly to total setup/cold wall time. FFTW-reported flop counts concern the selected transform plan, not the complete estimator.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "n": args.n, "threads": args.threads,
                      "planner": args.planner, "aggregation": args.aggregation,
                      "prepared_median_s": report["prepared_pipeline"]["median_wall_s"],
                      "cold_median_s": report["cold_pipeline"]["median_wall_s"]}))


if __name__ == "__main__":
    main()
