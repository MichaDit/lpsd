#!/usr/bin/env python3
"""Serial, complete-call comparisons against an independently built baseline.

The two FFTW estimators retain their original definitions. Timings exclude
signal synthesis, imports, independent result checks and file output. Input
validation performed by each estimator remains inside its timer. No performance tests
run concurrently. The baseline checkout and native binary are kept separate.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks import bench_fftw as bf
from benchmarks.fftw_comparison_signals import CASES, DESCRIPTIONS, signal, theory
from benchmarks.fftw_comparison_metrics import summarize_noise, tone_metrics
from benchmarks.matched_smoothing import MatchedSmoothing, estimate_once
import lpsd_fast
from lpsd_fast import api


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def measured(function):
    gc_was_enabled = gc.isenabled()
    gc.disable()
    cpu0, wall0 = time.process_time_ns(), time.perf_counter_ns()
    try:
        output = function()
        wall = (time.perf_counter_ns() - wall0) / 1e9
        cpu = (time.process_time_ns() - cpu0) / 1e9
    finally:
        if gc_was_enabled:
            gc.enable()
    return output, {"wall_s": wall, "cpu_s": cpu}


def summarize(rows):
    return {"repetitions": rows,
            "median_wall_s": statistics.median(row["wall_s"] for row in rows),
            "min_wall_s": min(row["wall_s"] for row in rows),
            "max_wall_s": max(row["wall_s"] for row in rows),
            "median_cpu_s": statistics.median(row["cpu_s"] for row in rows)}


def difference(actual, reference):
    actual, reference = np.asarray(actual, float), np.asarray(reference, float)
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise AssertionError("Invalid optimized spectrum shape or nonfinite output")
    error = np.abs(actual - reference)
    positive = np.abs(reference) > 0
    relative = error[positive] / np.abs(reference[positive])
    peak = float(np.max(np.abs(reference)))
    return {"exact": bool(np.array_equal(actual, reference)),
            "max_abs": float(np.max(error)),
            "max_relative_where_reference_nonzero": float(np.max(relative)) if relative.size else 0.0,
            "p99_relative_where_reference_nonzero": float(np.quantile(relative, .99)) if relative.size else 0.0,
            "max_absolute_over_reference_peak": float(np.max(error) / peak) if peak else float(np.max(error)),
            "different_points": int(np.count_nonzero(error)),
            "points": int(actual.size)}


def assert_equivalent(actual, reference, rtol=3e-6):
    """Check implementation error, separately from differences of estimators.

    A scale-relative absolute floor handles nearly zero tone-leakage bins.
    The raw maximum absolute and relative errors are always retained.
    """
    a, b = np.asarray(actual, float), np.asarray(reference, float)
    atol = max(float(np.max(np.abs(b))) * 2e-14, np.finfo(np.float32).tiny * 2)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    return difference(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--baseline-hybrid", type=Path, default=ROOT / "benchmarks/reference/matched_smoothing_20261010.py")
    parser.add_argument("--fftw-library", type=Path)
    parser.add_argument("--threads-library", type=Path)
    parser.add_argument("--mode", choices=("screen", "full", "quality", "timings"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000, 1_000_003, 10_000_000, 30_000_000])
    parser.add_argument("--main-n", type=int, default=1_000_000)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--large-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--operations", choices=("auto", "native", "native_grouped", "numpy"), default="native")
    parser.add_argument("--kernel-cache-mb", type=float, default=1536)
    parser.add_argument("--low-cache-mb", type=float, default=512)
    parser.add_argument("--total-cache-mb", type=float)
    parser.add_argument("--auto-low-cache", action="store_true",
                        help="Use the remaining total cache budget for LPSD coefficients")
    parser.add_argument("--smoothing-workers", type=int, default=4)
    parser.add_argument("--weight-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--bluestein", nargs="*", default=[], metavar="N:M",
                        help="Explicit original N and internal convolution M for selected lengths")
    parser.add_argument("--fftw-threads", nargs="*", default=[], metavar="N:T",
                        help="Explicit optimized FFTW thread count for selected sample lengths; baseline stays at --workers")
    parser.add_argument("--measure-lengths", type=int, nargs="*", default=[], metavar="N",
                        help="MEASURE for reused optimized objects at these lengths; complete fresh calls use ESTIMATE")
    parser.add_argument("--planner-time-limit", type=float, default=1.0)
    parser.add_argument("--baseline-kernel-cache-mb", type=float, default=256)
    parser.add_argument("--skip-fresh", action="store_true")
    parser.add_argument("--skip-lpsd-timings", action="store_true")
    args = parser.parse_args()
    args.baseline_repo = args.baseline_repo.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        bluestein_lengths = {int(pair.split(":")[0]): int(pair.split(":")[1])
                             for pair in args.bluestein if len(pair.split(":")) == 2}
        if len(bluestein_lengths) != len(args.bluestein):
            raise ValueError
        fftw_thread_counts = {int(pair.split(":")[0]): int(pair.split(":")[1])
                              for pair in args.fftw_threads if len(pair.split(":")) == 2}
        if len(fftw_thread_counts) != len(args.fftw_threads) or any(t < 1 for t in fftw_thread_counts.values()):
            raise ValueError
    except (ValueError, IndexError):
        parser.error("--bluestein and --fftw-threads require distinct integer N:M or N:T pairs; T must be positive")
    if args.auto_low_cache and args.total_cache_mb is None:
        parser.error("--auto-low-cache requires --total-cache-mb")
    old_package = load_module("lpsd_fast_baseline", args.baseline_repo / "lpsd_fast/__init__.py", package=True)
    old_api = sys.modules["lpsd_fast_baseline.api"]
    old_bf = load_module("fftw_baseline", args.baseline_repo / "benchmarks/bench_fftw.py")
    old_hybrid = load_module("matched_smoothing_baseline", args.baseline_hybrid.resolve())
    old_hybrid._fftw, old_hybrid._lpsd_api = old_bf, old_api
    api._native()
    old_api._native()
    backend = bf.FFTWLibrary(args.fftw_library, args.threads_library, args.workers)
    optimized_backends = {args.workers: backend}
    for threads in sorted(set(fftw_thread_counts.values()) - {args.workers}):
        optimized_backends[threads] = bf.FFTWLibrary(args.fftw_library, args.threads_library, threads)
    provider = bf.KaiserWindow("native", Path(api._LIB._name))
    old_provider = old_bf.KaiserWindow("native", Path(old_api._LIB._name))
    common = dict(sample_rate=args.sample_rate, psll=200., n_frequencies=1000, n_averages=100)
    lkwargs = dict(**common, window_function="kaiser", detrending_order=0,
                   workers=args.workers, kernel="fast", max_working_mb=4096)
    rng_order = np.random.default_rng(202610102)

    def fft_options(n, prepared=True):
        options = (dict(fft_algorithm="bluestein", fft_convolution_length=bluestein_lengths[n])
                   if n in bluestein_lengths else {})
        if prepared and n in args.measure_lengths:
            options.update(planner="measure", time_limit=args.planner_time_limit)
        else:
            options.update(planner="estimate", time_limit=None)
        return options

    def optimized_backend(n):
        return optimized_backends[fftw_thread_counts.get(n, args.workers)]

    def matched_options(n, prepared=True):
        return dict(**common, window_provider=provider, workers=args.workers,
                    max_kernel_cache_mb=args.kernel_cache_mb,
                    window_cache_mb=None if args.auto_low_cache else args.low_cache_mb,
                    total_cache_mb=args.total_cache_mb, smoothing_workers=args.smoothing_workers,
                    max_working_mb=2048, operations_backend="native", smoothing_backend="native",
                    weight_dtype=args.weight_dtype, **fft_options(n, prepared))

    def create(method, n, prepared=True):
        if method == "fftw_before":
            return old_bf.Periodogram(backend, n, **common, window_provider=old_provider,
                                      planner="estimate", aggregation="log", output_dtype="float32")
        if method in ("fftw_after", "fftw_native", "fftw_grouped", "fftw_numpy"):
            operation = {"fftw_after": args.operations, "fftw_native": "native",
                         "fftw_grouped": "native_grouped", "fftw_numpy": "numpy"}[method]
            return bf.Periodogram(optimized_backend(n), n, **common, window_provider=provider,
                                  aggregation="log", output_dtype="float32",
                                  operations=operation, operations_library=api._LIB, **fft_options(n, prepared))
        if method == "matched_before":
            return old_hybrid.MatchedSmoothing(
                backend, n, **common, window_provider=old_provider, workers=args.workers,
                max_kernel_cache_mb=args.baseline_kernel_cache_mb, max_working_mb=2048)
        if method == "matched_after":
            return MatchedSmoothing(optimized_backend(n), n, **matched_options(n, prepared))
        raise ValueError(method)

    def compute(obj, method, samples, outputs):
        if method == "fftw_before":
            if outputs == "both":
                obj.density = "psd"
                psd = obj.compute(samples)
                obj.density = "nsd"
                nsd = obj.compute(samples)
                return pd.DataFrame({"psd": psd.psd, "nsd": nsd.nsd})
            obj.density = outputs
            return obj.compute(samples)
        return obj.compute(samples, outputs=("psd", "nsd") if outputs == "both" else outputs)

    def fresh(method, samples, outputs):
        backend.forget_wisdom()
        if method == "matched_once":
            options = matched_options(len(samples), prepared=False)
            # No cached smoothing dots need a separate worker tuning here.
            options["smoothing_workers"] = args.workers
            selected = ("psd", "nsd") if outputs == "both" else outputs
            return estimate_once(optimized_backend(len(samples)), samples, outputs=selected, **options)
        with create(method, len(samples), prepared=False) as obj:
            return compute(obj, method, samples, outputs)

    sources = [Path(__file__).resolve(), ROOT / "benchmarks/bench_fftw.py",
               ROOT / "benchmarks/_fftw_native.py", ROOT / "benchmarks/matched_smoothing.py",
               ROOT / "benchmarks/_fftw_bluestein.py",
               ROOT / "lpsd_fast/prepared.py", ROOT / "lpsd_fast/api.py",
               *sorted((ROOT / "lpsd_fast/_native").glob("*.c")),
               *sorted((ROOT / "lpsd_fast/_native").glob("*.h"))]
    cpu_path = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip() for line in cpu_path.read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if cpu_path.exists() else platform.processor()
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "baseline": {"commit": command(["git", "rev-parse", "HEAD"], args.baseline_repo),
                     "version": old_package.__version__, "native_binary": str(old_api._LIB._name),
                     "native_sha256": sha256(old_api._LIB._name),
                     "periodogram_sha256": sha256(args.baseline_repo / "benchmarks/bench_fftw.py"),
                     "hybrid_sha256": sha256(args.baseline_hybrid)},
        "optimized": {"base_commit": command(["git", "rev-parse", "HEAD"]),
                      "branch": command(["git", "branch", "--show-current"]),
                      "version": lpsd_fast.__version__, "native_binary": str(api._LIB._name),
                      "native_sha256": sha256(api._LIB._name),
                      "sources": {str(path.relative_to(ROOT)): sha256(path) for path in sources}},
        "machine": {"cpu": cpu, "platform": platform.platform(), "python": platform.python_version(),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
                    "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip() if Path("/sys/fs/cgroup/memory.max").exists() else None,
                    "thread_environment": {key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "fftw": {"version": backend.version, "library": backend.evidence,
                 "threads_library": backend.threads_evidence},
        "timing_scope": {
            "reuse": "Complete data-dependent compute, validation, detrending, window, FFT, powers, smoothing, low LPSD subset and owning DataFrame; prepared allocations/plans/coefficients excluded.",
            "fresh": "Forget FFTW wisdom, construct the complete object, compute and release; all window, kernel and cached coefficient preparation included.",
            "matched_once": "Complete estimate_once API, with both permanent caches disabled and uncached normalizers calculated together with the first numerator; construct, compute, release and metadata included.",
            "planner_selection": "All complete fresh calls use ESTIMATE. Explicit --measure-lengths only select bounded MEASURE for reused optimized objects; its observed preparation cost is reported separately.",
            "lpsd": "Complete public LPSD/NSD API using the pinned original native binary, including all preparation.",
            "paired": "Serial randomized balanced rounds, at most one pair of prepared estimators retained at a time; no concurrent benchmark jobs.",
            "both": "The previous basic FFTW API needs two complete calls for PSD and NSD; optimized returns both after one transform. The previous hybrid already supports a combined call.",
            "excluded": "Imports, shared-library loading, synthesis, warmups, independent result validation, hashing and file output. Each estimator's input validation remains timed.",
            "limits": "Observed min/max are not confidence intervals. CPU clocks and other tenants on the virtual host are uncontrolled."},
        "timings": [], "cases": [], "profiles": [], "validation": [], "preparation": []}

    def checkpoint():
        write_json(args.output, report)

    if args.mode in ("full", "quality"):
        spectra = {}
        with ExitStack() as stack:
            methods = ("fftw_before", "fftw_after", "matched_before", "matched_after")
            objects = {method: stack.enter_context(create(method, args.main_n)) for method in methods}
            for case in CASES:
                print("QUALITY", case, flush=True)
                x, tones = signal(case, args.main_n, args.sample_rate, args.seed)
                fingerprint = bf.digest_array(x)
                reference = old_package.lpsd(x, **lkwargs, outputs=("psd", "nsd", "enbw"))
                f = reference.index.to_numpy().copy()
                spectra[case + "_f"] = f
                spectra[case + "_lpsd_psd"] = reference.psd.to_numpy().copy()
                spectra[case + "_lpsd_nsd"] = reference.nsd.to_numpy().copy()
                frames = {}
                for method, obj in objects.items():
                    frame = compute(obj, method, x, "both")
                    np.testing.assert_array_equal(frame.index.to_numpy(), f)
                    frames[method] = frame
                    for output in ("psd", "nsd"):
                        values = frame[output].to_numpy().copy()
                        assert np.isfinite(values).all() and np.all(values >= 0)
                        spectra[case + "_" + method + "_" + output] = values
                    # A later call must not overwrite a returned DataFrame.
                    retained = frame.to_numpy().copy()
                    compute(obj, method, x * .5, "psd")
                    np.testing.assert_array_equal(frame.to_numpy(), retained)
                    repeated = compute(obj, method, x, "both")
                    np.testing.assert_array_equal(repeated.to_numpy(), retained)
                assert bf.digest_array(x) == fingerprint
                changes = {method: {output: assert_equivalent(frames[method][output], frames[before][output])
                                    for output in ("psd", "nsd")}
                           for method, before in (("fftw_after", "fftw_before"), ("matched_after", "matched_before"))}
                once_frame = fresh("matched_once", x, "both")
                np.testing.assert_array_equal(once_frame.index.to_numpy(), f)
                changes["matched_once"] = {output: assert_equivalent(once_frame[output], frames["matched_before"][output])
                                            for output in ("psd", "nsd")}
                expected = theory(case, f, args.sample_rate)
                metrics = {}
                for method, frame in {"lpsd": reference, **frames}.items():
                    entry = {}
                    if expected is not None:
                        excluded = np.zeros(len(f), dtype=bool)
                        for tone in tones:
                            excluded |= np.abs(f - tone["frequency_hz"]) <= 3 * reference.enbw.to_numpy().real
                        entry["noise"] = summarize_noise(f, frame.nsd, expected, excluded)
                        mask = (f >= .05) & (f <= 20) & ~excluded
                        entry["normalized_nsd_std_005_to_20_hz"] = float(np.std(frame.nsd.to_numpy()[mask] / np.sqrt(expected[mask]), ddof=1))
                    if case == "offbin_tone":
                        entry["peak"] = tone_metrics(f, frame.psd, tones[0]["frequency_hz"], tones[0]["rms_v"])
                        far = (f > .01) & (f < 20) & (np.abs(f - tones[0]["frequency_hz"]) > 1)
                        entry["peak"]["far_from_tone_nsd_max"] = float(np.max(frame.nsd.to_numpy()[far]))
                    metrics[method] = entry
                report["cases"].append({"case": case, "description": DESCRIPTIONS[case],
                                        "n": len(x), "input_sha256": fingerprint, "tones": tones,
                                        "implementation_difference": changes, "metrics": metrics,
                                        "input_unchanged": True, "output_ownership_and_repeatability": True})
                checkpoint()
            for method, obj in objects.items():
                report["preparation"].append({"n": args.main_n, "method": method,
                                              "metadata": getattr(obj, "metadata", {}),
                                              "setup_phases": getattr(obj, "setup_phases", {})})
        np.savez_compressed(args.output.with_suffix(".spectra.npz"), **spectra)
        gc.collect()

    if args.mode in ("full", "timings", "screen"):
        for n in args.sizes:
            repetitions = args.repeats if n < 10_000_000 else args.large_repeats
            x, _ = signal("white", n, args.sample_rate, args.seed)
            fingerprint = bf.digest_array(x)
            pairs = [("fftw_before", "fftw_after"), ("matched_before", "matched_after")]
            if args.mode == "screen":
                pairs = [("fftw_before", "fftw_native"), ("fftw_before", "fftw_grouped"),
                         ("matched_before", "matched_after")]
            for methods in pairs:
                print("TIMING", n, "/".join(methods), flush=True)
                with ExitStack() as stack:
                    objects = {}
                    reference_frames = {}
                    for method in methods:
                        backend.forget_wisdom()
                        obj, setup_time = measured(lambda method=method: create(method, n))
                        objects[method] = stack.enter_context(obj)
                        report["preparation"].append({"n": n, "method": method,
                            "observed_setup": setup_time, "metadata": getattr(obj, "metadata", {}),
                            "setup_phases": getattr(obj, "setup_phases", {})})
                        reference_frames[method] = compute(obj, method, x, "both")
                        compute(obj, method, x, "both")
                    before, after = methods
                    for output in ("psd", "nsd"):
                        report["validation"].append({"n": n, "method": after, "output": output,
                            **assert_equivalent(reference_frames[after][output], reference_frames[before][output])})
                    outputs = ("nsd",) if args.mode == "screen" else ("psd", "nsd", "both")
                    rows = {(method, output): [] for method in methods for output in outputs}
                    for repetition in range(repetitions):
                        jobs = list(rows)
                        rng_order.shuffle(jobs)
                        for method, output in jobs:
                            frame, timing = measured(lambda: compute(objects[method], method, x, output))
                            timing["round"] = repetition
                            rows[(method, output)].append(timing)
                            columns = ("psd", "nsd") if output == "both" else (output,)
                            for column in columns:
                                np.testing.assert_array_equal(frame[column], reference_frames[method][column])
                    for (method, output), times in rows.items():
                        entry = {"n": n, "method": method, "mode": "reuse", "outputs": output, **summarize(times)}
                        report["timings"].append(entry)
                        print(" ", method, output, round(entry["median_wall_s"] * 1000, 4), "ms", flush=True)
                    for method, obj in objects.items():
                        if method.startswith("fftw_"):
                            frame = obj.compute(x, profile=True)
                            report["profiles"].append({"n": n, "method": method, "profile": obj.last_profile})
                        elif hasattr(obj, "profile"):
                            frame = obj.profile(x, outputs="nsd")
                            report["profiles"].append({"n": n, "method": method, "profile": obj.last_profile})
                gc.collect()
                checkpoint()
                if args.mode != "screen" and not args.skip_fresh:
                    fresh_methods = (*methods, "matched_once") if "matched_after" in methods else methods
                    rows = {(method, output): [] for method in fresh_methods for output in ("psd", "nsd", "both")}
                    # No large prepared comparison objects remain alive during
                    # fresh measurements, keeping the memory budget meaningful.
                    del objects, reference_frames, obj
                    gc.collect()
                    for repetition in range(repetitions):
                        jobs = list(rows)
                        rng_order.shuffle(jobs)
                        for method, output in jobs:
                            _, timing = measured(lambda: fresh(method, x, output))
                            timing["round"] = repetition
                            rows[(method, output)].append(timing)
                    for (method, output), times in rows.items():
                        entry = {"n": n, "method": method, "mode": "fresh", "outputs": output, **summarize(times)}
                        report["timings"].append(entry)
                        print(" ", method, output, "fresh", round(entry["median_wall_s"] * 1000, 4), "ms", flush=True)
                    checkpoint()
            if args.mode != "screen" and not args.skip_lpsd_timings:
                print("TIMING", n, "lpsd", flush=True)
                for output in ("psd", "nsd"):
                    function = lambda output=output: old_package.lpsd(x, **lkwargs, outputs=output)
                    function()
                    times = [measured(function)[1] for _ in range(repetitions)]
                    entry = {"n": n, "method": "lpsd", "mode": "complete", "outputs": output, **summarize(times)}
                    report["timings"].append(entry)
                    print(" ", output, round(entry["median_wall_s"] * 1000, 4), "ms", flush=True)
            assert bf.digest_array(x) == fingerprint
            checkpoint()
            del x
            gc.collect()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["max_rss_bytes"] = bf.max_rss_bytes()
    checkpoint()
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
