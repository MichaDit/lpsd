#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Alternating complete-call full-C2C/packed-real comparison on one machine.

Uses the same FFTW binary, input, output grid, resource allowances and native
library for both algorithms. Signal creation and hashes are outside each
timer. Fresh planning, preprocessing, both spectra and workspace cleanup are
inside. No speed threshold is a test; all-point numerical failures fail the
run. This small performance screen creates no remote reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def differences(reference, candidate):
    import numpy as np
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"passed": False, "shape_or_finite_failure": True}
    absolute = np.abs(b - a)
    nonzero = np.abs(a) > 0
    relative = np.zeros_like(absolute)
    np.divide(absolute, np.abs(a), out=relative, where=nonzero)
    zero_mismatch = int(np.count_nonzero((~nonzero) & (absolute > 0)))
    return {
        "passed": bool(np.all(absolute <= .01 * np.abs(a))),
        "user_relative_limit": .01, "absolute_tolerance": 0.,
        "strict_3ppm_passed": bool(np.all(absolute <= 3e-6 * np.abs(a))),
        "bit_identical": reference.tobytes() == candidate.tobytes(),
        "maximum_absolute_error": float(absolute.max(initial=0.)),
        "maximum_relative_error_nonzero": float(relative.max(initial=0.)),
        "reference_zero_mismatches": zero_mismatch,
        "points": len(a),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000_000])
    parser.add_argument("--methods", choices=("fftw", "matched"), nargs="+", default=["fftw", "matched"])
    parser.add_argument("--signals", nargs="+", default=["white"],
                        choices=("white", "pink", "brown", "mixed_tones", "offbin_tone", "dc_nanovolt"))
    parser.add_argument("--point-factors", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fft-memory-mib", type=float, default=64.)
    parser.add_argument("--native-working-mib", type=float, default=128.)
    parser.add_argument("--max-seconds", type=float, default=600.)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (min(args.sizes) < 4 or args.threads < 1 or args.repeats < 1 or
            any(value < 1 for value in args.point_factors) or
            any(not math.isfinite(value) or value <= 0 for value in
                (args.fft_memory_mib, args.native_working_mib, args.max_seconds))):
        parser.error("Positive sizes, workers, repeats, factors, budgets and deadline required")
    import numpy as np
    from benchmarks import bench_fftw as bf
    from benchmarks.bench_fftw_disk import disk_pipeline, input_file
    from benchmarks.fftw_comparison_signals import signal
    from lpsd_benchmark.resources import detect_resources, MIB
    from lpsd_benchmark.worker import _load_fftw, resolve_point_plan
    from lpsd_fast import api

    api._native()
    requested_directory = args.work_dir.resolve() if args.work_dir else None
    if requested_directory:
        requested_directory.mkdir(parents=True, exist_ok=True)
    backend = _load_fftw(bf, args.threads)
    report = {
        "schema": "lpsd-disk-algorithm-pairs-v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(), "machine": platform.machine(),
        "python": platform.python_version(), "numpy": np.__version__,
        "requested_threads": args.threads, "actual_fftw_threads": backend.threads,
        "fftw_version": backend.version, "fftw_sha256": backend.evidence.get("binary_sha256"),
        "native_sha256": hashlib.sha256(Path(api._LIB._name).read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "benchmarks/bench_disk_algorithms.py", "benchmarks/_fftw_disk.py",
            "benchmarks/_fftw_packed_real.py", "benchmarks/bench_fftw_disk.py",
            "benchmarks/_lpsd_disk.py", "benchmarks/matched_smoothing.py")},
        "settings": {key: value for key, value in vars(args).items() if key not in ("output", "work_dir")},
        "scope": "Alternating fresh complete PSD+NSD calls on the same input. All planning, "
                 "preprocessing, bounded diagnostics and cleanup timed. Signal generation/hash, "
                 "imports and library loading excluded. Same machine and dependency cohort. "
                 "Small CI timings do not measure out-of-RAM storage performance. Deadline is "
                 "cooperative between calls; ordinary portable runner has process supervision.",
        "cases": [],
    }
    deadline = time.perf_counter() + args.max_seconds
    write_report(args.output, report)
    failed = False
    try:
        with tempfile.TemporaryDirectory(prefix="lpsd-disk-pairs-", dir=requested_directory) as name:
            directory = Path(name)
            resources = detect_resources(directory, threads=args.threads)
            threads = min(args.threads, resources.threads)
            if threads != args.threads:
                backend = _load_fftw(bf, threads)
                report["actual_fftw_threads"] = backend.threads
            report["admitted_native_workers"] = threads
            for n in args.sizes:
                for signal_name in args.signals:
                    ram_need = int((args.fft_memory_mib + args.native_working_mib) * MIB) + 256 * MIB
                    if signal_name != "white":
                        ram_need += 72 * n
                    if ram_need > resources.ram_budget or 28 * n + 256 * MIB > resources.disk_budget:
                        report["cases"].append({"n": n, "signal": signal_name, "status": "skipped_resource_budget"})
                        failed = True
                        continue
                    path = directory / "input.f64"
                    if signal_name == "white":
                        input_file(path, n, 50., 20261010)
                    else:
                        values, _ = signal(signal_name, n, 50., 20261010)
                        mapped = np.memmap(path, dtype=np.float64, mode="w+", shape=(n,))
                        mapped[:] = values
                        del values
                        mapped.flush()
                        mapped._mmap.close()
                        del mapped
                    values = np.memmap(path, dtype=np.float64, mode="r", shape=(n,))
                    fingerprint = bf.digest_array(values)
                    try:
                        for method in args.methods:
                            for factor in args.point_factors:
                                _, plan = resolve_point_plan(api, n, 50., 1000, factor, 1000, 100)
                                row = {"n": n, "method": method, "signal": signal_name,
                                       "point_factor": factor, **plan, "status": "running",
                                       "input_sha256": fingerprint, "pairs": []}
                                report["cases"].append(row)
                                for repeat in range(args.repeats):
                                    frames, samples = {}, {}
                                    order = ("full", "packed-real") if repeat % 2 == 0 else ("packed-real", "full")
                                    for algorithm in order:
                                        if time.perf_counter() >= deadline:
                                            raise TimeoutError("Comparison deadline reached between calls")
                                        options = SimpleNamespace(sample_rate=50., n_frequencies=plan["target_points"],
                                                                  n_averages=100, fft_memory_mb=args.fft_memory_mib,
                                                                  low_working_mb=args.native_working_mib, workers=threads,
                                                                  smoothing_workers=min(4, threads), power_storage="external" if n % 2 else "workspace",
                                                                  disk_fft_algorithm=algorithm)
                                        gc.collect()
                                        wall, cpu = time.perf_counter(), time.process_time()
                                        frame, details = disk_pipeline(method, values, backend, directory, options)
                                        cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
                                        if (not np.isfinite(frame.to_numpy()).all() or
                                                not (frame.to_numpy() >= 0).all()):
                                            raise AssertionError("Nonfinite or negative spectrum")
                                        frames[algorithm] = frame
                                        samples[algorithm] = {"wall_s": wall, "cpu_s": cpu,
                                                              "phases": details["phases"]}
                                    a, b = frames["full"], frames["packed-real"]
                                    same_grid = np.array_equal(a.index.to_numpy(), b.index.to_numpy())
                                    comparison = {column: differences(a[column].to_numpy(), b[column].to_numpy())
                                                  for column in ("psd", "nsd")}
                                    good = same_grid and all(item["passed"] for item in comparison.values())
                                    row["pairs"].append({"repeat": repeat, "order": order, "samples": samples,
                                                         "identical_frequency_grid": same_grid,
                                                         "comparison": comparison, "passed": good})
                                    failed |= not good
                                    write_report(args.output, report)
                                row["median_wall_s"] = {algorithm: statistics.median(
                                    pair["samples"][algorithm]["wall_s"] for pair in row["pairs"])
                                    for algorithm in ("full", "packed-real")}
                                row["median_cpu_s"] = {algorithm: statistics.median(
                                    pair["samples"][algorithm]["cpu_s"] for pair in row["pairs"])
                                    for algorithm in ("full", "packed-real")}
                                row["ratio_of_wall_medians"] = row["median_wall_s"]["full"] / row["median_wall_s"]["packed-real"]
                                row["status"] = "completed" if all(pair["passed"] for pair in row["pairs"]) else "failed_numerical_gate"
                                print(json.dumps({key: row[key] for key in ("n", "method", "signal", "point_factor", "status", "ratio_of_wall_medians")}), flush=True)
                        if bf.digest_array(values) != fingerprint:
                            raise AssertionError("Input changed")
                    finally:
                        values._mmap.close()
                        del values
                        path.unlink(missing_ok=True)
        report["status"] = "failed" if failed else "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(args.output, report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
