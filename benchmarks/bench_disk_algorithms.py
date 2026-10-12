#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Alternating complete-call full-C2C/packed-real comparison on one machine.

Uses the same FFTW binary, input, output grid, resource allowances and native
library for both algorithms. Signal creation and hashes are outside each
timer. Fresh planning, preprocessing, both spectra and workspace cleanup are
inside. No speed threshold is a test; all-point numerical failures fail the
run. --compare-packed-phase instead isolates the frozen fast.8 phase loop
versus current packed reconstruction and requires bit-identical outputs.
--compare-direct-power compares current materialized versus ABI3 direct-log
reduction after the same original-N full C2C transform. It requires all
returned float32 PSD/NSD values to agree within 3 ppm, with zero absolute
tolerance; bit identity is reported separately.
This small performance screen creates no remote reports.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import inspect
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
        "strict_relative_limit": 3e-6,
        "bit_identical": reference.tobytes() == candidate.tobytes(),
        "maximum_absolute_error": float(absolute.max(initial=0.)),
        "maximum_relative_error_nonzero": float(relative.max(initial=0.)),
        "reference_zero_mismatches": zero_mismatch,
        "points": len(a),
    }


@contextmanager
def packed_phase_variant(variant):
    """Select only the frozen reconstruction method outside the call timer.

    This standalone harness executes serial calls. Restore the private class
    selector even if a numerical check or the pipeline itself raises.
    """
    from benchmarks import bench_fftw_disk as pipeline
    from benchmarks.reference.packed_real_phase_fast8 import LegacyPhasePackedRealDiskFFT
    if variant not in ("legacy-phase", "packed-real"):
        raise ValueError("Unknown packed phase comparison variant")
    original = pipeline.disk_fft_class
    def select(algorithm):
        if algorithm == "packed-real" and variant == "legacy-phase":
            return LegacyPhasePackedRealDiskFFT
        return original(algorithm)
    pipeline.disk_fft_class = select
    try:
        yield
    finally:
        pipeline.disk_fft_class = original


def direct_power_sample(variant, n, output_points, frame, details):
    """Verify the measured full-N route, then retain path evidence without paths."""
    import numpy as np
    if variant not in ("materialized", "direct-log"):
        raise AssertionError("Unknown direct power comparison variant")
    direct = variant == "direct-log"
    fft, storage, phases = details["fft"], details["power_storage"], details["phases"]
    factors = fft["factors"]
    mode = "direct_log_groups" if direct else "workspace"
    if (details["disk_fft_algorithm"] != "full" or fft["n"] != n
            or "inner_n" in fft or math.prod(factors) != n or factors[1] % 2
            or fft["matrix_shape"] != factors
            or fft["memory"]["memory_budget_bytes"] != 128 * 1024**2
            or details["disk_power_reduction"] != variant
            or details["power_vector_created"] is not (not direct)
            or fft["power_storage"] != mode or storage["mode"] != mode
            or storage["maximum_large_file_bytes"] != 24 * n
            or storage["extra_power_file_bytes"] != 0
            or storage["workspace_power_offset_bytes"] != (None if direct else 8 * n)):
        raise AssertionError("Direct power comparison did not execute the required full-N reduction route")
    if (len(frame) != output_points or details["point_plan"]["output_points"] != output_points
            or any(frame[column].dtype != np.dtype("float32") for column in ("psd", "nsd"))):
        raise AssertionError("Direct power comparison requires the planned float32 PSD/NSD outputs")
    same_nsd = np.array_equal(frame["nsd"].to_numpy(),
                             np.sqrt(frame["psd"].to_numpy().astype(np.complex64)).real)
    if not same_nsd:
        raise AssertionError("Returned NSD differs from the current float32 PSD square root")
    parseval = details["parseval"]
    if (not math.isfinite(parseval["relative_error"]) or not 0 <= parseval["relative_error"] <= 1e-10
            or parseval["diagnostic_passes_included_in_timer"] is not True):
        raise AssertionError("Direct power comparison requires the included strict Parseval check")
    evidence = {
        "disk_fft_algorithm": "full", "fft_n": fft["n"], "fft_factors": factors,
        "fft_column_strategy": fft["column_strategy"],
        "fft_memory_budget_bytes": fft["memory"]["memory_budget_bytes"],
        "disk_power_reduction": variant, "power_vector_created": details["power_vector_created"],
        "power_storage": {key: storage[key] for key in ("mode", "maximum_large_file_bytes",
                          "extra_power_file_bytes", "workspace_power_offset_bytes")},
        "output_dtypes": {column: str(frame[column].dtype) for column in ("psd", "nsd")},
        "nsd_matches_sqrt_psd": bool(same_nsd), "parseval": parseval,
        "route_verified": True,
    }
    if direct:
        reduction = details["direct_group_reduction"]
        if (reduction["abi_version"] != 3 or reduction["n"] != n
                or reduction["groups"] != output_points or reduction["matrix_shape"] != factors
                or reduction["matrix_stride_complex"] != factors[1]
                or reduction["energy_bins"] != n // 2 + 1 or reduction["dc_bins"] != 1
                or reduction["nyquist_bins"] != int(n % 2 == 0)
                or reduction["power_vector_created"] is not False or reduction["matrix_written"] is not False
                or reduction["additional_workers"] != 0 or reduction["ordinary_result_arrays"] is not True
                or reduction["array_state_bytes"] != 48 * output_points + 48
                or "direct_log_groups_and_energy_s" not in phases or "ordered_one_sided_power_s" in phases):
            raise AssertionError("Direct power comparison did not execute the bounded ABI3 reducer")
        evidence["direct_group_reduction"] = reduction
    elif ("direct_group_reduction" in details or "ordered_one_sided_power_s" not in phases
          or "direct_log_groups_and_energy_s" in phases):
        raise AssertionError("Direct power baseline did not materialize its ordered power vector")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000_000])
    parser.add_argument("--methods", choices=("fftw", "matched"), nargs="+",
                        help="Default: fftw and matched; direct power comparison: fftw only")
    parser.add_argument("--signals", nargs="+", default=["white"],
                        choices=("white", "pink", "brown", "mixed_tones", "offbin_tone", "dc_nanovolt"))
    parser.add_argument("--point-factors", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fft-memory-mib", type=float,
                        help="FFT scratch allowance (default 64 MiB; direct power comparison requires 128 MiB)")
    parser.add_argument("--native-working-mib", type=float, default=128.)
    parser.add_argument("--max-seconds", type=float, default=600.)
    comparison_mode = parser.add_mutually_exclusive_group()
    comparison_mode.add_argument("--compare-packed-phase", action="store_true",
                                 help="Compare frozen fast.8 versus current packed phase construction, requiring identical spectra")
    comparison_mode.add_argument("--compare-direct-power", action="store_true",
                                 help="Compare materialized versus direct-log ABI3 power after full original-N C2C, requiring 3 ppm PSD/NSD agreement")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.methods is None:
        args.methods = ["fftw"] if args.compare_direct_power else ["fftw", "matched"]
    if args.fft_memory_mib is None:
        args.fft_memory_mib = 128. if args.compare_direct_power else 64.
    if (min(args.sizes) < 4 or args.threads < 1 or args.repeats < 1 or
            any(value < 1 for value in args.point_factors) or
            any(not math.isfinite(value) or value <= 0 for value in
                (args.fft_memory_mib, args.native_working_mib, args.max_seconds))):
        parser.error("Positive sizes, workers, repeats, factors, budgets and deadline required")
    if args.compare_packed_phase and any(n % 2 for n in args.sizes):
        parser.error("Packed phase comparison requires even lengths >= 4")
    if args.compare_direct_power and (args.methods != ["fftw"] or args.fft_memory_mib != 128.
                                      or any(n > 1_000_000_000 for n in args.sizes)):
        parser.error("Direct power comparison requires fftw only, 128 MiB FFT scratch and N<=1B")
    import numpy as np
    from benchmarks import bench_fftw as bf
    from benchmarks.bench_fftw_disk import disk_pipeline, input_file
    from benchmarks.fftw_comparison_signals import signal
    from lpsd_benchmark.resources import detect_resources, MIB
    from lpsd_benchmark.worker import _load_fftw, resolve_point_plan
    from lpsd_fast import api

    if args.compare_direct_power:
        from benchmarks._fftw_disk import DiskFFT
        from benchmarks._fftw_direct_power import DirectLibrary
        for n in args.sizes:
            if not DiskFFT.memory_estimate(n, 128.)["workspace_power_supported"]:
                parser.error("Direct power comparison requires original full-N geometry with an even B factor for workspace power")
    variants = (("materialized", "direct-log") if args.compare_direct_power else
                ("legacy-phase", "packed-real") if args.compare_packed_phase else ("full", "packed-real"))
    baseline, candidate = variants
    api._native()
    requested_directory = args.work_dir.resolve() if args.work_dir else None
    if requested_directory:
        requested_directory.mkdir(parents=True, exist_ok=True)
    backend = _load_fftw(bf, args.threads)
    report = {
        "schema": ("lpsd-direct-power-pairs-v1" if args.compare_direct_power else
                   "lpsd-packed-phase-pairs-v1" if args.compare_packed_phase else "lpsd-disk-algorithm-pairs-v1"),
        "status": "running", "baseline_variant": baseline, "candidate_variant": candidate,
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
    if args.compare_packed_phase:
        from benchmarks.reference import packed_real_phase_fast8 as reference
        method_hash = hashlib.sha256(inspect.getsource(
            reference.LegacyPhasePackedRealDiskFFT._tile).encode("utf-8")).hexdigest()
        if method_hash != reference.REFERENCE_METHOD_SHA256:
            raise RuntimeError("Frozen fast.8 reconstruction method was changed")
        reference_name = "benchmarks/reference/packed_real_phase_fast8.py"
        report["source_sha256"][reference_name] = hashlib.sha256((ROOT / reference_name).read_bytes()).hexdigest()
        report["comparison_reference"] = {
            "commit": reference.REFERENCE_COMMIT, "file": reference.REFERENCE_FILE,
            "method": "PackedRealDiskFFT._tile", "method_sha256": method_hash,
            "scope": "Frozen method subclass of current packed FFT. Only reconstruction differs; all other code, buffers and settings are shared.",
        }
        report["scope"] += " Both variants request packed-real. Class selection is outside the timer; all returned PSD/NSD bytes must agree."
    if args.compare_direct_power:
        report["direct_log_power_abi"] = DirectLibrary(api._LIB).abi_version
        for source in ("benchmarks/_fftw_direct_power.py", "lpsd_fast/_native/fftw_direct_log_power.c",
                       "lpsd_fast/_native/fast_dft.c"):
            report["source_sha256"][source] = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        report["scope"] += (" Both variants request the original-N full C2C transform, fs=50 Hz, K=100, "
                            "Kaiser PSLL=200 dB, 128 MiB FFT scratch and workspace storage. Only the power "
                            "reduction option differs. Actual full-N/materialized/ABI3 no-power-vector routes "
                            "are checked after timing. All returned float32 PSD/NSD values must agree within "
                            "3 ppm with zero absolute tolerance; bit identity is also reported.")
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
                                    order = variants if repeat % 2 == 0 else variants[::-1]
                                    for variant in order:
                                        if time.perf_counter() >= deadline:
                                            raise TimeoutError("Comparison deadline reached between calls")
                                        options = SimpleNamespace(sample_rate=50., n_frequencies=plan["target_points"],
                                                                  n_averages=100, fft_memory_mb=args.fft_memory_mib,
                                                                  low_working_mb=args.native_working_mib, workers=threads,
                                                                  smoothing_workers=min(4, threads), power_storage="external" if n % 2 else "workspace",
                                                                  disk_fft_algorithm=("full" if args.compare_direct_power else
                                                                                      "packed-real" if args.compare_packed_phase else variant),
                                                                  disk_power_reduction=variant if args.compare_direct_power else "materialized")
                                        with packed_phase_variant(variant) if args.compare_packed_phase else nullcontext():
                                            gc.collect()
                                            wall, cpu = time.perf_counter(), time.process_time()
                                            frame, details = disk_pipeline(method, values, backend, directory, options)
                                            cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
                                        if (not np.isfinite(frame.to_numpy()).all() or
                                                not (frame.to_numpy() >= 0).all()):
                                            raise AssertionError("Nonfinite or negative spectrum")
                                        if args.compare_packed_phase and (
                                                details["disk_fft_algorithm"] != "packed-real"
                                                or details["fft"]["route"] != "packed_even"
                                                or details["fft"]["inner_n"] != n // 2):
                                            raise AssertionError("Packed phase comparison did not execute the packed route")
                                        frames[variant] = frame
                                        samples[variant] = {"wall_s": wall, "cpu_s": cpu,
                                                            "phases": details["phases"]}
                                        if args.compare_packed_phase:
                                            samples[variant].update(disk_fft_algorithm=details["disk_fft_algorithm"],
                                                                    fft_inner_n=details["fft"]["inner_n"],
                                                                    fft_factors=details["fft"]["factors"])
                                        if args.compare_direct_power:
                                            samples[variant].update(direct_power_sample(
                                                variant, n, plan["actual_points"], frame, details))
                                    a, b = frames[baseline], frames[candidate]
                                    same_grid = np.array_equal(a.index.to_numpy(), b.index.to_numpy())
                                    comparison = {column: differences(a[column].to_numpy(), b[column].to_numpy())
                                                  for column in ("psd", "nsd")}
                                    good = same_grid and all(item["passed"] for item in comparison.values())
                                    if args.compare_packed_phase:
                                        good &= all(item["bit_identical"] for item in comparison.values())
                                    if args.compare_direct_power:
                                        good &= all(item["strict_3ppm_passed"] for item in comparison.values())
                                    row["pairs"].append({"repeat": repeat, "order": order, "samples": samples,
                                                         "identical_frequency_grid": same_grid,
                                                         "comparison": comparison, "passed": good})
                                    failed |= not good
                                    write_report(args.output, report)
                                row["median_wall_s"] = {algorithm: statistics.median(
                                    pair["samples"][algorithm]["wall_s"] for pair in row["pairs"])
                                    for algorithm in variants}
                                row["median_cpu_s"] = {algorithm: statistics.median(
                                    pair["samples"][algorithm]["cpu_s"] for pair in row["pairs"])
                                    for algorithm in variants}
                                row["ratio_of_wall_medians"] = row["median_wall_s"][baseline] / row["median_wall_s"][candidate]
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
