#!/usr/bin/env python3
"""Separate out-of-core complete-call experiment; not the RAM pipeline timer.

Uses an exact N-point blocked FFTW factorization and the existing spectral
definitions. All large workspaces are shared file mappings. The cgroup RAM
limit is not modified. Input creation is outside the measured call, whereas
workspace construction, checks, required paging, computation and cleanup are
inside. A PSD/NSD pair is produced by each call. No timings are extrapolated.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

try:
    import resource
except ImportError:
    resource = None

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from benchmarks import bench_fftw as bf
from benchmarks.bench_fftw_optimized import measured, write_json, sha256, assert_equivalent
from benchmarks.bench_fftw_large_n import process_memory, read_file, trim
from benchmarks.fftw_comparison_metrics import summarize_noise
from benchmarks.matched_smoothing import _Kernel, _NativeSmoothing, MatchedSmoothing, estimate_once
from benchmarks._fftw_native import FFTWOperations
from benchmarks._fftw_disk import DiskFFT
from benchmarks._lpsd_disk import DiskLPSDSubset
from lpsd_fast import api

CHUNK = 262144
GIB = 1024**3
METHODS = ("fftw", "matched", "lpsd")


def close_map(array):
    if array is not None:
        array._mmap.close()


def io_counters():
    return {key: int(value) for key, value in
            (line.split(": ") for line in (read_file("/proc/self/io") or "").splitlines())}


def input_file(path, n, sample_rate, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, n]))
    values = np.memmap(path, dtype=np.float64, mode="w+", shape=(n,))
    scale = 10e-9 * np.sqrt(sample_rate / 2)
    for first in range(0, n, CHUNK):
        part = values[first:first + CHUNK]
        rng.standard_normal(len(part), out=part)
        part *= scale
    values.flush()
    fingerprint = bf.digest_array(values)
    close_map(values)
    return {"sha256": fingerprint, "samples": n, "dtype": "float64", "bytes": 8 * n,
            "seed_sequence": [seed, 0, n], "chunk_samples": CHUNK,
            "distribution": "Gaussian white noise; one-sided NSD=10 nV/sqrt(Hz)"}


def fill_tukey(window, fraction=.05):
    """The same periodic Tukey formula as matched_smoothing, in small blocks."""
    n = len(window)
    window[:] = 1.0
    if not fraction:
        return
    half_ramp = fraction * n / 2
    count = min(n, int(math.ceil(half_ramp)))
    for first in range(0, count, CHUNK):
        stop = min(count, first + CHUNK)
        indices = np.arange(first, stop, dtype=np.float64)
        left = .5 - .5 * np.cos(np.pi * indices / half_ramp)
        window[first:stop] = left
        lo = max(1, first)
        if stop > lo:
            window[n - stop + 1:n - lo + 1] = left[lo - first:][::-1]


def finite_chunks(values):
    for first in range(0, len(values), CHUNK):
        if not np.isfinite(values[first:first + CHUNK]).all():
            raise ValueError("Input contains nonfinite samples")


def vector_square_sum(values):
    return sum(float(np.dot(values[first:first + CHUNK], values[first:first + CHUNK]))
               for first in range(0, len(values), CHUNK))


def mapped_sum(values):
    return sum(float(np.sum(values[first:first + CHUNK]))
               for first in range(0, len(values), CHUNK))


def frame_from_psd(frequencies, density):
    psd = np.array(density, dtype=np.float32, copy=True)
    nsd = np.sqrt(psd.astype(np.complex64)).real.copy()
    return pd.DataFrame({"psd": psd, "nsd": nsd},
                        index=pd.Index(frequencies, name="frequency"), copy=False)


def disk_pipeline(method, values, backend, directory, args, *, progress=None, fft_progress=None):
    n = len(values)
    phases, metadata = {}, {}
    tick = time.perf_counter()
    def mark(name):
        nonlocal tick
        now = time.perf_counter()
        phases[name] = now - tick
        tick = now
        if progress:
            progress(name, phases[name])
    overlap = api._kaiser_rov(api._kaiser_alpha(200.0))
    beta = api._kaiser_alpha(200.0) * np.pi
    plan = api._frequency_plan(n, args.sample_rate, overlap, 1, 0, 1000, 100)
    frequencies = np.asarray(plan[0], dtype=np.float64)
    workspace_path = directory / "workspace.c128"
    powers_path = directory / "powers.f64"
    if method == "lpsd":
        workspace = np.memmap(workspace_path, dtype=np.complex128, mode="w+", shape=(n,))
        try:
            estimator = DiskLPSDSubset(n, args.sample_rate, plan=plan, psll=200.0,
                                       overlap=overlap, workers=args.workers,
                                       max_working_mb=args.low_working_mb)
            mark("workspace_and_plan_s")
            result = estimator.compute(values, workspace, outputs=("psd", "nsd"), profile=True)
            mark("lpsd_complete_s")
            metadata["lpsd"] = estimator.metadata
            metadata["lpsd_profile"] = estimator.last_profile
        finally:
            close_map(workspace)
            workspace_path.unlink(missing_ok=True)
        mark("workspace_cleanup_s")
        return result, {"phases": phases, **metadata}

    fft, powers = None, None
    try:
        backend.forget_wisdom()
        fft = DiskFFT(backend, n, workspace_path, memory_mb=args.fft_memory_mb,
                      progress=fft_progress)
        # The window occupies the second half of the file before complex
        # expansion. Preprocessing consumes it before execute overwrites it.
        window = fft.matrix.reshape(-1).view(np.float64)[n:2 * n]
        powers = np.memmap(powers_path, dtype=np.float64, mode="w+", shape=(n // 2 + 1,))
        mark("workspace_and_fftw_planning_s")
        if method == "fftw":
            status = api._LIB.generate_kaiser_series(api._pointer(window), n, beta)
            if status:
                raise RuntimeError("Kaiser generation failed")
        else:
            fill_tukey(window)
        s1, s2 = ct.c_double(), ct.c_double()
        if api._LIB.window_sums(api._pointer(window), n, ct.byref(s1), ct.byref(s2)):
            raise RuntimeError("Window normalization failed")
        s1, s2 = s1.value, s2.value
        metadata["window"] = {"name": "kaiser" if method == "fftw" else "periodic_tukey",
                              "taper_fraction": .05 if method == "matched" else None,
                              "s1": s1, "s2": s2, "target_kaiser_psll_db": 200.0}
        mark("window_and_normalization_s")
        finite_chunks(values)
        operations = FFTWOperations("native", api._LIB)
        operations.prepare(values, window, fft.x)
        mark("input_validation_copy_mean_window_s")
        # Unlike the RAM timers, this separate experimental complete-call
        # timer explicitly includes its two bounded Parseval diagnostic passes.
        expected = vector_square_sum(fft.x) / s2
        mark("time_energy_diagnostic_s")
        del window
        fft.execute()
        mark("blocked_fftw_s")
        metadata["fft"] = fft.metadata
        metadata["fft_profile"] = getattr(fft, "last_profile", None)
        fft.power_into(powers, args.sample_rate, s2)
        mark("ordered_one_sided_power_s")
        obtained = mapped_sum(powers) * args.sample_rate / n
        np.testing.assert_allclose(obtained, expected, rtol=1e-10, atol=0.0)
        metadata["parseval"] = {"time_power_v2": expected, "fft_power_v2": obtained,
                                "relative_error": abs(obtained - expected) / expected,
                                "diagnostic_passes_included_in_timer": True}
        mark("frequency_energy_diagnostic_s")
        if method == "fftw":
            grid = bf.partitions(n, args.sample_rate, frequencies)
            density = np.asarray([np.sum(powers[first:stop], dtype=np.float64) / (stop - first)
                                  for first, stop in zip(grid["cuts"][:-1], grid["cuts"][1:])])
            result = frame_from_psd(frequencies, density)
            mark("logarithmic_grouping_and_outputs_s")
        else:
            lengths = np.asarray(plan[3], dtype=np.int64)
            low_mask = (lengths > n / 16) | (lengths < 32)
            high_indices = np.flatnonzero(~low_mask)
            halfwidth = max(8.5, beta / np.pi + .5)
            df = args.sample_rate / n
            kernels = []
            for j in high_indices:
                frequency, length = float(frequencies[j]), int(lengths[j])
                radius = halfwidth * args.sample_rate / length
                first = max(0, int(math.ceil((frequency - radius) / df)))
                stop = min(len(powers), int(math.floor((frequency + radius) / df)) + 1)
                kernels.append(_Kernel(int(j), first, stop, length, frequency, math.nan, None))
            smoother = _NativeSmoothing(api._LIB, "native", "float32", defer_normalization=True)
            smoother.bind(kernels)
            density = np.zeros(len(frequencies), dtype=np.float64)
            smoother.bind_buffers(powers, density)
            with ThreadPoolExecutor(max_workers=args.smoothing_workers) as executor:
                owner = SimpleNamespace(n=n, sample_rate=args.sample_rate, beta=beta,
                                        halfwidth=halfwidth, powers=powers, _kernels=kernels,
                                        _executor=executor, smoothing_workers=args.smoothing_workers)
                ranges = MatchedSmoothing._kernel_ranges(owner)
                list(executor.map(lambda bounds: smoother.apply(owner, *bounds, density), ranges))
            if not np.isfinite(smoother.denominators).all() or not (smoother.denominators > 0).all():
                raise AssertionError("Smoothing normalization invalid")
            metadata["smoothing"] = {"kernel_halfwidth_segment_bins": halfwidth,
                                      "kernel": "unchanged analytic Kaiser-power response and reflection",
                                      "kernel_cache_bytes": 0, "deferred_normalization": True,
                                      "high_points": int(len(high_indices)), "low_points": int(low_mask.sum()),
                                      "low_mask": low_mask, "workers": args.smoothing_workers}
            mark("matched_frequency_smoothing_s")
            # Nothing still needs FFT content or powers. Close the plans and
            # all large FFT-only RAM tiles before reusing the same 16N file.
            del owner, smoother, kernels
            close_map(powers)
            powers = None
            powers_path.unlink(missing_ok=True)
            fft.close()
            fft = None
            trim()
            mark("release_fft_before_low_lpsd_s")
            workspace = np.memmap(workspace_path, dtype=np.complex128, mode="r+", shape=(n,))
            try:
                estimator = DiskLPSDSubset(n, args.sample_rate, plan=plan, psll=200.0,
                                           overlap=overlap, plan_mask=low_mask,
                                           workers=args.workers, max_working_mb=args.low_working_mb)
                low = estimator.compute(values, workspace, outputs="psd", profile=True)
                density[low_mask] = low.psd.to_numpy()
                metadata["low_lpsd"] = estimator.metadata
                metadata["low_lpsd_profile"] = estimator.last_profile
                result = frame_from_psd(frequencies, density)
            finally:
                close_map(workspace)
            mark("low_lpsd_and_outputs_s")
    finally:
        if powers is not None:
            close_map(powers)
        if fft is not None:
            fft.close()
        powers_path.unlink(missing_ok=True)
        workspace_path.unlink(missing_ok=True)
    mark("workspace_cleanup_s")
    return result, {"phases": phases, **metadata}


def child(args):
    n, method = args.n, args.child
    # Files consume virtual address space, while their physical pages remain
    # subject to the original 8-GiB cgroup limit. Permit only bounded anonymous
    # headroom above the explicitly known shared file mappings.
    mapped_bytes = (24 if method == "lpsd" else 28) * n + 8
    guard = mapped_bytes + 2 * GIB
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (guard, guard))
    api._native()
    backend = bf.FFTWLibrary(args.fftw_library, args.threads_library, args.fftw_threads)
    values = np.memmap(args.input, dtype=np.float64, mode="r", shape=(n,))
    report = {"n": n, "method": method, "mode": "disk_complete", "outputs": "both",
              "status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "settings": vars(args), "memory_cgroup_limit": read_file("/sys/fs/cgroup/memory.max"),
              "address_space_guard_bytes": guard, "maximum_large_mapping_bytes": mapped_bytes,
              "input_sha256_before": bf.digest_array(values),
              "native_binary_sha256": sha256(api._LIB._name),
              "fftw": {"version": backend.version, "binary": backend.evidence,
                       "threads_binary": backend.threads_evidence}, "progress": [],
              "fft_progress_events": []}
    def progress(name, seconds):
        print(f"PHASE {method} N={n} {name}: {seconds:.3f} s", flush=True)
        report["progress"].append({"phase": name, "wall_s": seconds})
        report["current_process_memory"] = process_memory()
        write_json(args.output, report)
    def fft_progress(event):
        entry = {**event, "call_elapsed_wall_s": time.perf_counter() - call_started}
        report["fft_progress_events"].append(entry)
        report["current_process_memory"] = process_memory()
        report["current_io_counters"] = io_counters()
        print(f"FFT {method} N={n} {entry['phase']}: "
              f"{entry['completed']}/{entry['total']} at {entry['call_elapsed_wall_s']:.3f} s", flush=True)
        write_json(args.output, report)
    write_json(args.output, report)
    try:
        with tempfile.TemporaryDirectory(prefix=f"disk_{method}_{n}_", dir=args.work_dir) as directory:
            io_before = io_counters()
            call_started = time.perf_counter()
            (frame, details), timing = measured(lambda: disk_pipeline(
                method, values, backend, Path(directory), args,
                progress=progress, fft_progress=fft_progress))
            io_after = io_counters()
        finite = np.isfinite(frame.to_numpy()).all() and (frame.to_numpy() >= 0).all()
        assert finite
        np.testing.assert_array_equal(np.sqrt(frame.psd.to_numpy().astype(np.complex64)).real,
                                      frame.nsd.to_numpy())
        report.update(status="completed", timing=timing, details=details,
                      io_delta={key: io_after[key] - io_before.get(key, 0) for key in io_after},
                      spectrum={"frequency_hz": frame.index.to_numpy(), "psd": frame.psd.to_numpy(),
                                "nsd": frame.nsd.to_numpy()},
                      noise_summary=summarize_noise(frame.index, frame.nsd, np.full(len(frame), 1e-16)))
        report["input_sha256_after"] = bf.digest_array(values)
        report["input_unchanged"] = report["input_sha256_before"] == report["input_sha256_after"]
        assert report["input_unchanged"]
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        close_map(values)
        report["process_memory"] = process_memory()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100_000_000, 1_000_000_000])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--fftw-library", type=Path, required=True)
    parser.add_argument("--threads-library", type=Path, required=True)
    parser.add_argument("--fftw-threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoothing-workers", type=int, default=4)
    parser.add_argument("--fft-memory-mb", type=float, default=128)
    parser.add_argument("--low-working-mb", type=float, default=2048)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=METHODS)
    parser.add_argument("--n", type=int)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=2400)
    args = parser.parse_args()
    for name in ("fftw_library", "threads_library", "work_dir", "output", "input"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.child:
        child(args)
        return
    report = {"schema": "lpsd-fftw-disk-v1", "settings": vars(args), "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                                [Path(__file__), ROOT / "benchmarks/_fftw_disk.py",
                                 ROOT / "benchmarks/_lpsd_disk.py", ROOT / "benchmarks/bench_fftw.py",
                                 ROOT / "benchmarks/bench_fftw_optimized.py", ROOT / "benchmarks/bench_fftw_large_n.py",
                                 ROOT / "benchmarks/fftw_comparison_metrics.py", ROOT / "benchmarks/matched_smoothing.py",
                                 ROOT / "benchmarks/_fftw_native.py", ROOT / "lpsd_fast/api.py",
                                 ROOT / "lpsd_fast/planning.py", ROOT / "lpsd_fast/prepared.py",
                                 ROOT / "lpsd/_helpers.py"]},
              "scope": "One observed complete PSD+NSD call per N/method. Includes bounded energy diagnostics, workspace creation, progress checkpoints, actual paging and deletion of disposable work files. Excludes signal creation, input hashes, process/import/library loading. Work files are not durable outputs; no final fsync is forced. Original cgroup RAM limit unchanged; shared file mappings are normal pageable filesystem I/O. Different FFT factorization and workspace strategy from the optimized RAM implementations; no RAM reuse timing or in-RAM speedup claim.",
              "memory_events_before": read_file("/sys/fs/cgroup/memory.events"), "inputs": [], "jobs": []}
    jobs_dir = args.output.parent / (args.output.stem + "_jobs")
    jobs_dir.mkdir(exist_ok=True)
    write_json(args.output, report)
    failure = False
    try:
        for n in args.sizes:
            need = 28 * n + 512 * 1024**2
            free = shutil.disk_usage(args.work_dir).free
            if free < need:
                report["jobs"].append({"n": n, "status": "not_executed_disk_capacity", "required_bytes_with_reserve": need,
                                       "available_bytes": free})
                write_json(args.output, report)
                failure = True
                continue
            path = args.work_dir / f"input_{n}.f64"
            try:
                print(f"INPUT N={n}", flush=True)
                source = input_file(path, n, args.sample_rate, args.seed)
                report["inputs"].append(source)
                write_json(args.output, report)
                for method in args.methods:
                    job_path = jobs_dir / f"{method}_{n}.json"
                    job_path.unlink(missing_ok=True)
                    command = [sys.executable, str(Path(__file__).resolve()), "--child", method, "--n", str(n),
                               "--input", str(path), "--output", str(job_path), "--work-dir", str(args.work_dir),
                               "--fftw-library", str(args.fftw_library), "--threads-library", str(args.threads_library),
                               "--fftw-threads", str(args.fftw_threads), "--workers", str(args.workers),
                               "--smoothing-workers", str(args.smoothing_workers), "--fft-memory-mb", str(args.fft_memory_mb),
                               "--low-working-mb", str(args.low_working_mb), "--sample-rate", str(args.sample_rate),
                               "--seed", str(args.seed)]
                    timed_out = False
                    with job_path.with_suffix(".log").open("w") as log:
                        try:
                            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                       env=dict(os.environ, MALLOC_ARENA_MAX="2"), timeout=args.timeout_seconds)
                            code = completed.returncode
                        except subprocess.TimeoutExpired:
                            code, timed_out = -1, True
                    job = json.loads(job_path.read_text()) if job_path.exists() else {"n": n, "method": method}
                    job["child_exit_code"] = code
                    if timed_out:
                        job.update(status="timeout", timeout_seconds=args.timeout_seconds)
                    elif code != 0:
                        job["status"] = "failed"
                    if job.get("status") == "completed":
                        assert job["input_sha256_before"] == source["sha256"]
                        print(f"JOB N={n} {method}: {job['timing']['wall_s']:.3f} s", flush=True)
                    else:
                        failure = True
                        print(f"JOB N={n} {method}: {job.get('status', 'unknown')}", flush=True)
                        print(job_path.with_suffix(".log").read_text()[-5000:], flush=True)
                    report["jobs"].append(job)
                    write_json(args.output, report)
                    # A killed child cannot run TemporaryDirectory cleanup.
                    for abandoned in args.work_dir.glob(f"disk_{method}_{n}_*"):
                        shutil.rmtree(abandoned)
            finally:
                path.unlink(missing_ok=True)
    except BaseException as exc:
        failure = True
        report["parent_error"] = {"error_type": type(exc).__name__, "error": str(exc)}
        raise
    finally:
        report["status"] = "completed" if not failure else "incomplete"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["memory_events_after"] = read_file("/sys/fs/cgroup/memory.events")
        write_json(args.output, report)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
