#!/usr/bin/env python3
"""Memory-aware, serial large-N measurements of the existing FFTW estimators.

Every N/method gets a separate child process. There is only one prepared
estimator in a child at a time. An analytical memory lower bound rejects
impossible jobs before allocating their input. No runtime is extrapolated.
The process address-space limit is an additional Linux safety guard, not an
RSS measurement. All arrays, windows and estimator definitions are unchanged.
On platforms without physical-memory discovery, including Windows, pass
--memory-limit-bytes explicitly.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import ctypes as ct
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

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
from benchmarks import bench_fftw as bf
from benchmarks.bench_fftw_optimized import measured, summarize, write_json, sha256
from benchmarks.fftw_comparison_metrics import summarize_noise
from benchmarks.matched_smoothing import MatchedSmoothing, estimate_once
from lpsd_fast import api

MIB, GIB = 1024**2, 1024**3
METHODS = ("fftw", "matched", "matched_once", "lpsd")


def read_file(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def process_memory():
    values = {}
    for line in (read_file("/proc/self/status") or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "VmSize", "VmPeak"):
            values[key + "_bytes"] = int(value.split()[0]) * 1024
    values["max_rss_bytes"] = bf.max_rss_bytes()
    return values


def memory_lower_bound(method, n):
    # L=N at the first actual LPSD frequency. Window and both real Fourier
    # coefficient vectors coexist in the present native Python implementation.
    if method == "lpsd":
        return 32 * n, "8N input + 24N window and two coefficients at L=N"
    if method in ("matched", "matched_once"):
        return 24 * n + 24 * (n // 2 + 1), "8N input + 8N FFT input + 16(floor(N/2)+1) FFT output + 8N window + 8(floor(N/2)+1) powers; excludes all caches and low-LPSD workspace"
    return 24 * n + 16 * (n // 2 + 1), "8N input + 8N FFT input + 16(floor(N/2)+1) FFT output + 8N window; no full powers array"


def trim():
    gc.collect()
    try:
        function = ct.CDLL(None).malloc_trim
        function.argtypes, function.restype = [ct.c_size_t], ct.c_int
        function(0)
    except AttributeError:
        pass


def bounded_parseval(obj, chunk=262144):
    """Independent diagnostic, with O(chunk) instead of O(N) temporaries."""
    time_sum = 0.0
    for first in range(0, obj.n, chunk):
        values = obj.fft.x[first:first + chunk]
        time_sum += float(np.dot(values, values))
    power_sum = 0.0
    for first in range(0, len(obj.fft.y), chunk):
        values = obj.fft.y[first:first + chunk]
        assert np.isfinite(values).all()
        power_sum += float(np.sum(values.real * values.real + values.imag * values.imag))
    dc_square = float(abs(obj.fft.y[0])**2)
    nyquist_square = float(abs(obj.fft.y[-1])**2) if obj.n % 2 == 0 else 0.0
    expected = time_sum / obj.s2
    obtained = (2 * power_sum - dc_square - nyquist_square) / (obj.n * obj.s2)
    np.testing.assert_allclose(obtained, expected, rtol=2e-12, atol=0.0)
    result = {"time_power_v2": expected, "fft_power_v2": obtained,
              "relative_parseval_error": abs(obtained - expected) / expected,
              "outside_timed_region": True, "temporary_block_samples": chunk}
    if getattr(obj, "last_band_powers", None) is not None:
        bands = float(np.sum(obj.last_band_powers))
        positive = obtained - dc_square / (obj.n * obj.s2)
        np.testing.assert_allclose(bands, positive, rtol=2e-12, atol=0.0)
        result["relative_grouped_power_error"] = abs(bands - positive) / positive
    return result


def validate_frame(frame, reference, frequencies):
    np.testing.assert_array_equal(frame.index.to_numpy(), frequencies)
    assert len(frame.columns) > 0
    for column in frame:
        values = frame[column].to_numpy()
        assert values.dtype == np.float32
        assert np.isfinite(values).all() and (values >= 0).all()
        if reference is not None:
            np.testing.assert_array_equal(values, reference[column].to_numpy())
    if "psd" in frame and "nsd" in frame:
        np.testing.assert_array_equal(np.sqrt(frame.psd.to_numpy().astype(np.complex64)).real,
                                      frame.nsd.to_numpy())


def run_child(args):
    n, method = args.n, args.child
    lower, definition = memory_lower_bound(method, n)
    result = {"n": n, "method": method, "started_utc": datetime.now(timezone.utc).isoformat(),
              "settings": vars(args), "timings": [], "status": "starting",
              "memory_lower_bound_bytes": lower, "memory_lower_bound_definition": definition,
              "memory_cgroup_limit_bytes": args.memory_limit_bytes,
              "address_space_guard_bytes": args.address_space_guard_bytes}
    def checkpoint():
        result["process_memory"] = process_memory()
        write_json(args.output, result)
    if lower > args.memory_limit_bytes:
        result.update(status="not_executed_memory_limit",
                      reason="The known simultaneously allocated arrays alone exceed the machine memory limit. No input or estimator was allocated; no execution time exists.")
        checkpoint()
        print(f"SKIP N={n} {method}: {lower / GIB:.3f} GiB lower bound > {args.memory_limit_bytes / GIB:.3f} GiB limit", flush=True)
        return
    if sys.platform.startswith("linux") and args.address_space_guard_bytes:
        resource.setrlimit(resource.RLIMIT_AS, (args.address_space_guard_bytes,
                                               args.address_space_guard_bytes))
    try:
        api._native()
        backend = bf.FFTWLibrary(args.fftw_library, args.threads_library, args.fftw_threads)
        provider = bf.KaiserWindow("native", Path(api._LIB._name))
        result["native_binary"] = {"path": api._LIB._name, "sha256": sha256(api._LIB._name)}
        build_path = Path(api._LIB._name).with_suffix(".build.json")
        if build_path.exists():
            result["native_binary"]["build_metadata"] = json.loads(build_path.read_text())
        result["fftw"] = {"version": backend.version, "binary": backend.evidence,
                          "threads_binary": backend.threads_evidence}
        overlap = api._kaiser_rov(api._kaiser_alpha(200.0))
        plan = api._frequency_plan(n, args.sample_rate, overlap, 1, 0, 1000, 100)
        frequencies = np.asarray(plan[0], dtype=np.float64)
        result["plan"] = {"frequency_hz": frequencies, "lengths": plan[3],
                          "segment_counts": plan[4], "frequency_bins": plan[2],
                          "points": len(frequencies), "overlap": overlap}
        synthesis_start = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0, n]))
        x = rng.standard_normal(n)
        x *= 10e-9 * np.sqrt(args.sample_rate / 2)
        result["input"] = {"dtype": str(x.dtype), "samples": n, "bytes": x.nbytes,
                           "sha256": bf.digest_array(x), "seed_sequence": [args.seed, 0, n],
                           "distribution": "Gaussian white noise, one-sided NSD=10 nV/sqrt(Hz)",
                           "synthesis_and_hash_s_excluded": time.perf_counter() - synthesis_start}
        common = dict(sample_rate=args.sample_rate, psll=200.0, n_frequencies=1000, n_averages=100)
        matched_options = dict(**common, window_provider=provider, workers=args.workers,
                               max_working_mb=2048, total_cache_mb=args.total_cache_mb,
                               max_kernel_cache_mb=1536, window_cache_mb=None,
                               smoothing_workers=args.smoothing_workers, weight_dtype="float32",
                               operations_backend="native", smoothing_backend="native",
                               planner="estimate")
        def make():
            if method == "fftw":
                return bf.Periodogram(backend, n, **common, window_provider=provider,
                                      operations="native_grouped", operations_library=api._LIB,
                                      aggregation="log", output_dtype="float32", planner="estimate")
            return MatchedSmoothing(backend, n, **matched_options)
        def complete(output):
            selected = ("psd", "nsd") if output == "both" else output
            if method == "lpsd":
                return api.lpsd(x, **common, window_function="kaiser", detrending_order=0,
                                workers=args.workers, kernel="fast", max_working_mb=4096,
                                window_cache_mb=128, outputs=selected)
            backend.forget_wisdom()
            if method == "matched_once":
                return estimate_once(backend, x, outputs=selected, **matched_options)
            with make() as obj:
                return obj.compute(x, outputs=selected)
        result["memory_after_input"] = process_memory()
        result["status"] = "running"
        checkpoint()
        references = None
        order = np.random.default_rng(np.random.SeedSequence([args.seed, n, METHODS.index(method)]))
        if method in ("fftw", "matched"):
            backend.forget_wisdom()
            print(f"SETUP N={n} {method}", flush=True)
            obj, setup = measured(make)
            try:
                result["preparation"] = {"observed_setup": setup, "metadata": obj.metadata,
                                         "memory_after_setup": process_memory()}
                references = obj.compute(x, outputs=("psd", "nsd"))
                validate_frame(references, None, frequencies)
                snapshot = references.to_numpy(copy=True)
                warm = obj.compute(x, outputs=("psd", "nsd"))
                validate_frame(warm, references, frequencies)
                np.testing.assert_array_equal(references.to_numpy(), snapshot)
                result["memory_after_warmup"] = process_memory()
                rows = {output: [] for output in args.outputs}
                checkpoint()
                for repetition in range(args.repeats):
                    jobs = list(args.outputs)
                    order.shuffle(jobs)
                    for output in jobs:
                        print(f"RUN N={n} {method} reuse {output} {repetition + 1}/{args.repeats}", flush=True)
                        selected = ("psd", "nsd") if output == "both" else output
                        frame, timing = measured(lambda: obj.compute(x, outputs=selected))
                        timing["round"] = repetition
                        rows[output].append(timing)
                        validate_frame(frame, references, frequencies)
                        result["timings"] = [{"n": n, "method": method, "mode": "reuse",
                                              "outputs": name, **summarize(times)}
                                             for name, times in rows.items() if times]
                        checkpoint()
                result["validation"] = bounded_parseval(obj)
                if method == "matched":
                    frame = obj.profile(x, outputs="nsd")
                    result["profile"] = obj.last_profile
                else:
                    frame = obj.compute(x, outputs="nsd", profile=True)
                    result["profile"] = obj.last_profile
                validate_frame(frame, references, frequencies)
            finally:
                obj.close()
                del obj
                trim()
        else:
            print(f"WARMUP N={n} {method} complete", flush=True)
            references = complete("both")
            validate_frame(references, None, frequencies)
        rows = {output: [] for output in args.outputs}
        complete_mode = "complete" if method == "lpsd" else "fresh"
        previous_rows = list(result["timings"])
        for repetition in range(args.repeats):
            jobs = list(args.outputs)
            order.shuffle(jobs)
            for output in jobs:
                trim()
                print(f"RUN N={n} {method} {complete_mode} {output} {repetition + 1}/{args.repeats}", flush=True)
                frame, timing = measured(lambda: complete(output))
                timing["round"] = repetition
                rows[output].append(timing)
                validate_frame(frame, references, frequencies)
                result["timings"] = previous_rows + [
                    {"n": n, "method": method, "mode": complete_mode, "outputs": name,
                     **summarize(times)} for name, times in rows.items() if times]
                checkpoint()
        if method == "lpsd":
            # A separate instrumented call is explicitly not a timing median.
            profiled = api.lpsd(x, **common, window_function="kaiser", detrending_order=0,
                               workers=args.workers, kernel="fast", max_working_mb=4096,
                               window_cache_mb=128, outputs="nsd", profile=True)
            validate_frame(profiled, references, frequencies)
            result["profile"] = profiled.attrs["lpsd_profile"]
        result["spectrum"] = {"frequency_hz": frequencies,
                              "psd": references.psd.to_numpy(), "nsd": references.nsd.to_numpy()}
        result["noise_summary"] = summarize_noise(frequencies, references.nsd,
                                                   np.full(len(frequencies), 1e-16))
        result["input_unchanged"] = bf.digest_array(x) == result["input"]["sha256"]
        assert result["input_unchanged"]
        result["repeatability_and_output_checks_passed"] = True
        result["status"] = "completed"
    except MemoryError as exc:
        result.update(status="memory_allocation_failed", error_type=type(exc).__name__, error=str(exc))
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        checkpoint()
        raise
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    print(f"DONE N={n} {method}: {result['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10_000_000, 100_000_000, 1_000_000_000])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outputs", nargs="+", choices=("psd", "nsd", "both"), default=["psd", "nsd"])
    parser.add_argument("--fftw-library", type=Path, required=True)
    parser.add_argument("--threads-library", type=Path, required=True)
    parser.add_argument("--fftw-threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoothing-workers", type=int, default=4)
    parser.add_argument("--total-cache-mb", type=float, default=1024)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=METHODS)
    parser.add_argument("--n", type=int)
    parser.add_argument("--memory-limit-bytes", type=int,
                        help="RAM budget in bytes; required when automatic discovery is unavailable, including Windows")
    parser.add_argument("--address-space-guard-bytes", type=int)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 5 for n in args.sizes) or args.total_cache_mb < 0:
        parser.error("Require positive repeats, N>=5 and nonnegative cache budget")
    for name in ("output", "fftw_library", "threads_library"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    memory_max = read_file("/sys/fs/cgroup/memory.max")
    if args.memory_limit_bytes is None:
        if memory_max not in (None, "max"):
            args.memory_limit_bytes = int(memory_max)
        else:
            try:
                args.memory_limit_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            except (AttributeError, OSError, ValueError):
                parser.error("Cannot detect physical memory on this platform; pass --memory-limit-bytes explicitly")
    if args.address_space_guard_bytes is None:
        args.address_space_guard_bytes = max(GIB, args.memory_limit_bytes - 512 * MIB)
    if args.child:
        if args.n is None:
            parser.error("--child requires --n")
        run_child(args)
        return
    report = {"schema": "lpsd-fftw-large-n-v1", "settings": vars(args),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                [Path(__file__), ROOT / "benchmarks/bench_fftw.py",
                                 ROOT / "benchmarks/matched_smoothing.py", ROOT / "lpsd_fast/prepared.py",
                                 ROOT / "benchmarks/_fftw_native.py", ROOT / "lpsd_fast/api.py"]},
              "machine": {"platform": platform.platform(), "python": platform.python_version(),
                          "numpy": np.__version__, "cpu": platform.processor(),
                          "cpu_info": (read_file("/proc/cpuinfo") or "").split("\n\n")[0],
                          "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                          "cpu_max": read_file("/sys/fs/cgroup/cpu.max"),
                          "memory_max": memory_max, "swap_max": read_file("/sys/fs/cgroup/memory.swap.max"),
                          "memory_events_before": read_file("/sys/fs/cgroup/memory.events")},
              "scope": {"reuse": "Complete compute with one existing plan/window/cache; includes estimator input validation. Setup and close excluded.",
                        "fresh": "Forget FFTW wisdom, construct, compute, close. Includes plan/window/kernel preparation. Excludes process/import/library loading and signal creation.",
                        "complete": "Public native LPSD API call, including validation, planning, coefficients and all segments.",
                        "excluded": "Signal synthesis, input hashes, independent validation, plotting, process/import/library loading. OS caches are not cleared.",
                        "memory": "Process-lifetime peak RSS per isolated method process includes setup, warmups, computation and validation; not per-call RSS.",
                        "quality": "Original estimator definitions retained. White-noise summary describes this realization, not a general equivalence guarantee.",
                        "cache_change": f"{args.total_cache_mb:g} MiB total hybrid cache versus 2048 MiB in the previous study; all requested sizes in this invocation use the same settings.",
                        "timing_order": "Methods serialized in separate processes; PSD/NSD order shuffled within each repetition. Two prepared warmups or one complete warmup."},
              "jobs": []}
    jobs_dir = args.output.parent / (args.output.stem + "_jobs")
    jobs_dir.mkdir(exist_ok=True)
    write_json(args.output, report)
    for n in args.sizes:
        for method in args.methods:
            job_path = jobs_dir / f"{method}_{n}.json"
            job_path.unlink(missing_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--child", method, "--n", str(n),
                       "--fftw-library", str(args.fftw_library), "--threads-library", str(args.threads_library),
                       "--repeats", str(args.repeats), "--fftw-threads", str(args.fftw_threads),
                       "--workers", str(args.workers), "--smoothing-workers", str(args.smoothing_workers),
                       "--total-cache-mb", str(args.total_cache_mb), "--sample-rate", str(args.sample_rate),
                       "--seed", str(args.seed), "--memory-limit-bytes", str(args.memory_limit_bytes),
                       "--address-space-guard-bytes", str(args.address_space_guard_bytes),
                       "--output", str(job_path), "--outputs", *args.outputs]
            env = dict(os.environ, MALLOC_ARENA_MAX="2")
            with job_path.with_suffix(".log").open("w") as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if job_path.exists():
                job = json.loads(job_path.read_text())
            else:
                job = {"n": n, "method": method, "status": "child_failed_without_result"}
            job["child_exit_code"] = completed.returncode
            if completed.returncode != 0 and job.get("status") == "completed":
                job["status"] = "child_failed_after_checkpoint"
            job["raw_job_file"] = str(job_path)
            report["jobs"].append(job)
            write_json(args.output, report)
            print(f"JOB N={n} {method}: {job['status']} exit={completed.returncode}", flush=True)
            for row in job.get("timings", []):
                print(f"  {row['mode']} {row['outputs']}: median={row['median_wall_s']:.6f} s", flush=True)
            if completed.returncode != 0:
                print(job_path.with_suffix(".log").read_text()[-6000:], flush=True)
                break
        else:
            continue
        break
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["machine"]["memory_events_after"] = read_file("/sys/fs/cgroup/memory.events")
    report["cross_method_input_and_grid_checks"] = []
    for n in args.sizes:
        jobs = [job for job in report["jobs"] if job["n"] == n and job["status"] == "completed"]
        if jobs:
            for job in jobs[1:]:
                assert job["input"]["sha256"] == jobs[0]["input"]["sha256"]
                assert job["plan"]["frequency_hz"] == jobs[0]["plan"]["frequency_hz"]
            report["cross_method_input_and_grid_checks"].append({"n": n, "methods": [job["method"] for job in jobs], "passed": True})
    accepted = {"completed", "not_executed_memory_limit"}
    report["status"] = ("completed_with_resource_checks" if
                        len(report["jobs"]) == len(args.sizes) * len(args.methods)
                        and all(job["status"] in accepted and job["child_exit_code"] == 0 for job in report["jobs"])
                        else "incomplete")
    write_json(args.output, report)
    if report["status"] == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
