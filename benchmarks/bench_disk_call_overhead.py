#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare fresh-process complete disk calls from two built checkouts.

Each observation runs in a new interpreter, matching the portable worker's
first-use diagnostics. The timer includes all disk_pipeline work and cleanup;
imports, input creation/hashes, FFTW discovery and source hashing are outside.
Run serially without concurrent benchmarks. No remote execution or uploads.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CHILD = r'''
import hashlib, json, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from benchmarks import bench_fftw as bf
from benchmarks.bench_fftw_disk import disk_pipeline, input_file
from lpsd_benchmark.worker import _load_fftw
from lpsd_fast import api
from lpsd_fast.build import source_manifest

n, algorithm, threads = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
api._native()
library = Path(api._LIB._name).resolve()
build_report = json.loads(library.with_suffix('.build.json').read_text())
native_hash = hashlib.sha256(library.read_bytes()).hexdigest()
assert native_hash == build_report['binary_sha256']
assert source_manifest(library.parent / build_report['source']) == build_report['source_sha256']
backend = _load_fftw(bf, threads)
assert backend.evidence['binary_sha256'] is not None
assert not backend.threadlib or backend.threads_evidence['binary_sha256'] is not None
args = SimpleNamespace(sample_rate=50., n_frequencies=1000, n_averages=100,
                       workers=threads, smoothing_workers=threads,
                       low_working_mb=64., fft_memory_mb=16.,
                       power_storage='workspace', disk_fft_algorithm=algorithm,
                       statistics='corrected')
with tempfile.TemporaryDirectory(prefix='disk-call-') as directory:
    directory = Path(directory)
    source = directory / 'input.f64'
    initial = input_file(source, n, 50., 20261021)
    values = np.memmap(source, dtype=np.float64, mode='r', shape=(n,))
    try:
        testing_before = 'numpy.testing' in sys.modules
        started = time.perf_counter()
        frame, details = disk_pipeline('fftw', values, backend, directory, args)
        wall = time.perf_counter() - started
        testing_after = 'numpy.testing' in sys.modules
        assert bf.digest_array(values) == initial['sha256']
        assert np.isfinite(frame.to_numpy()).all()
        assert (frame.to_numpy() >= 0.).all()
        assert np.array_equal(np.sqrt(frame.psd.to_numpy().astype(np.complex64)).real,
                              frame.nsd.to_numpy())
        assert set(directory.iterdir()) == {source}
    finally:
        values._mmap.close()
    row = dict(wall_s=wall, phases=details['phases'], parseval=details['parseval'],
               actual_points=len(frame), input_sha256=initial['sha256'],
               output_sha256=hashlib.sha256(frame.index.to_numpy().tobytes()
                                           + frame.to_numpy().tobytes()).hexdigest(),
               native_sha256=native_hash,
               fftw_sha256=backend.evidence['binary_sha256'],
               fftw_threads_sha256=(backend.threads_evidence or {}).get('binary_sha256'),
               actual_fftw_threads=backend.threads, input_unchanged=True,
               finite_nonnegative=True, psd_nsd_exact=True, owned_workspace_removed=True,
               numpy_testing_before=testing_before, numpy_testing_after=testing_after)
print(json.dumps(row, allow_nan=False))
'''


def source_hashes(root):
    """Relative source closure for the computation and measurement helpers."""
    paths = []
    for name in ("lpsd", "lpsd_fast", "lpsd_benchmark", "benchmarks"):
        paths.extend(path for path in (root / name).rglob("*")
                     if path.is_file() and path.suffix in (".py", ".c", ".h"))
    paths += [root / "pyproject.toml", root / "setup.py"]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--n", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--algorithms", nargs="+", choices=("full", "packed-real"),
                        default=["full", "packed-real"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 100 or args.threads < 1 or any(n < 1000 for n in args.n):
        parser.error("Require 2..100 pairs, positive threads and N >= 1000")
    roots = {"baseline": args.baseline_root.resolve(), "candidate": args.candidate_root.resolve()}
    for root in roots.values():
        if not (root / "benchmarks/bench_fftw_disk.py").is_file():
            parser.error("Both roots must be complete independently built checkouts")
    sources = {name: source_hashes(root) for name, root in roots.items()}
    report = {
        "schema": "lpsd-fresh-disk-call-v1", "state": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"system": platform.system(), "machine": platform.machine(),
                        "python": platform.python_version()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "child_sha256": hashlib.sha256(CHILD.encode()).hexdigest(),
        "source_sha256": sources,
        "methodology": {
            "pairs": args.repeats, "method": "fftw", "sample_rate": 50.,
            "requested_frequencies": 1000, "n_averages": 100, "threads": args.threads,
            "fft_memory_mb": 16., "power_storage": "workspace", "seed": 20261021,
            "timer": "Complete first disk_pipeline call including Parseval and cleanup",
            "order": "Fresh child per call; alternating A/B then B/A; all observations retained",
            "exclusions": "Imports, native/runtime loading, input creation/hashes, source evidence",
            "limits": "One shared host; warm filesystem caches; no cache flush or CPU-clock control",
        },
        "cases": [],
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.output)

    save()
    native_hashes = {}
    try:
        for n in args.n:
            for algorithm in args.algorithms:
                row = {"n": n, "algorithm": algorithm, "observations": []}
                report["cases"].append(row)
                reference = None
                for pair in range(args.repeats):
                    order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
                    for name in order:
                        env = os.environ.copy()
                        env["PYTHONPATH"] = str(roots[name])
                        for key in THREAD_KEYS:
                            env[key] = "1"
                        started = time.perf_counter()
                        child = subprocess.run([sys.executable, "-c", CHILD, str(n), algorithm,
                                                str(args.threads)], cwd=roots[name], env=env,
                                               text=True, capture_output=True, timeout=300)
                        elapsed = time.perf_counter() - started
                        if child.returncode:
                            raise RuntimeError(child.stderr)
                        found = json.loads(child.stdout)
                        row["observations"].append(dict(pair=pair, implementation=name,
                                                       process_wall_s=elapsed, **found))
                        if name not in native_hashes:
                            native_hashes[name] = found["native_sha256"]
                        if found["native_sha256"] != native_hashes[name]:
                            raise AssertionError("Native binary changed during measurement")
                        equality_keys = ("input_sha256", "output_sha256", "actual_points",
                                         "fftw_sha256", "fftw_threads_sha256", "actual_fftw_threads")
                        fingerprint = {key: found[key] for key in equality_keys}
                        if reference is None:
                            reference = fingerprint
                        if fingerprint != reference:
                            raise AssertionError("Input, output or FFTW runtime changed between observations")
                        save()
                timings = {name: [item["wall_s"] for item in row["observations"]
                                  if item["implementation"] == name] for name in roots}
                row["medians_s"] = {name: statistics.median(times) for name, times in timings.items()}
                row["ranges_s"] = {name: [min(times), max(times)] for name, times in timings.items()}
                row["speedup"] = row["medians_s"]["baseline"] / row["medians_s"]["candidate"]
                ratios = [before / after for before, after in zip(timings["baseline"], timings["candidate"])]
                row["faster_pairs"] = sum(ratio > 1 for ratio in ratios)
                row["paired_geometric_speedup"] = statistics.geometric_mean(ratios)
                row["outputs_bit_identical"] = True
                save()
                print(json.dumps({key: row[key] for key in ("n", "algorithm", "medians_s", "speedup")}), flush=True)
        if {name: source_hashes(root) for name, root in roots.items()} != sources:
            raise AssertionError("Sources changed during measurement")
        report["source_evidence_unchanged"] = True
        report["state"] = "completed"
    except BaseException as error:
        report["state"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
