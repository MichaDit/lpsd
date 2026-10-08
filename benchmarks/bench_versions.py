#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare complete Kaiser/PSD APIs from two separately built checkouts.

Run as ``python -m benchmarks.bench_versions`` from the candidate checkout.
This is an integration comparison, including each version's Python API;
the segment experiment separately isolates the native kernel.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import pandas as pd

from benchmarks.bench_lpsd import fingerprint, json_scalar, profile_summary
from benchmarks.bench_rolling_boxcar import evidence, load_package, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--n", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 1000 or min(args.workers, args.profile_workers) < 1 or args.repeats < 2:
        parser.error("Require n>=1000, positive worker counts, and at least two repetitions")

    modules = {
        "baseline": load_package(args.baseline_root, "lpsd_version_baseline"),
        "candidate": load_package(args.candidate_root, "lpsd_version_candidate"),
    }
    values = pd.Series(np.random.default_rng(20261008).standard_normal(args.n), copy=False)
    parameters = dict(sample_rate=1., n_frequencies=1000, n_averages=100,
                      window_function="kaiser", psll=200., detrending_order=0,
                      kernel="fast", outputs="psd", workers=args.workers,
                      max_working_mb=4096, window_cache_mb=128)
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "n": args.n, "input_sha256": hashlib.sha256(values.to_numpy().tobytes()).hexdigest(),
        "input": "resident float64 pandas Series, PCG64 seed20261008",
        "parameters": parameters,
        "implementations": {name: evidence(module) for name, module in modules.items()},
        "methodology": {
            "timing": "complete public API wall time; alternating A/B then B/A",
            "warmup": "one complete problem per implementation before timing",
            "exclusions": "imports, data creation, builds, provenance, warmup, comparisons and JSON output",
            "profiles": "separate calls at the stated profile worker count; not timing repetitions",
            "reuse": "each ordinary API call prepares its own plan, windows and coefficients",
            "limits": "shared host and CPU clocks uncontrolled; all observations retained",
        },
        "calls": [],
    }
    reference = modules["baseline"].lpsd(values, **parameters)
    modules["candidate"].lpsd(values, **parameters)
    report["reference_fingerprint"] = fingerprint(reference)
    report["actual_frequencies"] = len(reference)
    repetitions = {name: [] for name in modules}
    for pair in range(args.repeats):
        order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        for name in order:
            started = time.perf_counter()
            result = modules[name].lpsd(values, **parameters)
            elapsed = time.perf_counter() - started
            repetitions[name].append(elapsed)
            comparison = metrics(reference, result)
            if comparison["points_at_or_above_one_percent"] or comparison["changed_exact_zeros"]:
                raise AssertionError("Version comparison exceeds the noise-case numerical criterion")
            report["calls"].append({"pair": pair, "implementation": name, "wall_s": elapsed,
                                    "fingerprint": fingerprint(result), "comparison": comparison})
            print(json.dumps({"pair": pair, "implementation": name, "wall_s": elapsed}), flush=True)
    medians = {name: statistics.median(times) for name, times in repetitions.items()}
    report["wall_repetitions_s"] = repetitions
    report["median_wall_s"] = medians
    report["baseline_over_candidate"] = medians["baseline"] / medians["candidate"]
    report["profiles"] = {}
    for name, module in modules.items():
        profile_parameters = dict(parameters, workers=args.profile_workers, profile=True)
        started = time.perf_counter()
        profiled = module.lpsd(values, **profile_parameters)
        elapsed = time.perf_counter() - started
        profile = profiled.attrs["lpsd_profile"]
        report["profiles"][name] = {
            "workers": args.profile_workers, "full_api_wall_s": elapsed,
            "fingerprint": fingerprint(profiled), "comparison": metrics(reference, profiled),
            "summary": profile_summary(profile), "full_profile": profile,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=json_scalar) + "\n")
    print(json.dumps({"median_wall_s": medians, "baseline_over_candidate": report["baseline_over_candidate"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
