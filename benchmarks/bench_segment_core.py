#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare complete PSD or CSD calls from two separately built checkouts.

Run as ``python -m benchmarks.bench_segment_core``. Each comparison keeps
the input, requested estimator and worker count fixed, warms both versions,
then alternates A/B and B/A order. Profiles are separate diagnostic calls.
Serialize this process with other benchmarks on a shared machine.
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

from benchmarks.bench_lpsd import (
    cgroup_value, fingerprint, json_scalar, profile_summary,
)
from benchmarks.bench_rolling_boxcar import evidence, load_package


def compare_spectra(reference, result):
    """Keep complex CSD errors, exact zeros and every returned column visible."""
    if not reference.index.equals(result.index):
        raise AssertionError("Frequency grids differ")
    if not reference.columns.equals(result.columns):
        raise AssertionError("Output columns differ")
    if not reference.dtypes.equals(result.dtypes):
        raise AssertionError("Output dtypes differ")
    output = {}
    for column in reference.columns:
        dtype = np.complex128 if np.iscomplexobj(reference[column]) else np.float64
        expected = reference[column].to_numpy().astype(dtype)
        actual = result[column].to_numpy().astype(dtype)
        if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
            raise AssertionError(f"Nonfinite benchmark output: {column}")
        delta = np.abs(actual - expected)
        nonzero = expected != 0
        relative = delta[nonzero] / np.abs(expected[nonzero])
        output[str(column)] = {
            "bitwise_equal": reference[column].to_numpy().tobytes()
                             == result[column].to_numpy().tobytes(),
            "max_absolute_error": float(delta.max(initial=0)),
            "max_relative_error_nonzero_reference": float(relative.max(initial=0)),
            "points_at_or_above_one_percent": int(np.count_nonzero(relative >= .01)),
            "changed_exact_zeros": int(np.count_nonzero(actual[~nonzero])),
        }
    # The seeded ordinary-noise workload is required to pass, independently
    # of the legacy deviation columns, which are still fully reported above.
    for column in ("ps", "psd", "asd", "nsd", "asdrms", "enbw"):
        if column in output and (
            output[column]["points_at_or_above_one_percent"]
            or output[column]["changed_exact_zeros"]
        ):
            raise AssertionError(f"Ordinary-noise numerical criterion failed: {column}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--kind", choices=("psd", "csd"), default="psd")
    parser.add_argument("--kernel", choices=("auto", "fast"), default="fast")
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--frequencies", type=int, default=1000)
    parser.add_argument("--averages", type=int, default=100)
    parser.add_argument("--window", choices=("kaiser", "hann", "hamming", "blackman",
                                            "bartlett", "boxcar"), default="kaiser")
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--outputs", nargs="+", default=["psd"])
    parser.add_argument("--profile-workers", type=int, default=1,
                        help="0 disables separate diagnostic calls")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n < 1000 or args.workers < 1 or args.repeats < 2 or args.profile_workers < 0:
        parser.error("Require n>=1000, workers>=1, repeats>=2, profile-workers>=0")
    if args.frequencies < 2 or args.averages < 1:
        parser.error("Require frequencies>=2 and averages>=1")
    if args.overlap is not None and not (np.isfinite(args.overlap) and 0 <= args.overlap < 1):
        parser.error("overlap must be finite and in [0, 1)")
    if args.window != "kaiser" and args.overlap is None:
        parser.error("Non-Kaiser windows require explicit overlap")
    if args.kind == "csd" and "nsd" in args.outputs:
        parser.error("NSD is defined only for an auto spectrum")

    modules = {
        "baseline": load_package(args.baseline_root, "lpsd_segment_baseline"),
        "candidate": load_package(args.candidate_root, "lpsd_segment_candidate"),
    }
    rng = np.random.default_rng(20261008)
    x = rng.standard_normal(args.n)
    if args.kind == "csd":
        values = pd.DataFrame({"left": x, "right": .7 * x + .6 * rng.standard_normal(args.n)})
    else:
        values = pd.Series(x, copy=False)
    parameters = dict(
        sample_rate=1., n_frequencies=args.frequencies, n_averages=args.averages,
        window_function=args.window, psll=200., overlap=args.overlap,
        detrending_order=0, kernel=args.kernel, outputs=args.outputs,
        csd=args.kind == "csd", workers=args.workers,
        max_working_mb=4096, window_cache_mb=128,
    )
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "cpu_max": cgroup_value("cpu.max"),
            "cpuset_cpus_effective": cgroup_value("cpuset.cpus.effective"),
        },
        "kind": args.kind, "n": args.n,
        "input_sha256": hashlib.sha256(values.to_numpy().tobytes()).hexdigest(),
        "input": "resident float64 pandas input, PCG64 seed 20261008",
        "parameters": parameters,
        "implementations": {name: evidence(module) for name, module in modules.items()},
        "methodology": {
            "timing": "complete public API wall time; alternating A/B then B/A pairs",
            "warmup": "one complete problem per implementation before timing",
            "exclusions": "imports, input creation, builds, warmup, comparisons, provenance and file output",
            "profiles": "separate instrumented calls, excluded from timing repetitions",
            "reuse": "each API call independently prepares its plan, windows and coefficients",
            "limits": "shared host; uncontrolled clocks; all observations retained",
        },
        "calls": [], "profiles": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, default=json_scalar) + "\n")

    reference = modules["baseline"].lpsd(values, **parameters)
    compare_spectra(reference, modules["candidate"].lpsd(values, **parameters))
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
            call = {"pair": pair, "implementation": name, "wall_s": elapsed,
                    "fingerprint": fingerprint(result),
                    "comparison": compare_spectra(reference, result)}
            report["calls"].append(call)
            save()
            print(json.dumps({key: call[key] for key in ("pair", "implementation", "wall_s")}),
                  flush=True)

    medians = {name: statistics.median(times) for name, times in repetitions.items()}
    report["wall_repetitions_s"] = repetitions
    report["median_wall_s"] = medians
    report["baseline_over_candidate"] = medians["baseline"] / medians["candidate"]
    report["candidate_faster_pairs"] = sum(
        candidate < baseline for baseline, candidate in zip(
            repetitions["baseline"], repetitions["candidate"]))
    if args.profile_workers:
        for name, module in modules.items():
            started = time.perf_counter()
            result = module.lpsd(values, **dict(parameters, workers=args.profile_workers, profile=True))
            elapsed = time.perf_counter() - started
            profile = result.attrs["lpsd_profile"]
            report["profiles"][name] = {
                "workers": args.profile_workers, "full_api_wall_s": elapsed,
                "fingerprint": fingerprint(result), "comparison": compare_spectra(reference, result),
                "summary": profile_summary(profile), "full_profile": profile,
            }
            save()
    # Never report timings from a library rebuilt or sources edited during
    # a run as if they belonged to the initial source/binary evidence.
    if report["implementations"] != {name: evidence(module) for name, module in modules.items()}:
        raise AssertionError("Source or native build changed during the comparison")
    report["completed"] = True
    save()
    print(json.dumps({"median_wall_s": medians,
                      "baseline_over_candidate": report["baseline_over_candidate"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
