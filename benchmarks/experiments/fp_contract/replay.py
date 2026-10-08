#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated compiler experiment; production libraries are never replaced.

Usage: replay.py {baseline,fma} {bench,audit} -- <ordinary CLI arguments>
Audit scalar calls always use the pinned -ffp-contract=off library.
The switch happens only between complete, synchronous public API calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("baseline", "fma"))
    parser.add_argument("operation", choices=("bench", "audit"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if "--output" not in arguments:
        parser.error("An explicit downstream --output path is required.")
    output = Path(arguments[arguments.index("--output") + 1])

    import lpsd_fast
    from lpsd_fast import api

    from support import load_libraries
    baseline, experiment, reports = load_libraries()
    candidate = experiment if args.variant == "fma" else baseline
    api._LIB = candidate

    if args.operation == "audit":
        original_api = lpsd_fast.lpsd

        def pinned_reference(*positional, **keywords):
            api._LIB = baseline if keywords.get("kernel") == "scalar" else candidate
            return original_api(*positional, **keywords)

        lpsd_fast.lpsd = pinned_reference
        from benchmarks import check_accuracy
        status = check_accuracy.main(arguments)
    else:
        from benchmarks import bench_lpsd
        ordinary_evidence = bench_lpsd.build_evidence

        def actual_evidence():
            evidence = ordinary_evidence()
            evidence["fast"] = reports[args.variant]
            return evidence

        bench_lpsd.build_evidence = actual_evidence
        status = bench_lpsd.main(arguments)

    payload = json.loads(output.read_text())
    payload["compiler_experiment"] = {
        "variant": args.variant,
        "candidate": reports[args.variant],
        "scalar_reference": reports["baseline"],
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Experimental global contraction flag; scalar audit calls remain pinned to the final off library. No production API or library was modified.",
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return status or 0


if __name__ == "__main__":
    raise SystemExit(main())
