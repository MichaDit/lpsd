# SPDX-License-Identifier: GPL-3.0-or-later
"""Small native-runner comparison; writes JSON evidence, never a timing gate.

Run after ``make compile``. The ordinary API repetitions and the additional
profiled call are measured separately by bench_lpsd.py. No signal files are saved.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


def _revision_evidence(root):
    """Keep revision identity separate from the build/source hashes per call."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root)
    if dirty.returncode not in (0, 1):
        raise RuntimeError("Cannot inspect comparison checkout")
    return {"commit": revision, "tracked_changes": bool(dirty.returncode)}


def _paired_summary(pairs):
    """Summarize same-configuration pairs without pooling unrelated runners."""
    baseline = [pair["baseline"]["wall_s"] for pair in pairs]
    current = [pair["current"]["wall_s"] for pair in pairs]
    baseline_median = statistics.median(baseline)
    current_median = statistics.median(current)
    outputs_identical = all(pair["outputs_identical"] for pair in pairs)
    return {
        "baseline_median_wall_s": baseline_median,
        "current_median_wall_s": current_median,
        "baseline_range_wall_s": [min(baseline), max(baseline)],
        "current_range_wall_s": [min(current), max(current)],
        "ratio_of_medians": baseline_median / current_median,
        "wall_time_change_percent": 100. * (current_median / baseline_median - 1.),
        "outputs_identical": outputs_identical,
        "comparison_status": (
            "identical_output_fingerprints" if outputs_identical else "numerical_comparison_required"
        ),
    }


def revision_comparison(root, baseline_root, output, sizes, workers):
    """Run paired fresh-process complete API calls with one shared harness.

    The source trees use the same interpreter/dependencies and the identical
    current bench_lpsd.py. PYTHONPATH and cwd select the implementation. Each
    process warms only a small prefix, just like the existing architecture
    screen. The timed call still creates its plan, coefficients and outputs.
    """
    roots = {"baseline": baseline_root.resolve(), "current": root.resolve()}
    harness = root / "benchmarks" / "bench_lpsd.py"
    expected_builds = {
        name: json.loads((path / "lpsd_fast/_native/liblpsd_fast.build.json").read_text())
        for name, path in roots.items()
    }
    report = {
        "schema_version": 1,
        "kind": "same_runner_alternating_revision_comparison",
        "state": "running",
        "revisions": {name: _revision_evidence(path) for name, path in roots.items()},
        "input": "seeded float64 standard-normal samples; seed 20261008",
        "timing_scope": "complete_lpsd_api_call; imports, input, small warmup and output hashing excluded",
        "repetitions_per_variant": 3,
        "worker_counts": sorted({1, min(4, workers)}),
        "cases": [],
        "notes": [
            "Pairs alternate baseline/current, current/baseline, baseline/current on one CI runner.",
            "Shared CI timings are observations, not hard speed guarantees or comparisons between CPUs.",
            "Every raw call retains loaded binary hashes and source/compiler evidence.",
            "Output fingerprints compare frequencies, names, dtypes and values. A mismatch requires a separate numerical comparison; it does not prove an error.",
            "The existing numerical regression gates remain authoritative; these white-noise cases do not replace them.",
        ],
    }
    report["expected_case_count"] = len(sizes) * 2 * len(report["worker_counts"])
    (output / "revision-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for n in sizes:
        for kernel, outputs in (("auto", None), ("fast", ["psd", "nsd"])):
            for count in report["worker_counts"]:
                case = {"n": n, "kernel": kernel, "outputs": outputs or "all",
                        "workers": count, "pairs": []}
                for repeat in range(3):
                    order = ["baseline", "current"] if repeat % 2 == 0 else ["current", "baseline"]
                    pair = {"repeat": repeat, "execution_order": order}
                    for variant in order:
                        stem = f"ab-n{n}-{kernel}-w{count}-p{repeat}-{variant}"
                        destination = output / f"{stem}.json"
                        command = [sys.executable, str(harness), "--backend", "fast",
                                   "--n", str(n), "--workers", str(count), "--kernel", kernel,
                                   "--repeats", "1", "--warmups", "1", "--output", str(destination)]
                        if outputs is not None:
                            command += ["--outputs", *outputs]
                        environment = dict(os.environ, PYTHONPATH=str(roots[variant]))
                        subprocess.run(command, cwd=roots[variant], env=environment,
                                       check=True, timeout=180, capture_output=True, text=True)
                        measured = json.loads(destination.read_text())
                        # Mandatory verified build evidence prevents unnoticed
                        # lazy/system backends from turning this into another comparison.
                        build = measured.get("build_reports", {}).get("fast")
                        if not build or not build.get("binary_sha256"):
                            raise RuntimeError("Comparison requires a verified native build report")
                        if build != expected_builds[variant]:
                            raise RuntimeError("Comparison imported a different native implementation")
                        if measured["fast_metadata"]["workers"] != count:
                            raise RuntimeError("Revision comparison admitted a different worker count")
                        trial = measured["repeats"][0]
                        pair[variant] = {
                            "report": destination.name, "wall_s": trial["wall_s"],
                            "cpu_s": trial["cpu_s"], "output_sha256": trial["output_sha256"],
                            "actual_frequencies": trial["actual_frequencies"],
                            "native_sha256": build["binary_sha256"],
                            "native_mode": measured["fast_metadata"]["native_mode"],
                        }
                    if pair["baseline"]["actual_frequencies"] != pair["current"]["actual_frequencies"]:
                        raise RuntimeError("Revision comparison returned different frequency counts")
                    pair["outputs_identical"] = (
                        pair["baseline"]["output_sha256"] == pair["current"]["output_sha256"]
                    )
                    case["pairs"].append(pair)
                case.update(_paired_summary(case["pairs"]))
                report["cases"].append(case)
                (output / "revision-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({key: value for key, value in case.items() if key != "pairs"}), flush=True)
    report["state"] = "completed"
    (output / "revision-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--output-directory", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--baseline-directory", type=Path,
                        help="Optional already-built checkout for paired revision comparisons")
    parser.add_argument("--baseline-commit", help="Require this exact baseline commit")
    parser.add_argument("--comparison-sizes", nargs="+", type=int, default=[100_000, 1_000_000])
    parser.add_argument("--comparison-only", action="store_true",
                        help="Skip the existing upstream/fast screen and run only paired revisions")
    args = parser.parse_args()
    if args.n < 16:
        parser.error("--n must be at least 16")
    if any(n < 16 or n > 1_000_000 for n in args.comparison_sizes):
        parser.error("comparison sizes must lie between 16 and 1,000,000")
    if (args.comparison_only or args.baseline_commit) and args.baseline_directory is None:
        parser.error("revision comparisons require --baseline-directory")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    import lpsd
    import lpsd_fast
    from lpsd_fast.api import available_workers

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workers = available_workers()
    if args.baseline_directory is not None and args.baseline_commit is not None:
        observed = _revision_evidence(args.baseline_directory)["commit"]
        if observed != args.baseline_commit:
            parser.error("baseline checkout differs from --baseline-commit")
    if args.comparison_only:
        revision_comparison(root, args.baseline_directory, output, args.comparison_sizes, workers)
        return
    reports = [
        Path(lpsd.__file__).parent / "ltpda_dft.build.json",
        Path(lpsd_fast.__file__).parent / "_native" / "liblpsd_fast.build.json",
    ]
    environment = {
        "schema_version": 1,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "available_workers": workers,
        "numpy_long_double": {
            "bytes": np.dtype(np.longdouble).itemsize,
            "mantissa_bits_including_leading_bit": np.finfo(np.longdouble).nmant + 1,
            "epsilon": str(np.finfo(np.longdouble).eps),
        },
        "github": {key: os.environ.get(key) for key in (
            "GITHUB_SHA", "GITHUB_RUN_ID", "RUNNER_OS", "RUNNER_ARCH", "ImageOS", "ImageVersion"
        )},
        "build_reports": [json.loads(path.read_text()) for path in reports],
        "note": "Shared CI hardware and small inputs; no cross-machine speedup or hard timing claim.",
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    # Force both benchmark backends to this checkout, matching the source-tree
    # numerical tests and the build reports above, even with another installation.
    child_environment = dict(os.environ, PYTHONPATH=str(root))
    common = [sys.executable, str(root / "benchmarks" / "bench_lpsd.py"), "--n", str(args.n)]
    commands = [common + ["--backend", "original", "--repeats", "1",
                          "--output", str(output / "original-w1.json")]]
    for count in sorted({1, workers}):
        commands.append(common + [
            "--backend", "fast", "--kernel", "auto", "--workers", str(count),
            "--repeats", "3", "--profile", "--output", str(output / f"fast-auto-w{count}.json"),
        ])
        commands.append(common + [
            "--backend", "fast", "--kernel", "fast", "--workers", str(count),
            "--outputs", "psd", "--repeats", "3", "--profile",
            "--output", str(output / f"fast-selected-w{count}.json"),
        ])
    for command in commands:
        subprocess.run(command, cwd=root, env=child_environment, check=True)
        report = json.loads(Path(command[-1]).read_text())
        print(json.dumps({
            "report": Path(command[-1]).name,
            "median_wall_s": report["median_wall_s"],
            "fast_metadata": report.get("fast_metadata"),
            "profile_summary": report.get("additional_profiled_call", {}).get("summary"),
        }), flush=True)
    if args.baseline_directory is not None:
        revision_comparison(root, args.baseline_directory, output, args.comparison_sizes, workers)


if __name__ == "__main__":
    main()
