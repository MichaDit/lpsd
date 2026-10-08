# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure one public LPSD configuration per process; no sample files are saved.

Input generation and warm-up are outside the API timer. An optional additional
profiled call is kept separate from the ordinary timing repetitions.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd

import lpsd as original_module
from lpsd import lpsd as original_lpsd
from lpsd._helpers import c_core_available
from lpsd.flattop import HFT248D
import lpsd_fast as fast_module
from lpsd_fast import lpsd as fast_lpsd

try:
    import resource
except ImportError:
    resource = None

OUTPUTS = ("ps", "psd", "ps_std", "psd_std", "enbw", "asd", "asdrms", "nsd")
WINDOWS = {
    "kaiser": np.kaiser,
    "hann": np.hanning,
    "hanning": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "bartlett": np.bartlett,
    "boxcar": np.ones,
    "hft248d": HFT248D,
}


def max_rss_bytes():
    """Process-lifetime high-water mark; only normalize known platform units."""
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    if sys.platform.startswith("linux"):
        return int(value * 1024)
    return None


def cgroup_value(name):
    try:
        return (Path("/sys/fs/cgroup") / name).read_text().strip()
    except OSError:
        return None


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_evidence():
    """Read optional build sidecars and verify they describe the binary present."""
    reports = {}
    for name, module, relative in (
        ("original", original_module, "ltpda_dft"),
        ("fast", fast_module, "_native/liblpsd_fast"),
    ):
        prefix = Path(module.__file__).resolve().parent / relative
        path = prefix.with_suffix(".build.json")
        if path.is_file():
            report = json.loads(path.read_text())
            library = path.parent / report["library"]
            if hashlib.sha256(library.read_bytes()).hexdigest() != report["binary_sha256"]:
                raise RuntimeError(f"Build report does not match {library}")
            reports[name] = report
    return reports


def fingerprint(frame):
    """Hash index and individual columns without coercing mixed output dtypes."""
    digest = hashlib.sha256()
    for name, values in [("frequency", frame.index.to_numpy())] + [
        (str(column), frame[column].to_numpy()) for column in frame.columns
    ]:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(values.dtype.str.encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def profile_summary(profile):
    rows = profile.get("frequencies", [])
    stages = (
        "window_and_sums_s",
        "coefficients_s",
        "c_kernel_s",
        "c_preparation_s",
        "c_segments_s",
        "worker_elapsed_s",
    )
    stages += tuple(
        stage for stage in ("window_generation_s", "window_sums_s", "memory_gate_wait_s")
        if rows and all(stage in row for row in rows)
    )
    return {
        "sample_iterations": sum(int(row["sample_iterations"]) for row in rows),
        "window_cache_hits": sum(bool(row["window_cache_hit"]) for row in rows),
        "sum_worker_stage_seconds": {
            stage: sum(float(row[stage]) for row in rows) for stage in stages
        },
        "channel_phase_seconds": {
            stage: float(profile[stage])
            for stage in ("planning_s", "input_conversion_s", "output_assembly_s")
            if profile.get(stage) is not None
        },
        "note": (
            "Worker elapsed times overlap; these sums are not total wall time. "
            "C preparation and segments subdivide C kernel time. "
            "Window generation and sums subdivide window_and_sums time; "
            "do not add subdivisions to their parent phase."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("original", "fast"), required=True)
    parser.add_argument("--entry-point", choices=("lpsd", "lnsd"), default="lpsd")
    parser.add_argument(
        "--outputs", nargs="+", choices=("all",) + OUTPUTS,
        help="Fast LPSD output columns, in order; omitted means all seven legacy columns.",
    )
    parser.add_argument("--window", choices=tuple(WINDOWS), default="kaiser")
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--psll", type=float, default=200.)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--kernel", choices=("auto", "fast", "scalar", "simd", "projected"), default="auto"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--n-frequencies", type=int, default=1000)
    parser.add_argument("--n-averages", type=int, default=100)
    parser.add_argument("--order", type=int, choices=range(-1, 11), default=0)
    parser.add_argument("--max-working-mb", type=float)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n < 16 or args.repeats < 1 or args.warmups < 0 or args.workers < 1:
        parser.error("Require n >= 16, repeats >= 1, warmups >= 0 and workers >= 1.")
    if (
        args.n_frequencies < 2
        or args.n_averages < 1
        or not np.isfinite(args.sample_rate)
        or args.sample_rate <= 0
    ):
        parser.error(
            "Require at least 2 target frequencies, positive averages "
            "and a positive finite sample rate."
        )
    if args.max_working_mb is not None and (
        not np.isfinite(args.max_working_mb) or args.max_working_mb <= 0
    ):
        parser.error("--max-working-mb must be positive and finite.")
    if not np.isfinite(args.psll):
        parser.error("--psll must be finite.")
    if args.overlap is not None and (
        not np.isfinite(args.overlap) or not 0 <= args.overlap < 1
    ):
        parser.error("--overlap must be finite and in [0, 1).")
    if args.window != "kaiser" and args.overlap is None:
        parser.error("Non-Kaiser windows require an explicit --overlap.")
    if args.outputs and (
        len(set(args.outputs)) != len(args.outputs)
        or ("all" in args.outputs and args.outputs != ["all"])
    ):
        parser.error("Output names must be unique; 'all' must be used alone.")
    if args.entry_point == "lnsd" and (args.backend != "fast" or args.outputs is not None):
        parser.error("--entry-point lnsd requires --backend fast and cannot be combined with --outputs.")
    if args.backend == "original" and args.outputs not in (None, ["all"]):
        parser.error("Selected outputs require --backend fast; original always computes all outputs.")
    if args.backend == "original" and (
        args.profile or args.workers != 1 or args.max_working_mb is not None
    ):
        parser.error(
            "Profiling, multiple workers and memory-budget options "
            "require --backend fast."
        )
    if (
        args.backend == "fast"
        and args.kernel == "projected"
        and args.order not in (0, 1)
    ):
        parser.error("The projected kernel supports only detrending orders 0 and 1.")
    if args.backend == "original" and not c_core_available():
        parser.error("Original native C core unavailable. Build/install it before benchmarking.")

    input_start = time.perf_counter()
    samples = np.random.default_rng(args.seed).normal(size=args.n)
    # Keep the public input container identical for both implementations.
    series = pd.Series(samples, copy=False)
    setup_input_s = time.perf_counter() - input_start
    common = {
        "sample_rate": args.sample_rate,
        "n_frequencies": args.n_frequencies,
        "n_averages": args.n_averages,
        "detrending_order": args.order,
        "window_function": WINDOWS[args.window],
        "overlap": args.overlap,
        "psll": args.psll,
        "use_c_core": True,
    }
    implementation = original_lpsd if args.backend == "original" else (
        fast_module.lnsd if args.entry_point == "lnsd" else fast_lpsd
    )
    extra = {} if args.backend == "original" else {
        "workers": args.workers,
        "kernel": args.kernel,
        "max_working_mb": args.max_working_mb,
    }
    if args.backend == "fast" and args.outputs is not None:
        extra["outputs"] = args.outputs[0] if len(args.outputs) == 1 else args.outputs
    warmup_start = time.perf_counter()
    warmup_kwargs = dict(common, n_frequencies=min(args.n_frequencies, 100))
    for _ in range(args.warmups):
        implementation(series.iloc[:4096], **warmup_kwargs, **extra)
    setup_warmup_s = time.perf_counter() - warmup_start

    timings = []
    for repeat in range(args.repeats):
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        result = implementation(series, **common, **extra)
        wall_s = time.perf_counter() - wall_start
        cpu_s = time.process_time() - cpu_start
        timings.append({
            "repeat": repeat,
            "wall_s": wall_s,
            "cpu_s": cpu_s,
            "process_max_rss_bytes": max_rss_bytes(),
            "actual_frequencies": len(result),
            "output_sha256": fingerprint(result),
        })

    report = {
        "schema_version": 1,
        "backend": args.backend,
        "entry_point": args.entry_point,
        "parameters": dict(common, window_function=args.window,
                           n=args.n, seed=args.seed, **extra),
        "requested_outputs": "nsd" if args.entry_point == "lnsd" else args.outputs,
        "input": "pandas.Series of seeded float64 standard-normal samples",
        "warmup": {
            "calls": args.warmups,
            "samples_per_call": min(args.n, 4096),
            "target_frequencies": warmup_kwargs["n_frequencies"],
            "elapsed_s": setup_warmup_s,
        },
        "setup_input_s": setup_input_s,
        "repeats": timings,
        "median_wall_s": statistics.median(row["wall_s"] for row in timings),
        "actual_frequencies": len(result),
        "output_dtypes": {str(column): str(result[column].dtype) for column in result},
        "fast_metadata": result.attrs.get("lpsd_fast"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lpsd_distribution": installed_version("lpsd"),
            "lpsd_module_version": getattr(original_module, "__version__", None),
            "lpsd_fast_module_version": getattr(fast_module, "__version__", None),
            "platform": platform.platform(),
            "cpu_affinity_count": (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "cgroup_cpu_max": cgroup_value("cpu.max"),
            "cgroup_memory_max": cgroup_value("memory.max"),
        },
        "build_reports": build_evidence(),
        "notes": [
            "Imports, input generation, warm-up, fingerprints and file output are outside API timers.",
            "With zero warm-ups the first timed call may include native loading or lazy-build setup.",
            "RSS is a process-lifetime high-water mark, not a per-call allocation or memory-budget guarantee.",
            "Output fingerprints are reproducibility metadata, not a numerical correctness proof.",
            "Build sidecars are recorded when present and checked against binary SHA256; missing reports require separate compiler evidence.",
        ],
    }
    if args.profile:
        profile_start = time.perf_counter()
        profiled = implementation(series, **common, **extra, profile=True)
        profile_wall_s = time.perf_counter() - profile_start
        profile = profiled.attrs["lpsd_profile"]
        report["additional_profiled_call"] = {
            "wall_s": profile_wall_s,
            "summary": profile_summary(profile),
            "detail": profile,
        }

    payload = json.dumps(report, indent=2, default=json_scalar, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": args.backend,
        "n": args.n,
        "actual_frequencies": len(result),
        "median_wall_s": report["median_wall_s"],
        "measured_calls": len(timings),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
