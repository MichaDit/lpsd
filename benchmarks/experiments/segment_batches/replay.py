# SPDX-License-Identifier: GPL-3.0-or-later
"""Alternate complete LPSD calls using captured native segment kernels.

Each configuration receives one full problem warm-up per library. Every API
call creates its own window cache and coefficient workspaces. Timed work does
not include imports, input creation, hashes or JSON writes. Profiled calls are
separate from ordinary timing samples.
"""
import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_pair(artifacts):
    report = json.loads((artifacts / "builds.json").read_text())
    client = artifacts / report["client_directory"]
    for relative, expected in report["client_sources"].items():
        if sha256((client / Path(relative).name).read_bytes()) != expected:
            raise RuntimeError(f"Captured Python client changed: {relative}")
    for name, build in report["builds"].items():
        path = artifacts / build["binary"]
        if sha256(path.read_bytes()) != build["build"]["binary_sha256"]:
            raise RuntimeError(f"Captured native library changed: {path}")
        sources = build["source_sha256"]
        manifest_hash = sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode())
        if manifest_hash != build["source_snapshot_sha256"]:
            raise RuntimeError("Captured source manifest changed")
        for relative, expected in sources.items():
            if sha256((artifacts / build["source_directory"] / relative).read_bytes()) != expected:
                raise RuntimeError(f"Captured native source changed: {relative}")
    module_name = "_lpsd_segment_batch_client"
    spec = importlib.util.spec_from_file_location(module_name, client / "__init__.py", submodule_search_locations=[str(client)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = package
    spec.loader.exec_module(package)
    api = sys.modules[module_name + ".api"]
    api._native()
    declarations = api._LIB
    libraries = {}
    for name, build in report["builds"].items():
        path = artifacts / build["binary"]
        library = ct.CDLL(str(path))
        for symbol, declaration in vars(declarations).copy().items():
            if isinstance(declaration, ct._CFuncPtr):
                function = getattr(library, symbol)
                function.argtypes, function.restype = declaration.argtypes, declaration.restype
        libraries[name] = library
    return api, libraries, report


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--output", type=Path, default=HERE / "replay-results.json")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000_000])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--profiles", action="store_true")
    args = parser.parse_args(argv)
    if min(args.sizes) < 16 or min(args.workers) < 1 or args.pairs < 1:
        parser.error("Require sizes >= 16, workers >= 1 and pairs >= 1")
    api, libraries, builds = load_pair(args.artifacts.resolve())
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
            "memory_max": read_optional("/sys/fs/cgroup/memory.max"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "builds": builds,
        "configuration": {
            "sizes": args.sizes, "workers": args.workers, "pairs": args.pairs,
            "seed": args.seed, "sample_rate": 1.0, "n_frequencies": 1000,
            "n_averages": 100, "kernel": "fast", "outputs": "psd",
            "window": "kaiser", "psll": 200.0, "order": 0,
            "max_working_mb": 4096, "input": "float64 pandas Series",
            "warmups": "One complete call per library/configuration; fresh preparation on every call",
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for n in args.sizes:
        samples = pd.Series(np.random.default_rng(args.seed).normal(size=n), copy=False)
        for workers in args.workers:
            kwargs = dict(sample_rate=1.0, n_frequencies=1000, n_averages=100,
                          kernel="fast", outputs="psd", workers=workers, max_working_mb=4096)
            for name in ("baseline", "candidate"):
                api._LIB = libraries[name]
                api.lpsd(samples, **kwargs)
            case = {"n": n, "workers": workers, "samples": [], "profiles": {}}
            report["cases"].append(case)
            reference = None
            for pair in range(args.pairs):
                names = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
                for name in names:
                    api._LIB = libraries[name]
                    start = time.perf_counter()
                    result = api.lpsd(samples, **kwargs)
                    wall_s = time.perf_counter() - start
                    values = result.psd.to_numpy()
                    if np.iscomplexobj(values) or not np.isfinite(values).all():
                        raise RuntimeError("Finite white-noise PSD unexpectedly became complex or nonfinite")
                    if reference is None:
                        reference = values.copy()
                        case["frequency"] = result.index.to_list()
                        case["baseline_psd"] = reference.tolist()
                    entry = {"name": name, "pair": pair, "wall_s": wall_s,
                             "dtype": str(values.dtype), "psd_sha256": sha256(values.tobytes()),
                             "byte_equal_to_baseline": values.tobytes() == reference.tobytes(),
                             "max_absolute_psd_error": float(np.max(np.abs(values.astype(np.float64) - reference)))}
                    case["samples"].append(entry)
                    print(n, workers, entry, flush=True)
                    save()
            case["median_wall_s"] = {
                name: statistics.median(row["wall_s"] for row in case["samples"] if row["name"] == name)
                for name in ("baseline", "candidate")
            }
            case["median_ratio"] = case["median_wall_s"]["baseline"] / case["median_wall_s"]["candidate"]
            if args.profiles:
                for name in ("baseline", "candidate"):
                    api._LIB = libraries[name]
                    result = api.lpsd(samples, profile=True, **kwargs)
                    case["profiles"][name] = result.attrs["lpsd_profile"]
            save()
    print(args.output)


if __name__ == "__main__":
    main()
