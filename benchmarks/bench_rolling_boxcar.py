#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Matched full-API Boxcar timing against a separately built source checkout.

Run builds and this benchmark without other CPU benchmarks. Both implementations
receive the same input array and complete parameters. Every case warms each API
on the complete input and alternates baseline/candidate then candidate/baseline
timing order. Profiles are separate instrumented calls, never speed samples.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

from lpsd_fast.build import source_manifest


def load_package(root, name):
    package = Path(root).resolve() / "lpsd_fast"
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evidence(module):
    api = sys.modules[module.__name__ + ".api"]
    api._native()
    package = Path(module.__file__).resolve().parent
    library = Path(api._LIB._name).resolve()
    sources = sorted(package.glob("*.py")) + sorted((package / "_native").glob("*.c"))
    sources += sorted((package / "_native").glob("*.h"))
    report = library.with_suffix(".build.json")
    library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
    build_report = json.loads(report.read_text())
    if build_report["binary_sha256"] != library_sha256:
        raise AssertionError("Native build report does not match the loaded library")
    native_sources = source_manifest(package / "_native" / build_report["source"])
    bound_sources = build_report.get("source_sha256")
    if bound_sources is not None and bound_sources != native_sources:
        raise AssertionError("Native sources do not match the sources used to build the library")
    shared_sources = {}
    for name, imported in tuple(sys.modules.items()):
        filename = getattr(imported, "__file__", None)
        if filename and (name == "lpsd" or name.startswith("lpsd.")):
            path = Path(filename).resolve()
            shared_sources[name] = {"path": str(path),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "package": str(package), "version": module.__version__,
        "library": str(library),
        "library_sha256": library_sha256,
        "build_report": build_report,
        "native_source_binding": "verified" if bound_sources is not None else "unavailable_legacy_build_report",
        "native_source_sha256": native_sources,
        "shared_python_sources": shared_sources,
        "source_sha256": {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }


def metrics(expected, actual):
    if expected.index.to_numpy().tobytes() != actual.index.to_numpy().tobytes():
        raise AssertionError("Frequency grids differ")
    if not expected.dtypes.equals(actual.dtypes):
        raise AssertionError("PSD output dtypes differ")
    if np.iscomplexobj(expected.psd.to_numpy()) or np.iscomplexobj(actual.psd.to_numpy()):
        raise AssertionError("A real auto spectrum became complex")
    left = expected.psd.to_numpy().astype(np.float64)
    right = actual.psd.to_numpy().astype(np.float64)
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise AssertionError("A benchmark spectrum is nonfinite")
    absolute = np.abs(left-right)
    nonzero = left != 0
    relative = absolute[nonzero]/np.abs(left[nonzero])
    return {"index_exact": True, "dtypes_equal": expected.dtypes.equals(actual.dtypes),
            "max_absolute_error": float(absolute.max()),
            "max_relative_error_nonzero_reference": float(relative.max()) if len(relative) else None,
            "points_at_or_above_one_percent": int((relative >= .01).sum()),
            "changed_exact_zeros": int(np.count_nonzero(right[~nonzero]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--n", nargs="+", type=int, default=[1_000_000, 10_000_000])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--overlap", nargs="+", type=float, default=[.8, .9, .95])
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--frequencies", type=int, default=1000)
    parser.add_argument("--averages", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 2 or any(n < 1000 for n in args.n) or any(w < 1 for w in args.workers):
        parser.error("Use at least two repetitions, n>=1000, workers>=1")
    baseline = load_package(args.baseline_root, "lpsd_boxcar_baseline")
    candidate = load_package(Path(__file__).resolve().parents[1], "lpsd_boxcar_candidate")
    output = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline": evidence(baseline), "candidate": evidence(candidate),
        "methodology": {
            "seed": 20261008, "signal": "float64 standard normal, resident in RAM",
            "sample_rate": 1., "warmup": "one complete call per implementation per case",
            "timing": "wall clock, complete lpsd API, alternating within each pair",
            "excludes": "input creation, module imports, build, provenance, separate profiles",
            "profile": "one separate call per implementation after ordinary repetitions",
            "statistics": "median of saved wall-clock repetitions; shared-host noise remains",
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        values = np.random.default_rng(20261008).standard_normal(n)
        for overlap in args.overlap:
            for workers in args.workers:
                kwargs = dict(sample_rate=1., window_function="boxcar", overlap=overlap,
                              n_frequencies=args.frequencies, n_averages=args.averages,
                              kernel="fast", workers=workers, max_working_mb=4096,
                              outputs="psd")
                modules = {"baseline": baseline, "candidate": candidate}
                for module in modules.values():
                    module.lpsd(values, **kwargs)
                repetitions = {name: [] for name in modules}
                observed_order = []
                spectra = {}
                for repetition in range(args.repeat):
                    order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
                    for name in order:
                        started = time.perf_counter()
                        spectra[name] = modules[name].lpsd(values, **kwargs)
                        repetitions[name].append(time.perf_counter()-started)
                        observed_order.append(name)
                profiles = {name: module.lpsd(values, **kwargs, profile=True).attrs["lpsd_profile"]
                            for name, module in modules.items()}
                baseline_median = statistics.median(repetitions["baseline"])
                candidate_median = statistics.median(repetitions["candidate"])
                case = {
                    "n": n, "overlap": overlap, "workers": workers,
                    "parameters": kwargs, "actual_frequencies": len(spectra["candidate"]),
                    "rolling_frequencies": spectra["candidate"].attrs["lpsd_fast"]["rolling_boxcar_frequencies"],
                    "observed_order": observed_order, "wall_repetitions_s": repetitions,
                    "baseline_median_s": baseline_median, "candidate_median_s": candidate_median,
                    "speedup": baseline_median/candidate_median,
                    "numerical_comparison": metrics(spectra["baseline"], spectra["candidate"]),
                    "profiles": profiles,
                }
                output["cases"].append(case)
                args.output.write_text(json.dumps(output, indent=2)+"\n")
                print(json.dumps({key: case[key] for key in (
                    "n", "overlap", "workers", "rolling_frequencies",
                    "baseline_median_s", "candidate_median_s", "speedup")}), flush=True)


if __name__ == "__main__":
    main()
