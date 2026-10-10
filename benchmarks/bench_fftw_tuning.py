#!/usr/bin/env python3
"""Serial tuning experiments; final paired timings use bench_fftw_optimized.py.

Each configuration runs in a separate prepared block, keeping only one large
cache resident. These exploratory block timings choose candidates; they are
not presented as the final balanced before/after speedup measurement.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import argparse
from functools import partial
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import sys
import time
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from benchmarks import bench_fftw as bf
from benchmarks.matched_smoothing import MatchedSmoothing
from benchmarks.fftw_comparison_signals import signal
from benchmarks.bench_fftw_optimized import measured, summarize, difference, write_json, sha256
from lpsd_fast import api


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("smoothing", "lowcache", "totalcache", "lowworkers", "basic", "prime", "convolution", "fresh"), required=True)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--fftw-library", type=Path)
    p.add_argument("--threads-library", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weight-dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--smoothing-workers", type=int, default=8)
    p.add_argument("--low-workers", type=int, default=8)
    p.add_argument("--low-cache-mb", type=float, default=512)
    p.add_argument("--kernel-cache-mb", type=float, default=1536)
    p.add_argument("--total-cache-budgets", type=float, nargs="+", default=[2048, 4096])
    p.add_argument("--fftw-threads", type=int, default=8)
    p.add_argument("--convolution-lengths", type=int, nargs="+", default=[1512000, 1536000, 1600000, 2097152])
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    api._native()
    provider = bf.KaiserWindow("native", Path(api._LIB._name))
    x, _ = signal("white", args.n, 50., 20261010)
    fingerprint = bf.digest_array(x)
    common = dict(sample_rate=50., psll=200., n_frequencies=1000, n_averages=100,
                  window_provider=provider)
    default = dict(weight_dtype=args.weight_dtype, smoothing_workers=args.smoothing_workers,
                   workers=args.low_workers, window_cache_mb=args.low_cache_mb,
                   max_kernel_cache_mb=args.kernel_cache_mb, max_working_mb=2048,
                   operations_backend="native", smoothing_backend="native")
    configurations = []
    if args.stage == "smoothing":
        configurations = [{"id": f"{dtype}_s{workers}", "weight_dtype": dtype, "smoothing_workers": workers}
                          for dtype in ("float64", "float32") for workers in (1, 2, 4, 8)]
    elif args.stage == "lowcache":
        configurations = [{"id": f"low_cache_{cache}", "window_cache_mb": cache}
                          for cache in (0, 256, 512, 1024, 1536, 2048, 3072)]
    elif args.stage == "totalcache":
        configurations = [{"id": f"total_cache_{cache:g}", "total_cache_mb": cache,
                           "window_cache_mb": None} for cache in args.total_cache_budgets]
    elif args.stage == "lowworkers":
        configurations = [{"id": f"low_workers_{workers}", "workers": workers} for workers in (1, 2, 4, 8)]
    elif args.stage == "basic":
        configurations = [{"id": f"{operation}_t{threads}", "operations": operation, "fftw_threads": threads}
                          for operation in ("numpy", "native", "native_grouped") for threads in (1, 2, 4, 8)]
    elif args.stage == "prime":
        configurations = [{"id": f"{family}_{transform}_t{threads}", "family": family,
                           "transform": transform, "fftw_threads": threads}
                          for family in ("basic", "matched")
                          for transform in ("native", "bluestein") for threads in (4, 8)]
    elif args.stage == "convolution":
        configurations = [{"id": f"bluestein_m{length}_t{threads}_{operation}",
                           "family": "basic", "transform": "bluestein",
                           "fftw_threads": threads, "convolution_length": length,
                           "bluestein_ops": operation}
                          for length in args.convolution_lengths
                          for threads in (4, 8) for operation in ("native", "numpy")]
    elif args.stage == "fresh":
        configurations = [{"id": "prepared_defaults"},
                          {"id": "once", "max_kernel_cache_mb": 0, "window_cache_mb": 0,
                           "defer_normalization": True}]
    output = {"settings": vars(args), "purpose": __doc__,
              "native_binary_sha256": sha256(api._LIB._name),
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                                [Path(__file__), ROOT / "benchmarks/matched_smoothing.py",
                                 ROOT / "lpsd_fast/prepared.py", ROOT / "benchmarks/bench_fftw.py",
                                 ROOT / "benchmarks/_fftw_native.py", ROOT / "benchmarks/_fftw_bluestein.py"]},
              "input_sha256": fingerprint, "configurations": []}
    order = np.random.default_rng(202610104)
    references = {}
    for block in range(args.blocks):
        jobs = list(configurations)
        order.shuffle(jobs)
        for configuration in jobs:
            settings = dict(configuration)
            name = settings.pop("id")
            transform = settings.pop("transform", "native")
            family = settings.pop("family", "basic" if args.stage == "basic" else "matched")
            threads = settings.pop("fftw_threads", args.fftw_threads)
            backend = bf.FFTWLibrary(args.fftw_library, args.threads_library, threads)
            constructor = bf.Periodogram if family == "basic" else MatchedSmoothing
            parameters = (dict(operations="native_grouped", output_dtype="float32")
                          if family == "basic" else dict(default))
            parameters.update(settings)
            original_fft = bf.RealFFT
            if transform == "bluestein":
                from benchmarks._fftw_bluestein import BluesteinRealFFT
                parameters.pop("convolution_length", None)
                parameters.pop("bluestein_ops", None)
                bf.RealFFT = partial(BluesteinRealFFT,
                                     convolution_length=settings.get("convolution_length"),
                                     native_ops=settings.get("bluestein_ops", "auto"))
            print("TUNE", args.stage, args.n, "block", block, name, flush=True)
            try:
                def create():
                    backend.forget_wisdom()
                    return constructor(backend, args.n, **common, **parameters)
                if args.stage == "fresh":
                    def calculate():
                        with create() as obj:
                            return obj.compute(x, outputs="nsd")
                    calculate()
                    times = []
                    for _ in range(args.repeats):
                        result, timing = measured(calculate)
                        times.append(timing)
                    metadata, profile, setup = {}, None, None
                else:
                    obj, setup = measured(create)
                    with obj:
                        result = obj.compute(x, outputs="nsd")
                        obj.compute(x, outputs="nsd")
                        times = []
                        for _ in range(args.repeats):
                            again, timing = measured(lambda: obj.compute(x, outputs="nsd"))
                            times.append(timing)
                            np.testing.assert_array_equal(again.nsd, result.nsd)
                        metadata = {"pipeline": getattr(obj, "metadata", {}), "fft": obj.fft.describe()}
                        if hasattr(obj, "profile"):
                            obj.profile(x, outputs="nsd")
                            profile = obj.last_profile
                            obj._low_prepared._compute_validated(x, profile=True)
                            profile = {**profile, "separate_low_profile": obj._low_prepared.last_profile}
                        else:
                            obj.compute(x, outputs="nsd", profile=True)
                            profile = obj.last_profile
                values = result.nsd.to_numpy().copy()
                if family not in references:
                    references[family] = values
                reference = references[family]
                np.testing.assert_allclose(values, reference, rtol=3e-6, atol=np.max(reference) * 2e-14)
                row = {"id": name, "block": block, "family": family, "transform": transform,
                       "fftw_threads": threads, "parameters": parameters, "setup": setup,
                       "timing": summarize(times), "difference_to_first_family_configuration": difference(values, reference),
                       "metadata": metadata, "profile": profile}
                # A native provider is not a JSON value; its provenance is
                # already recorded through metadata and binary hashes.
                output["configurations"].append(row)
                write_json(args.output, output)
                print("  median", round(row["timing"]["median_wall_s"] * 1000, 4), "ms", flush=True)
            finally:
                bf.RealFFT = original_fft
            if "obj" in locals():
                del obj
            gc.collect()
            # Drop allocator-held pages between large configurations. This is
            # outside all measured intervals and is optional off glibc.
            try:
                trim = ct.CDLL(None).malloc_trim
                trim.argtypes, trim.restype = [ct.c_size_t], ct.c_int
                trim(0)
            except AttributeError:
                pass
    assert bf.digest_array(x) == fingerprint
    output["input_unchanged"] = True
    output["max_rss_bytes"] = bf.max_rss_bytes()
    write_json(args.output, output)


if __name__ == "__main__":
    main()
