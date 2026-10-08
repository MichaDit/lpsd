# SPDX-License-Identifier: GPL-3.0-or-later
"""Show differences between LPSD and a whole-record FFTW log-periodogram.

These are different estimators, not implementations of one estimator. Relative
differences are descriptive, never an equivalence gate. This script is a small
numerical comparison, not a controlled wall-time benchmark. It saves the small
logarithmic output spectra, but never the input samples or full FFT spectrum.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

import lpsd_fast
from lpsd_fast import api

if __package__:
    from . import bench_fftw
    from .bench_lpsd import build_evidence
    from .check_accuracy import make_signal, source_evidence
else:
    import bench_fftw
    from bench_lpsd import build_evidence
    from check_accuracy import make_signal, source_evidence


CASES = ("white", "pink", "offbin_tone", "dc_nanovolt")


def signal_case(name, n, sample_rate, seed):
    if name == "offbin_tone":
        # Approximately 0.06246*fs, but unambiguously off the full-record grid
        # at every allowed N. This need not be off every individual LPSD segment.
        record_bin = np.floor(0.06246 * n) + 0.37
        frequency = record_bin * sample_rate / n
        values = np.sin(2 * np.pi * record_bin * np.arange(n, dtype=np.float64) / n)
        return values, {"generator": "unit-amplitude sine", "frequency_hz": float(frequency),
                        "full_record_bin": float(record_bin), "seed": None}
    values, actual_seed = make_signal(name, n, sample_rate, seed)
    description = {
        "white": "PCG64 standard-normal samples",
        "pink": "PCG64 white noise; rfft bins divided by sqrt(k); DC zero; irfft normalized to unit standard deviation",
        "dc_nanovolt": "10 + 1e-9 * PCG64 standard-normal samples, represented in float64",
    }[name]
    return values, {"generator": description, "seed": actual_seed}


def difference_summary(lpsd_values, fftw_values, frequencies, worst_points=5):
    """No amplitude floor; relative difference uses nonzero LPSD values only."""
    lpsd_values = np.asarray(lpsd_values, dtype=np.float64)
    fftw_values = np.asarray(fftw_values, dtype=np.float64)
    if lpsd_values.shape != fftw_values.shape or lpsd_values.shape != frequencies.shape:
        raise ValueError("Pointwise comparison requires exactly matching grids")
    if not np.isfinite(lpsd_values).all() or not np.isfinite(fftw_values).all():
        raise RuntimeError("Nonfinite density encountered; no finite comparison can be reported")
    absolute = np.abs(fftw_values - lpsd_values)
    nonzero = np.flatnonzero(lpsd_values != 0)
    relative = absolute[nonzero] / np.abs(lpsd_values[nonzero])
    zero = lpsd_values == 0

    def point(index):
        return {"index": int(index), "frequency_hz": float(frequencies[index]),
                "lpsd_psd": float(lpsd_values[index]), "fftw_log_psd": float(fftw_values[index]),
                "absolute_difference": float(absolute[index]),
                "relative_difference_to_lpsd": float(absolute[index] / abs(lpsd_values[index])) if lpsd_values[index] else None}

    largest_absolute = np.argsort(absolute)[::-1][:worst_points]
    largest_relative = nonzero[np.argsort(relative)[::-1][:worst_points]]
    return {
        "points": len(lpsd_values), "finite_pairs": len(lpsd_values),
        "nonzero_lpsd_points": len(nonzero),
        "zero_lpsd_points": int(np.count_nonzero(zero)),
        "zero_lpsd_nonzero_fftw_points": int(np.count_nonzero(zero & (fftw_values != 0))),
        "max_absolute_difference": float(np.max(absolute)),
        "mean_absolute_difference": float(np.mean(absolute)),
        "max_relative_difference_to_nonzero_lpsd": float(np.max(relative)) if len(relative) else None,
        "relative_difference_quantiles": {str(q): float(np.quantile(relative, q)) if len(relative) else None
                                          for q in (0.5, 0.9, 0.95, 0.99)},
        "descriptive_relative_counts": {
            "at_least_1_percent": int(np.count_nonzero(relative >= 0.01)),
            "at_least_10_percent": int(np.count_nonzero(relative >= 0.1)),
            "at_least_100_percent": int(np.count_nonzero(relative >= 1.0)),
        },
        "largest_absolute_difference_points": [point(i) for i in largest_absolute],
        "largest_relative_difference_points": [point(i) for i in largest_relative],
        "note": "Different estimators: none of these descriptive counts is an accuracy pass/fail criterion. No DC-scaled, peak-scaled or other floor is used.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=131073)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--n-frequencies", type=int, default=96)
    parser.add_argument("--n-averages", type=int, default=16)
    parser.add_argument("--psll", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--lpsd-kernel", choices=("fast", "auto", "scalar"), default="fast")
    parser.add_argument("--lpsd-workers", type=int, default=1)
    parser.add_argument("--fftw-threads", type=int, default=1)
    parser.add_argument("--fftw-library", type=Path)
    parser.add_argument("--threads-library", type=Path)
    parser.add_argument("--planner", choices=tuple(bench_fftw.PLANNERS), default="estimate")
    parser.add_argument("--planner-time-limit", type=float)
    parser.add_argument("--worst-points", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.n < 64 or args.n > np.iinfo(np.int32).max or args.n_frequencies < 2 or
            args.n_averages < 1 or args.seed < 0 or args.lpsd_workers < 1 or args.fftw_threads < 1):
        parser.error("Require 64 <= n <= INT_MAX, at least two frequencies, positive averages/workers/threads and a nonnegative seed")
    if not np.isfinite(args.sample_rate) or args.sample_rate <= 0 or not np.isfinite(args.psll):
        parser.error("Sample rate must be positive and finite; PSLL must be finite")
    if not 1 <= args.worst_points <= 20 or len(set(args.cases)) != len(args.cases):
        parser.error("Require 1..20 worst points and unique cases")
    if args.planner_time_limit is not None and (not np.isfinite(args.planner_time_limit) or args.planner_time_limit < 0):
        parser.error("Planner time limit must be finite and nonnegative")

    sources = source_evidence()
    sources["comparison_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sources["fftw_adapter_sha256"] = hashlib.sha256(Path(bench_fftw.__file__).read_bytes()).hexdigest()
    backend = bench_fftw.FFTWLibrary(args.fftw_library, args.threads_library, args.fftw_threads)
    # Sharing the actual loaded LPSD library avoids accidentally comparing
    # fast native Kaiser against an unrelated, expensive NumPy setup path.
    provider = bench_fftw.KaiserWindow("native", Path(api._LIB._name))
    common = dict(sample_rate=args.sample_rate, psll=args.psll,
                  n_frequencies=args.n_frequencies, n_averages=args.n_averages)
    backend.forget_wisdom()
    reports = []
    with bench_fftw.Periodogram(backend, args.n, **common, planner=args.planner,
                               time_limit=args.planner_time_limit, window_provider=provider,
                               aggregation="log", density="psd", output_dtype="float32") as pipeline:
        grid = {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in pipeline.grid.items()}
        plan = pipeline.fft.describe()
        frequencies = pipeline.frequencies
        widths = pipeline.grid["discrete_bandwidth_hz"]
        for name in args.cases:
            values, generator = signal_case(name, args.n, args.sample_rate, args.seed)
            input_hash = bench_fftw.digest_array(values)
            data = pd.Series(values, copy=False)
            lpsd_result = lpsd_fast.lpsd(data, **common, window_function="kaiser",
                                        detrending_order=0, use_c_core=True, outputs="psd",
                                        workers=args.lpsd_workers, kernel=args.lpsd_kernel)
            fftw_result = pipeline.compute(data)
            lpsd_frequencies = lpsd_result.index.to_numpy()
            exact = lpsd_frequencies.dtype == frequencies.dtype and np.array_equal(lpsd_frequencies, frequencies)
            if not exact:
                raise RuntimeError("Actual LPSD and FFTW group-label grids differ")
            if input_hash != bench_fftw.digest_array(values):
                raise RuntimeError("An estimator mutated the shared input")
            if np.iscomplexobj(lpsd_result.psd.to_numpy()):
                raise RuntimeError("Expected a real auto spectrum from LPSD")
            lpsd_values = lpsd_result.psd.to_numpy()
            fftw_values = fftw_result.psd.to_numpy()
            differences = difference_summary(lpsd_values, fftw_values, frequencies, args.worst_points)
            validation = pipeline.validation()
            if (not validation["finite"] or
                    not np.isclose(validation["full_periodogram_integrated_power"],
                                   validation["weighted_detrended_time_power"], rtol=2e-12, atol=1e-30) or
                    not np.isclose(validation["aggregated_positive_power"],
                                   validation["positive_bin_power"], rtol=2e-12, atol=1e-30)):
                raise RuntimeError("FFTW periodogram failed Parseval or bin-power conservation")
            centered = values - values[0]
            centered -= np.mean(centered)
            lpsd_quadrature = float(np.sum(lpsd_values.astype(np.float64) * widths))
            fftw_integral = float(np.sum(fftw_values.astype(np.float64) * widths))
            reports.append({
                "case": name, "signal": generator, "input_sha256": input_hash,
                "frequency_index_exact": True, "input_unchanged": True,
                "dtypes": {"input": str(values.dtype), "lpsd_psd": str(lpsd_values.dtype),
                           "fftw_log_psd": str(fftw_values.dtype), "fftw_internal": "float64"},
                "lpsd_metadata": lpsd_result.attrs.get("lpsd_fast"),
                "pointwise_differences": differences,
                "integrals": {
                    "global_detrended_unweighted_mean_square": float(np.mean(centered * centered)),
                    "lpsd_point_values_times_fft_discrete_widths": lpsd_quadrature,
                    "fftw_output_psd_times_discrete_widths": fftw_integral,
                    "fftw_output_integral_over_lpsd_quadrature": fftw_integral / lpsd_quadrature if lpsd_quadrature else None,
                    "fftw_output_rounding_integral_absolute_difference": abs(fftw_integral - validation["positive_bin_power"]),
                    "fftw_internal_validation": validation,
                    "note": "The LPSD weighted sum is a comparison quadrature of point estimates; it is not an exact LPSD Parseval identity. FFTW internal band sums conserve all positive-bin power; DC is separate.",
                },
                "small_log_spectra": {"lpsd_psd": lpsd_values.tolist(), "fftw_log_psd": fftw_values.tolist()},
                "output_sha256": {"lpsd_psd": bench_fftw.digest_array(lpsd_values),
                                  "fftw_log_psd": bench_fftw.digest_array(fftw_values)},
            })

    report = {
        "schema_version": 1, "comparison": "different estimators on identical frequency labels",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {**common, "n": args.n, "seed": args.seed, "cases": args.cases,
                       "detrending_order": 0, "lpsd_kernel": args.lpsd_kernel,
                       "lpsd_workers": args.lpsd_workers, "fftw_threads": args.fftw_threads,
                       "fftw_planner": args.planner, "planner_time_limit_s": args.planner_time_limit,
                       "padding_samples": 0, "requested_outputs": "psd"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
        "source_evidence": sources, "lpsd_build_reports": build_evidence(),
        "fftw": {"version": backend.version, "library": backend.evidence,
                 "threads_library": backend.threads_evidence, "plan": plan},
        "window_generation": provider.metadata, "grid": grid,
        "frequency_sha256": bench_fftw.digest_array(frequencies), "cases": reports,
        "notes": [
            "This is reproducible numerical evidence, not a controlled speed benchmark. Planning information is metadata; no runtime comparison is inferred from these calls.",
            "Default fs=50, target frequencies=96 and desired averages=16 form a compact numerical comparison, different from the throughput profile fs=1, target frequencies=1000, averages=100. Set --sample-rate 1 --n-frequencies 1000 --n-averages 100 to match those planning parameters.",
            "LPSD uses frequency-dependent segment lengths, overlapping segment estimates and detrending within each segment. FFTW uses one globally detrended complete record, one full-length window and averages neighboring Fourier-bin powers.",
            "The labels are exactly the actual LPSD frequencies, but the filters, leakage, resolution and variance differ. A common label does not make the estimators equivalent.",
            "Pointwise relative differences use exactly nonzero LPSD reference values without any amplitude floor. Counts above 1%, 10% and 100% are descriptive; there is no 1% pass/fail rule or universal equivalence assertion.",
            "The plotted PSD outputs share float32 precision; FFTW normalization and aggregation are done in float64 before that final rounding. The loaded native LPSD Kaiser generator also supplies the complete-record FFTW window.",
            "The FFTW bin-mean PSD integrates using counts*fs/N, not nominal midpoint boundary widths. DC is excluded from log groups and reported separately; even-N Nyquist is single-sided, while the highest odd-N bin is doubled.",
            "The full FFTW periodogram obeys a window-weighted Parseval identity. The sum of LPSD point values weighted by the FFT groups' discrete widths is only a comparison quadrature, not a conservation identity of LPSD.",
            "Finite-sample white and colored spectra have statistical variability. A single realization cannot establish a general bias, variance ratio or distributional equivalence.",
            "Only small logarithmic spectra and summary metadata are saved. Input data and the full-resolution FFT spectrum are regenerated, not stored.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(reports),
                      "frequencies": len(frequencies), "all_frequency_labels_exact": True,
                      "comparison": "different estimators; descriptive differences only"}))


if __name__ == "__main__":
    main()
