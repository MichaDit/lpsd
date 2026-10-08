# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit candidate PSD/NSD against scalar LPSD without hiding spectral nulls.

This produces numerical evidence, not a speed benchmark or a universal error
guarantee. Maxima and a few worst points are saved, not complete spectra.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

import lpsd
from lpsd._helpers import _kaiser_alpha, _kaiser_rov, c_core_available
from lpsd.flattop import HFT248D
import lpsd_fast
from lpsd_fast import api


SIGNALS = ("white", "pink", "tone_noise", "dc_nanovolt", "ramp_nanovolt",
           "offbin_tone", "onbin_tone", "weak_tone")
WINDOWS = {
    "kaiser200": (np.kaiser, 200., None),
    "kaiser80": (np.kaiser, 80., None),
    "hann": (np.hanning, 200., .5),
    "blackman": (np.blackman, 200., .5),
    "boxcar": (np.ones, 200., .5),
    "hft248d": (HFT248D, 200., .841),
}
CASES = {f"{signal}_kaiser200": (signal, "kaiser200") for signal in SIGNALS}
CASES.update({f"{signal}_{window}": (signal, window)
              for window in tuple(WINDOWS)[1:]
              for signal in ("white", "offbin_tone")})


def digest(values):
    values = np.ascontiguousarray(values)
    payload = values.dtype.str.encode() + str(values.shape).encode() + values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def make_signal(kind, n, sample_rate, base_seed):
    """Seeds depend on signal family, not case selection or execution order."""
    seed = base_seed + SIGNALS.index(kind)
    t = np.arange(n, dtype=np.float64) / sample_rate
    if kind in SIGNALS[:5]:
        noise = np.random.default_rng(seed).standard_normal(n)
    if kind == "white":
        values = noise
    elif kind == "pink":
        spectrum = np.fft.rfft(noise)
        spectrum[0] = 0
        spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
        values = np.fft.irfft(spectrum, n=n)
        values /= values.std()
    elif kind == "tone_noise":
        values = np.sin(2*np.pi*3.123*t) + .07*noise
    elif kind == "dc_nanovolt":
        values = 10. + 1e-9*noise
    elif kind == "ramp_nanovolt":
        values = np.linspace(-10., 10., n) + 1e-9*noise
    elif kind == "offbin_tone":
        values = np.sin(2*np.pi*3.123*t)
    elif kind == "onbin_tone":
        # An integer bin of the complete input record, not of every segment.
        values = np.sin(2*np.pi*17*np.arange(n, dtype=np.float64)/n)
    else:
        values = np.sin(2*np.pi*3.01*t) + 1e-8*np.sin(2*np.pi*8.04*t)
    return values, seed


def error_metrics(reference, candidate, frequencies, relative_limit=.01, worst_points=5):
    """Evaluate actual output values in float64; no scale-dependent floor."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if reference.shape != candidate.shape or reference.shape != frequencies.shape:
        raise ValueError("Error metrics require matching one-dimensional arrays.")
    if reference.ndim != 1 or not len(reference):
        raise ValueError("Error metrics require nonempty one-dimensional arrays.")
    finite = np.isfinite(reference) & np.isfinite(candidate)
    finite_indices = np.flatnonzero(finite)
    nonzero_indices = np.flatnonzero(finite & (reference != 0))
    zero_indices = np.flatnonzero(finite & (reference == 0))
    absolute = np.abs(candidate[finite_indices] - reference[finite_indices])
    relative = np.abs(candidate[nonzero_indices] - reference[nonzero_indices]) / np.abs(reference[nonzero_indices])
    changed_zeros = zero_indices[candidate[zero_indices] != 0]

    def point(index):
        error = abs(candidate[index] - reference[index])
        return {"index": int(index), "frequency": float(frequencies[index]),
                "reference": float(reference[index]), "candidate": float(candidate[index]),
                "absolute_error": float(error),
                "relative_error": float(error / abs(reference[index])) if reference[index] != 0 else None}

    worst_abs = int(finite_indices[np.argmax(absolute)]) if len(absolute) else None
    worst_rel = int(nonzero_indices[np.argmax(relative)]) if len(relative) else None
    top_relative = nonzero_indices[np.argsort(relative)[::-1][:worst_points]]
    top_zero = changed_zeros[np.argsort(np.abs(candidate[changed_zeros]))[::-1][:worst_points]]
    reference_nonfinite = int(np.count_nonzero(~np.isfinite(reference)))
    candidate_nonfinite = int(np.count_nonzero(~np.isfinite(candidate)))
    at_or_above = int(np.count_nonzero(relative >= relative_limit))
    return {
        "count": len(reference),
        "finite_pair_count": int(np.count_nonzero(finite)),
        "reference_nonfinite_count": reference_nonfinite,
        "candidate_nonfinite_count": candidate_nonfinite,
        "nonzero_reference_finite_pair_count": len(nonzero_indices),
        "reference_zero_count": int(np.count_nonzero(reference == 0)),
        "reference_zero_candidate_nonzero_count": len(changed_zeros),
        "reference_zero_candidate_nonfinite_count": int(np.count_nonzero((reference == 0) & ~np.isfinite(candidate))),
        "max_absolute_error": float(absolute.max()) if len(absolute) else None,
        "max_absolute_error_at": point(worst_abs) if worst_abs is not None else None,
        "max_relative_error_nonzero_reference": float(relative.max()) if len(relative) else None,
        "max_relative_error_at": point(worst_rel) if worst_rel is not None else None,
        "count_above_relative_limit": int(np.count_nonzero(relative > relative_limit)),
        "count_at_or_above_relative_limit": at_or_above,
        "relative_limit": relative_limit,
        "strict_limit_satisfied": not (reference_nonfinite or candidate_nonfinite or at_or_above or len(changed_zeros)),
        "worst_relative_points": [point(int(index)) for index in top_relative],
        "worst_changed_zero_points": [point(int(index)) for index in top_zero],
    }


def compare_outputs(reference, candidate, relative_limit=.01, worst_points=5):
    expected_f = reference.index.to_numpy()
    actual_f = candidate.index.to_numpy()
    exact = expected_f.dtype == actual_f.dtype and expected_f.tobytes() == actual_f.tobytes()
    result = {"frequency_index_exact": exact,
              "reference_frequencies": len(reference), "candidate_frequencies": len(candidate),
              "reference_frequency_sha256": digest(expected_f),
              "candidate_frequency_sha256": digest(actual_f)}
    if not exact:
        result["status"] = "frequency_mismatch"
        return result
    result["status"] = "compared"
    result["metrics"] = {}
    for label, ref_name, candidate_name in (("psd", "psd", "psd"), ("nsd", "asd", "nsd")):
        ref_values = reference[ref_name].to_numpy()
        candidate_values = candidate[candidate_name].to_numpy()
        if np.iscomplexobj(ref_values) or np.iscomplexobj(candidate_values):
            raise ValueError("This audit is for real auto spectra, not complex CSD.")
        metrics = error_metrics(ref_values, candidate_values, expected_f, relative_limit, worst_points)
        metrics.update({"reference_dtype": str(ref_values.dtype),
                        "candidate_dtype": str(candidate_values.dtype),
                        "reference_sha256": digest(ref_values),
                        "candidate_sha256": digest(candidate_values)})
        result["metrics"][label] = metrics
    return result


def source_evidence():
    manifest_path = Path(__file__).resolve().parents[1] / "test_fast" / "reference_sources.json"
    manifest = json.loads(manifest_path.read_text())
    original_root = Path(lpsd.__file__).resolve().parent
    original_hashes = {name: hashlib.sha256((original_root / name).read_bytes()).hexdigest()
                       for name in manifest["sha256"]}
    if original_hashes != manifest["sha256"]:
        raise RuntimeError("The original reference sources differ from the pinned upstream commit.")
    fast_root = Path(lpsd_fast.__file__).resolve().parent
    paths = [fast_root / "api.py", fast_root / "planning.py"]
    paths += sorted((fast_root / "_native").glob("*.c"))
    paths += sorted((fast_root / "_native").glob("*.h"))
    api._native()
    library = Path(api._LIB._name).resolve()
    original_library = original_root / ("ltpda_dft.dll" if sys.platform == "win32" else "ltpda_dft.so")
    return {
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_commit": manifest["commit"], "upstream_sources_match_pinned": True,
        "upstream_source_sha256": original_hashes,
        "upstream_native_library": str(original_library),
        "upstream_native_library_sha256": hashlib.sha256(original_library.read_bytes()).hexdigest() if original_library.exists() else None,
        "fast_module_path": str(fast_root),
        "fast_source_sha256": {str(path.relative_to(fast_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "native_library": str(library),
        "native_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }


def scalar_anchor():
    """Small independent original-C gate before using scalar as the reference."""
    if not c_core_available():
        raise RuntimeError("The original native C backend is required for the scalar anchor.")
    values = pd.Series(np.random.default_rng(177).standard_normal(257))
    kwargs = dict(sample_rate=50., n_frequencies=32, n_averages=4)
    original = lpsd.lpsd(values, use_c_core=True, **kwargs)
    reference = lpsd_fast.lpsd(values, kernel="scalar", workers=1, **kwargs)
    same = tuple(original.columns) == tuple(reference.columns)
    same = same and digest(original.index.to_numpy()) == digest(reference.index.to_numpy())
    same = same and all(digest(original[name].to_numpy()) == digest(reference[name].to_numpy()) for name in original)
    if not same:
        raise RuntimeError("Scalar failed the original-C anchor; do not use it as an independent reference.")
    return {"samples": len(values), "all_seven_columns_and_frequencies_bitwise_equal": True,
            "scalar_metadata": reference.attrs["lpsd_fast"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32769)
    parser.add_argument("--sample-rate", type=float, default=50.)
    parser.add_argument("--n-frequencies", type=int, default=96)
    parser.add_argument("--n-averages", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--kernel", default="auto", help="Candidate kernel; reference is always scalar.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES))
    parser.add_argument("--relative-limit", type=float, default=.01)
    parser.add_argument("--worst-points", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--fail-on-limit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n < 35 or args.n_frequencies < 2 or args.n_averages < 1 or args.workers < 1 or args.seed < 0:
        parser.error("Require n >= 35 (record bin 17 below Nyquist), frequencies >= 2, averages/workers >= 1 and seed >= 0.")
    if not np.isfinite(args.sample_rate) or args.sample_rate <= 2*8.04:
        parser.error("The fixed signal cases require a sample rate above 16.08 Hz.")
    if not np.isfinite(args.relative_limit) or args.relative_limit <= 0 or not 0 <= args.worst_points <= 20:
        parser.error("Require a positive finite relative limit and 0..20 saved worst points.")
    names = args.cases or list(CASES)
    if len(set(names)) != len(names):
        parser.error("Case names must not be repeated.")
    report = {
        "schema_version": 1, "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {"kernel": "scalar", "workers": 1, "outputs": "all", "nsd_reference_column": "asd"},
        "candidate": {"kernel": args.kernel, "workers": args.workers, "outputs": ["psd", "nsd"]},
        "configuration": {"n": args.n, "sample_rate": args.sample_rate,
                          "n_frequencies": args.n_frequencies, "n_averages": args.n_averages,
                          "base_seed": args.seed, "case_names": names,
                          "saved_worst_points_per_metric": args.worst_points},
        "relative_limit": args.relative_limit, "criterion": "strictly less than the relative limit",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform(),
                        "lpsd_fast_version": lpsd_fast.__version__,
                        "rng": type(np.random.default_rng(0).bit_generator).__name__},
        "source_evidence": source_evidence(), "scalar_anchor": scalar_anchor(),
        "notes": [
            "No DC-scaled, peak-scaled or other amplitude floor is applied.",
            "Relative errors use abs(candidate-reference)/abs(reference) at exactly nonzero finite references.",
            "Maxima use finite pairs; nonfinite values are counted separately and fail the strict criterion.",
            "Changed exact zeros are reported separately; their relative error is undefined and they fail --fail-on-limit.",
            "The 1% default is strict: exactly 1% does not pass. Both > and >= counts are recorded.",
            "Native scalar uses the strict coefficient/window path; an independent small original-C anchor gates it.",
            "Source and binary fingerprints are recorded before the audit; do not rebuild during the run.",
            "A finite synthetic audit cannot establish a universal guarantee, especially at spectral nulls.",
            "These calls are not controlled wall-time benchmarks; use bench_lpsd.py for speed comparisons.",
        ],
        "cases": [],
    }
    for name in names:
        kind, window = CASES[name]
        function, psll, overlap = WINDOWS[window]
        values, seed = make_signal(kind, args.n, args.sample_rate, args.seed)
        order = 1 if kind == "ramp_nanovolt" else None if kind in ("offbin_tone", "onbin_tone") else 0
        kwargs = dict(sample_rate=args.sample_rate, n_frequencies=args.n_frequencies,
                      n_averages=args.n_averages, detrending_order=order,
                      window_function=function, overlap=overlap, psll=psll)
        data = pd.Series(values, copy=False)
        reference = lpsd_fast.lpsd(data, kernel="scalar", workers=1, **kwargs)
        if reference.attrs["lpsd_fast"]["native_mode"] != 0:
            raise RuntimeError("Scalar reference unexpectedly selected another native mode.")
        candidate = lpsd_fast.lpsd(data, kernel=args.kernel, workers=args.workers,
                                  outputs=("psd", "nsd"), **kwargs)
        result = compare_outputs(reference, candidate, args.relative_limit, args.worst_points)
        result.update({"name": name, "signal": kind, "input_sha256": digest(values),
                       "parameters": dict(kwargs, window_function=window, n=args.n, seed=seed,
                           resolved_overlap=overlap if overlap is not None else _kaiser_rov(_kaiser_alpha(psll))),
                       "candidate_metadata": candidate.attrs["lpsd_fast"]})
        report["cases"].append(result)
    mismatches = [row["name"] for row in report["cases"] if not row["frequency_index_exact"]]
    outside = [row["name"] for row in report["cases"] if "metrics" in row
               and not all(metric["strict_limit_satisfied"] for metric in row["metrics"].values())]
    report["summary"] = {"cases": len(names), "frequency_mismatch_cases": mismatches,
                          "cases_not_meeting_strict_limit": outside,
                          "all_cases_meet_strict_limit": not mismatches and not outside}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(report["summary"], output=str(args.output))))
    return int(bool(mismatches) or (args.fail_on_limit and bool(outside)))


if __name__ == "__main__":
    sys.exit(main())
