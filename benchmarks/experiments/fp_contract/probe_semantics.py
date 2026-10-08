#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small semantic probe of experimental FMA; no timing measurements.

Compare complex outputs without dropping their imaginary parts. This is a
diagnostic probe, not an approval gate or a general numerical guarantee.
"""
import argparse
import json
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4097)
    args = parser.parse_args()
    if args.n < 35:
        parser.error("n must be at least 35")
    import numpy as np
    import pandas as pd
    from lpsd_fast import api

    from support import load_libraries
    baseline, candidate, libraries = load_libraries()

    def maximum(values):
        return float(np.max(values)) if len(values) else None

    noise = np.random.default_rng(20261008).standard_normal(args.n)
    t = np.arange(args.n, dtype=np.float64) / 50.
    tone = np.sin(2*np.pi*3.123*t)
    cases = []
    for amplitude in (1., 1e6, 1e9):
        for name, values in (("white", noise), ("offbin_tone", tone),
                             ("dc_nanovolt", 10.+1e-9*noise)):
            cases.append((name, amplitude, False, amplitude*values))
        cases.append(("phase_shifted_csd", amplitude, True, pd.DataFrame({
            "x": amplitude*(tone+.07*noise),
            "y": amplitude*(np.sin(2*np.pi*3.123*t+.7)+.03*noise),
        })))
    records = []
    for name, amplitude, csd, values in cases:
        frames = {}
        captured = {}
        outputs = ("psd", "psd_std", "asd") if csd else ("psd", "psd_std", "nsd")
        for variant, library in (("baseline", baseline), ("fma", candidate)):
            api._LIB = library
            with warnings.catch_warnings(record=True) as found:
                warnings.simplefilter("always")
                frames[variant] = api.lpsd(values, sample_rate=50.,
                    kernel="fast", workers=1, n_frequencies=32, n_averages=4,
                    detrending_order=None if name == "offbin_tone" else 0,
                    csd=csd, outputs=outputs)
            captured[variant] = [str(item.message) for item in found]
        reference, result = frames["baseline"], frames["fma"]
        columns = {}
        for column in outputs:
            a = reference[column].to_numpy().astype(np.complex128)
            b = result[column].to_numpy().astype(np.complex128)
            finite = np.isfinite(a) & np.isfinite(b)
            nonzero = finite & (a != 0)
            zero = finite & (a == 0)
            absolute = np.abs(b[finite]-a[finite])
            relative = np.abs(b[nonzero]-a[nonzero])/np.abs(a[nonzero])
            columns[column] = {
                "baseline_dtype": str(reference[column].dtype),
                "fma_dtype": str(result[column].dtype),
                "baseline_nonfinite": int(np.count_nonzero(~np.isfinite(a))),
                "fma_nonfinite": int(np.count_nonzero(~np.isfinite(b))),
                "baseline_imag_max": maximum(np.abs(a[np.isfinite(a)].imag)),
                "fma_imag_max": maximum(np.abs(b[np.isfinite(b)].imag)),
                "new_imaginary_points": int(np.count_nonzero(finite & (a.imag == 0) & (b.imag != 0))),
                "complex_max_absolute_error": maximum(absolute),
                "complex_max_relative_error": maximum(relative),
                "changed_exact_zeros": int(np.count_nonzero(zero & (b != 0))),
            }
        records.append({"signal": name, "amplitude": amplitude, "csd": csd,
                        "n": args.n, "frequencies": len(reference),
                        "frequency_index_exact": bool(reference.index.equals(result.index)),
                        "columns": columns, "warnings": captured})
    payload = {"libraries": libraries, "cases": records,
               "summary": {"cases": len(records),
                   "dtype_mismatch_columns": sum(c["baseline_dtype"] != c["fma_dtype"] for r in records for c in r["columns"].values()),
                   "new_imaginary_psd_points": sum(r["columns"]["psd"]["new_imaginary_points"] for r in records if not r["csd"]),
                   "nonfinite_values": sum(c["baseline_nonfinite"] + c["fma_nonfinite"] for r in records for c in r["columns"].values())},
               "notes": ["No execution times are measured.",
                         "Complex differences retain imaginary components; no float casting hides them.",
                         "Baseline and candidate both use public kernel=fast; only compiler contraction differs.",
                         "This finite probe does not prove a universal error bound."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print(json.dumps(payload["summary"]))


if __name__ == "__main__":
    main()
