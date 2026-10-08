# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the stored FFTW/LPSD evidence without running either estimator.

Example, from the repository root:
    python benchmarks/plot_comparisons.py \
        --results benchmarks/results_fftw.json \
        --estimators benchmarks/results_estimators.json \
        --output-directory docs/figures

Requires matplotlib and numpy. Figures retain source SHA256 values in their
metadata. Fixed SVG ids and omitted creation dates make reruns deterministic
within the same matplotlib/font environment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SIZES = (1_000_000, 1_000_003, 10_000_000, 30_000_000)
SERIES = (
    ("lpsd", "LPSD fast.2 — complete API", "#0072B2"),
    ("fftw_raw", "FFTW — raw r2c, reused plan", "#777777"),
    ("fftw_prepared", "FFTW — prepared log-PSD pipeline", "#009E73"),
    ("fftw_cold", "FFTW — fresh-plan log-PSD pipeline", "#E69F00"),
)


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "savefig.facecolor": "white",
        "figure.facecolor": "white", "svg.fonttype": "none",
        "svg.hashsalt": "lpsd-fftw-evidence-v1",
    })


def save(fig, directory, stem, title, sources):
    provenance = json.dumps({"sources": sources, "matplotlib": matplotlib.__version__,
                             "script": "benchmarks/plot_comparisons.py"}, sort_keys=True)
    paths = []
    for extension in ("svg", "png"):
        path = directory / f"{stem}.{extension}"
        metadata = {"Title": title, "Description": provenance}
        if extension == "svg":
            metadata.update(Date=None, Creator="benchmarks/plot_comparisons.py")
        else:
            metadata["Software"] = f"matplotlib {matplotlib.__version__}"
        fig.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata)
        if extension == "svg":
            # Matplotlib leaves spaces before path-data newlines. Normalize
            # those without changing SVG tokens or rendering.
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        paths.append(str(path))
    plt.close(fig)
    return paths


def performance_figure(report):
    selected = [r for r in report["api_comparison"] if r["threads"] == 8 and r["n"] in SIZES]
    if len(selected) != len(SIZES) or {r["n"] for r in selected} != set(SIZES):
        raise ValueError("Need exactly one 8-thread API row for each of the four specified sizes")
    by_n = {r["n"]: r for r in selected}
    rows = [by_n[n] for n in SIZES]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.25, top=0.76)
    x = np.arange(len(rows), dtype=float)
    width = 0.19
    all_values = []
    for column, (prefix, label, color) in enumerate(SERIES):
        median, lower, upper = [], [], []
        for row in rows:
            values = np.asarray(row[f"{prefix}_wall_repetitions_s"], dtype=float)
            center = float(row[f"{prefix}_median_wall_s"])
            if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"Invalid positive timing repetitions: {row['n']} {prefix}")
            if not np.isclose(center, statistics.median(values), rtol=1e-12, atol=0):
                raise ValueError(f"Stored median differs from repetitions: {row['n']} {prefix}")
            median.append(center)
            lower.append(center - float(values.min()))
            upper.append(float(values.max()) - center)
            all_values.extend(values)
        positions = x + (column - 1.5) * width
        ax.bar(positions, median, width * 0.91, color=color, label=label,
               yerr=np.array([lower, upper]), capsize=3,
               error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#222222"})
        for position, center, high in zip(positions, median, upper):
            ax.annotate(f"{center:.3g}", (position, center + high),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(min(all_values) * 0.42, max(all_values) * 1.9)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", color="#d4d4d4", linewidth=0.8)
    ax.grid(axis="y", which="minor", color="#ececec", linewidth=0.5)
    labels = []
    for row in rows:
        prime = " (prime)" if row["n"] == 1_000_003 else ""
        labels.append(f"N = {row['n']:,}{prime}\n{row['actual_frequencies']} log-frequency labels")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylabel("Wall time per call (seconds; logarithmic scale)")
    ax.set_xlim(-0.55, len(rows) - 0.45)
    fig.suptitle("LPSD and FFTW API runtimes — 8 threads", fontsize=15, y=0.985)
    fig.text(0.5, 0.935, "Different estimators: segmented LPSD and whole-record FFTW power averaging",
             ha="center", fontsize=11, color="#333333")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=2, frameon=False, fontsize=9.5, handlelength=1.6, columnspacing=2.2)
    fig.text(0.10, 0.13,
             "Bars and numbers: medians. Whiskers: observed minimum–maximum, not confidence intervals.\n"
             "Float64 input; PSD output float32; periodic Kaiser PSLL = 200 dB; FFTW ESTIMATE; no padding.",
             fontsize=9, va="top", linespacing=1.55)
    fig.text(0.10, 0.055,
             "Raw FFT excludes preprocessing. Prepared FFTW reuses plan/window/grid; fresh-plan FFTW includes their setup.\n"
             "All timed LPSD API calls prepare their own windows and coefficients. Shared-host CPU frequency was not controlled.",
             fontsize=8.5, va="top", color="#444444", linespacing=1.5)
    return fig


def spectra_figure(report):
    cases = {case["case"]: case for case in report["cases"]}
    frequencies = np.asarray(report["grid"]["frequency_labels_hz"], dtype=float)
    if frequencies.ndim != 1 or np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0):
        raise ValueError("Spectral labels must be finite, positive and strictly increasing")
    parameters = report["parameters"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.75, wspace=0.24)
    omitted = []
    styles = (("lpsd_psd", "LPSD fast.2", "#0072B2", "o"),
              ("fftw_log_psd", "FFTW log-periodogram", "#D55E00", "x"))
    for ax, name in zip(axes, ("white", "offbin_tone")):
        case = cases[name]
        if not case["frequency_index_exact"]:
            raise ValueError(f"Frequency labels did not match for {name}")
        plotted = []
        for key, label, color, marker in styles:
            values = np.asarray(case["small_log_spectra"][key], dtype=float)
            if values.shape != frequencies.shape or np.any(~np.isfinite(values)):
                raise ValueError(f"Invalid saved spectrum: {name} {key}")
            positive = values > 0
            if not positive.any():
                raise ValueError(f"No positive PSD values to plot: {name} {key}")
            count = int(np.count_nonzero(~positive))
            if count:
                omitted.append(f"{name}/{key}: {count}")
            # Markers only: no resampling, interpolated curves or amplitude floor.
            ax.plot(frequencies[positive], values[positive], linestyle="none", marker=marker,
                    markersize=2.6 if marker == "o" else 3.0,
                    markeredgewidth=0 if marker == "o" else 0.7,
                    color=color, alpha=0.85, label=label)
            plotted.extend(values[positive])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(frequencies[0] / 1.1, parameters["sample_rate"] / 2)
        ax.set_ylim(min(plotted) / 1.7, max(plotted) * 2.0)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", color="#dedede", linewidth=0.7)
        ax.set_xlabel("Frequency (Hz)")
        if name == "white":
            ax.set_title("White noise", pad=10)
        else:
            tone = case["signal"]["frequency_hz"]
            ax.set_title(f"Off-bin tone: $f_0$ = {tone:.6g} Hz", pad=10)
            ax.axvline(tone, color="#777777", linestyle=":", linewidth=0.8, zorder=0)
    axes[0].set_ylabel("PSD (signal unit² / Hz)")
    axes[1].set_ylabel("PSD (signal unit² / Hz)")
    fig.suptitle("Same frequency labels, different spectral estimates", fontsize=15, y=0.985)
    fig.text(0.5, 0.927,
             f"N = {parameters['n']:,}; $f_s$ = {parameters['sample_rate']:g} Hz; "
             f"{len(frequencies)} actual labels; targets J = {parameters['n_frequencies']}, K = {parameters['n_averages']}",
             ha="center", fontsize=10.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.885),
               ncol=2, frameon=False, fontsize=10, markerscale=1.7)
    fig.text(0.085, 0.12,
             "Markers show the saved float32 PSD values; no interpolation, smoothing or amplitude floor is added.\n"
             "LPSD averages overlapping, frequency-dependent segments; FFTW averages powers from one full-record transform.",
             fontsize=9, va="top", linespacing=1.55)
    detail = "Nonpositive values omitted from log axes: " + "; ".join(omitted) if omitted else "All saved values in these two spectra are positive."
    fig.text(0.085, 0.045,
             detail + " Differences include leakage, resolution and finite-sample variability.",
             fontsize=8.5, va="top", color="#444444")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results_fftw.json")
    parser.add_argument("--estimators", type=Path, help="Optional saved estimator comparison for the two-panel PSD figure")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args(argv)
    configure()
    performance, performance_sha = load(args.results)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    sources = {args.results.name: performance_sha}
    paths = save(performance_figure(performance), args.output_directory, "fftw-api-comparison",
                 "LPSD and FFTW API runtimes, eight threads", sources)
    if args.estimators:
        estimates, estimates_sha = load(args.estimators)
        paths += save(spectra_figure(estimates), args.output_directory, "fftw-estimator-comparison",
                      "LPSD and FFTW spectra on common frequency labels", {args.estimators.name: estimates_sha})
        sources[args.estimators.name] = estimates_sha
    print(json.dumps({"figures": paths, "source_sha256": sources, "matplotlib": matplotlib.__version__}, indent=2))


if __name__ == "__main__":
    main()
