#!/usr/bin/env python3
"""Render measured FFTW optimizations without running spectral benchmarks.

Usage:
    python benchmarks/render_fftw_optimized.py --results measurements.json \
        --output-dir figures

The accompanying measurements.spectra.npz is inferred from the JSON filename;
--spectra may override that location. Missing measurement categories are
omitted, never reconstructed from assumed execution times or fitted spectra.
All HTML figures are embedded SVGs; the report has no network dependencies.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from benchmarks.fftw_comparison_signals import TITLES, theory
except ImportError:
    # Spectra and the report remain renderable without SciPy. The optional
    # source-model curve is then omitted rather than guessed from the data.
    TITLES = {}
    theory = None

INK = "#223043"
MUTED = "#65758A"
GRID = "#DEE5EE"
PURPLE = "#4E365F"
BLUE = "#1768AC"
ORANGE = "#D45E00"
GREY = "#8995A4"
MODEL = "#64816A"
METHODS = ("lpsd", "fftw_after", "matched_after")
LABEL = {"lpsd": "LPSD / LNSD", "fftw_after": "FFTW einfach", "matched_after": "FFTW-Hybrid",
         "matched_once": "FFTW-Hybrid · einmalig"}
COLOR = {"lpsd": PURPLE, "fftw_after": GREY, "matched_after": BLUE}
CASE_ORDER = ("white", "pink", "brown", "mixed_tones", "offbin_tone", "dc_nanovolt")
CASE_COLORS = dict(zip(CASE_ORDER, ("#1768AC", "#B04C92", "#8A5D3B", "#198B77", "#D45E00", "#6D66BB")))
SHORT_CASE = {"white": "Weiß", "pink": "Rosa", "brown": "Braun", "mixed_tones": "Mischsignal",
              "offbin_tone": "Isolierter Ton", "dc_nanovolt": "10 V + 1 nV"}
DENSITY = {"psd": "PSD", "nsd": "NSD", "both": "PSD + NSD"}
UNIT = {"psd": "nV²/Hz", "nsd": "nV/√Hz"}
SCALE = {"psd": 1e18, "nsd": 1e9}


def number(value, digits=4):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)
    if not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    return f"{value:.{digits}g}".replace(".", ",")


def integer(value):
    return f"{int(value):,}".replace(",", " ")


def ms(value):
    return number(float(value) * 1e3) if value is not None else "—"


def h(value):
    return escape(str(value), quote=True)


def median(row):
    return None if row is None else row.get("median_wall_s")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.0,
        "axes.titlesize": 11.0, "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#95A4B6", "axes.spines.top": False,
        "axes.spines.right": False, "axes.facecolor": "white",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none", "axes.formatter.use_mathtext": True,
        "lines.solid_capstyle": "round",
    })


@dataclass
class Dataset:
    path: Path
    raw: dict
    spectra: dict

    @property
    def settings(self):
        return self.raw.get("settings", {})

    @property
    def cases(self):
        found = {row["case"]: row for row in self.raw.get("cases", [])}
        return [found[key] for key in CASE_ORDER if key in found] + [
            row for key, row in found.items() if key not in CASE_ORDER]

    @property
    def main_n(self):
        return int(self.settings.get("main_n", self.cases[0]["n"] if self.cases else 0))

    def array(self, case, suffix):
        return self.spectra.get(f"{case}_{suffix}")

    def preparation(self, method="matched_after", n=None):
        n = self.main_n if n is None else n
        rows = [row for row in self.raw.get("preparation", [])
                if row.get("method") == method and row.get("n") == n]
        return rows[-1].get("metadata", {}) if rows else {}

    @property
    def low_cutoff(self):
        return self.preparation().get("exact_low_frequency_max_hz")

    @property
    def low_count(self):
        return self.preparation().get("exact_low_frequency_points")

    def reference_timing(self, n, output):
        found = [row for row in self.raw.get("timings", []) if
                 row.get("method") == "lpsd" and row.get("n") == n and
                 row.get("outputs") == output and row.get("mode") == "complete"]
        return found[-1] if found else None

    def time_pairs(self, family, mode, output, *, after_method=None):
        """Match recorded pairs, including optional screening method aliases.

        Screen files contain repeated baseline rows. Match an optimized row
        to the immediately preceding baseline of the same N/mode/output;
        never average baselines from different paired experiments.
        """
        rows = self.raw.get("timings", [])
        before = "fftw_before" if family == "basic" else "matched_before"
        after = after_method or ("fftw_after" if family == "basic" else "matched_after")
        matching = [row for row in rows if row.get("method") == after and
                    row.get("mode") == mode and row.get("outputs") == output]
        if not matching and family == "basic" and after_method is None:
            preferred = {"native_grouped": "fftw_grouped", "numpy": "fftw_numpy"}.get(
                self.settings.get("operations"), "fftw_native")
            after = preferred
        latest_before, result = {}, {}
        for row in rows:
            if row.get("mode") != mode or row.get("outputs") != output:
                continue
            n, method = row.get("n"), row.get("method")
            if method == before:
                latest_before[n] = row
            elif method == after and n in latest_before:
                result[n] = (latest_before[n], row)
        return dict(sorted(result.items()))

    def once_pairs(self, mode, output):
        if mode != "fresh":
            return {}
        return self.time_pairs("matched", mode, output, after_method="matched_once")


@dataclass
class CacheStudy:
    path: Path
    raw: dict
    groups: list

    @property
    def n(self):
        return int(self.raw["settings"]["n"])


def load_cache_studies(paths):
    """Summarize separate completed tuning blocks, never mutate main results."""
    studies = []
    for path in paths:
        path = path.resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("settings", {}).get("stage") != "totalcache":
            raise ValueError(f"Cache-Studie benötigt stage=totalcache: {path}")
        if raw.get("input_unchanged") is not True:
            raise ValueError(f"Cache-Studie ist noch nicht mit Schlussprüfung abgeschlossen: {path}")
        grouped, definitions = {}, {}
        for entry in raw.get("configurations", []):
            parameters = entry.get("parameters", {})
            budget = parameters.get("total_cache_mb")
            if entry.get("family") != "matched" or budget is None or not math.isfinite(float(budget)):
                raise ValueError(f"Ungültige Hybrid-Cachekonfiguration: {path}")
            key = entry["id"]
            definition = (parameters, entry.get("fftw_threads"), entry.get("transform"))
            if key in definitions and definition != definitions[key]:
                raise ValueError(f"Konfiguration ändert sich zwischen Blöcken: {path}/{key}")
            definitions[key] = definition
            grouped.setdefault(key, []).append(entry)
        if not grouped:
            raise ValueError(f"Keine abgeschlossenen Cachekonfigurationen: {path}")
        groups = []
        for key, entries in grouped.items():
            entries = sorted(entries, key=lambda row: row["block"])
            if len({entry["block"] for entry in entries}) != len(entries):
                raise ValueError(f"Doppelter Tuningblock: {path}/{key}")
            all_times, blocks = [], []
            for entry in entries:
                values = [float(row["wall_s"]) for row in entry.get("timing", {}).get("repetitions", [])]
                if not values or any(not math.isfinite(value) or value <= 0 for value in values):
                    raise ValueError(f"Fehlende oder ungültige Rohzeiten: {path}/{key}")
                all_times.extend(values)
                blocks.append({"block": int(entry["block"]), "values": values,
                               "median_wall_s": float(np.median(values)),
                               "min_wall_s": min(values), "max_wall_s": max(values)})
            groups.append({"id": key, "budget_mib": float(entries[0]["parameters"]["total_cache_mb"]),
                           "fftw_threads": entries[0].get("fftw_threads"), "entries": entries,
                           "blocks": blocks, "repetitions": len(all_times), "values": all_times,
                           "median_wall_s": float(np.median(all_times)),
                           "min_wall_s": min(all_times), "max_wall_s": max(all_times)})
        studies.append(CacheStudy(path, raw, sorted(groups, key=lambda group: group["budget_mib"])))
    return sorted(studies, key=lambda study: study.n)


def load_dataset(path, spectra_path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Nicht unterstützte schema_version; erwartet wird 1.")
    arrays = {}
    if spectra_path.is_file():
        with np.load(spectra_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    elif raw.get("cases"):
        raise FileNotFoundError(f"Gemessene Spektren fehlen: {spectra_path}")
    data = Dataset(path, raw, arrays)
    for case in data.cases:
        key = case["case"]
        f = data.array(key, "f")
        if f is None or f.ndim != 1 or not np.isfinite(f).all() or np.any(f <= 0) or np.any(np.diff(f) <= 0):
            raise ValueError(f"Ungültiges oder fehlendes Frequenzraster: {key}")
        for method in ("lpsd", "fftw_before", "fftw_after", "matched_before", "matched_after"):
            for density in ("psd", "nsd"):
                a = data.array(key, f"{method}_{density}")
                if a is None or a.shape != f.shape or not np.isfinite(a).all() or np.any(a < 0):
                    raise ValueError(f"Ungültige oder fehlende Messwerte: {key}/{method}/{density}")
    return data


def decorate(ax, *, logarithmic_x=False, logarithmic_y=False):
    if logarithmic_x:
        ax.set_xscale("log")
    if logarithmic_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.grid(which="major", color=GRID, lw=.65)
    ax.grid(which="minor", color=GRID, lw=.4, alpha=.35)
    ax.set_axisbelow(True)


def lines_for_methods(dashed_hybrid=False):
    return [Line2D([], [], color=COLOR[method], lw=1.8,
                   ls=(0, (4, 2)) if dashed_hybrid and method == "matched_after" else "-",
                   label=LABEL[method]) for method in METHODS]


def header(fig, title, subtitle, handles=None, legend_columns=4):
    fig.text(.065, .968, title, fontsize=20, fontweight="bold", va="top")
    fig.text(.065, .923, subtitle, fontsize=10.5, color=MUTED, va="top")
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.062, .892),
                   ncol=legend_columns, frameon=False, fontsize=10, handlelength=2.7,
                   columnspacing=2.3, borderaxespad=0)


def save_figure(fig, output_dir, stem, title):
    files = {}
    for extension in ("png", "svg"):
        target = output_dir / f"{stem}.{extension}"
        fig.savefig(target, dpi=185, bbox_inches="tight", pad_inches=.16,
                    metadata={"Creator": "render_fftw_optimized.py"})
        files[extension] = target.name
    plt.close(fig)
    return {"title": title, **files}


def shade_native(ax, data, f):
    cut = data.low_cutoff
    if cut is not None and f.size and f[0] < cut:
        ax.axvspan(float(f[0]), min(float(cut), float(f[-1])), color="#E8EBEF", alpha=.65, zorder=0)


def native_note(data):
    if data.low_cutoff is None or data.low_count is None:
        return "FFTW-Hybrid: Frequenzmittelung mit dem dokumentierten nativen LPSD-Teilplan."
    return (f"Grauer Frequenzbereich: {integer(data.low_count)} unverändert native LPSD-Punkte "
            f"bis {number(data.low_cutoff, 6)} Hz im Hybrid; ihre Rechenzeit ist enthalten.")


def overview(data, output_dir, density):
    cases = data.cases
    rows = math.ceil(len(cases) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14.6, 4.0 * rows + 2.3), squeeze=False)
    fig.subplots_adjust(left=.075, right=.985, bottom=.105, top=.835, hspace=.40, wspace=.27)
    f0 = data.array(cases[0]["case"], "f")
    model_available = theory is not None
    handles = lines_for_methods()
    if model_available:
        handles.append(Line2D([], [], color=MODEL, lw=1.25, ls=":", label="Eingangs-Rauschmodell"))
    title = f"{density.upper()} nach Optimierung · {len(cases)} Eingangssignale"
    header(fig, title,
           f"N = {integer(data.main_n)} · fs = {number(data.settings.get('sample_rate', 50))} Hz · "
           f"{integer(len(f0))} gemeinsame Ausgabepunkte · gemessene Spektren ohne zusätzliche Glättung",
           handles)
    for index, ax in enumerate(axes.flat):
        if index >= len(cases):
            ax.set_axis_off()
            continue
        row = cases[index]
        key = row["case"]
        f = data.array(key, "f")
        ax.set_title(f"({chr(97 + index)})  {TITLES.get(key, key)}", loc="left", fontweight="bold", pad=11)
        shade_native(ax, data, f)
        if model_available:
            expected = theory(key, f, data.settings.get("sample_rate", 50))
            if expected is not None:
                if density == "nsd":
                    expected = np.sqrt(expected)
                ax.plot(f, expected * SCALE[density], color=MODEL, lw=1.0, ls=":", zorder=5)
        for method in METHODS:
            values = data.array(key, f"{method}_{density}").astype(np.float64) * SCALE[density]
            ax.plot(f, np.where(values > 0, values, np.nan), color=COLOR[method],
                    lw=1.0 if method == "fftw_after" else 1.35,
                    alpha=.78 if method == "fftw_after" else .97,
                    zorder=2 if method == "fftw_after" else (4 if method == "matched_after" else 3))
        decorate(ax, logarithmic_x=True, logarithmic_y=True)
        ax.set_xlim(f[0], f[-1])
        ax.set_xlabel("Frequenz [Hz]")
        ax.set_ylabel(f"{density.upper()} [{UNIT[density]}]")
    fig.text(.075, .061, native_note(data), fontsize=9.1, color=MUTED)
    fig.text(.075, .041, "Modellkurve: Rauschanteil des Eingangs; Fenster und Mittelwertabzug verändern den endlichen Schätzer.", fontsize=9.1, color=MUTED)
    fig.text(.075, .021, "Laufzeiten sind separat mit weißem Eingang gemessen. Für die übrigen fünf Signale werden keine individuellen Zeiten behauptet.", fontsize=9.1, color=MUTED)
    return save_figure(fig, output_dir, f"{density}_fftw_optimized", title)


def size_label(n):
    return integer(n) + ("\n(Primzahl)" if n == 1_000_003 else "")


def fft_path_note(data, *, short=False):
    """Describe only FFT choices explicitly present in the measured settings."""
    selections = data.settings.get("bluestein") or []
    if not selections:
        return ("FFTW R2C: ursprüngliche N-Punkt-DFT; die Primzahlmessung gilt für diesen Adapterpfad."
                if short else "FFTW verwendet den ursprünglichen N-Punkt-R2C-Pfad. "
                "Andere Transformationspfade werden dadurch nicht allgemein bewertet.")
    variants = []
    for selection in selections:
        n, m = map(int, str(selection).split(":"))
        variants.append(f"N = {integer(n)}, M = {integer(m)}")
    choices = "; ".join(variants)
    if short:
        return f"Nachher Bluestein bei {choices}; sonst R2C. Unveränderte N-Punkt-DFT."
    return (f"Die optimierten Pfade verwenden ausdrücklich Bluestein bei {choices}; "
            "die übrigen Längen und die ursprünglichen Vergleichsverfahren verwenden direktes R2C. "
            "Bluestein berechnet dieselbe ursprüngliche N-Punkt-DFT; "
            "M bezeichnet allein die interne komplexe Faltungslänge und verändert weder Raster noch Normalisierung.")


def planner_note(data, *, short=False):
    lengths = data.settings.get("measure_lengths") or []
    if not lengths:
        return "Planer: ESTIMATE für vorbereitete Objekte und sämtliche vollständigen Neuaufrufe."
    selection = ", ".join(integer(n) for n in lengths)
    limit = number(data.settings.get("planner_time_limit"))
    if short:
        return (f"Nur Reuse nachher bei N = {selection}: MEASURE ({limit} s je Plan); "
                "alle Neuaufrufe und Baselines: ESTIMATE.")
    return (f"Nur wiederverwendete optimierte Objekte bei N = {selection} verwenden MEASURE mit "
            f"dem FFTW-Zeitlimit {limit} s je Plan. Die dabei beobachtete Vorbereitung wird separat "
            "ausgewiesen und liegt außerhalb der Reuse-Zeiten. Sämtliche vollständigen Neuaufrufe, "
            "einschließlich Kaltstart der Klasse und estimate_once, verwenden ESTIMATE; ebenso alle "
            "bisherigen Vergleichsverfahren und die übrigen wiederverwendeten Längen. Das FFTW-Limit "
            "gilt pro Plan, nicht für den gesamten Objektaufbau; Bluestein benötigt zwei Faltungspläne.")


def thread_note(data):
    selections = data.settings.get("fftw_threads") or []
    if not selections:
        return f"FFTW-Threads vorher/nachher: {data.settings.get('workers', '—')}."
    return ("FFTW-Threads nachher N:T = " + " · ".join(map(str, selections))
            + f"; sonst {data.settings.get('workers', '—')}; vorher immer {data.settings.get('workers', '—')}.")


def runtime_plot(data, output_dir, density):
    available_modes = [mode for mode in ("reuse", "fresh") if any(
        data.time_pairs(family, mode, density) for family in ("basic", "matched"))
        or data.once_pairs(mode, density)]
    if not available_modes:
        return None
    ncols = len(available_modes)
    fig, axes = plt.subplots(2, ncols, figsize=(15.6 if ncols == 2 else 10.6, 12.3), squeeze=False)
    fig.subplots_adjust(left=.075, right=.985, bottom=.29, top=.82, hspace=.61, wspace=.25)
    title = f"Komplette Berechnungszeiten · {DENSITY[density]}"
    handles = [Patch(facecolor=GREY, label="Vorher"), Patch(facecolor=BLUE, label="Optimierte Klasse")]
    if data.once_pairs("fresh", density):
        handles.append(Patch(facecolor=ORANGE, label="Optimiert · Einzelaufruf"))
    if any(row.get("method") == "lpsd" and row.get("outputs") == density for row in data.raw.get("timings", [])):
        handles.append(Line2D([], [], color=PURPLE, marker="D", ls="--", label="LPSD / LNSD komplett"))
    header(fig, title, "Vorher / nachher · weißer Eingang · Median mit beobachteter Minimum–Maximum-Spanne",
           handles, 4)
    for i, family in enumerate(("basic", "matched")):
        for j, mode in enumerate(available_modes):
            ax = axes[i, j]
            label = "Einfaches FFTW" if family == "basic" else "FFTW-Hybrid"
            state = "vorbereitet" if mode == "reuse" else "komplett neu"
            ax.set_title(f"({chr(97 + i * ncols + j)})  {label} · {state}", loc="left", fontweight="bold", pad=12)
            pairs = data.time_pairs(family, mode, density)
            once = data.once_pairs(mode, density) if family == "matched" else {}
            if not pairs and not once:
                ax.text(.5, .5, "Dieser Betriebsfall wurde nicht gemessen.", transform=ax.transAxes, ha="center", color=MUTED)
                ax.set_axis_off()
                continue
            sizes = sorted(set(pairs) | set(once))
            x = np.arange(len(sizes), dtype=float)
            all_positive = []
            baselines = {n: (pairs[n] if n in pairs else once[n])[0] for n in sizes}
            series = [(GREY, baselines), (BLUE, {n: pair[1] for n, pair in pairs.items()})]
            if once:
                series.append((ORANGE, {n: pair[1] for n, pair in once.items()}))
            offsets = (-.26, 0, .26) if once else (-.17, .17)
            for k, (color, series_rows) in enumerate(series):
                available = [i for i, n in enumerate(sizes) if n in series_rows]
                records = [series_rows[sizes[i]] for i in available]
                if not records:
                    continue
                center = np.asarray([row["median_wall_s"] * 1000 for row in records])
                lower = np.asarray([row["min_wall_s"] * 1000 for row in records])
                upper = np.asarray([row["max_wall_s"] * 1000 for row in records])
                positions = x[available] + offsets[k]
                ax.bar(positions, center, width=.225 if once else .285, color=color, zorder=3)
                ax.errorbar(positions, center, yerr=[center - lower, upper - center],
                            fmt="none", ecolor=INK, elinewidth=.7, capsize=2.2, zorder=5)
                for pos, high, val in zip(positions, upper, center):
                    ax.text(pos, high * 1.10, number(val, 3), ha="center", va="bottom",
                            fontsize=7.5, color=color, rotation=55 if once else 0)
                all_positive.extend(lower.tolist() + upper.tolist())
            ref = [data.reference_timing(n, density) for n in sizes]
            rx = np.asarray([k for k, row in enumerate(ref) if row is not None])
            ry = np.asarray([row["median_wall_s"] * 1000 for row in ref if row is not None])
            if len(rx):
                ax.plot(rx, ry, color=PURPLE, marker="D", ms=3.3, ls="--", lw=1.15, zorder=6)
                all_positive.extend(ry.tolist())
            decorate(ax, logarithmic_y=True)
            if all_positive:
                ax.set_ylim(min(v for v in all_positive if v > 0) * .55, max(all_positive) * (2.6 if once else 1.85))
            ax.set_xticks(x, [size_label(n) for n in sizes])
            ax.set_ylabel("Zeit [ms] · logarithmisch")
            ax.set_xlabel("Anzahl Eingangswerte N")
            ratios = " · ".join(f"{number(pairs[n][0]['median_wall_s'] / pairs[n][1]['median_wall_s'], 3)}×"
                                if n in pairs else "—" for n in sizes)
            ratio_label = "Faktor vorher / Kaltstart" if once else "Faktor vorher / nachher"
            ax.text(0, -.33, f"{ratio_label} (von links): {ratios}", transform=ax.transAxes, fontsize=8.25, color=MUTED)
            if once:
                ratios = " · ".join(f"{number(once[n][0]['median_wall_s'] / once[n][1]['median_wall_s'], 3)}×"
                                    if n in once else "—" for n in sizes)
                ax.text(0, -.42, f"Faktor vorher / Einzelaufruf: {ratios}", transform=ax.transAxes, fontsize=8.25, color=ORANGE)
    fig.text(.075, .160, "Vorbereitet: kompletter Datenlauf mit vorhandenen Plänen und Caches. Komplett neu: einschließlich Planung, Vorbereitung und Freigabe.", fontsize=9, color=MUTED)
    if data.once_pairs("fresh", density):
        fig.text(.075, .139, "Blau im Neuaufruf: Kaltstart der vorbereitbaren Klasse. Orange: estimate_once ohne dauerhafte Caches, mit aufgeschobener Normierung.", fontsize=9, color=MUTED)
    fig.text(.075, .118, planner_note(data, short=True), fontsize=8.7, color=MUTED)
    fig.text(.075, .097, fft_path_note(data, short=True), fontsize=8.7, color=MUTED)
    fig.text(.075, .076, thread_note(data), fontsize=8.7, color=MUTED)
    fig.text(.075, .055, "PSD und NSD teilen fast die gesamte Rechnung. Mediane können streuen; die Minimum–Maximum-Spannen sind keine Konfidenzintervalle.", fontsize=9, color=MUTED)
    if density == "both":
        fig.text(.075, .034, "PSD + NSD: einfaches FFTW zuvor zwei komplette Aufrufe, danach ein gemeinsamer; der Hybrid unterstützte kombinierte Ausgaben bereits zuvor.", fontsize=8.7, color=MUTED)
    return save_figure(fig, output_dir, f"runtime_before_after_{density}", title)


def cache_study_plot(studies, output_dir):
    if not studies:
        return None
    ncols = min(len(studies), 2)
    nrows = math.ceil(len(studies) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.8 if ncols == 2 else 9.3, 4.0 * nrows + 3.4), squeeze=False)
    fig.subplots_adjust(left=.085, right=.98, top=.765, bottom=.29, hspace=.48, wspace=.24)
    title = "NSD und Cachebudget · zusätzliche serielle Studie"
    handles = [Line2D([], [], color=INK, marker="_", markersize=13, lw=0, label="Median aller Rohzeiten"),
               Line2D([], [], color=INK, marker="D", mfc="white", markersize=5, lw=0, label="Blockmediane"),
               Line2D([], [], color=MUTED, marker=".", markersize=6, lw=0, label="Einzelne Aufrufe")]
    header(fig, title, "Vorbereiteter FFTW-Hybrid · ein Objekt zur Zeit · Min/Max und einzelne Tuningblöcke sichtbar", handles, 3)
    budget_values = sorted({group["budget_mib"] for study in studies for group in study.groups})
    palette = [BLUE, "#148773", ORANGE, PURPLE]
    colors = {budget: palette[i % len(palette)] for i, budget in enumerate(budget_values)}
    for index, ax in enumerate(axes.flat):
        if index >= len(studies):
            ax.set_axis_off()
            continue
        study = studies[index]
        groups = study.groups
        threads = sorted({group["fftw_threads"] for group in groups if group["fftw_threads"] is not None})
        ax.set_title(f"({chr(97 + index)})  N = {integer(study.n)} · FFTW-Threads: {', '.join(map(str, threads))}",
                     loc="left", fontweight="bold", pad=12)
        x = np.arange(len(groups), dtype=float)
        upper_limit = max(group["max_wall_s"] for group in groups) * 1000
        for position, group in zip(x, groups):
            center = group["median_wall_s"] * 1000
            low, high = group["min_wall_s"] * 1000, group["max_wall_s"] * 1000
            ax.bar(position, center, width=.56, color=colors[group["budget_mib"]], alpha=.55, zorder=2)
            ax.errorbar(position, center, yerr=[[center - low], [high - center]], fmt="none",
                        ecolor=INK, elinewidth=1.0, capsize=5, zorder=4)
            ax.plot([position - .22, position + .22], [center, center], color=INK, lw=1.6, zorder=6)
            offsets = np.linspace(-.16, .16, len(group["blocks"])) if len(group["blocks"]) > 1 else [0.0]
            for offset, block in zip(offsets, group["blocks"]):
                raw_x = position + offset + np.linspace(-.035, .035, len(block["values"]))
                ax.scatter(raw_x, np.asarray(block["values"]) * 1000, s=12, color=MUTED, alpha=.75, zorder=5)
                ax.plot(position + offset, block["median_wall_s"] * 1000, marker="D", ms=5.5,
                        mfc="white", mec=INK, mew=1.0, ls="none", zorder=7)
            ax.text(position, high + upper_limit * .07, f"{number(center, 5)} ms", ha="center", va="bottom",
                    color=colors[group["budget_mib"]], fontsize=11, fontweight="bold")
        ax.set_xticks(x, [f"{number(group['budget_mib'] / 1024)} GiB" for group in groups])
        ax.set_xlabel("Gemeinsame Cacheobergrenze")
        ax.set_ylabel("Vollständiger NSD-Aufruf [ms]")
        ax.set_ylim(0, upper_limit * 1.32)
        decorate(ax)
        counts = " · ".join(f"{number(group['budget_mib'] / 1024)} GiB: {len(group['blocks'])} Blöcke, {group['repetitions']} Aufrufe"
                            for group in groups)
        ax.text(0, -.25, counts, transform=ax.transAxes, color=MUTED, fontsize=8.8)
    fig.text(.085, .109, "Gesamtmedian: direkt aus allen aufgezeichneten Einzelzeiten, nicht aus den Blockmedianen. Spannweiten sind keine Konfidenzintervalle.", fontsize=9, color=MUTED)
    fig.text(.085, .080, "Dies sind serielle Tuningblöcke; sie sind keine gepaarte Messung gegen die ursprüngliche Implementierung. Hostschwankungen bleiben sichtbar.", fontsize=9, color=MUTED)
    fig.text(.085, .051, "Das Cachelimit umfasst gespeicherte Gewichte und projizierte LPSD-Koeffizienten. Tatsächliche Belegung und q-Punktzahlen stehen im Bericht.", fontsize=9, color=MUTED)
    return save_figure(fig, output_dir, "cache_budget_nsd", title)


def peak_figure(data, output_dir):
    case = next((row for row in data.cases if row["case"] == "offbin_tone"), None)
    if case is None:
        return None
    f = data.array("offbin_tone", "f")
    center = case["tones"][0]["frequency_hz"]
    widths = [case["metrics"][method]["peak"].get("nsd_fwhm_hz") for method in METHODS]
    step = np.median(np.diff(f[max(0, np.searchsorted(f, center) - 2):np.searchsorted(f, center) + 3]))
    half_range = max(1.25 * max((width for width in widths if width), default=.1), 4 * step)
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.6))
    fig.subplots_adjust(left=.077, right=.98, top=.82, bottom=.14, hspace=.41, wspace=.25)
    title = "Isolierter Sinus · Peak-Breite und ferner Leckboden"
    header(fig, title, f"f₀ = {number(center, 10)} Hz · {number(case['tones'][0]['rms_v'] * 1e6)} µV RMS · N = {integer(case['n'])} · Grau: native LPSD im Hybrid", lines_for_methods(True), 3)
    for i, density in enumerate(("psd", "nsd")):
        near, far = axes[i]
        near.set_title(f"({chr(97 + 2 * i)})  {density.upper()} · native Punkte, auf eigenen Peak normiert", loc="left", fontweight="bold", pad=12)
        far.set_title(f"({chr(98 + 2 * i)})  {density.upper()} · |f − f₀| > 1 Hz", loc="left", fontweight="bold", pad=12)
        shade_native(far, data, f)
        for k, method in enumerate(METHODS):
            values = data.array("offbin_tone", f"{method}_{density}").astype(float)
            metrics = case["metrics"][method]["peak"]
            selected = np.abs(f - center) <= half_range * 1.3
            maximum = float(np.max(values[np.abs(f - center) <= .6]))
            line_style = (0, (4, 2)) if method == "matched_after" else "-"
            near.plot(f[selected] - center, values[selected] / maximum, color=COLOR[method],
                      lw=1.4, ls=line_style, marker="o", ms=3.2, mec="white", mew=.4, zorder=3 + k)
            crossing = metrics.get(f"{density}_half_max_crossings")
            if crossing:
                near.plot(np.asarray([crossing["left_hz"], crossing["right_hz"]]) - center,
                          [.5, .5], color=COLOR[method], marker="|", ms=8, lw=.85, zorder=6)
            near.text(.025, .955 - .075 * k, f"{LABEL[method]}: FWHM ≈ {number(metrics.get(density + '_fwhm_hz'), 5)} Hz",
                      transform=near.transAxes, va="top", fontsize=8.3, color=COLOR[method])
            allowed = (f > .01) & (f < 20) & (np.abs(f - center) > 1)
            visible = np.where(allowed & (values > 0), values * SCALE[density], np.nan)
            far.plot(f, visible, color=COLOR[method], lw=1.3, ls=line_style, zorder=3 + k)
        near.axvline(0, color="#8190A2", ls=":", lw=.75)
        near.axhline(.5, color="#8190A2", ls=":", lw=.75)
        near.set_xlim(-half_range, half_range)
        near.set_ylim(-.015, 1.075)
        near.set_xlabel("Abstand zur Eingangsfrequenz f₀ [Hz]")
        near.set_ylabel(f"{density.upper()} / eigener Peak")
        decorate(near)
        far.set_xlim(.01, 20)
        far.set_xlabel("Frequenz [Hz]")
        far.set_ylabel(f"{density.upper()} [{UNIT[density]}]")
        decorate(far, logarithmic_x=True, logarithmic_y=True)
    fig.text(.077, .065, "FWHM: halbe PSD-Höhe beziehungsweise halbe NSD-Höhe (= Viertel der PSD-Höhe). Kreuzungen sind zwischen nativen Rasterpunkten interpoliert.", fontsize=9, color=MUTED)
    fig.text(.077, .044, "Verbindungslinien erzeugen keine zusätzlichen Spektralwerte. Die grobe Frequenzstützung begrenzt die Genauigkeit der Breitenangaben.", fontsize=9, color=MUTED)
    fig.text(.077, .023, "Rechts: absolute Leckwerte, ohne Normierung oder künstlichen Rauschboden. Ähnliche Peak-Breite gewährleistet keine gleiche Nebenkeulenunterdrückung.", fontsize=9, color=MUTED)
    return save_figure(fig, output_dir, "peak_leakage_fftw_optimized", title)


def difference_figure(data, output_dir, density):
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.5))
    fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.125, hspace=.36, wspace=.27)
    handles = [Line2D([], [], color=CASE_COLORS.get(row["case"], BLUE), lw=1.6,
                      label=SHORT_CASE.get(row["case"], row["case"])) for row in data.cases]
    title = f"Implementierungsänderung · {density.upper()} nachher minus vorher"
    header(fig, title, f"N = {integer(data.main_n)} · Jede FFTW-Methode gegen ihren eigenen Vorgänger, auf identischen Eingängen.", handles, 6)
    for i, (after, before, name) in enumerate((("fftw_after", "fftw_before", "Einfaches FFTW"),
                                             ("matched_after", "matched_before", "FFTW-Hybrid"))):
        nonzero = [0, 0]
        totals, positive_references, max_relative = 0, 0, 0.0
        for row in data.cases:
            key = row["case"]
            f = data.array(key, "f")
            actual = data.array(key, f"{after}_{density}").astype(float)
            reference = data.array(key, f"{before}_{density}").astype(float)
            error = actual - reference
            totals += len(error)
            color = CASE_COLORS.get(key, BLUE)
            absolute = np.abs(error) * SCALE[density]
            positive = reference != 0
            positive_references += int(np.count_nonzero(positive))
            relative = np.divide(error * 100, np.abs(reference), out=np.full_like(error, np.nan), where=positive)
            if np.any(error != 0):
                axes[i, 0].plot(f, np.where(absolute > 0, absolute, np.nan), color=color, lw=.9, marker=".", ms=2.1)
                axes[i, 1].plot(f, relative, color=color, lw=.85, alpha=.87)
                nonzero[0] += int(np.count_nonzero(error))
                nonzero[1] += int(np.count_nonzero(positive & (error != 0)))
            if np.any(positive):
                max_relative = max(max_relative, float(np.nanmax(np.abs(relative))))
        for j, ax in enumerate(axes[i]):
            kind = "absoluter Betrag" if j == 0 else "relative, vorzeichenbehaftete Änderung"
            ax.set_title(f"({chr(97 + 2 * i + j)})  {name} · {kind}", loc="left", fontweight="bold", pad=12)
            decorate(ax, logarithmic_x=True, logarithmic_y=j == 0 and nonzero[j] > 0)
            if j == 1 and max_relative > 0:
                ax.set_yscale("symlog", linthresh=max(max_relative * 1e-3, 1e-12))
                ax.axhline(0, color="#8593A4", lw=.7)
            if not nonzero[j]:
                ax.set_ylim(-.5, .5)
                ax.set_yticks([0])
                ax.axhline(0, color=BLUE, lw=1.2)
                equality = (f"Alle {integer(totals)} gespeicherten Werte bitidentisch\nDifferenz = 0" if not nonzero[0]
                            else f"Relative Differenz = 0 für {integer(positive_references)} Werte\n"
                                 f"{integer(totals - positive_references)} vorherige Nullwerte: Quotient nicht definiert")
                ax.text(.5, .55, equality, transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color=BLUE,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 8})
            else:
                ax.text(.025, .97, f"{integer(nonzero[j])} / {integer(totals)} Werte mit Differenz", transform=ax.transAxes,
                        va="top", fontsize=8.6, color=MUTED, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86})
            first_f = data.array(data.cases[0]["case"], "f")
            ax.set_xlim(first_f[0], first_f[-1])
            ax.set_xlabel("Frequenz [Hz]")
            ax.set_ylabel(f"|Δ{density.upper()}| [{UNIT[density]}]" if j == 0 else f"100 · Δ{density.upper()} / |{density.upper()} vorher| [%]")
    fig.text(.08, .061, "Links bleiben auch kleine absolute Leckdifferenzen sichtbar. Exakte Nulldifferenzen werden auf logarithmischen Ordinaten nicht als positive Werte dargestellt.", fontsize=9, color=MUTED)
    fig.text(.08, .039, "Rechts sind Änderungen relativ zum jeweiligen vorherigen Wert gezeigt; nahe einem sehr kleinen Leckboden kann dieser Quotient groß werden.", fontsize=9, color=MUTED)
    fig.text(.08, .017, "Diese Zahlen beschreiben die Implementierung, nicht den Unterschied zwischen dem LPSD-Schätzer und den beiden FFTW-Schätzern.", fontsize=9, color=MUTED)
    return save_figure(fig, output_dir, f"implementation_difference_{density}", title)


def table(headers, rows, *, classes=None):
    head = "".join(f"<th>{h(value)}</th>" for value in headers)
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            cls = f' class="{classes[index]}"' if classes and classes[index] else ""
            cells.append(f"<td{cls}>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def time_cell(row):
    if row is None:
        return "—"
    return (f"<strong>{ms(row['median_wall_s'])}</strong>"
            f"<small>{ms(row['min_wall_s'])}–{ms(row['max_wall_s'])} · n={len(row.get('repetitions', []))}</small>")


def timings_html(data):
    sections = []
    for family in ("basic", "matched"):
        for mode in ("reuse", "fresh"):
            rows = []
            include_once = family == "matched" and mode == "fresh" and any(
                data.once_pairs(mode, output) for output in ("psd", "nsd", "both"))
            for output in ("psd", "nsd", "both"):
                pairs = data.time_pairs(family, mode, output)
                once = data.once_pairs(mode, output) if include_once else {}
                for n in sorted(set(pairs) | set(once)):
                    before = (pairs[n] if n in pairs else once[n])[0]
                    after = pairs[n][1] if n in pairs else None
                    ratio = h(number(before["median_wall_s"] / after["median_wall_s"], 5) + "×") if after else "—"
                    cells = [h(integer(n)), h(DENSITY[output]), time_cell(before), time_cell(after), ratio]
                    if include_once:
                        once_pair = once.get(n)
                        once_ratio = (h(number(once_pair[0]["median_wall_s"] / once_pair[1]["median_wall_s"], 5) + "×")
                                      if once_pair else "—")
                        cells.extend([time_cell(once_pair[1]) if once_pair else "—", once_ratio])
                    cells.append(time_cell(data.reference_timing(n, output)))
                    rows.append((n, output, cells))
            if not rows:
                continue
            rows.sort(key=lambda row: (row[0], ("psd", "nsd", "both").index(row[1])))
            name = "Einfaches FFTW" if family == "basic" else "FFTW-Hybrid"
            state = "vorbereitet" if mode == "reuse" else "komplett neu"
            columns = (["N", "Ausgabe", "Vorher [ms]", "Kaltstart Klasse [ms]", "Faktor Kaltstart",
                        "Einmalig optimiert [ms]", "Faktor einmalig", "LPSD komplett [ms]"] if include_once else
                       ["N", "Ausgabe", "Vorher [ms]", "Nachher [ms]", "Vorher / nachher", "LPSD komplett [ms]"])
            note = ("<p class='table-note'>Kaltstart Klasse: vollständiger Neuaufbau der vorbereitbaren "
                    "optimierten Klasse einschließlich ihrer Caches. Einmalig optimiert: vollständiges "
                    "<code>estimate_once</code> ohne dauerhafte Caches und mit aufgeschobener nativer "
                    "Normierung. Beide Faktoren verwenden die jeweils gepaarte ursprüngliche "
                    "Hybridmessung als Zähler.</p>" if include_once else "")
            sections.append(f"<h3>{name} · {state}</h3>" + note + table(
                columns, [row[2] for row in rows], classes=[None, None] + ["numeric"] * (len(columns) - 2)))
    return "".join(sections)


def preparation_html(data):
    rows = []
    thread_counts = {int(str(value).split(":")[0]): int(str(value).split(":")[1])
                     for value in data.settings.get("fftw_threads") or []}
    convolution_lengths = {int(str(value).split(":")[0]): int(str(value).split(":")[1])
                           for value in data.settings.get("bluestein") or []}
    measured_lengths = set(data.settings.get("measure_lengths") or [])
    for row in data.raw.get("preparation", []):
        observed = row.get("observed_setup")
        if not observed:
            continue
        n, method = row["n"], row["method"]
        md = row.get("metadata", {})
        fft = md.get("fft") or {}
        original = method.endswith("_before")
        label = ("FFTW einfach" if method.startswith("fftw_") else "FFTW-Hybrid")
        label += " · vorher" if original else " · optimiert"
        planner = md.get("planner", fft.get("planner")) or (
            "measure" if n in measured_lengths and not original else "estimate")
        threads = md.get("fftw_threads", fft.get("threads"))
        if threads is None:
            threads = data.settings.get("workers", "—") if original else thread_counts.get(
                n, data.settings.get("workers", "—"))
        convolution = md.get("fft_convolution_length", fft.get("convolution_length"))
        if convolution is None and not original:
            convolution = convolution_lengths.get(n)
        algorithm = f"Bluestein · M={integer(convolution)}" if convolution is not None else "R2C"
        plan_time = fft.get("planning_s", row.get("setup_phases", {}).get("fftw_planning_s"))
        rows.append([h(integer(n)), h(label), h(algorithm), h(str(planner).upper()), h(threads),
                     h(ms(observed.get("wall_s"))), h(ms(plan_time)), h(ms(observed.get("cpu_s")))])
    if not rows:
        return ""
    return ("<h3>Vorbereitung der wiederverwendeten Objekte</h3>"
            "<p>Jede Zeile ist die protokollierte Einzelbeobachtung eines vollständigen Konstruktoraufrufs "
            "mit dem Reuse-Profil. Diese Vorbereitung enthält Planer, Allokation, Fenster und konfigurierte "
            "Caches; sie gehört nicht zu den Reuse-Medianen. Die separat erfasste FFTW-Planungszeit ist "
            "eine Teilphase. Die CPU-Zeit kann bei mehreren aktiven Threads größer als die verstrichene "
            "Zeit sein. Vollständige Neuaufrufe verwenden dagegen überall ESTIMATE.</p>" + table(
                ["N", "Methode", "DFT-Algorithmus", "Reuse-Planer", "FFTW-Threads", "Aufbau gesamt [ms]",
                 "Davon FFTW-Planung [ms]", "CPU Aufbau [ms]"], rows,
                classes=[None, None, None, None, "numeric", "numeric", "numeric", "numeric"]))


def changes_html(data):
    sections = []
    for density in ("psd", "nsd"):
        rows = []
        for row in data.cases:
            for method in ("fftw_after", "matched_after", "matched_once"):
                values = row.get("implementation_difference", {}).get(method, {}).get(density)
                if not values:
                    continue
                rows.append([h(TITLES.get(row["case"], row["case"])), h(LABEL[method]),
                             "ja" if values.get("exact") else "nein",
                             h(f"{values.get('different_points', '—')} / {values.get('points', '—')}"),
                             h(number(values.get("max_abs", 0) * SCALE[density], 7)),
                             h(number(values.get("max_relative_where_reference_nonzero", 0) * 100, 7)),
                             h(number(values.get("p99_relative_where_reference_nonzero", 0) * 100, 7))])
        if rows:
            sections.append(f"<h3>{density.upper()}: maschinell erfasste Abweichungen</h3>" + table(
                ["Signal", "Methode", "Bitidentisch", "Geänderte Werte", f"Max. |Δ| [{UNIT[density]}]", "Max. relativ [%]", "P99 relativ [%]"],
                rows, classes=[None, None, None, "numeric", "numeric", "numeric", "numeric"]))
    return "".join(sections)


def load_prime_quality(path):
    """Read an independent completed quality run without mixing spectral grids."""
    if path is None:
        return None
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not raw.get("finished_utc") or not raw.get("cases"):
        raise ValueError(f"Zusätzliche Qualitätsprüfung ist nicht vollständig: {path}")
    data = Dataset(path, raw, {})
    if any(row.get("n") != data.main_n for row in data.cases):
        raise ValueError(f"Qualitätsprüfung enthält verschiedene Eingangslängen: {path}")
    return data


def prime_quality_html(data):
    if data is None:
        return ""
    rows = []
    for method in ("fftw_after", "matched_after", "matched_once"):
        for density in ("psd", "nsd"):
            changes = [row.get("implementation_difference", {}).get(method, {}).get(density)
                       for row in data.cases]
            changes = [row for row in changes if row]
            if not changes:
                continue
            rows.append([h(LABEL[method]), h(density.upper()),
                         h(f"{sum(bool(row.get('exact')) for row in changes)} / {len(changes)}"),
                         h(number(max(row["max_abs"] for row in changes) * SCALE[density], 9)),
                         h(number(max(row["max_relative_where_reference_nonzero"] for row in changes) * 100, 9)),
                         h(number(max(row["max_absolute_over_reference_peak"] for row in changes) * 100, 9))])
    if not rows:
        return ""
    raw_json = h(json.dumps(data.raw, ensure_ascii=False, indent=2, allow_nan=False))
    return (f"<h3>Zusätzliche Qualitätsprüfung bei N = {h(integer(data.main_n))}</h3>"
            f"<p>Diese eigenständige Prüfung verwendet <code>{h(data.path.name)}</code> und "
            f"{len(data.cases)} Signale. Ihre Frequenzpunkte werden nicht mit dem Hauptdatensatz vermischt. "
            f"{h(fft_path_note(data))} {h(planner_note(data))}</p>"
            "<p>Die Maxima jeder Tabellenspalte werden getrennt über die geprüften Signale bestimmt; "
            "sie müssen nicht zum gleichen Signal oder Frequenzpunkt gehören. Die letzte Spalte "
            "setzt die größte absolute Änderung jedes Signals ins Verhältnis zu dessen Referenzmaximum. "
            "Sie hilft, hohe relative Werte an extrem kleinen Leckböden einzuordnen.</p>" + table(
                ["Methode", "Ausgabe", "Bitidentische Signale", "Max. |Δ| [nV²/Hz bzw. nV/√Hz]",
                 "Max. punktweise relativ [%]", "Max. |Δ| / Referenzmaximum [%]"], rows,
                classes=[None, None, "numeric", "numeric", "numeric", "numeric"]) +
            "<p class='table-note'>Bluestein berechnet dieselbe mathematische N-Punkt-DFT mit anderer "
            "Rundung. Beim isolierten Ton können relative Abweichungen an beinahe verschwindenden "
            "Leckwerten deshalb größer ausfallen, obwohl der absolute Fehler extrem klein bleibt. "
            "Die Prüfung des Messharness verwendet gemeinsam <code>rtol=3e-6</code> und "
            "<code>atol=max(Referenzmaximum × 2e-14, 2 × float32.tiny)</code>. Das ist keine "
            "ausschließlich relative Fehlergarantie und kein Grenzwert für Unterschiede zwischen "
            "LPSD und FFTW.</p>" + changes_html(data) +
            "<details><summary>Vollständiges zusätzliches Qualitätsprotokoll</summary>"
            f"<pre>{raw_json}</pre></details>")


def quality_html(data):
    rows = []
    for row in data.cases:
        cells = []
        for method in METHODS:
            value = row.get("metrics", {}).get(method, {}).get("normalized_nsd_std_005_to_20_hz")
            cells.append(h(number(value * 100 if value is not None else None, 6)))
        if any(cell != "—" for cell in cells):
            rows.append([h(TITLES.get(row["case"], row["case"]))] + cells)
    result = ""
    if rows:
        purely_fftw = data.low_cutoff is not None and data.low_cutoff < .05
        band_title = "Rauschwelligkeit im FFTW-Bereich 0,05–20 Hz" if purely_fftw else "Rauschwelligkeit im Band 0,05–20 Hz"
        band_note = ("Dieses Band liegt oberhalb des nativen LPSD-Teilplans dieser Spektralmessung. " if purely_fftw else
                     "Dieses Band kann auch native LPSD-Punkte des Hybrids enthalten; maßgeblich ist die angegebene Teilplan-Grenze. ")
        result += (f"<h3>{band_title}</h3><p>{band_note}</p>"
                   "<p>Ausgewiesen ist die Stichproben-Standardabweichung von NSD / √PSD<sub>Rauschmodell</sub> "
                   "über die ausgewählten Frequenzpunkte, in Prozent. Sinusumgebungen werden anhand der "
                   "LPSD-ENBW ausgeschlossen. Diese korrelierten Frequenzwerte beschreiben die sichtbare "
                   "Welligkeit eines Datensatzes; sie sind keine unabhängigen Wiederholungen.</p>"
                   + table(["Signal", "LNSD [%]", "FFTW einfach [%]", "FFTW-Hybrid [%]"], rows,
                           classes=[None, "numeric", "numeric", "numeric"]))
    tone = next((row for row in data.cases if row["case"] == "offbin_tone"), None)
    if tone:
        rows = []
        for method in METHODS:
            value = tone["metrics"][method]["peak"]
            rows.append([h(LABEL[method]), h(number(value.get("psd_fwhm_hz"), 9)),
                         h(number(value.get("nsd_fwhm_hz"), 9)),
                         h(number(value.get("far_from_tone_nsd_max", 0) * 1e15, 8))])
        result += ("<h3>Isolierter Ton: native Linienbreiten und ferner Leckboden</h3>" + table(
            ["Methode", "PSD-FWHM [Hz]", "NSD-FWHM [Hz]", "Max. ferne NSD [fV/√Hz]"], rows,
            classes=[None, "numeric", "numeric", "numeric"]) +
            "<p class='table-note'>FWHM wird aus nativen PSD-Stützstellen beziehungsweise deren Quadratwurzel bestimmt; "
            "Kreuzungen sind linear interpoliert. Die Abstände des groben Ausgaberasters begrenzen die Genauigkeit, "
            "auch wenn hier zusätzliche Stellen für die Reproduzierbarkeit stehen. Der ferne Leckbereich ist "
            "0,01 &lt; f &lt; 20 Hz und |f − f₀| &gt; 1 Hz. Die dortige NSD stammt aus der gespeicherten öffentlichen Ausgabe.</p>")
    return result


def cache_html(data):
    rows, seen = [], set()
    for row in data.raw.get("preparation", []):
        if row.get("method") != "matched_after":
            continue
        n = row["n"]
        if n in seen:
            continue
        seen.add(n)
        md = row.get("metadata", {})
        low = md.get("low_frequency_preparation") or {}
        mib = lambda value: number(value / 1024**2, 6) if isinstance(value, (int, float)) else "—"
        rows.append([h(integer(n)), h(md.get("kernel_weight_dtype", "—")), h(mib(md.get("cached_kernel_bytes"))),
                     h(mib(md.get("kernel_cache_budget_bytes"))), h(mib(low.get("cached_coefficient_bytes"))),
                     h(mib(md.get("low_frequency_cache_budget_bytes"))),
                     h(mib(md.get("persistent_cache_bytes"))), h(mib(md.get("total_cache_budget_bytes"))),
                     h(str(md.get("streamed_frequency_kernels", "—"))),
                     h(mib(md.get("fftw_array_bytes")))])
    if not rows:
        return ""
    return ("<h3>Tatsächliche Cachebelegung des optimierten Hybrids</h3>" + table(
        ["N", "Gewichtstyp", "Gewichte [MiB]", "Gewichtslimit [MiB]", "LPSD-Koeffizienten [MiB]", "LPSD-Cachelimit [MiB]",
         "Caches belegt [MiB]", "Gesamtlimit [MiB]", "Kerne ohne Gewichtscache", "FFT-/Fensterpuffer [MiB]"],
        rows, classes=[None, None] + ["numeric"] * 8) +
        "<p class='table-note'>Dauerhafte Cachegrenzen und temporäres LPSD-Arbeitsbudget sind getrennte Größen. "
        "Ein protokolliertes Gesamtlimit umfasst ausschließlich beide dauerhaften Caches; bei automatischer "
        "Aufteilung kann der LPSD-Koeffizientencache den nach den Gewichten verbleibenden Anteil nutzen. "
        "Sie sind keine Zusicherung einer gleich hohen Prozess-RSS; Eingang, FFTW-Plan, Interpreter und weitere "
        "Messobjekte kommen hinzu. Die Puffer-Spalte erfasst nur die im Metadatenfeld aufgeführten "
        "FFT-/Fensterarrays, keinen zusätzlichen internen Bluestein-Faltungsraum. Ein nicht gespeicherter "
        "Kernel behält seinen vollständigen Kaiser-Stützbereich. Der einmalig optimierte Hybrid baut "
        "keinen dieser beiden dauerhaften Caches auf.</p>")


def cache_values(group, *keys):
    values = []
    for entry in group["entries"]:
        value = entry.get("metadata", {}).get("pipeline", {})
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(value)
    return values


def value_range(values, *, divisor=1, digits=7, exact_integer=False):
    if not values:
        return "—"
    lower, upper = min(values), max(values)
    format_value = integer if exact_integer else lambda value: number(value / divisor, digits)
    return format_value(lower) if lower == upper else f"{format_value(lower)}–{format_value(upper)}"


def cache_bytes_cell(values):
    if not values:
        return "—"
    return (h(value_range(values, divisor=1024**2))
            + f"<small>{h(value_range(values, exact_integer=True))} Bytes</small>")


def cache_studies_html(studies, output_dir, figure):
    if not studies:
        return ""
    times, memory, files, raw_sections = [], [], [], []
    for study in studies:
        files.append([h(study.path.name), h(integer(study.n)),
                      h(number(study.raw.get("max_rss_bytes", 0) / 1024**3, 7))
                      if study.raw.get("max_rss_bytes") else "—"])
        for group in study.groups:
            block_medians = "<br>".join(f"B{block['block'] + 1}: {ms(block['median_wall_s'])} ms"
                                          for block in group["blocks"])
            times.append([h(integer(study.n)), h(number(group["budget_mib"] / 1024)),
                          h(group["fftw_threads"]), f"<strong>{ms(group['median_wall_s'])}</strong>",
                          h(f"{ms(group['min_wall_s'])}–{ms(group['max_wall_s'])}"),
                          f"{len(group['blocks'])} / {group['repetitions']}", block_medians])
            cached_q = cache_values(group, "low_frequency_preparation", "cached_frequencies")
            total_q = cache_values(group, "exact_low_frequency_points")
            memory.append([h(integer(study.n)), h(number(group["budget_mib"] / 1024)),
                           cache_bytes_cell(cache_values(group, "cached_kernel_bytes")),
                           cache_bytes_cell(cache_values(group, "low_frequency_preparation", "cached_coefficient_bytes")),
                           cache_bytes_cell(cache_values(group, "persistent_cache_bytes")),
                           h(f"{value_range(cached_q, exact_integer=True)} / {value_range(total_q, exact_integer=True)}")])
        raw_sections.append(f"<details><summary>Cache-Studie: {h(study.path.name)} · vollständige Rohdaten</summary>"
                            f"<pre>{h(json.dumps(study.raw, ensure_ascii=False, indent=2, allow_nan=False))}</pre></details>")
    figure_html = embedded_figure(output_dir, figure,
        "Zusätzliche serielle NSD-Tuningblöcke. Die Balken zeigen den Median aller einzelnen Rohzeiten; "
        "Rauten markieren die separat berechneten Blockmediane. Gezeigte Min/Max-Spannen sind keine Konfidenzintervalle.")
    return ("<h3>Zusatzstudie: gemeinsames Cachebudget</h3>"
            "<p>Hier wird ausschließlich die gemeinsame Obergrenze für gespeicherte Kaiser-Gewichte und "
            "projizierte LPSD-Koeffizienten untersucht. Jede Konfiguration läuft in einem eigenen "
            "vorbereiteten Block; jeweils ein großes Objekt bleibt resident. Gemessen werden vollständige "
            "wiederverwendete NSD-Aufrufe auf weißem Eingang. Diese seriellen Tuningblöcke sind eine "
            "Zusatzstudie und keine gepaarte Messung gegen die ursprüngliche Implementierung. "
            "Sie verändern weder die Hauptmessdaten noch das dort gewählte Cacheprofil.</p>" + figure_html +
            "<p>Der Gesamtmedian wird direkt aus sämtlichen aufgezeichneten Rohzeiten dieser Konfiguration "
            "berechnet. Zusätzlich bleiben Blockmediane, Aufrufzahlen und Minimum/Maximum sichtbar. "
            "Unterschiede zwischen Blöcken können Schwankungen des gemeinsam genutzten Hosts enthalten.</p>" +
            table(["N", "Cachelimit [GiB]", "FFTW-Threads", "Gesamtmedian NSD [ms]", "Minimum–Maximum [ms]",
                   "Blöcke / Aufrufe", "Blockmediane"], times,
                  classes=[None, "numeric", "numeric", "numeric", "numeric", "numeric", "numeric"]) +
            "<h3>Tatsächliche Cachebelegung in der Zusatzstudie</h3>" +
            table(["N", "Cachelimit [GiB]", "Gewichte [MiB]", "LPSD-Koeffizienten [MiB]",
                   "Caches zusammen [MiB]", "q-Frequenzen gespeichert / Teilplan"], memory,
                  classes=[None] + ["numeric"] * 5) +
            "<p class='table-note'>Angegeben sind die aufgezeichneten Nutzdatenbytes, mit exakten Bytewerten "
            "unter der MiB-Darstellung. Falls sich Werte zwischen Blöcken unterscheiden, wird deren Bereich "
            "ausgewiesen. Nicht belegtes Budget wird nicht als belegter Speicher gezählt. Eingang, FFTW-Plan, "
            "temporäre Arbeitsräume und Interpreter kommen hinzu. Der folgende RSS-Höchstwert gilt für "
            "den gesamten Studienprozess mit mehreren nacheinander ausgeführten Konfigurationen; er "
            "kann keiner einzelnen Cachegrenze zugerechnet werden.</p>" +
            table(["Eigenständige Messdatei", "N", "Prozess-Höchst-RSS der gesamten Studie [GiB]"], files,
                  classes=[None, "numeric", "numeric"]) + "".join(raw_sections))


def embedded_figure(output_dir, files, caption):
    if not files:
        return ""
    svg = (output_dir / files["svg"]).read_text(encoding="utf-8")
    svg = svg[svg.index("<svg"):]
    return f'<figure><div class="chart">{svg}</div><figcaption>{caption}</figcaption></figure>'


def write_html(data, output_dir, figures, cache_studies=(), prime_quality=None):
    rows = data.raw
    settings = data.settings
    cards = []
    for family in ("basic", "matched"):
        for mode in ("reuse", "fresh"):
            pair = data.time_pairs(family, mode, "nsd").get(data.main_n)
            if pair:
                before, after = pair
                name = "Einfaches FFTW" if family == "basic" else "FFTW-Hybrid"
                state = "vorbereitet" if mode == "reuse" else "Kaltstart Klasse"
                ratio = before["median_wall_s"] / after["median_wall_s"]
                cards.append(f'<div class="card"><span>{name} · {state}</span><strong>{number(ratio, 4)}×</strong>'
                             f'<small>NSD: {ms(before["median_wall_s"])} → {ms(after["median_wall_s"])} ms<br>'
                             f'N = {integer(data.main_n)} · vorher / nachher</small></div>')
    once_pair = data.once_pairs("fresh", "nsd").get(data.main_n)
    if once_pair:
        before, after = once_pair
        ratio = before["median_wall_s"] / after["median_wall_s"]
        cards.append(f'<div class="card once"><span>FFTW-Hybrid · einmalig</span><strong>{number(ratio, 4)}×</strong>'
                     f'<small>NSD: {ms(before["median_wall_s"])} → {ms(after["median_wall_s"])} ms<br>'
                     f'N = {integer(data.main_n)} · vorher / Einzelaufruf</small></div>')
    runtime_block = "".join(embedded_figure(output_dir, figures.get(f"runtime_{density}"),
        "Gemessene vollständige Aufrufe. Fehlerbalken sind beobachtete Minimum–Maximum-Spannen; "
        "Eine vorhandene LPSD-/LNSD-Linie zeigt den vollständigen öffentlichen Referenzaufruf. "
        "Orange kennzeichnet den vollständig einmalig optimierten Hybridaufruf; Blau im Neuaufruf "
        "den Kaltstart der vorbereitbaren Klasse.") for density in ("psd", "nsd", "both"))
    spectra_block = "".join(embedded_figure(output_dir, figures.get(density),
        h(native_note(data)) + " Die Eingangsmodellkurve beschreibt den Rauschanteil und ist kein exakter Erwartungswert des endlichen Schätzers.")
        for density in ("nsd", "psd"))
    difference_block = "".join(embedded_figure(output_dir, figures.get(f"difference_{density}"),
        "Absolute und relative Unterschiede auf genau denselben gespeicherten Spektralpunkten. "
        "Nulldifferenzen werden nicht durch ein künstliches positives Minimum ersetzt.") for density in ("psd", "nsd"))
    signal_rows = [[h(TITLES.get(row["case"], row["case"])), h(row.get("description", "")), h(integer(row["n"]))]
                   for row in data.cases]
    signals_table = table(["Signal", "Definition", "N"], signal_rows) if signal_rows else ""
    md = data.preparation()
    low = md.get("low_frequency_preparation") or {}
    weight_dtype = md.get("kernel_weight_dtype", settings.get("weight_dtype", "nicht protokolliert"))
    once_accuracy = any("matched_once" in row.get("implementation_difference", {}) for row in data.cases)
    once_accuracy_text = ("Die Genauigkeit des einmalig optimierten Hybrids wird zusätzlich in den "
                          "Abweichungstabellen ausgewiesen. Für diesen Modus ist keine eigene Spektralkurve "
                          "in der NPZ-Datei gespeichert; die Diagramme zeigen weiterhin die gespeicherten "
                          "Kurven der vorbereitbaren Verfahren." if once_accuracy else "")
    hybrid_text = (
        "Der Hybrid behält sein vollständiges Tukey-Fenster, die kontinuierliche Kaiser-Leistungsantwort, "
        "deren dokumentierten Stützbereich, die reflektierten Frequenzbeiträge und die bisherige Normierung. "
        "Native SIMD-Dots nutzen vorbereitete Gewichte; ohne Cache werden dieselben Gewichte unmittelbar mit "
        "der FFTW-Leistung verrechnet. Der tiefe LPSD-Teilplan verwendet wiederverwendbare projizierte Koeffizienten "
        "und gebündelte unabhängige Frequenzaufgaben. Die festen Puffer sind einmalig für die native Vor- und "
        "Nachverarbeitung gebunden; neue Eingänge werden im öffentlichen Aufruf weiterhin geprüft. "
        "Kein Gewicht wird an eine gemessene Referenzkurve angepasst.")
    environment = {
        "Messbeginn UTC": rows.get("created_utc", "—"), "Messende UTC": rows.get("finished_utc", "nicht im Protokoll"),
        "CPU": rows.get("machine", {}).get("cpu", "—"),
        "CPU-Kontingent (Quota / Periode)": rows.get("machine", {}).get("cpu_quota", "—"),
        "FFTW": rows.get("fftw", {}).get("version", "—"),
        "LPSD-Worker / FFTW-Threads vorher": settings.get("workers", "—"),
        "FFTW-Threads nachher (N:T)": "; ".join(map(str, settings.get("fftw_threads") or [])) or settings.get("workers", "—"),
        "Glättungsworker, vorbereitete Klasse": settings.get("smoothing_workers", md.get("smoothing_workers", "—")),
        "Glättungsworker, einmalig": settings.get("workers", "—") if any(
            row.get("method") == "matched_once" for row in data.raw.get("timings", [])) else "nicht separat gemessen",
        "Speichertyp der Glättungsgewichte": weight_dtype,
        "Gesamtcachelimit angefordert [MiB]": number(settings.get("total_cache_mb")),
        "Restbudget automatisch für LPSD-Cache": ("ja" if settings["auto_low_cache"] else "nein") if "auto_low_cache" in settings else "nicht protokolliert",
        "Explizite Bluestein-Auswahl N:M": "; ".join(map(str, settings.get("bluestein") or [])) or "keine",
        "MEASURE nur für Reuse nachher, N": "; ".join(integer(n) for n in settings.get("measure_lengths") or []) or "keine",
        "FFTW-Zeitlimit je MEASURE-Plan [s]": number(settings.get("planner_time_limit")) if settings.get("measure_lengths") else "nicht verwendet",
        "Python / NumPy / Pandas": " / ".join(str(rows.get("machine", {}).get(key, "—")) for key in ("python", "numpy", "pandas")),
        "Ausgangsversion": rows.get("baseline", {}).get("version", "—"),
        "Ausgangscommit": rows.get("baseline", {}).get("commit", "—"),
        "Optimierter Branch": rows.get("optimized", {}).get("branch", "—"),
        "Basis des optimierten Arbeitsstands": rows.get("optimized", {}).get("base_commit", "—"),
        "Beobachtete maximale Prozess-RSS [GiB]": number(rows.get("max_rss_bytes", 0) / 1024**3, 5) if rows.get("max_rss_bytes") else "—",
    }
    facts = "".join(f"<dt>{h(key)}</dt><dd>{h(value)}</dd>" for key, value in environment.items())
    validated = [row for row in data.cases if row.get("input_unchanged") and row.get("output_ownership_and_repeatability")]
    validation_text = (f"Für {len(validated)} von {len(data.cases)} Spektralfällen dokumentiert das Messprotokoll "
                       "unveränderte Eingänge sowie wiederholbare, eigenständige Ausgangspuffer.") if data.cases else ""
    raw_json = h(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    body = f'''<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FFTW-Optimierung · vollständige Zeiten und unveränderte Spektralschätzer</title><style>
:root{{--ink:{INK};--muted:{MUTED};--blue:{BLUE};--purple:{PURPLE};--line:#dce4ed;--wash:#f2f6fa}}*{{box-sizing:border-box}}
body{{margin:0;background:#fff;color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1360px;margin:auto;padding:44px 36px 60px}}
header{{border-top:6px solid var(--blue);padding-top:20px;margin-bottom:24px}}.eyebrow{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.09em;font-weight:650}}
h1{{font-size:34px;line-height:1.16;letter-spacing:-.025em;margin:12px 0 16px}}h2{{font-size:24px;line-height:1.25;border-top:1px solid var(--line);padding-top:18px;margin:36px 0 17px}}h3{{font-size:17px;margin:25px 0 12px}}
p{{max-width:1120px;margin:11px 0 15px}}.lead{{font-size:17px}}.muted,.table-note,.meta{{color:var(--muted)}}.meta,.table-note{{font-size:12px}}a{{color:var(--blue);text-underline-offset:2px}}code{{background:var(--wash);padding:2px 5px;border-radius:3px;font:0.92em ui-monospace,monospace}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:13px;margin:25px 0}}.card{{background:var(--wash);padding:17px 16px;border-top:3px solid var(--blue)}}.card.once{{border-top-color:{ORANGE}}}.card.once strong{{color:{ORANGE}}}.card span{{display:block;font-size:13px;font-weight:650}}.card strong{{display:block;font-size:32px;margin:5px 0;line-height:1.3}}.card small{{font-size:11px;color:var(--muted)}}
.note{{padding:15px 18px;background:var(--wash);border-left:3px solid var(--purple);margin:24px 0}}figure{{margin:23px 0 30px;border:1px solid var(--line);padding:7px 9px 12px}}.chart svg{{display:block;width:100%;height:auto}}figcaption{{font-size:12px;line-height:1.5;color:var(--muted);margin:7px 12px 0}}
.table-scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums;margin:5px 0 13px}}th,td{{padding:10px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:var(--wash);font-size:11.5px;line-height:1.35;font-weight:650}}tr:nth-child(even) td{{background:#fafbfd}}td.numeric{{white-space:nowrap;text-align:right}}td.numeric small{{display:block;font-size:10px;color:var(--muted);margin-top:3px}}dl.environment{{display:grid;grid-template-columns:240px 1fr;gap:8px 18px;font-size:13px}}dt{{font-weight:650}}dd{{margin:0;overflow-wrap:anywhere}}details{{border:1px solid var(--line);padding:15px 18px;margin:24px 0}}summary{{cursor:pointer;font-weight:650}}pre{{font:11px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}}footer{{border-top:1px solid var(--line);padding-top:17px;margin-top:35px;font-size:12px;color:var(--muted)}}
@media(max-width:850px){{main{{padding:25px 15px 38px}}h1{{font-size:28px}}h2{{font-size:22px}}.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}.lead{{font-size:15px}}figure{{padding:2px;margin-left:-7px;margin-right:-7px}}figcaption{{margin:8px 9px}}th,td{{padding:8px;font-size:11px}}dl.environment{{grid-template-columns:155px 1fr}}}}
@media(max-width:430px){{.cards{{grid-template-columns:1fr}}.card{{padding:11px 14px}}.card strong{{font-size:28px}}}}
@media print{{@page{{size:A4;margin:12mm}}body{{font-size:10pt}}main{{padding:0;max-width:none}}h1{{font-size:24pt}}h2{{font-size:16pt;break-after:avoid}}h3{{break-after:avoid}}figure{{break-inside:avoid;padding:0}}table{{font-size:9pt}}thead{{display:table-header-group}}tr{{break-inside:avoid}}.table-scroll{{overflow:visible}}details{{display:none}}}}
</style></head><body><main><header><div class="eyebrow">Serialisierte Messung · zwei FFTW-Pfade</div><h1>FFTW-Optimierung: vollständige Zeiten und Spektren</h1>
<p class="lead">Die einfache logarithmische FFTW-Auswertung und der an LPSD angenäherte FFTW-Hybrid werden jeweils mit ihrer vorherigen Implementierung verglichen.
Die Zeitmessungen umfassen vollständige Aufrufe. Zusätzliche Spektralplots vergleichen die optimierten Verfahren mit der schnellen LPSD-/LNSD-Referenz.</p>
<p class="meta">Messprotokoll: <code>{h(data.path.name)}</code> · verwendete einfache FFTW-Operationen: <code>{h(settings.get('operations', '—'))}</code></p></header>
<div class="cards">{''.join(cards)}</div><p class="meta">Faktor = Median vorher / Median nachher. Werte über 1 bedeuten einen kürzeren optimierten Aufruf; Werte unter 1 einen längeren.</p>
<div class="note"><strong>Zwei getrennte Vergleiche</strong><p>Vorher/nachher prüft Implementierungsänderungen am selben Schätzer.
LPSD gegenüber FFTW vergleicht unterschiedliche Schätzer. Ähnliche Rauschwelligkeit und Peak-Breite verlangen keine identischen Rauschzacken an jeder Frequenz.</p></div>
<h2>1 · Vollständige Berechnungszeiten</h2>
<p>Die Messungen verwenden weißes Rauschen und wechseln die Methoden in serialisierten, zufällig angeordneten Runden.
„Vorbereitet“ enthält den vollständigen datenabhängigen Aufruf mit wiederverwendeten Plänen, Fenstern und Caches.
„Kaltstart Klasse“ enthält zusätzlich das Verwerfen der FFTW-Wisdom, Objektaufbau, Allokationen, Fenster- und Cachevorbereitung sowie Freigabe.
„Einmalig optimiert“ enthält den vollständigen <code>estimate_once</code>-Aufruf nach Wisdom-Löschung: beide dauerhaften Caches sind deaktiviert, nicht gespeicherte native Kaiser-Nenner werden zusammen mit dem ersten Zähler berechnet, anschließend wird das Objekt freigegeben.</p>
<p>Die Tabellen zeigen den Median in Millisekunden; darunter stehen beobachtetes Minimum–Maximum und die Anzahl der Wiederholungen.
Diese Spannweite ist kein Konfidenzintervall. PSD und NSD teilen nahezu die gesamte Rechnung; Unterschiede ihrer Mediane auf dem gemeinsam genutzten virtuellen Rechner können Messstreuung sein.</p>
<p>Importe, Bibliotheksladen, Signalerzeugung, Aufwärmläufe, zusätzliche Ergebnisvergleiche, Hashbildung und Dateiausgabe liegen außerhalb der Zeitmessung.
Die reguläre Eingabeprüfung innerhalb eines öffentlichen Aufrufs bleibt enthalten. {h(fft_path_note(data))}</p>
<p>{h(planner_note(data))} {h(thread_note(data))} Die ausgewiesenen Beschleunigungen vergleichen das gesamte aufgezeichnete Profil einschließlich dieser Thread- und Planerauswahl.</p>
{runtime_block}{timings_html(data)}
<p class="table-note">Kombinierte PSD+NSD-Ausgabe: die vorherige einfache FFTW-API benötigt zwei vollständige Aufrufe, die optimierte einen gemeinsamen. Der Hybrid unterstützte den gemeinsamen Aufruf bereits zuvor. Nicht gemessene Kombinationen erscheinen nicht als Schätzwerte.</p>
{preparation_html(data)}
<h2>2 · Spektren der optimierten Verfahren</h2>{signals_table}{spectra_block}
<p>{h(native_note(data))} Alle gezeigten Werte stammen aus der gespeicherten NPZ-Datei; der Renderer führt keine Spektralberechnung und keine zusätzliche Glättung oder Kurvenanpassung aus.</p>
{quality_html(data)}
{embedded_figure(output_dir, figures.get('peak'), 'Native Linienprofile und absoluter ferner Leckboden. Die Leistungs- und Amplituden-FWHM sind unterschiedliche Größen; die Verbindung zwischen vorhandenen Punkten ist keine zusätzliche Messung.')}
<h2>3 · Numerische Änderungen durch die Optimierung</h2>
<p>{h(validation_text)} Die folgenden Abweichungen beziehen sich immer auf dieselbe Methode vor und nach der Optimierung.
Der relative Fehler wird nur für von null verschiedene vorherige Werte ausgewiesen. Große relative Werte an einem extrem kleinen Leckboden müssen zusammen mit dem absoluten Fehler gelesen werden.</p>
<p>{h(once_accuracy_text)}</p>
{difference_block}{changes_html(data)}
{prime_quality_html(prime_quality)}
<h2>4 · Umsetzung und Speichergrenzen</h2>
<p>Im einfachen FFTW-Pfad werden die gemeinsamen Vor- und Nachverarbeitungsschritte nativ ausgeführt. Der gemessene Operationsmodus steht im Kopf des Berichts und in den Rohdaten.
Die vorherige und die optimierte Variante behalten dasselbe vollständige Fenster, denselben Frequenzplan und dieselbe Einseiten-Normierung.</p>
<p>{h(hybrid_text)}</p><p>PSD wird nach der Leistungsrechnung auf float32 gerundet. NSD folgt der erhaltenen Ausgabe-Route: komplexe float32-Quadratwurzel dieser gerundeten PSD, anschließend Realteil.
Eingangsdaten, Fenster, interne FFT, Leistungsspektrum und Akkumulation bleiben float64.
Der gespeicherte Gewichtstyp dieses Profils ist <code>{h(weight_dtype)}</code>. Bei Float32-Gewichten werden ausschließlich die zuvor in Float64 normalisierten Cachewerte einmal gerundet; LPSD-Koeffizienten bleiben Float64.</p>
<p>Die native LPSD-Teilberechnung erhält die im Ausgangsstand vorhandene Segment-Rekurrenz: bei mindestens zwei Segmenten entfällt in exakter Arithmetik der Beitrag des ersten Segments durch den Divisor <code>ii</code> statt <code>ii + 1</code>.
Die geerbte zweite Momentenaktualisierung ist ebenfalls keine validierte laufende Varianz. <code>psd_std</code> wird hier nicht als unabhängige statistische Unsicherheit verwendet.</p>
<p>Die angegebene Kaiser-ENBW beschreibt das Ziel-Fenster; sie ist keine separat verifizierte ENBW der gesamten Vollfenster-/Frequenzfaltung.
Das Tukey-Vollfenster übernimmt nicht automatisch die Nebenkeulenunterdrückung des Ziel-Kaiserfensters. Deshalb zeigt der Bericht zusätzlich den absoluten fernen Leckboden.</p>
{cache_html(data)}
{cache_studies_html(cache_studies, output_dir, figures.get('cache_study'))}
<p class="meta">LPSD-Arbeitsbudget der Hauptmessung: {h(number(md.get('low_frequency_working_budget_bytes', 0) / 1024**2)) if md else '—'} MiB.
Native LPSD-Punkte mit gespeicherten Koeffizienten: {h(low.get('cached_frequencies', '—'))}; ohne Koeffizientencache: {h(low.get('uncached_frequencies', '—'))}.</p>
<p class="table-note">FFTW-Planerzustand, Threadkonfiguration, Wisdom und Zeitlimit sind global. Konstruktoren und Änderungen dieses Zustands müssen seriell ausgeführt werden.
Ein Berechnungsobjekt ist nicht reentrant; verschiedene vorbereitete Instanzen dürfen unabhängig ausführen. Diese Messung führt die Vergleichsaufrufe seriell aus.</p>
<h2>5 · Reproduzierbarkeit und Rohwerte</h2><dl class="environment">{facts}</dl>
<p>Der optimierte Stand ist ein Arbeitsstand auf der angegebenen Ausgangsbasis. Das Messprotokoll enthält Hashes der tatsächlich verwendeten Quelldateien und Native-Bibliotheken.
Alle Wiederholungen, Vorbereitungsphasen, Prüfwerte und Parameter sind unten eingebettet. Die originalen Spektralvektoren stehen in der begleitenden <code>.spectra.npz</code>-Datei.</p>
<p class="table-note">Zusätzliche instrumentierte Phasenprofile sind keine Medianmessungen. Die einfachen FFTW-Profile wurden ohne gesonderte Ausgabewahl aufgerufen: optimiert mit der Standard-PSD, die Referenz mit ihrem zuletzt eingestellten Dichtemodus. Hybridprofile verlangen ausdrücklich NSD. Deshalb werden diese Profilsummen nicht als NSD-Gesamtzeiten oder als Ersatz für die separat gemessenen Aufrufzeiten dargestellt.</p>
<details><summary>Vollständiges JSON-Messprotokoll mit maschinellen Werten</summary><pre>{raw_json}</pre></details>
<footer>Alle Diagramme sind als SVG in dieser Datei eingebettet und zusätzlich als PNG/SVG verfügbar. Der Bericht benötigt keine Internetverbindung.
Erzeugt ausschließlich aus gespeicherten Messwerten durch <code>render_fftw_optimized.py</code>.</footer></main></body></html>'''
    target = output_dir / "report_fftw_optimized.html"
    target.write_text(body, encoding="utf-8")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="JSON aus bench_fftw_optimized.py")
    parser.add_argument("--spectra", type=Path, help="Optionaler NPZ-Pfad; sonst <JSON-Stamm>.spectra.npz")
    parser.add_argument("--cache-study", type=Path, nargs="+", default=[],
                        help="Optionale abgeschlossene totalcache-Tuningdateien, getrennt vom Hauptvergleich")
    parser.add_argument("--prime-quality", type=Path,
                        help="Optionale zusätzliche abgeschlossene Qualitätsprüfung, nur separate HTML-Tabellen")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    data = load_dataset(args.results.resolve(), (args.spectra or args.results.with_suffix(".spectra.npz")).resolve())
    cache_studies = load_cache_studies(args.cache_study)
    prime_quality = load_prime_quality(args.prime_quality)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    figures = {}
    if cache_studies:
        figures["cache_study"] = cache_study_plot(cache_studies, args.output_dir)
    for density in ("psd", "nsd", "both"):
        files = runtime_plot(data, args.output_dir, density)
        if files:
            figures[f"runtime_{density}"] = files
    if data.cases:
        for density in ("psd", "nsd"):
            figures[density] = overview(data, args.output_dir, density)
            figures[f"difference_{density}"] = difference_figure(data, args.output_dir, density)
        peak = peak_figure(data, args.output_dir)
        if peak:
            figures["peak"] = peak
    report = write_html(data, args.output_dir, figures, cache_studies, prime_quality)
    print(json.dumps({"report": str(report.resolve()), "figures": figures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
