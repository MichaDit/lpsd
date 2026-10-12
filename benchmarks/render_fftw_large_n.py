#!/usr/bin/env python3
"""Render recorded large-N spectra and timings, without running a benchmark.

Usage:
    python benchmarks/render_fftw_large_n.py --results results_large_n_final.json \
        --output-dir figures [--disk-results results_disk.json] \
        [--validation validation_large_n.json]

Only jobs with status="completed" contribute spectra or measured time values.
Resource rejections, failures and historical attempts remain visible in HTML.
Optional file-backed calls are a separate experiment, never pooled with RAM
timings. Dependencies: Python standard library, NumPy and Matplotlib only.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


MIB, GIB = 1024**2, 1024**3
INK, MUTED, GRID = "#223043", "#65758A", "#DEE5EE"
BLUE, PURPLE, GREY, ORANGE = "#1768AC", "#4E365F", "#8391A1", "#D45E00"
METHODS = ("fftw", "matched", "matched_once", "lpsd")
COLORS = dict(zip(METHODS, (GREY, BLUE, ORANGE, PURPLE)))
LABELS = {"fftw": "FFTW einfach", "matched": "FFTW-Hybrid · mit Cache",
          "matched_once": "FFTW-Hybrid · Einzelaufruf", "lpsd": "LPSD / LNSD"}
UNITS = {"psd": "nV²/Hz", "nsd": "nV/√Hz"}
SCALES = {"psd": 1e18, "nsd": 1e9}
# Recorded releases for the original RAM and file-backed workspace revisions.
# Later native releases come from the explicit endpoint transition provenance.
RECORDED_RELEASES = {
    "cfa8a5a3b1a619d71ca756132ebed185f65a5df2": "1.0.6+fast.5",
    "5d5f5d8564928cd27135fdad63bfd042c9be5853": "1.0.6+fast.5",
}
STATUS = {
    "completed": "Abgeschlossen",
    "completed_with_resource_checks": "Abgeschlossen mit Ressourcenprüfung",
    "not_executed_memory_limit": "Nicht ausgeführt: RAM-Untergrenze überschritten",
    "memory_allocation_failed": "Speicheranforderung unter Adressraumgrenze gescheitert",
    "not_executed_disk_capacity": "Nicht ausgeführt: unzureichender Dateispeicher",
    "failed": "Fehlgeschlagen", "child_failed_without_result": "Prozess ohne Ergebnis beendet",
    "child_failed_after_checkpoint": "Prozess nach Zwischenstand fehlgeschlagen",
    "timeout": "Zeitlimit erreicht", "running": "Läuft", "starting": "Gestartet",
    "interrupted": "Abgebrochen",
    "incomplete": "Unvollständig", "missing": "Kein abgeschlossener Datensatz",
}


def h(value):
    return escape(str(value), quote=True)


def number(value, digits=5):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "—"
    return f"{value:.{digits}g}".replace(".", ",")


def integer(value):
    return f"{int(value):,}".replace(",", " ")


def label(method, density=None, *, disk=False):
    if disk and method == "matched":
        return "FFTW-Hybrid"
    if method == "lpsd" and density in ("psd", "nsd"):
        return "LPSD" if density == "psd" else "LNSD"
    return LABELS.get(method, "Alle angeforderten Methoden" if method is None else str(method))


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlesize": 11,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.edgecolor": "#95A4B6",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
        "axes.formatter.use_mathtext": True, "lines.solid_capstyle": "round",
    })


def decorate(ax, *, log_x=False, log_y=False):
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(which="major", color=GRID, lw=.65)
    ax.grid(which="minor", color=GRID, lw=.4, alpha=.4)
    ax.set_axisbelow(True)


def header(fig, title, subtitle, handles=None, columns=4):
    fig.text(.065, .955, title, fontsize=22, fontweight="bold", va="top")
    fig.text(.065, .902, subtitle, fontsize=11, color=MUTED, va="top")
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.065, .862),
                   ncol=columns, frameon=False, fontsize=10, borderaxespad=0, columnspacing=1.7)


def save_figure(fig, output_dir, stem, title):
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=190, bbox_inches="tight", pad_inches=.16)
    plt.close(fig)
    return {"title": title, "png": f"{stem}.png", "svg": f"{stem}.svg"}


@dataclass
class Report:
    path: Path
    raw: dict

    @property
    def settings(self):
        return self.raw.get("settings", {})

    @property
    def jobs(self):
        return self.raw.get("jobs", [])

    @property
    def completed(self):
        return [job for job in self.jobs if job.get("status") == "completed"]

    def job(self, n, method):
        return next((job for job in self.jobs if job.get("n") == n and job.get("method") == method), None)

    @property
    def cache_mib(self):
        values = sorted({job.get("settings", {}).get("total_cache_mb", self.settings.get("total_cache_mb"))
                         for job in self.completed if job.get("method") == "matched"}
                        - {None})
        return " / ".join(number(value) for value in values) if values else number(self.settings.get("total_cache_mb"))


def load_report(path, schema):
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != schema or not isinstance(raw.get("jobs"), list):
        raise ValueError(f"Nicht unterstütztes Messformat: {path}; erwartet {schema}")
    seen = set()
    for job in raw["jobs"]:
        key = (job.get("n"), job.get("method"))
        if key in seen:
            raise ValueError(f"Mehrere aktuelle Jobs für N/Methode {key}; historische Versuche getrennt ablegen.")
        seen.add(key)
        if job.get("status") != "completed":
            continue
        if schema == "lpsd-fftw-large-n-v1":
            for row in job.get("timings", []):
                times = [entry.get("wall_s") for entry in row.get("repetitions", [])]
                if not times or any(not finite(value) or value <= 0 for value in times):
                    raise ValueError(f"Ungültige vollständige Zeitreihe: {key}/{row.get('outputs')}")
                if not math.isclose(statistics.median(times), row["median_wall_s"], rel_tol=1e-12):
                    raise ValueError(f"Median widerspricht Rohzeiten: {key}/{row.get('outputs')}")
        elif not finite(job.get("timing", {}).get("wall_s")) or job["timing"]["wall_s"] <= 0:
            raise ValueError(f"Fehlende beobachtete Disk-Gesamtzeit: {key}")
        spectrum = job.get("spectrum")
        if spectrum:
            f = np.asarray(spectrum.get("frequency_hz"), dtype=float)
            if f.ndim != 1 or len(f) == 0 or not np.isfinite(f).all() or np.any(f <= 0) or np.any(np.diff(f) <= 0):
                raise ValueError(f"Ungültiges Spektralraster: {key}")
            for density in ("psd", "nsd"):
                values = np.asarray(spectrum.get(density), dtype=float)
                if values.shape != f.shape or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError(f"Ungültige gespeicherte {density.upper()}: {key}")
    return Report(path, raw)


def timing(report, n, method, density, modes):
    job = report.job(n, method)
    if not job or job.get("status") != "completed":
        return None
    rows = [row for row in job.get("timings", []) if row.get("outputs") == density and row.get("mode") in modes]
    if len(rows) > 1:
        raise ValueError(f"Mehrdeutige Zeitreihe: {n}/{method}/{density}/{modes}")
    return rows[0] if rows else None


def missing_note(job):
    status = (job or {}).get("status", "missing")
    if status == "memory_allocation_failed":
        return "Speicherfehler"
    if status == "not_executed_memory_limit":
        return "RAM-Grenze"
    if status == "not_executed_disk_capacity":
        return "Dateispeicher"
    if status in ("running", "starting"):
        return "noch offen"
    if status == "timeout":
        return "Prozesszeitlimit"
    return "nicht gemessen" if status in ("missing", "completed") else "fehlgeschlagen"


def runtime_figure(report, output_dir, density):
    lengths = sorted({job["n"] for job in report.completed if any(
        row.get("outputs") == density for row in job.get("timings", []))})
    if not lengths:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 8.3))
    fig.subplots_adjust(left=.075, right=.985, top=.775, bottom=.28, wspace=.24)
    title = f"Große Datensätze · vollständige {density.upper()}-Aufrufe"
    handles = [Patch(facecolor=COLORS[method], label=label(method, density)) for method in METHODS]
    header(fig, title, f"Isolierte Methodenprozesse · Hybridklasse: {report.cache_mib} MiB Cachelimit · beobachtete Zeiten", handles)
    for panel, (ax, modes, methods) in enumerate(zip(
            axes, (("reuse",), ("fresh", "complete")), (("fftw", "matched"), METHODS))):
        ax.set_title(("(a)  FFTW mit vorhandenen Plänen und Caches" if panel == 0 else
                      "(b)  Vollständiger Neuaufbau / öffentlicher LPSD-Aufruf"), loc="left", fontweight="bold", pad=15)
        available = [timing(report, n, method, density, modes) for n in lengths for method in methods]
        available = [row for row in available if row]
        if not available:
            ax.text(.5, .5, "Keine abgeschlossene Messung", transform=ax.transAxes, ha="center", color=MUTED)
            ax.set_axis_off()
            continue
        lo = min(row["min_wall_s"] for row in available)
        hi = max(row["max_wall_s"] for row in available)
        width = .74 / len(methods)
        for i, n in enumerate(lengths):
            for k, method in enumerate(methods):
                x = i + (k - (len(methods) - 1) / 2) * width
                row = timing(report, n, method, density, modes)
                if row is None:
                    ax.text(x, .025, missing_note(report.job(n, method)), transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", rotation=90, fontsize=7.5, color=MUTED)
                    continue
                value, low, high = row["median_wall_s"], row["min_wall_s"], row["max_wall_s"]
                color = COLORS[method]
                ax.bar(x, value, width=width * .82, color=color, alpha=.85, zorder=3)
                ax.errorbar(x, value, yerr=[[value - low], [high - value]], fmt="none", ecolor=INK,
                            capsize=3, elinewidth=.9, zorder=5)
                raw = np.asarray([entry["wall_s"] for entry in row["repetitions"]])
                ax.scatter(x + np.linspace(-width * .15, width * .15, len(raw)), raw,
                           color="white", edgecolors=INK, linewidths=.5, s=15, zorder=6)
                ax.text(x, high * 1.12, f"{number(value, 4)} s", color=color, ha="center", va="bottom",
                        fontsize=8.8, fontweight="bold")
        ax.set_xticks(np.arange(len(lengths)), [integer(n) for n in lengths])
        ax.set_xlim(-.58, len(lengths) - .42)
        ax.set_ylim(lo / 1.7, hi * 2.1)
        ax.set_xlabel("Anzahl Eingangswerte N", labelpad=10)
        ax.set_ylabel("Verstrichene Zeit [s] · logarithmisch")
        decorate(ax, log_y=True)
    fig.text(.075, .166, "Balken: Median · Punkte: einzelne Aufrufe · Spanne: Minimum–Maximum, kein Konfidenzintervall. Nur abgeschlossene Jobs liefern Zeitwerte.", fontsize=9, color=MUTED)
    fig.text(.075, .135, "Links: Aufbau und Freigabe außerhalb der Zeit. Rechts: Aufbau, Berechnung und Freigabe enthalten; LPSD/LNSD als vollständiger öffentlicher Aufruf.", fontsize=9, color=MUTED)
    fig.text(.075, .104, f"FFTW: ESTIMATE, {report.settings.get('fftw_threads', '—')} Threads · LPSD: {report.settings.get('workers', '—')} Worker · Glättung: {report.settings.get('smoothing_workers', '—')} Worker · keine extrapolierten Zeiten.", fontsize=9, color=MUTED)
    fig.text(.075, .073, f"Hybridklasse: {report.cache_mib} MiB gemeinsamer Cache statt 2048 MiB in der Vorstudie. Der optimierte Einzelaufruf baut keine dauerhaften Caches auf.", fontsize=9, color=MUTED)
    fig.text(.075, .042, "PSD und NSD teilen fast die gesamte Rechnung. Unterschiede separater Mediane können Messstreuung auf dem virtuellen Host sein.", fontsize=9, color=MUTED)
    return save_figure(fig, output_dir, f"runtime_large_n_{density}", title)


def memory_figure(report, output_dir, disk=None):
    jobs = [job for job in report.jobs if job.get("n") == 1_000_000_000
            and finite(job.get("memory_lower_bound_bytes"))]
    if not jobs:
        return None
    jobs.sort(key=lambda job: METHODS.index(job["method"]) if job.get("method") in METHODS else len(METHODS))
    limits = {job.get("memory_cgroup_limit_bytes", report.settings.get("memory_limit_bytes")) for job in jobs}
    limits.discard(None)
    if len(limits) != 1:
        raise ValueError("Die Speicherabbildung benötigt ein eindeutiges Maschinenlimit.")
    limit = limits.pop() / GIB
    fig, ax = plt.subplots(figsize=(13.8, 7.3))
    fig.subplots_adjust(left=.235, right=.97, top=.765, bottom=.25)
    title = "1 Milliarde Proben · Speichergrenze der bestehenden RAM-Pfade"
    header(fig, title, "Bekannte gleichzeitig benötigte Puffer · analytische Untergrenze, keine ausgeführte RSS-Messung")
    values = [job["memory_lower_bound_bytes"] / GIB for job in jobs]
    labels = [label(job["method"]) + (" · L = N" if job["method"] == "lpsd" else "") for job in jobs]
    y = np.arange(len(jobs))
    ax.axvspan(0, limit, color="#E8F1EC", zorder=0)
    for k, (job, value) in enumerate(zip(jobs, values)):
        ax.barh(k, value, color=COLORS.get(job.get("method"), GREY), height=.58, alpha=.83)
        ax.text(value + .5, k, f"≥ {number(value, 4)} GiB", va="center", fontweight="bold", fontsize=11)
    ax.axvline(limit, color="#B54D51", lw=1.5, ls="--")
    ax.text(limit + .35, -.55, f"Maschinenlimit: {number(limit)} GiB", color="#B54D51", fontsize=10, va="bottom")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(values) * 1.2, limit * 1.3))
    ax.set_xlabel("Bekannte Puffer einschließlich Float64-Eingang [GiB]", labelpad=10)
    ax.grid(axis="x", color=GRID, lw=.65)
    ax.set_axisbelow(True)
    fig.text(.065, .145, "Untergrenzen aus den Job-Protokollen. FFTW-interner Planspeicher, weitere Arbeitsbereiche und Caches können zusätzlich erforderlich sein.", fontsize=9, color=MUTED)
    fig.text(.065, .109, "LPSD: längster Frequenzpunkt mit Segmentlänge L = N. Hybrid: ohne Gewichts-/Koeffizientencaches und ohne tiefen LPSD-Arbeitsbereich.", fontsize=9, color=MUTED)
    disk_note = ("Die bereits durchgeführten dateibasierten Messungen werden separat ausgewiesen."
                 if disk is not None and disk.completed else
                 "Ein optionaler Versuch mit dateibasierten Puffern wird separat ausgewiesen.")
    fig.text(.065, .073, "Für diese RAM-Pfade wird daraus keine Laufzeit abgeleitet. " + disk_note, fontsize=9, color=MUTED)
    return save_figure(fig, output_dir, "memory_large_n_1b", title)


def spectrum_figure(report, output_dir):
    n = 100_000_000
    jobs = [job for job in report.completed if job.get("n") == n and job.get("spectrum")]
    if not jobs:
        return None
    jobs.sort(key=lambda job: METHODS.index(job["method"]) if job.get("method") in METHODS else len(METHODS))
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 8.0))
    fig.subplots_adjust(left=.075, right=.985, top=.775, bottom=.235, wspace=.23)
    title = "100 Millionen Proben · PSD und NSD des weißen Rauschens"
    points = sorted({len(job["spectrum"]["frequency_hz"]) for job in jobs})
    handles = [Line2D([], [], color=COLORS[job["method"]], lw=1.5,
                      ls=":" if job["method"] == "matched_once" else "-", label=label(job["method"])) for job in jobs]
    handles.append(Line2D([], [], color="#27332A", lw=1.1, ls="--", label="Quellmodell"))
    header(fig, title, f"fₛ = {number(report.settings.get('sample_rate'))} Hz · {', '.join(map(str, points))} native Ausgabepunkte · Hybridcache {report.cache_mib} MiB", handles, len(handles))
    matched = report.job(n, "matched")
    md = (matched or {}).get("preparation", {}).get("metadata", {})
    cutoff = md.get("exact_low_frequency_max_hz")
    for ax, density in zip(axes, ("psd", "nsd")):
        ax.set_title(f"{density.upper()} · gespeicherte Spektralwerte", loc="left", fontweight="bold", pad=15)
        for job in jobs:
            spectrum, method = job["spectrum"], job["method"]
            f = np.asarray(spectrum["frequency_hz"])
            y = np.asarray(spectrum[density]) * SCALES[density]
            ax.plot(f, np.where(y > 0, y, np.nan), color=COLORS[method],
                    lw=1.0 if method == "matched_once" else 1.4,
                    ls=":" if method == "matched_once" else "-", alpha=.9)
        first, last = jobs[0]["spectrum"]["frequency_hz"][0], jobs[0]["spectrum"]["frequency_hz"][-1]
        if finite(cutoff) and cutoff > first:
            ax.axvspan(first, cutoff, color="#EEF1F4", zorder=0)
        ax.axhline(100 if density == "psd" else 10, color="#27332A", lw=1.05, ls="--")
        ax.set_xlim(first, last)
        ax.set_xlabel("Frequenz [Hz]")
        ax.set_ylabel(f"{density.upper()} [{UNITS[density]}]")
        decorate(ax, log_x=True, log_y=True)
    native = (f"Grau hinterlegt: unveränderte native LPSD-Teilpunkte im Hybrid bis {number(cutoff, 6)} Hz. "
              if finite(cutoff) else "")
    fig.text(.075, .144, native + "Die Kosten dieser Teilberechnung sind in den Hybridzeiten enthalten.", fontsize=9, color=MUTED)
    fig.text(.075, .106, "Quellmodell: einseitige NSD 10 nV/√Hz, entsprechend PSD 100 nV²/Hz. Kein exakter Erwartungswert des endlichen gefensterten Schätzers.", fontsize=9, color=MUTED)
    fig.text(.075, .068, "Nur gespeicherte native Punkte: keine zusätzliche Glättung, Kurvenanpassung oder künstliche Verfeinerung des Frequenzrasters.", fontsize=9, color=MUTED)
    return save_figure(fig, output_dir, "white_spectra_100m", title)


def disk_figure(report, output_dir):
    if report is None or not report.completed:
        return None
    lengths = sorted({job["n"] for job in report.jobs})
    methods = [method for method in ("fftw", "matched", "lpsd")
               if method in report.settings.get("methods", []) or any(job.get("method") == method for job in report.jobs)]
    fig, ax = plt.subplots(figsize=(13.8, 7.5))
    fig.subplots_adjust(left=.08, right=.98, top=.765, bottom=.275)
    title = "Dateibasierte Puffer · ein beobachteter PSD+NSD-Gesamtaufruf"
    header(fig, title, "Eigenständiges Experiment · andere exakte N-Punkt-FFT-Zerlegung · kein RAM-Reuse-Vergleich",
           [Patch(facecolor=COLORS[method], label=label(method, disk=True)) for method in methods], len(methods))
    values = [job["timing"]["wall_s"] for job in report.completed]
    width = .72 / max(1, len(methods))
    for i, n in enumerate(lengths):
        for k, method in enumerate(methods):
            x = i + (k - (len(methods) - 1) / 2) * width
            job = report.job(n, method) or report.job(n, None)
            if not job or job.get("status") != "completed":
                ax.text(x, .035, label(method, disk=True) + "\n" + missing_note(job),
                        transform=ax.get_xaxis_transform(), rotation=90, ha="center", va="bottom",
                        color=COLORS[method], fontsize=8)
                continue
            value = job["timing"]["wall_s"]
            ax.bar(x, value, width=width * .8, color=COLORS[method], alpha=.85)
            ax.text(x, value * 1.12, f"{number(value, 5)} s", ha="center", va="bottom",
                    color=COLORS[method], fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(len(lengths)), [integer(n) for n in lengths])
    ax.set_xlabel("Anzahl Eingangswerte N", labelpad=10)
    ax.set_ylabel("Ein vollständiger PSD+NSD-Aufruf [s] · logarithmisch")
    ax.set_ylim(min(values) / 1.8, max(values) * 2.0)
    ax.set_xlim(-.55, len(lengths) - .45)
    decorate(ax, log_y=True)
    fig.text(.08, .16, "Je Balken genau eine Beobachtung: kein Median, keine Wiederholungsstreuung und kein Konfidenzintervall.", fontsize=9, color=MUTED)
    fig.text(.08, .124, "Enthalten: Arbeitsdateien, Planung, Berechnung, tatsächliche Seiteneinlagerung, Kontrollprotokolle und enthaltene Energiediagnostik.", fontsize=9, color=MUTED)
    fig.text(.08, .088, "Ausgeschlossen: Eingangserzeugung, Hashbildung, Prozessstart und Importe. Temporäre Dateien ohne erzwungene abschließende fsync-Sicherung.", fontsize=9, color=MUTED)
    fig.text(.08, .052, cohort_summary(report) or "Andere Speicherstrategie bei unverändertem mathematischem Schätzer; Rundung und Messumfang können von den RAM-Aufrufen abweichen.", fontsize=9, color=MUTED)
    return save_figure(fig, output_dir, "runtime_large_n_disk", title)


def disk_spectrum_jobs(report, n):
    if report is None:
        return []
    return [job for method in ("lpsd", "fftw", "matched") for job in report.completed
            if job.get("n") == n and job.get("method") == method and job.get("spectrum")]


def disk_native_cutoff(job):
    """Return the last point of the explicitly recorded contiguous low mask."""
    if not job or job.get("status") != "completed" or not job.get("spectrum"):
        return None
    f = np.asarray(job["spectrum"]["frequency_hz"], dtype=float)
    mask = np.asarray(job.get("details", {}).get("smoothing", {}).get("low_mask", []), dtype=bool)
    if mask.shape != f.shape or not len(mask) or not mask[0]:
        return None
    first_high = np.flatnonzero(~mask)
    stop = int(first_high[0]) if len(first_high) else len(mask)
    return float(f[stop - 1])


def comparable_disk_spectra(job, reference):
    if not job or job.get("status") != "completed" or not job.get("spectrum"):
        return False, "Kein abgeschlossener Spektraljob"
    if not reference or reference.get("status") != "completed" or not reference.get("spectrum"):
        return False, "Keine abgeschlossene dateibasierte LNSD-Referenz"
    fingerprint = job.get("input_sha256_before")
    if not fingerprint or not reference.get("input_sha256_before"):
        return False, "Eingangs-Hash fehlt"
    if fingerprint != reference["input_sha256_before"]:
        return False, "Unterschiedliche Eingangs-Hashes"
    f = np.asarray(job["spectrum"]["frequency_hz"], dtype=float)
    rf = np.asarray(reference["spectrum"]["frequency_hz"], dtype=float)
    if not np.array_equal(f, rf):
        return False, "Unterschiedliche native Frequenzraster; keine Interpolation"
    return True, "Identischer Eingang und identisches Raster"


def disk_spectrum_figure(report, output_dir):
    if not disk_spectrum_jobs(report, 1_000_000_000):
        return None
    lengths = (100_000_000, 1_000_000_000)
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 11.7), squeeze=False)
    fig.subplots_adjust(left=.075, right=.985, top=.79, bottom=.145, hspace=.39, wspace=.22)
    title = "Dateibasierte Spektren · 100 Millionen und 1 Milliarde Proben"
    handles = [Line2D([], [], color=COLORS[method], lw=1.5, label=label(method, disk=True))
               for method in ("lpsd", "fftw", "matched")]
    handles.append(Line2D([], [], color="#27332A", lw=1.05, ls="--", label="Quellmodell"))
    header(fig, title, "Je N ein gespeichertes PSD/NSD-Paar pro abgeschlossener Methode · keine RAM-Spektren bei 1 Milliarde", handles)
    for column, n in enumerate(lengths):
        jobs = disk_spectrum_jobs(report, n)
        cutoff = disk_native_cutoff(report.job(n, "matched"))
        for row, density in enumerate(("psd", "nsd")):
            ax = axes[row, column]
            ax.set_title(f"({chr(97 + 2 * row + column)})  {density.upper()} · N = {integer(n)}",
                         loc="left", fontweight="bold", pad=12)
            if not jobs:
                ax.text(.5, .5, "Kein abgeschlossenes dateibasiertes Spektrum",
                        ha="center", va="center", transform=ax.transAxes, color=MUTED)
                ax.set_axis_off()
                continue
            for job in jobs:
                f = np.asarray(job["spectrum"]["frequency_hz"], dtype=float)
                y = np.asarray(job["spectrum"][density], dtype=float) * SCALES[density]
                ax.plot(f, np.where(y > 0, y, np.nan), color=COLORS[job["method"]],
                        lw=1.25, ls="--" if job["method"] == "matched" else "-")
            first = min(job["spectrum"]["frequency_hz"][0] for job in jobs)
            last = max(job["spectrum"]["frequency_hz"][-1] for job in jobs)
            if finite(cutoff) and cutoff > first:
                ax.axvspan(first, cutoff, color="#EEF1F4", zorder=0)
            ax.axhline(100 if density == "psd" else 10, color="#27332A", ls="--", lw=1.0)
            ax.set_xlim(first, last)
            ax.set_xlabel("Frequenz [Hz]")
            ax.set_ylabel(f"{density.upper()} [{UNITS[density]}]")
            points = sorted({len(job["spectrum"]["frequency_hz"]) for job in jobs})
            ax.text(.025, .96, f"{', '.join(map(str, points))} native Ausgabepunkte", transform=ax.transAxes,
                    color=MUTED, fontsize=8.5, va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
            present = {job["method"] for job in jobs}
            missing = [f"{label(method, disk=True)}: {missing_note(report.job(n, method))}"
                       for method in ("lpsd", "fftw", "matched") if method not in present
                       and method in report.settings.get("methods", ("lpsd", "fftw", "matched"))]
            if missing:
                ax.text(.975, .035, "Keine Kurve für:\n" + "\n".join(missing), transform=ax.transAxes,
                        ha="right", va="bottom", color=MUTED, fontsize=8.5,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94})
            decorate(ax, log_x=True, log_y=True)
    fig.text(.075, .080, "Spalten verwenden ihre jeweils eigene aufgezeichnete Eingangslänge und ihr natives Raster. Es erfolgt keine Interpolation zwischen 100M und 1B.", fontsize=9, color=MUTED)
    fig.text(.075, .055, "Grau: dokumentierter tiefer LPSD-Teilplan des Hybrids. Quellmodell: 10 nV/√Hz; kein exakter Erwartungswert eines endlichen Schätzers.", fontsize=9, color=MUTED)
    fig.text(.075, .030, cohort_summary(report) or "Dateibasierte Aufrufe erzeugen PSD und NSD gemeinsam. Diese gemessenen Spektren heben die Speichergrenze der bestehenden RAM-Pfade nicht auf.", fontsize=9, color=MUTED)
    return save_figure(fig, output_dir, "white_spectra_disk_100m_1b", title)


def disk_ratio_figure(report, output_dir):
    if not disk_spectrum_jobs(report, 1_000_000_000):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.9))
    fig.subplots_adjust(left=.075, right=.985, top=.765, bottom=.245, wspace=.23)
    title = "Dateibasierte NSD · Verhältnis zur jeweiligen LNSD"
    handles = [Line2D([], [], color=COLORS[method], lw=1.5, label=label(method, disk=True)) for method in ("fftw", "matched")]
    handles.append(Line2D([], [], color=PURPLE, lw=1.0, ls="--", label="LNSD-Verhältnis = 1"))
    header(fig, title, "Punktweiser Vergleich nur bei identischem Eingangs-Hash und identischem nativem Frequenzraster", handles, 3)
    for ax, n in zip(axes, (100_000_000, 1_000_000_000)):
        ax.set_title(f"N = {integer(n)} · dateibasierte Verfahren", loc="left", fontweight="bold", pad=12)
        reference = report.job(n, "lpsd")
        drawn, reasons = False, []
        for method in ("fftw", "matched"):
            job = report.job(n, method)
            valid, reason = comparable_disk_spectra(job, reference)
            if not valid:
                reasons.append(f"{label(method, disk=True)}: {reason}")
                continue
            f = np.asarray(reference["spectrum"]["frequency_hz"], dtype=float)
            baseline = np.asarray(reference["spectrum"]["nsd"], dtype=float)
            values = np.asarray(job["spectrum"]["nsd"], dtype=float)
            ratio = np.divide(values, baseline, out=np.full_like(values, np.nan), where=baseline > 0)
            ax.plot(f, np.where(ratio > 0, ratio, np.nan), color=COLORS[method], lw=1.2,
                    ls="--" if method == "matched" else "-")
            drawn = True
        if not drawn:
            ax.text(.5, .5, "Kein gültiger punktweiser Vergleich\n" + "\n".join(reasons),
                    transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9, wrap=True)
            ax.set_axis_off()
            continue
        cutoff = disk_native_cutoff(report.job(n, "matched"))
        if finite(cutoff) and cutoff > f[0]:
            ax.axvspan(f[0], cutoff, color="#EEF1F4", zorder=0)
        ax.axhline(1, color=PURPLE, ls="--", lw=1.0)
        ax.set_xlim(f[0], f[-1])
        ax.set_xlabel("Frequenz [Hz]")
        ax.set_ylabel("NSD / dateibasierte LNSD · logarithmisch")
        decorate(ax, log_x=True, log_y=True)
        if reasons:
            ax.text(.02, .02, "\n".join(reasons), transform=ax.transAxes, fontsize=8, color=MUTED,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    fig.text(.075, .145, "Verglichen werden unterschiedliche Spektralschätzer auf demselben Eingang, keine Optimierungsversionen derselben Implementierung.", fontsize=9, color=MUTED)
    fig.text(.075, .109, "Verhältnis 1 bedeutet gleichen NSD-Wert am nativen Punkt. Ähnliche Rauschwelligkeit verlangt keine identischen Zufallszacken.", fontsize=9, color=MUTED)
    fig.text(.075, .073, "Keine Interpolation; Nullwerte werden auf der logarithmischen Achse nicht als künstlich positive Werte ersetzt. Bandstatistiken stehen im Bericht.", fontsize=9, color=MUTED)
    if cohort_summary(report):
        fig.text(.075, .037, cohort_summary(report), fontsize=9, color=MUTED)
    return save_figure(fig, output_dir, "nsd_ratio_disk_100m_1b", title)


def disk_comparison_table(report):
    rows = []
    for n in sorted({job["n"] for job in report.completed if job.get("spectrum")}):
        reference = report.job(n, "lpsd")
        for method in ("fftw", "matched"):
            job = report.job(n, method)
            valid, reason = comparable_disk_spectra(job, reference)
            if not valid:
                rows.append([h(integer(n)), h(label(method, disk=True)), h(reason), "—", "—", "—", "—"])
                continue
            f = np.asarray(reference["spectrum"]["frequency_hz"], dtype=float)
            baseline = np.asarray(reference["spectrum"]["nsd"], dtype=float)
            values = np.asarray(job["spectrum"]["nsd"], dtype=float)
            band = (f >= .01) & (f <= 20) & (baseline > 0)
            ratio = values[band] / baseline[band]
            delta = 100 * (ratio - 1)
            rows.append([h(integer(n)), h(label(method, disk=True)), h(reason), h(int(np.count_nonzero(band))),
                         h(number(np.median(ratio), 7)) if len(ratio) else "—",
                         h(number(np.sqrt(np.mean(delta**2)), 7)) if len(ratio) else "—",
                         h(number(np.percentile(np.abs(delta), 95), 7)) if len(ratio) else "—"])
    return table(["N", "Methode", "Vergleich mit dateibasierter LNSD", "Punkte 0,01–20 Hz", "Median NSD/LNSD", "RMS der relativen Differenz [%]", "P95 des relativen Betrags [%]"], rows)


def table(headers, rows):
    if not rows:
        return "<p class='muted'>Keine entsprechenden Datensätze im Protokoll.</p>"
    return ("<div class='table-scroll'><table><thead><tr>" + "".join(f"<th>{h(value)}</th>" for value in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def status_cell(job):
    status = job.get("status", "missing")
    kind = "good" if status == "completed" else "muted" if status.startswith("not_executed") else "problem"
    return f"<span class='{kind}'>{h(STATUS.get(status, status))}</span><small><code>{h(status)}</code></small>"


def rss(job):
    md = job.get("process_memory") or job.get("current_process_memory") or {}
    return md.get("max_rss_bytes", md.get("VmHWM_bytes"))


def rss_cell(job):
    value = rss(job)
    if not finite(value):
        return "—"
    result = h(number(value / GIB, 7))
    if not job.get("process_memory") and job.get("current_process_memory"):
        result += "<small>Bis zum letzten gespeicherten Zwischenstand</small>"
    return result


def timing_table(report):
    rows = []
    modes = {"reuse": "Vorbereitet", "fresh": "Komplett neu", "complete": "LPSD/LNSD komplett"}
    for job in sorted(report.completed, key=lambda j: (j["n"], METHODS.index(j["method"]))):
        for row in job.get("timings", []):
            rows.append([h(integer(job["n"])), h(label(job["method"])), h(str(row.get("outputs", "—")).upper()),
                         h(modes.get(row.get("mode"), row.get("mode"))),
                         f"<strong>{number(row['median_wall_s'], 7)}</strong>",
                         h(number(row["min_wall_s"], 7) + "–" + number(row["max_wall_s"], 7)),
                         h(len(row["repetitions"])), h(number(row.get("median_cpu_s"), 7))])
    return table(["N", "Methode", "Ausgabe", "Messumfang", "Median [s]", "Minimum–Maximum [s]", "Aufrufe", "Median CPU-Zeit [s]"], rows)


def jobs_table(jobs, *, disk=False):
    rows = []
    for job in jobs:
        setup = job.get("preparation", {}).get("observed_setup", {})
        cache = job.get("settings", {}).get("total_cache_mb") if job.get("method") == "matched" else None
        reason = job.get("error") or job.get("reason") or job.get("error_type") or ""
        if job.get("status") == "timeout":
            limit = job.get("controller_process_timeout_seconds", job.get("parent_timeout_seconds", job.get("timeout_seconds")))
            reason = f"Prozesszeitlimit: {number(limit)} s; kein abgeschlossener Spektralaufruf"
        if job.get("reason_not_selected"):
            reason = (reason + "; " if reason else "") + str(job["reason_not_selected"])
        if job.get("source_file"):
            reason = (reason + "; " if reason else "") + "Quelle: " + str(job["source_file"])
        cells = [h(integer(job["n"])), h(label(job.get("method"), disk=disk)), status_cell(job)]
        if not disk:
            cells.extend([h(number(cache)), h(number(setup.get("wall_s"), 7))])
        rows.append(cells + [rss_cell(job), h(reason)])
    headers = ["N", "Methode", "Status"]
    if not disk:
        headers.extend(["Hybridcachelimit [MiB]", "Einmal beobachteter Aufbau [s]"])
    headers.extend(["Prozess-RSS-Maximum [GiB]", "Protokollierter Grund / Fehler"])
    return table(headers, rows)


def historical_jobs(raw):
    found = []
    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if "n" in value and "status" in value and ("method" in value or "error" in value):
                found.append(value)
            else:
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        visit(child)
    visit(raw.get("historical_attempts", []))
    return found


def cohort_version(report, cohort):
    version = cohort.get("version")
    if version:
        return str(version)
    transition = report.raw.get("native_endpoint_fix_provenance", {}).get("details", {})
    if (cohort.get("git_commit") == transition.get("commit")
            and cohort.get("native_binary_sha256") == transition.get("native_binary_after_sha256")
            and transition.get("version")):
        return str(transition["version"])
    return RECORDED_RELEASES.get(cohort.get("git_commit"), "nicht gesondert protokolliert")


def cohort_summary(report):
    pieces = []
    for cohort in report.raw.get("source_cohorts", []):
        methods = cohort.get("methods", [])
        name = "FFTW/Hybrid" if methods == ["fftw", "matched"] else "/".join(
            "LPSD/LNSD" if method == "lpsd" else label(method, disk=True) for method in methods)
        version = cohort_version(report, cohort)
        version = version.split("+", 1)[-1] if "+" in version else version
        pieces.append(f"{name}: {version}")
    return "Gemessene Quellenkohorten je Methodenpaar: " + " · ".join(pieces) if pieces else ""


def source_cohorts_html(report):
    cohorts = report.raw.get("source_cohorts") or []
    if not cohorts:
        return ""
    rows = []
    for cohort in cohorts:
        for method in cohort.get("methods", []):
            selected = [job for job in cohort.get("selected_jobs", []) if job.get("method") == method]
            lengths = sorted({job["n"] for job in selected if "n" in job})
            filenames = list(dict.fromkeys(job["source_file"] for job in selected if job.get("source_file")))
            rows.append([
                h(label(method, disk=True)), "<br>".join(h(integer(n)) for n in lengths) or "—",
                h(cohort_version(report, cohort)),
                f"<code>{h(cohort.get('git_commit', '—'))}</code>",
                f"<code>{h(cohort.get('native_binary_sha256', '—'))}</code>",
                "<br>".join(f"<code>{h(name)}</code>" for name in filenames) or "—",
            ])
    transition = report.raw.get("native_endpoint_fix_provenance", {}).get("details", {})
    endpoint_note = (
        "<p>Die neu gemessene volle LPSD verwendet bei beiden Datenlängen den nativen Endpunkt-Fix: "
        "Er begrenzt ausschließlich einen durch Rundungsdrift ungültigen letzten Segmentstart auf "
        "den letzten gültigen Start. Die geerbte Mittelungsrekurrenz bleibt erhalten. Die früheren "
        "nativen Beobachtungen bleiben getrennt in der Historie; die RAM-Zeitreihe wird dadurch "
        "nicht neu gemessen. Spätere Paketversions-Metadaten und ihre CI-Nachweise ersetzen weder "
        "den hier ausgewiesenen Messcommit noch die gemessene native Bibliothek.</p>"
        if transition else ""
    )
    return (
        "<h3>Gemessene Quellenkohorte je Methodenpaar</h3>"
        "<p>Jedes Methodenpaar umfasst die ausgewählten Messungen bei 100 Millionen und einer "
        "Milliarde Proben. Die Konsolidierung prüft innerhalb des Paars gleiche Quellhashes, native "
        "und FFTW-Binärhashes sowie gleiche Recheneinstellungen. Methoden-/Größenauswahl und "
        "Dateipfade werden separat aufgezeichnet. Die verschiedenen Kohorten besitzen eigene "
        "gemessene Revisionen und Bibliotheken; ein gemeinsamer Messcommit für alle Verfahren "
        "wird nicht vorausgesetzt.</p>" +
        table(["Methodenpaar", "Ausgewählte N", "Messstand", "Gemessener Quellcommit",
               "Native Bibliothek · SHA-256", "Quelldateien"], rows) + endpoint_note +
        "<p class='muted'>Ein terminaler Quellcontroller darf wegen eines nicht ausgewählten "
        "Fehlversuchs den Status „unvollständig“ behalten. Der konsolidierte Status ist nur dann "
        "„abgeschlossen“, wenn alle sechs ausgewählten Jobs mit Exitcode 0 abgeschlossen sind, "
        "alle zugehörigen Controller terminal sind und keiner einen <code>parent_error</code> "
        "enthält. Die zugehörigen Auswahlprüfungen stehen im Rohprotokoll.</p>"
    )


def source_runs_html(report):
    runs = report.raw.get("source_runs") or []
    if not runs:
        return ""
    rows = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        filename = run.get("file", run.get("source_file", run.get("path", "—")))
        selected = run.get("selected_jobs")
        if isinstance(selected, list):
            selected_text = "<br>".join(h(f"N = {integer(job['n'])} · {label(job.get('method'), disk=report.raw.get('schema') == 'lpsd-fftw-disk-v1')}")
                                        for job in selected if isinstance(job, dict) and "n" in job) or "keine"
        else:
            selected_text = "nicht gesondert zugeordnet"
        hashes = run.get("source_sha256")
        if isinstance(hashes, dict):
            hash_text = (f"<details><summary>{len(hashes)} Quellhashes</summary><pre>"
                         + h(json.dumps(hashes, ensure_ascii=False, indent=2)) + "</pre></details>")
        else:
            hash_text = h(hashes) if hashes is not None else "—"
        file_hash = run.get("file_sha256", run.get("sha256"))
        if isinstance(file_hash, str):
            hash_text += "<small>Datei-SHA256: <code>" + h(file_hash) + "</code></small>"
        rows.append([f"<code>{h(filename)}</code>", h(run.get("git_commit", run.get("commit", "—"))),
                     selected_text, hash_text])
    return ("<h3>Quelldateien und Auswahl der Messungen</h3>"
            "<p>Die aktuellen <code>jobs</code> enthalten höchstens einen ausgewählten Job je "
            "Datenlänge und Methode. Historische Versuche werden weder zu diesen Zeiten hinzugefügt "
            "noch mit ihnen gemittelt. Die Optionen jedes ausgewählten Jobs sind maßgeblich; "
            "gemeinsame Einstellungen des konsolidierten Protokolls liefern den übergeordneten Kontext. "
            "Insbesondere Prozesszeitlimits und Fortschrittsprotokollierung können sich zwischen "
            "Quellläufen unterscheiden.</p>" + table(["Quelldatei", "Quellcommit", "Ausgewählte Jobs", "Quelltext- und vorhandene Dateihashes"], rows))


def cache_table(report):
    rows = []
    for job in report.completed:
        if job.get("method") != "matched":
            continue
        md = job.get("preparation", {}).get("metadata", {})
        low = md.get("low_frequency_preparation", {})
        byte_cell = lambda value: (h(number(value / MIB, 7)) + f"<small>{integer(value)} Bytes</small>" if finite(value) else "—")
        rows.append([h(integer(job["n"])), byte_cell(md.get("total_cache_budget_bytes")),
                     byte_cell(md.get("cached_kernel_bytes")), byte_cell(low.get("cached_coefficient_bytes")),
                     byte_cell(md.get("persistent_cache_bytes")),
                     h(low.get("cached_frequencies", "—")), h(low.get("uncached_frequencies", "—")),
                     h(number(md.get("exact_low_frequency_max_hz"), 8))])
    return table(["N", "Gesamtlimit [MiB]", "Gewichte [MiB]", "LPSD-Koeffizienten [MiB]", "Tatsächlich belegt [MiB]", "q-Frequenzen im Cache", "q-Frequenzen ohne Cache", "Oberste tiefe LPSD-Frequenz [Hz]"], rows)


def noise_table(report):
    rows = []
    for job in report.completed:
        md = job.get("noise_summary", {}).get("overall", {})
        if not md:
            continue
        rows.append([h(integer(job["n"])), h(label(job["method"], disk=report.raw.get("schema") == "lpsd-fftw-disk-v1")),
                     h(number(md.get("low_hz")) + "–" + number(md.get("high_hz"))), h(md.get("points", "—")),
                     h(number(md.get("mean_nsd_over_source"), 7)),
                     h(number(md.get("normalized_nsd_std") * 100, 7)) if finite(md.get("normalized_nsd_std")) else "—",
                     h(number(md.get("adjacent_difference_rms_over_sqrt2") * 100, 7)) if finite(md.get("adjacent_difference_rms_over_sqrt2")) else "—"])
    return table(["N", "Methode", "Band [Hz]", "Punkte", "Mittel NSD / Quell-NSD", "Normierte NSD-Streuung [%]", "Benachbarte Differenz-RMS / √2 [%]"], rows)


def memory_table(report):
    rows = []
    for job in report.jobs:
        if job.get("n") != 1_000_000_000:
            continue
        lower = job.get("memory_lower_bound_bytes")
        rows.append([h(label(job.get("method"))), h(number(lower / GIB, 8)) if finite(lower) else "—",
                     h(integer(lower)) if finite(lower) else "—", h(job.get("memory_lower_bound_definition", "—")), status_cell(job)])
    return table(["Bestehender RAM-Pfad", "Pufferuntergrenze [GiB]", "Pufferuntergrenze [Bytes]", "Definition aus dem Messharness", "Status"], rows)


def embedded_figure(output_dir, figure, caption):
    if not figure:
        return ""
    raw = (output_dir / figure["svg"]).read_text(encoding="utf-8")
    raw = raw[raw.index("<svg"):]
    return f"<figure><div class='chart'>{raw}</div><figcaption>{h(caption)}</figcaption></figure>"


def raw_section(report, title):
    payload = json.dumps(report.raw, ensure_ascii=False, indent=2, allow_nan=False)
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return (f"<p><a download='{h(report.path.name)}' href='data:application/json;base64,{encoded}'>"
            f"{h(report.path.name)} als JSON herunterladen</a></p>"
            f"<details><summary>{h(title)}</summary><pre>{h(payload)}</pre></details>")


def load_validation(path):
    if path is None:
        return None
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Validierungsprotokoll muss ein JSON-Objekt sein: {path}")
    comparison = raw.get("ram_vs_disk_100m", {})
    if not isinstance(comparison, dict) or not isinstance(comparison.get("checks", []), list):
        raise ValueError(f"Ungültiges ram_vs_disk_100m-Prüfprotokoll: {path}")
    return Report(path, raw)


def boolean_cell(value, *, gate=False):
    if value is True:
        return "<span class='good'>bestanden</span>" if gate else "<span class='good'>ja</span>"
    if value is False:
        return "<span class='problem'>fehlgeschlagen</span>" if gate else "<span class='problem'>nein</span>"
    return "—"


def validation_html(validation):
    if validation is None:
        return ""
    sections = ["<h2>Prüfevidenz der verwendeten Implementierungen</h2>",
                f"<p>Die folgenden Werte stammen aus <code>{h(validation.path.name)}</code>. "
                "Sie sind zusätzliche Prüfungen außerhalb der dargestellten Laufzeitmessungen.</p>"]
    unit = validation.raw.get("unit_tests")
    if isinstance(unit, dict):
        sections.append("<h3>Zentrale Tests</h3>" + table(
            ["Prüfgruppe", "Bestanden", "Fehlgeschlagen"],
            [["Zentrale Implementierungstests", h(unit.get("passed", "—")), h(unit.get("failed", "—"))]]))
    checks = validation.raw.get("ram_vs_disk_100m", {}).get("checks", [])
    rows = []
    for check in checks:
        for density in ("psd", "nsd"):
            values = check.get("outputs", {}).get(density)
            if not isinstance(values, dict):
                continue
            relative = values.get("max_relative_difference")
            rows.append([h(integer(check["n"])), h(label(check.get("method"), disk=True)), h(density.upper()),
                         boolean_cell(check.get("same_input")), boolean_cell(check.get("same_frequency_grid")),
                         h(number(relative * 100, 8)) if finite(relative) else "—",
                         boolean_cell(values.get("gate_passed"), gate=True),
                         boolean_cell(values.get("bit_value_equal")), boolean_cell(check.get("passed"), gate=True)])
    if rows:
        sections.append("<h3>100 Millionen Proben: RAM- und dateibasierte Implementierung</h3>"
                        "<p>Jede Methode wird mit ihrer eigenen Umsetzung mit anderen Puffern verglichen. "
                        "Eingangs- und Rastergleichheit, numerische Prüfung sowie Bitwertgleichheit "
                        "werden getrennt ausgewiesen. Die numerische Prüfung übernimmt das "
                        "gespeicherte Ergebnis <code>gate_passed</code>. Diese Werte betreffen "
                        "100 Millionen Proben und sind kein punktweiser RAM-Nachweis für eine Milliarde.</p>" + table(
                            ["N", "Methode", "Ausgabe", "Eingang identisch", "Raster identisch", "Max. relativ [%]",
                             "Numerische Prüfung", "Bitwerte gleich", "Gesamtprüfung der Methode"], rows))
    sections.append("<p class='muted'>Weitere Prüffelder bleiben unverändert im vollständigen "
                    "Validierungs-JSON erhalten. Der Renderer führt diese Tests nicht erneut aus.</p>")
    sections.append(raw_section(validation, "Vollständiges Validierungsprotokoll einschließlich weiterer Prüffelder"))
    return "".join(sections)


def disk_html(report, output_dir, figures):
    if report is None:
        return ""
    rows, diagnostics = [], []
    histories = historical_jobs(report.raw)
    history_block = ("<h3>Frühere dateibasierte Versuche</h3>"
                     "<p>Diese Originaljobs sind historische Versuche und gehören nicht zur ausgewählten "
                     "Zeitreihe. Ein Timeout bezeichnet das Limit der Prozesshülle, nicht eine "
                     "abgeschlossene PSD-/NSD-Berechnungszeit. Auch Fortschritts- oder Phasenzeiten "
                     "werden nicht als Ersatz für eine fehlende Gesamtzeit verwendet. Ein RSS-Wert "
                     "aus dem letzten Zwischenstand kann unter dem später erreichten Prozessmaximum liegen.</p>"
                     + jobs_table(histories, disk=True)) if histories else ""
    for job in report.jobs:
        observed = job.get("timing", {}) if job.get("status") == "completed" else {}
        details = job.get("details", {})
        fft = details.get("fft", {})
        factors = fft.get("factors", fft.get("memory", {}).get("factors"))
        rows.append([h(integer(job["n"])), h(label(job.get("method"), disk=True)), status_cell(job),
                     h(number(observed.get("wall_s"), 8)), h(number(observed.get("cpu_s"), 8)),
                     rss_cell(job),
                     "ja" if job.get("input_unchanged") is True else "—"])
        if job.get("status") == "completed":
            io = job.get("io_delta", {})
            diagnostics.append([h(integer(job["n"])), h(label(job.get("method"), disk=True)),
                                h(" × ".join(map(str, factors))) if factors else "—",
                                h(number(io.get("read_bytes") / GIB, 7)) if finite(io.get("read_bytes")) else "—",
                                h(number(io.get("write_bytes") / GIB, 7)) if finite(io.get("write_bytes")) else "—",
                                h(number(details.get("parseval", {}).get("relative_error"), 7))])
    return ("<h2>Dateibasierte Puffer: eigenständiges Experiment</h2>"
            "<p>Jeder abgeschlossene Job erzeugt PSD und NSD gemeinsam in genau einem beobachteten "
            "vollständigen Aufruf. Es gibt hier weder einen Wiederholungsmedian noch ein Konfidenzintervall. "
            "Die gemeinsame PSD+NSD-Zeit wird nicht halbiert oder als getrennte PSD-/NSD-Zeit ausgegeben. "
            "Die FFTW-Verfahren verwenden eine andere exakte N-Punkt-Zerlegung mit dateibasierten Puffern; "
            "der mathematische Frequenzplan und die Spektralschätzer bleiben erhalten, ihre Rundung kann "
            "sich ändern. Der dateibasierte LPSD-Pfad berechnet seine ursprünglichen Segmentprojektionen.</p>"
            "<p>Die Zeit enthält das Anlegen und Löschen der Arbeitsdateien, Planung, Berechnung, tatsächliche "
            "Seiteneinlagerung und Fortschrittsprotokolle. Bei den FFTW-Verfahren liegen zusätzlich die "
            "begrenzten Zeit-/Frequenz-Energiediagnosen innerhalb des gemessenen Aufrufs. Eingangserzeugung, "
            "Hashbildung, Prozessstart, Importe und Bibliotheksladen sind ausgeschlossen. Für temporäre "
            "Arbeitsdateien wird kein abschließendes <code>fsync</code> erzwungen. Diese Messgrenze und "
            "Speicherstrategie unterscheiden sich von den RAM-Zeiten; daraus wird kein RAM-Reuse-Speedup abgeleitet.</p>" +
            source_cohorts_html(report) +
            embedded_figure(output_dir, figures.get("disk"), "Einzelne beobachtete gemeinsame PSD+NSD-Aufrufe. Nicht abgeschlossene Jobs liefern keinen Zeitwert.") +
            table(["N", "Methode", "Status", "Ein beobachteter Gesamtaufruf [s]", "CPU-Zeit [s]", "Prozess-RSS-Maximum [GiB]", "Eingang unverändert"], rows) +
            "<p class='note'>Das RSS-Maximum gilt für den gesamten Methodenprozess. Bei "
            "<code>--power-storage workspace</code> wird die Powerregion innerhalb der vorhandenen "
            "FFT-Datei zusätzlich gemappt. Das Prozess-RSS kann dadurch dieselben gemeinsam residenten "
            "Dateiseiten mehrfach zählen; es ist keine Messung der physischen cgroup-Speichernutzung. "
            "Für diese bleibt die protokollierte cgroup-RAM-Grenze maßgeblich. Im Workspace-Modus der "
            "FFTW-Pfade umfassen die großen Dateien einschließlich Eingang 24N Bytes, ihre virtuellen "
            "Ansichten zusammen 28N + 8 Bytes. Beide Größen sind von der gleichzeitig residenten "
            "physischen Belegung zu unterscheiden. Soweit im Job vorhanden, dokumentiert "
            "<code>details.power_storage</code> die Dateistrategie, maximale große Dateigröße "
            "und zugehörige Erläuterung.</p>" +
            "<h3>Dateipuffer: FFT-Zerlegung, I/O und Energiediagnose</h3>" +
            table(["N", "Methode", "FFT-Faktoren", "I/O read_bytes [GiB]", "I/O write_bytes [GiB]", "Relativer Parseval-Fehler"], diagnostics) +
            "<p class='muted'>Die I/O-Spalten sind die protokollierten Änderungen der Prozesszähler "
            "<code>read_bytes</code> und <code>write_bytes</code>. Sie sind weder Dateigrößen noch die "
            "Summe aller logischen Arrayzugriffe. Die Energiediagnose ist kein vollständiger "
            "punktweiser Vergleich mit einer anderen Spektralimplementierung.</p>" +
            jobs_table([job for job in report.jobs if job.get("status") != "completed"], disk=True) +
            history_block + source_runs_html(report) +
            "<h3>Dateibasierte PSD-/NSD-Spektren und Rauschstreuung</h3>" +
            embedded_figure(output_dir, figures.get("disk_spectrum"),
                            "100M und 1B besitzen jeweils eigene gemessene Spektren und eigene native Raster. Die Kurven stammen ausschließlich aus abgeschlossenen dateibasierten Jobs.") +
            noise_table(report) +
            "<p class='muted'>Die Quellnormierung verwendet die vorgegebene weiße NSD von 10 nV/√Hz. "
            "Die Streuung über korrelierte Frequenzpunkte einer Realisierung ist eine beschreibende "
            "Qualitätsgröße; die beiden Datenlängen sind keine Monte-Carlo-Wiederholungen. Eine "
            "vollständige dateibasierte Messung erzeugt PSD und NSD gemeinsam.</p>" +
            embedded_figure(output_dir, figures.get("disk_ratio"),
                            "Verhältnis zur tatsächlich gespeicherten dateibasierten LNSD derselben Datenlänge; nur bei gleichem Eingang und identischem Raster.") +
            disk_comparison_table(report) +
            "<p class='muted'>Die Vergleichsstatistiken werden direkt aus den gespeicherten Vektoren "
            "im Band 0,01–20 Hz berechnet, unter Ausschluss verschwindender LNSD-Referenzwerte. "
            "RMS bezeichnet √mean([100 × (NSD/LNSD − 1)]²), P95 das 95. Perzentil des absoluten "
            "prozentualen Unterschieds. Diese Zahlen vergleichen unterschiedliche Schätzer und sind "
            "keine Garantie zur Implementierungsgenauigkeit. Unterschiedliche Raster werden weder "
            "interpoliert noch zwischen Datenlängen vermischt.</p>" +
            raw_section(report, "Vollständiges separates Protokoll der dateibasierten Aufrufe"))


def write_html(report, disk, output_dir, figures, validation=None):
    histories = historical_jobs(report.raw)
    source = report.raw
    machine = source.get("machine", {})
    cpu = machine.get("cpu") or next((line.split(":", 1)[1].strip() for line in machine.get("cpu_info", "").splitlines() if line.startswith("model name")), "nicht protokolliert")
    memory = report.settings.get("memory_limit_bytes")
    guard = report.settings.get("address_space_guard_bytes")
    ram_release = source.get("lpsd_fast_version") or RECORDED_RELEASES.get(source.get("git_commit"))
    ram_revision_note = (
        f"<p class='note'><strong>RAM-Messstand: {h(ram_release)}.</strong> Diese Zeitreihe enthält "
        "die ursprünglichen RAM-Aufrufe. Eine später ausgewiesene dateibasierte Quellenkohorte "
        "ändert weder ihre gemessenen Zeiten noch ihre native Bibliothek. Der gemessene "
        "RAM-Quellcommit steht im eigenen Herkunftsabschnitt.</p>"
        if ram_release else ""
    )
    facts = {
        "RAM-Messprotokoll": report.path.name,
        "RAM-Protokollstatus": STATUS.get(source.get("status"), source.get("status", "nicht abgeschlossen")),
        "RAM-Beginn / Ende UTC": f"{source.get('started_utc', '—')} / {source.get('finished_utc', '—')}",
        "RAM-Messstand": ram_release or "nicht gesondert protokolliert",
        "Gemessener RAM-Quellcommit": source.get("git_commit", "—"), "CPU": cpu,
        "CPU-Kontingent": machine.get("cpu_max", "—"),
        "cgroup-RAM-Limit [GiB]": number(memory / GIB) if finite(memory) else "—",
        "RAM-Adressraumgrenze [GiB]": number(guard / GIB) if finite(guard) else "—",
        "Python / NumPy": f"{machine.get('python', '—')} / {machine.get('numpy', '—')}",
        "FFTW-Threads / LPSD-Worker / Glättungsworker": " / ".join(str(report.settings.get(key, "—")) for key in ("fftw_threads", "workers", "smoothing_workers")),
        "Hybridklasse: gemeinsames Cachelimit [MiB]": report.cache_mib,
        "Abtastrate [Hz]": number(report.settings.get("sample_rate")),
        "FFTW-Versionen": "; ".join(sorted({str(job.get("fftw", {}).get("version")) for job in report.completed if job.get("fftw", {}).get("version")})) or "—",
    }
    facts_html = "".join(f"<dt>{h(key)}</dt><dd>{h(value)}</dd>" for key, value in facts.items())
    validated = [job for job in report.completed if job.get("input_unchanged") and job.get("repeatability_and_output_checks_passed")]
    completed_count = len(report.completed)
    history_block = ("<h3>Frühere Versuche und Ressourcenfehler</h3><p>Diese separat erhaltenen Versuche "
                     "gehören nicht zur aktuellen Zeitreihe. Ein Speicherfehler unter der Adressraumgrenze "
                     "ist kein ausgeführter erfolgreicher Aufruf und liefert hier keinen Benchmarkwert. "
                     "Die Cachekonfiguration steht pro Versuch in der Tabelle.</p>" + jobs_table(histories)) if histories else ""
    document = f'''<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Große Datensätze · LPSD/LNSD und FFTW</title><style>
:root{{--ink:{INK};--muted:{MUTED};--blue:{BLUE};--line:#dce4ed;--wash:#f2f6fa}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:white;font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1360px;margin:auto;padding:42px 34px 55px}}header{{border-top:6px solid var(--blue);padding-top:18px}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}h1{{font-size:34px;line-height:1.18;letter-spacing:-.02em}}h2{{font-size:24px;line-height:1.25;border-top:1px solid var(--line);padding-top:20px;margin-top:34px}}h3{{font-size:18px;margin:23px 0 11px}}p{{max-width:1150px;margin:12px 0 16px}}.lead{{font-size:17px}}.muted,small,figcaption{{color:var(--muted)}}.good{{color:#25765A}}.problem{{color:#A54136}}.note{{background:var(--wash);border-left:3px solid {PURPLE};padding:14px 17px;margin:20px 0}}code{{font:.92em ui-monospace,monospace;background:var(--wash);padding:2px 4px;overflow-wrap:anywhere}}a{{color:var(--blue);text-underline-offset:2px}}.table-scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}th,td{{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:var(--wash);font-size:11.5px}}tr:nth-child(even) td{{background:#fafbfd}}td small{{display:block;font-size:10px;margin-top:4px}}figure{{margin:24px 0 30px;border:1px solid var(--line);padding:7px 9px 12px}}.chart svg{{display:block;width:100%;height:auto}}figcaption{{font-size:12px;margin:8px 11px 0}}dl{{display:grid;grid-template-columns:250px 1fr;gap:8px 20px;font-size:13px}}dt{{font-weight:650}}dd{{margin:0;overflow-wrap:anywhere}}details{{border:1px solid var(--line);padding:14px 16px;margin:20px 0}}summary{{cursor:pointer;font-weight:650}}pre{{font:11px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}}footer{{border-top:1px solid var(--line);margin-top:32px;padding-top:16px;font-size:12px;color:var(--muted)}}
@media(max-width:800px){{main{{padding:24px 15px}}h1{{font-size:28px}}h2{{font-size:22px}}figure{{margin-left:-7px;margin-right:-7px;padding:2px}}th,td{{padding:8px;font-size:11px}}dl{{grid-template-columns:150px 1fr}}}}
@media print{{@page{{size:A4;margin:12mm}}main{{padding:0;max-width:none}}body{{font-size:10pt}}h1{{font-size:23pt}}h2{{font-size:16pt;break-after:avoid}}h3{{break-after:avoid}}figure{{break-inside:avoid}}.table-scroll{{overflow:visible}}thead{{display:table-header-group}}tr{{break-inside:avoid}}details{{display:none}}}}
</style></head><body><main><header><div class="eyebrow">Isolierte Methodenprozesse · aufgezeichnete Messwerte</div><h1>Große Datensätze: LPSD/LNSD und FFTW</h1>
<p class="lead">Gemessene Berechnungszeiten, weiße Rauschspektren und Speichergrenzen für die vorhandenen RAM-Pfade. Optionale dateibasierte Versuche besitzen einen eigenen Messumfang und werden getrennt ausgewiesen.</p>
<p class="note"><strong>Aktuelles Hybridcachelimit: {h(report.cache_mib)} MiB.</strong> Es betrifft die vorbereitbare Hybridklasse und die Nutzdaten ihrer beiden dauerhaften Caches. Die Vorstudie verwendete 2048 MiB. Der optimierte Hybrid-Einzelaufruf deaktiviert diese Caches. Die 10-Millionen-Messung dient als Brücke innerhalb des neuen Profils; historische Zeiten werden nicht zu einer gemeinsamen Reihe vermischt.</p></header>
<h2>1 · Vollständige RAM-Aufrufe</h2>{ram_revision_note}<p>Methoden und Datenlängen laufen seriell in eigenen Prozessen. Innerhalb einer Wiederholungsrunde wird die Reihenfolge von PSD und NSD gemischt. „Vorbereitet“ enthält den vollständigen datenabhängigen Aufruf, jedoch keinen Aufbau oder Abbau. „Komplett neu“ enthält das Verwerfen der FFTW-Wisdom, Planung mit ESTIMATE, Fenster und konfigurierte Caches, Berechnung und Freigabe. LPSD/LNSD wird als vollständiger öffentlicher <code>kernel="fast"</code>-Aufruf gemessen.</p>
<p>Signalerzeugung, Hashbildung, unabhängige Prüfungen, Prozessstart, Importe, Bibliotheksladen und Plotten sind ausgeschlossen. Die Eingangsprüfung im regulären Aufruf bleibt enthalten. Betriebssystemcaches werden nicht geleert. Mediane, Rohpunkte und Minimum–Maximum beschreiben die beobachteten Wiederholungen; die Spannen sind keine Konfidenzintervalle. PSD und NSD teilen nahezu die gesamte Rechnung, sodass unterschiedliche Mediane allein keinen grundsätzlichen Geschwindigkeitsunterschied belegen.</p>
{embedded_figure(output_dir, figures.get('runtime_psd'), 'PSD: nur abgeschlossene Jobs. Unterschiedliche Messumfänge stehen in getrennten Panels.')}
{embedded_figure(output_dir, figures.get('runtime_nsd'), 'NSD: vollständige datenabhängige beziehungsweise vollständige neue Aufrufe, keine reinen FFT-Kernzeiten.')}
{timing_table(report)}
<h3>Status, Vorbereitung und Prozessspeicher</h3><p>Ein Aufbauwert ist eine separat beobachtete Konstruktorzeit, kein Median. Das RSS-Maximum gilt für den gesamten isolierten Methodenprozess einschließlich Vorbereitung, Aufwärmen, Berechnung und Validierung; es ist kein pro Aufruf gemessenes Maximum. RSS und virtueller Adressraum sind verschiedene Größen. Nur <code>status=completed</code> liefert Zeitwerte in den Diagrammen und der obigen Tabelle.</p>
{jobs_table(report.jobs)}{history_block}
<h3>Tatsächliche Cachebelegung</h3>{cache_table(report)}
<p class="muted">Das gemeinsame Cachelimit umfasst gespeicherte Kaiser-Gewichte und projizierte LPSD-Koeffizienten. Eingang, FFTW-Pläne, Fenster, temporärer Arbeitsraum und Interpreter kommen hinzu. Das Limit ist keine RSS-Garantie; nicht gespeicherte Kernel behalten ihren vollständigen Stützbereich.</p>
<h2>2 · Gemessene Rauschspektren</h2>{embedded_figure(output_dir, figures.get('spectrum'), '100 Millionen Proben. Dargestellt werden ausschließlich die gespeicherten Spektralwerte der abgeschlossenen Methodenjobs.')}
<p>Die Eingangsdaten sind weißes Gaußrauschen mit einseitiger Quell-NSD 10 nV/√Hz. Das Quellmodell ist kein exakter Erwartungswert des endlichen, gefensterten und detrendeten Schätzers. PSD hat die Einheit V²/Hz, NSD V/√Hz. Intern bleiben Eingang, FFT und Leistungsrechnung Float64; die öffentliche PSD ist Float32 und die NSD folgt der erhaltenen Quadratwurzel dieser gerundeten PSD.</p>
<p>Die neuen Großdatenversuche bei 100 Millionen und einer Milliarde Proben verwenden ausschließlich weißes Rauschen. Neue Ton- oder Peakbreitenmessungen gehören nicht zu diesen Versuchen; die früheren Tonprüfungen betreffen gesonderte Datensätze. Vergleiche werden jeweils innerhalb der RAM- beziehungsweise der dateibasierten Verfahren gelesen.</p>
<p>Einfaches FFTW gruppiert Leistungen eines vollständigen Kaiser-Periodogramms. Der Hybrid verwendet das unveränderte periodische Tukey-Vollfenster, Kaiser-Leistungsgewichte im Frequenzbereich und einen expliziten nativen LPSD-Teilplan. Diese unterschiedlichen Schätzer verlangen keine identischen Rauschzacken. Die geerbte LPSD-Segmentrekurrenz bleibt erhalten: bei mindestens zwei Segmenten entfällt in exakter Arithmetik der Beitrag des ersten Segments durch den Divisor <code>ii</code> statt <code>ii+1</code>. <code>psd_std</code> wird nicht als validierte Unsicherheit verwendet.</p>
{noise_table(report)}<p class="muted">Die normierte Streuung ist die Stichproben-Standardabweichung von NSD/Quell-NSD über die ausgewählten Frequenzpunkte einer Realisierung. Die Differenz-RMS benachbarter Punkte ist eine separate Feinheitskennzahl. Diese Frequenzwerte sind korreliert; sie bilden keine unabhängige Monte-Carlo-Serie und kein Konfidenzintervall.</p>
<p>Für {len(validated)} von {completed_count} abgeschlossenen RAM-Jobs dokumentiert das Protokoll unveränderte Eingänge sowie wiederholbare öffentliche Ausgaben. Gesonderte Prüfungen von Eingangs-Hashes, Frequenzrastern und Energie stehen vollständig im eingebetteten JSON. Instrumentierte Phasenprofile sind eigene Aufrufe und ersetzen keine Timingmediane.</p>
<h2>3 · Eine Milliarde Proben: bestehende RAM-Pfade</h2>{embedded_figure(output_dir, figures.get('memory'), 'Analytische Untergrenzen der vorhandenen RAM-Puffer, keine extrapolierten Laufzeiten. Optionale dateibasierte Verfahren werden gesondert bewertet.')}
{memory_table(report)}<p>Die Speicheruntergrenze bezieht sich auf bekannte gleichzeitig benötigte Arrays einschließlich Eingang. Zusätzliche Bibliotheks- und Arbeitsbereiche können den Bedarf erhöhen. Diese bekannten Arrays werden durch ein kleineres Cachelimit nicht kleiner; die ursprünglichen Parameter einer Ressourcenprüfung bleiben im Rohprotokoll erhalten. Ein vorhersehbar nicht ausführbarer RAM-Aufruf erhält keinen Zeitwert; ebenso wenig wird die 100-Millionen-Zeit auf eine Milliarde hochgerechnet.</p>
{disk_html(disk, output_dir, figures)}
{validation_html(validation)}
<h2>Herkunft der RAM-Messung und vollständige Rohwerte</h2><dl>{facts_html}</dl>
{source_runs_html(report)}
<p>Der Renderer liest gespeicherte Messdaten und berechnet keine neuen Spektren oder Laufzeiten. Quelldatei-Hashes, Native-Bibliotheken, einzelne Wiederholungen, Ressourcenstatus und alle vorhandenen Spektralvektoren bleiben im folgenden Protokoll enthalten.</p>
{raw_section(report, 'Vollständiges RAM-Messprotokoll einschließlich historischer Versuche')}
<footer>Alle Abbildungen sind als SVG in dieser Datei eingebettet und zusätzlich als PNG/SVG verfügbar. Die JSON-Downloads sind ebenfalls eingebettet; der Bericht benötigt keine Netzwerkverbindung.</footer></main></body></html>'''
    target = output_dir / "report_fftw_large_n.html"
    target.write_text(document, encoding="utf-8")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="Konsolidiertes lpsd-fftw-large-n-v1-JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separater Ausgabeordner für PNG/SVG/HTML")
    parser.add_argument("--disk-results", type=Path, help="Optionales getrenntes lpsd-fftw-disk-v1-JSON")
    parser.add_argument("--validation", type=Path, help="Optionales Validierungs-JSON mit unit_tests und ram_vs_disk_100m")
    args = parser.parse_args(argv)
    report = load_report(args.results, "lpsd-fftw-large-n-v1")
    disk = load_report(args.disk_results, "lpsd-fftw-disk-v1") if args.disk_results else None
    validation = load_validation(args.validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    figures = {}
    for density in ("psd", "nsd"):
        figures[f"runtime_{density}"] = runtime_figure(report, args.output_dir, density)
    figures["memory"] = memory_figure(report, args.output_dir, disk)
    figures["spectrum"] = spectrum_figure(report, args.output_dir)
    figures["disk"] = disk_figure(disk, args.output_dir)
    figures["disk_spectrum"] = disk_spectrum_figure(disk, args.output_dir)
    figures["disk_ratio"] = disk_ratio_figure(disk, args.output_dir)
    report_path = write_html(report, disk, args.output_dir, figures, validation)
    print(json.dumps({"report": str(report_path.resolve()),
                      "figures": {key: value for key, value in figures.items() if value}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
