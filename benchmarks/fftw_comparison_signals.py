"""Fixed, reproducible sources shared by the FFTW comparison reports.

SPDX-License-Identifier: GPL-3.0-or-later
"""
import numpy as np
from scipy.fft import irfft, rfft, rfftfreq


CASES = ("white", "pink", "brown", "mixed_tones", "offbin_tone", "dc_nanovolt")
TITLES = {
    "white": "Weißes Rauschen",
    "pink": "Rosa Rauschen · PSD ∝ 1/f",
    "brown": "Braunes Spektrum · PSD ∝ 1/f²",
    "mixed_tones": "Rauschmischung + drei Sinuslinien",
    "offbin_tone": "Reiner Sinus zwischen FFT-Bins",
    "dc_nanovolt": "10 V Gleichspannung + 1 nV RMS Rauschen",
}
DESCRIPTIONS = {
    "white": "Gaußrauschen mit theoretischer einseitiger NSD 10 nV/√Hz.",
    "pink": "Gaußsche Fourier-Synthese: S(f) = (10 nV/√Hz)² · 0,1 Hz/f; DC = 0; unterste Frequenz fs/N.",
    "brown": "Gaußsche Fourier-Synthese: S(f) = (10 nV/√Hz)² · (0,01 Hz/f)²; DC = 0, endlicher Datensatz. Keine unbeschränkte Zufallswanderung.",
    "mixed_tones": "Unabhängige weiße, rosa und braune Anteile wie oben; Sinuslinien bei 0,12345 / 3,123456 / 12,34567 Hz mit 30 / 50 / 20 nV RMS. Sollkurve zeigt nur das Rauschen.",
    "offbin_tone": "Reiner Sinus mit 1 µV RMS; sein vollständiger FFT-Bin-Index besitzt den Bruchteil 0,37.",
    "dc_nanovolt": "10 V + weißes Gaußrauschen mit 1 nV Zeitbereichs-RMS, als float64 dargestellt. Theoretische NSD bei fs=50 Hz: 0,2 nV/√Hz.",
}


def signal(name, n, fs, seed):
    """Fixed absolute source PSD, no per-realization variance normalization."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, CASES.index(name), n]))
    a = 10e-9
    sigma = a * np.sqrt(fs / 2)
    tones = []

    def colored(alpha, corner):
        z = rng.standard_normal(n) * sigma
        transform = rfft(z, workers=1)
        f = rfftfreq(n, 1 / fs)
        transform[0] = 0
        transform[1:] *= (corner / f[1:]) ** (alpha / 2)
        return irfft(transform, n=n, workers=1)

    if name == "white":
        x = rng.standard_normal(n) * sigma
    elif name == "pink":
        x = colored(1, .1)
    elif name == "brown":
        x = colored(2, .01)
    elif name == "mixed_tones":
        x = rng.standard_normal(n) * sigma
        x += colored(1, .1)
        x += colored(2, .01)
        t = np.arange(n, dtype=np.float64) / fs
        for f, rms, phase in ((.12345, 30e-9, .2), (3.123456, 50e-9, .7), (12.34567, 20e-9, 1.1)):
            x += np.sqrt(2) * rms * np.sin(2 * np.pi * f * t + phase)
            tones.append({"frequency_hz": f, "rms_v": rms, "phase_rad": phase})
    elif name == "offbin_tone":
        record_bin = np.floor(.06246 * n) + .37
        f = record_bin * fs / n
        x = np.sqrt(2) * 1e-6 * np.sin(2 * np.pi * record_bin * np.arange(n, dtype=np.float64) / n + .3)
        tones.append({"frequency_hz": float(f), "rms_v": 1e-6, "phase_rad": .3, "full_record_bin": float(record_bin)})
    elif name == "dc_nanovolt":
        x = 10.0 + 1e-9 * rng.standard_normal(n)
    else:
        raise ValueError(name)
    return np.ascontiguousarray(x, dtype=np.float64), tones


def theory(name, f, fs):
    base = 1e-16
    if name == "white":
        return np.full_like(f, base)
    if name == "pink":
        return base * (.1 / f)
    if name == "brown":
        return base * (.01 / f) ** 2
    if name == "mixed_tones":
        return base * (1 + .1 / f + (.01 / f) ** 2)
    if name == "dc_nanovolt":
        return np.full_like(f, 2e-18 / fs)
    return None
