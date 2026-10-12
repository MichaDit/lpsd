"""Explicit comparison metrics; no smoothing or fitting of measured spectra."""
import numpy as np

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

BANDS = [(0.01, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 20.0)]


def summarize_noise(frequencies, nsd, source_psd, exclude=None):
    f = np.asarray(frequencies, float)
    a = np.asarray(nsd, float)
    target = np.sqrt(np.asarray(source_psd, float))
    allowed = np.isfinite(a) & (target > 0)
    if exclude is not None:
        allowed &= ~np.asarray(exclude, bool)

    def one(lo, hi):
        keep = allowed & (f >= lo) & (f < hi if hi < 20 else f <= hi)
        ratio = a[keep] / target[keep]
        if len(ratio) < 2:
            return {"low_hz": lo, "high_hz": hi, "points": int(len(ratio))}
        adjacent = keep[:-1] & keep[1:]
        all_ratio = a / target
        rough = np.diff(all_ratio)[adjacent]
        return {"low_hz": lo, "high_hz": hi, "points": int(len(ratio)),
                "mean_nsd_over_source": float(np.mean(ratio)),
                "median_nsd_over_source": float(np.median(ratio)),
                "normalized_nsd_std": float(np.std(ratio, ddof=1)),
                "relative_std": float(np.std(ratio, ddof=1)/np.mean(ratio)),
                "p10_nsd_over_source": float(np.quantile(ratio, .1)),
                "p90_nsd_over_source": float(np.quantile(ratio, .9)),
                "adjacent_difference_rms_over_sqrt2": float(np.sqrt(np.mean(rough*rough)/2)) if len(rough) else None}
    return {"overall": one(.01, 20), "bands": [one(*b) for b in BANDS],
            "definition": "Sample standard deviation of NSD/sqrt(source PSD) across the named frequency band in one realization. Correlated frequency samples: descriptive waviness, not a confidence interval. Adjacent difference RMS is a separate fine-scale roughness measure."}


def width_at_fraction(f, y, fraction):
    f, y = np.asarray(f, float), np.asarray(y, float)
    p = int(np.argmax(y))
    h = y[p] * fraction
    left = next((k for k in range(p-1, -1, -1) if y[k] <= h), None)
    right = next((k for k in range(p+1, len(y)) if y[k] <= h), None)
    if left is None or right is None:
        return None
    def crossing(i, j):
        return float(f[i]+(h-y[i])*(f[j]-f[i])/(y[j]-y[i]))
    l, r = crossing(left, left+1), crossing(right-1, right)
    return {"width_hz": r-l, "left_hz": l, "right_hz": r,
            "left_grid_interval_hz": float(f[left+1]-f[left]),
            "right_grid_interval_hz": float(f[right]-f[right-1])}


def tone_metrics(frequencies, psd, tone_hz, rms_v, half_span=.6):
    f, p = np.asarray(frequencies, float), np.asarray(psd, float)
    keep = np.abs(f-tone_hz) <= half_span
    f, p = f[keep], p[keep]
    a = np.sqrt(p)
    w_p, w_a = width_at_fraction(f,p,.5), width_at_fraction(f,a,.5)
    return {"center_hz": float(tone_hz), "peak_psd": float(p.max()), "peak_nsd": float(a.max()),
            "peak_label_hz": float(f[np.argmax(p)]),
            "psd_fwhm_hz": w_p["width_hz"] if w_p else None,
            "nsd_fwhm_hz": w_a["width_hz"] if w_a else None,
            "psd_half_max_crossings": w_p, "nsd_half_max_crossings": w_a,
            "output_grid_line_power_v2": float(_trapezoid(p,f)),
            "output_grid_line_power_over_source": float(_trapezoid(p,f)/rms_v**2),
            "definition": "FWHM at half the native sampled PSD maximum, or half the native sampled NSD maximum; linear interpolation only between adjacent native points. This is a grid-dependent width estimate. Integrated line power is a trapezoid estimate within the stated span, not a Parseval identity.",
            "half_span_hz": half_span}


def compare_to_reference(frequencies, psd, reference, enbw, tones=()):
    f, p, r = np.asarray(frequencies), np.asarray(psd,float), np.asarray(reference,float)
    keep = (f >= .01) & (f <=20) & (r>0)
    for tone in tones:
        keep &= np.abs(f-tone["frequency_hz"]) > 3*np.asarray(enbw)
    rel=np.sqrt(p[keep]/r[keep])-1
    return {"points":int(keep.sum()), "median_abs_relative_nsd_difference":float(np.median(np.abs(rel))) if len(rel) else None,
            "p95_abs_relative_nsd_difference":float(np.quantile(np.abs(rel),.95)) if len(rel) else None,
            "rms_relative_nsd_difference":float(np.sqrt(np.mean(rel**2))) if len(rel) else None,
            "note":"Pointwise difference on the same realization; estimator difference, not floating-point error."}
