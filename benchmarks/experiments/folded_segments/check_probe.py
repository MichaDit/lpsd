"""Numerical screening only; no performance claims or timing assertions."""
from pathlib import Path
import ctypes as ct
import argparse
import json
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parent
DP = ct.POINTER(ct.c_double)
ARGS = [DP, DP, DP, DP, ct.POINTER(ct.c_long), DP, DP,
        ct.c_long, ct.c_long, DP, DP, ct.c_double, ct.c_int, ct.c_bool, ct.c_int]


def library(path):
    lib = ct.CDLL(str(path))
    lib.fast_dft_selected.argtypes = ARGS + [ct.c_bool, ct.c_bool]
    lib.fast_dft_selected.restype = ct.c_int
    lib.generate_window.argtypes = [DP, ct.c_long, ct.c_int, ct.c_double]
    lib.generate_window.restype = ct.c_int
    lib.generate_coefficients_blocked.argtypes = [DP, DP, DP, ct.c_long, ct.c_double]
    lib.generate_coefficients_blocked.restype = ct.c_int
    if hasattr(lib, 'probe_folded_dft'):
        lib.probe_folded_dft.argtypes = ARGS + [ct.c_bool, DP, ct.c_bool,
                                              ct.c_double, ct.c_int, ct.POINTER(ct.c_long)]
        lib.probe_folded_dft.restype = ct.c_int
    return lib


def pointer(x):
    return x.ctypes.data_as(DP)


def case(lib, length=4097, segments=33, frequency_bin=4.371, signal='white',
         window_kind=0, beta=23.7):
    count = length + round((segments - 1) * (length / 4 + .123))
    rng = np.random.default_rng(90201 + length)
    x = rng.standard_normal(count)
    phase = np.arange(count) * (2*np.pi*frequency_bin/length)
    if signal == 'dc': x = 10. + 1e-9 * x
    elif signal == 'dc_ulp': x = 10. + 16*np.spacing(10.) * x
    elif signal == 'constant': x.fill(10.)
    elif signal == 'zero': x.fill(0.)
    elif signal == 'ramp': x = np.linspace(-1e6, 1e6, count) + 1e-4*x
    elif signal == 'tone': x = np.sin(phase + .47)
    elif signal == 'offtone': x = np.sin(np.arange(count) * (2*np.pi*.817391/length))
    elif signal == 'weak_tone': x = np.sin(phase + .47) + 1e-9*np.sin(7.313*phase)
    elif signal == 'alternating': x = np.where(np.arange(count)%2, 1., -1.)
    elif signal == 'transient': x[1] += 1e12
    elif signal == 'departed': x.fill(0.); x[0] = 1e12
    elif signal == 'huge': x *= 1e155
    elif signal == 'tiny': x *= 1e-165
    w = np.empty(length)
    assert lib.generate_window(pointer(w), length, window_kind, beta) == 0
    cr, ci = np.empty(length), np.empty(length)
    assert lib.generate_coefficients_blocked(pointer(cr), pointer(ci), pointer(w), length, frequency_bin) == 0
    return dict(x=x, cr=cr, ci=ci, w=w, frequency_bin=frequency_bin,
                periodic=window_kind==0, label=f'L{length}-K{segments}-m{frequency_bin}-w{window_kind}-b{beta}-{signal}')


def evaluate(lib, item, *, folded=False, width=8, statistics=True):
    values = [ct.c_double() for _ in range(4)]
    count, folded_count = ct.c_long(), ct.c_long()
    x, cr, ci = item['x'], item['cr'], item['ci']
    args = [*(ct.byref(value) for value in values), ct.byref(count),
            pointer(x), None, len(x), len(cr), pointer(cr), pointer(ci),
            75., 0, False, 2]
    if folded:
        result = lib.probe_folded_dft(*args, statistics, pointer(item['w']), item['periodic'],
                                      item['frequency_bin'], width, ct.byref(folded_count))
    else:
        result = lib.fast_dft_selected(*args, statistics, False)
    assert result == 0, (result, item['label'])
    return np.array([value.value for value in values]), count.value, folded_count.value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefficient-reuse', action='store_true')
    args = parser.parse_args()
    prefix = 'libfolded_reuse_' if args.coefficient_reuse else 'libfolded_'
    plain = library(ROOT / (prefix + 'plain.so'))
    fused = library(ROOT / (prefix + 'fma.so'))
    rows = []
    variants = [(0, 0.), (0, 1.), (0, 23.7), (0, 32.), (0, 50.),
                (1, 0.), (2, 0.), (3, 0.)]
    configurations = []
    for length in (129, 256, 1025, 4096, 4097):
        for window_kind, beta in variants:
            for signal in ('white', 'dc', 'tone', 'offtone', 'constant', 'transient'):
                configurations.append(dict(length=length, window_kind=window_kind, beta=beta, signal=signal))
    for window_kind, beta in variants:
        for signal in ('dc_ulp', 'ramp', 'alternating', 'departed', 'huge', 'tiny'):
            configurations.append(dict(length=4097, window_kind=window_kind, beta=beta, signal=signal))
        for m in (0., .5, 1., 1304., 10000.):
            configurations.append(dict(length=4097, window_kind=window_kind, beta=beta, signal='offtone', frequency_bin=m))
    for length in (3, 4, 5, 128, 255, 1024, 2048, 65537):
        for kind in (0, 1):
            configurations.append(dict(length=length, segments=1 if length==3 else 17, window_kind=kind, signal='white'))
    for config in configurations:
        item = case(plain, **config)
        expected, count, _ = evaluate(plain, item)
        normalization = 2. / (50. * np.square(item['w']).sum())
        for name, lib in (('plain', plain), ('fma', fused)):
            actual, actual_count, folded_count = evaluate(lib, item, folded=True)
            assert actual_count == count
            classes = bool(np.array_equal(np.isnan(expected), np.isnan(actual))
                           and np.array_equal(np.isposinf(expected), np.isposinf(actual))
                           and np.array_equal(np.isneginf(expected), np.isneginf(actual)))
            finite = np.isfinite(expected) & np.isfinite(actual)
            absolute = float(abs(expected[0] - actual[0])*normalization) if finite[0] else None
            relative = float(abs((expected[0] - actual[0])/expected[0])) if finite[0] and expected[0] else None
            with np.errstate(over='ignore', invalid='ignore'):
                e32 = np.float32(expected[0]*normalization)
                a32 = np.float32(actual[0]*normalization)
                ea, aa = np.sqrt(e32), np.sqrt(a32)
            nsd_absolute = float(abs(float(ea) - float(aa))) if np.isfinite(ea) and np.isfinite(aa) else None
            nsd_relative = float(abs((float(ea) - float(aa))/float(ea))) if nsd_absolute is not None and ea else None
            passed = classes and (not finite[0] or (relative is not None and relative < .01) or absolute <= 1e-24)
            nsd_passed = nsd_absolute is None or (nsd_relative is not None and nsd_relative < .01) or nsd_absolute <= 1e-12
            rows.append(dict(label=item['label'], implementation=name, count=count,
                             check_kind='numeric_tolerance' if finite[0] else 'exceptional_semantics',
                             input_sha256=hashlib.sha256(item['x'].tobytes()).hexdigest(),
                             folded_count=folded_count, classes_match=classes,
                             psd_abs=absolute, psd_rel=relative, nsd_abs=nsd_absolute,
                             nsd_rel=nsd_relative, combined_pass=bool(passed and nsd_passed)))
    summary = dict(cases=len(configurations), comparisons=len(rows),
                   numeric_comparisons=sum(r['check_kind']=='numeric_tolerance' for r in rows),
                   exceptional_comparisons=sum(r['check_kind']=='exceptional_semantics' for r in rows),
                   passes=sum(row['combined_pass'] for row in rows),
                   class_failures=[row for row in rows if not row['classes_match']],
                   failures=[row for row in rows if not row['combined_pass']],
                   worst_relative=sorted((r for r in rows if r['psd_rel'] is not None),
                                          key=lambda r:r['psd_rel'], reverse=True)[:12])
    provenance = dict(prototype_sha256=hashlib.sha256((ROOT/'folded_probe.c').read_bytes()).hexdigest(),
                      checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      baseline_commit='9f3071ca2b7134a585f1727506a4e1472a7da299',
                      binary_hashes={Path(lib._name).name:hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()
                                     for lib in (plain,fused)})
    (ROOT / ('numerical-smoke-reuse.json' if args.coefficient_reuse else 'numerical-smoke.json')).write_text(json.dumps(dict(provenance=provenance, summary=summary, rows=rows), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
