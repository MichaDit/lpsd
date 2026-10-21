"""Replay the rejected PSD-statistics screen; argument 1 holds baseline.so/candidate.so.

Run with PYTHONPATH pointing to the pinned baseline checkout. New results go
into that artifact directory; the archived original observations are unchanged.
"""
import ctypes as ct
import json
import math
from pathlib import Path
import statistics as statistics_module
import time
import numpy as np
from lpsd_fast import api

ROOT = Path(__import__('sys').argv[1]).resolve()
api._native()
original = api._LIB
libraries = {}
for kind in ('baseline', 'candidate'):
    library = ct.CDLL(str(ROOT/(kind+'.so')))
    for name in ('fast_dft_selected_corrected', 'fast_dft_selected', 'fast_dft_selected_profile'):
        getattr(library, name).argtypes = getattr(original, name).argtypes
        getattr(library, name).restype = ct.c_int
    libraries[kind] = library

def calculate(kind, samples, length, statistics=True, mode=2, legacy=False, batched=True):
    n = len(samples)
    phase = np.arange(length)*(2*np.pi*7.371/length)
    window = np.kaiser(length+1,23.7)[:-1]
    real = window*np.cos(phase)
    imag = window*np.sin(phase)
    results = [ct.c_double() for _ in range(4)]
    count = ct.c_long()
    arguments = (*(ct.byref(v) for v in results), ct.byref(count),
                 api._pointer(samples), None, n, length,
                 api._pointer(real), api._pointer(imag), 75.0, 0, False, mode,
                 statistics, batched)
    if legacy:
        function = libraries[kind].fast_dft_selected
    else:
        function = libraries[kind].fast_dft_selected_corrected
        arguments += (math.inf, None, None, None)
    start = time.perf_counter()
    status = function(*arguments)
    elapsed = time.perf_counter() - start
    assert status == 0
    return elapsed, np.array([v.value for v in results]), count.value

rng = np.random.default_rng(20261021)
base = rng.standard_normal(2_000_000)
report = {'schema':'native-psd-statistics-screen-v1', 'timings':[], 'checks':[]}
for length in (16, 64, 128, 129, 255, 256, 511, 1024, 2048, 4096, 16384):
    row={'length':length, 'samples':len(base), 'pairs':[]}
    for statistics in (False, True):
        for kind in libraries:
            calculate(kind,base,length,statistics)
        series={kind:[] for kind in libraries}
        for pair in range(6):
            values={}
            for kind in (('baseline','candidate') if pair%2==0 else ('candidate','baseline')):
                elapsed,result,count=calculate(kind,base,length,statistics)
                values[kind]=(result,count)
                series[kind].append(elapsed)
            a,b=values['baseline'],values['candidate']
            assert a[1]==b[1]
            np.testing.assert_array_equal(a[0],b[0])
        row['pairs'].append({'statistics':statistics, 'times':series,
                            'ratio':statistics_module.median(series['baseline'])/statistics_module.median(series['candidate'])})
    report['timings'].append(row)
    print(length,[(x['statistics'],round(x['ratio'],4)) for x in row['pairs']],flush=True)

signals={
 'white':rng.standard_normal(4097),
 'dc_noise':10+1e-9*rng.standard_normal(4097),
 'transient':np.r_[1e150,rng.standard_normal(4096)],
 'tiny':1e-155*rng.standard_normal(4097),
 'overflow':1e155*rng.standard_normal(4097),
 'huge':1e308*rng.standard_normal(4097),
 'zero':np.zeros(4097),
 'constant':np.full(4097,10.0),
 'nan':np.r_[np.nan,rng.standard_normal(4096)],
 'inf':np.r_[np.inf,rng.standard_normal(4096)]}
for name,samples in signals.items():
    for length in (16,129,256,2048):
        for mode in (0,1,2,3):
            for legacy in (False,True):
                for statistics in (False,True):
                    a=calculate('baseline',samples,length,statistics,mode,legacy)
                    b=calculate('candidate',samples,length,statistics,mode,legacy)
                    assert a[2]==b[2]
                    np.testing.assert_array_equal(a[1],b[1])
                    report['checks'].append({'signal':name,'length':length,'mode':mode,'legacy':legacy,'statistics':statistics,'status':'exact'})
(ROOT/'native-screen-replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('Checks',len(report['checks']),flush=True)
