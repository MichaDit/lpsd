"""Replay the rejected PSD-statistics screen; argument 1 holds baseline.so/candidate.so.

Run with PYTHONPATH pointing to the pinned baseline checkout. New results go
into that artifact directory; the archived original observations are unchanged.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import lpsd_fast
from lpsd_fast import api
ROOT=Path(__import__('sys').argv[1]).resolve()
api._native()
original=api._LIB
libraries={name:ct.CDLL(str(ROOT/(name+'.so'))) for name in ('baseline','candidate')}
for name,fn in original.__dict__.items():
    if isinstance(fn,ct._CFuncPtr):
        for library in libraries.values():
            getattr(library,name).argtypes=fn.argtypes
            getattr(library,name).restype=fn.restype
rng=np.random.default_rng(20261021)
report={'schema':'native-psd-statistics-api-v1','runs':[]}
for n in (1000000,):
    x=rng.standard_normal(n)
    for workers in (1,4):
        for outputs in ('psd',None):
            values={}
            options=dict(sample_rate=1.0,n_frequencies=1000,n_averages=100,kernel='auto',workers=workers,statistics='corrected',max_working_mb=4096)
            if outputs is not None:options['outputs']=outputs
            def call(kind):
                api._LIB=libraries[kind]
                begin=time.perf_counter()
                result=lpsd_fast.lpsd(x,**options)
                elapsed=time.perf_counter()-begin
                return elapsed,result
            for kind in libraries:call(kind)
            times={kind:[] for kind in libraries}
            hashes={}
            for pair in range(6):
                for kind in (('baseline','candidate') if pair%2==0 else ('candidate','baseline')):
                    elapsed,result=call(kind)
                    times[kind].append(elapsed)
                    values[kind]=result
                    hashes[kind]=hashlib.sha256(result.to_numpy().tobytes()).hexdigest()
                np.testing.assert_array_equal(values['baseline'].to_numpy(),values['candidate'].to_numpy())
            row=dict(n=n,workers=workers,outputs=outputs,times=times,hashes=hashes,ratio=statistics.median(times['baseline'])/statistics.median(times['candidate']))
            report['runs'].append(row)
            (ROOT/'api-screen-replay.json').write_text(json.dumps(report,indent=2)+'\n')
            print(n,workers,outputs,row['ratio'],flush=True)
api._LIB=original
