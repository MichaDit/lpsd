"""Serial native-frequency microbenchmark; window/phase generation excluded."""
from pathlib import Path
import ctypes as ct
import argparse
import hashlib
import json
import statistics
import time
import numpy as np
from check_probe import ROOT, DP, ARGS, pointer, library, case

parser=argparse.ArgumentParser()
parser.add_argument('--coefficient-reuse',action='store_true')
parser.add_argument('--baseline-library',type=Path,required=True)
parser.add_argument('--fma-library',type=Path,required=True)
args=parser.parse_args()
prefix='libfolded_reuse_' if args.coefficient_reuse else 'libfolded_'
plain=library(ROOT/(prefix+'plain.so'))
fused=library(ROOT/(prefix+'fma.so'))
old=library(args.baseline_library.resolve())
target=library(args.fma_library.resolve())
for lib in (plain,fused):
    lib.probe_folded_dft_profile.argtypes=lib.probe_folded_dft.argtypes+[DP,DP]
    lib.probe_folded_dft_profile.restype=ct.c_int
for lib in (old,plain,target):
    lib.fast_dft_selected_profile.argtypes=ARGS+[ct.c_bool,ct.c_bool,DP,DP]
    lib.fast_dft_selected_profile.restype=ct.c_int
target.fast_dft_selected_bounded.argtypes=ARGS+[ct.c_bool,ct.c_bool,ct.c_double,DP,DP,ct.POINTER(ct.c_long)]
target.fast_dft_selected_bounded.restype=ct.c_int


def prepare(lib,item,method,width=8,profile=False):
    results=[ct.c_double() for _ in range(4)]
    count,used=ct.c_long(),ct.c_long()
    prep,seg=ct.c_double(),ct.c_double()
    x,cr,ci=item['x'],item['cr'],item['ci']
    args=(*(ct.byref(v) for v in results),ct.byref(count),pointer(x),None,len(x),len(cr),
          pointer(cr),pointer(ci),75.,0,False,2)
    if method=='folded':
        func=lib.probe_folded_dft_profile if profile else lib.probe_folded_dft
        args+= (False,pointer(item['w']),item['periodic'],item['frequency_bin'],width,ct.byref(used))
        if profile: args+=(ct.byref(prep),ct.byref(seg))
    elif method=='bounded':
        func=lib.fast_dft_selected_bounded
        args+=(False,False,float(np.max(np.abs(x))),ct.byref(prep) if profile else None,
               ct.byref(seg) if profile else None,ct.byref(used))
    else:
        func=lib.fast_dft_selected_profile if profile else lib.fast_dft_selected
        args+=(False,False)
        if profile: args+=(ct.byref(prep),ct.byref(seg))
    def call():
        status=func(*args)
        if status: raise RuntimeError(status)
    return call,results,count,used,prep,seg

configs=[(128,129),(256,129),(512,65),(1024,65),(2048,65),(8192,65),(32768,65),(131072,33)]
methods=[('baseline_portable',old,'baseline',8),('baseline_native',plain,'baseline',8),
         ('targeted_fma',target,'bounded',8)]
methods += [(f'folded_{name}_{width}',lib,'folded',width) for name,lib in [('plain',plain),('fma',fused)] for width in [1,4,8]]
rows=[]
for length,segments in configs:
    for window in (0,1):
        item=case(plain,length=length,segments=segments,window_kind=window)
        expected=None
        for name,lib,method,width in methods:
            call,out,count,used,prep,seg=prepare(lib,item,method,width)
            start=time.perf_counter();call();elapsed=time.perf_counter()-start
            if expected is None: expected=out[0].value
            relative=abs((out[0].value-expected)/expected)
            iterations=max(1,min(200,int(.020/max(elapsed,1e-9))))
            repeats=[]
            for _ in range(3):
                start=time.perf_counter()
                for __ in range(iterations):call()
                repeats.append((time.perf_counter()-start)/iterations)
            prof,*rest=prepare(lib,item,method,width,profile=True);prof()
            _,_,used_p,prep_p,seg_p=rest
            rows.append(dict(label=item['label'],length=length,segments=count.value,window=window,
                             method=name,iterations=iterations,repeats_s=repeats,
                             median_s=statistics.median(repeats),preparation_s=prep_p.value,
                             segments_s=seg_p.value,used=used_p.value,relative_power_error=relative))
summary=[]
for length,segments in configs:
    for window in (0,1):
        group={row['method']:row for row in rows if row['length']==length and row['window']==window}
        best=min((row for row in group.values() if row['method'].startswith('folded')),key=lambda row:row['median_s'])
        summary.append(dict(length=length,window=window,best=best['method'],
                            native_speedup=group['baseline_native']['median_s']/best['median_s'],
                            targeted_fma_speedup=group['targeted_fma']['median_s']/best['median_s'],
                            total_us=best['median_s']*1e6,prep_us=best['preparation_s']*1e6,
                            segments_us=best['segments_s']*1e6,
                            max_relative_error=max(row['relative_power_error'] for row in group.values())))
paths=[Path(lib._name) for lib in (plain,fused,old,target)]
hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
report=dict(note='Serial native-frequency wall time including projection preparation and allocations. Original window and Fourier coefficient generation excluded for every method. Profiles are separate calls; repeated wall times are uninstrumented. Microbenchmark screening, not full API results.',
            binary_hashes=hashes,prototype_source_sha256=hashlib.sha256((ROOT/'folded_probe.c').read_bytes()).hexdigest(),
            summary=summary,rows=rows)
(ROOT/('micro-results-reuse.json' if args.coefficient_reuse else 'micro-results.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
