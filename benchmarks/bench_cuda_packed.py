# SPDX-License-Identifier: GPL-3.0-or-later
"""Serial paired complete calls and strict numerical audits of packed CUDA."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from lpsd_fast import lpsd
from benchmarks.cuda_lpsd import CUDAEstimator,HERE
from benchmarks.cuda_packed import PackedCUDAEstimator,ResidentPackedPlan
from benchmarks.check_accuracy import CASES,WINDOWS,make_signal,compare_outputs,digest
from benchmarks.gpu_telemetry import snapshot


def passed(row):
    return row['frequency_index_exact'] and all(m['combined_limit_satisfied'] for m in row.get('metrics',{}).values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('timing','audit','reuse'),default='timing')
    p.add_argument('--method',choices=('packed','fft'),default='packed')
    p.add_argument('--n',type=int,default=1000000)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--batch-mb',type=int,default=1024)
    p.add_argument('--precision',choices=('fp64','twofold'),default='fp64')
    p.add_argument('--threshold',type=int,default=16384)
    p.add_argument('--fma',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--split-length',type=int,default=32768)
    p.add_argument('--no-dc-guard',action='store_true')
    p.add_argument('--prefetch',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--legacy',action='store_true')
    p.add_argument('--schedule',choices=('alternating','burst'),default='alternating')
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        parameters={k:v for k,v in vars(args).items() if k!='output'},calls=[],completed=False,
        source_sha256={str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in HERE.glob('*') if f.suffix in ('.c','.cu','.cl')})
    root=Path(__file__).resolve().parents[1]
    report['python_source_sha256']={str(f.relative_to(root)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (Path(__file__),root/'benchmarks/cuda_lpsd.py',root/'benchmarks/cuda_packed.py',root/'benchmarks/cuda_fft_lpsd.py',root/'benchmarks/gpu_telemetry.py')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    started=time.perf_counter()
    if args.method=='fft':
        if args.mode=='reuse': p.error('FFT correlation currently has no resident-plan mode')
        from benchmarks.cuda_fft_lpsd import FFTCorrelationEstimator
        gpu=FFTCorrelationEstimator()
    else:
        gpu=PackedCUDAEstimator(args.workers,args.batch_mb,args.precision,args.threshold,args.fma,args.split_length,not args.no_dc_guard,args.prefetch)
    cp=gpu.cp
    from cupy.cuda import cufft
    props=cp.cuda.runtime.getDeviceProperties(0)
    report['runtime']=dict(cupy=cp.__version__,cufft=cufft.getVersion(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),free_total_bytes=cp.cuda.runtime.memGetInfo(),
        gpu=props['name'].decode())
    report['adapter_initialization_s']=time.perf_counter()-started
    if args.mode=='audit':
        for name in list(CASES)+['zero_kaiser200','constant_kaiser200']:
            if name in CASES:
                kind,window=CASES[name]
                x,_=make_signal(kind,args.n,50.,20261008)
            else:
                kind,window=name.split('_',1)
                x=np.zeros(args.n) if kind=='zero' else np.full(args.n,10.)
            function,psll,overlap=WINDOWS[window]
            order=1 if kind=='ramp_nanovolt' else None if kind in ('offbin_tone','onbin_tone') else 0
            kwargs=dict(sample_rate=50.,n_frequencies=96,n_averages=16,detrending_order=order,
                        window_function=function,overlap=overlap,psll=psll)
            reference=lpsd(x,workers=1,kernel='scalar',**kwargs)
            actual=gpu(x,**kwargs)
            row=compare_outputs(reference,actual,psd_absolute_limit=1e-24,nsd_absolute_limit=1e-12)
            row.update(name=name,execution=actual.attrs['cuda_experiment'],input_sha256=digest(x))
            report['calls'].append(row)
            print(name,'PASS' if passed(row) else 'FAIL',flush=True)
            save()
    else:
        x=np.random.default_rng(20261008).normal(size=args.n)
        report['input_sha256']=digest(x)
        def public(): return lpsd(x,sample_rate=1.,workers=args.workers,kernel='fast',max_working_mb=4096,outputs=('psd','nsd'))
        reference=public().rename(columns={'nsd':'asd'})
        functions={'public_cpu':public,'packed_gpu':lambda:gpu(x)}
        if args.legacy:
            t=time.perf_counter();legacy=CUDAEstimator()
            report['legacy_adapter_initialization_s']=time.perf_counter()-t
            functions['legacy_gpu']=lambda:legacy(x)
        if args.mode=='reuse':
            t=time.perf_counter()
            plan=ResidentPackedPlan(gpu,args.n)
            report['resident_plan']=dict(total_construction_s=time.perf_counter()-t,
                host_preparation_s=plan.host_preparation_s,pack_upload_s=plan.upload_s,payload_bytes=plan.payload_bytes)
            functions={'public_cpu':public,'prepared_cpu':lambda:plan.cpu_call(x),'resident_gpu':lambda:plan(x)}
        first={}
        for name,fn in functions.items():
            t=time.perf_counter();actual=fn();wall=time.perf_counter()-t
            first[name]=dict(wall_s=wall,accuracy=compare_outputs(reference,actual,psd_absolute_limit=1e-24,nsd_absolute_limit=1e-12))
        report['first_calls']=first
        for fn in functions.values(): fn()
        order=[(repeat,name) for repeat in range(args.repeats)
               for name in (list(functions) if repeat%2==0 else list(functions)[::-1])]
        if args.schedule=='burst':
            order=[(repeat,name) for name in functions for repeat in range(args.repeats)]
        previous=None
        for repeat,name in order:
                if args.schedule=='burst' and name!=previous:
                    for _ in range(5): functions[name]()
                previous=name
                t=time.perf_counter(); actual=functions[name](); wall=time.perf_counter()-t
                row=dict(name=name,repeat=repeat,wall_s=wall,
                    gpu_snapshot_after_call=snapshot(),
                    accuracy=compare_outputs(reference,actual,psd_absolute_limit=1e-24,nsd_absolute_limit=1e-12))
                report['calls'].append(row)
                print(name,repeat,round(wall,6),flush=True);save()
        report['median_wall_s']={name:statistics.median(r['wall_s'] for r in report['calls'] if r['name']==name) for name in functions}
        if args.mode=='timing':
            gpu(x,profile=True);report['additional_profile']=gpu.last_profile
        else: plan.close()
    report['completed']=True
    report['all_combined_limits_passed']=all(passed(r.get('accuracy',r)) for r in report['calls'])
    save()
    if not report['all_combined_limits_passed']: raise SystemExit(1)


if __name__=='__main__': main()
