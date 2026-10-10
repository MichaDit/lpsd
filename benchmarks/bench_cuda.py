# SPDX-License-Identifier: GPL-3.0-or-later
"""Run optional CUDA experiments serially; all transfers synchronize.

First-call timing includes constructing the adapter, CUDA/CuPy initialization
and NVRTC JIT, but excludes process launch and ordinary CPU-module imports.
Warm calls rebuild the plan and stream every coefficient/start/power payload.
The tune mode is resident device-kernel diagnostics, never API speedup evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import pandas as pd
from lpsd_fast import api, lpsd
from lpsd._helpers import _kaiser_alpha, _kaiser_rov
from benchmarks.cuda_lpsd import CUDAEstimator, OpenCLEstimator, HERE
from benchmarks.bench_lpsd import build_evidence, fingerprint
from benchmarks.check_accuracy import CASES, WINDOWS, make_signal, compare_outputs, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['timing','audit','tune'],default='timing')
    parser.add_argument('--backend',choices=['cuda','opencl'],default='cuda')
    parser.add_argument('--platform',default='AMD')
    parser.add_argument('--n',type=int,default=1000000)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--strategy',choices=['warp','block','adaptive'],default='adaptive')
    parser.add_argument('--threads',type=int,default=128)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='tune' and args.backend!='cuda': parser.error('tune currently requires CUDA')
    def estimator():
        return CUDAEstimator(args.strategy,args.threads) if args.backend=='cuda' else OpenCLEstimator(args.platform,args.threads)
    report=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
                mode=args.mode,backend=args.backend,n=args.n,source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in HERE.glob('*') if p.suffix in ('.c','.cu','.cl')},
                build_reports=build_evidence(),calls=[],completed=False,
                bridge_build=json.loads((HERE/'bridge.build.json').read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    if args.mode=='timing':
        x=pd.Series(np.random.default_rng(20261008).normal(size=args.n),copy=False)
        kwargs=dict(sample_rate=1.,n_frequencies=1000,n_averages=100,detrending_order=0,
                    outputs=('psd','nsd'))
        report['input_sha256']=digest(x.to_numpy())
        report['parameters']=dict(kwargs,workers=args.workers,strategy=args.strategy,threads=args.threads)
        t=time.perf_counter()
        gpu=estimator()
        result=gpu(x,**kwargs)
        report['first_complete_gpu_host_call_s']=time.perf_counter()-t
        reference=lpsd(x,workers=args.workers,kernel='fast',max_working_mb=4096,**kwargs)
        reference_for_accuracy=reference.rename(columns={'nsd':'asd'})
        report['first_call_accuracy']=compare_outputs(reference_for_accuracy,result,psd_absolute_limit=1e-24,nsd_absolute_limit=1e-12)
        gpu(x,**kwargs)
        for repeat in range(args.repeats):
            names=['cpu','gpu'] if repeat%2==0 else ['gpu','cpu']
            for name in names:
                t=time.perf_counter()
                result=gpu(x,**kwargs) if name=='gpu' else lpsd(x,workers=args.workers,kernel='fast',max_working_mb=4096,**kwargs)
                wall=time.perf_counter()-t
                accuracy=compare_outputs(reference_for_accuracy,result,psd_absolute_limit=1e-24,nsd_absolute_limit=1e-12)
                item=dict(name=name,repeat=repeat,wall_s=wall,output_sha256=fingerprint(result),accuracy=accuracy)
                report['calls'].append(item)
                print(name,repeat,round(wall,6),flush=True)
                save()
        result=gpu(x,profile=True,**kwargs)
        report['additional_gpu_profile']=gpu.last_profile
        report['median_wall_s']={name:statistics.median(r['wall_s'] for r in report['calls'] if r['name']==name) for name in ('cpu','gpu')}
        report['cpu_over_gpu_ratio']=report['median_wall_s']['cpu']/report['median_wall_s']['gpu']
    else:
        gpu=estimator()
    if args.backend=='cuda':
        cp=gpu.cp
        props=cp.cuda.runtime.getDeviceProperties(0)
        report['gpu']={k:v.decode(errors='replace') if isinstance(v,bytes) else v for k,v in props.items()}
        report['runtime']=dict(cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                               cuda_driver=cp.cuda.runtime.driverGetVersion(),free_total_bytes=cp.cuda.runtime.memGetInfo())
    else:
        report['gpu']=dict(name=gpu.device.name,platform=gpu.device.platform.name,
                           global_memory_bytes=gpu.device.global_mem_size,double_fp_config=int(gpu.device.double_fp_config))
        report['runtime']=dict(pyopencl=gpu.cl.VERSION_TEXT,driver=gpu.device.driver_version)
    if args.mode=='audit':
        report['criterion']='Relative <1% OR absolute PSD <=1e-24 / NSD <=1e-12 in synthetic units; no amplitude floor; order 1+ explicitly CPU fallback'
        names=list(CASES)+['zero_kaiser200','constant_kaiser200']
        for name in names:
            if name in CASES:
                kind,window=CASES[name]
                x,seed=make_signal(kind,args.n,50.,20261008)
            else:
                kind,window=name.split('_',1)
                x=np.zeros(args.n) if kind=='zero' else np.full(args.n,10.)
            function,psll,overlap=WINDOWS[window]
            order=1 if kind=='ramp_nanovolt' else None if kind in ('offbin_tone','onbin_tone') else 0
            kwargs=dict(sample_rate=50.,n_frequencies=96,n_averages=16,detrending_order=order,
                        window_function=function,overlap=overlap,psll=psll)
            reference=lpsd(x,workers=1,kernel='scalar',**kwargs)
            result=gpu(x,**kwargs)
            row=compare_outputs(reference,result,psd_absolute_limit=1e-24,nsd_absolute_limit=1e-12)
            row.update(name=name,input_sha256=digest(x),execution=result.attrs['cuda_experiment'])
            report['calls'].append(row)
            print(name,[m['count_not_meeting_combined_limit'] for m in row['metrics'].values()],flush=True)
            save()
    if args.mode=='tune':
        api._native()
        x=np.random.default_rng(20261008).normal(size=max(args.n,1048576))
        dx=cp.asarray(x)
        report['timing_scope']='Resident segmented projection only; coefficient H2D, D2H, preparation, aggregation, allocations excluded'
        report['configurations']=[]
        for length in (16,64,128,256,512,1024,4096,16384,65536,262144,1048576):
            count=int(np.floor((len(x)-length)*4./length+1.+.5))
            cr,ci,starts,s2=gpu.prepare_frequency(len(x),length,4.371,count,.75,np.kaiser,_kaiser_alpha(200.)*np.pi,0)
            dc,di,ds=cp.asarray(cr),cp.asarray(ci),cp.asarray(starts)
            out=cp.empty(count,dtype=cp.float64)
            strategies=[(s,t) for s in ('warp','block') for t in (64,128,256)]
            rows=[]
            for repeat in range(-1,args.repeats):
                random.Random(length+repeat).shuffle(strategies)
                for strategy,threads in strategies:
                    blocks=(count+threads//32-1)//(threads//32) if strategy=='warp' else count
                    start,end=cp.cuda.Event(),cp.cuda.Event()
                    start.record()
                    gpu.kernels[strategy]((blocks,),(threads,),(dx,dc,di,ds,np.int32(length),np.int32(count),np.int32(1),out))
                    end.record(); end.synchronize()
                    rows.append(dict(strategy=strategy,threads=threads,repeat=repeat,device_s=cp.cuda.get_elapsed_time(start,end)/1000.))
            entry=dict(length=length,count=count,samples=rows,
                       medians={f'{s}-{t}':statistics.median(r['device_s'] for r in rows if r['repeat']>=0 and r['strategy']==s and r['threads']==t) for s,t in strategies})
            report['configurations'].append(entry)
            print(length,entry['medians'],flush=True)
            save()
    report['completed']=True
    failures=[]
    if args.mode=='audit':
        failures=[row['name'] for row in report['calls']
                  if not row['frequency_index_exact'] or not all(m['combined_limit_satisfied'] for m in row['metrics'].values())]
        report['summary']={'cases':len(report['calls']),'failed_cases':failures,'all_cases_pass':not failures}
    save()
    return int(bool(failures))


if __name__=='__main__':
    sys.exit(main())
