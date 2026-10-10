# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare matched prepared CPU/GPU plans and the public CPU API.

This changes the reuse policy, not the spectral estimator. Host coefficient
preparation and device plan construction are measured separately. GPU host
calls copy the input each time; resident-input diagnostics omit only that copy.
All GPU calculations still return every segment power to the host recurrence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
import numpy as np
import pandas as pd
from lpsd_fast import api,lpsd
from lpsd._helpers import _kaiser_alpha,_kaiser_rov
from benchmarks.cuda_lpsd import CUDAEstimator
from benchmarks.check_accuracy import compare_outputs
from benchmarks.bench_lpsd import build_evidence


class PreparedPlan:
    def __init__(self,gpu,n,device=True):
        self.gpu,self.n=gpu,n
        self.rows=[]
        overlap=_kaiser_rov(_kaiser_alpha(200.))
        self.plan=api._frequency_plan(n,1.,overlap,1,0,1000,100)
        api._native()
        t=time.perf_counter()
        for j,length in enumerate(self.plan[3]):
            cr,ci,starts,s2=gpu.prepare_frequency(n,int(length),self.plan[2][j],int(self.plan[4][j]),overlap,np.kaiser,_kaiser_alpha(200.)*np.pi,0)
            self.rows.append((cr,ci,starts,s2))
        self.host_preparation_s=time.perf_counter()-t
        self.payload_bytes=n*8+sum(cr.nbytes+ci.nbytes+starts.nbytes+len(starts)*8 for cr,ci,starts,_ in self.rows)
        self.host_plan_bytes=sum(cr.nbytes+ci.nbytes+starts.nbytes for cr,ci,starts,_ in self.rows)
        if not device:
            self.device_plan_s=0.
            return
        if self.payload_bytes>gpu.pool_limit: raise MemoryError('Resident plan exceeds the recorded free-device-memory limit')
        t=time.perf_counter()
        cp=gpu.cp
        self.dx=cp.empty(n,dtype=cp.float64)
        self.device_rows=[(cp.asarray(cr),cp.asarray(ci),cp.asarray(starts),cp.empty(len(starts))) for cr,ci,starts,_ in self.rows]
        gpu.synchronize()
        self.device_plan_s=time.perf_counter()-t

    def finish(self,powers,row,x):
        cr,ci,starts,s2=row
        self.gpu.bridge.bench_initial(api._pointer(x),api._pointer(cr),api._pointer(ci),len(cr),self.gpu.ip(starts),len(starts),0,api._pointer(powers))
        return 2*self.gpu.bridge.bench_aggregate(api._pointer(powers),len(starts))/s2

    def frame(self,values):
        psd=np.asarray(values,dtype=np.complex64)
        result=pd.DataFrame({'psd':psd.real,'nsd':np.sqrt(psd).real},index=self.plan[0])
        result.index.name='frequency'
        return result

    def gpu_call(self,x,transfer=True):
        x=api._array(x)
        if transfer: self.dx.set(x)
        values=[]
        for row,device in zip(self.rows,self.device_rows):
            cr,ci,starts,_=row
            dcr,dci,ds,out=device
            strategy,threads=('warp',128) if len(cr)<65536 else ('block',256)
            blocks=(len(starts)+threads//32-1)//(threads//32) if strategy=='warp' else len(starts)
            self.gpu.kernels[strategy]((blocks,),(threads,),(self.dx,dcr,dci,ds,np.int32(len(cr)),np.int32(len(starts)),np.int32(1),out))
            values.append(self.finish(out.get(),row,x))
        result=self.frame(values)
        self.gpu.synchronize()
        return result

    def cpu_call(self,x,workers):
        x=api._array(x)
        def one(row):
            cr,ci,starts,s2=row
            powers=np.empty(len(starts))
            self.gpu.bridge.bench_reference(api._pointer(x),api._pointer(cr),api._pointer(ci),len(cr),self.gpu.ip(starts),len(starts),0,api._pointer(powers))
            return 2*self.gpu.bridge.bench_aggregate(api._pointer(powers),len(starts))/s2
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return self.frame(list(pool.map(one,self.rows)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=1000000)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-only',action='store_true',help='Host prepared-plan study without allocating any GPU buffers')
    args=parser.parse_args()
    x=np.random.default_rng(20261008).normal(size=args.n)
    t=time.perf_counter()
    if args.cpu_only:
        gpu=object.__new__(CUDAEstimator)
        gpu.load_bridge()
        gpu.pool_limit=None
    else: gpu=CUDAEstimator()
    plan=PreparedPlan(gpu,args.n,not args.cpu_only)
    result=plan.cpu_call(x,args.workers) if args.cpu_only else plan.gpu_call(x)
    first=time.perf_counter()-t
    reference=lpsd(x,sample_rate=1.,workers=args.workers,kernel='fast',outputs=('psd','asd'),max_working_mb=4096)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),n=args.n,workers=args.workers,
                build_reports=build_evidence(),pool_limit_bytes=gpu.pool_limit,
                resident_payload_bytes=plan.payload_bytes,host_preparation_s=plan.host_preparation_s,
                host_plan_bytes=plan.host_plan_bytes,cpu_only=args.cpu_only,
                device_plan_construction_s=plan.device_plan_s,first_complete_with_plan_s=first,calls=[],completed=False,
                scope='Prepared CPU/GPU use exactly the same projected coefficients, starts and CPU initial projections; GPU returns powers for host inherited aggregation; public CPU regenerates its plan')
    functions={'public_cpu':lambda:lpsd(x,sample_rate=1.,workers=args.workers,kernel='fast',outputs=('psd','nsd'),max_working_mb=4096),
               'prepared_cpu':lambda:plan.cpu_call(x,args.workers),
               'prepared_gpu_host':lambda:plan.gpu_call(x),
               'prepared_gpu_resident_input':lambda:plan.gpu_call(x,False)}
    if args.cpu_only:
        functions={name:fn for name,fn in functions.items() if name in ('public_cpu','prepared_cpu')}
    for function in functions.values(): function()
    for repeat in range(args.repeats):
        names=list(functions) if repeat%2==0 else list(functions)[::-1]
        for name in names:
            t=time.perf_counter()
            result=functions[name]()
            wall=time.perf_counter()-t
            row=dict(name=name,repeat=repeat,wall_s=wall,
                     accuracy=compare_outputs(reference,result,psd_absolute_limit=1e-24,nsd_absolute_limit=1e-12))
            report['calls'].append(row)
            print(name,repeat,round(wall,6),flush=True)
    report['median_wall_s']={name:statistics.median(r['wall_s'] for r in report['calls'] if r['name']==name) for name in functions}
    report['completed']=True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__': main()
