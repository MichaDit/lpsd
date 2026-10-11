# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare real cuFFT/VkFFT and FFTW with identical periodogram semantics.

This is a different estimator from LPSD. Raw resident FFT, prepared complete
pipeline (including input transfer and output), and fresh plans are separate.
"""
import argparse
import gc
import hashlib
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time
import numpy as np
import pandas as pd
from lpsd._helpers import _kaiser_alpha
from benchmarks.bench_fftw import FFTWLibrary,Periodogram,KaiserWindow,lpsd_grid
from benchmarks.check_accuracy import digest
from benchmarks.gpu_telemetry import snapshot


class GPUPeriodogram:
    def __init__(self,n,backend='cufft'):
        import cupy as cp
        self.cp,self.n,self.backend=cp,n,backend
        self.fused='_' in backend
        self.graph_enabled=backend.endswith('_graph')
        self.stream=cp.cuda.Stream(non_blocking=True)
        provider=KaiserWindow()
        w=provider.generate(n,_kaiser_alpha(200.)*np.pi)
        self.s1,self.s2=provider.sums(w)
        self.window=cp.asarray(w)
        self.grid=lpsd_grid(n,1.,200.,1000,100)
        self.cuts=cp.asarray(self.grid['cuts'][:-1])
        self.counts=cp.asarray(self.grid['counts'])
        self.dx=cp.empty(n,dtype=cp.float64)
        self.dy=cp.empty(n//2+1,dtype=cp.complex128)
        self.powers=cp.empty(n//2+1,dtype=cp.float64)
        self.input=cp.empty_like(self.dx)
        self.all_cuts=cp.asarray(self.grid['cuts'])
        self.parts=cp.empty((n+8191)//8192,dtype=cp.float64)
        self.mean=cp.empty(1,dtype=cp.float64)
        self.density=cp.empty(len(self.counts),dtype=cp.float32)
        source=(Path(__file__).parent/'experiments/cuda_segments/fft_pipeline.cu').read_text()
        self.kernels={name:cp.RawKernel(source,name,options=('--fmad=false',))
                      for name in ('anchored_parts','anchored_mean','window_input','logarithmic_power')}
        if backend.startswith('cufft'):
            from cupy.cuda import cufft
            self.plan=cufft.Plan1d(n,cufft.CUFFT_D2Z,1)
            self.direction=cufft.CUFFT_FORWARD
        else:
            from pyvkfft.cuda import VkFFTApp
            kw={}
            if 'tuned' in backend:
                from pyvkfft.tune import tune_vkfft
                with self.stream:
                    self.dx.fill(1)
                    kw,results=tune_vkfft({'backend':'cupy','src':self.dx,'dest':self.dy},
                        (n,),np.float64,ndim=1,inplace=False,r2c=True,norm=0,stream=self.stream)
                self.tuning_evidence=dict(selected=kw,candidates=results,
                    scope='Best of three forward/inverse event batches, only a raw-FFT selection heuristic')
                if not results: raise RuntimeError('VkFFT autotuning produced no evidence')
            self.plan=VkFFTApp((n,),np.float64,ndim=1,inplace=False,r2c=True,norm=0,stream=self.stream,**kw)
        cp.cuda.get_current_stream().synchronize()
        if self.graph_enabled:
            with self.stream:
                self.input.fill(0)
                self.pipeline();self.pipeline()
            self.stream.synchronize()
            with self.stream:
                self.stream.begin_capture()
                self.pipeline()
                self.graph=self.stream.end_capture()

    def execute(self):
        if self.backend.startswith('cufft'): self.plan.fft(self.dx,self.dy,self.direction)
        else: self.plan.fft(self.dx,self.dy)

    def pipeline(self):
        k=self.kernels
        k['anchored_parts']((len(self.parts),),(256,),(self.input,np.int32(self.n),self.parts))
        k['anchored_mean']((1,),(256,),(self.parts,np.int32(len(self.parts)),np.int32(self.n),self.mean))
        k['window_input'](((self.n+255)//256,),(256,),(self.input,self.window,self.mean,np.int32(self.n),self.dx))
        self.execute()
        k['logarithmic_power']((len(self.counts),),(256,),
            (self.dy,self.all_cuts,np.int32(len(self.counts)),np.int32(self.n),np.float64(self.s2),self.density))

    def compute(self,x):
        cp=self.cp
        with self.stream:
            if self.fused:
                self.input.set(x,stream=self.stream)
                if self.graph_enabled: self.graph.launch(stream=self.stream)
                else: self.pipeline()
                result=self.density.get(stream=self.stream)
            else:
                self.dx.set(x,stream=self.stream)
                self.dx-=self.dx[0].copy()
                self.dx-=cp.mean(self.dx)
                self.dx*=self.window
                self.execute()
                cp.square(self.dy.real,out=self.powers)
                self.powers+=self.dy.imag*self.dy.imag
                self.powers*=2/self.s2
                self.powers[0]*=.5
                if self.n%2==0: self.powers[-1]*=.5
                density=cp.add.reduceat(self.powers,self.cuts)/self.counts
                result=density.astype(cp.float32).get(stream=self.stream)
        return pd.DataFrame({'psd':result},index=pd.Index(self.grid['frequency_labels_hz'],name='frequency'))

    def validation(self):
        cp=self.cp
        self.stream.synchronize()
        if self.fused:
            cp.square(self.dy.real,out=self.powers)
            self.powers+=self.dy.imag*self.dy.imag
            self.powers*=2/self.s2
            self.powers[0]*=.5
            if self.n%2==0: self.powers[-1]*=.5
        weighted=float(cp.sum(self.dx*self.dx).get()/self.s2)
        full=float(cp.sum(self.powers).get()/self.n)
        positive=float(cp.sum(self.powers[1:]).get()/self.n)
        aggregated=float(cp.sum(cp.add.reduceat(self.powers,self.cuts)).get()/self.n)
        return dict(finite=bool(cp.isfinite(self.powers).all().get()),weighted_power=weighted,
            integrated_power=full,parseval_absolute_error=abs(full-weighted),
            aggregation_absolute_error=abs(positive-aggregated))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=1000000)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--threads',type=int,nargs='+',default=[1,4,8,16])
    p.add_argument('--fftw-library',required=True)
    p.add_argument('--backends',nargs='+',choices=['cufft','vkfft','cufft_fused','cufft_graph','vkfft_graph','vkfft_tuned_graph'],default=['cufft','vkfft','cufft_fused','cufft_graph','vkfft_graph'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--schedule',choices=('alternating','burst'),default='alternating')
    args=p.parse_args()
    import cupy as cp
    from cupy.cuda import cufft
    x=np.random.default_rng(20261008).normal(size=args.n)
    report=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),n=args.n,
        repeats=args.repeats,input_sha256=digest(x),estimator='Full-record anchored mean-detrended periodic Kaiser200 log-bin periodogram; NOT LPSD',
        runtime=dict(cupy=cp.__version__,cufft=cufft.getVersion(),free_total_bytes=cp.cuda.runtime.memGetInfo()),
        schedule=args.schedule,calls=[],first_plan_and_pipeline=[],validations={},tuning={},device_memory_after_plan={},completed=False)
    root=Path(__file__).resolve().parents[1]
    report['source_sha256']={str(f.relative_to(root)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (Path(__file__),root/'benchmarks/experiments/cuda_segments/fft_pipeline.cu',root/'benchmarks/bench_fftw.py',root/'benchmarks/gpu_telemetry.py')}
    try:
        from pyvkfft.cuda import vkfft_version
        import pyvkfft.version
        report['runtime']['vkfft']=vkfft_version()
        report['runtime']['pyvkfft']=pyvkfft.version.__version__
    except ImportError: pass
    def save(): args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plans={}
    for threads in args.threads:
        name=f'fftw_{threads}'
        t=time.perf_counter()
        backend=FFTWLibrary(args.fftw_library,args.fftw_library,threads)
        report['runtime']['fftw']=backend.version
        plan=Periodogram(backend,args.n)
        frame=plan.compute(x)
        plans[name]=plan
        report['first_plan_and_pipeline'].append(dict(name=name,wall_s=time.perf_counter()-t))
        report['validations'][name]=plan.validation()
    reference=next(iter(plans.values())).compute(x)
    for name in args.backends:
        t=time.perf_counter()
        plan=GPUPeriodogram(args.n,name)
        frame=plan.compute(x)
        plans[name]=plan
        report['first_plan_and_pipeline'].append(dict(name=name,wall_s=time.perf_counter()-t))
        report['validations'][name]=plan.validation()
        if hasattr(plan,'tuning_evidence'): report['tuning'][name]=plan.tuning_evidence
        report['device_memory_after_plan'][name]=cp.cuda.runtime.memGetInfo()
        if report['device_memory_after_plan'][name][0]<512*1024**2:
            report['failure']='Combined GPU plans exhausted the 512 MiB device reserve; compare one backend per process'
            save()
            raise MemoryError(report['failure'])
    # FFTW's thread setting is global and affects subsequent plan creation;
    # each already-created FFTW plan retains its own thread configuration.
    for plan in plans.values(): plan.compute(x)
    order=[(repeat,name) for repeat in range(args.repeats)
           for name in (list(plans) if repeat%2==0 else list(plans)[::-1])]
    if args.schedule=='burst':
        order=[(repeat,name) for name in plans for repeat in range(args.repeats)]
    previous=None
    for repeat,name in order:
            if args.schedule=='burst' and name!=previous:
                for _ in range(5): plans[name].compute(x)
            previous=name
            plan=plans[name]
            t=time.perf_counter();actual=plan.compute(x);wall=time.perf_counter()-t
            a,b=actual.psd.to_numpy(),reference.psd.to_numpy()
            row=dict(name=name,scope='prepared_complete',repeat=repeat,wall_s=wall,
                gpu_snapshot_after_call=snapshot(),
                max_relative_error=float(np.max(np.abs(a.astype(float)-b)/np.abs(b))),
                frequencies_exact=actual.index.to_numpy().tobytes()==reference.index.to_numpy().tobytes(),
                output_sha256=digest(a))
            report['calls'].append(row)
            print(name,repeat,round(wall,6),flush=True);save()
    for repeat in range(args.repeats):
        names=list(plans) if repeat%2==0 else list(plans)[::-1]
        for name in names:
            plan=plans[name]
            execute=plan.execute if isinstance(plan,GPUPeriodogram) else plan.fft.execute
            if isinstance(plan,GPUPeriodogram):
                with plan.stream: execute()
                plan.stream.synchronize()
            else: execute()
            # Batch timings avoid timer granularity for short transforms.
            loops=10 if args.n<2000000 else 3
            t=time.perf_counter()
            if isinstance(plan,GPUPeriodogram):
                with plan.stream:
                    for _ in range(loops): execute()
                plan.stream.synchronize()
            else:
                for _ in range(loops): execute()
            report['calls'].append(dict(name=name,scope='raw_resident_fft',repeat=repeat,wall_s=(time.perf_counter()-t)/loops,loops=loops))
    # Release compared GPU plans before timing fresh allocations: otherwise
    # four large plans can exceed VRAM and cause WDDM paging in the fresh row.
    plan_names=list(plans)
    del plan,execute
    for name in args.backends: del plans[name]
    gc.collect();cp.get_default_memory_pool().free_all_blocks()
    report['device_memory_before_fresh']=cp.cuda.runtime.memGetInfo()
    # Fresh plan + all preprocessing + transfer + complete output, separately.
    for repeat in range(min(args.repeats,3)):
        names=plan_names if repeat%2==0 else plan_names[::-1]
        for name in names:
            t=time.perf_counter()
            if name in args.backends: new=GPUPeriodogram(args.n,name)
            else:
                threads=int(name.rsplit('_',1)[1])
                backend=FFTWLibrary(args.fftw_library,args.fftw_library,threads)
                backend.forget_wisdom()
                new=Periodogram(backend,args.n)
            new.compute(x)
            report['calls'].append(dict(name=name,scope='fresh_complete',repeat=repeat,wall_s=time.perf_counter()-t))
            if isinstance(new,Periodogram): new.close()
            del new
    report['medians']={scope:{name:statistics.median(r['wall_s'] for r in report['calls'] if r['scope']==scope and r['name']==name)
        for name in plan_names if any(r['scope']==scope and r['name']==name for r in report['calls'])}
        for scope in ('prepared_complete','raw_resident_fft','fresh_complete')}
    report['completed']=True;save()
    if any(r.get('max_relative_error',0)>=.01 or not r.get('frequencies_exact',True) for r in report['calls']): raise SystemExit(1)
    if any(not v['finite'] or v['parseval_absolute_error']>1e-11 or v['aggregation_absolute_error']>1e-11 for v in report['validations'].values()): raise SystemExit(1)
    for plan in plans.values():
        if isinstance(plan,Periodogram): plan.close()


if __name__=='__main__': main()
