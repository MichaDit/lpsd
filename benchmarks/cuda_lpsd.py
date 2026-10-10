# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental FP64 PSD/NSD adapter. No production GPU dispatch is enabled.

One input copy, host projected coefficients and exact starts per frequency,
GPU segment powers, inherited host recurrence. Order 1+ is an explicit CPU
fallback. Requires optional CuPy/CUDA and a separately built bridge.
"""
import ctypes as ct
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from lpsd_fast import api, lpsd as cpu_lpsd
from lpsd._helpers import _kaiser_alpha, _kaiser_rov
from lpsd_fast.build import build_shared

HERE = Path(__file__).parent / 'experiments' / 'cuda_segments'
LIBRARY = HERE / ('bridge.dll' if os.name == 'nt' else 'bridge.so')


def build_bridge():
    return build_shared(HERE / 'bridge.c', LIBRARY, simd=True, native=True)


class CUDAEstimator:
    def __init__(self, strategy='adaptive', threads=128):
        import cupy as cp
        self.cp = cp
        # Respect memory already occupied by unrelated desktop applications.
        # Keep a small driver reserve; CuPy releases cached blocks at the cap.
        free,_=cp.cuda.runtime.memGetInfo()
        self.pool_limit=max(0,free-64*1024**2)
        cp.get_default_memory_pool().set_limit(size=self.pool_limit)
        self.strategy, self.threads = strategy, threads
        if strategy not in ('warp', 'block', 'adaptive') or threads not in (64,128,256):
            raise ValueError('strategy=warp/block/adaptive; threads=64/128/256')
        self.load_bridge()
        options=('--std=c++11','--fmad=false')
        source=(HERE / 'segments.cu').read_text()
        self.kernels={name:cp.RawKernel(source,name+'_segments',options=options) for name in ('warp','block')}
        self.last_profile={}

    def load_bridge(self):
        self.bridge = ct.CDLL(str(LIBRARY))
        dp=ct.POINTER(ct.c_double)
        ip=ct.POINTER(ct.c_int32)
        self.bridge.bench_prepare.argtypes=[dp,dp,ct.c_long,ct.c_int]
        self.bridge.bench_starts.argtypes=[ip,ct.c_long,ct.c_long,ct.c_double]
        self.bridge.bench_starts.restype=ct.c_long
        self.bridge.bench_initial.argtypes=[dp,dp,dp,ct.c_long,ip,ct.c_long,ct.c_int,dp]
        self.bridge.bench_reference.argtypes=self.bridge.bench_initial.argtypes
        self.bridge.bench_aggregate.argtypes=[dp,ct.c_long]
        self.bridge.bench_aggregate.restype=ct.c_double

    def synchronize(self):
        self.cp.cuda.get_current_stream().synchronize()

    def to_device(self, x):
        return self.cp.asarray(x)

    def host_powers(self, device_powers):
        return device_powers.get()

    @staticmethod
    def ip(array):
        return array.ctypes.data_as(ct.POINTER(ct.c_int32))

    def prepare_frequency(self, n, length, m, count, overlap, window, beta, order):
        w=np.empty(length)
        if window is np.kaiser:
            status=api._LIB.generate_kaiser_series(api._pointer(w),length,beta)
            if status: raise RuntimeError('Window generation failed')
        else:
            w=np.ascontiguousarray(window(length),dtype=np.float64)
        s1,s2=ct.c_double(),ct.c_double()
        status=api._LIB.window_sums(api._pointer(w),length,ct.byref(s1),ct.byref(s2))
        if status or not np.isfinite(s2.value) or not s2.value: raise ValueError('Invalid window normalization')
        cr,ci=api._coefficients(w,m,length,strict=False,blocked=True)
        status=self.bridge.bench_prepare(api._pointer(cr),api._pointer(ci),length,order)
        if status: raise ValueError('Unsupported projection')
        starts=np.empty(count,dtype=np.int32)
        actual=self.bridge.bench_starts(self.ip(starts),n,length,overlap*100)
        if actual != count: raise AssertionError(('Segment count mismatch',actual,count))
        return cr,ci,starts,s2.value

    def project(self, dx, cr, ci, starts, order, strategy=None, threads=None):
        cp=self.cp
        strategy=strategy or self.strategy
        threads=threads or self.threads
        if strategy == 'adaptive':
            # Fixed selection from the retained six-layout resident screen.
            strategy='warp' if len(cr)<65536 else 'block'
            if strategy=='block': threads=256
        dcr,dci,ds=cp.asarray(cr),cp.asarray(ci),cp.asarray(starts)
        out=cp.empty(len(starts),dtype=cp.float64)
        blocks=(len(starts)+(threads//32)-1)//(threads//32) if strategy=='warp' else len(starts)
        self.kernels[strategy]((blocks,),(threads,),
                              (dx,dcr,dci,ds,np.int32(len(cr)),np.int32(len(starts)),np.int32(order==0),out))
        return out

    def __call__(self, data, sample_rate=1., n_frequencies=1000, n_averages=100,
                 detrending_order=0, window_function=np.kaiser, overlap=None,
                 psll=200., outputs=('psd','nsd'), profile=False):
        if detrending_order not in (None,0):
            result=cpu_lpsd(data,sample_rate=sample_rate,n_frequencies=n_frequencies,
                            n_averages=n_averages,detrending_order=detrending_order,
                            window_function=window_function,overlap=overlap,psll=psll,
                            workers=1,kernel='fast',outputs=outputs)
            result.attrs['cuda_experiment']={'executed':'CPU fallback','reason':'Order 1+ not implemented on GPU'}
            return result
        if tuple(outputs) not in (('psd',),('psd','nsd')): raise ValueError('Experimental PSD or PSD/NSD only')
        self.synchronize()
        started=time.perf_counter()
        api._native()
        x=api._array(data)
        window=api._resolve_window(window_function)
        overlap=_kaiser_rov(_kaiser_alpha(psll)) if overlap is None else overlap
        if not np.isfinite(overlap) or not 0<=overlap<1: raise ValueError('Invalid overlap')
        plan=api._frequency_plan(len(x),sample_rate,overlap,1,0,n_frequencies,n_averages)
        f,r,m,lengths,counts=plan
        dx=self.to_device(x)
        values=[]
        stages={'preparation_s':0.,'gpu_and_transfers_s':0.,'initial_and_aggregation_s':0.}
        order=-1 if detrending_order is None else 0
        for j,length in enumerate(lengths):
            length,count=int(length),int(counts[j])
            t=time.perf_counter()
            cr,ci,starts,s2=self.prepare_frequency(len(x),length,m[j],count,overlap,window,_kaiser_alpha(psll)*np.pi,order)
            if profile: stages['preparation_s']+=time.perf_counter()-t
            t=time.perf_counter()
            powers=self.host_powers(self.project(dx,cr,ci,starts,order))
            if profile: stages['gpu_and_transfers_s']+=time.perf_counter()-t
            t=time.perf_counter()
            self.bridge.bench_initial(api._pointer(x),api._pointer(cr),api._pointer(ci),length,self.ip(starts),count,order,api._pointer(powers))
            mean=self.bridge.bench_aggregate(api._pointer(powers),count)
            values.append(2.*mean/sample_rate/s2)
            if profile: stages['initial_and_aggregation_s']+=time.perf_counter()-t
        # Preserve complex64 PSD conversion and the existing NSD square root.
        psd=np.asarray(values,dtype=np.complex64)
        columns={'psd':psd.real}
        if 'nsd' in outputs: columns['nsd']=np.sqrt(psd).real
        result=pd.DataFrame(columns,index=f)
        result.index.name='frequency'
        self.synchronize()
        self.last_profile=dict(stages,wall_s=time.perf_counter()-started)
        result.attrs['cuda_experiment']={'executed':'FP64 GPU segments with CPU preparation/initial projections/aggregation',
                                         'strategy':self.strategy,'threads':self.threads,'fma':False,
                                         'pool_limit_bytes':self.pool_limit,
                                         'profile':self.last_profile if profile else None}
        return result


class OpenCLEstimator(CUDAEstimator):
    """Same host contract; actual FP64 OpenCL on an explicitly selected GPU."""
    def __init__(self, platform='AMD',threads=128):
        import pyopencl as cl
        self.cl=cl
        candidates=[d for p in cl.get_platforms() if platform.lower() in p.name.lower() for d in p.get_devices()]
        if len(candidates)!=1: raise ValueError('Select exactly one OpenCL device')
        self.device=candidates[0]
        if not self.device.double_fp_config: raise RuntimeError('FP64 unavailable')
        self.context=cl.Context([self.device])
        self.queue=cl.CommandQueue(self.context)
        self.threads,self.strategy=threads,'OpenCL block'
        self.pool_limit=None
        self.load_bridge()
        self.program=cl.Program(self.context,(HERE/'segments.cl').read_text()).build(options=['-cl-std=CL1.2'])
        self.kernel=cl.Kernel(self.program,'block_segments')
        self.last_profile={}

    def synchronize(self):
        self.queue.finish()

    def to_device(self,x):
        return self.cl.Buffer(self.context,self.cl.mem_flags.READ_ONLY|self.cl.mem_flags.COPY_HOST_PTR,hostbuf=x)

    def project(self,dx,cr,ci,starts,order):
        cl=self.cl
        dc,di,ds=(self.to_device(a) for a in (cr,ci,starts))
        out=cl.Buffer(self.context,cl.mem_flags.WRITE_ONLY,size=len(starts)*8)
        self.kernel(self.queue,(len(starts)*self.threads,),(self.threads,),dx,dc,di,ds,
                    np.int32(len(cr)),np.int32(len(starts)),np.int32(order==0),out,
                    cl.LocalMemory(self.threads*8),cl.LocalMemory(self.threads*8))
        return out,len(starts)

    def host_powers(self,device_powers):
        buffer,count=device_powers
        out=np.empty(count)
        self.cl.enqueue_copy(self.queue,out,buffer,is_blocking=True)
        return out


if __name__=='__main__':
    build_bridge()
