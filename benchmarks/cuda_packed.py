# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded packed CUDA batches and optional reusable plans.

The same CPU-projected coefficients, exact segment starts, first eight CPU
projections and upstream mean recurrence are retained. FP64 is the default;
twofold FP32 products/accumulators are an explicitly selected experiment.
No production backend is changed.
"""
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import numpy as np
import pandas as pd
from lpsd_fast import api
from lpsd._helpers import _kaiser_alpha, _kaiser_rov
from benchmarks.cuda_lpsd import CUDAEstimator, HERE


class PackedBatch:
    def __init__(self, gpu, rows, threshold=65536, precision='fp64',workspace=None):
        cp=gpu.cp
        def empty(size,dtype,key):
            if workspace is None: return cp.empty(size,dtype=dtype)
            old=workspace.get(key)
            if old is None or old.size<size or old.dtype!=np.dtype(dtype):
                workspace[key]=cp.empty(size,dtype=dtype)
            return workspace[key][:size]
        def upload(host,key):
            if workspace is None: return cp.asarray(host)
            device=empty(host.size,host.dtype,key)
            device.set(host)
            return device
        self.gpu, self.rows = gpu, rows
        self.precision=precision
        self.lengths=np.asarray([len(r[0]) for r in rows],dtype=np.int32)
        counts=np.asarray([len(r[2]) for r in rows],dtype=np.int64)
        self.cuts=np.r_[0, np.cumsum(counts)]
        offsets=np.r_[0,np.cumsum(self.lengths[:-1],dtype=np.int64)]
        frequencies=np.repeat(np.arange(len(rows),dtype=np.int32),counts)
        starts=np.concatenate([r[2] for r in rows])
        def joined(column):
            arrays=[r[column] for r in rows]
            base=arrays[0].base
            if (isinstance(base,np.ndarray) and base.shape==(int(np.sum(self.lengths)),)
                    and all(a.base is base and a.ctypes.data==base.ctypes.data+int(o)*8
                            for a,o in zip(arrays,offsets))):
                return base
            return np.concatenate(arrays)
        self.dcr=upload(joined(0),'cr')
        self.dci=upload(joined(1),'ci')
        self.meta=tuple(upload(a,key) for a,key in zip((offsets,self.lengths,frequencies,starts),('offsets','lengths','fids','starts')))
        ids=np.arange(len(starts),dtype=np.int32)
        short=self.lengths[frequencies]<threshold
        # These powers are always replaced by CPU projections in finish().
        keep=(ids-np.repeat(self.cuts[:-1],counts))>=8
        self.groups=[upload(ids[short & keep],'short'),upload(ids[~short & keep],'long')]
        self.out=empty(len(starts),np.float64,'powers')
        self.payload_bytes=sum(a.nbytes for a in (self.dcr,self.dci,*self.meta,*self.groups,self.out))
        self.split=None
        if gpu.split_length and precision=='fp64':
            long_ids=ids[~short & keep]
            num=(self.lengths[frequencies[long_ids]]+gpu.split_length-1)//gpu.split_length
            cuts=np.r_[0,np.cumsum(num,dtype=np.int64)]
            segments=np.repeat(long_ids,num)
            parts=(np.arange(len(segments))-np.repeat(cuts[:-1],num)).astype(np.int32)
            self.split=(upload(segments,'segments'),upload(parts,'parts'),upload(cuts,'cuts'),
                        empty(len(segments),np.float64,'rr'),empty(len(segments),np.float64,'ii'))
            self.payload_bytes+=sum(a.nbytes for a in self.split)

    def launch(self, dx, order=0):
        prefix='packed' if self.precision=='fp64' else 'twofold'
        for layout,ids in zip(('warp','block'),self.groups):
            if not len(ids): continue
            if layout=='block' and self.split is not None:
                segments,parts,cuts,rr,ii=self.split
                self.gpu.packed_kernels['split_partials']((len(segments),),(256,),
                    (dx,self.dcr,self.dci,*self.meta,segments,parts,np.int32(len(segments)),
                     np.int32(self.gpu.split_length),np.int32(order==0),rr,ii))
                self.gpu.packed_kernels['finish_partials'](((len(ids)+255)//256,),(256,),
                    (rr,ii,ids,cuts,np.int32(len(ids)),self.out))
                continue
            threads=128 if layout=='warp' else 256
            blocks=(len(ids)+threads//32-1)//(threads//32) if layout=='warp' else len(ids)
            self.gpu.packed_kernels[prefix+'_'+layout]((blocks,),(threads,),
                (dx,self.dcr,self.dci,*self.meta,ids,np.int32(len(ids)),np.int32(order==0),self.out))

    def finish(self, x, powers, pool, sample_rate, order=0):
        def one(j):
            cr,ci,starts,s2=self.rows[j]
            p=powers[self.cuts[j]:self.cuts[j+1]]
            self.gpu.bridge.bench_initial(api._pointer(x),api._pointer(cr),api._pointer(ci),
                len(cr),self.gpu.ip(starts),len(starts),order,api._pointer(p))
            return 2*self.gpu.bridge.bench_aggregate(api._pointer(p),len(p))/sample_rate/s2
        return list(pool.map(one,range(len(self.rows))))


class PackedCUDAEstimator(CUDAEstimator):
    def __init__(self, workers=16, batch_mb=1024, precision='fp64', threshold=16384, fma=True,split_length=32768,guard_dc=True,prefetch=True):
        super().__init__()
        if workers<1 or batch_mb<=0 or split_length<0 or threshold<1 or precision not in ('fp64','twofold'):
            raise ValueError('Positive workers/batch_mb; precision=fp64/twofold')
        self.workers,self.batch_mb,self.precision,self.threshold=workers,batch_mb,precision,threshold
        self.fma=fma
        self.split_length=split_length
        self.guard_dc=guard_dc
        self.prefetch=prefetch
        self.workspace={}
        self.window_lock=threading.Lock()
        source=(HERE/'packed.cu').read_text()
        options=('--std=c++11','--fmad='+('true' if fma else 'false'))
        self.packed_kernels={name:self.cp.RawKernel(source,name,options=options)
            for name in ('packed_warp','packed_block','twofold_warp','twofold_block','split_partials','finish_partials')}

    def rows(self, n, plan, overlap, window, beta, order, pool, indices):
        lengths=np.asarray([plan[3][j] for j in indices],dtype=np.int64)
        offsets=np.r_[0,np.cumsum(lengths)]
        cr=np.empty(int(offsets[-1]));ci=np.empty_like(cr)
        def one(k):
            j=indices[k]
            buffers=cr[offsets[k]:offsets[k+1]],ci[offsets[k]:offsets[k+1]]
            args=(n,int(plan[3][j]),plan[2][j],int(plan[4][j]),overlap,window,beta,order,buffers)
            if window is np.kaiser: return self.prepare_frequency(*args)
            # A callback may return a shared work buffer; keep its full use private.
            with self.window_lock: return self.prepare_frequency(*args)
        return list(pool.map(one,range(len(indices))))

    @staticmethod
    def frame(values, frequencies, outputs):
        psd=np.asarray(values,dtype=np.complex64)
        columns={'psd':psd.real}
        if 'nsd' in outputs: columns['nsd']=np.sqrt(psd).real
        result=pd.DataFrame(columns,index=frequencies)
        result.index.name='frequency'
        return result

    def dc_sensitive(self,x,order):
        if not self.guard_dc or order!=0: return False
        low,high=float(np.min(x)),float(np.max(x))
        return high-low<=1e-6*max(abs(low),abs(high))

    def __call__(self,data,sample_rate=1.,n_frequencies=1000,n_averages=100,
                 detrending_order=0,window_function=np.kaiser,overlap=None,
                 psll=200.,outputs=('psd','nsd'),profile=False):
        if detrending_order not in (None,0):
            from lpsd_fast import lpsd
            result=lpsd(data,sample_rate=sample_rate,n_frequencies=n_frequencies,n_averages=n_averages,
                detrending_order=detrending_order,window_function=window_function,overlap=overlap,psll=psll,
                outputs=outputs,workers=self.workers,kernel='auto')
            result.attrs['cuda_experiment']=dict(executed='CPU fallback',
                reason='Order 1+ not implemented on GPU',workers=self.workers,kernel='auto')
            return result
        if tuple(outputs) not in (('psd',),('psd','nsd')): raise ValueError('PSD/NSD only')
        self.synchronize()
        started=time.perf_counter()
        api._native()
        x=api._array(data)
        if self.dc_sensitive(x,detrending_order):
            from lpsd_fast import lpsd
            result=lpsd(x,sample_rate=sample_rate,n_frequencies=n_frequencies,n_averages=n_averages,
                detrending_order=0,window_function=window_function,overlap=overlap,psll=psll,
                outputs=outputs,workers=self.workers,kernel='auto')
            result.attrs['cuda_experiment']=dict(executed='CPU auto residual fallback',
                reason='Projected coefficients cannot reproduce reference residual rounding for DC-dominated input',
                guard='range <= 1e-6 * peak')
            return result
        window=api._resolve_window(window_function)
        overlap=_kaiser_rov(_kaiser_alpha(psll)) if overlap is None else overlap
        if not np.isfinite(overlap) or not 0<=overlap<1: raise ValueError('Invalid overlap')
        plan=api._frequency_plan(len(x),sample_rate,overlap,1,0,n_frequencies,n_averages)
        order=-1 if detrending_order is None else 0
        dx=self.to_device(x)
        # Allow space for input, metadata, readback and driver/FFT caches.
        limit=min(self.batch_mb*1024**2,max(0,(self.pool_limit-x.nbytes)*.7))
        batches=[]; current=[]; size=0
        for j,length in enumerate(plan[3]):
            cost=int(length)*16+int(plan[4][j])*24
            if current and size+cost>limit:
                batches.append(current); current=[]; size=0
            if cost>limit: raise MemoryError('One frequency exceeds the configured batch budget')
            current.append(j); size+=cost
        if current: batches.append(current)
        values=[]
        phases=dict(host_preparation_s=0.,pack_and_upload_s=0.,projection_readback_s=0.,cpu_finish_s=0.)
        phases['host_preparation_work_s']=0.
        with ThreadPoolExecutor(max_workers=self.workers) as pool, ThreadPoolExecutor(max_workers=1) as coordinator:
            def prepare(indices):
                t=time.perf_counter()
                rows=self.rows(len(x),plan,overlap,window,_kaiser_alpha(psll)*np.pi,order,pool,indices)
                return rows,time.perf_counter()-t
            future=coordinator.submit(prepare,batches[0]) if self.prefetch else None
            for b,indices in enumerate(batches):
                t=time.perf_counter()
                rows,work=future.result() if self.prefetch else prepare(indices)
                phases['host_preparation_s']+=time.perf_counter()-t
                phases['host_preparation_work_s']+=work
                if self.prefetch and b+1<len(batches): future=coordinator.submit(prepare,batches[b+1])
                t=time.perf_counter()
                batch=PackedBatch(self,rows,self.threshold,self.precision,self.workspace)
                self.synchronize()
                phases['pack_and_upload_s']+=time.perf_counter()-t
                t=time.perf_counter()
                batch.launch(dx,order)
                powers=batch.out.get()
                phases['projection_readback_s']+=time.perf_counter()-t
                t=time.perf_counter()
                values.extend(batch.finish(x,powers,pool,sample_rate,order))
                phases['cpu_finish_s']+=time.perf_counter()-t
                del batch, rows, powers
        self.synchronize()
        self.last_profile=dict(phases,wall_s=time.perf_counter()-started,batches=len(batches))
        result=self.frame(values,plan[0],outputs)
        result.attrs['cuda_experiment']=dict(executed='Packed '+self.precision+' GPU segments; CPU preparation/initial/aggregation',
            workers=self.workers,batch_mb=self.batch_mb,fma=self.fma,profile=self.last_profile if profile else None)
        return result


class ResidentPackedPlan:
    def __init__(self,gpu,n,n_frequencies=1000,n_averages=100):
        self.gpu=gpu
        self.closed=False
        self.n_frequencies,self.n_averages=n_frequencies,n_averages
        api._native()
        overlap=_kaiser_rov(_kaiser_alpha(200.))
        self.plan=api._frequency_plan(n,1.,overlap,1,0,n_frequencies,n_averages)
        estimate=n*8+int(np.sum(self.plan[3]))*16+int(np.sum(self.plan[4]))*24
        if estimate>gpu.pool_limit*.85:
            raise MemoryError('Resident plan exceeds available VRAM reserve')
        self.pool=ThreadPoolExecutor(max_workers=gpu.workers)
        started=time.perf_counter()
        rows=gpu.rows(n,self.plan,overlap,np.kaiser,_kaiser_alpha(200.)*np.pi,0,self.pool,range(len(self.plan[0])))
        self.host_preparation_s=time.perf_counter()-started
        t=time.perf_counter()
        self.batch=PackedBatch(gpu,rows,gpu.threshold,gpu.precision)
        self.dx=gpu.cp.empty(n,dtype=gpu.cp.float64)
        gpu.synchronize()
        self.upload_s=time.perf_counter()-t
        self.payload_bytes=self.batch.payload_bytes+self.dx.nbytes

    def __call__(self,x):
        if self.closed: raise RuntimeError('Plan is closed')
        x=api._array(x)
        if len(x)!=len(self.dx): raise ValueError('Input length differs from plan')
        if self.gpu.dc_sensitive(x,0):
            return self.gpu(x,n_frequencies=self.n_frequencies,n_averages=self.n_averages)
        self.dx.set(x)
        self.batch.launch(self.dx)
        values=self.batch.finish(x,self.batch.out.get(),self.pool,1.)
        return self.gpu.frame(values,self.plan[0],('psd','nsd'))

    def close(self):
        if not self.closed:
            self.pool.shutdown()
            self.batch=None
            self.dx=None
            self.closed=True

    def cpu_call(self,x):
        if self.closed: raise RuntimeError('Plan is closed')
        x=api._array(x)
        if self.gpu.dc_sensitive(x,0):
            return self.gpu(x,n_frequencies=self.n_frequencies,n_averages=self.n_averages)
        def one(row):
            cr,ci,starts,s2=row
            powers=np.empty(len(starts))
            self.gpu.bridge.bench_reference(api._pointer(x),api._pointer(cr),api._pointer(ci),
                len(cr),self.gpu.ip(starts),len(starts),0,api._pointer(powers))
            return 2*self.gpu.bridge.bench_aggregate(api._pointer(powers),len(powers))/s2
        return self.gpu.frame(list(self.pool.map(one,self.batch.rows)),self.plan[0],('psd','nsd'))
