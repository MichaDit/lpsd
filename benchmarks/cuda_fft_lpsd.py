# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual cuFFT convolution of the same LPSD segment projection.

A reversed projected coefficient vector is convolved with the signal; only
the exact existing segment-start samples are gathered. Fractional bins are
preserved. This is distinct from replacing LPSD with a full periodogram.
FFT arithmetic has its own cancellation behavior and requires an audit.
"""
import numpy as np
from scipy.fft import next_fast_len
from benchmarks.cuda_lpsd import CUDAEstimator


class FFTCorrelationEstimator(CUDAEstimator):
    def __call__(self,*args,**kwargs):
        self._fft_input=None
        self._fft_length=None
        try: return super().__call__(*args,**kwargs)
        finally:
            self._fft_input=None

    def project(self,dx,cr,ci,starts,order,**kwargs):
        cp=self.cp
        length=len(cr)
        fft_length=next_fast_len(len(dx)+length-1,real=True)
        if fft_length!=getattr(self,'_fft_length',None) or getattr(self,'_fft_input',None) is None:
            centered=dx-dx[0] if order==0 else dx
            self._fft_input=cp.fft.rfft(centered,n=fft_length)
            self._fft_length=fft_length
        coeff=cp.asarray(np.stack((cr[::-1],ci[::-1])))
        transformed=cp.fft.rfft(coeff,n=fft_length,axis=-1)
        transformed*=self._fft_input[None,:]
        convolved=cp.fft.irfft(transformed,n=fft_length,axis=-1)
        selected=convolved[:,cp.asarray(starts,dtype=cp.int64)+length-1]
        if order==0:
            # Restore exactly the local anchoring of each direct dot.
            sums=cp.asarray(np.asarray([np.sum(cr,dtype=np.longdouble),np.sum(ci,dtype=np.longdouble)],dtype=np.float64))
            selected-=(dx[cp.asarray(starts)]-dx[0])[None,:]*sums[:,None]
        return cp.sum(selected*selected,axis=0)
