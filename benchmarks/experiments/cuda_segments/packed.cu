// SPDX-License-Identifier: GPL-3.0-or-later
// Packed frequency batches: two launches and one power readback per batch.
// The twofold option uses explicit FP32 FMA residuals; it is experimental.
struct Twofold {
    float hi, lo;
    __device__ Twofold():hi(0),lo(0) {}
    __device__ void add_product(double a, double b) {
        float ah=(float)a, bh=(float)b;
        float al=(float)(a-(double)ah), bl=(float)(b-(double)bh);
        float p=ah*bh;
        float e=fmaf(ah,bh,-p);
        e+=ah*bl; e+=al*bh; e+=al*bl;
        float s=hi+p, v=s-hi;
        float t=((p-v)+(hi-(s-v)))+(lo+e);
        hi=s+t; lo=t-(hi-s);
    }
    __device__ double value() {return (double)hi+(double)lo;}
};

template<bool mixed, bool block>
__device__ void packed_projection(const double *x, const double *cr,
    const double *ci, const long long *offsets, const int *lengths,
    const int *frequency_ids, const int *starts, const int *ids,
    int size, int anchored, double *powers)
{
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    int k=block ? blockIdx.x : (blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(k>=size) return;
    int segment=ids[k], f=frequency_ids[segment], start=starts[segment];
    int length=lengths[f], step=block ? blockDim.x : 32;
    long long offset=offsets[f];
    double anchor=anchored ? x[start] : 0., r=0., im=0.;
    Twofold tr, ti;
    for(int j=block ? threadIdx.x : lane;j<length;j+=step) {
        double v=x[start+j]-anchor;
        if(mixed) {tr.add_product(cr[offset+j],v); ti.add_product(ci[offset+j],v);}
        else {r+=cr[offset+j]*v; im+=ci[offset+j]*v;}
    }
    if(mixed) {r=tr.value(); im=ti.value();}
    for(int d=16;d>0;d/=2) {
        r+=__shfl_down_sync(0xffffffff,r,d);
        im+=__shfl_down_sync(0xffffffff,im,d);
    }
    if(block) {
        __shared__ double real[32], imag[32];
        if(!lane) {real[warp]=r; imag[warp]=im;}
        __syncthreads();
        if(!warp) {
            r=lane<blockDim.x/32 ? real[lane] : 0.;
            im=lane<blockDim.x/32 ? imag[lane] : 0.;
            for(int d=16;d>0;d/=2) {
                r+=__shfl_down_sync(0xffffffff,r,d);
                im+=__shfl_down_sync(0xffffffff,im,d);
            }
            if(!lane) powers[segment]=r*r+im*im;
        }
    } else if(!lane) powers[segment]=r*r+im*im;
}

#define ENTRY(name, mixed, block) \
extern "C" __global__ void name(const double *x,const double *cr,const double *ci, \
    const long long *o,const int *l,const int *f,const int *s,const int *ids, \
    int size,int anchored,double *powers) { \
    packed_projection<mixed,block>(x,cr,ci,o,l,f,s,ids,size,anchored,powers); }
ENTRY(packed_warp,false,false)
ENTRY(packed_block,false,true)
ENTRY(twofold_warp,true,false)
ENTRY(twofold_block,true,true)

extern "C" __global__ void split_partials(const double *x,const double *cr,
    const double *ci,const long long *offsets,const int *lengths,const int *fids,
    const int *starts,const int *segments,const int *parts,int size,int chunk,
    int anchored,double *rr,double *ii)
{
    int k=blockIdx.x;
    if(k>=size) return;
    int segment=segments[k], f=fids[segment], start=starts[segment];
    int begin=parts[k]*chunk, end=min(lengths[f],begin+chunk);
    long long offset=offsets[f];
    double anchor=anchored ? x[start] : 0.,r=0.,im=0.;
    for(int j=begin+threadIdx.x;j<end;j+=blockDim.x) {
        double v=x[start+j]-anchor;
        r+=cr[offset+j]*v;im+=ci[offset+j]*v;
    }
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int d=16;d>0;d/=2) {
        r+=__shfl_down_sync(0xffffffff,r,d);im+=__shfl_down_sync(0xffffffff,im,d);
    }
    __shared__ double real[32],imag[32];
    if(!lane) {real[warp]=r;imag[warp]=im;}
    __syncthreads();
    if(!warp) {
        r=lane<blockDim.x/32 ? real[lane] : 0.;im=lane<blockDim.x/32 ? imag[lane] : 0.;
        for(int d=16;d>0;d/=2) {
            r+=__shfl_down_sync(0xffffffff,r,d);im+=__shfl_down_sync(0xffffffff,im,d);
        }
        if(!lane) {rr[k]=r;ii[k]=im;}
    }
}

extern "C" __global__ void finish_partials(const double *rr,const double *ii,
    const int *ids,const long long *cuts,int count,double *powers)
{
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=count) return;
    double r=0.,im=0.;
    for(long long k=cuts[j];k<cuts[j+1];++k) {r+=rr[k];im+=ii[k];}
    powers[ids[j]]=r*r+im*im;
}
