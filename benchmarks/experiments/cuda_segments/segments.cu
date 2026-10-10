// SPDX-License-Identifier: GPL-3.0-or-later
// FP64 input, coefficients, accumulation and powers. FMA disabled by NVRTC.
extern "C" __global__ void warp_segments(const double *x, const double *qr,
    const double *qi, const int *starts, int length, int count, int anchored,
    double *powers)
{
    int lane=threadIdx.x%32;
    int segment=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if (segment >= count) return;
    int start=starts[segment];
    double anchor=anchored ? x[start] : 0.;
    double r=0., im=0.;
    for (int j=lane; j<length; j+=32) {
        double v=x[start+j]-anchor;
        r+=qr[j]*v; im+=qi[j]*v;
    }
    for (int d=16; d>0; d/=2) {
        r+=__shfl_down_sync(0xffffffff,r,d);
        im+=__shfl_down_sync(0xffffffff,im,d);
    }
    if (!lane) powers[segment]=r*r+im*im;
}

extern "C" __global__ void block_segments(const double *x, const double *qr,
    const double *qi, const int *starts, int length, int count, int anchored,
    double *powers)
{
    int segment=blockIdx.x;
    if (segment >= count) return;
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    int start=starts[segment];
    double anchor=anchored ? x[start] : 0.;
    double r=0., im=0.;
    for (int j=threadIdx.x; j<length; j+=blockDim.x) {
        double v=x[start+j]-anchor;
        r+=qr[j]*v; im+=qi[j]*v;
    }
    for (int d=16; d>0; d/=2) {
        r+=__shfl_down_sync(0xffffffff,r,d);
        im+=__shfl_down_sync(0xffffffff,im,d);
    }
    __shared__ double real[32], imag[32];
    if (!lane) {real[warp]=r; imag[warp]=im;}
    __syncthreads();
    if (!warp) {
        r=lane<blockDim.x/32 ? real[lane] : 0.;
        im=lane<blockDim.x/32 ? imag[lane] : 0.;
        for (int d=16; d>0; d/=2) {
            r+=__shfl_down_sync(0xffffffff,r,d);
            im+=__shfl_down_sync(0xffffffff,im,d);
        }
        if (!lane) powers[segment]=r*r+im*im;
    }
}
