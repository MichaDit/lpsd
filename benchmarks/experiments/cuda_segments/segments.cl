// SPDX-License-Identifier: GPL-3.0-or-later
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF
__kernel void block_segments(__global const double *x, __global const double *qr,
    __global const double *qi, __global const int *starts, int length,
    int count, int anchored, __global double *powers,
    __local double *real, __local double *imag)
{
    int segment=get_group_id(0), lane=get_local_id(0), width=get_local_size(0);
    if (segment>=count) return;
    int start=starts[segment];
    double anchor=anchored ? x[start] : 0.;
    double r=0.,im=0.;
    for (int j=lane;j<length;j+=width) {
        double v=x[start+j]-anchor;
        r+=qr[j]*v; im+=qi[j]*v;
    }
    real[lane]=r; imag[lane]=im;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int d=width/2;d;d/=2) {
        if (lane<d) {real[lane]+=real[lane+d];imag[lane]+=imag[lane+d];}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (!lane) powers[segment]=real[0]*real[0]+imag[0]*imag[0];
}
