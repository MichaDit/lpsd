// SPDX-License-Identifier: GPL-3.0-or-later
// Stable graph-capturable periodogram preprocessing and log aggregation.
__device__ double reduce(double v) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int d=16;d>0;d/=2) v+=__shfl_down_sync(0xffffffff,v,d);
    __shared__ double sums[32];
    if(!lane) sums[warp]=v;
    __syncthreads();
    if(!warp) {
        v=lane<blockDim.x/32 ? sums[lane] : 0.;
        for(int d=16;d>0;d/=2) v+=__shfl_down_sync(0xffffffff,v,d);
    }
    return v;
}
extern "C" __global__ void anchored_parts(const double *x,int n,double *parts) {
    int begin=blockIdx.x*8192,end=min(n,begin+8192);
    double sum=0.,anchor=x[0];
    for(int j=begin+threadIdx.x;j<end;j+=blockDim.x) sum+=x[j]-anchor;
    sum=reduce(sum);
    if(!threadIdx.x) parts[blockIdx.x]=sum;
}
extern "C" __global__ void anchored_mean(const double *parts,int count,int n,double *mean) {
    double sum=0.;
    for(int j=threadIdx.x;j<count;j+=blockDim.x) sum+=parts[j];
    sum=reduce(sum);
    if(!threadIdx.x) mean[0]=sum/n;
}
extern "C" __global__ void window_input(const double *x,const double *w,
    const double *mean,int n,double *out) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<n) out[j]=((x[j]-x[0])-mean[0])*w[j];
}
extern "C" __global__ void logarithmic_power(const double2 *fft,
    const long long *cuts,int groups,int n,double s2,float *out) {
    int group=blockIdx.x;
    if(group>=groups) return;
    long long begin=cuts[group],end=cuts[group+1];
    double sum=0.;
    for(long long j=begin+threadIdx.x;j<end;j+=blockDim.x) {
        double2 z=fft[j];
        double p=z.x*z.x+z.y*z.y;
        if(n%2==0 && j==n/2) p*=.5;
        sum+=p;
    }
    sum=reduce(sum);
    if(!threadIdx.x) out[group]=(float)(2*sum/s2/(end-begin));
}
