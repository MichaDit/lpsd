/* SPDX-License-Identifier: GPL-3.0-or-later
 * Benchmark-only bridge to the unchanged native projector and initial dots. */
#include <stdint.h>
#include "../../../lpsd_fast/_native/fast_dft.c"

int bench_prepare(double *cr, double *ci, long length, int order)
{
    if (order == 0) prepare_projected_coefficients(cr, ci, length, order, cr, ci);
    return order == 0 || order == -1 ? 0 : 1;
}

long bench_starts(int32_t *out, long n, long length, double overlap)
{
    const double ovfact = 1.0 / (1.0 - overlap / 100.0);
    const long count = (long)floor(((double)(n-length)*ovfact)/length + 1.0 + 0.5);
    if (count < 1 || count > n-length+1) return -1;
    const double shift = count == 1 ? 1.0 : (double)(n-length)/(double)(count-1);
    double start = 0.;
    for (long i=0; i<count; ++i) {
        out[i] = (int32_t)floor(start+0.5);
        if (out[i] < 0 || out[i] > n-length) return -1;
        start += shift;
    }
    return count;
}

void bench_initial(const double *x, const double *cr, const double *ci,
                   long length, const int32_t *starts, long count,
                   int order, double *powers)
{
    /* Preserve the CPU's first group and its cancellation-sensitive ii=1. */
    double r[8], im[8];
    long done=0;
    if (order == 0 && length >= 128 && (length < 256 || length >= 1024) &&
        use_eight_segment_batch() && count >= 8) {
        dot_psd_eight_anchored_simd(x+starts[0],x+starts[1],x+starts[2],x+starts[3],
                                  x+starts[4],x+starts[5],x+starts[6],x+starts[7],
                                  cr,ci,length,r,im);
        done=8;
    } else if (order == 0 && length >= 2048 && count >= 4) {
        dot_psd_four_anchored_simd(x+starts[0],x+starts[1],x+starts[2],x+starts[3],
                                 cr,ci,length,r,im);
        done=4;
    }
    for (long i=done; i<8 && i<count; ++i) {
        if (order == 0) dot_psd_anchored_simd(x+starts[i],cr,ci,length,r+i,im+i);
        else dot_psd_simd(x+starts[i],cr,ci,length,r+i,im+i);
    }
    for (long i=0; i<8 && i<count; ++i) powers[i]=r[i]*r[i]+im[i]*im[i];
}

double bench_aggregate(const double *powers, long count)
{
    double mean = powers[0];
    for (long i=1; i<count; ++i) mean += (powers[i]-mean)/i;
    return mean;
}

void bench_reference(const double *x, const double *cr, const double *ci,
                     long length, const int32_t *starts, long count,
                     int order, double *powers)
{
    for (long i=0; i<count; i+=8)
        bench_initial(x,cr,ci,length,starts+i,count-i,order,powers+i);
}
