
/* Isolated screening prototype. Not a supported API or a shipped kernel. */
#include "fast_dft.c"

#if PROBE_FMA
#define PROBE_ADD_PRODUCT(a,b,c) fma((a),(b),(c))
#else
#define PROBE_ADD_PRODUCT(a,b,c) ((a)*(b)+(c))
#endif

static int prepare_folded(const double *window, long int length, double m,
                          bool periodic, double *fr, double *fi,
                          double *center_r, double *mean_i)
{
    const double pi = 3.141592653589793238462643383279502884;
    const double omega = (2.0 * pi) * m / (double)length;
    const double center = .5 * (double)(periodic ? length : length - 1);
    const long int pairs = periodic ? (length - 1) / 2 : length / 2;
    const bool has_center = periodic ? length % 2 == 0 : length % 2 != 0;
    long double sum_r = 0.0L;
    for (long int j = 0; j < pairs; ++j) {
        const long int input_j = j + (periodic ? 1 : 0);
        const double phase = omega * ((double)input_j - center);
        fr[j] = window[input_j] * cos(phase);
        fi[j] = window[input_j] * sin(phase);
        sum_r += 2.0L * (long double)fr[j];
    }
    if (has_center) sum_r += window[length / 2];
    if (periodic) sum_r += (long double)window[0] * cos(-omega * center);
    const long double mr = sum_r / (long double)length;
    *mean_i = periodic ? window[0] * sin(-omega * center) / (double)length : 0.0;
    *center_r = has_center ? (double)((long double)window[length / 2] - mr) : 0.0;
    for (long int j = 0; j < pairs; ++j) fr[j] = (double)((long double)fr[j] - mr);
    return isfinite(*center_r) && isfinite(*mean_i) ? 0 : 1;
}
static LPSD_ALWAYS_INLINE void folded_symmetric_1(const double **x, long int length,
                  const double *fr, const double *fi, double center_r,
                  double mean_i, double *rr, double *ri)
{
    const long int pairs = length / 2;
    const bool has_center = length % 2 != 0;
    const double *x0 = x[0];
    const double a0 = x0[0];
    double r0 = 0.0, i0 = 0.0;
    #pragma omp simd reduction(+:r0,i0)
    for (long int j = 0; j < pairs; ++j) {
        const long int left = j;
        const long int right = length - 1 - left;
        const double real = fr[j], imag = fi[j];
        const double vl0 = x0[left] - a0;
        const double vr0 = x0[right] - a0;
        const double vp0 = vl0 + vr0;
        const double vm0 = vl0 - vr0;
        r0 = PROBE_ADD_PRODUCT(real, vp0, r0);
        i0 = PROBE_ADD_PRODUCT(imag, vm0, i0);
    }
    if (has_center) {
        const double vc0 = x0[length / 2] - a0;
        r0 = PROBE_ADD_PRODUCT(center_r, vc0, r0);
    }
    rr[0] = r0;
    ri[0] = i0;
}
static LPSD_ALWAYS_INLINE void folded_symmetric_4(const double **x, long int length,
                  const double *fr, const double *fi, double center_r,
                  double mean_i, double *rr, double *ri)
{
    const long int pairs = length / 2;
    const bool has_center = length % 2 != 0;
    const double *x0 = x[0];
    const double a0 = x0[0];
    double r0 = 0.0, i0 = 0.0;
    const double *x1 = x[1];
    const double a1 = x1[0];
    double r1 = 0.0, i1 = 0.0;
    const double *x2 = x[2];
    const double a2 = x2[0];
    double r2 = 0.0, i2 = 0.0;
    const double *x3 = x[3];
    const double a3 = x3[0];
    double r3 = 0.0, i3 = 0.0;
    #pragma omp simd reduction(+:r0,i0,r1,i1,r2,i2,r3,i3)
    for (long int j = 0; j < pairs; ++j) {
        const long int left = j;
        const long int right = length - 1 - left;
        const double real = fr[j], imag = fi[j];
        const double vl0 = x0[left] - a0;
        const double vr0 = x0[right] - a0;
        const double vp0 = vl0 + vr0;
        const double vm0 = vl0 - vr0;
        r0 = PROBE_ADD_PRODUCT(real, vp0, r0);
        i0 = PROBE_ADD_PRODUCT(imag, vm0, i0);
        const double vl1 = x1[left] - a1;
        const double vr1 = x1[right] - a1;
        const double vp1 = vl1 + vr1;
        const double vm1 = vl1 - vr1;
        r1 = PROBE_ADD_PRODUCT(real, vp1, r1);
        i1 = PROBE_ADD_PRODUCT(imag, vm1, i1);
        const double vl2 = x2[left] - a2;
        const double vr2 = x2[right] - a2;
        const double vp2 = vl2 + vr2;
        const double vm2 = vl2 - vr2;
        r2 = PROBE_ADD_PRODUCT(real, vp2, r2);
        i2 = PROBE_ADD_PRODUCT(imag, vm2, i2);
        const double vl3 = x3[left] - a3;
        const double vr3 = x3[right] - a3;
        const double vp3 = vl3 + vr3;
        const double vm3 = vl3 - vr3;
        r3 = PROBE_ADD_PRODUCT(real, vp3, r3);
        i3 = PROBE_ADD_PRODUCT(imag, vm3, i3);
    }
    if (has_center) {
        const double vc0 = x0[length / 2] - a0;
        r0 = PROBE_ADD_PRODUCT(center_r, vc0, r0);
        const double vc1 = x1[length / 2] - a1;
        r1 = PROBE_ADD_PRODUCT(center_r, vc1, r1);
        const double vc2 = x2[length / 2] - a2;
        r2 = PROBE_ADD_PRODUCT(center_r, vc2, r2);
        const double vc3 = x3[length / 2] - a3;
        r3 = PROBE_ADD_PRODUCT(center_r, vc3, r3);
    }
    rr[0] = r0;
    ri[0] = i0;
    rr[1] = r1;
    ri[1] = i1;
    rr[2] = r2;
    ri[2] = i2;
    rr[3] = r3;
    ri[3] = i3;
}
static LPSD_ALWAYS_INLINE void folded_symmetric_8(const double **x, long int length,
                  const double *fr, const double *fi, double center_r,
                  double mean_i, double *rr, double *ri)
{
    const long int pairs = length / 2;
    const bool has_center = length % 2 != 0;
    const double *x0 = x[0];
    const double a0 = x0[0];
    double r0 = 0.0, i0 = 0.0;
    const double *x1 = x[1];
    const double a1 = x1[0];
    double r1 = 0.0, i1 = 0.0;
    const double *x2 = x[2];
    const double a2 = x2[0];
    double r2 = 0.0, i2 = 0.0;
    const double *x3 = x[3];
    const double a3 = x3[0];
    double r3 = 0.0, i3 = 0.0;
    const double *x4 = x[4];
    const double a4 = x4[0];
    double r4 = 0.0, i4 = 0.0;
    const double *x5 = x[5];
    const double a5 = x5[0];
    double r5 = 0.0, i5 = 0.0;
    const double *x6 = x[6];
    const double a6 = x6[0];
    double r6 = 0.0, i6 = 0.0;
    const double *x7 = x[7];
    const double a7 = x7[0];
    double r7 = 0.0, i7 = 0.0;
    #pragma omp simd reduction(+:r0,i0,r1,i1,r2,i2,r3,i3,r4,i4,r5,i5,r6,i6,r7,i7)
    for (long int j = 0; j < pairs; ++j) {
        const long int left = j;
        const long int right = length - 1 - left;
        const double real = fr[j], imag = fi[j];
        const double vl0 = x0[left] - a0;
        const double vr0 = x0[right] - a0;
        const double vp0 = vl0 + vr0;
        const double vm0 = vl0 - vr0;
        r0 = PROBE_ADD_PRODUCT(real, vp0, r0);
        i0 = PROBE_ADD_PRODUCT(imag, vm0, i0);
        const double vl1 = x1[left] - a1;
        const double vr1 = x1[right] - a1;
        const double vp1 = vl1 + vr1;
        const double vm1 = vl1 - vr1;
        r1 = PROBE_ADD_PRODUCT(real, vp1, r1);
        i1 = PROBE_ADD_PRODUCT(imag, vm1, i1);
        const double vl2 = x2[left] - a2;
        const double vr2 = x2[right] - a2;
        const double vp2 = vl2 + vr2;
        const double vm2 = vl2 - vr2;
        r2 = PROBE_ADD_PRODUCT(real, vp2, r2);
        i2 = PROBE_ADD_PRODUCT(imag, vm2, i2);
        const double vl3 = x3[left] - a3;
        const double vr3 = x3[right] - a3;
        const double vp3 = vl3 + vr3;
        const double vm3 = vl3 - vr3;
        r3 = PROBE_ADD_PRODUCT(real, vp3, r3);
        i3 = PROBE_ADD_PRODUCT(imag, vm3, i3);
        const double vl4 = x4[left] - a4;
        const double vr4 = x4[right] - a4;
        const double vp4 = vl4 + vr4;
        const double vm4 = vl4 - vr4;
        r4 = PROBE_ADD_PRODUCT(real, vp4, r4);
        i4 = PROBE_ADD_PRODUCT(imag, vm4, i4);
        const double vl5 = x5[left] - a5;
        const double vr5 = x5[right] - a5;
        const double vp5 = vl5 + vr5;
        const double vm5 = vl5 - vr5;
        r5 = PROBE_ADD_PRODUCT(real, vp5, r5);
        i5 = PROBE_ADD_PRODUCT(imag, vm5, i5);
        const double vl6 = x6[left] - a6;
        const double vr6 = x6[right] - a6;
        const double vp6 = vl6 + vr6;
        const double vm6 = vl6 - vr6;
        r6 = PROBE_ADD_PRODUCT(real, vp6, r6);
        i6 = PROBE_ADD_PRODUCT(imag, vm6, i6);
        const double vl7 = x7[left] - a7;
        const double vr7 = x7[right] - a7;
        const double vp7 = vl7 + vr7;
        const double vm7 = vl7 - vr7;
        r7 = PROBE_ADD_PRODUCT(real, vp7, r7);
        i7 = PROBE_ADD_PRODUCT(imag, vm7, i7);
    }
    if (has_center) {
        const double vc0 = x0[length / 2] - a0;
        r0 = PROBE_ADD_PRODUCT(center_r, vc0, r0);
        const double vc1 = x1[length / 2] - a1;
        r1 = PROBE_ADD_PRODUCT(center_r, vc1, r1);
        const double vc2 = x2[length / 2] - a2;
        r2 = PROBE_ADD_PRODUCT(center_r, vc2, r2);
        const double vc3 = x3[length / 2] - a3;
        r3 = PROBE_ADD_PRODUCT(center_r, vc3, r3);
        const double vc4 = x4[length / 2] - a4;
        r4 = PROBE_ADD_PRODUCT(center_r, vc4, r4);
        const double vc5 = x5[length / 2] - a5;
        r5 = PROBE_ADD_PRODUCT(center_r, vc5, r5);
        const double vc6 = x6[length / 2] - a6;
        r6 = PROBE_ADD_PRODUCT(center_r, vc6, r6);
        const double vc7 = x7[length / 2] - a7;
        r7 = PROBE_ADD_PRODUCT(center_r, vc7, r7);
    }
    rr[0] = r0;
    ri[0] = i0;
    rr[1] = r1;
    ri[1] = i1;
    rr[2] = r2;
    ri[2] = i2;
    rr[3] = r3;
    ri[3] = i3;
    rr[4] = r4;
    ri[4] = i4;
    rr[5] = r5;
    ri[5] = i5;
    rr[6] = r6;
    ri[6] = i6;
    rr[7] = r7;
    ri[7] = i7;
}
static LPSD_ALWAYS_INLINE void folded_periodic_1(const double **x, long int length,
                  const double *fr, const double *fi, double center_r,
                  double mean_i, double *rr, double *ri)
{
    const long int pairs = (length - 1) / 2;
    const bool has_center = length % 2 == 0;
    const double *x0 = x[0];
    const double a0 = x0[0];
    double r0 = 0.0, i0 = 0.0, s0 = 0.0;
    #pragma omp simd reduction(+:r0,i0,s0)
    for (long int j = 0; j < pairs; ++j) {
        const long int left = j + 1;
        const long int right = length - left;
        const double real = fr[j], imag = fi[j];
        const double vl0 = x0[left] - a0;
        const double vr0 = x0[right] - a0;
        const double vp0 = vl0 + vr0;
        const double vm0 = vl0 - vr0;
        r0 = PROBE_ADD_PRODUCT(real, vp0, r0);
        i0 = PROBE_ADD_PRODUCT(imag, vm0, i0);
        s0 += vp0;
    }
    if (has_center) {
        const double vc0 = x0[length / 2] - a0;
        r0 = PROBE_ADD_PRODUCT(center_r, vc0, r0);
        s0 += vc0;
    }
    rr[0] = r0;
    ri[0] = i0 - mean_i * s0;
}
static LPSD_ALWAYS_INLINE void folded_periodic_4(const double **x, long int length,
                  const double *fr, const double *fi, double center_r,
                  double mean_i, double *rr, double *ri)
{
    const long int pairs = (length - 1) / 2;
    const bool has_center = length % 2 == 0;
    const double *x0 = x[0];
    const double a0 = x0[0];
    double r0 = 0.0, i0 = 0.0, s0 = 0.0;
    const double *x1 = x[1];
    const double a1 = x1[0];
    double r1 = 0.0, i1 = 0.0, s1 = 0.0;
    const double *x2 = x[2];
    const double a2 = x2[0];
    double r2 = 0.0, i2 = 0.0, s2 = 0.0;
    const double *x3 = x[3];
    const double a3 = x3[0];
    double r3 = 0.0, i3 = 0.0, s3 = 0.0;
    #pragma omp simd reduction(+:r0,i0,s0,r1,i1,s1,r2,i2,s2,r3,i3,s3)
    for (long int j = 0; j < pairs; ++j) {
        const long int left = j + 1;
        const long int right = length - left;
        const double real = fr[j], imag = fi[j];
        const double vl0 = x0[left] - a0;
        const double vr0 = x0[right] - a0;
        const double vp0 = vl0 + vr0;
        const double vm0 = vl0 - vr0;
        r0 = PROBE_ADD_PRODUCT(real, vp0, r0);
        i0 = PROBE_ADD_PRODUCT(imag, vm0, i0);
        s0 += vp0;
        const double vl1 = x1[left] - a1;
        const double vr1 = x1[right] - a1;
        const double vp1 = vl1 + vr1;
        const double vm1 = vl1 - vr1;
        r1 = PROBE_ADD_PRODUCT(real, vp1, r1);
        i1 = PROBE_ADD_PRODUCT(imag, vm1, i1);
        s1 += vp1;
        const double vl2 = x2[left] - a2;
        const double vr2 = x2[right] - a2;
        const double vp2 = vl2 + vr2;
        const double vm2 = vl2 - vr2;
        r2 = PROBE_ADD_PRODUCT(real, vp2, r2);
        i2 = PROBE_ADD_PRODUCT(imag, vm2, i2);
        s2 += vp2;
        const double vl3 = x3[left] - a3;
        const double vr3 = x3[right] - a3;
        const double vp3 = vl3 + vr3;
        const double vm3 = vl3 - vr3;
        r3 = PROBE_ADD_PRODUCT(real, vp3, r3);
        i3 = PROBE_ADD_PRODUCT(imag, vm3, i3);
        s3 += vp3;
    }
    if (has_center) {
        const double vc0 = x0[length / 2] - a0;
        r0 = PROBE_ADD_PRODUCT(center_r, vc0, r0);
        s0 += vc0;
        const double vc1 = x1[length / 2] - a1;
        r1 = PROBE_ADD_PRODUCT(center_r, vc1, r1);
        s1 += vc1;
        const double vc2 = x2[length / 2] - a2;
        r2 = PROBE_ADD_PRODUCT(center_r, vc2, r2);
        s2 += vc2;
        const double vc3 = x3[length / 2] - a3;
        r3 = PROBE_ADD_PRODUCT(center_r, vc3, r3);
        s3 += vc3;
    }
    rr[0] = r0;
    ri[0] = i0 - mean_i * s0;
    rr[1] = r1;
    ri[1] = i1 - mean_i * s1;
    rr[2] = r2;
    ri[2] = i2 - mean_i * s2;
    rr[3] = r3;
    ri[3] = i3 - mean_i * s3;
}
static LPSD_ALWAYS_INLINE void folded_periodic_8(const double **x, long int length,
                  const double *fr, const double *fi, double center_r,
                  double mean_i, double *rr, double *ri)
{
    const long int pairs = (length - 1) / 2;
    const bool has_center = length % 2 == 0;
    const double *x0 = x[0];
    const double a0 = x0[0];
    double r0 = 0.0, i0 = 0.0, s0 = 0.0;
    const double *x1 = x[1];
    const double a1 = x1[0];
    double r1 = 0.0, i1 = 0.0, s1 = 0.0;
    const double *x2 = x[2];
    const double a2 = x2[0];
    double r2 = 0.0, i2 = 0.0, s2 = 0.0;
    const double *x3 = x[3];
    const double a3 = x3[0];
    double r3 = 0.0, i3 = 0.0, s3 = 0.0;
    const double *x4 = x[4];
    const double a4 = x4[0];
    double r4 = 0.0, i4 = 0.0, s4 = 0.0;
    const double *x5 = x[5];
    const double a5 = x5[0];
    double r5 = 0.0, i5 = 0.0, s5 = 0.0;
    const double *x6 = x[6];
    const double a6 = x6[0];
    double r6 = 0.0, i6 = 0.0, s6 = 0.0;
    const double *x7 = x[7];
    const double a7 = x7[0];
    double r7 = 0.0, i7 = 0.0, s7 = 0.0;
    #pragma omp simd reduction(+:r0,i0,s0,r1,i1,s1,r2,i2,s2,r3,i3,s3,r4,i4,s4,r5,i5,s5,r6,i6,s6,r7,i7,s7)
    for (long int j = 0; j < pairs; ++j) {
        const long int left = j + 1;
        const long int right = length - left;
        const double real = fr[j], imag = fi[j];
        const double vl0 = x0[left] - a0;
        const double vr0 = x0[right] - a0;
        const double vp0 = vl0 + vr0;
        const double vm0 = vl0 - vr0;
        r0 = PROBE_ADD_PRODUCT(real, vp0, r0);
        i0 = PROBE_ADD_PRODUCT(imag, vm0, i0);
        s0 += vp0;
        const double vl1 = x1[left] - a1;
        const double vr1 = x1[right] - a1;
        const double vp1 = vl1 + vr1;
        const double vm1 = vl1 - vr1;
        r1 = PROBE_ADD_PRODUCT(real, vp1, r1);
        i1 = PROBE_ADD_PRODUCT(imag, vm1, i1);
        s1 += vp1;
        const double vl2 = x2[left] - a2;
        const double vr2 = x2[right] - a2;
        const double vp2 = vl2 + vr2;
        const double vm2 = vl2 - vr2;
        r2 = PROBE_ADD_PRODUCT(real, vp2, r2);
        i2 = PROBE_ADD_PRODUCT(imag, vm2, i2);
        s2 += vp2;
        const double vl3 = x3[left] - a3;
        const double vr3 = x3[right] - a3;
        const double vp3 = vl3 + vr3;
        const double vm3 = vl3 - vr3;
        r3 = PROBE_ADD_PRODUCT(real, vp3, r3);
        i3 = PROBE_ADD_PRODUCT(imag, vm3, i3);
        s3 += vp3;
        const double vl4 = x4[left] - a4;
        const double vr4 = x4[right] - a4;
        const double vp4 = vl4 + vr4;
        const double vm4 = vl4 - vr4;
        r4 = PROBE_ADD_PRODUCT(real, vp4, r4);
        i4 = PROBE_ADD_PRODUCT(imag, vm4, i4);
        s4 += vp4;
        const double vl5 = x5[left] - a5;
        const double vr5 = x5[right] - a5;
        const double vp5 = vl5 + vr5;
        const double vm5 = vl5 - vr5;
        r5 = PROBE_ADD_PRODUCT(real, vp5, r5);
        i5 = PROBE_ADD_PRODUCT(imag, vm5, i5);
        s5 += vp5;
        const double vl6 = x6[left] - a6;
        const double vr6 = x6[right] - a6;
        const double vp6 = vl6 + vr6;
        const double vm6 = vl6 - vr6;
        r6 = PROBE_ADD_PRODUCT(real, vp6, r6);
        i6 = PROBE_ADD_PRODUCT(imag, vm6, i6);
        s6 += vp6;
        const double vl7 = x7[left] - a7;
        const double vr7 = x7[right] - a7;
        const double vp7 = vl7 + vr7;
        const double vm7 = vl7 - vr7;
        r7 = PROBE_ADD_PRODUCT(real, vp7, r7);
        i7 = PROBE_ADD_PRODUCT(imag, vm7, i7);
        s7 += vp7;
    }
    if (has_center) {
        const double vc0 = x0[length / 2] - a0;
        r0 = PROBE_ADD_PRODUCT(center_r, vc0, r0);
        s0 += vc0;
        const double vc1 = x1[length / 2] - a1;
        r1 = PROBE_ADD_PRODUCT(center_r, vc1, r1);
        s1 += vc1;
        const double vc2 = x2[length / 2] - a2;
        r2 = PROBE_ADD_PRODUCT(center_r, vc2, r2);
        s2 += vc2;
        const double vc3 = x3[length / 2] - a3;
        r3 = PROBE_ADD_PRODUCT(center_r, vc3, r3);
        s3 += vc3;
        const double vc4 = x4[length / 2] - a4;
        r4 = PROBE_ADD_PRODUCT(center_r, vc4, r4);
        s4 += vc4;
        const double vc5 = x5[length / 2] - a5;
        r5 = PROBE_ADD_PRODUCT(center_r, vc5, r5);
        s5 += vc5;
        const double vc6 = x6[length / 2] - a6;
        r6 = PROBE_ADD_PRODUCT(center_r, vc6, r6);
        s6 += vc6;
        const double vc7 = x7[length / 2] - a7;
        r7 = PROBE_ADD_PRODUCT(center_r, vc7, r7);
        s7 += vc7;
    }
    rr[0] = r0;
    ri[0] = i0 - mean_i * s0;
    rr[1] = r1;
    ri[1] = i1 - mean_i * s1;
    rr[2] = r2;
    ri[2] = i2 - mean_i * s2;
    rr[3] = r3;
    ri[3] = i3 - mean_i * s3;
    rr[4] = r4;
    ri[4] = i4 - mean_i * s4;
    rr[5] = r5;
    ri[5] = i5 - mean_i * s5;
    rr[6] = r6;
    ri[6] = i6 - mean_i * s6;
    rr[7] = r7;
    ri[7] = i7 - mean_i * s7;
}

static int probe_folded_impl(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
                     long int *Navs,
                     const double *x1data, const double *x2data,
                     long int nData, long int segLen,
                     double *Cr, double *Ci,
                     double olap, int order, bool csd, int mode,
                     bool statistics, const double *window, bool periodic,
                     double frequency_bin, int requested_width, long int *folded_count,
                     double *preparation_seconds, double *segments_seconds)
{
    const bool profile = preparation_seconds != NULL && segments_seconds != NULL;
    const double preparation_started = profile ? monotonic_seconds() : 0.0;
    *folded_count = 0;
    if (mode < 2 || order != 0 || csd || segLen < 4) {
        return fast_dft_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs,
                             x1data, x2data, nData, segLen, Cr, Ci,
                             olap, order, csd, mode, statistics, true,
                             NULL, NULL, NULL, NULL);
    }
    const double ovfact = 1.0 / (1.0 - olap / 100.0);
    const double davg = ((double)(nData - segLen) * ovfact) / segLen + 1.0;
    const long int navg = (long int)floor(davg + 0.5);
    if (navg < 1 || navg > nData - segLen + 1) return 2;
    const double shift = navg == 1 ? 1.0 : (double)(nData - segLen) / (double)(navg - 1);
    double *qr = malloc((size_t)segLen * sizeof(double));
    double *qi = malloc((size_t)segLen * sizeof(double));
    double *fr = malloc((size_t)((segLen + 1) / 2) * sizeof(double));
    double *fi = malloc((size_t)((segLen + 1) / 2) * sizeof(double));
    if (!qr || !qi || !fr || !fi) { free(qr); free(qi); free(fr); free(fi); return 3; }
    if (mode != 3 || !prepare_projected_coefficients_dd(Cr, Ci, segLen, qr, qi)) {
        prepare_projected_coefficients(Cr, Ci, segLen, 0, qr, qi);
    }
    double center_r, mean_i;
    int fallback = prepare_folded(window, segLen, frequency_bin, periodic,
                                   fr, fi, &center_r, &mean_i);
    *Pr_r = *Pr_i = NAN;
    if (Vr_r) *Vr_r = NAN;
    if (Vr_i) *Vr_i = NAN;
    *Navs = 0;
    const bool baseline_eight = segLen >= 128 && (segLen < 256 || segLen >= 1024)
                                && use_eight_segment_batch();
    /* Preserve exactly the original first batch, or the first two singleton
     * dots, so the cancellation-prone first mean reset is unchanged. */
    const long int prefix = baseline_eight && navg >= 8 ? 8 :
                            segLen >= 2048 && navg >= 4 ? 4 : 2;
    const double segments_started = profile ? monotonic_seconds() : 0.0;
    if (profile) *preparation_seconds = segments_started - preparation_started;
    double start = 0.0;
    double Mr_r = 0.0, Mr_i = 0.0, M2_r = 0.0, M2_i = 0.0;
    for (long int ii = 0; ii < navg && !fallback;) {
        int batch = 1;
        if (ii == 0 && prefix == 8) batch = 8;
        else if (ii == 0 && prefix == 4) batch = 4;
        else if (ii >= prefix && requested_width >= 8 && navg - ii >= 8) batch = 8;
        else if (ii >= prefix && requested_width >= 4 && navg - ii >= 4) batch = 4;
        const double *segments[8];
        double rr[8], ri[8];
        for (int b = 0; b < batch; ++b) {
            const long int istart = (long int)floor(start + 0.5);
            start += shift;
            if (istart < 0 || istart > nData - segLen) { fallback = 1; break; }
            segments[b] = x1data + istart;
        }
        if (fallback) break;
        if (ii < prefix) {
            if (batch == 8) dot_psd_eight_anchored_simd(
                segments[0], segments[1], segments[2], segments[3],
                segments[4], segments[5], segments[6], segments[7],
                qr, qi, segLen, rr, ri);
            else if (batch == 4) dot_psd_four_anchored_simd(
                segments[0], segments[1], segments[2], segments[3],
                qr, qi, segLen, rr, ri);
            else dot_psd_anchored_simd(segments[0], qr, qi, segLen, rr, ri);
        } else {
            if (periodic) {
                if (batch == 8) folded_periodic_8(segments, segLen, fr, fi, center_r, mean_i, rr, ri);
                else if (batch == 4) folded_periodic_4(segments, segLen, fr, fi, center_r, mean_i, rr, ri);
                else folded_periodic_1(segments, segLen, fr, fi, center_r, mean_i, rr, ri);
            } else {
                if (batch == 8) folded_symmetric_8(segments, segLen, fr, fi, center_r, mean_i, rr, ri);
                else if (batch == 4) folded_symmetric_4(segments, segLen, fr, fi, center_r, mean_i, rr, ri);
                else folded_symmetric_1(segments, segLen, fr, fi, center_r, mean_i, rr, ri);
            }
            *folded_count += batch;
            /* A limited screening guard: no claim of arbitrary-signal
             * accuracy. Prevent changed overflow/fourth-power semantics. */
            const double limit = .25 * (statistics ? sqrt(sqrt(DBL_MAX)) : sqrt(DBL_MAX));
            for (int b = 0; b < batch; ++b) {
                if (!isfinite(rr[b]) || !isfinite(ri[b]) || fabs(rr[b]) + fabs(ri[b]) >= limit) {
                    fallback = 1;
                }
            }
        }
        if (fallback) break;
        for (int b = 0; b < batch; ++b) {
            update_original_statistics(ii + b, rr[b], ri[b], rr[b], ri[b],
                                       statistics && ii + b == navg - 1,
                                       &Mr_r, &Mr_i, &M2_r, &M2_i);
        }
        ii += batch;
    }
    if (profile) *segments_seconds = monotonic_seconds() - segments_started;
    free(qr); free(qi); free(fr); free(fi);
    if (fallback) {
        *folded_count = -1;
        return fast_dft_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs,
                             x1data, x2data, nData, segLen, Cr, Ci,
                             olap, order, csd, mode, statistics, true,
                             NULL, NULL, NULL, NULL);
    }
    *Pr_r = Mr_r; *Pr_i = Mr_i; *Navs = navg;
    if (statistics) {
        if (navg == 1) { *Vr_r = Mr_r * Mr_r - Mr_i * Mr_i; *Vr_i = 2.0 * Mr_i * Mr_r; }
        else { *Vr_r = M2_r / (navg - 1); *Vr_i = M2_i / (navg - 1); }
    }
    return 0;
}

int probe_folded_dft(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
                     long int *Navs,
                     const double *x1data, const double *x2data,
                     long int nData, long int segLen,
                     double *Cr, double *Ci,
                     double olap, int order, bool csd, int mode,
                     bool statistics, const double *window, bool periodic,
                     double frequency_bin, int requested_width, long int *folded_count)
{
    return probe_folded_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs, x1data, x2data,
                             nData, segLen, Cr, Ci, olap, order, csd, mode,
                             statistics, window, periodic, frequency_bin,
                             requested_width, folded_count, NULL, NULL);
}

int probe_folded_dft_profile(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
                     long int *Navs,
                     const double *x1data, const double *x2data,
                     long int nData, long int segLen,
                     double *Cr, double *Ci,
                     double olap, int order, bool csd, int mode,
                     bool statistics, const double *window, bool periodic,
                     double frequency_bin, int requested_width, long int *folded_count, double *preparation_seconds, double *segments_seconds)
{
    return probe_folded_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs, x1data, x2data,
                             nData, segLen, Cr, Ci, olap, order, csd, mode,
                             statistics, window, periodic, frequency_bin,
                             requested_width, folded_count, preparation_seconds, segments_seconds);
}
