/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Optimized derivative of lpsd 1.0.6 ltpda_dft.c.
 * Original DFT: M Hewitson, 2008-01-15.
 * CSD modifications: Artem Basalaev, 2022-11-11.
 * Polynomial detrending: Gerhard Heinzel, AEI, 2008-01-17.
 *
 * This derivative preserves the supplied frequencies and coefficients,
 * segment starts, long-double polynomial detrending, and the original
 * mean/variance recurrence, INCLUDING its known 1.0.6 defects. Correcting
 * that recurrence is a separate estimator change, not a speed optimization.
 *
 * Build with -O3 -fopenmp-simd -ffp-contract=off. -march=native is optional.
 * Do not use global -ffast-math. There is no internal thread parallelism.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "fast_dft.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

/* Unmodified upstream implementation. */
#include "../../lpsd/c_sources/polyreg.c"

/* NumPy/Cephes window port; its original BSD notice is retained. */
#include "numpy_kaiser.c"
#include "cosine_windows.c"

/* Optional order-0 preparation without software long-double arithmetic. */
#include "projected_dd.c"

static LPSD_ALWAYS_INLINE void detrend_original(int order, const double *px, int length,
                             double *out, double *a)
{
    /* Upstream signatures are non-const; these functions only read px. */
    double *input = (double *)px;
    switch (order) {
    case 0: polyreg0(input, length, out, a); break;
    case 1: polyreg1(input, length, out, a); break;
    case 2: polyreg2(input, length, out, a); break;
    case 3: polyreg3(input, length, out, a); break;
    case 4: polyreg4(input, length, out, a); break;
    case 5: polyreg5(input, length, out, a); break;
    case 6: polyreg6(input, length, out, a); break;
    case 7: polyreg7(input, length, out, a); break;
    case 8: polyreg8(input, length, out, a); break;
    case 9: polyreg9(input, length, out, a); break;
    case 10: polyreg10(input, length, out, a); break;
    }
}

static LPSD_ALWAYS_INLINE void dot_psd_serial(const double *x, const double *cr, const double *ci,
                           long int length, double *rr, double *ri)
{
    double r = 0.0, im = 0.0;
    for (long int j = 0; j < length; ++j) {
        const double value = x[j];
        r += cr[j] * value;
        im += ci[j] * value;
    }
    *rr = r;
    *ri = im;
}

static LPSD_ALWAYS_INLINE void dot_psd_simd(const double *x, const double *cr, const double *ci,
                         long int length, double *rr, double *ri)
{
    double r = 0.0, im = 0.0;
    #pragma omp simd reduction(+:r,im)
    for (long int j = 0; j < length; ++j) {
        const double value = x[j];
        r += cr[j] * value;
        im += ci[j] * value;
    }
    *rr = r;
    *ri = im;
}

static LPSD_ALWAYS_INLINE void dot_csd_serial(const double *x1, const double *x2,
                           const double *cr, const double *ci,
                           long int length, double *rr1, double *ri1,
                           double *rr2, double *ri2)
{
    double r1 = 0.0, i1 = 0.0, r2 = 0.0, i2 = 0.0;
    for (long int j = 0; j < length; ++j) {
        const double value1 = x1[j], value2 = x2[j];
        r1 += cr[j] * value1;
        i1 += ci[j] * value1;
        r2 += cr[j] * value2;
        i2 += ci[j] * value2;
    }
    *rr1 = r1;
    *ri1 = i1;
    *rr2 = r2;
    *ri2 = i2;
}

static LPSD_ALWAYS_INLINE void dot_csd_simd(const double *x1, const double *x2,
                         const double *cr, const double *ci,
                         long int length, double *rr1, double *ri1,
                         double *rr2, double *ri2)
{
    double r1 = 0.0, i1 = 0.0, r2 = 0.0, i2 = 0.0;
    #pragma omp simd reduction(+:r1,i1,r2,i2)
    for (long int j = 0; j < length; ++j) {
        const double value1 = x1[j], value2 = x2[j];
        r1 += cr[j] * value1;
        i1 += ci[j] * value1;
        r2 += cr[j] * value2;
        i2 += ci[j] * value2;
    }
    *rr1 = r1;
    *ri1 = i1;
    *rr2 = r2;
    *ri2 = i2;
}

/*
 * If c is the supplied complex window/exponential vector and P the real
 * polynomial least-squares projector, c^T (I-P) x = ((I-P)c)^T x.
 * Prepare that projected coefficient vector once, rather than detrending
 * every overlapping segment. Long-double moments follow polyreg0/1.
 *
 * The projected coefficients are rounded to double. This is algebraically
 * equivalent but is NOT a bitwise reproduction of upstream residuals.
 */
static LPSD_ALWAYS_INLINE void prepare_projected_coefficients(const double *cr, const double *ci,
                                           long int length, int order,
                                           double *qr, double *qi)
{
    const long double n = length;
    const long double n1 = order == 1 ? 2.L / (n - 1.L) : 0.L;
    long double sum_r = 0.L, sum_i = 0.L;
    long double sum_zr = 0.L, sum_zi = 0.L;
    for (long int j = 0; j < length; ++j) {
        sum_r += cr[j];
        sum_i += ci[j];
        if (order == 1) {
            const long double z = n1 * j - 1.L;
            sum_zr += (long double)cr[j] * z;
            sum_zi += (long double)ci[j] * z;
        }
    }
    const long double a0r = sum_r / n;
    const long double a0i = sum_i / n;
    long double a1r = 0.L, a1i = 0.L;
    if (order == 1) {
        /* Same coefficient expression as the original polyreg1. */
        const long double t1 = (n - 1.L) /
            ((0.3333333333333333333333L * n +
              0.3333333333333333333333L) * n);
        a1r = t1 * sum_zr;
        a1i = t1 * sum_zi;
    }
    for (long int j = 0; j < length; ++j) {
        const long double z = order == 1 ? n1 * j - 1.L : 0.L;
        qr[j] = (long double)cr[j] - (a1r * z + a0r);
        qi[j] = (long double)ci[j] - (a1i * z + a0i);
    }
}

static LPSD_ALWAYS_INLINE void dot_psd_anchored_simd(const double *x,
                                 const double *qr, const double *qi,
                                 long int length, double *rr, double *ri)
{
    /* A constant is removed by the projector. Local centering avoids a
     * raw large-DC dot product followed by cancellation. For nearby double
     * samples, x[j]-anchor is exact by Sterbenz's lemma. */
    const double anchor = x[0];
    double r = 0.0, im = 0.0;
    #pragma omp simd reduction(+:r,im)
    for (long int j = 0; j < length; ++j) {
        const double value = x[j] - anchor;
        r += qr[j] * value;
        im += qi[j] * value;
    }
    *rr = r;
    *ri = im;
}

/* Four independent overlapping segments share each coefficient load.
 * Each segment keeps its own anchor and SIMD reduction; there is no
 * decimation, change of segment starts, or cross-segment averaging. */
static LPSD_ALWAYS_INLINE void dot_psd_four_anchored_simd(const double *x0, const double *x1,
                                      const double *x2, const double *x3,
                                      const double *qr, const double *qi,
                                      long int length, double *rr, double *ri)
{
    const double a0 = x0[0], a1 = x1[0], a2 = x2[0], a3 = x3[0];
    double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
    double i0 = 0.0, i1 = 0.0, i2 = 0.0, i3 = 0.0;
    #pragma omp simd reduction(+:r0,r1,r2,r3,i0,i1,i2,i3)
    for (long int j = 0; j < length; ++j) {
        const double real = qr[j], imag = qi[j];
        const double v0 = x0[j] - a0, v1 = x1[j] - a1;
        const double v2 = x2[j] - a2, v3 = x3[j] - a3;
        r0 += real * v0;
        i0 += imag * v0;
        r1 += real * v1;
        i1 += imag * v1;
        r2 += real * v2;
        i2 += imag * v2;
        r3 += real * v3;
        i3 += imag * v3;
    }
    rr[0] = r0; rr[1] = r1; rr[2] = r2; rr[3] = r3;
    ri[0] = i0; ri[1] = i1; ri[2] = i2; ri[3] = i3;
}

/* Eight segments use sixteen independent SIMD accumulators. AVX-512 has
 * enough registers to retain those accumulators and the eight input anchors.
 * Keep the four-segment path on narrower x86 targets: spilling accumulators
 * defeats the reduction in coefficient loads. The run-time branch is needed
 * for portable target_clones builds, whose preprocessing has baseline flags. */
static LPSD_ALWAYS_INLINE bool use_eight_segment_batch(void)
{
#if defined(__AVX512F__)
    return true;
#elif defined(LPSD_HAVE_TARGET_CLONES) && LPSD_HAVE_TARGET_CLONES && \
      defined(__ELF__) && (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx512f") != 0;
#else
    return false;
#endif
}

static LPSD_ALWAYS_INLINE void dot_psd_eight_anchored_simd(
    const double *x0, const double *x1, const double *x2, const double *x3,
    const double *x4, const double *x5, const double *x6, const double *x7,
    const double *qr, const double *qi, long int length, double *rr, double *ri)
{
    const double a0 = x0[0], a1 = x1[0], a2 = x2[0], a3 = x3[0];
    const double a4 = x4[0], a5 = x5[0], a6 = x6[0], a7 = x7[0];
    double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
    double r4 = 0.0, r5 = 0.0, r6 = 0.0, r7 = 0.0;
    double i0 = 0.0, i1 = 0.0, i2 = 0.0, i3 = 0.0;
    double i4 = 0.0, i5 = 0.0, i6 = 0.0, i7 = 0.0;
    #pragma omp simd reduction(+:r0,r1,r2,r3,r4,r5,r6,r7,i0,i1,i2,i3,i4,i5,i6,i7)
    for (long int j = 0; j < length; ++j) {
        const double real = qr[j], imag = qi[j];
        const double v0 = x0[j] - a0, v1 = x1[j] - a1;
        const double v2 = x2[j] - a2, v3 = x3[j] - a3;
        const double v4 = x4[j] - a4, v5 = x5[j] - a5;
        const double v6 = x6[j] - a6, v7 = x7[j] - a7;
        r0 += real * v0; i0 += imag * v0;
        r1 += real * v1; i1 += imag * v1;
        r2 += real * v2; i2 += imag * v2;
        r3 += real * v3; i3 += imag * v3;
        r4 += real * v4; i4 += imag * v4;
        r5 += real * v5; i5 += imag * v5;
        r6 += real * v6; i6 += imag * v6;
        r7 += real * v7; i7 += imag * v7;
    }
    rr[0] = r0; rr[1] = r1; rr[2] = r2; rr[3] = r3;
    rr[4] = r4; rr[5] = r5; rr[6] = r6; rr[7] = r7;
    ri[0] = i0; ri[1] = i1; ri[2] = i2; ri[3] = i3;
    ri[4] = i4; ri[5] = i5; ri[6] = i6; ri[7] = i7;
}

static LPSD_ALWAYS_INLINE void dot_csd_anchored_simd(const double *x1, const double *x2,
                                 const double *qr, const double *qi,
                                 long int length, double *rr1, double *ri1,
                                 double *rr2, double *ri2)
{
    const double anchor1 = x1[0], anchor2 = x2[0];
    double r1 = 0.0, i1 = 0.0, r2 = 0.0, i2 = 0.0;
    #pragma omp simd reduction(+:r1,i1,r2,i2)
    for (long int j = 0; j < length; ++j) {
        const double value1 = x1[j] - anchor1;
        const double value2 = x2[j] - anchor2;
        r1 += qr[j] * value1;
        i1 += qi[j] * value1;
        r2 += qr[j] * value2;
        i2 += qi[j] * value2;
    }
    *rr1 = r1;
    *ri1 = i1;
    *rr2 = r2;
    *ri2 = i2;
}

static double monotonic_seconds(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
        return NAN;
    }
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static LPSD_ALWAYS_INLINE void update_original_statistics(long int ii,
                                        double rxsum1, double ixsum1,
                                        double rxsum2, double ixsum2,
                                        bool statistics,
                                        double *Mr_r, double *Mr_i,
                                        double *M2_r, double *M2_i)
{
    ixsum2 = -1.0 * ixsum2;
    const double Xr_r = rxsum1 * rxsum2 - ixsum2 * ixsum1;
    const double Xr_i = rxsum1 * ixsum2 + ixsum1 * rxsum2;
    if (ii == 0) {
        *Mr_r = Xr_r;
        *Mr_i = Xr_i;
    } else {
        const double Qr_r = Xr_r - *Mr_r;
        const double Qr_i = Xr_i - *Mr_i;
        /* Preserve upstream 1.0.6 exactly: ii, NOT ii + 1. */
        *Mr_r += Qr_r / ii;
        *Mr_i += Qr_i / ii;
        if (statistics) {
            const double XM_diff_r = Xr_r - *Mr_r;
            const double XM_diff_i = Xr_i - *Mr_i;
            /* Preserve assignment, NOT the corrected accumulating +=. */
            *M2_r = Qr_r * XM_diff_r - Qr_i * XM_diff_i;
            *M2_i = Qr_r * XM_diff_i + Qr_i * XM_diff_r;
        }
    }
}

static LPSD_TARGET_CLONES int fast_dft_impl(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
                         long int *Navs,
                         const double *x1data, const double *x2data,
                         long int nData, long int segLen,
                         const double *Cr, const double *Ci,
                         double olap, int order, bool csd, int mode,
                         bool statistics, bool batched,
                         double *inplace_r, double *inplace_i,
                         double *preparation_seconds, double *segments_seconds)
{
    if (Pr_r == NULL || Pr_i == NULL ||
        (statistics && (Vr_r == NULL || Vr_i == NULL)) ||
        Navs == NULL || x1data == NULL || Cr == NULL || Ci == NULL ||
        (csd && x2data == NULL) ||
        ((inplace_r == NULL) != (inplace_i == NULL))) {
        return 1;
    }
    *Pr_r = *Pr_i = NAN;
    if (Vr_r != NULL) {
        *Vr_r = NAN;
    }
    if (Vr_i != NULL) {
        *Vr_i = NAN;
    }
    *Navs = 0;
    if (mode < 0 || mode > 3 ||
        (mode == 2 && order != 0 && order != 1) ||
        (mode == 3 && order != 0)) {
        return 4;
    }
    if (order < -1 || order > 10 || !isfinite(olap) ||
        olap < 0.0 || olap >= 100.0) {
        return 1;
    }
    /* Upstream polynomial and rounding helpers use signed int lengths. */
    if (nData < 1 || nData > INT_MAX || segLen < 1 ||
        segLen > nData || segLen > INT_MAX || order >= segLen) {
        return 2;
    }

    /* Keep the upstream arithmetic order, including repeated start += shift. */
    const double ovfact = 1.0 / (1.0 - olap / 100.0);
    const double davg = ((double)(nData - segLen) * ovfact) / segLen + 1.0;
    if (!isfinite(davg) || davg + 0.5 >= (double)INT_MAX) {
        return 2;
    }
    const long int navg = (long int)floor(davg + 0.5);
    if (navg < 1 || navg > nData - segLen + 1) {
        /* Upstream otherwise clamps shift to 1 but reads past the data. */
        return 2;
    }
    double shift = navg == 1 ? 1.0 :
        (double)(nData - segLen) / (double)(navg - 1);
    if (shift < 1.0) {
        shift = 1.0;
    }

    const bool profile = preparation_seconds != NULL && segments_seconds != NULL;
    const double preparation_started = profile ? monotonic_seconds() : 0.0;
    double *residual1 = NULL;
    double *residual2 = NULL;
    double *projected_r = NULL;
    double *projected_i = NULL;
    bool owns_projected = false;
    double coefficients1[11] = {0.0};
    double coefficients2[11] = {0.0};
    if (mode >= 2) {
        owns_projected = inplace_r == NULL;
        projected_r = owns_projected ?
            (double *)malloc((size_t)segLen * sizeof(double)) : inplace_r;
        projected_i = owns_projected ?
            (double *)malloc((size_t)segLen * sizeof(double)) : inplace_i;
        if (projected_r == NULL || projected_i == NULL) {
            free(projected_r);
            free(projected_i);
            return 3;
        }
        if (mode != 3 || !prepare_projected_coefficients_dd(
                Cr, Ci, segLen, projected_r, projected_i)) {
            prepare_projected_coefficients(Cr, Ci, segLen, order,
                                            projected_r, projected_i);
        }
    } else if (order >= 0) {
        residual1 = (double *)malloc((size_t)segLen * sizeof(double));
        if (csd) {
            residual2 = (double *)malloc((size_t)segLen * sizeof(double));
        }
        if (residual1 == NULL || (csd && residual2 == NULL)) {
            free(residual1);
            free(residual2);
            return 3;
        }
    }

    const double segments_started = profile ? monotonic_seconds() : 0.0;
    if (profile) {
        *preparation_seconds = segments_started - preparation_started;
    }
    double start = 0.0;
    double Mr_r = 0.0, Mr_i = 0.0, M2_r = 0.0, M2_i = 0.0;
    const bool batch_eight = batched && mode >= 2 && !csd &&
        segLen >= 128 && use_eight_segment_batch();
    for (long int ii = 0; ii < navg; ++ii) {
        if (batch_eight && navg - ii >= 8) {
            const double *segments[8];
            double rr[8], ri[8];
            for (int b = 0; b < 8; ++b) {
                const long int istart = (long int)floor(start + 0.5);
                start += shift;
                if (istart < 0 || istart > nData - segLen) {
                    free(residual1);
                    free(residual2);
                    if (owns_projected) {
                        free(projected_r);
                        free(projected_i);
                    }
                    return 2;
                }
                segments[b] = x1data + istart;
            }
            dot_psd_eight_anchored_simd(
                segments[0], segments[1], segments[2], segments[3],
                segments[4], segments[5], segments[6], segments[7],
                projected_r, projected_i, segLen, rr, ri);
            for (int b = 0; b < 8; ++b) {
                update_original_statistics(ii + b, rr[b], ri[b], rr[b], ri[b],
                                             statistics && ii + b == navg - 1,
                                             &Mr_r, &Mr_i, &M2_r, &M2_i);
            }
            ii += 7;
            continue;
        }
        /* Short coefficient vectors already fit well in L1. Their extra
         * four-way batch bookkeeping did not give a consistent measured
         * benefit on the narrower SIMD path. */
        if (batched && mode >= 2 && !csd && segLen >= 2048 && navg - ii >= 4) {
            const double *segments[4];
            double rr[4], ri[4];
            for (int b = 0; b < 4; ++b) {
                const long int istart = (long int)floor(start + 0.5);
                start += shift;
                if (istart < 0 || istart > nData - segLen) {
                    free(residual1);
                    free(residual2);
                    if (owns_projected) {
                        free(projected_r);
                        free(projected_i);
                    }
                    return 2;
                }
                segments[b] = x1data + istart;
            }
            dot_psd_four_anchored_simd(segments[0], segments[1],
                                       segments[2], segments[3],
                                       projected_r, projected_i, segLen, rr, ri);
            for (int b = 0; b < 4; ++b) {
                /* Legacy M2 is overwritten, so only its final assignment
                 * can affect the output. Every mean update is retained. */
                update_original_statistics(ii + b, rr[b], ri[b], rr[b], ri[b],
                                             statistics && ii + b == navg - 1,
                                             &Mr_r, &Mr_i,
                                             &M2_r, &M2_i);
            }
            ii += 3;
            continue;
        }
        const long int istart = (long int)floor(start + 0.5);
        start += shift;
        if (istart < 0 || istart > nData - segLen) {
            free(residual1);
            free(residual2);
            if (owns_projected) {
                free(projected_r);
                free(projected_i);
            }
            return 2;
        }
        const double *segment1 = x1data + istart;
        const double *segment2 = csd ? x2data + istart : segment1;
        if (mode < 2 && order >= 0) {
            detrend_original(order, segment1, (int)segLen,
                             residual1, coefficients1);
            segment1 = residual1;
            if (csd) {
                detrend_original(order, segment2, (int)segLen,
                                 residual2, coefficients2);
                segment2 = residual2;
            }
        }

        double rxsum1, ixsum1, rxsum2, ixsum2;
        if (csd) {
            if (mode >= 2) {
                dot_csd_anchored_simd(segment1, segment2,
                                       projected_r, projected_i, segLen,
                                       &rxsum1, &ixsum1, &rxsum2, &ixsum2);
            } else if (mode == 0) {
                dot_csd_serial(segment1, segment2, Cr, Ci, segLen,
                                &rxsum1, &ixsum1, &rxsum2, &ixsum2);
            } else {
                dot_csd_simd(segment1, segment2, Cr, Ci, segLen,
                              &rxsum1, &ixsum1, &rxsum2, &ixsum2);
            }
        } else {
            if (mode >= 2) {
                dot_psd_anchored_simd(segment1, projected_r, projected_i,
                                       segLen, &rxsum1, &ixsum1);
            } else if (mode == 0) {
                dot_psd_serial(segment1, Cr, Ci, segLen, &rxsum1, &ixsum1);
            } else {
                dot_psd_simd(segment1, Cr, Ci, segLen, &rxsum1, &ixsum1);
            }
            rxsum2 = rxsum1;
            ixsum2 = ixsum1;
        }

        /* Only the final (overwriting) legacy M2 value is observable. */
        update_original_statistics(ii, rxsum1, ixsum1, rxsum2, ixsum2,
                                     statistics && ii == navg - 1,
                                     &Mr_r, &Mr_i, &M2_r, &M2_i);
    }

    if (profile) {
        *segments_seconds = monotonic_seconds() - segments_started;
    }
    free(residual1);
    free(residual2);
    if (owns_projected) {
        free(projected_r);
        free(projected_i);
    }
    *Pr_r = Mr_r;
    *Pr_i = Mr_i;
    if (statistics) {
        if (navg == 1) {
            *Vr_r = Mr_r * Mr_r - Mr_i * Mr_i;
            *Vr_i = 2.0 * Mr_i * Mr_r;
        } else {
            *Vr_r = M2_r / (navg - 1);
            *Vr_i = M2_i / (navg - 1);
        }
    }
    *Navs = navg;
    return 0;
}

int fast_dft(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
             long int *Navs,
             const double *x1data, const double *x2data,
             long int nData, long int segLen,
             const double *Cr, const double *Ci,
             double olap, int order, bool csd, int mode)
{
    return fast_dft_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs,
                          x1data, x2data, nData, segLen, Cr, Ci,
                          olap, order, csd, mode, true, false,
                          NULL, NULL, NULL, NULL);
}

int fast_dft_profile(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
                     long int *Navs,
                     const double *x1data, const double *x2data,
                     long int nData, long int segLen,
                     const double *Cr, const double *Ci,
                     double olap, int order, bool csd, int mode,
                     double *preparation_seconds, double *segments_seconds)
{
    if (preparation_seconds == NULL || segments_seconds == NULL) {
        return 1;
    }
    *preparation_seconds = *segments_seconds = NAN;
    return fast_dft_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs,
                          x1data, x2data, nData, segLen, Cr, Ci,
                          olap, order, csd, mode, true, false, NULL, NULL,
                          preparation_seconds, segments_seconds);
}

int fast_dft_selected(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
                      long int *Navs,
                      const double *x1data, const double *x2data,
                      long int nData, long int segLen,
                      double *Cr, double *Ci,
                      double olap, int order, bool csd, int mode,
                      bool statistics, bool inplace)
{
    return fast_dft_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs,
                          x1data, x2data, nData, segLen, Cr, Ci,
                          olap, order, csd, mode, statistics, true,
                          inplace ? Cr : NULL, inplace ? Ci : NULL,
                          NULL, NULL);
}

int fast_dft_selected_profile(double *Pr_r, double *Pr_i,
                              double *Vr_r, double *Vr_i, long int *Navs,
                              const double *x1data, const double *x2data,
                              long int nData, long int segLen,
                              double *Cr, double *Ci,
                              double olap, int order, bool csd, int mode,
                              bool statistics, bool inplace,
                              double *preparation_seconds, double *segments_seconds)
{
    if (preparation_seconds == NULL || segments_seconds == NULL) {
        return 1;
    }
    *preparation_seconds = *segments_seconds = NAN;
    return fast_dft_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs,
                          x1data, x2data, nData, segLen, Cr, Ci,
                          olap, order, csd, mode, statistics, true,
                          inplace ? Cr : NULL, inplace ? Ci : NULL,
                          preparation_seconds, segments_seconds);
}

#include "rolling_boxcar.c"

int window_sums(const double *window, long int length,
                 double *sum_window, double *sum_squares)
{
    if (length < 0 || sum_window == NULL || sum_squares == NULL ||
        (length > 0 && window == NULL)) {
        return 1;
    }
    double s1 = 0.0, s2 = 0.0;
    for (long int j = 0; j < length; ++j) {
        const double value = window[j];
        const double square = value * value;
        s1 += value;
        s2 += square;
    }
    *sum_window = s1;
    *sum_squares = s2;
    return 0;
}

int native_long_double_mantissa_bits(void)
{
    return LDBL_MANT_DIG;
}

LPSD_TARGET_CLONES int generate_coefficients(double *Cr, double *Ci, const double *window,
                           long int length, double frequency_bin)
{
    if (Cr == NULL || Ci == NULL || window == NULL ||
        length < 1 || length > INT_MAX || !isfinite(frequency_bin)) {
        return 1;
    }
    /* Match NumPy's phase construction; preserve this operation order.
     * GCC can combine the paired ordinary calls into one libm sincos. */
    const double pi = 3.141592653589793238462643383279502884;
    const double scale = (2.0 * pi) * frequency_bin / (double)length;
    for (long int j = 0; j < length; ++j) {
        const double phase = scale * (double)j;
        const double value = window[j];
        const double real = cos(phase);
        const double imag = sin(phase);
        Cr[j] = value * real;
        Ci[j] = value * imag;
    }
    return 0;
}

LPSD_TARGET_CLONES int generate_coefficients_blocked(double *Cr, double *Ci, const double *window,
                                   long int length, double frequency_bin)
{
    if (Cr == NULL || Ci == NULL || window == NULL ||
        length < 1 || length > INT_MAX || !isfinite(frequency_bin)) {
        return 1;
    }
    enum { BLOCK = 64 };
    const double pi = 3.141592653589793238462643383279502884;
    const double scale = (2.0 * pi) * frequency_bin / (double)length;
    /* Short vectors cannot amortize the offset table. For unusually large
     * total phase, keep direct argument reduction rather than magnifying
     * differences between rounded block+offset and independent phases. */
    if (length < 4 * BLOCK || !isfinite(scale) ||
        fabs(scale) * (double)(length - 1) > 8192.0) {
        return generate_coefficients(Cr, Ci, window, length, frequency_bin);
    }
    double offsets_r[BLOCK], offsets_i[BLOCK], offsets_phase[BLOCK];
    for (int j = 0; j < BLOCK; ++j) {
        const double phase = scale * (double)j;
        offsets_phase[j] = phase;
        offsets_r[j] = cos(phase);
        offsets_i[j] = sin(phase);
    }
    for (long int start = 0; start < length;) {
        const double phase = scale * (double)start;
        const double base_r = cos(phase), base_i = sin(phase);
        const int first = (int)start;
        const int count = length - start < BLOCK ? (int)(length - start) : BLOCK;
        #pragma omp simd
        for (int j = 0; j < count; ++j) {
            const double real = base_r * offsets_r[j] - base_i * offsets_i[j];
            const double imag = base_i * offsets_r[j] + base_r * offsets_i[j];
            /* Recover the independently rounded sample phase. Subtraction
             * of nearby block/sample phases avoids the larger rounding
             * error of simply treating rounded phases as exactly additive.
             * The correction is tiny under the total-phase guard above;
             * the omitted second-order rotation term is below 2e-24. */
            const double delta = (scale * (double)(first + j) - phase) -
                offsets_phase[j];
            const double value = window[start + j];
            Cr[start + j] = value * (real - delta * imag);
            Ci[start + j] = value * (imag + delta * real);
        }
        /* The final partial block ends at length, including on LLP64
         * where a fixed BLOCK increment could overflow a 32-bit long. */
        start += count;
    }
    return 0;
}
