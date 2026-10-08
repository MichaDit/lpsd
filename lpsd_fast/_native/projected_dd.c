/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Mean-projector preparation using compensated pairs of FP64 values.
 * This avoids software binary128 arithmetic on platforms whose long double
 * is wider than the hardware floating-point unit. It is a separate mode;
 * the original long-double projector and polynomial detrending are retained.
 */

#include <float.h>
#include <math.h>

/* Error-free addition of two finite doubles, unless their sum overflows.
 * Its roundoff is retained separately and later folded into the low part. */
static LPSD_ALWAYS_INLINE void dd_two_sum(double a, double b,
                                          double *hi, double *lo)
{
    const double sum = a + b;
    const double z = sum - a;
    *hi = sum;
    *lo = (a - (sum - z)) + (b - z);
}

static LPSD_ALWAYS_INLINE void dd_add(double value, double *hi, double *lo)
{
    double sum, error;
    dd_two_sum(*hi, value, &sum, &error);
    dd_two_sum(sum, *lo + error, hi, lo);
}

static LPSD_ALWAYS_INLINE void dd_divide(double hi, double lo, double divisor,
                                         double *quotient_hi, double *quotient_lo)
{
    const double q = hi / divisor;
    /* An explicit fused operation recovers the division residual. This is
     * not an invitation to fuse/reassociate the surrounding algorithm. */
    const double residual = fma(-q, divisor, hi) + lo;
    *quotient_hi = q;
    *quotient_lo = residual / divisor;
}

static LPSD_ALWAYS_INLINE double dd_subtract(double value,
                                              double mean_hi, double mean_lo)
{
    double hi, lo;
    dd_two_sum(value, -mean_hi, &hi, &lo);
    return hi + (lo - mean_lo);
}

/* Return false before writing coefficients if an extreme input requires
 * the exponent range of the original long-double implementation. */
static LPSD_ALWAYS_INLINE bool prepare_projected_coefficients_dd(
    const double *cr, const double *ci, long int length, double *qr, double *qi)
{
    enum { LANES = 8 };
    double hr[LANES] = {0.0}, lr[LANES] = {0.0};
    double hi[LANES] = {0.0}, li[LANES] = {0.0};
    double peak[LANES] = {0.0};
    for (long int start = 0; start < length;) {
        const int count = length - start < LANES ? (int)(length - start) : LANES;
        #pragma omp simd
        for (int j = 0; j < count; ++j) {
            const double real = cr[start + j], imag = ci[start + j];
            dd_add(real, &hr[j], &lr[j]);
            dd_add(imag, &hi[j], &li[j]);
            const double ar = fabs(real), ai = fabs(imag);
            if (ar > peak[j]) {
                peak[j] = ar;
            }
            if (ai > peak[j]) {
                peak[j] = ai;
            }
        }
        start += count;
    }
    double sum_r_hi = 0.0, sum_r_lo = 0.0;
    double sum_i_hi = 0.0, sum_i_lo = 0.0;
    for (int j = 0; j < LANES; ++j) {
        if (peak[j] > DBL_MAX * 0.25) {
            return false;
        }
        dd_add(hr[j], &sum_r_hi, &sum_r_lo);
        dd_add(lr[j], &sum_r_hi, &sum_r_lo);
        dd_add(hi[j], &sum_i_hi, &sum_i_lo);
        dd_add(li[j], &sum_i_hi, &sum_i_lo);
    }
    if (!isfinite(sum_r_hi) || !isfinite(sum_r_lo) ||
        !isfinite(sum_i_hi) || !isfinite(sum_i_lo)) {
        return false;
    }
    double mean_r_hi, mean_r_lo, mean_i_hi, mean_i_lo;
    dd_divide(sum_r_hi, sum_r_lo, (double)length, &mean_r_hi, &mean_r_lo);
    dd_divide(sum_i_hi, sum_i_lo, (double)length, &mean_i_hi, &mean_i_lo);
    #pragma omp simd
    for (long int j = 0; j < length; ++j) {
        qr[j] = dd_subtract(cr[j], mean_r_hi, mean_r_lo);
        qi[j] = dd_subtract(ci[j], mean_i_hi, mean_i_lo);
    }
    return true;
}
