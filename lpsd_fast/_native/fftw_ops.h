/* SPDX-License-Identifier: GPL-3.0-or-later
 * Elementwise FFTW pipeline operations, included by fast_dft.c.
 *
 * No FFTW dependency, allocations, or internal worker threads. The enclosing
 * build must retain -fno-fast-math -ffp-contract=off. Native center/window and
 * power preserve the NumPy baseline's elementary double operations. The
 * optional grouped reduction changes only the floating-point reduction order.
 */
#ifndef LPSD_FFTW_OPS_H
#define LPSD_FFTW_OPS_H

#include <stdint.h>
#include <math.h>
#include "fast_dft.h"

int fftw_ops_version(void) { return 1; }

LPSD_TARGET_CLONES int fftw_ops_center_window(double *data, const double *window,
                                             int64_t n, double mean)
{
    if (!data || !window || n < 1) return 1;
    /* data already holds input - input[0]. Keep the mean computation in the
     * Python caller unchanged, then combine these two elementwise passes. */
    for (int64_t i = 0; i < n; ++i) {
        const double centered = data[i] - mean;
        data[i] = centered * window[i];
    }
    return 0;
}

LPSD_TARGET_CLONES int fftw_ops_power(const double *transform, int64_t n,
                                     double scale, double *powers)
{
    if (!transform || !powers || n < 2) return 1;
    const int64_t bins = n / 2 + 1;
    for (int64_t i = 0; i < bins; ++i) {
        const double real_square = transform[2 * i] * transform[2 * i];
        const double imag_square = transform[2 * i + 1] * transform[2 * i + 1];
        const double squared_magnitude = real_square + imag_square;
        powers[i] = squared_magnitude * scale;
    }
    powers[0] *= 0.5;
    if (n % 2 == 0) powers[bins - 1] *= 0.5;
    return 0;
}

LPSD_TARGET_CLONES int fftw_ops_log_power(const double *transform, int64_t n,
                                         double scale, const int64_t *cuts,
                                         int64_t groups, double df,
                                         double *density, double *band_powers)
{
    if (!transform || !cuts || !density || !band_powers || n < 2 || groups < 1)
        return 1;
    const int64_t bins = n / 2 + 1;
    if (cuts[0] != 1 || cuts[groups] != bins) return 1;
    for (int64_t j = 0; j < groups; ++j) {
        const int64_t start = cuts[j], stop = cuts[j + 1];
        if (start < 1 || stop > bins || start >= stop) return 1;
        double sum = 0.0;
        /* This opt-in reduction is mathematical-equivalent, not bitwise
         * equivalent to np.add.reduceat. No FFT bins change groups. */
        #pragma omp simd reduction(+:sum)
        for (int64_t i = start; i < stop; ++i) {
            const double real_square = transform[2 * i] * transform[2 * i];
            const double imag_square = transform[2 * i + 1] * transform[2 * i + 1];
            double power = (real_square + imag_square) * scale;
            if (n % 2 == 0 && i == bins - 1) power *= 0.5;
            sum += power;
        }
        density[j] = sum / (double)(stop - start);
        band_powers[j] = sum * df;
    }
    return 0;
}

#endif
