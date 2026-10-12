/* SPDX-License-Identifier: GPL-3.0-or-later
 * Optional elementary Bluestein operations, included by fast_dft.c.
 *
 * No FFTW dependency, allocations, phase generation, reduction, or worker
 * threads. The enclosing build retains -fno-fast-math -ffp-contract=off.
 * Complex values are interleaved double real/imaginary pairs. Buffers are
 * private to the prepared FFT adapter, contiguous and mutually nonaliasing.
 */
#ifndef LPSD_FFTW_BLUESTEIN_OPS_H
#define LPSD_FFTW_BLUESTEIN_OPS_H

#include <stdint.h>
#include <math.h>
#include "fast_dft.h"

int fftw_bluestein_ops_version(void) { return 1; }

LPSD_TARGET_CLONES int fftw_bluestein_prepare(
    const double *input, const double *chirp, int64_t n, int64_t m,
    double *work)
{
    if (!input || !chirp || !work || n < 2 || m < n || m > INT32_MAX)
        return 1;
    /* input is already centered/windowed by the unchanged owning pipeline.
     * This replaces two strided NumPy writes with one interleaved write. */
    for (int64_t j = 0; j < n; ++j) {
        const double value = input[j];
        work[2 * j] = value * chirp[2 * j];
        work[2 * j + 1] = value * chirp[2 * j + 1];
    }
    for (int64_t j = n; j < m; ++j) {
        work[2 * j] = 0.0;
        work[2 * j + 1] = 0.0;
    }
    return 0;
}

LPSD_TARGET_CLONES int fftw_bluestein_finish(
    const double *work, const double *chirp, int64_t n, double inverse_m,
    double *output)
{
    if (!work || !chirp || !output || n < 2 || n > INT32_MAX ||
        !isfinite(inverse_m) || inverse_m <= 0.0)
        return 1;
    const int64_t bins = n / 2 + 1;
    for (int64_t k = 0; k < bins; ++k) {
        const double real = work[2 * k] * chirp[2 * k] -
                            work[2 * k + 1] * chirp[2 * k + 1];
        const double imag = work[2 * k] * chirp[2 * k + 1] +
                            work[2 * k + 1] * chirp[2 * k];
        output[2 * k] = real * inverse_m;
        output[2 * k + 1] = imag * inverse_m;
    }
    /* Match the exact-real endpoint representation of an N-point r2c FFT. */
    output[1] = 0.0;
    if (n % 2 == 0) output[2 * (bins - 1) + 1] = 0.0;
    return 0;
}

#endif
