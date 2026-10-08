/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Periodic cosine-series windows supplied by lpsd/flattop.py.
 * Coefficients and window conventions are supplied by the caller. Evaluate
 * all harmonics from one cosine using the Chebyshev recurrence, vectorized
 * across independent samples. Small fixed workspaces stay in cache.
 * This changes rounding; the scalar Python API keeps the upstream formula.
 */
#include <limits.h>
#include <math.h>
#include <stddef.h>

enum { LPSD_COSINE_BLOCK = 128, LPSD_COSINE_MAX_TERMS = 32 };

LPSD_TARGET_CLONES int generate_cosine_window(double *window, long int length,
                            const double *coefficients, int count)
{
    if (window == NULL || coefficients == NULL || length < 1 ||
        length > INT_MAX || count < 1 || count > LPSD_COSINE_MAX_TERMS) {
        return 1;
    }
    for (int k = 0; k < count; ++k) {
        if (!isfinite(coefficients[k])) {
            return 1;
        }
    }
    const double pi = 3.141592653589793238462643383279502884;
    const double step = pi / (double)length;
    const long int half = length / 2;
    double cosine[LPSD_COSINE_BLOCK];
    double previous[LPSD_COSINE_BLOCK];
    double current[LPSD_COSINE_BLOCK];
    double values[LPSD_COSINE_BLOCK];

    for (long int start = 0; start <= half; start += LPSD_COSINE_BLOCK) {
        const long int remaining = half - start + 1;
        const int block = remaining < LPSD_COSINE_BLOCK ?
            (int)remaining : LPSD_COSINE_BLOCK;
        for (int j = 0; j < block; ++j) {
            /* Same first-harmonic phase as upstream: 2*(pi/L*j). */
            cosine[j] = cos(2.0 * (step * (double)(start + j)));
        }
        #pragma omp simd
        for (int j = 0; j < block; ++j) {
            previous[j] = 1.0;
            current[j] = cosine[j];
            values[j] = coefficients[0];
            if (count > 1) {
                values[j] += coefficients[1] * current[j];
            }
        }
        for (int k = 2; k < count; ++k) {
            const double coefficient = coefficients[k];
            #pragma omp simd
            for (int j = 0; j < block; ++j) {
                const double next = (2.0 * cosine[j]) * current[j] - previous[j];
                values[j] += coefficient * next;
                previous[j] = current[j];
                current[j] = next;
            }
        }
        for (int j = 0; j < block; ++j) {
            const long int index = start + j;
            const long int mirror = length - index;
            window[index] = values[j];
            if (mirror < length && mirror != index) {
                window[mirror] = values[j];
            }
        }
    }
    return 0;
}
