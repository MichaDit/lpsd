/* SPDX-License-Identifier: BSD-3-Clause
 *
 * C adaptation of the i0/Chebyshev and Kaiser algorithms in
 * NumPy 2.3.5, numpy/lib/_function_base_impl.py.
 * That source identifies its i0 approximation as originating in Cephes.
 * The approximation coefficients and arithmetic recurrence are retained.
 * The periodic LPSD window is np.kaiser(length + 1, beta)[:-1].
 *
 * Copyright (c) 2005-2025, NumPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *     * Neither the name of the NumPy Developers nor the names of any
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <limits.h>
#include <math.h>
#include <stddef.h>

static const double numpy_i0_a[30] = {
    -4.41534164647933937950E-18,
     3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
     1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
     7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
     2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
     9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
     2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
     6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
     1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
     1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
     1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
     1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
     1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
     4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
     1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
     6.76795274409476084995E-1
};

static const double numpy_i0_b[25] = {
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
     4.46562142029675999901E-17,
     3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
     1.77256013305652638360E-15,
     3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
     1.54008621752140982691E-14,
     3.85277838274214270114E-13,
     7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
     1.18891471078464383424E-11,
     4.94060238822496958910E-10,
     3.39623202570838634515E-9,
     2.26666899049817806459E-8,
     2.04891858946906374183E-7,
     2.89137052083475648297E-6,
     6.88975834691682398426E-5,
     3.36911647825569408990E-3,
     8.04490411014108831608E-1
};

enum { NUMPY_KAISER_BLOCK = 128 };

static double numpy_cheb_scalar(double x, const double *values, int count)
{
    double b0 = values[0], b1 = 0.0, b2 = 0.0;
    for (int i = 1; i < count; ++i) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + values[i];
    }
    return 0.5 * (b0 - b2);
}

static double numpy_i0_scalar(double x)
{
    x = fabs(x);
    if (x <= 8.0) {
        return exp(x) * numpy_cheb_scalar(x / 2.0 - 2.0, numpy_i0_a, 30);
    }
    return exp(x) * numpy_cheb_scalar(32.0 / x - 2.0, numpy_i0_b, 25) / sqrt(x);
}

/* SIMD across independent sample positions, without reassociating the
 * Chebyshev recurrence within an individual sample. Small workspaces stay
 * in cache and avoid NumPy's full-length intermediate vectors. */
static void numpy_cheb_block(const double *x, double *out, int length,
                             const double *values, int count)
{
    double b0[NUMPY_KAISER_BLOCK];
    double b1[NUMPY_KAISER_BLOCK];
    double b2[NUMPY_KAISER_BLOCK];
    #pragma omp simd
    for (int j = 0; j < length; ++j) {
        b0[j] = values[0];
        b1[j] = 0.0;
        b2[j] = 0.0;
    }
    for (int i = 1; i < count; ++i) {
        const double coefficient = values[i];
        #pragma omp simd
        for (int j = 0; j < length; ++j) {
            b2[j] = b1[j];
            b1[j] = b0[j];
            b0[j] = x[j] * b1[j] - b2[j] + coefficient;
        }
    }
    #pragma omp simd
    for (int j = 0; j < length; ++j) {
        out[j] = 0.5 * (b0[j] - b2[j]);
    }
}

int generate_kaiser(double *window, long int length, double beta)
{
    if (window == NULL || length < 1 || length > INT_MAX || !isfinite(beta)) {
        return 1;
    }
    beta = fabs(beta);
    if (beta == 0.0) {
        #pragma omp simd
        for (long int j = 0; j < length; ++j) {
            window[j] = 1.0;
        }
        return 0;
    }
    const double denominator = numpy_i0_scalar(beta);
    const double alpha = (double)length / 2.0;
    const long int half = length / 2;
    double argument[NUMPY_KAISER_BLOCK];
    double transformed[NUMPY_KAISER_BLOCK];
    double cheb[NUMPY_KAISER_BLOCK];

    for (long int start = 0; start <= half; start += NUMPY_KAISER_BLOCK) {
        const long int remaining = half - start + 1;
        const int count = remaining < NUMPY_KAISER_BLOCK ?
            (int)remaining : NUMPY_KAISER_BLOCK;
        int count_a = 0;
        /* Arguments are monotone on the left half of this symmetric window,
         * so the <=8 and >8 approximation regions form contiguous blocks. */
        for (int j = 0; j < count; ++j) {
            const double normalized = ((double)(start + j) - alpha) / alpha;
            const double x = beta * sqrt(1.0 - normalized * normalized);
            argument[j] = x;
            if (x <= 8.0) {
                ++count_a;
                transformed[j] = x / 2.0 - 2.0;
            } else {
                transformed[j] = 32.0 / x - 2.0;
            }
        }
        if (count_a > 0) {
            numpy_cheb_block(transformed, cheb, count_a, numpy_i0_a, 30);
        }
        if (count_a < count) {
            numpy_cheb_block(transformed + count_a, cheb + count_a,
                             count - count_a, numpy_i0_b, 25);
        }
        for (int j = 0; j < count; ++j) {
            double value = exp(argument[j]) * cheb[j];
            if (j >= count_a) {
                value /= sqrt(argument[j]);
            }
            value /= denominator;
            const long int index = start + j;
            window[index] = value;
            const long int mirror = length - index;
            /* The right endpoint of the length+1 window is intentionally
             * omitted, matching upstream LPSD's periodic convention. */
            if (mirror < length && mirror != index) {
                window[mirror] = value;
            }
        }
    }
    return 0;
}
