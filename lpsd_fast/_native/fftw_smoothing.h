/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef LPSD_FFTW_SMOOTHING_H
#define LPSD_FFTW_SMOOTHING_H

/* Additive helpers for benchmarks/matched_smoothing.py.  These functions
 * evaluate the existing continuous Kaiser *power* kernel; they do not replace
 * it with a table, a narrower support, or a different spectral estimator.
 * All storage belongs to the caller.  There are no allocations or worker
 * threads here; independent frequency ranges may be dispatched by Python.
 * Compile without fast-math and with floating-point contraction disabled. */
#include <math.h>
#include <stdint.h>
#include <stddef.h>

#ifndef LPSD_TARGET_CLONES
#define LPSD_TARGET_CLONES
#endif

static inline double lpsd_fftw_kaiser_power(double q, double beta,
                                         double halfwidth, double scale)
{
    const double pi = 3.141592653589793238462643383279502884;
    if (fabs(q) > halfwidth) return 0.0;
    if (beta == 0.0) {
        if (q == 0.0) return 1.0;
        const double phase = pi * q;
        const double sinc = sin(phase) / phase;
        return sinc * sinc;
    }
    const double phase = pi * q;
    const double value = beta * beta - phase * phase;
    const double argument = sqrt(fabs(value));
    double response = 1.0;
    if (argument > 1e-12)
        response = value > 0.0 ? sinh(argument) / argument
                               : sin(argument) / argument;
    response *= scale;
    return response * response;
}

static inline double lpsd_fftw_smoothing_weight(int64_t bin, double df,
                                               double frequency,
                                               double length_over_fs,
                                               double sample_rate,
                                               double beta, double halfwidth,
                                               double response_scale)
{
    /* Keep the operations and both reflected response contributions used by
     * the original NumPy implementation, including near DC and Nyquist. */
    const double bin_frequency = (double)bin * df;
    const double direct = (bin_frequency - frequency) * length_over_fs;
    const double summed = bin_frequency + frequency;
    const double reflected = fmin(summed, sample_rate - summed) * length_over_fs;
    return lpsd_fftw_kaiser_power(direct, beta, halfwidth, response_scale)
         + lpsd_fftw_kaiser_power(reflected, beta, halfwidth, response_scale);
}

/* Compute normalization and, optionally, a cached normalized weight vector.
 * Passing weights=NULL computes only the denominator in constant space.
 * Endpoint quadrature halves only the denominator: the numerator retains
 * both reflected contributions to the one-sided PSD, exactly as before.
 * Returns 0 on success, 1 for invalid arguments, 2 for invalid normalization. */
LPSD_TARGET_CLONES int lpsd_fftw_smoothing_prepare(
    int64_t n, double sample_rate, double beta, double halfwidth,
    int64_t first, int64_t stop, int64_t length, double frequency,
    double *weights, double *denominator)
{
    if (n < 2 || first < 0 || stop <= first || stop > n / 2 + 1 || length < 1 ||
        !isfinite(sample_rate) || sample_rate <= 0.0 || !isfinite(beta) ||
        beta < 0.0 || beta > 100.0 || !isfinite(halfwidth) || halfwidth <= 0.0 ||
        !isfinite(frequency) || frequency < 0.0 || frequency > sample_rate / 2.0 ||
        denominator == NULL) return 1;
    const double df = sample_rate / (double)n;
    const double length_over_fs = (double)length / sample_rate;
    const double response_scale = beta == 0.0 ? 1.0 : beta / sinh(beta);
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int64_t bin = first; bin < stop; ++bin) {
        const double weight = lpsd_fftw_smoothing_weight(
            bin, df, frequency, length_over_fs, sample_rate, beta, halfwidth,
            response_scale);
        if (weights != NULL) weights[bin - first] = weight;
        sum += weight;
    }
    if (first == 0) {
        const double endpoint = weights != NULL ? weights[0]
            : lpsd_fftw_smoothing_weight(0, df, frequency, length_over_fs,
                                        sample_rate, beta, halfwidth, response_scale);
        sum -= 0.5 * endpoint;
    }
    if (n % 2 == 0 && stop == n / 2 + 1) {
        const double endpoint = weights != NULL ? weights[stop - first - 1]
            : lpsd_fftw_smoothing_weight(stop - 1, df, frequency, length_over_fs,
                                        sample_rate, beta, halfwidth, response_scale);
        sum -= 0.5 * endpoint;
    }
    if (!isfinite(sum) || sum <= 0.0) return 2;
    *denominator = sum;
    if (weights != NULL) {
        #pragma omp simd
        for (int64_t k = 0; k < stop - first; ++k) weights[k] /= sum;
    }
    return 0;
}

/* Evaluate a range of independently prepared output kernels.  Cached
 * normalized weights use a SIMD dot.  A NULL cached pointer evaluates the
 * full original Kaiser formula and multiplies it by power in the same loop,
 * without any frequency, mask, response, or temporary weight arrays.
 * Index/buffer validation is performed by the Python preparation wrapper. */
static inline int lpsd_fftw_smoothing_apply_impl(
    const double *power, int64_t n, double sample_rate, double beta, double halfwidth,
    int64_t begin, int64_t end, const int64_t *indices, const int64_t *firsts,
    const int64_t *stops, const int64_t *lengths, const double *frequencies,
    const double *denominators, const void *const *cached, double *output,
    int cached_float32, double *deferred_denominators)
{
    if (power == NULL || n < 2 || !isfinite(sample_rate) || sample_rate <= 0.0 ||
        !isfinite(beta) || beta < 0.0 || beta > 100.0 ||
        !isfinite(halfwidth) || halfwidth <= 0.0 || begin < 0 || end < begin ||
        indices == NULL || firsts == NULL || stops == NULL || lengths == NULL ||
        frequencies == NULL || denominators == NULL || cached == NULL ||
        output == NULL) return 1;
    const double df = sample_rate / (double)n;
    const double response_scale = beta == 0.0 ? 1.0 : beta / sinh(beta);
    for (int64_t j = begin; j < end; ++j) {
        const int64_t first = firsts[j], stop = stops[j];
        const int deferred = deferred_denominators != NULL && cached[j] == NULL &&
                             isnan(denominators[j]);
        if (first < 0 || stop <= first || stop > n / 2 + 1 || lengths[j] < 1 ||
            (!deferred && (!isfinite(denominators[j]) || denominators[j] <= 0.0))) return 1;
        double sum = 0.0;
        if (cached[j] != NULL && cached_float32) {
            const float *weights = (const float *)cached[j];
            /* Only stored weights are narrowed. Products and the reduction
             * remain double; samples, FFT, powers and low LPSD remain double. */
            #pragma omp simd reduction(+:sum)
            for (int64_t k = 0; k < stop - first; ++k)
                sum += (double)weights[k] * power[first + k];
        } else if (cached[j] != NULL) {
            const double *weights = cached[j];
            #pragma omp simd reduction(+:sum)
            for (int64_t k = 0; k < stop - first; ++k)
                sum += weights[k] * power[first + k];
        } else {
            const double frequency = frequencies[j];
            const double length_over_fs = (double)lengths[j] / sample_rate;
            if (deferred) {
                if (!isfinite(frequency) || frequency < 0.0 ||
                    frequency > sample_rate / 2.0) return 1;
                /* First-use streaming computes each interior weight once.
                 * The denominator is independent of power and is retained
                 * for subsequent calls. Both reductions stay in double. */
                double normalization = 0.0;
                #pragma omp simd reduction(+:sum,normalization)
                for (int64_t bin = first; bin < stop; ++bin) {
                    const double weight = lpsd_fftw_smoothing_weight(
                        bin, df, frequency, length_over_fs, sample_rate,
                        beta, halfwidth, response_scale);
                    sum += weight * power[bin];
                    normalization += weight;
                }
                /* Preserve the existing endpoint quadrature and its order:
                 * halve only normalization, never the one-sided numerator. */
                if (first == 0)
                    normalization -= 0.5 * lpsd_fftw_smoothing_weight(
                        0, df, frequency, length_over_fs, sample_rate,
                        beta, halfwidth, response_scale);
                if (n % 2 == 0 && stop == n / 2 + 1)
                    normalization -= 0.5 * lpsd_fftw_smoothing_weight(
                        stop - 1, df, frequency, length_over_fs, sample_rate,
                        beta, halfwidth, response_scale);
                if (!isfinite(normalization) || normalization <= 0.0) return 2;
                deferred_denominators[j] = normalization;
                sum /= normalization;
            } else {
                #pragma omp simd reduction(+:sum)
                for (int64_t bin = first; bin < stop; ++bin)
                    sum += lpsd_fftw_smoothing_weight(
                        bin, df, frequency, length_over_fs, sample_rate,
                        beta, halfwidth, response_scale) * power[bin];
                sum /= denominators[j];
            }
        }
        output[indices[j]] = sum;
    }
    return 0;
}

LPSD_TARGET_CLONES int lpsd_fftw_smoothing_apply(
    const double *power, int64_t n, double sample_rate, double beta, double halfwidth,
    int64_t begin, int64_t end, const int64_t *indices, const int64_t *firsts,
    const int64_t *stops, const int64_t *lengths, const double *frequencies,
    const double *denominators, const double *const *cached, double *output)
{
    return lpsd_fftw_smoothing_apply_impl(
        power, n, sample_rate, beta, halfwidth, begin, end, indices, firsts,
        stops, lengths, frequencies, denominators,
        (const void *const *)cached, output, 0, NULL);
}

/* Explicit alternative for a narrower persistent cache. It keeps the same
 * kernel but rounds its normalized, nonnegative weights once to float32.
 * The double-cache ABI remains available as the numerical reference. */
LPSD_TARGET_CLONES int lpsd_fftw_smoothing_apply_f32(
    const double *power, int64_t n, double sample_rate, double beta, double halfwidth,
    int64_t begin, int64_t end, const int64_t *indices, const int64_t *firsts,
    const int64_t *stops, const int64_t *lengths, const double *frequencies,
    const double *denominators, const float *const *cached, double *output)
{
    return lpsd_fftw_smoothing_apply_impl(
        power, n, sample_rate, beta, halfwidth, begin, end, indices, firsts,
        stops, lengths, frequencies, denominators,
        (const void *const *)cached, output, 1, NULL);
}

/* Additive first-use-normalization variants. Initialize an uncached kernel's
 * denominator to NaN to defer preparation. These functions write only those
 * pending denominators; valid prepared ones and cached weights are unchanged.
 * Parallel calls must address disjoint [begin,end) ranges. A pipeline itself
 * is not reentrant while denominators or its output buffer are being written.
 * The original const-denominator entry points above retain their contract. */
LPSD_TARGET_CLONES int lpsd_fftw_smoothing_apply_deferred(
    const double *power, int64_t n, double sample_rate, double beta, double halfwidth,
    int64_t begin, int64_t end, const int64_t *indices, const int64_t *firsts,
    const int64_t *stops, const int64_t *lengths, const double *frequencies,
    double *denominators, const double *const *cached, double *output)
{
    return lpsd_fftw_smoothing_apply_impl(
        power, n, sample_rate, beta, halfwidth, begin, end, indices, firsts,
        stops, lengths, frequencies, denominators,
        (const void *const *)cached, output, 0, denominators);
}

LPSD_TARGET_CLONES int lpsd_fftw_smoothing_apply_deferred_f32(
    const double *power, int64_t n, double sample_rate, double beta, double halfwidth,
    int64_t begin, int64_t end, const int64_t *indices, const int64_t *firsts,
    const int64_t *stops, const int64_t *lengths, const double *frequencies,
    double *denominators, const float *const *cached, double *output)
{
    return lpsd_fftw_smoothing_apply_impl(
        power, n, sample_rate, beta, halfwidth, begin, end, indices, firsts,
        stops, lengths, frequencies, denominators,
        (const void *const *)cached, output, 1, denominators);
}

#endif
