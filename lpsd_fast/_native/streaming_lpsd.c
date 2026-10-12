/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Bounded-memory Kaiser/order-zero auto-spectrum for long segments.
 * Included after fast_dft.c's ordinary kernel and coefficient generators.
 *
 * The first pass reproduces the original window sums and coefficient
 * projector moments in their original order. The second regenerates those
 * same coefficients in a small buffer, projects them and accumulates the
 * unchanged overlapping segments. No L-sized coefficient array is read or
 * written. Segment starts, local anchors and the legacy statistics recurrence
 * are retained. Blockwise dot reductions change floating-point summation
 * order; this is deliberately a separately reported, tested execution path.
 */

enum { STREAMING_WINDOW_BLOCK = 128, STREAMING_PHASE_BLOCK = 64,
       STREAMING_MAX_DEGREE = 63, STREAMING_MAX_SEGMENTS = 256 };

typedef struct {
    long int length;
    double alpha, beta;
    double coefficients[STREAMING_MAX_DEGREE + 1];
    int degree;
    double phase_scale;
    bool direct_phase;
    double offsets_r[STREAMING_PHASE_BLOCK];
    double offsets_i[STREAMING_PHASE_BLOCK];
    double offsets_phase[STREAMING_PHASE_BLOCK];
} streaming_kaiser_plan;

static LPSD_ALWAYS_INLINE int streaming_kaiser_prepare(
    streaming_kaiser_plan *plan, long int length, double beta, double frequency_bin)
{
    if (length < 2048 || !isfinite(beta) || beta <= 0.0 || beta > 32.0 ||
        !isfinite(frequency_bin)) {
        return 4;
    }
    plan->length = length;
    plan->alpha = (double)length / 2.0;
    plan->beta = beta;
    const double denominator = numpy_i0_scalar(beta);
    const long double a = (long double)beta * (long double)beta / 4.0L;
    long double term = 1.0L / (long double)denominator;
    plan->coefficients[0] = (double)term;
    plan->degree = 0;
    for (int k = 1; k <= STREAMING_MAX_DEGREE; ++k) {
        term *= a / ((long double)k * (long double)k);
        plan->coefficients[k] = (double)term;
        const long double next = term * a /
            ((long double)(k + 1) * (long double)(k + 1));
        const long double ratio = a /
            ((long double)(k + 2) * (long double)(k + 2));
        if (ratio < 1.0L && next <= 1.0e-20L * (1.0L - ratio)) {
            plan->degree = k;
            break;
        }
    }
    if (plan->degree == 0) {
        return 4;
    }
    const double pi = 3.141592653589793238462643383279502884;
    plan->phase_scale = (2.0 * pi) * frequency_bin / (double)length;
    if (!isfinite(plan->phase_scale)) {
        return 4;
    }
    plan->direct_phase = fabs(plan->phase_scale) * (double)(length - 1) > 8192.0;
    for (int j = 0; j < STREAMING_PHASE_BLOCK; ++j) {
        const double phase = plan->phase_scale * (double)j;
        plan->offsets_phase[j] = phase;
        plan->offsets_r[j] = cos(phase);
        plan->offsets_i[j] = sin(phase);
    }
    return 0;
}

static LPSD_ALWAYS_INLINE void streaming_kaiser_window(
    const streaming_kaiser_plan *plan, long int first, long int count, double *window)
{
    double u[STREAMING_WINDOW_BLOCK], value[STREAMING_WINDOW_BLOCK];
    for (long int offset = 0; offset < count;) {
        const int size = count - offset < STREAMING_WINDOW_BLOCK ?
            (int)(count - offset) : STREAMING_WINDOW_BLOCK;
        #pragma omp simd
        for (int j = 0; j < size; ++j) {
            const long int absolute = first + offset + j;
            // The old generator mirrors its left-half values. Reuse that
            // same integer index, rather than recomputing the right half.
            const long int index = absolute <= plan->length / 2 ?
                absolute : plan->length - absolute;
            const double normalized = ((double)index - plan->alpha) / plan->alpha;
            u[j] = 1.0 - normalized * normalized;
            value[j] = plan->coefficients[plan->degree];
        }
        for (int k = plan->degree - 1; k >= 0; --k) {
            const double coefficient = plan->coefficients[k];
            #pragma omp simd
            for (int j = 0; j < size; ++j) {
                value[j] = value[j] * u[j] + coefficient;
            }
        }
        #pragma omp simd
        for (int j = 0; j < size; ++j) {
            const long int index = first + offset + j;
            window[offset + j] = plan->length % 2 == 0 && index == plan->length / 2 ?
                1.0 : value[j];
        }
        offset += size;
    }
}

static LPSD_ALWAYS_INLINE void streaming_kaiser_coefficients(
    const streaming_kaiser_plan *plan, long int first, long int count,
    double *real, double *imag)
{
    if (plan->direct_phase) {
        for (long int j = 0; j < count; ++j) {
            const double phase = plan->phase_scale * (double)(first + j);
            const double window = real[j];
            real[j] = window * cos(phase);
            imag[j] = window * sin(phase);
        }
        return;
    }
    for (long int offset = 0; offset < count;) {
        const long int absolute = first + offset;
        const double phase = plan->phase_scale * (double)absolute;
        const double base_r = cos(phase), base_i = sin(phase);
        const int size = count - offset < STREAMING_PHASE_BLOCK ?
            (int)(count - offset) : STREAMING_PHASE_BLOCK;
        #pragma omp simd
        for (int j = 0; j < size; ++j) {
            const double rr = base_r * plan->offsets_r[j] - base_i * plan->offsets_i[j];
            const double ri = base_i * plan->offsets_r[j] + base_r * plan->offsets_i[j];
            const double delta = (plan->phase_scale * (double)(absolute + j) - phase) -
                plan->offsets_phase[j];
            const double window = real[offset + j];
            real[offset + j] = window * (rr - delta * ri);
            imag[offset + j] = window * (ri + delta * rr);
        }
        offset += size;
    }
}

/* This small observable helper is useful for validating generation at
 * arbitrary long-vector offsets, without allocating the whole vector. */
LPSD_TARGET_CLONES int generate_kaiser_coefficients_range(
    double *real, double *imag, long int length, long int first, long int count,
    double beta, double frequency_bin, bool coefficients)
{
    if (real == NULL || (coefficients && imag == NULL) || length > INT_MAX ||
        first < 0 || first % STREAMING_WINDOW_BLOCK || count < 1 ||
        count > length || first > length - count) {
        return 1;
    }
    streaming_kaiser_plan plan;
    const int status = streaming_kaiser_prepare(&plan, length, beta, frequency_bin);
    if (status != 0) {
        return status;
    }
    streaming_kaiser_window(&plan, first, count, real);
    if (coefficients) {
        streaming_kaiser_coefficients(&plan, first, count, real, imag);
    }
    return 0;
}

static LPSD_TARGET_CLONES int fast_dft_kaiser_streaming_impl(
    double *Pr_r, double *Pr_i, long int *Navs,
    const double *samples, long int nData, long int segLen,
    double beta, double frequency_bin, double olap, int mode,
    long int block_length, long int max_segments,
    double *sum_window, double *sum_squares,
    double *preparation_seconds, double *segments_seconds,
    bool corrected_statistics)
{
    if (Pr_r == NULL || Pr_i == NULL || Navs == NULL || samples == NULL ||
        sum_window == NULL || sum_squares == NULL ||
        preparation_seconds == NULL || segments_seconds == NULL ||
        !isfinite(olap) || olap < 0.0 || olap >= 100.0 ||
        block_length < STREAMING_WINDOW_BLOCK || block_length > 1048576 ||
        block_length % STREAMING_WINDOW_BLOCK ||
        max_segments < 1 || max_segments > STREAMING_MAX_SEGMENTS) {
        return 1;
    }
    *Pr_r = *Pr_i = *sum_window = *sum_squares = NAN;
    *preparation_seconds = *segments_seconds = NAN;
    *Navs = 0;
    if (nData < 1 || nData > INT_MAX || segLen > nData || segLen < 2048) {
        return 2;
    }
    if (mode != 2 && mode != 3) {
        return 4;
    }
    const double ovfact = 1.0 / (1.0 - olap / 100.0);
    const double davg = ((double)(nData - segLen) * ovfact) / segLen + 1.0;
    if (!isfinite(davg) || davg + 0.5 >= (double)INT_MAX) {
        return 2;
    }
    const long int navg = (long int)floor(davg + 0.5);
    if (navg < 1 || navg > nData - segLen + 1) {
        return 2;
    }
    if (navg > max_segments) {
        return 4;
    }
    double shift = navg == 1 ? 1.0 : (double)(nData - segLen) / (double)(navg - 1);
    if (shift < 1.0) {
        shift = 1.0;
    }
    long int starts[STREAMING_MAX_SEGMENTS];
    double anchors[STREAMING_MAX_SEGMENTS];
    long double projections_r[STREAMING_MAX_SEGMENTS] = {0.L};
    long double projections_i[STREAMING_MAX_SEGMENTS] = {0.L};
    double start = 0.0;
    for (long int segment = 0; segment < navg; ++segment) {
        starts[segment] = lpsd_segment_start(start, segment, navg, nData - segLen);
        start += shift;
        if (starts[segment] < 0 || starts[segment] > nData - segLen) {
            return 2;
        }
        anchors[segment] = samples[starts[segment]];
    }

    const double preparation_started = monotonic_seconds();
    streaming_kaiser_plan plan;
    int status = streaming_kaiser_prepare(&plan, segLen, beta, frequency_bin);
    if (status != 0) {
        return status;
    }
    const long int allocated = block_length < segLen ? block_length : segLen;
    double *buffer = (double *)malloc((size_t)allocated * 2 * sizeof(double));
    if (buffer == NULL) {
        return 3;
    }
    double *real = buffer, *imag = buffer + allocated;
    double s1 = 0.0, s2 = 0.0;
    long double sum_r = 0.L, sum_i = 0.L;
    double hr[8] = {0.0}, lr[8] = {0.0}, hi[8] = {0.0}, li[8] = {0.0};
    for (long int first = 0; first < segLen;) {
        const long int count = segLen - first < block_length ? segLen - first : block_length;
        streaming_kaiser_window(&plan, first, count, real);
        for (long int j = 0; j < count; ++j) {
            const double value = real[j], square = value * value;
            s1 += value;
            s2 += square;
        }
        streaming_kaiser_coefficients(&plan, first, count, real, imag);
        if (mode == 3) {
            for (long int offset = 0; offset < count;) {
                const int size = count - offset < 8 ? (int)(count - offset) : 8;
                #pragma omp simd
                for (int j = 0; j < size; ++j) {
                    dd_add(real[offset + j], &hr[j], &lr[j]);
                    dd_add(imag[offset + j], &hi[j], &li[j]);
                }
                offset += size;
            }
        } else {
            for (long int j = 0; j < count; ++j) {
                sum_r += real[j];
                sum_i += imag[j];
            }
        }
        first += count;
    }
    long double mean_r = sum_r / (long double)segLen;
    long double mean_i = sum_i / (long double)segLen;
    double mean_r_hi = 0.0, mean_r_lo = 0.0, mean_i_hi = 0.0, mean_i_lo = 0.0;
    if (mode == 3) {
        double total_r_hi = 0.0, total_r_lo = 0.0, total_i_hi = 0.0, total_i_lo = 0.0;
        for (int j = 0; j < 8; ++j) {
            dd_add(hr[j], &total_r_hi, &total_r_lo);
            dd_add(lr[j], &total_r_hi, &total_r_lo);
            dd_add(hi[j], &total_i_hi, &total_i_lo);
            dd_add(li[j], &total_i_hi, &total_i_lo);
        }
        dd_divide(total_r_hi, total_r_lo, (double)segLen, &mean_r_hi, &mean_r_lo);
        dd_divide(total_i_hi, total_i_lo, (double)segLen, &mean_i_hi, &mean_i_lo);
    }
    const double segments_started = monotonic_seconds();
    *preparation_seconds = segments_started - preparation_started;
    for (long int first = 0; first < segLen;) {
        const long int count = segLen - first < block_length ? segLen - first : block_length;
        streaming_kaiser_window(&plan, first, count, real);
        streaming_kaiser_coefficients(&plan, first, count, real, imag);
        if (mode == 3) {
            #pragma omp simd
            for (long int j = 0; j < count; ++j) {
                real[j] = dd_subtract(real[j], mean_r_hi, mean_r_lo);
                imag[j] = dd_subtract(imag[j], mean_i_hi, mean_i_lo);
            }
        } else {
            for (long int j = 0; j < count; ++j) {
                real[j] = (long double)real[j] - mean_r;
                imag[j] = (long double)imag[j] - mean_i;
            }
        }
        for (long int segment = 0; segment < navg; ++segment) {
            const double *input = samples + starts[segment] + first;
            const double anchor = anchors[segment];
            double rr = 0.0, ri = 0.0;
            #pragma omp simd reduction(+:rr,ri)
            for (long int j = 0; j < count; ++j) {
                const double value = input[j] - anchor;
                rr += real[j] * value;
                ri += imag[j] * value;
            }
            projections_r[segment] += rr;
            projections_i[segment] += ri;
        }
        first += count;
    }
    double Mr_r = 0.0, Mr_i = 0.0, unused_r = 0.0, unused_i = 0.0;
    for (long int segment = 0; segment < navg; ++segment) {
        const double rr = (double)projections_r[segment], ri = (double)projections_i[segment];
        update_statistics(segment, navg, rr, ri, rr, ri, false,
                          corrected_statistics,
                          &Mr_r, &Mr_i, &unused_r, &unused_i);
    }
    *segments_seconds = monotonic_seconds() - segments_started;
    free(buffer);
    *Pr_r = Mr_r;
    *Pr_i = Mr_i;
    *Navs = navg;
    *sum_window = s1;
    *sum_squares = s2;
    return 0;
}

LPSD_TARGET_CLONES int fast_dft_kaiser_streaming(
    double *Pr_r, double *Pr_i, long int *Navs,
    const double *samples, long int nData, long int segLen,
    double beta, double frequency_bin, double olap, int mode,
    long int block_length, long int max_segments,
    double *sum_window, double *sum_squares,
    double *preparation_seconds, double *segments_seconds)
{
    return fast_dft_kaiser_streaming_impl(Pr_r, Pr_i, Navs,
        samples, nData, segLen, beta, frequency_bin, olap, mode,
        block_length, max_segments, sum_window, sum_squares,
        preparation_seconds, segments_seconds, false);
}

LPSD_TARGET_CLONES int fast_dft_kaiser_streaming_corrected(
    double *Pr_r, double *Pr_i, long int *Navs,
    const double *samples, long int nData, long int segLen,
    double beta, double frequency_bin, double olap, int mode,
    long int block_length, long int max_segments,
    double *sum_window, double *sum_squares,
    double *preparation_seconds, double *segments_seconds)
{
    return fast_dft_kaiser_streaming_impl(Pr_r, Pr_i, Navs,
        samples, nData, segLen, beta, frequency_bin, olap, mode,
        block_length, max_segments, sum_window, sum_squares,
        preparation_seconds, segments_seconds, true);
}
