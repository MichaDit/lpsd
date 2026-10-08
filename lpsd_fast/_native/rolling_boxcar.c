/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Overlap reuse for an order-zero Boxcar auto spectrum. The record starts,
 * frequency, order of statistics updates and normalization stay unchanged.
 * Only the way each segment's Fourier projection is evaluated changes.
 *
 * For S(s) = sum_j (x[s+j]-a) exp(i*w*j), an advance h obeys
 * S(s+h) = exp(-i*w*h) * (S(s) - outgoing + exp(i*w*L)*incoming).
 * The unweighted anchored sum advances with the same boundaries; subtracting
 * its product with the coefficient mean applies the order-zero projector.
 * Every 32 segments all three sums and the anchor are rebuilt from the data.
 * There is no signal decimation, FFT-bin rounding or unbounded recurrence.
 *
 * Included after fast_dft_impl so the extreme-value fallback uses the
 * ordinary projected kernel, including its finite/nonfinite conventions.
 */

static LPSD_ALWAYS_INLINE void boxcar_rebase(const double *x,
                                             const double *cr, const double *ci,
                                             long int length, double anchor,
                                             double *rr, double *ri, double *ss,
                                             double *error_mass)
{
    double r = 0.0, im = 0.0, sum = 0.0, mass = 0.0;
    #pragma omp simd reduction(+:r,im,sum,mass)
    for (long int j = 0; j < length; ++j) {
        const double value = x[j] - anchor;
        r += value * cr[j];
        im += value * ci[j];
        sum += value;
        mass += fabs(value);
    }
    *rr = r;
    *ri = im;
    *ss = sum;
    *error_mass = mass;
}

static LPSD_ALWAYS_INLINE void boxcar_boundary(const double *outgoing,
                                               const double *incoming,
                                               const double *cr, const double *ci,
                                               long int shift, double anchor,
                                               double *out_r, double *out_i,
                                               double *in_r, double *in_i,
                                               double *delta_sum, double *error_mass)
{
    double or_ = 0.0, oi = 0.0, ir = 0.0, ii = 0.0, ds = 0.0, mass = 0.0;
    #pragma omp simd reduction(+:or_,oi,ir,ii,ds,mass)
    for (long int j = 0; j < shift; ++j) {
        const double old = outgoing[j] - anchor;
        const double next = incoming[j] - anchor;
        or_ += old * cr[j];
        oi += old * ci[j];
        ir += next * cr[j];
        ii += next * ci[j];
        ds += next - old;
        mass += fabs(next) + fabs(old);
    }
    *out_r = or_;
    *out_i = oi;
    *in_r = ir;
    *in_i = ii;
    *delta_sum = ds;
    *error_mass += mass;
}

/* Plain Boxcar coefficients are bounded by one. Compensated sums suffice
 * even when C long double is software binary128 or has only 53 mantissa
 * bits. Eight independent compensated lanes allow SIMD without fast-math. */
static LPSD_ALWAYS_INLINE void boxcar_coefficient_means(const double *cr,
                                                        const double *ci,
                                                        long int length,
                                                        double *mr, double *mi)
{
    enum { LANES = 8 };
    double hr[LANES] = {0.0}, lr[LANES] = {0.0};
    double hi[LANES] = {0.0}, li[LANES] = {0.0};
    for (long int start = 0; start < length;) {
        const int count = length - start < LANES ? (int)(length - start) : LANES;
        #pragma omp simd
        for (int j = 0; j < count; ++j) {
            dd_add(cr[start+j], &hr[j], &lr[j]);
            dd_add(ci[start+j], &hi[j], &li[j]);
        }
        start += count;
    }
    double sr = 0.0, er = 0.0, si = 0.0, ei = 0.0;
    for (int j = 0; j < LANES; ++j) {
        dd_add(hr[j], &sr, &er);
        dd_add(lr[j], &sr, &er);
        dd_add(hi[j], &si, &ei);
        dd_add(li[j], &si, &ei);
    }
    double qr, qr_low, qi, qi_low;
    dd_divide(sr, er, (double)length, &qr, &qr_low);
    dd_divide(si, ei, (double)length, &qi, &qi_low);
    *mr = qr + qr_low;
    *mi = qi + qi_low;
}

LPSD_TARGET_CLONES int fast_dft_boxcar(double *Pr_r, double *Pr_i,
                                      double *Vr_r, double *Vr_i, long int *Navs,
                                      const double *x, long int nData, long int segLen,
                                      double *Cr, double *Ci, double olap,
                                      double frequency_bin, int mode, bool statistics,
                                      double *preparation_seconds, double *segments_seconds,
                                      long int *rebase_count)
{
    if (Pr_r == NULL || Pr_i == NULL || Navs == NULL || rebase_count == NULL ||
        x == NULL || Cr == NULL || Ci == NULL ||
        (statistics && (Vr_r == NULL || Vr_i == NULL)) ||
        !isfinite(frequency_bin) || !isfinite(olap) || olap < 0.0 || olap >= 100.0 ||
        ((preparation_seconds == NULL) != (segments_seconds == NULL))) {
        return 1;
    }
    if (nData < 1 || nData > INT_MAX || segLen < 1 || segLen > nData ||
        segLen > INT_MAX) {
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
    double shift = navg == 1 ? 1.0 : (double)(nData - segLen) / (double)(navg - 1);
    if (shift < 1.0) {
        shift = 1.0;
    }
    const bool profile = preparation_seconds != NULL;
    const double preparation_started = profile ? monotonic_seconds() : 0.0;
    double segments_started = preparation_started;
    *rebase_count = -1;
    /* The reuse must reduce arithmetic enough to cover its extra states.
     * The Python eligibility check avoids entering here for ordinary cases;
     * this native guard also makes direct calls safe and deterministic. */
    if (segLen < 256 || navg < 32 || ceil(shift) > (double)segLen * 0.25) {
        return fast_dft_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs, x, x,
                             nData, segLen, Cr, Ci, olap, 0, false, mode,
                             statistics, true, Cr, Ci,
                             preparation_seconds, segments_seconds, NAN, NULL);
    }

    *Pr_r = *Pr_i = NAN;
    if (Vr_r != NULL) *Vr_r = NAN;
    if (Vr_i != NULL) *Vr_i = NAN;
    *Navs = 0;
    double mean_r, mean_i;
    boxcar_coefficient_means(Cr, Ci, segLen, &mean_r, &mean_i);
    const double pi = 3.141592653589793238462643383279502884;
    const double scale = (2.0 * pi) * frequency_bin / (double)segLen;
    const double wrap_r = cos(scale * (double)segLen);
    const double wrap_i = sin(scale * (double)segLen);
    /* A conservative cancellation trigger, not a universal error proof.
     * It compares the projection to the anchored absolute data mass used
     * since the last rebuild. Departed large transients remain in this mass
     * so their residual rounding cannot masquerade as later quiet samples.
     * Phase magnitude covers arbitrary noninteger frequency coefficients. */
    const double cancellation_scale = 65536.0 * DBL_EPSILON *
                                        (1.0 + fabs(scale * (double)segLen));
    /* Keep squared PSDs, or the legacy fourth-power variance intermediate,
     * away from exponent boundaries in this alternate accumulation path. */
    const double projection_limit = .25 * (statistics ? sqrt(sqrt(DBL_MAX)) : sqrt(DBL_MAX));
    segments_started = profile ? monotonic_seconds() : 0.0;
    if (profile) *preparation_seconds = segments_started - preparation_started;

    enum { REBASE_INTERVAL = 32 };
    double start = 0.0, anchor = 0.0, r = 0.0, im = 0.0, sum = 0.0, error_mass = 0.0;
    double Mr_r = 0.0, Mr_i = 0.0, M2_r = 0.0, M2_i = 0.0, first_amplitude = 0.0;
    long int previous_start = 0;
    *rebase_count = 0;
    for (long int ii = 0; ii < navg; ++ii) {
        /* This repeated addition and rounding is exactly the original
         * start arithmetic; ii*shift is not interchangeable with it. */
        const long int istart = (long int)floor(start + 0.5);
        start += shift;
        if (istart < 0 || istart > nData - segLen) {
            return 2;
        }
        if (ii % REBASE_INTERVAL == 0) {
            anchor = x[istart];
            boxcar_rebase(x + istart, Cr, Ci, segLen, anchor, &r, &im, &sum, &error_mass);
            ++*rebase_count;
        } else {
            const long int h = istart - previous_start;
            if (h < 1 || h >= segLen) {
                return 2;
            }
            double old_r, old_i, new_r, new_i, ds;
            boxcar_boundary(x + previous_start, x + previous_start + segLen,
                             Cr, Ci, h, anchor, &old_r, &old_i, &new_r, &new_i, &ds,
                             &error_mass);
            const double middle_r = r - old_r + wrap_r * new_r - wrap_i * new_i;
            const double middle_i = im - old_i + wrap_i * new_r + wrap_r * new_i;
            const double rotation_r = Cr[h], rotation_i = Ci[h];
            r = rotation_r * middle_r + rotation_i * middle_i;
            im = rotation_r * middle_i - rotation_i * middle_r;
            sum += ds;
        }
        previous_start = istart;
        const double rr = r - sum * mean_r;
        const double ri = im - sum * mean_i;
        const double amplitude = fabs(rr) + fabs(ri);
        /* ii=1 performs P0+(P1-P0), discarding the first power through a
         * potentially ill-conditioned subtraction. Preserve the direct
         * evaluation when a large first transient precedes a quiet segment.
         * An amplitude ratio avoids forming extra powers near overflow. */
        const bool first_reset_cancellation = ii == 1 && first_amplitude > 0.0 &&
                                               amplitude <= .001 * first_amplitude;
        if (ii == 0) first_amplitude = amplitude;
        if (!isfinite(r) || !isfinite(im) || !isfinite(sum) ||
            !isfinite(rr) || !isfinite(ri) || !isfinite(error_mass) ||
            amplitude >= projection_limit || first_reset_cancellation ||
            (error_mass > 0.0 && amplitude <= cancellation_scale * error_mass)) {
            /* Intermediate unprojected states have less exponent headroom
             * than projected dots for some finite extreme signals. Severe
             * cancellation also deserves the direct projected evaluation.
             * Recompute the whole frequency, retaining every mean update. */
            const double wasted_preparation = profile ? *preparation_seconds : 0.0;
            const double wasted_segments = profile ? monotonic_seconds() - segments_started : 0.0;
            *rebase_count = -1;
            const int status = fast_dft_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs, x, x,
                                            nData, segLen, Cr, Ci, olap, 0, false, mode,
                                            statistics, true, Cr, Ci,
                                            preparation_seconds, segments_seconds, NAN, NULL);
            if (profile) {
                *preparation_seconds += wasted_preparation;
                *segments_seconds += wasted_segments;
            }
            return status;
        }
        update_original_statistics(ii, rr, ri, rr, ri,
                                     statistics && ii == navg - 1,
                                     &Mr_r, &Mr_i, &M2_r, &M2_i);
    }
    if (profile) *segments_seconds = monotonic_seconds() - segments_started;
    *Pr_r = Mr_r;
    *Pr_i = Mr_i;
    if (statistics) {
        *Vr_r = M2_r / (navg - 1);
        *Vr_i = M2_i / (navg - 1);
    }
    *Navs = navg;
    return 0;
}
