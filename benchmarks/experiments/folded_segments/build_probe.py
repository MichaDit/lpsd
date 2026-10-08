"""Generate an isolated order-zero folded-DFT screening prototype."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True,
                    help='Read-only lpsd source checkout providing fast_dft.c and its includes.')
args = parser.parse_args()
SOURCE = args.source_root.resolve() / 'lpsd_fast/_native/fast_dft.c'
EXPECTED_BASELINE_SHA256 = 'c53af885aed60e812416693518e87b7cbdd975d314d67b5d69f9b20572ad27f9'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_BASELINE_SHA256:
    raise SystemExit('This experiment requires the pinned fast.3 baseline; see README.md.')

base = r'''
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

/* Reuse the already generated/projected coefficients. A single constant
 * rotation exposes their nearly even-real/odd-imaginary symmetry. Keep the
 * original projector's long-double means for the periodic imaginary term. */
static void project_with_means(const double *cr, const double *ci, long int length,
                               double *qr, double *qi, long double *mr, long double *mi)
{
    long double sr = 0.0L, si = 0.0L;
    for (long int j = 0; j < length; ++j) { sr += cr[j]; si += ci[j]; }
    *mr = sr / (long double)length;
    *mi = si / (long double)length;
    for (long int j = 0; j < length; ++j) {
        qr[j] = (double)((long double)cr[j] - *mr);
        qi[j] = (double)((long double)ci[j] - *mi);
    }
}

static int prepare_folded_reusing(const double *qr, const double *qi,
                                  long int length, double m, bool periodic,
                                  long double mean_r, long double mean_i,
                                  double *fr, double *fi, double *center_r, double *fold_mean_i)
{
    const double pi = 3.141592653589793238462643383279502884;
    const double omega = (2.0 * pi) * m / (double)length;
    const double center = .5 * (double)(periodic ? length : length - 1);
    const double rotation_r = cos(omega * center), rotation_i = sin(omega * center);
    const long int pairs = periodic ? (length - 1) / 2 : length / 2;
    const bool has_center = periodic ? length % 2 == 0 : length % 2 != 0;
    #pragma omp simd
    for (long int j = 0; j < pairs; ++j) {
        const long int left = j + (periodic ? 1 : 0);
        const long int right = periodic ? length - left : length - 1 - left;
        const double real_even = .5 * (qr[left] + qr[right]);
        const double imag_even = .5 * (qi[left] + qi[right]);
        const double real_odd = .5 * (qr[left] - qr[right]);
        const double imag_odd = .5 * (qi[left] - qi[right]);
        fr[j] = rotation_r * real_even + rotation_i * imag_even;
        fi[j] = rotation_r * imag_odd - rotation_i * real_odd;
    }
    *center_r = has_center ? rotation_r * qr[length / 2] + rotation_i * qi[length / 2] : 0.0;
    *fold_mean_i = periodic ? (double)((long double)rotation_r * mean_i -
                                       (long double)rotation_i * mean_r) : 0.0;
    return isfinite(*center_r) && isfinite(*fold_mean_i) ? 0 : 1;
}
'''

for periodic in (False, True):
    for width in (1, 4, 8):
        name = f"folded_{'periodic' if periodic else 'symmetric'}_{width}"
        lines = [
            f'static LPSD_ALWAYS_INLINE void {name}(const double **x, long int length,',
            '                  const double *fr, const double *fi, double center_r,',
            '                  double mean_i, double *rr, double *ri)',
            '{',
            f'    const long int pairs = {"(length - 1) / 2" if periodic else "length / 2"};',
            f'    const bool has_center = length % 2 {"== 0" if periodic else "!= 0"};',
        ]
        accumulators = []
        for b in range(width):
            lines += [f'    const double *x{b} = x[{b}];',
                      f'    const double a{b} = x{b}[0];',
                      f'    double r{b} = 0.0, i{b} = 0.0' + (f', s{b} = 0.0;' if periodic else ';')]
            accumulators += [f'r{b}', f'i{b}'] + ([f's{b}'] if periodic else [])
        lines += [f'    #pragma omp simd reduction(+:{",".join(accumulators)})',
                  '    for (long int j = 0; j < pairs; ++j) {',
                  f'        const long int left = j{ " + 1" if periodic else ""};',
                  f'        const long int right = {"length - left" if periodic else "length - 1 - left"};',
                  '        const double real = fr[j], imag = fi[j];']
        for b in range(width):
            lines += [f'        const double vl{b} = x{b}[left] - a{b};',
                      f'        const double vr{b} = x{b}[right] - a{b};',
                      f'        const double vp{b} = vl{b} + vr{b};',
                      f'        const double vm{b} = vl{b} - vr{b};',
                      f'        r{b} = PROBE_ADD_PRODUCT(real, vp{b}, r{b});',
                      f'        i{b} = PROBE_ADD_PRODUCT(imag, vm{b}, i{b});']
            if periodic:
                lines += [f'        s{b} += vp{b};']
        lines += ['    }', '    if (has_center) {']
        for b in range(width):
            lines += [f'        const double vc{b} = x{b}[length / 2] - a{b};',
                      f'        r{b} = PROBE_ADD_PRODUCT(center_r, vc{b}, r{b});']
            if periodic:
                lines += [f'        s{b} += vc{b};']
        lines += ['    }']
        for b in range(width):
            lines += [f'    rr[{b}] = r{b};',
                      f'    ri[{b}] = i{b}' + (f' - mean_i * s{b};' if periodic else ';')]
        lines += ['}\n']
        base += '\n'.join(lines)

base += r'''
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
    double center_r, mean_i;
    int fallback;
#if PROBE_COEFFICIENT_REUSE
    long double coefficient_mean_r, coefficient_mean_i;
    project_with_means(Cr, Ci, segLen, qr, qi, &coefficient_mean_r, &coefficient_mean_i);
    if (mode == 3) prepare_projected_coefficients_dd(Cr, Ci, segLen, qr, qi);
    fallback = prepare_folded_reusing(qr, qi, segLen, frequency_bin, periodic,
                                      coefficient_mean_r, coefficient_mean_i,
                                      fr, fi, &center_r, &mean_i);
#else
    if (mode != 3 || !prepare_projected_coefficients_dd(Cr, Ci, segLen, qr, qi)) {
        prepare_projected_coefficients(Cr, Ci, segLen, 0, qr, qi);
    }
    fallback = prepare_folded(window, segLen, frequency_bin, periodic,
                              fr, fi, &center_r, &mean_i);
#endif
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
'''

for profiled in (False, True):
    profile_args = ', double *preparation_seconds, double *segments_seconds' if profiled else ''
    name = 'probe_folded_dft_profile' if profiled else 'probe_folded_dft'
    timer_args = 'preparation_seconds, segments_seconds' if profiled else 'NULL, NULL'
    base += f'''
int {name}(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
                     long int *Navs,
                     const double *x1data, const double *x2data,
                     long int nData, long int segLen,
                     double *Cr, double *Ci,
                     double olap, int order, bool csd, int mode,
                     bool statistics, const double *window, bool periodic,
                     double frequency_bin, int requested_width, long int *folded_count{profile_args})
{{
    return probe_folded_impl(Pr_r, Pr_i, Vr_r, Vr_i, Navs, x1data, x2data,
                             nData, segLen, Cr, Ci, olap, order, csd, mode,
                             statistics, window, periodic, frequency_bin,
                             requested_width, folded_count, {timer_args});
}}
'''

source_path = HERE / 'folded_probe.c'
source_path.write_text(base)
flags = ['cc', '-shared', '-O3', '-std=c11', '-fPIC', '-fopenmp-simd',
         '-fno-fast-math', '-ffp-contract=off', '-march=native',
         '-DLPSD_HAVE_TARGET_CLONES=0', '-I', str(SOURCE.parent)]
for reused in (0, 1):
    for fused in (0, 1):
        output = HERE / f'libfolded_{"reuse_" if reused else ""}{"fma" if fused else "plain"}.so'
        subprocess.run(flags + [f'-DPROBE_FMA={fused}', f'-DPROBE_COEFFICIENT_REUSE={reused}',
                                str(source_path), '-o', str(output), '-lm'], check=True)
record = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'prototype_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(), 'flags': flags}
(HERE / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
