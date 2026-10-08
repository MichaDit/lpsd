/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef LPSD_FAST_DFT_H
#define LPSD_FAST_DFT_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Native LPSD 1.0.6-compatible kernel.
 * mode 0: original summation order, one PSD projection instead of two.
 * mode 1: same long-double detrending, explicit SIMD dot-product reduction.
 * mode 2: experimental projection of order-0/1 detrending into the kernel;
 *         SIMD projections use a separate input anchor for every segment.
 *
 * Returns 0 on success, 1 for invalid arguments, 2 for unsupported dimensions
 * or invalid segment starts, 3 on allocation failure, 4 for unsupported mode.
 * Frequencies, windows and normalization are supplied by the caller.
 * Segment order and upstream 1.0.6 mean/variance semantics are preserved.
 */
int fast_dft(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
             long int *Navs,
             const double *x1data, const double *x2data,
             long int nData, long int segLen,
             const double *Cr, const double *Ci,
             double olap, int order, bool csd, int mode);

/* Same computation, with only three timer reads per frequency.
 * Preparation includes buffer allocation and projected-kernel preparation.
 * Segments includes the entire segment loop, including detrending in modes
 * 0/1 and the preserved statistics. Frees/output assignment are untimed. */
int fast_dft_profile(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
                     long int *Navs,
                     const double *x1data, const double *x2data,
                     long int nData, long int segLen,
                     const double *Cr, const double *Ci,
                     double olap, int order, bool csd, int mode,
                     double *preparation_seconds, double *segments_seconds);

/* Additive API for selected outputs and private coefficient workspaces.
 * statistics=false skips variance work, retaining the original mean update
 * and Navs exactly. Vr_r/Vr_i may be NULL; non-NULL variance outputs are NAN.
 * inplace=true lets mode 2 replace Cr/Ci with projected coefficients. These
 * arrays must be writable, distinct, nonoverlapping and private to the call.
 * The input samples remain unchanged. Modes 0/1 do not modify Cr/Ci.
 * Existing fast_dft[_profile] retain their original read-only contract. */
int fast_dft_selected(double *Pr_r, double *Pr_i, double *Vr_r, double *Vr_i,
                      long int *Navs,
                      const double *x1data, const double *x2data,
                      long int nData, long int segLen,
                      double *Cr, double *Ci,
                      double olap, int order, bool csd, int mode,
                      bool statistics, bool inplace);

int fast_dft_selected_profile(double *Pr_r, double *Pr_i,
                              double *Vr_r, double *Vr_i, long int *Navs,
                              const double *x1data, const double *x2data,
                              long int nData, long int segLen,
                              double *Cr, double *Ci,
                              double olap, int order, bool csd, int mode,
                              bool statistics, bool inplace,
                              double *preparation_seconds, double *segments_seconds);

/* Replace Python sum(window) and sum(window**2) without changing their
 * sequential addition order. Compile with -ffp-contract=off and without
 * -ffast-math. Returns 0 on success, 1 for invalid pointers/length. */
int window_sums(const double *window, long int length,
                 double *sum_window, double *sum_squares);

/* NumPy/Cephes-based periodic Kaiser window, equivalent to
 * np.kaiser(length + 1, beta)[:-1]. The same approximation coefficients
 * are used; scalar libm exp can differ from NumPy by a few ulps.
 * Returns 0 on success, 1 for invalid arguments. */
int generate_kaiser(double *window, long int length, double beta);

enum lpsd_window_kind {
    LPSD_WINDOW_KAISER = 0,
    LPSD_WINDOW_HANN = 1,
    LPSD_WINDOW_HAMMING = 2,
    LPSD_WINDOW_BLACKMAN = 3,
    LPSD_WINDOW_BARTLETT = 4,
    LPSD_WINDOW_BOXCAR = 5
};

/* Kaiser retains the LPSD periodic convention. The other kinds reproduce
 * NumPy's symmetric length-L definitions (L=1 gives a single one); beta is
 * used only for Kaiser. Native libm can differ from NumPy by a few ulps.
 * Returns 0 on success, 1 for invalid pointers, dimensions or kind. */
int generate_window(double *window, long int length, int kind, double beta);

/* Coefficients for the supplied (possibly fractional) frequency bin:
 * window[j] * exp(+i * ((2*pi)*frequency_bin/length) * j).
 * Independent samples use ordinary libm sin/cos; no phase recurrence.
 * Returns 0 on success, 1 for invalid arguments. */
int generate_coefficients(double *Cr, double *Ci, const double *window,
                           long int length, double frequency_bin);

#ifdef __cplusplus
}
#endif

#endif
