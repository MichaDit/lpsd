/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef LPSD_FAST_DFT_H
#define LPSD_FAST_DFT_H

#include <stdbool.h>

/* The build enables this only after compiling and linking an ELF/x86
 * target_clones probe. Other targets retain their ordinary compiler ISA. */
#if defined(LPSD_HAVE_TARGET_CLONES) && LPSD_HAVE_TARGET_CLONES && \
    defined(__ELF__) && (defined(__x86_64__) || defined(__i386__))
#define LPSD_TARGET_CLONES \
    __attribute__((target_clones("default", "avx2", "avx512f")))
#else
#define LPSD_TARGET_CLONES
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LPSD_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define LPSD_ALWAYS_INLINE inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Native LPSD 1.0.6-compatible kernel.
 * mode 0: original summation order, one PSD projection instead of two.
 * mode 1: same long-double detrending, explicit SIMD dot-product reduction.
 * mode 2: experimental projection of order-0/1 detrending into the kernel;
 *         SIMD projections use a separate input anchor for every segment.
 * mode 3: order-0 projection with compensated FP64 pairs; avoids software
 *         binary128 arithmetic, while retaining the anchored segment dots.
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
 * inplace=true lets modes 2/3 replace Cr/Ci with projected coefficients. These
 * arrays must be writable, distinct, nonoverlapping and private to the call.
 * The input samples remain unchanged. Modes 0/1 do not modify Cr/Ci.
 * Modes 2/3 PSD may process four or eight segments together to share
 * coefficient loads. Eight-way batches require x86-64 AVX-512 and lengths
 * 128..255 or >=1024; four-way batches remain available for length >=2048
 * and batch remainders. The intervening lengths retain the single-segment loop.
 * Means are still updated in the original segment order.
 * Projected CSD may share these helpers between two/four segments, with
 * a separate anchor per channel and segment. The first two CSD projections
 * retain the original single-segment path to protect the legacy mean reset.
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

/* High-overlap order-0 auto spectrum for a known all-ones window. Cr/Ci must
 * be the unprojected exp(i*w*j) coefficients for frequency_bin and length.
 * It may replace them in place only when falling back to the ordinary
 * projected kernel. Every 32 segments are recomputed directly. The optional
 * timers must be both NULL or both non-NULL. rebase_count is nonnegative for
 * overlap reuse, -1 for a direct fallback. Modes 2/3 are supported. */
LPSD_TARGET_CLONES int fast_dft_boxcar(double *Pr_r, double *Pr_i,
                                      double *Vr_r, double *Vr_i, long int *Navs,
                                      const double *x, long int nData, long int segLen,
                                      double *Cr, double *Ci, double olap,
                                      double frequency_bin, int mode, bool statistics,
                                      double *preparation_seconds, double *segments_seconds,
                                      long int *rebase_count);

/* Replace Python sum(window) and sum(window**2) without changing their
 * sequential addition order. Compile with -ffp-contract=off and without
 * -ffast-math. Returns 0 on success, 1 for invalid pointers/length. */
int window_sums(const double *window, long int length,
                 double *sum_window, double *sum_squares);

/* Inspect actual C precision rather than inferring it from type size or
 * CPU family. Mode 3 is useful when long double is software binary128. */
int native_long_double_mantissa_bits(void);

/* NumPy/Cephes-based periodic Kaiser window, equivalent to
 * np.kaiser(length + 1, beta)[:-1]. The same approximation coefficients
 * are used; scalar libm exp can differ from NumPy by a few ulps.
 * Returns 0 on success, 1 for invalid arguments. */
LPSD_TARGET_CLONES int generate_kaiser(double *window, long int length, double beta);

/* Optional positive-series Kaiser evaluation for long windows and |beta|<=32.
 * Keeps the periodic convention and original normalized coordinate. The
 * adaptive tail bound limits series truncation, not floating-point error.
 * Vectors shorter than 2048, beta=0 and |beta|>32 use generate_kaiser.
 * Original exports and their window evaluation are unchanged. */
LPSD_TARGET_CLONES int generate_kaiser_series(double *window, long int length,
                                              double beta);

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
LPSD_TARGET_CLONES int generate_window(double *window, long int length, int kind, double beta);

/* Periodic sum of signed coefficients[k]*cos(2*pi*k*j/length).
 * Uses one cosine per position and a SIMD Chebyshev harmonic recurrence.
 * count must lie in [1,32]. Rounding differs from independent cosine calls.
 * Returns 0 on success, 1 for invalid arguments. */
LPSD_TARGET_CLONES int generate_cosine_window(double *window, long int length,
                            const double *coefficients, int count);

/* Coefficients for the supplied (possibly fractional) frequency bin:
 * window[j] * exp(+i * ((2*pi)*frequency_bin/length) * j).
 * Independent samples use ordinary libm sin/cos; no phase recurrence.
 * Returns 0 on success, 1 for invalid arguments. */
LPSD_TARGET_CLONES int generate_coefficients(double *Cr, double *Ci, const double *window,
                           long int length, double frequency_bin);

/* SIMD block rotations with a directly evaluated phase every 64 samples.
 * There is no accumulating phase recurrence or change of frequency. Small
 * vectors and unusually large total phases retain independent sin/cos.
 * A small rotation correction recovers independently rounded sample phases.
 * Coefficient rounding can still differ from generate_coefficients; the
 * original function remains available as the independent-phase reference. */
LPSD_TARGET_CLONES int generate_coefficients_blocked(double *Cr, double *Ci, const double *window,
                                   long int length, double frequency_bin);

#ifdef __cplusplus
}
#endif

#endif
