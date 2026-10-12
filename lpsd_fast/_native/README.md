# Native LPSD kernels

These sources keep LPSD 1.0.6's frequency plan, valid segment starts and
normalization. The existing exports retain its legacy mean/variance recurrence.
Additive `*_corrected` exports average all actual segments and accumulate the
real sample variance when requested; these support the current Python default
`statistics='corrected'`. The numerical computation mode is independent of
this statistics choice. See the [statistics contract](../../docs/statistics.md).
The polynomial routines are included directly from the unchanged original
`lpsd/c_sources/polyreg.c`; there is no second maintained copy.

## Computation modes

| Mode | Computation | Floating-point behavior |
| --- | --- | --- |
| 0 / scalar | Original polynomial residuals and serial Fourier accumulation; one PSD projection instead of two | Closest reproduction of the original C path |
| 1 / SIMD | Original residuals followed by vectorized Fourier reductions | Reordered sums can change rounding |
| 2 / projected | Prepare detrended coefficients once per frequency, then use an input anchor for every segment | Algebraically equivalent; intended for mean removal in automatic mode |
| 3 / compensated | Prepare the order-0 mean projector with compensated FP64 pairs, then use the same anchored segment dots | Avoids software binary128 arithmetic; opt-in alternative to mode 2 |

The Python API selects mode 2 only for detrending order 0 in automatic mode.
Explicit projection for order 1 can lose accuracy for a large ramp with a
very small residual. See [numerical limits](../../docs/numerics.md).

On targets where C `long double` has the same precision and exponent range
as `double`, mode 1 with order-0 detrending subtracts the original mean
directly in the SIMD dot. It preserves `polyreg0`'s serial sum,
reciprocal-then-multiply mean and double residual rounding. No projected
coefficients are substituted. The length-sized residual vectors and their
write/read pass are omitted; the scalar mode and wider long-double targets
retain the existing path. `native_inline_mean_supported()` reports this
compiled capability, and public results report whether the call used it in
`inline_mean_residual`. A native test compares both channel projections with
the original residual-vector helpers bitwise, including odd vector tails,
DC plus nanovolt noise, constants and a large starting transient.

Mode 3 keeps an upper and lower FP64 part for each coefficient sum and for
the mean. It uses independent SIMD lanes, compensated division and
compensated subtraction. Extreme coefficient magnitudes fall back to the
original long-double preparation before any coefficients are modified.
`native_long_double_mantissa_bits()` returns the compiler's actual
`LDBL_MANT_DIG`, allowing the caller to use this alternative only where
long double has more than 64 significand bits. Its value must not be
inferred from `sizeof(long double)` or the processor family. Mode 3 is
limited to detrending order 0; the other original modes remain available.

Segment starts deliberately use `floor(start + 0.5)` followed by the repeated
`start += shift` update for the next segment. Multiplying the segment number
by the shift can select a different sample through rounding. Every previously
valid start is retained, including a final start that rounds below `N - L`.

There is one narrow endpoint repair shared by the direct and rolling kernels:
if the **last** segment's finite rounded start would exceed `N - L`, use the
exact endpoint `N - L`. Negative rounded starts, nonfinite starts and nonfinal out-of-bounds starts
are still rejected. Bounds are checked before converting to a C `long`.
This avoids cumulative binary64 drift aborting a valid large plan. For
`N = 1,000,000,000`, `L = 146` and `K = 29,251,388`, the recorded Kaiser
overlap produces a final recursive start of `999,999,854.5296893`, which
formerly rounded one sample beyond the endpoint `999,999,854`. No earlier
start, segment count, shift, projection, normalization or statistics update
is changed. The helper regression walks this actual recurrence using constant
memory; it does not allocate a billion-sample signal.

The original statistics have two known defects: the mean update uses the
zero-based segment index as divisor, and M2 is overwritten rather than
accumulated. The legacy exports preserve these rules for compatibility.
Corrected exports use the actual count `ii + 1` and accumulate M2. The Python
caller converts corrected sample variance into a nominal SEM; for CSD this is
a real RMS complex error. A single segment has a valid mean and undefined
sample variance (`NaN`). No correlation correction for overlapping segments
or colored noise is applied, so SEM is not a validated measurement uncertainty.

## Preparation and profiling exports

- `window_sums` replaces Python's sequential window sums without changing
  their order of addition.
- `generate_kaiser` computes `np.kaiser(L + 1, beta)[:-1]`, using the
  NumPy 2.3.5/Cephes approximation coefficients, symmetry and SIMD blocks.
- `generate_coefficients` computes coefficients for the actual fractional
  bin, with independent phases and paired sine/cosine where the compiler
  supports it. It does not round frequencies to FFT bins.
- `generate_window` additionally provides NumPy's symmetric Hann, Hamming,
  Blackman, Bartlett and boxcar definitions. Kaiser alone keeps the original
  LPSD periodic convention. Native libm can differ from NumPy by a few ulps.
- `fast_dft_profile` performs the same calculation with three timer reads
  per frequency. `fast_dft` contains no timing calls on its execution path.
- `fast_dft_selected` and `fast_dft_selected_profile` add optional variance
  calculation and in-place coefficient projection without changing the
  existing exports. Skipping variance retains every mean update and segment.
  In-place modes 2/3 reuse the caller's private, writable coefficient arrays,
  saving two length-L double buffers (16L bytes) per active frequency. It
  never changes the input samples. Variance outputs may be NULL when not
  requested; supplied variance outputs are then NAN.
- `fast_dft_selected_corrected` has the same signature as
  `fast_dft_selected_bounded`, including the input bound and optional timers.
  `fast_dft_boxcar_corrected`, `fast_dft_prepared_order0_corrected` and
  `fast_dft_kaiser_streaming_corrected` have the same argument contracts as
  their legacy counterparts. These exports select the corrected recurrence;
  the existing symbols and original prepared ABI version remain unchanged.
- Selected modes 2/3 PSD share coefficient loads across four segments when
  L is at least 2048. On x86-64, AVX-512 uses eight segments for lengths
  128..255 or at least 1024, followed by four-way and single-segment
  remainders where applicable. Paired probes over all 135 actual lengths
  in the 256..2047 band support retaining the single-segment loop for
  256..1023; those lengths did not benefit from the wider batch on this host.
  The wider batch keeps sixteen SIMD accumulators and eight input anchors
  in the larger register file; narrower targets retain the four-way path.
  Portable x86 builds check AVX-512 at run time, including in target clones,
  because baseline preprocessing cannot identify the selected clone.
  Both batches retain separate anchors, every input sample, the original
  repeated segment-start updates, and the original mean order. No global
  floating-point contraction or statistics reassociation is enabled.
- Only the final assignment to the legacy M2 is computed, since earlier
  M2 assignments are overwritten. This speeds up full-output calls too;
  it deliberately preserves the original variance defect. Corrected full
  outputs accumulate every M2 contribution. Both modes skip M2 entirely
  when no deviation column is requested.
- `generate_coefficients_blocked` is an optional faster coefficient path.
  It directly evaluates one phase per 64-sample block and rotates a short
  offset table with SIMD. A per-sample correction recovers the independently
  rounded phase. Lengths below 256 and total phase above 8192 radians use
  the original independent sine/cosine path. Floating-point multiplication
  can still change very deep spectral sidelobes; use the original generator
  when those differences matter. There is no unbounded phase recurrence.

Signatures and return codes are documented in `fast_dft.h`. Argument errors,
unsupported dimensions and failed allocations are returned to the caller.
The original polynomial implementation limits lengths to positive `int`.

## Build and licenses

The build uses `-O3 -fopenmp-simd -ffp-contract=off`, with `-fPIC` on Unix
and `-lm`. It does not use global `-ffast-math`. Supported ELF/x86 compilers
can generate runtime-dispatched default, AVX2 and AVX512F versions after a
successful compile/link probe enables `LPSD_HAVE_TARGET_CLONES`. The hot
helpers are inlined into each version, so the ISA choice applies to the
actual loops. Other targets keep the compiler's ordinary ISA selection.
`--native` selects the local CPU (`-march=native` on x86 or `-mcpu=native`
on AArch64), without these portable clones; rebuild before using that
library on a machine with different CPU capabilities. SIMD reductions do
not create OpenMP threads. Python controls frequency-level concurrency.

`fast_dft.c` and `fast_dft.h` are GPL-3.0-or-later derivatives of the
original LPSD/LTPDA implementation. Original author credits remain in
the sources. `numpy_kaiser.c` retains its BSD-3-Clause notice, and
`LICENSE.numpy.txt` contains the NumPy license. Preserve these notices in
source and binary distributions. See the repository's root `LICENSE`.
