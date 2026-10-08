# Native LPSD kernels

These sources implement the LPSD 1.0.6 estimator. They keep its frequency
plan, segment starts, normalization and legacy mean/variance recurrence.
The polynomial routines are included directly from the unchanged original
`lpsd/c_sources/polyreg.c`; there is no second maintained copy.

## Computation modes

| Mode | Computation | Floating-point behavior |
| --- | --- | --- |
| 0 / scalar | Original polynomial residuals and serial Fourier accumulation; one PSD projection instead of two | Closest reproduction of the original C path |
| 1 / SIMD | Original residuals followed by vectorized Fourier reductions | Reordered sums can change rounding |
| 2 / projected | Prepare detrended coefficients once per frequency, then use an input anchor for every segment | Algebraically equivalent; intended for mean removal in automatic mode |

The Python API selects mode 2 only for detrending order 0 in automatic mode.
Explicit projection for order 1 can lose accuracy for a large ramp with a
very small residual. See [numerical limits](../../docs/numerics.md).

Segment starts deliberately use repeated `start += shift`, followed by
`floor(start + 0.5)`. Multiplying the segment number by the shift can select
a different sample through rounding.

The original statistics have two known defects: the mean update uses the
zero-based segment index as divisor, and M2 is overwritten rather than
accumulated. These rules are preserved for compatibility; the returned
standard-deviation columns are not validated statistical uncertainties.

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
  In-place mode 2 reuses the caller's private, writable coefficient arrays,
  saving two length-L double buffers (16L bytes) per active frequency. It
  never changes the input samples. Variance outputs may be NULL when not
  requested; supplied variance outputs are then NAN.

Signatures and return codes are documented in `fast_dft.h`. Argument errors,
unsupported dimensions and failed allocations are returned to the caller.
The original polynomial implementation limits lengths to positive `int`.

## Build and licenses

The build uses `-O3 -fopenmp-simd -ffp-contract=off`, with `-fPIC` on Unix
and `-lm`. It does not use global `-ffast-math`. `--native` adds
`-march=native` for a local CPU; rebuild before using that library on a
machine with different CPU capabilities. SIMD reductions do not create
OpenMP threads. Python controls frequency-level concurrency.

`fast_dft.c` and `fast_dft.h` are GPL-3.0-or-later derivatives of the
original LPSD/LTPDA implementation. Original author credits remain in
the sources. `numpy_kaiser.c` retains its BSD-3-Clause notice, and
`LICENSE.numpy.txt` contains the NumPy license. Preserve these notices in
source and binary distributions. See the repository's root `LICENSE`.
