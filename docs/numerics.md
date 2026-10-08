# Numerical behavior and compatibility

`lpsd_fast` preserves the intended LPSD frequency plan, segment starts,
window definition, normalization and output layout. It also deliberately
preserves the original C core's mean/variance recurrence, including the
defects described below. The original `lpsd` module and its defaults remain
available unchanged. Faster arithmetic is selected through the separate
`lpsd_fast` API.

## Choosing a kernel

| Kernel | Segment computation | Appropriate use |
|---|---|---|
| `scalar` | Original long-double detrending and serial Fourier accumulation order; original window and coefficient path | Closest compatibility with the original implementation |
| `simd` | Original long-double residual detrending, followed by a SIMD Fourier reduction | Faster reductions with changed rounding order |
| `projected` | Preprojected Fourier coefficients and locally anchored input; orders 0 and 1 only | Explicit numerical tradeoff; order 1 remains experimental |
| `auto` | `projected` for mean removal (order 0), otherwise `simd` | Default optimized choice, with the order-1 stability fallback |

Frequency-level workers do not combine partial results across frequencies.
They therefore do not inherently change each frequency's reduction order.
SIMD and coefficient projection do change arithmetic order within a frequency.
No global `-ffast-math` is used, and fused multiply-add contraction is disabled
in the supplied build configuration. CPU, compiler, math-library and
long-double-format changes can still affect results.

The native Kaiser path implements the upstream periodic window
`np.kaiser(L + 1, beta)[:-1]`. It uses NumPy's documented source approximation
and coefficients, evaluated in C. The faster Fourier-coefficient path keeps
the fractional bin and the phase expression `(2*pi*m/L)*j`; it does not snap
frequencies to integer DFT bins or use a phase recurrence. Scalar mode retains
the original NumPy window/complex-exponential route. These implementation
choices and successful comparisons on one build are not a cross-platform
bitwise guarantee.

## Why the projected mean-removal path is faster

For one segment, write the complex windowed coefficient as

\[
c_n = w_n e^{i\omega n},\qquad
q_n = c_n - \frac{1}{L}\sum_{r=0}^{L-1}c_r.
\]

Then, in exact arithmetic,

\[
\sum_n c_n (x_{s+n}-\bar{x}_s)
 = \sum_n q_n x_{s+n}
 = \sum_n q_n (x_{s+n}-x_s).
\]

`q` is prepared once per output frequency. Each segment needs only the
anchored dot product, avoiding repeated long-double mean computation and
residual materialization. Coefficient moments are accumulated in long double;
the projected coefficients and Fourier reduction use double precision.

Subtracting the segment's first sample before the dot product matters for a
large DC level with small fluctuations. A raw Fourier dot minus a separately
computed large DC correction can lose the fluctuations through cancellation.
The anchored formula avoids that particular subtraction of two large sums.
It still has coefficient and reduction rounding error, especially when the
true Fourier component is nearly zero.

Linear detrending projects out both a constant and a centered linear term.
A constant anchor alone does not eliminate a large linear ramp from the dot
product. In an independent small test with a ramp spanning -10 V to +10 V
and 1 nV noise, explicit order-1 projection reached approximately `8.65e-6`
relative power error in one bin, versus approximately `1.56e-8` for the
original/SIMD residual-detrending calculation. This is why `auto` uses SIMD
residual detrending for order 1. The reference experiment held the binary64
Fourier coefficients fixed and used an independently centered long-double
residual and dot product; it was a native-kernel check, not a separate
validation of the entire coefficient-generation path.

## Recorded validation results

The full suite included noise, tones, impulses, steps, small/odd lengths,
large DC offsets, a large linear ramp with tiny noise, detrending orders up
to 10, different windows/overlaps, input layouts, pandas containers and CSD.
The frequency index was checked exactly. Primary columns were `ps`, `psd`,
`asd`, `enbw` and `asdrms`; uncertainty columns were `ps_std` and `psd_std`.

The recorded gates used relative tolerance `2e-6` for primary columns and
`1e-5` for uncertainty columns, with **zero absolute tolerance**. Exact
bitwise equality was recorded separately. These are regression thresholds,
not universal error bounds. The summaries below are retained in
[the curated results](../benchmarks/results.json).

| Test group | Complete comparisons passing | Primary comparisons passing | Bitwise-equal column arrays |
|---|---:|---:|---:|
| `scalar`, one worker, full suite | 49/49 cases | 49/49 cases | 350/350 |
| `auto`, one worker, full suite | 37/49 cases | 46/49 cases | 296/350 |
| `scalar`, memory-layout suite | 4/4 cases | 4/4 cases | 28/28 |
| `auto`, memory-layout suite | 1/4 cases | 4/4 cases | 22/28; all 20 primary arrays |
| `auto`, one versus eight workers | 10/10 cases | 10/10 cases | 70/70 |

One multicolumn case produced two DataFrames, so the 49 full-suite cases
contained 350 column arrays. The scalar results establish compatibility for
that tested source/build and suite; they do not prove equality for every
possible input or platform.

### Three primary failures near spectral nulls

The fast default did **not** pass every strict comparison. The three primary
failures were tone tests with tiny sidelobes:

| Input | Maximum absolute PSD difference | Maximum relative PSD difference | Maximum absolute ASD difference |
|---|---:|---:|---:|
| Off-bin tone | 4.796e-28 | 6.576e-5 | 2.808e-17 |
| On-bin tone | 5.170e-26 | 5.922e-6 | 1.076e-16 |
| Weak tone beside a strong tone | 2.221e-27 | 1.144e-5 | 4.424e-17 |

These values describe the supplied test signals and their units. Maxima in
different columns need not occur at the same frequency. Small absolute errors
can have large relative errors at a nearly zero component. The tests retain
these failures; no DC-scaled absolute tolerance was added to hide them.

The other **nine** failed full-suite cases differed only in the legacy
uncertainty columns through cancellation when two segments were used. The
three memory-layout failures in `auto` were also confined to those uncertainty
columns. Use `scalar` when matching the original rounding is the requirement;
assess both absolute and relative errors when extremely small sidelobes matter.

## Inherited statistical defects

The original native recurrence initializes the mean from segment zero. For
subsequent zero-based segment index `ii`, it updates the mean with
`Mr += (Xr - Mr) / ii`. A conventional incremental mean would divide by
`ii + 1`. In exact arithmetic, once at least two segments are present, the
existing recurrence drops the first segment's contribution and averages the
remaining segments. Floating-point cancellation can leave additional effects.

The original code also assigns the new second-moment term to `M2` rather than
accumulating it with `+=`. The returned variance consequently does not implement
the usual running sample variance. All optimized modes retain both operations
to keep compatibility with the supplied estimator. `ps_std` and `psd_std`
must therefore not be treated as independently validated statistical
uncertainties. Correcting these recurrences would change the estimator and
requires a separately selected, validated behavior change.

The output precision is also inherited: most real PSD columns are exposed
as float32 after the upstream complex64 storage step, `enbw` remains complex64,
and `asdrms` is float64. Equal final float32 values can conceal smaller
differences in internal double-precision calculations. An independent native
reference is useful alongside public-output comparisons for that reason.
