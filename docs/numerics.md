# Numerical behavior and compatibility

`lpsd_fast` preserves the intended LPSD frequency plan, segment starts,
window definition, normalization and default output layout. It also deliberately
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
| `auto` | `projected` for order 0 when C long double has more than 53 significand bits, otherwise original residual detrending with `simd`; independent Fourier phases | Default optimized choice, retaining residual detrending for order 1 |
| `fast` | Projected order 0, residual SIMD for other orders, bounded block rotations; compensated order-0 preparation when C `long double` has more than 64 significand bits | Opt-in lower wall time with additional rounding differences |

`fast` is an explicit arithmetic choice, not an accuracy target. It does not
compare results against a tolerance or rerun suspect frequencies with
`scalar`. Use `scalar` for the closest reproduction of upstream rounding,
or compare the chosen faster mode with that reference using both relative
and application-specific absolute limits. None of the faster modes has a
universal 1% relative-error guarantee for arbitrary signals and spectral nulls.

Frequency-level workers do not combine partial results across frequencies.
They therefore do not inherently change each frequency's reduction order.
SIMD and coefficient projection do change arithmetic order within a frequency.
No global `-ffast-math` is used, and fused multiply-add contraction is disabled
in the supplied build configuration. The compensated preparation explicitly
uses `fma` to recover two division residuals per frequency; this does not
enable contraction elsewhere. CPU, compiler, math-library and
long-double-format changes can still affect results.

### Window definitions and Fourier coefficients

The [API](../lpsd_fast/api.py) recognizes window names and the corresponding
original function objects. Non-Kaiser windows require an explicit overlap.

| Window | Preserved definition |
|---|---|
| `kaiser` / `np.kaiser` | Periodic upstream convention: `np.kaiser(L + 1, beta)[:-1]`, using the original PSLL-to-beta mapping |
| `hann` / `hanning`, `hamming`, `blackman`, `bartlett` | NumPy's symmetric length-`L` window, including its length-one convention |
| `boxcar` / `np.ones` | `L` ones |
| The 19 functions from `lpsd.flattop` | Original periodic cosine-series definitions and signed coefficients |

The supported cosine windows are `SFT3F`, `SFT4F`, `SFT5F`, `SFT3M`, `SFT4M`,
`SFT5M`, `FTNI`, `FTHP`, `FTSR`, `Matlab`, `HFT70`, `HFT95`, `HFT90D`,
`HFT116D`, `HFT144D`, `HFT169D`, `HFT196D`, `HFT223D` and `HFT248D`.
Their names are case-insensitive. The [coefficient table](../lpsd_fast/_windows.py)
is copied from the original [window functions](../lpsd/flattop.py).

All non-scalar kernels use the native path for these recognized windows.
Kaiser uses the NumPy/Cephes approximation coefficients, evaluated with
symmetry and SIMD blocks. The cosine-series path computes one fundamental
cosine and obtains the harmonics through a short Chebyshev recurrence,
then mirrors the periodic window. The recurrence advances across harmonic
order, not across an unbounded sequence of samples. This preserves the
mathematical definitions but changes rounding compared with evaluating
every harmonic in a separate NumPy array. Deep flat-top sidelobes are
particularly sensitive to those differences. Arbitrary callbacks retain
their own definition and are not identified by a matching name; scalar
mode retains the original window-generation path.

The independent native Fourier generator keeps the fractional bin and
phase expression `(2*pi*m/L)*j`, using ordinary sine/cosine evaluations.
`scalar` retains the upstream NumPy complex-exponential route. Neither
rounds the frequency to an integer DFT bin.

`fast` adds the [blocked generator](../lpsd_fast/_native/fast_dft.c). It
evaluates a fresh base phase for every 64 samples and rotates a short table
of offsets with SIMD. A small per-sample correction accounts for the
independently rounded phase expression. Short vectors (`L < 256`) and total
phases above 8192 radians use the independent generator. There is no
unbounded phase recurrence, but multiplication and rotation still change
coefficient rounding. Agreement on one compiler and test set is not a
cross-platform bitwise guarantee.

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
residual materialization. In native mode 2, coefficient moments are accumulated
in long double; the projected coefficients and Fourier reduction use double
precision. The selected-output native entry point reuses private writable
coefficient arrays for this preparation. It can also process four PSD segments
together when `L >= 2048`, sharing coefficient loads while retaining separate
anchors. The same samples, segment starts and mean-update order are retained.

### Compensated preparation on wider-long-double targets

For `kernel='fast'`, order 0 uses native mode 3 if
`native_long_double_mantissa_bits()` reports more than 64 significand bits.
The [compensated implementation](../lpsd_fast/_native/projected_dd.c) stores
upper and lower FP64 parts of the coefficient sums and their means, then
uses compensated division and subtraction. It avoids long-double arithmetic
in the normal preparation path. Extreme coefficient magnitudes can fall back
to the original long-double preparation before coefficients are modified;
this is a preparation fallback, not a scalar spectrum retry.

Two compensated FP64 parts are not a guarantee of binary128's 113-bit
significand, nor of bitwise equality to the original long-double projector.
The arithmetic assumes the usual IEEE rounding environment. The precision
query reports the C type's actual precision; by itself it does not establish
whether a machine implements that type in hardware. The `auto` mode uses mode 2 only when long double has more than 53
significand bits. On targets such as Apple Silicon, where it has 53 bits,
`auto` retains the original residual arithmetic (mode 1) to reproduce
DC-dominated reference results more closely. The opt-in `fast` path still
uses projection there. Mode 3 does not extend the experimental order-1
projector. See [platforms and floating-point ABI](platforms.md) for build
metadata, architecture-specific formats and the status of native validation.

### Cancellation and linear detrending

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

## Selecting PSD, NSD and other outputs

`outputs=None` or `outputs='all'` preserves the original seven columns:
`ps`, `psd`, `ps_std`, `psd_std`, `enbw`, `asd`, `asdrms`. For auto spectra,
select only the required quantities with, for example:

```python
from lpsd_fast import lpsd, lnsd

spectrum = lpsd(x, sample_rate=fs, kernel='fast', outputs=('psd', 'nsd'))
noise = lnsd(x, sample_rate=fs, kernel='fast')
```

`nsd` is an explicit name for the existing `asd`: the square root of the
already rounded PSD, with the same arithmetic and output precision.
`lnsd` fixes `outputs='nsd'` and returns the same frequency-indexed DataFrame
or channel dictionary structure. NSD is restricted to auto spectra; complex
CSD has its existing output selection. With input unit `U` and sample rate
in Hz, PSD has unit `U²/Hz` and NSD has unit `U/√Hz`.

Output selection changes which quantities are calculated, not the frequency
grid, segment count or input samples. Omitting both `ps_std` and `psd_std`
skips the native second-moment calculation. Requested amplitude outputs still
derive from the same rounded PSD. An `enbw`-only request needs window
normalization but no Fourier coefficients or segment DFTs.

Full-output calls also avoid the earlier second-moment assignments that the
original code overwrites. Only the final such assignment affects its returned
variance, so computing it once preserves the original behavior. Consequently,
selecting PSD/NSD alone does not remove an entire per-sample calculation;
the main Fourier work remains. Neither optimization repairs the inherited
statistical defects below.

## Auditing relative and absolute error

The [accuracy audit](../benchmarks/check_accuracy.py) compares the public
PSD and NSD outputs with `scalar` and first checks that scalar matches the
original C implementation on a small anchor case. It records input, source
and binary fingerprints and checks the frequency index exactly. PSD and NSD
are evaluated separately: taking a square root changes error sensitivity,
especially close to zero.

For each finite reference value `r` and candidate value `c`, the combined
criterion is:

\[
|c-r| < \mathrm{relative\_limit}\,|r|
\quad\text{or}\quad
|c-r| \leq \mathrm{absolute\_limit}.
\]

The default relative limit is `0.01`. Equality at 1% does not satisfy the
relative part. `--psd-absolute-limit` and `--nsd-absolute-limit` both default
to zero; an exact reference zero then requires an exact candidate zero.
NaN or infinity fails either criterion. With positive absolute limits, a
near-zero component can satisfy the absolute part even when its relative
error is large or undefined.

For the normalized synthetic cases, one explicit investigation policy is
PSD absolute error at most `1e-24` and NSD absolute error at most `1e-12`,
alongside the strict 1% relative criterion:

```bash
python -m benchmarks.check_accuracy --kernel fast --n 131073 \
  --relative-limit 0.01 \
  --psd-absolute-limit 1e-24 --nsd-absolute-limit 1e-12 \
  --fail-on-limit --output accuracy-fast.json
```

These are selected audit limits in the respective output units, not API
defaults, a physical noise-floor claim, or proven bounds for every signal.
They do not alter the computed spectrum. Scaling the input by a factor `a`
scales PSD by `a²` and NSD by `|a|`; a corresponding absolute-error policy
must be rescaled too. Choose limits from the measurement's units and the
smallest component that matters. A fixed normalized example is not a suitable
absolute tolerance for arbitrary volts, metres or ADC counts.

The report retains the maximum relative error and the count of points at or
above the relative limit even when the combined criterion passes. Exact-zero
changes, nonfinite values, maximum absolute errors and worst-point values are
reported separately. Additional counters distinguish points accepted only
through the absolute limit. `--fail-on-limit` uses the combined status when
an absolute allowance is selected; with both absolute limits zero it retains
the strict comparison. No DC-scaled or peak-scaled floor replaces the
reference values.

### Why a relative-only limit can reject negligible absolute differences

A development audit of the unit-amplitude off-bin tone with HFT248D,
`N=131073`, sample rate 50 Hz, 96 requested frequencies, 16 requested averages,
overlap 0.841 and no detrending found a maximum relative PSD difference of
approximately **1.49%**. At that point the scalar PSD was `3.11844e-29`, the
candidate was `3.07196e-29`, and the absolute difference was `4.64754e-31`.
The maximum absolute difference in that entire case was `3.89500e-29` at a
different frequency. Separate on-bin HFT248D checks at `N=32769` produced
relative errors close to 100% in PSD values around `1e-30`.

Those observations concern cancellation in very small sidelobes. They explain
why the report preserves both kinds of error, and why an absolute allowance
must be explicit. They are development examples, not a final all-cases pass
summary or a worst-case bound. Large relative errors must still be assessed
if those very small components are the quantities being measured.

## Earlier recorded validation results

The following records describe the initial optimized implementation and its
fixed test suite, before the optional `fast` block rotations, compensated
preparation and expanded native cosine-window path. They remain useful
baseline evidence, not a validation tally for the current implementation.

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

The earlier `auto` default did **not** pass every strict comparison. The three
primary failures were tone tests with tiny sidelobes:

| Input | Maximum absolute PSD difference | Maximum relative PSD difference | Maximum absolute ASD difference |
|---|---:|---:|---:|
| Off-bin tone | 4.796e-28 | 6.576e-5 | 2.808e-17 |
| On-bin tone | 5.170e-26 | 5.922e-6 | 1.076e-16 |
| Weak tone beside a strong tone | 2.221e-27 | 1.144e-5 | 4.424e-17 |

These values describe the supplied test signals and their units. Maxima in
different columns need not occur at the same frequency. Small absolute errors
can have large relative errors at a nearly zero component. These are recorded
failures under the original zero-absolute-tolerance gates.

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

The [original core](../lpsd/ltpda_dft.c) also assigns the new second-moment term
to `M2` rather than accumulating it with `+=`. The returned variance
consequently does not implement the usual running sample variance. All
optimized modes retain the mean
recurrence and the final M2 result; intermediate M2 assignments can be omitted
because they are overwritten. `ps_std` and `psd_std` must therefore not be
treated as independently validated statistical
uncertainties. Correcting these recurrences would change the estimator and
requires a separately selected, validated behavior change.

The output precision is also inherited: most real PSD columns are exposed
as float32 after the upstream complex64 storage step, `enbw` remains complex64,
and `asdrms` is float64. Equal final float32 values can conceal smaller
differences in internal double-precision calculations. An independent native
reference is useful alongside public-output comparisons for that reason.
Near the float32 subnormal range, even one output-quantization step can be a
large relative change; absolute error and exact-zero changes remain necessary
parts of the comparison.
