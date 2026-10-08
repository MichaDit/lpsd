# Folded real-input DFT: rejected screening experiment

This experiment tested a larger arithmetic change than widening the existing
segment reduction. It is **not a production kernel** and is not imported by
`lpsd_fast`.

The native screening did not establish a consistent additional gain over the
bounded FP64 FMA candidate. The extra path was therefore rejected for this
optimization round. No full public-API speedup is claimed for it.

## Result

Two implementations were measured, each with ordinary FP64 and explicit FMA,
and with one, four, or eight independent segments per dot-product batch:

| Preparation | Measured conclusion |
|---|---|
| Generate new centered half-window coefficients with sin/cos | Extra preparation dominates; generally slower than bounded FMA |
| Rotate and symmetrize the existing projected coefficients | Preparation becomes inexpensive, but the additional pair additions, reversed input stream, reductions, and periodic mean correction leave no consistent gain |

The second experiment covered 16 window/length configurations. Even selecting
the fastest folded batch width retrospectively for each case gave geometric
mean speed ratios of **0.8880 for periodic Kaiser** and **0.9518 for symmetric
Hann**, relative to the bounded FMA candidate. Ratios below one mean that the
folded method is slower. The best single fixed choice, explicit FMA with four
segments, achieved **0.8856** across all 16 cases. Individual favorable cases
are retained in the raw report; they do not establish a consistent additional
15% improvement or justify a new production dispatch policy.

Measurements used one process at a time on the shared host. Every method used
the same samples, coefficients, frequency, segment starts, overlap, and
normalization. The reports contain three ordinary, uninstrumented wall-time
repetitions per case/method. Preparation and segment profiles are separate
calls and are diagnostic; their single-call timings can be noisier than the
ordinary repetitions. The measured native-frequency wall time includes
projection preparation, temporary allocation, segment work, and statistics.
The common original window and Fourier coefficient generation is excluded.
Thus these are native screening results, not complete Python API measurements.

`micro-results.json` records the initial centered-trigonometric prototype.
`micro-results-reuse.json` records coefficient reuse. Each report preserves
the measured library hashes, prototype hash, all repetitions, and separately
recorded profiles.

## Mathematics

For a symmetric window, let `c=(L-1)/2`, `v_j=x_j-x_0`, and rotate the complex
coefficient vector by `exp(-i*omega*c)`. Its real part is even and its imaginary
part is odd. After order-zero projection, a pair `j,k=L-1-j` contributes

```
real += projected_even_real[j] * (v_j + v_k)
imag += odd_imag[j]            * (v_j - v_k)
```

An odd-length window has one central sample. For an auto spectrum, the common
constant Fourier phase rotation does not change the mathematical power.

The upstream periodic Kaiser convention instead has `w_j=w_(L-j)`. Here the
pairs are `j,k=L-j`, centered at `L/2`. The unpaired sample `j=0` has zero
anchored input, but its contribution to the **coefficient mean must remain**.
In particular, the projected imaginary sum needs

```
imag -= rotated_coefficient_mean_imag * sum(v_j)
```

That additional real sum is required even though the direct unpaired
sample contributes zero. At small Kaiser beta the correction is not
necessarily small.

The coefficient-reuse variant rotates and symmetrizes the actual supplied
projected `cr/ci` vectors. It obtains the original coefficient means in the
same long-double summation pass as the original projector. Only two scalar
trigonometric calls are needed for the fixed center rotation; all pair
preparation uses additions and multiplications.

## Numerical screening and limits

Both preparation variants passed **344 configurations × 2 arithmetic modes =
688 comparisons**. Of these, 672 numerical comparisons used a strict
relative PSD/NSD limit below 1% **or** explicit synthetic absolute limits of
`1e-24` for PSD and `1e-12` for NSD. The other 16 checked preservation of
exceptional-value classes rather than applying a finite-value tolerance. The recorded
comparisons are against the retained native projected implementation with
identical supplied coefficients, not against an independent exact DFT or a
new public-API scalar audit.

The cases include odd and even lengths from 3 through 65,537; Kaiser beta 0,
1, 23.7, 32, and 50; symmetric Hann/Hamming/Blackman; fractional and integer
frequency bins; DC plus nanovolt and 16-ULP noise; large linear ramps; tones;
alternating samples; initial/departed transients; exact constants/zeros; and
overflow/underflow scales. There were no changes in NaN/infinity classes.

Large relative differences at deep spectral nulls remain visible. For example,
coefficient reuse at `L=4097`, beta 32, and bin 10,000 changed a tiny PSD by
approximately 70.4%, while its absolute difference was approximately
`1.06e-27` and the NSD difference approximately `1.18e-14`. Such points passed
only through the stated absolute allowance. The allowance belongs to these
normalized synthetic cases; it is not a unit-independent physical noise
floor or an API accuracy promise.

The first original eight- or four-segment batch, or the first two singleton
segments, is retained exactly. This protects the inherited cancellation-prone
first mean reset `P0 + (P1-P0)`. Every subsequent statistical update preserves
its original segment order. The prototype falls back to the entire original
frequency calculation when its folded result becomes nonfinite or approaches
the existing power/fourth-power exponent boundaries.

Additional production work would have been required even if speed had been
favorable:

- Enforce recognized window identity and genuine symmetry; opaque callbacks
  cannot enter merely because their names match.
- Gate excessive phase, input/coefficient magnitude, and severe cancellation.
  A finite synthetic audit does not prove arbitrary-signal accuracy.
- Include the two half-length FP64 coefficient arrays in the concurrency
  memory budget, approximately eight additional bytes per segment sample.
- Preserve the residual order-one detrending path, scalar arithmetic,
  full/selected output relationship, and actual float32/complex64 output
  quantization.
- Validate native vector widths and ABI precision on each supported platform.

The prototype deliberately implements only order-zero real auto spectra.
Its low-level argument validation is incomplete and its timers on unsupported
fallback cases are not a supported interface. It must not be promoted to the
public API without that work.

## Reproduce

Use an isolated checkout of the recorded baseline commit
`9f3071ca2b7134a585f1727506a4e1472a7da299`. The baseline `fast_dft.c` SHA-256 is
`c53af885aed60e812416693518e87b7cbdd975d314d67b5d69f9b20572ad27f9`.
The experiment calls internal helpers from that baseline and intentionally
does not compile against later modified internal signatures.

```sh
python build_probe.py --source-root /path/to/pinned/lpsd
python check_probe.py
python check_probe.py --coefficient-reuse
python bench_probe.py --baseline-library /path/to/baseline/liblpsd_fast.so \
  --fma-library /path/to/bounded-fma/liblpsd_fast.so
python bench_probe.py --coefficient-reuse \
  --baseline-library /path/to/baseline/liblpsd_fast.so \
  --fma-library /path/to/bounded-fma/liblpsd_fast.so
```

Run the benchmark only while no other task is measuring or consuming the
host's CPUs. The build uses the local compiler and `-march=native`; compiler
and host changes can change both timings and rounding. The bounded-FMA
comparison library must provide the additive `fast_dft_selected_bounded`
entry point documented by its snapshot; it is not an FFTW library.

The generated C source uses an ordinary `#include "fast_dft.c"`; the build
supplies the selected baseline directory through `-I`. No machine-specific
include path is required in the source. Historical report hashes correspond
to their recorded experiment stages; `source-manifest.json` explains the
portable archive representation of the initial prototype.
