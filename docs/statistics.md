# Segment statistics: corrected default and explicit legacy mode

`lpsd_fast.lpsd`, `lcsd`, `lnsd` and `PreparedLPSDSubset` now default to
`statistics='corrected'`. This mode includes every actual segment in the
periodogram mean and accumulates all second-moment contributions when a
deviation column is requested. `statistics='legacy'` reproduces the original
LPSD 1.0.6 recurrence for historical comparisons. The original `lpsd` package
and existing native entry points retain their historical behavior.

This is a correction to the estimator, not an approximate speed optimization.
Corrected and legacy PSD values can differ by more than a numerical tolerance,
especially with only a few segments or a transient in the first segment.
Earlier fast.1–fast.9 performance and numerical evidence describes its actual
legacy estimator; those measurements are not relabelled as corrected results.

```python
from lpsd_fast import lpsd

current = lpsd(samples, sample_rate=50, statistics='corrected')
historical = lpsd(samples, sample_rate=50, statistics='legacy')
```

The statistics selection is independent of `kernel='scalar'`, `'auto'` or
`'fast'`. Use `kernel='scalar', statistics='legacy'` when reproducing the
original C arithmetic. A scalar corrected call is the arithmetic reference
for a faster corrected call. Keep this selection identical within a numerical
comparison or a benchmark pair.

## What is averaged

For one output frequency, let `K` be the number of actual valid segments,
`X_k` their windowed Fourier projections and `Z_k = |X_k|²` their raw
periodogram values. Corrected mode computes

\[
\bar Z = \frac{1}{K}\sum_{k=1}^{K} Z_k.
\]

For CSD the corresponding value is `Z_k = X_k conjugate(Y_k)`, preserving
the existing channel ordering and Fourier-sign convention. Its mean is
generally complex. The segment starts, detrending, window, frequency grid and
one-sided normalization are unchanged. `K` comes from the actual segments;
the requested `n_averages` is a planning target, not the denominator.

With window sums `S1 = sum(window)` and `S2 = sum(window²)`, the API retains
the existing normalizations:

\[
\mathrm{PS}=\frac{2\bar Z}{S_1^2},\qquad
\mathrm{PSD}=\frac{2\bar Z}{f_s S_2}.
\]

NSD/ASD is still the square root of the already rounded PSD. It is not the
mean of individual segment amplitudes.

## Meaning of the deviation columns

For `K > 1`, corrected mode accumulates Welford's second moment and computes
the ordinary sample variance

\[
s_Z^2=\frac{1}{K-1}\sum_{k=1}^{K}|Z_k-\bar Z|^2.
\]

The reported deviation columns retain their original names but now have
an explicit **nominal standard error of the mean (SEM)** interpretation:

\[
\mathrm{ps\_std}=\frac{2}{S_1^2}\sqrt{\frac{s_Z^2}{K}},\qquad
\mathrm{psd\_std}=\frac{2}{f_s S_2}\sqrt{\frac{s_Z^2}{K}}.
\]

These formulas use the **independent-segment assumption**. The implementation
does not adjust for shared samples in overlapping segments, time correlation
in colored noise, nonstationarity, or the resulting effective degrees of
freedom. Consequently these columns describe empirical segment scatter with
the nominal `sqrt(K)` scaling; they are not a validated uncertainty of the
measurement or a confidence interval for an arbitrary input. Even zero
overlap does not establish independence for a correlated input.

For `K=1`, the spectrum is valid but neither sample variance nor its SEM can
be estimated from scatter. `ps_std` and `psd_std` are therefore `NaN`.
An exactly constant set of `K > 1` periodograms has zero empirical SEM.
PSD-only and NSD-only calls remain finite when their spectra are finite;
they do not compute these undefined single-segment deviation values.

For complex CSD, the absolute square above is the sum of the real and
imaginary squared deviations. The resulting SEM is a **real RMS complex
error**, not a complex square root of the pseudovariance
`mean((Z-mean(Z))²)`. It does not separately report the real/imaginary
covariance, phase uncertainty or a confidence region. Swapping the two
channels conjugates the mean and leaves this RMS SEM unchanged.

Corrected `ps_std` and `psd_std` are always **float32**, including complex
CSD. Complex CSD spectrum columns retain their complex64 representation.
Legacy mode retains the old data-dependent real/complex deviation dtypes.
The PSD/NSD rounding point and other output types are unchanged.

## Small exact examples

The following values precede the common spectral normalization:

| Segment values `Z_k` | Corrected mean | Corrected nominal SEM | Legacy behavior |
|---|---:|---:|---|
| `1, 4` | `2.5` | `1.5` | Mean `4`, SEM `0` |
| `1, 4, 9` | `14/3` | `7/3` | First contribution is discarded and variance is overwritten |
| `i, -i` | `0` | `1` | Complex pseudovariance does not represent RMS complex scatter |
| One value `z` | `z` | `NaN` | Original single-segment substitute remains available in legacy mode |

An impulse-window fixture with `fs=1` and raw powers `1,4` has the API's
normalization factor of two. It now returns **PSD 5 and `psd_std` 3**;
explicit legacy mode retains **PSD 8 and `psd_std` 0**.

## Output selection and reproducibility

Omitting both `ps_std` and `psd_std` skips all native second-moment updates.
The corrected mean still includes every segment. Prepared and streaming
PSD-only paths use that same mean without performing second-moment updates.
Rolling Boxcar and fused/batched direct paths use
the selected recurrence, including after a whole-frequency fallback.

The result's `attrs['lpsd_fast']` records `statistics`, `estimator`,
`legacy_statistics`, `variance_computed`, `sem_kind` and
`sem_correlation_adjusted`. With `profile=True`, each frequency also reports
its actual `K`. Prepared-plan metadata records the statistics mode too.

The corrected native functions are additive:
`fast_dft_selected_corrected`, `fast_dft_boxcar_corrected`,
`fast_dft_prepared_order0_corrected` and
`fast_dft_kaiser_streaming_corrected`. Every pre-existing native ABI remains
legacy-compatible; in particular the original prepared ABI version remains
one. Frozen reference source hashes are not updated to hide the correction.

The mandatory [statistics tests](../test_fast/test_statistics.py) construct
individual segment periodograms independently and use two-pass squared
distances as the variance reference. They cover exact real/complex examples,
window and detrending choices, selected/full output equality, parallel
determinism, single-segment behavior, prepared/streaming/rolling routes,
complex conjugation and scaling, and finite-mean exponent limits. This
finite validation set does not make overlapping samples independent or
remove the retained floating-point output limits.
