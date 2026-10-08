# Overlap reuse for high-overlap Boxcar spectra

`kernel="fast"` can reuse the outgoing and incoming portions of overlapping
Boxcar segments. It retains the LPSD frequencies, lengths, segment starts,
order-zero detrending, and upstream mean/deviation recurrence. It works for
auto spectra and all selected outputs, including PSD and NSD.

The automatic selection is deliberately narrow:

- The window is the recognized `"boxcar"` name or `numpy.ones` callable.
- `kernel="fast"`, `detrending_order=0`, and `csd=False`.
- Requested overlap is at least 0.8, segment length is at least 256, and
  the frequency has at least 32 segments.
- The native check also requires the largest integer shift to be at most
  one quarter of the segment length.

Other calls keep their direct segment projections. An opaque callback that
happens to return ones is not silently treated as a known Boxcar window.
The `rolling_boxcar_frequencies` metadata reports how many frequencies used
overlap reuse; per-frequency profiles identify the segment method and the
number of direct rebuilds.

`sample_iterations` in the profile remains the logical `L*K` segment coverage:
it describes the direct reference work, not the number of dot-product sample
visits executed by overlap reuse. It must not be used to infer measured FLOPS
or memory traffic for the rolling path. A fallback's measured segment time
includes both the discarded rolling attempt and the direct recomputation.

## Algorithm and numerical safeguards

For the supplied angular frequency `w`, segment length `L`, and a fixed local
anchor `a`, define the complex Fourier state

```text
F(s) = sum((x[s+j] - a) * exp(i*w*j), j=0,...,L-1)
```

Advancing the segment by `h` samples gives the algebraic identity

```text
F(s+h) = exp(-i*w*h) * (F(s) - outgoing + exp(i*w*L)*incoming)
```

`outgoing` and `incoming` are length-`h` projections using the first `h`
coefficients. An unweighted sum of the same anchored samples is updated from
the same boundaries. Subtracting its product with the coefficient mean
applies order-zero mean removal. Compensated FP64 coefficient means avoid
software binary128 preparation on platforms with that `long double` format.

All states and the local anchor are rebuilt directly every 32 segments.
This bounds the duration of each recurrence; it does not round frequencies
to FFT bins or discard input samples. The segment starts use the inherited
repeated floating-point `start += shift`, followed by the same rounding as
the direct kernel. Replacing this with `segment_number * shift` would not
preserve all original starts.

Anchoring before summation retains small fluctuations around a large DC
offset. If intermediate unprojected states become nonfinite, or the projection
is small compared with the anchored absolute data mass used since the last
rebuild, the entire frequency is recomputed by the ordinary projected kernel.
The cancellation check includes samples that have already left the segment,
so a departed large transient cannot leave a spurious numerical noise floor.
It is a conservative trigger, not a universal rounding-error proof. That fallback
retains its finite/nonfinite behavior and is included in profiling times.
Projections approaching the exponent limits of squared PSDs or the inherited
fourth-power variance intermediates also use the direct fallback.
The first legacy mean update is uniquely sensitive: it evaluates
`P0 + (P1-P0)`, rather than a stable assignment of `P1`. If the second
projection's absolute real-plus-imaginary amplitude is at most 0.001 times
the first, the frequency also falls back. This conservative selection keeps
that inherited arithmetic when a large initial transient precedes quiet data.
Input samples are never modified.

The real and imaginary segment projections feed the unchanged legacy
statistics function. Global FMA contraction remains disabled; the final
auto-spectrum's imaginary component therefore keeps its original exact
cancellation and dtype convention for finite projections.

These are algebraically equivalent projections, not bitwise identical
floating-point evaluations. As with the existing fast projected kernel,
there is no universal relative-error bound at spectral nulls. Validation
must include both relative and application-specific absolute differences.

## Why the other windows do not automatically use this method

A Boxcar needs one complex Fourier state. A Hann or Hamming window written
as a finite cosine sum needs three such states; a Blackman needs five. A
window with `p` nonconstant cosine harmonics generally needs `2*p+1` states.
Each state updates two boundaries of approximately `(1-overlap)*L` samples.

Consequently, the basic boundary work exceeds a direct length-`L` projection
for Hann/Hamming at overlap below about 5/6, and for Blackman below about
0.9, before recurrence bookkeeping and rebuilds are included. The commonly
used overlap settings for these windows do not imply a gain. High-order
flat-top windows need still more states. A finite small cosine expansion
also does not reproduce the existing Kaiser window exactly.

This implementation therefore uses measured eligibility for the narrow
Boxcar case. It does not substitute a different window or spectral estimator
to force a speedup.

## Initial comparison against the first eight-segment kernel

This series used the first eight-segment AVX-512 kernel, whose gate included
all lengths of at least 128. A subsequent CPU optimization keeps direct
projections for lengths 256–1023; that adaptive gate is not the baseline in
this archived series. These ratios measure the additional benefit of Boxcar
reuse relative to the specifically identified `de9e70a5…` binary. Final
incremental claims require the separate comparison against the adaptive gate.
All cases use a resident
float64 standard-normal record (seed 20261008), sample rate 1, 1,000 target
frequencies, 100 target averages, order-zero detrending, PSD-only output, and
a 4,096 MiB working-concurrency budget. Exact output counts depend on overlap
and record length and are included in the saved data.

| Samples | Workers | Overlap | Eight-segment baseline | With Boxcar reuse | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 1 | 0.8 | 0.549 s | 0.495 s | 1.11× |
| 1,000,000 | 8 | 0.8 | 0.150 s | 0.140 s | 1.07× |
| 1,000,000 | 1 | 0.9 | 1.080 s | 0.697 s | 1.55× |
| 1,000,000 | 8 | 0.9 | 0.250 s | 0.197 s | 1.27× |
| 1,000,000 | 1 | 0.95 | 1.964 s | 0.996 s | 1.97× |
| 1,000,000 | 8 | 0.95 | 0.473 s | 0.291 s | 1.63× |
| 10,000,000 | 1 | 0.9 | 12.795 s | 8.890 s | 1.44× |
| 10,000,000 | 8 | 0.8 | 1.508 s | 1.267 s | 1.19× |
| 10,000,000 | 8 | 0.9 | 2.647 s | 2.312 s | 1.14× |
| 10,000,000 | 8 | 0.95 | 5.533 s | 2.723 s | 2.03× |

These are medians of four complete API calls per implementation, with one
full-problem warm-up first. Timing order alternates baseline/candidate and
candidate/baseline. There is shared-host noise: the candidate was faster in
37 of the 40 individual pairs, not every pair. The smallest case improvement
is modest and should not be interpreted as an architecture-independent
guarantee. These are x86-64 native builds measured on the same host as the
other fast.3 CPU experiments; architecture tests establish correctness and
build support separately from performance claims.

For the separate instrumented **10-million-sample, one-worker, 0.9-overlap**
calls, total API time was 12.842 s for the baseline and 8.314 s for overlap
reuse. The segment phase fell from 10.435 s to 6.274 s. Coefficient projection
preparation fell from 1.201 s to 0.882 s; window generation, window sums, and
coefficient generation together were 1.111 s and 1.071 s. The principal gain
therefore comes from the segment computation. These instrumented calls are
separate from the ordinary-call medians above.

Both implementations retained the same logical 66,299,007,864 `L*K` sample
coverage in that case. Overlap reuse does not execute that many full-length
sample visits, so dividing this count by its segment time would overstate
executed throughput.

All ten timing cases produced **identical rounded float32 PSD values**
and matching frequency indices/dtypes. This observation does not imply
bitwise equality for arbitrary signals. The 46 focused regressions cover
eleven signal families, odd/even lengths and nonintegral shifts, selected
outputs and workers, exclusions, large departed transients, the first legacy
mean reset, power/variance exponent limits, and six longer 131,073-sample
scalar-reference PSD/NSD cases. The latter use a 1% relative limit or explicit
synthetic-case floors of PSD 1e-24 / NSD 1e-12; those floors do not define a
physical tolerance for other inputs. The inherited deviation recurrence is
retained; minute absolute differences in its output can remain under a
different order of segment projection.

The readable [result summary](../benchmarks/results_rolling_boxcar.json)
contains all timing repetitions, ratios, comparisons, phase summaries, and
provenance. The [compressed full data](../benchmarks/results_rolling_boxcar_raw.json.gz)
retain every per-frequency profile row and observed source/build identity.
Gzip compression is lossless and deterministic, and both compressed and
uncompressed SHA-256 hashes are included in the summary.

The measured candidate package source is local commit `881b308`; its native
library SHA-256 is
`f2cb01984194d2bb6acf3d44f0a873d7daaf2ec17a775c601088eb586a45602d`.
The baseline library is
`de9e70a504dbe82131fcca52b483c2fb2e23d17f765ca3c99e5dbb7ed886a773`.
The data preserve a provenance distinction: a later 64-bit-only dispatch
guard changed the observed baseline C file, while the loaded baseline binary
stayed unchanged. An independent native rebuild of that guard-only source
produced the same baseline binary. The subsequently added Python profile
note labels logical sample counts and does not change the measured native
kernel or ordinary-call calculation.

## Reproduce the comparison

Use separate checkouts and native builds of the baseline and candidate.
Do not run another CPU benchmark or compiler workload concurrently.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_rolling_boxcar.py \
  --baseline-root /absolute/path/to/baseline \
  --n 1000000 10000000 --workers 1 8 --overlap .8 .9 .95 \
  --repeat 4 --output benchmark-results/rolling-boxcar.json
```

The harness loads both separately built APIs, gives them the same resident
float64 input and parameters, warms each on the complete problem, and then
alternates baseline/candidate and candidate/baseline ordering. It saves all
wall-time repetitions, separate instrumented profiles, numerical comparisons,
source hashes, library hashes, and compiler build reports. Imports, input
creation, provenance collection, and separate profiles are excluded from
ordinary API timings.
