# Segment calculations: fast.4

Implementation and measurements: 2026-10-08. This report compares
`1.0.6+fast.4` with fast.3 commit
`9f3071ca2b7134a585f1727506a4e1472a7da299`.

This revision changes the segment projections themselves. Projected CSD
shares coefficient loads between multiple segments. The explicit fast PSD
path uses bounded FP64 fused multiply-add operations, with vector widths
chosen by segment length. Neither change removes frequencies, segments,
window coefficients or input samples. The inherited mean and deviation
recurrences remain intact. No GPU speedup has been measured.

## Implemented segment work

### Cross spectra

A cross-spectrum segment reads two input channels. The previous projected
CSD loop reloaded its two coefficient vectors for every segment. The new
path uses the existing four-stream helper for two CSD segments, and the
eight-stream helper for four CSD segments on supported AVX-512 targets.
Each channel in each segment retains its own local anchor. The first two
CSD projections and every subsequent statistical update keep their original
order. The [CSD implementation report](csd-segments.md) describes dispatch,
phase preservation, exceptional inputs and the isolated measurement series.

### Explicit fast auto spectra

On a supported AVX-512 build and CPU, single-worker `kernel="fast"` calls
can use explicit FP64
FMA for projected auto spectra. Eight-stream AVX-512 projections share
coefficient loads. Lengths `256 <= L < 1024` instead use two four-stream
256-bit AVX/FMA passes per logical group of eight segments. Experiments
with four streams for long vectors were slower and were rejected.

The input peak is checked once per channel without an input-sized absolute-
value allocation. The native code independently checks the projected
coefficients. Both magnitudes must be at most `2^100`; unsupported or extreme
cases retain the ordinary selected arithmetic. The count check requires
at least ten segments in the middle-length interval and sixteen elsewhere,
so it avoids scanning coefficients when no fused group could execute.
Global compilation still uses `-fno-fast-math -ffp-contract=off`.

The original initial eight projections, or the first two singleton
projections in the middle interval, are preserved. This protects the
cancellation-sensitive inherited update `P0 + (P1-P0)`. The final variance
assignment, normalizations and output quantization also remain unchanged.
See [fused segments](fused-segments.md) for the numerical bounds and native
interface. `kernel="auto"`, scalar PSD, parallel PSD and CSD do not select this FMA path.
The one-worker restriction follows complete-call measurements; an FMA speedup
with parallel frequency workers was not repeatable.

## Complete public API measurements

The main evidence is complete wall time, including planning, input-bound
checking, window and coefficient preparation, native segment calculation,
normalization and output assembly. Each final scenario uses six fixed
balanced pairs after one complete warm-up per implementation. Every
observation is retained; none is removed as an outlier.

The factor is always baseline time divided by candidate time. A factor of
1.10 means 10% higher throughput, but approximately 9.09% less wall time.
The geometric mean of paired factors, their observed range and the number
of faster pairs accompany the ratio of medians. These are descriptive
small-sample results on a shared host, not confidence intervals or hardware
guarantees.

### Final PSD build against fast.3

| Samples | Workers | FMA selected | fast.3 median | fast.4 median | Ratio of medians | Less median wall time | Paired geometric factor | Paired range | Faster pairs |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 1 | Eligible batches | 0.588074 s | 0.507496 s | 1.1588x | 13.70% | 1.1538x | 1.0274–1.2586x | 6/6 |
| 10,000,000 | 1 | Eligible batches | 8.189316 s | 7.888222 s | 1.0382x | 3.68% | 1.0531x | 1.0073–1.1323x | 6/6 |
| 10,000,000 | 8 | No | 1.532102 s | 1.650249 s | 0.9284x | -7.71% | 1.0058x | 0.7957–1.3873x | 3/6 |

Both one-worker cases improve in every pair. The preceding unrestricted
four/eight-stream build, which used the same native library and the same
one-worker arithmetic, showed 8.91% and 5.18% less median time in its own
separate six-pair series. Those observations are retained in the experiment
archive rather than pooled into the final wrapper's repetitions. The final
single-worker result supports a useful but workload-dependent gain.

The parallel row deliberately uses the ordinary selected path. Its median
is 7.71% slower in this session, while its paired geometric summary is close
to unity and only three of six pairs improve. Both slower and faster
observations remain visible. It establishes no parallel PSD speedup and
no precise general regression factor. Disabling unproven parallel FMA does
not make shared-host wall times identical.

Every timed and profiled final PSD output is bit-identical float32 to fast.3
for these seeded inputs, with exact frequency grids and matching dtypes.


The complete final timings and numerical summaries are retained in
[results_fast4.json](../benchmarks/results_fast4.json). Its
[lossless raw report](../benchmarks/results_fast4_raw.json.gz) also retains
every per-frequency profile and both full accuracy audits.

### CSD evidence, kept by measured build

The first two rows isolate the published CSD change. The next two rows use
the subsequent integrated eight-stream PSD candidate, before the final
four/eight-stream PSD dispatch. The CSD algorithm is the same, but the
candidate binaries and measurement sessions differ. They are therefore
shown separately and are not pooled into twelve repetitions of one build.

| Measured build | Samples | Workers | fast.3 median | Candidate median | Ratio of medians | Less median wall time | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolated CSD | 1,000,000 | 1 | 1.254964 s | 0.954886 s | 1.3143x | 23.91% | 6/6 |
| Isolated CSD | 10,000,000 | 8 | 3.687593 s | 3.604374 s | 1.0231x | 2.26% | 5/6 |
| Integrated eight-stream candidate | 1,000,000 | 1 | 1.181048 s | 0.946322 s | 1.2480x | 19.87% | 6/6 |
| Integrated eight-stream candidate | 10,000,000 | 8 | 4.569309 s | 3.075242 s | 1.4858x | 32.70% | 5/6 |

The single-worker result is consistent across both sessions: every pair
improved, with roughly 20–24% less median API time. In the integrated
session the paired geometric factor is 1.2574x, with all six factors between
1.1897x and 1.3759x. The eight-worker estimates vary substantially between
sessions. They support no precise, generally applicable multithreaded gain.
The large integrated-session factor must not replace the less favorable
earlier result. Every timed and profiled CSD output in these series is
bit-identical complex64 for the recorded seeded input.

The isolated data are in [results_csd_batches.json](../benchmarks/results_csd_batches.json).
The complete subsequent session is retained in the
[FMA dispatch experiment](../benchmarks/experiments/fma_dispatch/README.md).

## Profiles and selection experiments

Instrumented profiles are separate diagnostic calls. The historical candidate profiles use one
worker, including profiles attached to eight-worker timing scenarios. The
final parallel-PSD comparison omits a profile, because a one-worker profile
would activate a different FMA dispatch.
They cannot explain a particular eight-worker scheduling or scaling result,
and their times are not extra benchmark repetitions.

| Samples, one worker | fast.3 segment time | fast.4 segment time | fast.3 preparation | fast.4 preparation | fast.3 full profiled API | fast.4 full profiled API |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 0.327123 s | 0.263580 s | 0.070081 s | 0.073365 s | 0.553502 s | 0.496155 s |
| 10,000,000 | 6.620155 s | 5.720312 s | 0.653538 s | 0.720709 s | 8.695631 s | 7.854629 s |

The segment phase falls by about 19.4% and 13.6% in these separate profiles.
The input-bound scans cost about 0.00027 s and 0.00622 s. Preparation is
slightly more expensive, and ordinary wall-time medians remain the main
complete-call evidence; these individual profiles are not added to them.

The initial integrated eight-stream PSD candidate was not accepted as a
finished optimization. Its final six-pair series had ratio-of-median
factors of 1.0471x for one million samples and one worker, 1.0149x for ten
million and one worker, and 0.7597x for ten million and eight workers.
The corresponding paired geometric factors were 1.0172x, 1.0167x and
0.8900x. This combination of small single-worker gains and a noisy
multithreaded regression risk motivated the width comparison.

The following fixed native screen compared ordinary selected arithmetic,
the eight-stream FMA candidate, four-stream 256-bit and 512-bit variants,
and selective dispatch. Four-stream long-vector variants were slower.
The 256-bit four-stream kernel improved the previously problematic
256–1023 interval. A fixed three-way complete-API screen then compared
strict arithmetic, eight streams and the mixed four/eight candidate;
all four rounds and all three implementations are retained. This screen
selected the mixed candidate for the final comparison above; its medians
must not be added to, multiplied with or pooled into that final series.

Individual profiles can also expose changes in host load. For example,
the integrated CSD ten-million-sample profile reduced even the unchanged
coefficient-generation phase from about 0.832 to 0.380 seconds. Its entire
API ratio cannot be attributed to a code change in that unchanged phase.

### Rejected folded DFT

A separate experiment paired symmetric window samples and rotated the
complex coefficients to reduce projection multiplications. It tested both
new trigonometric coefficient preparation and reuse of existing coefficients,
with ordinary FP64 and FMA and several segment widths. Each variant passed
688 numerical/exceptional comparisons under the recorded synthetic criteria.

It did not establish a consistent speed advantage. Even retrospectively
selecting the fastest folded width for each of sixteen configurations gave
geometric speed ratios of 0.8880 for periodic Kaiser and 0.9518 for symmetric
Hann against the bounded FMA candidate. Ratios below one mean slower.
The sources, every repetition, numerical caveats and rejection decision are
retained in the [folded experiment](../benchmarks/experiments/folded_segments/README.md).
It is isolated from the public API.

## Numerical and platform validation

The final portable GCC build passed **1,032 tests**, with the same **two
strict expected failures** already documenting the upstream standard-error
defect and the absence of a uniform relative guarantee at vanishing tone
sidelobes. Neither whole signal families nor new failures are hidden behind
expected-failure marks. The new suites contain 229 fused-projection cases
and 176 CSD batching cases, in addition to the previous regressions.

Two independent scalar-reference audits each covered 19 signal/window
configurations: 32,769 samples with one worker and 131,073 with eight
workers. They include white and pink noise, mixed and pure tones, DC plus
nanovolt noise, a large ramp with tiny noise, Kaiser PSLL 200/80, Hann,
Blackman, Boxcar and HFT248D. All 38 frequency grids match exactly.
The separate 257-sample scalar anchor retains all seven original output
columns and frequencies bit for bit.

The criterion was fixed before running: relative error strictly below 1%,
**or** absolute error at most `1e-24` for PSD and `1e-12` for NSD, in these
synthetic signal units. All 5,802 PSD/NSD values pass. Seven PSD values and
six NSD values pass only through the absolute allowance. These allowances
are not a unit-independent physical noise floor.

The raw relative errors remain visible. The worst relative PSD difference
is approximately 99.83% at a deep HFT248D null: the reference is about
`6.12e-30` and the absolute difference about `6.11e-30`. The largest absolute
PSD difference across both audits is `5.29e-23`, at a different point that
passes the relative criterion. The largest absolute NSD difference is
`1.38e-14`. Thus the result must not be described as a uniform sub-1%
relative guarantee for every spectral null.

The CSD exceptional-input test now evaluates finite real/imaginary pairs
with a scaled complex norm, keeping the original `2e-12` relative limit.
Power and variance are compared separately. NaN and signed-infinity masks
remain exact; a lone finite component next to a nonfinite one retains its
strict component-wise check. This resolved a Clang-only test failure caused
by treating a nearly cancelled imaginary component as an independent scale.

The [native CI workflow](../.github/workflows/tests.yml) exercises Linux x64
GCC with Python 3.10/3.12, Linux x64 Clang, Linux ARM64 GCC and macOS ARM64
Clang, plus installed-wheel checks on Linux x64/ARM64 and macOS ARM64.
[Workflow runs](https://github.com/MichaDit/lpsd/actions/workflows/tests.yml)
retain the per-commit results. Unsupported FMA targets exercise the portable
fallback; compilation capability alone is not a performance claim.

## GPU and other accelerators

The execution environment exposes no usable CUDA, ROCm or GPU device.
A direct call to the installed OpenCL loader returns `-1001` and zero
platforms, so an installed loader is not an available accelerator. No GPU
kernel, FP64 device result, transfer measurement or complete GPU LPSD timing
could be obtained. A production GPU backend is therefore not enabled.

The [accelerator assessment](accelerators.md) records the device/runtime
checks, CPU SIMD capabilities, BLAS distinction, exact proposed offload
payloads, and the implementation and accuracy work an actual GPU comparison
would require. The implemented and measured additional acceleration here
is CPU segment reuse and bounded FMA. Unsupported CPUs keep the ordinary
portable paths; this is not a measured claim that all CPU architectures
receive the same gain.

## Reproduction and provenance

The local host reports AMD EPYC 9V74, nine visible CPUs, an eight-CPU cgroup
quota, and Python 3.12 with GCC 13.3. Portable native flags are
`-O3 -std=c11 -fno-fast-math -ffp-contract=off -fPIC -fopenmp-simd`, with
`-DLPSD_HAVE_TARGET_CLONES=1` for runtime dispatch. Both sides are built
with the same flags. This series uses portable builds, unlike older
explicitly host-native fast.3 screening; compare matched binaries within
each series.

All complete-call scenarios use resident float64 pandas input with PCG64
seed 20261008, sample rate 1, Kaiser PSLL 200, 1,000 requested frequencies,
100 requested averages, order-zero detrending, `kernel="fast"`, PSD-only
output, a 4,096 MiB concurrency budget and a 128 MiB window cache. The actual
plans have 577 frequencies for one million samples and 651 for ten million.
Each call independently prepares its plan and windows. Imports, input
creation, warm-up, comparisons, provenance collection and file writes are
outside the API timer. Intentional benchmarks and builds run serially;
shared-host load and CPU clocks are not controlled.

Build sidecars now bind each binary to the SHA-256 of its project source
and recursively quoted includes. They are verified before and after
measurement along with the binary hash, Python source hashes and actually
imported shared reference modules. A source edited after compilation can
no longer silently masquerade as the source of a benchmarked binary.
Every final raw report retains these manifests and the harness hashes.

Build the current checkout and a separate fast.3 checkout before comparing:

```sh
make compile
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -o addopts='' -q test_fast test
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.bench_segment_core \
  --baseline-root /path/to/fast3 --candidate-root /path/to/fast4 \
  --kind psd --kernel fast --n 1000000 --workers 1 --repeats 6 \
  --output benchmark-results/psd-1000000-w1.json
```

Repeat with `--n 10000000 --workers 1` for the large single-worker case.
Use `--n 10000000 --workers 8 --profile-workers 0` for the parallel PSD
scenario; this preserves its recorded omission of a one-worker profile
that would activate a different FMA dispatch.
Use `--kind csd` for a matched cross-spectrum comparison.
Do not edit or rebuild either package during a measurement.

Reproduce either accuracy audit with its stated size and worker count:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.check_accuracy \
  --n 32769 --kernel fast --workers 1 --relative-limit .01 \
  --psd-absolute-limit 1e-24 --nsd-absolute-limit 1e-12 --fail-on-limit \
  --output benchmark-results/accuracy-fast4-32769.json
```
