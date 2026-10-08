# Segment optimization: fast.3

Measurements and implementation work: 2026-10-08. This report covers
`1.0.6+fast.3`; the [fast.2 report](performance.md) and
[FFTW comparison](fftw-comparison.md) retain their original measurement series.

The remaining segment work had useful optimization headroom. Two changes
are implemented: adaptive eight-segment coefficient reuse on supported
64-bit AVX-512 CPUs, and rolling Fourier states for sufficiently overlapping
Boxcar auto spectra. Neither removes requested frequencies, segments or
input samples. No GPU performance result has been measured in this container.

## Complete-call results

### Final Kaiser API against fast.2

The integrated release, including its Python dispatch and output assembly,
was compared directly with the prior fast.2 package. All calls use the
same 10-million-sample Kaiser problem and eight workers. The first four
pairs showed substantial timing variation, so one fixed additional series
of eight pairs was run with the same inputs, libraries and harness. Both
series are retained completely in [results_fast3.json](../benchmarks/results_fast3.json).

| Series | Pairs | fast.2 median, s | Integrated fast.3 median, s | Ratio of medians | Less median wall time | Candidate faster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial series | 4 | 1.659236 | 1.608060 | 1.0318× | 3.08% | 3/4 |
| Fixed confirmation | 8 | 1.684089 | 1.322029 | 1.2739× | 21.50% | 8/8 |
| **All retained calls** | **12** | **1.684089** | **1.457337** | **1.1556×** | **13.46%** | **11/12** |

The pooled row is a summary of those two series, not a third independent
experiment. Baseline times range from 1.430451 to 2.840857 s; candidate times
range from 1.208880 to 1.961849 s. The variation between the two series is
material. The result supports an observed complete-call improvement on this
host, not a precise universal 13.46% speed guarantee. No observations were
removed, and no further series was run to select a preferred result. Every
ordinary and profiled output fingerprint is identical between these versions
for this seeded input and its 651 actual output frequencies.

A separate 30-million-sample experiment isolates the final adaptive segment
kernel using the unchanged fast.2 Python wrapper. Five paired eight-worker
calls reduced the median from **6.280836 to 5.919844 s**, **1.0610×**, or
**5.75% less wall time**, with 678 frequencies. Four of five pairs improved;
the candidate's 8.484444 s observation is retained. This is a native-change
comparison within a complete API timer, not another measurement of the
integrated release's newer Python wrapper. Full data and the private replay
are in the [segment experiment](../benchmarks/experiments/segment_batches/README.md).

The earlier ten-million-sample eight-way prototype and its subsequent
adaptive-threshold comparison are retained in that experiment. Their
speedup factors must not be multiplied across measurement series or used
instead of the direct integrated comparison above.

### Additional gain from Boxcar overlap reuse

Here the baseline already has the final adaptive segment kernel. The only
additional algorithm is guarded Boxcar reuse. Each row uses ten million
samples and four balanced pairs of complete API calls.

| Workers | Overlap | Actual frequencies | Adaptive direct median, s | With overlap reuse, s | Speedup | Less median wall time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.80 | 660 | 1.689370 | 1.527597 | 1.1059× | 9.58% |
| 8 | 0.90 | 699 | 2.877074 | 1.816709 | 1.5837× | 36.86% |
| 8 | 0.95 | 733 | 5.360742 | 2.673038 | 2.0055× | 50.14% |
| 1 | 0.90 | 699 | 13.816762 | 8.852593 | 1.5608× | 35.93% |

Fifteen of sixteen individual pairs improved. The 0.80 case has a smaller,
noisier gain; the larger 0.90 and 0.95 gains occur in every recorded pair.
All four cases have identical rounded float32 PSD values and matching
frequency grids/dtypes. These compare Boxcar with the same Boxcar estimator,
not Boxcar with Kaiser. The [Boxcar report](rolling-boxcar.md) includes every
repetition, exact source snapshots and a separate clearly labeled archive
of the earlier nonadaptive-baseline series.

## Granular diagnosis of the final kernels

These are independent **one-worker instrumented calls**, not the ordinary
eight-worker timings above. Both final Kaiser series retain their own
profile pair, so the second profile has not been substituted for the first.

| Phase, seconds | Kaiser initial: fast.2 | Kaiser initial: fast.3 | Kaiser confirmation: fast.2 | Kaiser confirmation: fast.3 |
| --- | ---: | ---: | ---: | ---: |
| Segment projections and inherited accumulation | 6.064769 | 5.069126 | 5.999242 | 5.170256 |
| Native coefficient projection preparation | 0.635231 | 0.643933 | 0.614997 | 0.620568 |
| Window generation and normalization sums | 0.755709 | 0.744508 | 0.731760 | 0.736998 |
| Fourier coefficient generation | 0.329353 | 0.338827 | 0.328611 | 0.328887 |
| **Full measured API** | **7.901913** | **6.866164** | **7.748580** | **6.930391** |

Almost all of the saving in both profile pairs comes from the segment
phase. The remaining difference between the displayed phases and the full
API includes allocation, call boundaries, scheduling, planning and output
assembly. Parent native/window phase totals are not added to their own
subdivisions. Instrumentation itself can change timings.

The first final Kaiser profile pair can be divided more finely by length:

| Segment length | fast.2 segment seconds | fast.3 segment seconds |
| --- | ---: | ---: |
| 128–255 | 0.46142 | 0.40081 |
| 256–1023 | 0.78455 | 0.81332 |
| 1024–2047 | 0.46166 | 0.34742 |
| 2048–65535 | 1.92577 | 1.37920 |
| 65536–1048575 | 1.84443 | 1.55196 |
| At least 1048576 | 0.58694 | 0.57642 |

The 2,048–65,535 group accounts for the largest observed absolute saving.
The unchanged 256–1023 path still varies between these separate calls;
that variation must not be interpreted as new code slowing those segments.
The complete profile rows and the separate 135-length threshold experiment
are retained, including their less favorable results.

For Boxcar at 0.90 overlap, the separate one-worker profiles reduce segment
time from **11.382179 to 6.301633 s**. Native preparation changes from
1.202777 to 0.846198 s, while window generation, sums and Fourier coefficient
generation together change from 1.153638 to 1.055437 s. Full profiled API
times are 13.844899 and 8.291671 s. Here too, the main gain is actual segment
work. These diagnostic timings are distinct from the ordinary-call medians.

## What changed in the segment calculation

### Adaptive eight-segment projection

The previous projected auto-spectrum kernel shared each Fourier coefficient
pair across four segments for lengths of at least 2,048. The new AVX-512 path
uses sixteen independent real/imaginary accumulators for eight segments.
Loading a coefficient pair once can therefore feed eight anchored input
values. The compiler vectorizes the loop within each segment; frequency
parallelism still uses the existing worker pool.

The larger register file matters: forcing this layout onto a narrower
instruction set can cause register spills and erase the benefit. The new
selection requires 64-bit x86 and AVX-512F, through either explicit native
compilation or the existing runtime dispatch. Other targets retain their
previous four/single-segment route. This is not a measured claim of an ARM
speedup from an eight-way layout.

The dispatch is also adaptive by length. It uses eight segments for
`128 <= L < 256` and `L >= 1024`, with at least eight segments remaining.
The `256 <= L < 1024` interval retains the previous route. This exception
comes from matched measurements across all 135 actual lengths in the
256–2,047 range, not from a single noisy profile. Four-way and single-segment
remainders remain available.

Every inherited mean update still occurs in its original order. The final
overwriting second-moment update is retained for deviation outputs. CSD
and nonprojected kernels do not use this new auto-spectrum batch.

### Reuse of overlapping Boxcar segments

For a Boxcar window, the Fourier projection of the next segment can be
computed from the previous projection plus the outgoing and incoming sample
boundaries. The implementation also updates the anchored unweighted sum
needed for order-zero detrending. All states and their local anchor are
rebuilt directly every 32 segments.

Automatic selection requires `kernel="fast"`, a recognized Boxcar window,
an auto spectrum, order-zero detrending, overlap of at least 0.8, length of
at least 256 and at least 32 segments. A native check further bounds the
integer hop. The frequency, original repeatedly added segment starts,
normalization and output selection are unchanged. PSD, NSD and the other
auto-spectrum outputs are supported.

Severe cancellation, departed large transients, a critically larger first
periodogram, nonfinite intermediate states and exponent limits trigger a
direct recomputation of the whole frequency. The wasted rolling attempt is
included in profiled time. The full algorithm and safeguards are described
in [rolling-boxcar.md](rolling-boxcar.md).

The same method is not automatically profitable for other windows. A cosine
window with `p` nonconstant harmonics needs `2*p+1` Fourier states. Updating
both boundaries for every state can cost more than a direct projection at
ordinary Hann, Hamming, Blackman or flat-top overlaps. A small exact cosine
expansion does not reproduce the current Kaiser window. These windows keep
their direct segment kernel, including the adaptive batching where eligible.

## Timing conditions

All local timings use the same Linux/x86-64 environment reporting an AMD
EPYC 9V74, nine visible CPUs, an eight-CPU cgroup quota and an 8 GiB memory
limit. The runtime is Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and GCC 13.3.0.
Native compilation uses `-O3 -std=c11 -fno-fast-math -ffp-contract=off -fPIC
-march=native -fopenmp-simd`, shared-library linking and `libm`.

The ordinary complete-call timer includes planning, preparation, native
segment work, normalization and output assembly. Imports, input generation,
provenance collection, warm-up, comparisons and JSON output are outside it.
Both candidates receive the same resident seeded float64 input, parameter
values and one full-problem warm-up. The Kaiser experiments use pandas Series;
the Boxcar harness uses NumPy arrays. A warm-up does not install a persistent
LPSD plan: each ordinary API call still prepares its windows and coefficients.
Independent instrumented calls provide profiles without contaminating the
ordinary timing repetitions.

The common large-record settings are `sample_rate=1`, target frequencies
`1000`, desired averages `100`, `detrending_order=0`, `kernel="fast"`,
`outputs="psd"`, a 4,096 MiB concurrency budget and a 128 MiB window cache.
The Kaiser cases retain PSLL 200 and its original overlap. Boxcar cases use
the explicitly stated overlap in both versions of each pair.

Intentional benchmark and compilation workloads ran sequentially. The shared
host, frequencies and other tenants were not controlled. Every repetition
and its order is retained. Medians and observed ranges are descriptive small
samples, not confidence intervals or universal hardware guarantees.

## What the profiling counts mean

For the direct kernel, logical segment coverage is `sum(L_j*K_j)`. Batching
changes coefficient reuse and instruction scheduling while retaining that
coverage. It does not reduce the requested observations.

For rolling Boxcar, the profile's `sample_iterations` field deliberately
remains this logical direct-reference count. Actual boundary work is smaller;
using `L*K` as an executed visit count would overstate it. Neither count is a
measurement of DRAM traffic or hardware FLOPS. Cache reuse, load instructions
and physical transfers are different quantities.

A simple algorithmic visit model for one eligible frequency is
`R*L + 2*h*(K-R)`, where `R=ceil(K/32)` direct rebuilds and `h` is an
approximately constant hop. With many segments and `h≈(1-overlap)*L`, its
ratio to the direct `K*L` coverage approaches
`1/32 + (31/32)*2*(1-overlap)`. This model excludes coefficient preparation,
additional rolling states, cancellation checks and direct fallbacks. It
therefore does not predict a complete-call speedup.

The general direct complexity remains
`Theta(N + sum(L_j) + sum(K_j*L_j))`. Fixed overlap and a fixed number `M`
of output frequencies give approximately `Theta(N*M)` work; fixed `M` is
linear in record length. If the requested logarithmic density grows the
frequency count as `log(N)`, the corresponding direct work can grow as
`N*log(N)`. These are conditions on the selected planner, not a lower bound
for every possible spectral algorithm. Fixed 32-segment rebasing also leaves
the rolling method in the same worst-case asymptotic class, with a smaller
eligible-case constant. Eight-way SIMD does not change Big-O.

## Numerical contract

FP64 evaluation order may change. The regression gates check frequency and
segment planning, PSD dtype and imaginary cancellation, selected outputs,
PSD/NSD errors, DC offsets, tones, transients, overflow behavior and the
cancellation-sensitive inherited first mean reset. The unchanged scalar C
path remains the reference.

The permitted comparison is below 1% relative error or an explicitly small
absolute difference. The synthetic audit uses PSD `1e-24` and NSD `1e-12`
absolute allowances and keeps every relative violation visible. Those
absolute values depend on signal units; they are not a universal physical
tolerance. The fast kernel does not promise an arbitrary-signal 1% bound
at every spectral null. Inherited standard-deviation statistics retain
their documented defects and are not corrected as part of a speed change.

The integrated native build passed **617 tests with two documented expected
failures and no skips**, including the optional FFTW adapter coverage. It
reproduced the Boxcar experiment's final native binary byte for byte:
`50951a1df37a27d1fe5045b93dbc655e62e2ab1336efd2d6d3f574bb8a9453c8`.
The [pre-release native CI](https://github.com/MichaDit/lpsd/actions/runs/37824539159)
also passed all eight jobs for the same integrated kernels, including GCC,
Clang, native Linux ARM64, Apple Silicon and installed wheels. Its package
version string preceded the final fast.3 bump; the final main publication
has its own workflow run. Platform support is distinct from a measured
performance advantage on every architecture.

Two additional scalar-reference audits, at 32,769 and 131,073 samples,
cover **38 signal/window cases and 5,802 PSD/NSD point comparisons**.
Every case retains the exact frequency grid and passes the combined
criterion. Seven PSD values and six NSD values do not meet the strict 1%
relative criterion near very small flat-top tone responses. They pass only
through the explicit absolute allowance: the largest absolute error among
those values is `1.914753e-26` for PSD and `1.142479e-14` for NSD. These
relative violations remain visible in the saved audits. Maximum errors over
all points are recorded separately; the quoted maxima here concern only
points accepted through the absolute allowance.

The 209 segment-batching tests include 36 adversarial first-mean-reset cases.
The 46 Boxcar regressions include long scalar comparisons, departed
transients and exponent boundaries. The Boxcar cases in the two broad audits
use 0.50 overlap and therefore do not select rolling reuse; its numerical
validation comes from the dedicated 46-case suite. At extreme input scales, requesting
legacy deviation outputs can activate a stricter Boxcar fallback than PSD
alone and change the final rounding. Within a call, NSD and ASD still use
the same square root of its same rounded PSD.

## Reproduce the final integration check

Use separate checkouts with the same compiler and Python dependencies. The
canonical fast.2 baseline after the coauthor repair is
`3f2864391db3bfb22942d99106026358fef5fe96`; its files are identical to the
previously published `b8b921b` tree. Build both native backends in each
checkout before running the comparison from the current checkout.

```bash
git worktree add --detach ../lpsd-fast2-compare 3f2864391db3bfb22942d99106026358fef5fe96
(cd ../lpsd-fast2-compare && python build_native.py && python -m lpsd_fast.build --native)
python build_native.py
python -m lpsd_fast.build --native
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.bench_versions \
  --baseline-root ../lpsd-fast2-compare --n 10000000 --workers 8 \
  --repeats 12 --profile-workers 1 --output benchmark-results/versions-10m.json
```

The recorded result consists of a four-pair run and a fixed eight-pair
confirmation. The command above gathers a new twelve-pair series; it does
not reproduce the uncontrollable historical host scheduling. The harness
records the full source/build identities, every ordinary output fingerprint,
all repetitions and complete separate profile rows. Its CPU baseline needs
no FFTW dependency. The [segment replay](../benchmarks/experiments/segment_batches/README.md)
and [Boxcar source archive replay](rolling-boxcar.md#reproduce-the-comparison)
reproduce their specifically isolated comparisons.

To repeat the additional numerical audit:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.check_accuracy \
  --n 131073 --kernel fast --workers 8 --relative-limit .01 \
  --psd-absolute-limit 1e-24 --nsd-absolute-limit 1e-12 --fail-on-limit \
  --output benchmark-results/accuracy-fast3.json
```

## GPU and other accelerators

The [accelerator assessment](accelerators.md) includes a reproducible device
inventory, exact proposed transfer payloads and primary hardware/runtime
sources. No accessible GPU was exposed in the local execution environment.
The native CI jobs additionally record their own device visibility; finding
a device is distinguished from actually running an FP64 computation.

An FP64 GPU segmented-reduction backend remains a plausible target for large
or repeated records. Input residency, coefficient reuse or on-device
preparation, exact start indexing and the inherited reduction all matter.
For the recorded ten-million-sample Kaiser plan, host-prepared coefficients
alone total about 3.58 GB,
so estimating offload cost from the 80 MB input alone is misleading. No
production GPU backend is enabled on the basis of advertised FLOPS or an
untested port. Tensor-core emulation, mixed precision, SYCL, NPUs and FPGAs
are assessed separately in the linked report.

## Remaining headroom

There is still no demonstrated hardware or mathematical limit. Further
portable CPU gains need measured improvement to the segment kernel, reuse
across records/channels, or an eligible algebraic reformulation. Prepared
plans may amortize repeated-record preparation but change the timing scope;
they cannot be credited to a fresh complete API call without measuring it.
A full-record FFT followed by logarithmic averaging remains a different
estimator, as the existing FFTW numerical comparison shows.

