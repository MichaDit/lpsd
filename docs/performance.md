# Performance and reproducible measurements

`lpsd_fast` accelerates the package's direct-DFT LPSD calculation while retaining
its frequency plan, segment lengths, overlap, windows and normalization.
This report records release `1.0.6+fast.2`. The subsequent
[fast.3 segment optimizations and matched measurements](performance-fast3.md)
are a separate series. Use `from lpsd_fast import lpsd` to select the optimized
implementation; the unchanged original `lpsd` API remains available.

For a later measurement series against actual FFTW 3.3.11, including complete
pipeline costs, prime lengths and remaining optimization headroom, see
[the FFTW comparison](fftw-comparison.md). The version-to-version measurements
below retain their original conditions and results.

## Recorded result: fast.2 versus fast.1

The final native build reduced the measured median at **ten million samples
from 3.976 to 1.888 seconds**, about **2.11 times faster** than the previously
published fast.1 version. At thirty million samples the median fell from
14.449 to 7.913 seconds, about 1.83 times faster. These are fresh incremental
comparisons, not comparisons with the much slower upstream implementation.

All rows below use eight workers, the same seeded float64 Series and Kaiser
parameters, and an explicit 4096 MiB concurrency budget. Fast.1 uses `auto`
with all seven outputs; final fast.2 uses `fast` with PSD selected.
This compares practical whole-version configurations. Output selection by
itself did not demonstrate a reproducible wall-time benefit.

| Samples | Actual frequencies | fast.1 median [range], s | Final fast.2 median [range], s | Ratio of medians | Calls, old/new |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 577 | 0.487 [0.352–0.504] | 0.315 [0.260–0.504] | 1.54× | 3 / 3 |
| 10,000,000 | 651 | 3.976 [3.763–5.125] | 1.888 [1.760–2.033] | 2.11× | 5 / 3 |
| 30,000,000 | 678 | 14.449 [13.685–16.278] | 7.913 [7.665–8.209] | 1.83× | 3 / 3 |

The three final 10M observations came from three separate processes in the
paired Kaiser experiment below. The 1M and 30M rows use three repetitions
within one process. At 1M the ranges overlap substantially. These small
samples describe observed wall times, not confidence intervals or universal
speedup guarantees. At 30M the final run reached 4.593 GiB process RSS versus
4.690 GiB for fast.1; the earlier intermediate build's lower RSS is not a
measurement of the final code.

### Evidence and build identity

Measurements were made on 2026-10-08. Two datasets distinguish successive
stages of the work:

- [First fast.2 integration](../benchmarks/results_fast2.json): native library
  SHA256 `c41db40cae0514ebfe066d82d7d246060b90153d8a4451e2d186026c87beb905`,
  source commit `3908eade83e1ae43183acbdde765ca54da957087`. This contains the
  window comparison, early phase profiles and fast.1 baselines.
- [Final fast.2 evidence](../benchmarks/results_fast2_final.json): native library
  SHA256 `b0ab2e54a6231bd54b8d01c9c364efadb924c8e41dcc0398ef637f29d3e9e623`,
  final code published at `0d80d10872fd0a6d632a272b31d356b28e15feb3`.
  This adds the optional SIMD Kaiser series, final timings, profiles and audits.

Both datasets retain individual wall/CPU timings, output fingerprints,
conditions, memory high-water marks, accuracy diagnostics and raw-file hashes.
The fast.1 baseline is commit `83f8f3ba30f78ef24d9a33d2b61d42cb3e5a018d`,
with native library SHA256
`0edc61184a698f6107f9fd63a5ece20919310ef90490fd04af55fabd3d02140d`.
It had no build sidecar during measurement; its hash was recorded afterward,
and its compiler/flags were reconstructed from the pinned builder and ELF
compiler identity. The later builds have binary-verified build sidecars.

### Common conditions and limits

The local machine reports an AMD EPYC 9V74, Linux x86-64, nine CPUs in the
process affinity mask, an eight-CPU cgroup quota and an 8 GiB memory limit.
The environment used Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and GCC 13.3.0.
Final native compilation used `-O3 -std=c11 -fno-fast-math -ffp-contract=off
-fPIC -march=native -fopenmp-simd`, with shared-library linking and `libm`.
These local measurements do not establish portable-wheel or ARM performance;
actual native ARM and Apple CI evidence is in [platforms.md](platforms.md).

The input is standard-normal noise, seed `20261008`, passed as a pandas Series
in both versions. Parameters are `sample_rate=1`, `n_frequencies=1000`,
`n_averages=100`, `detrending_order=0` and default Kaiser/overlap with `psll=200`.
The requested frequency count is a planning target, not the actual count.
Each benchmark process performs a small warm-up on 4096 samples and a target
of 100 frequencies. Imports, input generation, warm-up, fingerprints and file
output are outside the API timer. Planning, conversion, window generation,
DFT work and output construction are inside it. Profiling, when requested,
uses an additional call separate from the ordinary timing repetitions.

Runs were sequential with no other local benchmark competing for CPU. CPU
clocks and the underlying shared host were not controlled. Cache state,
allocator behavior and memory pressure can change wall time. Process CPU time
sums work across threads; RSS is a process-lifetime high-water mark, not an
isolated allocation of one call. The working budget limits admission of
concurrent tasks, not total RSS.

## Where the remaining time goes

The final **one-worker**, 10M profile took 9.474 seconds. Its uninstrumented
median was 8.232 seconds, so use the profile to locate work and the independent
uninstrumented measurements to assess API speed. There were still
**27,072,536,884 visits to segment samples**: no frequency, segment or input
sample was removed to obtain the speedup.

| Final one-worker phase | Seconds | Share of profiled wall time |
| --- | ---: | ---: |
| Segment projections and inherited accumulation | 7.2241 | 76.3% |
| Window generation and normalization | 0.9748 | 10.3% |
| Native coefficient preparation | 0.7492 | 7.9% |
| Fourier coefficient generation | 0.4321 | 4.6% |
| Remaining allocation, API work and profiling overhead | 0.0936 | 1.0% |
| **Profiled API total** | **9.4738** | **100%** |

Percentages are rounded. The window row contains 0.7485 seconds of window
generation and 0.2123 seconds of sequential normalization sums; the rest is
associated preparation. Planning took 0.00072 seconds, input conversion
0.00473 seconds and output assembly 0.00070 seconds. These are subdivisions
of the measured total. Native preparation and segment processing subdivide
the native kernel and must not be added to it again.

The earlier fast.1 one-worker profile took 15.335 seconds. Its Fourier
coefficient generation took 3.3458 seconds versus 0.4321 in the final profile;
its native preparation took 1.1182 versus 0.7492 seconds. Segment processing
fell from 8.6982 to 7.2241 seconds in those individual profiles. They compare
version, kernel and output configuration together, and run-to-run variation
remains. The separately timed one-worker API medians were 16.071 seconds for
fast.1 (two calls) and 8.232 seconds for final fast.2 (three calls).

### Segment length matters

The same final one-worker profile can be grouped by segment length `L`:

| Segment length | Frequencies | Segment-sample visits | Segment phase, s | Worker elapsed sum, s |
| --- | ---: | ---: | ---: | ---: |
| `L < 256` | 45 | 1.922 billion | 0.5030 | 0.5098 |
| `256 <= L < 2048` | 135 | 5.765 billion | 1.4551 | 1.4793 |
| `2048 <= L < 65536` | 224 | 9.553 billion | 2.2348 | 2.3096 |
| `65536 <= L < 1048576` | 198 | 8.222 billion | 2.2811 | 2.9788 |
| `L >= 1048576` | 49 | 1.611 billion | 0.7501 | 2.1492 |

Small segments incur many repeated segment operations; very long segments
make window/coefficient preparation substantial. For the 49 longest rows,
window/normalization, coefficient generation and native preparation together
took about 1.393 seconds. Optimizing only the longest segment's DFT would
therefore miss much of the actual cost. These elapsed sums are interpretable
as sequential work because this profile uses one worker. With multiple
workers the intervals overlap.

### Changes responsible for the improvements

The fast.2 native path processes up to four PSD segments together for eligible
lengths, sharing coefficient loads while retaining segment accumulation order.
Projected coefficients are prepared in place in private buffers, reducing
memory passes and temporary storage. Fourier coefficients in `fast` use
bounded 64-position phase blocks, with direct evaluation for short or
large-phase cases. On targets with more than 64 long-double mantissa bits,
`fast` uses compensated double-precision pairs during order-zero preparation
to avoid long-double arithmetic in the per-coefficient loop.

Native generators cover the recognized standard windows and all 19 upstream
flat-top windows. The cosine-series windows generate harmonics from one
cosine with a vectorizable Chebyshev recurrence, and reuse symmetry. The
Kaiser series described next removes the remaining expensive repeated Bessel
function evaluation for its eligible range. Normalization retains the
original sequential summation order. [Numerical notes](numerics.md) explain
the rounding changes and fallbacks.

## Additional Kaiser gain in the final build

For `kernel="fast"`, lengths at least 2048 and `0 < abs(beta) <= 32`, the
final build evaluates a positive power series for the normalized Kaiser
window with SIMD Horner blocks. It chooses the degree from a bound on the
remaining positive tail; other cases use the direct generator. The standard
`psll=200` case has beta approximately 25.4026 and is eligible. This bound
controls series truncation, not all floating-point rounding or final PSD error.
The series follows the standard
[modified Bessel function power series](https://dlmf.nist.gov/10.25.E2).

The following 10M/eight-worker experiment alternated the previous fast.2
binary and the final binary, with the order reversed in the middle group.
Both use `fast`, PSD output and the same explicit 4096 MiB budget.

| Pair | Before Kaiser series, s | Final Kaiser series, s | Reduction |
| ---: | ---: | ---: | ---: |
| 0 | 2.4088 | 1.7598 | 26.9% |
| 1 | 1.9168 | 1.8876 | 1.5% |
| 2 | 2.2097 | 2.0330 | 8.0% |
| **Median** | **2.2097** | **1.8876** | **14.6%** |

All three pairs improved, but the effect size varied. One-worker medians
fell from 9.3727 to 8.2321 seconds, a 12.2% reduction. The independent
one-worker profiles locate the effect: window generation fell from 2.1390
to 0.7485 seconds (2.86×), and window plus normalization from 2.3580 to
0.9748 seconds (2.42×). A focused generator experiment measured roughly
2.9–3.2× for eligible lengths; this is not a whole-API ratio.

The additional series gain is not established at every size. At 1M the final
median was 0.3155 seconds versus 0.2124 seconds in the earlier fast.2 series,
with substantial variation. At 30M the median fell from 8.4162 to 7.9129
seconds, but the observed ranges overlap. No broad uniform-gain claim follows
from those small samples.

## Worker count, memory and output selection

For the final 10M Kaiser PSD calculation, eight workers gave the lowest
observed wall time among the tested worker counts:

| Workers | Median wall time, s | Median process CPU time, s | Maximum process RSS, GiB |
| ---: | ---: | ---: | ---: |
| 1 | 8.2321 | 8.2297 | 0.461 |
| 2 | 4.6844 | 9.3732 | 0.870 |
| 4 | 3.0367 | 12.0853 | 1.251 |
| 8, automatic budget | 2.0621 | 14.5202 | 1.749 |

Each row contains three ordinary API calls. The 1/2/4-worker rows use the
explicit 4096 MiB budget; the last row tests the default budget separately
from the fixed-budget 1.888-second headline. In its extra profile, the
automatic budget was about 4.48 GiB, maximum reserved work about 1.40 GiB,
and summed memory-gate wait only 0.005 seconds. The budget did not bind
significantly. More workers reduce wall time here while increasing total
CPU consumption and memory. This is evidence for this machine and workload,
not a universal optimum.

The default Linux cgroup memory estimate now allows for reclaimable inactive
file cache, subtracting dirty and writeback pages. Counting all cached files
as unavailable memory had unnecessarily restricted concurrency after builds
and large runs. The heuristic remains conservative about anonymous/shared
memory and is capped by the cgroup and host availability estimates. See the
[Linux cgroup v2 memory definitions](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory).
It is still a soft scheduling estimate, not a guarantee against out-of-memory
conditions or a general cgroup-hierarchy discovery mechanism.

Selected outputs reduce result construction and skip unused native variance
updates, but they do not remove the dominant segment projections. In the
final paired 10M/eight-worker experiment, `fast` medians were **1.8876 seconds
for PSD only and 1.8577 seconds for all outputs**. Earlier pairs likewise
failed to demonstrate a robust standalone PSD-only wall-time gain. Select
`outputs="psd"` or `outputs="nsd"` for the desired result and reduced unused
work, without attributing the whole-version speedup to this option.
Requesting only `enbw` is different: no spectrum is requested, so it can omit
the DFT entirely. No frequency or segment subsampling is used for PSD/NSD.

## Other windows

These one-million-sample, eight-worker measurements come from the **first
fast.2 integration**, before the final Kaiser-only addition. Medians are
from three calls per configuration. Non-Kaiser code paths were retained in
the final implementation, but these timings identify the recorded stage.

| Window | fast.1 auto/all, s | First fast.2 auto/all, s | First fast.2 fast/PSD, s |
| --- | ---: | ---: | ---: |
| Hann | 0.3721 | 0.1936 | 0.1656 |
| Hamming | 0.4068 | 0.2291 | 0.1545 |
| Blackman | 0.5621 | 0.1835 | 0.1725 |
| Bartlett | 0.2907 | 0.1435 | 0.1240 |
| Boxcar | 0.1672 | 0.2077 | 0.2233 |
| HFT248D | 4.7948 | 0.5115 | 0.3165 |

HFT248D improves by about **15.15×** in the fast/PSD configuration and 9.37×
with auto/all. Its overlap was 0.841 and actual frequency count 603; the
other windows used overlap 0.5 and 523 frequencies. Compare versions within
a window, not windows as interchangeable estimators.

Boxcar regressed in that 1M series. A final 10M follow-up, with 605 actual
frequencies, measured fast.1 at **1.8614 seconds** [1.7647–2.0709] and final
fast.2 at **1.3908 seconds** [1.1910–1.7257], a 25.3% reduction. All three new
10M calls were faster than all three old calls. The earlier 1M regression
remains part of the evidence; neither result establishes uniform behavior
for all lengths.

## Accuracy of the faster path

The final audit covers **38 configurations and 2901 frequency points each
for PSD and NSD**, at lengths 32769 and 131073. All 38 pass the explicit
combined criterion; 35 pass the strict relative criterion alone. Seven PSD
points and six NSD points in HFT248D tone cases require the absolute allowance.
Relative errors at deep nulls reach 99.8332% for PSD and 95.9162% for NSD;
they remain visible in the reports. The largest errors accepted only through
the absolute criterion are approximately `1.915e-26` PSD and `1.143e-14` NSD.

The criterion is relative error **strictly below 1% OR** absolute error at
most `1e-24` PSD / `1e-12` NSD. These absolute allowances were explicitly
chosen for the audit's normalized synthetic amplitudes and units. They are
not library defaults and do not transfer automatically to arbitrary signals.
There were no nonfinite results or frequency mismatches in this matrix.
Its reference densities contained no exact zeros, so its zero-change count
does not establish exact-zero coverage. Pinned original sources and an
independent original-C/scalar anchor were checked.

`kernel="fast"` does not check a tolerance or retry scalar computation at
runtime. It uses the permitted rounding tradeoff while preserving the
estimator plan. Use the [audit and numerical notes](numerics.md) to assess
your signal's scale and relevant spectral floor. The inherited mean and
second-moment defects are intentionally retained for compatibility.

## Reproduce API timings and profiles

Install both native cores as described in the root README. A native build
can be selected explicitly; rebuild before moving it to a different CPU.

```bash
python -m lpsd_fast.build --native
python benchmarks/bench_lpsd.py --backend fast --n 10000000 --workers 8 \
  --kernel fast --outputs psd --max-working-mb 4096 --repeats 3 \
  --output benchmark-results/fast-psd-8-10m.json
python benchmarks/bench_lpsd.py --backend fast --n 10000000 --workers 1 \
  --kernel fast --outputs psd --max-working-mb 4096 --repeats 3 --profile \
  --output benchmark-results/profile-fast-psd-1-10m.json
python benchmarks/bench_lpsd.py --backend fast --entry-point lnsd \
  --n 10000000 --workers 8 --kernel fast --repeats 3 \
  --output benchmark-results/fast-nsd-8-10m.json
python benchmarks/bench_lpsd.py --backend fast --n 1000000 --workers 8 \
  --kernel fast --outputs psd --window hft248d --overlap 0.841 --repeats 3 \
  --output benchmark-results/fast-hft248d-8-1m.json
```

Omit `--outputs` to measure the full seven-column result. Use `--kernel auto`
for the standard kernel and `--kernel scalar` for the compatibility path.
To compare against upstream, run a separate command with `--backend original`
and one worker. Comparing with fast.1 requires a separate checkout at its
pinned commit and its own compiled library; this command set alone does not
reconstruct that baseline.

```bash
python benchmarks/bench_lpsd.py --backend original --n 10000000 --repeats 1 \
  --output benchmark-results/original-10m.json
python benchmarks/check_accuracy.py --n 131073 --kernel fast \
  --relative-limit 0.01 --psd-absolute-limit 1e-24 --nsd-absolute-limit 1e-12 \
  --fail-on-limit --output benchmark-results/accuracy-131073.json
```

The benchmark checks that the original C core exists instead of timing its
Python fallback. Run configurations sequentially without other CPU work.
The profile includes per-frequency `L`, `K`, `K*L`, preparation, windows,
coefficients, native segments and memory-gate waits. A profile is an extra
instrumented call. **Do not sum overlapping worker elapsed times as wall
time**, or count child phases in addition to their parents.

## Algorithmic work and scaling

Let `N` be the number of input samples and `M` the **actual** output frequency
count. At frequency `j`, let `L_j` be the segment length and `K_j` the number of
overlapping segments. For PSD, NSD and the other spectral outputs, with a
fixed supported detrending degree, the direct implementation performs work of
order

$$
T = \Theta\!\left(N + \sum_{j=1}^{M} L_j
                       + \sum_{j=1}^{M} K_j L_j\right).
$$

For fixed valid overlap below one and the package's segment plan,
`K_j * L_j` is of order `N`, giving **Theta(N M)** work. SIMD, fewer passes
and threads reduce constants and wall time; they do not change that work bound.
With fixed logarithmic point density from roughly `fs/N` to `fs/2`,
`M = Theta(log N)` and the work is `Theta(N log N)`. A bounded actual `M`
instead gives `Theta(N)`. If the actual output count is increased in proportion
to `N`, the same direct approach has `Theta(N^2)` work. There is no single
complexity for “maximum logarithmic resolution” without specifying how that
resolution and the resulting actual `M` grow with `N`. The requested
`n_frequencies` is a planning target, so substituting it for actual `M`
without inspecting the plan can mislead.

For the measured defaults, the segment iteration counts were 2,377,782,853
at one million samples and 27,072,536,884 at ten million. These count visits
to segment samples, not CPU instructions or floating-point operations.
A full FFT and LPSD may share an asymptotic `N log N` description under the
fixed-density assumption while computing different estimators and doing very
different amounts of real work. This benchmark makes no FFTW speed comparison.

## Historical comparison with the original 1.0.6 implementation

For ten million real samples, the unchanged `lpsd` 1.0.6 public API took
**151.412 seconds** in one measured call. `lpsd_fast`, using `kernel="auto"`
and eight workers, took **2.937 seconds**, the median of three measured calls.
The ratio is approximately **51.6**, on the machine and input described below.
This combines faster per-frequency computation with parallel frequencies.

| Samples | Actual frequencies | Implementation | Workers | Measured calls | Wall time, seconds |
|---:|---:|---|---:|---:|---:|
| 1,000,000 | 577 | `auto` | 1 | 3 | 1.740 median |
| 1,000,000 | 577 | `auto` | 8 | 3 | 0.423 median |
| 10,000,000 | 651 | Original C core | 1 | 1 | 151.412 |
| 10,000,000 | 651 | `auto` | 1 | 1 | 15.476 |
| 10,000,000 | 651 | `auto` | 8 | 3 | 2.937 median |
| 10,000,000 | 651 | `scalar` | 8 | 1 | 16.184 |
| 30,000,000 | 678 | `auto` | 8 | 1 | 12.506 |

The three ten-million-sample `auto`/eight-worker wall times were 3.715, 2.937
and 2.926 seconds. The median call used 23.134 seconds of process CPU time:
CPU time includes work on all threads and is not wall time. At this size the
single-worker optimization alone was approximately 9.8 times faster than the
original; eight workers then reduced the optimized wall time by another factor
of approximately 5.3. There was no original baseline at thirty million samples,
so that row has no measured speedup.

[The curated JSON](../benchmarks/results.json) contains the individual timings,
CPU times, process memory high-water marks, conditions and numerical summaries.
These are historical measurements of the optimized implementation before its
integration into this repository, not fresh measurements of every subsequent
commit. The original source baseline is the `lpsd` code at `2fd15da`, corresponding
to the unchanged 1.0.6 implementation used in the comparison.

### Historical measurement conditions

The measurements used a single Linux x86-64 machine reporting an AMD EPYC 9V74
80-Core Processor, with an eight-CPU cgroup quota and an 8 GiB memory limit,
Python 3.12.14, NumPy 2.3.5 and pandas 2.2.3. The saved build environment records
GCC 13.3.0. The optimized C build used `-O3`, `-fopenmp-simd`,
`-ffp-contract=off` and `-march=native`; it did not use global `-ffast-math`.
The eight-CPU quota governed available CPU time. CPU clock rates were not
controlled.

The input was float64 standard-normal noise generated with seed `20261008`.
Parameters were `sample_rate=1`, `n_frequencies=1000`, `n_averages=100`,
`detrending_order=0`, and the default Kaiser window/overlap with `psll=200`.
Each process made one small warm-up call using 4,096 samples and a target of
100 frequencies before the reported calls. This is not a full-size warm-up.
Input generation, imports, comparisons and result-file output were outside
the measured API call. Planning, input conversion, window/coefficient
generation, native computation and output construction were inside it.
The original baseline and the eight-worker headline timings were uninstrumented.
The historical single-worker `auto` calls also collected phase profiles, so
their API times include profiling probes. Each JSON measurement row records
whether profiling was enabled.

The historical original call received a pandas Series and the historical fast
calls received the same numerical samples as a NumPy array. Both public API
paths included their own conversion costs. The reproduction script above uses
a Series for **both** implementations to align this interface detail. Times
also depend on compiler, CPU, cache state, memory pressure and other processes.
Three repetitions expose some variation but are not a confidence interval.

At ten million samples, the eight-worker `auto` process reached approximately
1.96 GiB RSS, versus approximately 0.51 GiB for one worker. These are process
lifetime high-water marks, not isolated allocations of a particular call.
The concurrency memory budget controls active work reservations; it is not
a hard cap on total process RSS.

All seven output columns and the frequency index agreed bitwise with the
saved original result for the benchmark noise inputs that had a reference.
That observation does not extend to arbitrary signals: see
[numerical compatibility and known limitations](numerics.md).
