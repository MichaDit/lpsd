# Independent segment batches and measured length thresholds

This experiment improves the existing direct LPSD kernel. It keeps every
sample, frequency, segment start, per-segment input anchor and legacy mean
update. It does not use an FFT approximation, change detrending, alter the
original final M2 calculation or enable floating-point contraction.

The previous projected PSD kernel shared its coefficient loads across four
segments at `L >= 2048`. On AVX-512, the new kernel holds sixteen independent
SIMD accumulators and eight input anchors in the larger register file and
shares each coefficient load across eight segments. The final eight-way
path uses `128 <= L < 256` or `L >= 1024`; four-way and single-segment
remainders remain available. The AVX2, baseline x86 and AArch64 paths
retain the previous batching. AArch64
has enough vector registers to make another experiment plausible, but its
speed has not been measured here.

Portable x86 `target_clones` builds select the larger batch through a runtime
AVX-512 test. `__AVX512F__` alone would miss the AVX-512 clone because those
builds preprocess the source with baseline flags. A native AVX-512 build can
resolve the batch choice at compilation.

The initial candidate applied eight-way batching to every `L >= 128`. Its
complete original data remains in `results.json`; the accepted threshold
refinement and subsequent measurements are recorded separately below. The
two versions must not be combined into one timing series.

## Initial eight-way candidate: complete API timing

[results.json](results.json) contains every retained observation, the initial
prototypes, source/binary hashes, measured scripts, the exact frequency plan,
and complete frequency-level profiles. The host was an AMD EPYC 9V74 with an
eight-CPU quota and nine logical CPUs in its affinity mask. No other task
benchmark or build ran concurrently on this host. Frequencies and other tenants of the
shared host were not controlled.

The main measurement uses ten million float64 samples in a pandas Series,
PCG64 seed 20261008, sample rate 1, Kaiser PSLL 200, 1000 requested frequencies,
100 requested averages, order 0, `kernel="fast"`, `outputs="psd"`, eight
workers, and a 4096 MiB working-memory budget. The actual plan has 651
frequencies, beta 25.402566024390598 and overlap 0.7658465180619339. Both native
libraries use GCC 13.3, `-O3 -march=native -fopenmp-simd`,
`-fno-fast-math -ffp-contract=off`.

Every library/configuration receives one complete warm-up call. Each timed
call then recreates its window cache, coefficients, projection and output.
Imports, input creation, comparison and JSON writes are outside the timer.
The eight pairs alternate A/B, B/A ordering. No outliers were removed.

| Pair | Baseline wall seconds | Eight-way candidate wall seconds | Baseline / candidate |
| --- | ---: | ---: | ---: |
| 1 | 1.866587 | 1.597430 | 1.1685 |
| 2 | 1.792951 | 1.761909 | 1.0176 |
| 3 | 1.477408 | 1.324064 | 1.1158 |
| 4 | 1.806809 | 1.393725 | 1.2964 |
| 5 | 1.603568 | 1.461735 | 1.0970 |
| 6 | 1.569491 | 1.304986 | 1.2027 |
| 7 | 1.718975 | 1.498000 | 1.1475 |
| 8 | 1.979675 | 1.637803 | 1.2087 |
| **Median of each series** | **1.755963** | **1.479867** | **1.1866** |

That is a **15.7% median wall-time reduction** on this host and configuration.
Each paired comparison favored the candidate, although the observed margin
varied substantially. All resulting float32 PSD arrays were byte-identical.
This establishes neither an architecture-wide speed guarantee nor a
universal bound on numerical differences for other signals.

The earlier three-repetition prototype results remain in the JSON. In
particular, their eight-worker medians were noisy and mixed; they did not
justify a performance claim. The accepted candidate also retains four-way
remainders, which those prototypes did not. The later balanced series must
not be described as repeated samples of exactly the same prototype binary.

## Initial candidate: where the time changed

Two subsequent, separately instrumented one-worker calls isolate the segment
work. Their channel wall times were 8.2301 seconds for the baseline and
7.6594 seconds for the candidate; the complete API wall times were 8.2369
and 7.6663 seconds. These are diagnostic single calls, separate from the
ordinary eight-worker timing medians.

| Segment length | Baseline segment seconds | Candidate segment seconds |
| --- | ---: | ---: |
| `L < 256` | 0.534911 | 0.415784 |
| `256 <= L < 2048` | 1.330235 | 1.495752 |
| `2048 <= L < 65536` | 1.970617 | 1.418421 |
| `65536 <= L < 1048576` | 1.936700 | 1.755935 |
| `L >= 1048576` | 0.616397 | 0.614904 |
| **All segments** | **6.388860** | **5.700795** |

The medium/long segment work and the shortest group improve; the middle
group regresses in this single profile. Changing thresholds from that one
observation would risk fitting host noise or particular lengths. The saved
microbenchmark explores lengths 128 through 1048576 with randomized variant
ordering and five measured repetitions after an initial repetition. It
includes the imaginary-division shortcut, four/eight-way thresholds, and
split-half/four-quarter dot accumulators. The latter two changes did not
show a reliable general improvement and were not adopted. A later matched
per-length series, described below, provides the evidence for the final
threshold change.

The microbenchmark uses a separate, controlled coefficient problem: beta
23.7, fractional bin 13.712, 75% overlap and ten million samples. Its timings
exclude native coefficient preparation; complete native wall times are also
saved. Those per-length speed ratios are not whole-API speedups.

## Accepted adaptive threshold and final measurements

[results_adaptive.json](results_adaptive.json) preserves the follow-up data,
the exact measurement programs, source manifests, all complete-call samples,
and the portable-build validation. The raw `candidate` binary alias refers
to the initial eight-way implementation (`de9e70a5…`), while `adaptive` is
the final length-dependent implementation (`3454347b…`). The archive makes
these aliases explicit; the original raw program fields remain unchanged.

The diagnostic middle-band result was checked against **all 135 actual
segment lengths from 256 through 2047** in the ten-million-sample plan.
Frequency order was randomized. After one initial pair, each frequency
received five alternating pairs; the starting implementation alternated
with the randomized frequency index. There are therefore three/two starting
orders within a particular frequency, rather than an exactly balanced split.
The test used the real plan's Kaiser beta, fractional frequency bins, overlap,
segment counts and fresh coefficient copies. Native segment time excludes
coefficient preparation.

| Length range | Frequencies | Previous kernel: sum of per-frequency medians, s | Initial eight-way: same sum, s | Eight-way faster frequencies |
| --- | ---: | ---: | ---: | ---: |
| `256 <= L < 512` | 45 | 0.429790 | 0.517665 | 5 / 45 |
| `512 <= L < 1024` | 45 | 0.398346 | 0.438425 | 4 / 45 |
| `1024 <= L < 2048` | 45 | 0.489737 | 0.345008 | 40 / 45 |

This supports retaining the previous single-segment loop for
`256 <= L < 1024`, and retaining the eight-way loop from 1024 upwards.
The sums in this table are diagnostic aggregates of independent native
calls; they are not measurements of a complete API call.

The final native source was then compared in complete API calls:

| Comparison, eight workers | Samples per library | Reference median, s | Adaptive median, s | Ratio of medians | Median wall-time reduction | Adaptive wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10M: initial eight-way vs adaptive | 8 | 1.561236 | 1.486272 | 1.0504× | 4.80% | 4 / 8 pairs |
| 30M: original fast.2 vs adaptive | 5 | 6.280836 | 5.919844 | 1.0610× | 5.75% | 4 / 5 pairs |

Both series use one complete warm-up per library and alternating A/B, B/A
pairs. The five-pair 30M series has three A/B and two B/A orders. The 10M
refinement result is noisy: half of the individual pairs favor each version.
It does not establish a robust separate five-percent whole-call gain from
the threshold change. The 30M adaptive observations range from 4.8337 to
8.4844 seconds; the slower first observation is retained. All 26 output
PSD arrays are byte-identical to their corresponding reference, including
the stored 651-point and 678-point grids.

The initial fast.2-to-eight-way 10M ratio and the later
eight-way-to-adaptive 10M ratio must **not be multiplied** to imply a direct
measurement of fast.2 against the final kernel. They were measured in
different sessions and include substantial host variation.

The exact committed final sources independently rebuild to the measured
native binary hash. A separate portable GCC build requires
`LPSD_TARGET_CLONES=1`; its generated resolver was called and its returned
address matched `fast_dft_impl.avx512f` on this host. The source still provides
default and AVX2 clones, and the eight-way gate requires 64-bit x86. Clang
was unavailable in this container. The same final kernels subsequently passed
the integrated Linux-Clang CI job; this is separate from the local GCC results.

## Numerical and compatibility checks

The focused existing tests passed: 137 tests, with the two documented
preexisting expected failures. The initial 161 native tests cover lengths
on both sides of the batch thresholds, eight/four/single remainders,
fractional hops, projected order 0/1, compensated mode 3, white noise,
10 V DC plus nanovolt noise, off-bin tones and exact constant inputs.

The native tests also compare NaN/infinity/overflow categories with the
unbatched native entry point. The public Python API rejects nonfinite input,
but the C implementation still must not silently change its exceptional
arithmetic. PSD-only and full-output means are required to match exactly
within the same candidate. The existing scalar path and upstream source
files are unchanged.

The final suite expands this to **209 native batching tests**, including 36
cases where a first-hop impulse makes `P0/P1` approximately `2^50` through
`2^56`. That stresses the inherited `P0 + (P1-P0)` update: tiny projection
differences could otherwise become material after cancellation. The old
first-four path supplies the reference, and the new path preserves it within
the one-percent criterion. No special first-batch fallback was needed for
the measured native and portable GCC builds. Together with the selected
output and fast-mode tests, **280 tests passed in each build**. The private
reproducer also passed a small one/two-worker, profiled API smoke comparison
with complete PSD byte equality and verified source/binary hashes.

## Reproduce the isolated comparison

The original measured baseline commit was `b8b921b`; its canonical equivalent
after linking the coauthor identity is
`3f2864391db3bfb22942d99106026358fef5fe96`, with the same repository tree.
The experiment also records native source hashes, so attribution metadata
does not affect source identity. The baseline native binary SHA-256 is
`b0ab2e54a6231bd54b8d01c9c364efadb924c8e41dcc0398ef637f29d3e9e623`;
the measured initial candidate is
`de9e70a504dbe82131fcca52b483c2fb2e23d17f765ca3c99e5dbb7ed886a773`.
The final adaptive candidate is
`3454347b824ea51a55996db74f9e7cd09cf22068e00b8005c00fae35e1f413c9`.

From the repository root, with the project dependencies installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python benchmarks/experiments/segment_batches/prepare.py --native

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python benchmarks/experiments/segment_batches/replay.py \
  --sizes 10000000 --workers 1 8 --pairs 8 --profiles
```

`prepare.py` captures the baseline and current candidate C include trees and
builds private libraries from those snapshots. It also captures the baseline
Python wrapper for both libraries, isolating the native segment change from
later API additions. Each source and binary is hashed and verified before
replay. Production libraries and source files are not modified. `--baseline-ref`
can select another comparison base explicitly; omit `--native` to exercise
the portable compiler/runtime dispatch instead.

The replay makes a new measurement; it does not overwrite `results.json` or
pretend to reproduce the host's earlier scheduling and clock conditions.
The recorded native source hash determines whether the current candidate
contains further changes beyond this original experiment.
