# PSD FMA dispatch experiments

These are the complete retained dispatch-selection experiments from 2026-10-08.
They compare an eight-segment FP64 FMA kernel with two four-segment passes and a
restricted eight-segment dispatch. They are development measurements, separate
from the release comparison against the published fast.3 implementation.

The archive keeps all **324 timed native observations** and all **36 timed public
API calls**, including slow results. Warmups and separate diagnostic profiles
are excluded from those counts. No repetitions were dropped or added after
inspection of the results. The original JSON contents, including their original absolute paths, are
unchanged. The large API report is stored losslessly as
`raw/api-screen.json.gz`; its manifest hash refers to the decompressed bytes.
`raw-storage.json` records both storage and content hashes. The other raw
JSON files remain directly readable in `raw/`. `summary.json` contains derived medians;
the raw files remain the authoritative measurements.

## Exact source identities

`control8` is the frozen, then-uncommitted eight-segment FMA implementation with
the separate CSD optimization already integrated. Its native source SHA-256 is
`70d34dab5c064bc89b6d12d2f97e069e5e52ffccb5462a09a7b617bffa758730`.
It is reconstructed from commit
`fc838aafa5de75fb1375574e0d8146756d4ab831` plus
`patches/fc838-to-control8.patch`. That patch includes the five differing package
files and the two then-modified shared benchmark modules. It is necessary:
the native source at the named commit is not the measured snapshot.

Every variant is a patch against that exact `control8` snapshot. The source
manifest binds the original Git files, the reconstructed snapshot, each changed
variant file, and the untouched raw results and original scripts. The original
compiler and source/binary reports are also retained in `builds/`. The measured
builds used GCC 13.3.0, `-O3 -fno-fast-math -ffp-contract=off`, OpenMP SIMD and
portable x86 ELF target clones. They did not enable global floating-point
contraction or `-march=native`.

`strict` is a control for this dispatch experiment, **not a checkout of the
published fast.3 release**. It uses the same native library as `control8`, but
sets Python's `use_bounded` to false. Thus it performs no full-input FMA bound
scan, calls the existing selected entrypoint instead of the bounded entrypoint,
does not check projected coefficients for FMA eligibility, and executes no
fused PSD batches. The CSD code is identical in all these variants. A release
claim must use the separate final comparison against the real fast.3 checkout.

## Candidates

Here `L` is segment length. All fused candidates preserve the original segment
starts, the initial strict periodograms and the statistics updates. Each outer
group still contains eight segments, even when its dots use two four-segment
functions. Accordingly, `fused_segment_batches` counts outer groups of eight.

| Candidate | Change from the frozen eight-segment implementation |
| --- | --- |
| `control8` | Eight-segment AVX-512 FMA on the original eligible lengths. |
| `strict` | Disable the bounded/fused route completely in the Python client. |
| `split256_large` | Two four-segment AVX/FMA256 calls for `L >= 2048`; retain eight-segment FMA below that. |
| `split256_all` | Two four-segment AVX/FMA256 calls for every eligible `L >= 128`. |
| `split512_large` | Two four-segment AVX-512 calls for `L >= 2048`. An empty inline-assembly register constraint prevents GCC from duplicating each coefficient load into every memory-source FMA. |
| `selective8` | Restrict eight-segment FMA to `128 <= L < 256`, `1024 <= L < 2048`, and `8192 <= L < 65536`. |
| `split256_middle` | Use four-segment AVX/FMA256 only for `256 <= L < 1024`; keep eight-segment FMA elsewhere. Also require enough segments for a fused group before scanning coefficients. |

The first six candidates were measured together in the fixed native screen.
`split256_middle` was selected from that screen and then compared in the full
API screen. Its coefficient-bound guard requires at least ten segments in the
middle interval, where the first two projections are strict, and sixteen
elsewhere, where the original initial group of eight remains strict. This avoids
coefficient scans when there cannot be any fused group. The four-segment helper
has its own FMA instruction-set check; AVX2 support alone is not used as evidence
of FMA support.

The archived middle candidate native source SHA-256 is
`77b2d04053f5f4ebc5fad0b609804473af52f13b7a02b001a23b88bcb61c8808`.
It uses the experimental helper name `dot_psd_four_fused_private` and evaluates
the extra FMA capability gate within each outer batch. The final integration
renames the helper and moves the invariant capability decision before the
segment timer. Those final sources and their timings belong to the subsequent
release validation, not to these archived identities.

Subsequent release validation kept the native kernel but restricted Python's
bounded/FMA selection to calls with one effective worker, because the
eight-worker gain was not reliable. That final wrapper restriction was not in
effect during this archived screen. In particular, the archived eight-worker
FMA results describe an experimental route rather than the shipped dispatch.

## Fixed native frequency screen

The input was one resident array of 10,000,000 float64 values from
`numpy.random.default_rng(20261008).standard_normal`. All candidates received
the same periodic Kaiser window coefficients, actual frequency bin, segment
length and overlap. The frequency plan requested 1,000 frequencies and 100
averages with Kaiser PSLL 200 dB. Coefficients were read-only for these calls.

The target lengths were fixed before timing:

```text
160, 224, 320, 512, 768, 1280, 1792, 3072, 6144,
12288, 24576, 49152, 98304, 196608, 393216, 786432,
1572864, 3145728
```

For each target, the nearest actual length in that plan was used. Each method
had one native warmup, followed by three repetitions in forward, reverse,
forward method order. With an odd repetition count, each method has two forward
positions and one reversed position; this is not a fully balanced crossover.
The raw rows preserve that order. The native call wall timer includes projection
preparation, while native preparation and segment timers are reported separately.

The largest relative power difference in the saved native screen was
`3.561907063341268e-14`. This ordinary-noise check is not a replacement for the
separate regression tests covering small AC signals on DC, early transients,
tails, nonfinite values, overflow and the exact eligibility boundary.

The four-segment AVX/FMA256 candidate improved the middle interval. Its segment
time ratios against `strict` were 1.81 at `L=320`, 1.88 at `L=509`, and 1.12 at
`L=772`. Extending the four-segment choice to long vectors was frequently slower.
For example, at `L=6098`, segment medians were 7.36 ms for `strict`, 6.96 ms for
`control8`, and 9.04 ms for `split256_all`. The four-segment AVX-512 alternative,
even with coefficient loads retained in registers, also failed to provide a
consistent long-vector benefit. All lengths and every candidate are retained,
including those negative results.

The selected interval follows an existing baseline dispatch boundary; no new
fine-grained length threshold was fitted to an isolated timing result. Fewer
simultaneous input streams and a smaller short-vector working set are plausible
reasons for the middle-interval benefit. These experiments do not isolate
cache conflicts, memory bandwidth, or processor clock effects, so those remain
explanations to investigate rather than measured causal conclusions.

## Complete API confirmation screen

The three methods were `strict`, `control8` and `split256_middle`. The workload
used the same seeded float64 pandas Series, Kaiser PSLL 200 dB, 1,000 requested
frequencies, 100 requested averages, order-zero detrending, `kernel="fast"`,
`outputs="psd"`, a 4096 MiB working-memory limit and a 128 MiB window cache. Each
call independently prepared its plan, windows and coefficients. Input creation,
imports, warmups, output comparisons, provenance collection and JSON writing
were outside the wall timer.

Each case had one full warmup per method and four repetitions in the orders
`strict/control8/middle`, `middle/control8/strict`, repeated twice. All three
methods have four timed calls per case. Separate instrumented calls were made
for the one-worker cases only. Package and native identities were checked again
at the end, and the report completed successfully.

| Samples | Workers | Strict median | Eight-segment median | Middle candidate median | Candidate time reduction vs strict | Candidate faster pairs vs strict / eight-segment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 1 | 0.609308 s | 0.532690 s | 0.546762 s | 10.27% | 3/4 / 1/4 |
| 10,000,000 | 1 | 7.849814 s | 7.566484 s | 7.386275 s | 5.91% | 4/4 / 3/4 |
| 10,000,000 | 8 | 1.818901 s | 1.534028 s | 1.670528 s | 8.16% | 2/4 / 1/4 |

The middle candidate improved the large single-worker case against the
eight-segment implementation and the strict control. It was slower than the
eight-segment implementation in the million-sample case and the eight-worker
case. The latter was variable, with only two of four pairs faster than strict;
its lower median does not establish a universal parallel speedup. The result
supported a focused final candidate, followed by independent release validation.
The 10.27% first-row time reduction corresponds to 11.44% higher throughput;
those percentages use different denominators and must not be interchanged.

## Replay

The scripts require Python with the project's NumPy, pandas and SciPy
dependencies, Git, and a compatible C compiler. The recorded experiment used a
Linux x86-64 host supporting AVX-512 FMA; hardware without the supported ISA uses
fallback paths and cannot reproduce the same dispatch comparison. Clocks and
load on the shared host were uncontrolled, so identical source/input hashes do
not promise identical timings.

From a checkout containing the pinned commit, choose a new scratch directory:

```sh
CC=cc LPSD_TARGET_CLONES=auto python benchmarks/experiments/fma_dispatch/width_screen/prepare.py \
    --repo . --output /tmp/lpsd-fma-dispatch --build
python benchmarks/experiments/fma_dispatch/width_screen/replay_frequency.py \
    --artifacts /tmp/lpsd-fma-dispatch --output /tmp/lpsd-fma-dispatch/frequency-screen.json
python benchmarks/experiments/fma_dispatch/width_screen/replay_api.py \
    --artifacts /tmp/lpsd-fma-dispatch --output /tmp/lpsd-fma-dispatch/api-screen.json
```

Serialize these builds and benchmarks with other CPU-intensive work. Omitting
`--build` reconstructs and verifies source files only. The preparation script
refuses an existing output directory, reads the repository through pinned Git
objects, and creates independent temporary repositories solely to apply patches;
it does not change or commit anything in the source checkout. It copies the
newly built control library to `strict`, matching the original experiment.

The replay scripts keep the original inputs, targets, warmups, repetition counts,
method order and timer boundaries. Their only configuration additions are the
artifact/output paths and source preflight. They import the frozen benchmark
helpers from the reconstructed `control8` tree instead of the evolving checkout.
Rebuilding with another compiler or toolchain may produce a different binary;
compare the new build reports with the originals instead of presenting the
recorded binary hashes as hashes of the rebuilt files. The path-specific scripts
used for the recorded observations are preserved unchanged in `original/`.

The source-only reconstruction was executed successfully from the pinned commit:
all seven candidates passed all 25 expected source-file hashes each, and no
native libraries were created. The original and portable Python scripts also
passed source parsing without importing the packages. The recorded reconstruction
command and script/patch hashes are retained in `source-validation.json` and
`manifest.json`. The portable benchmark entrypoints themselves were not rerun
while the final release timings occupied the shared CPU slot.
