# FMA segment dispatch: retained decisions and measurements

This archive records the candidates that led to the single-worker FMA
selection in fast.4. It includes less favorable complete-call results, the
source snapshots that produced them, and the later width-selection screen.
The release results are separate in [performance-fast4.md](../../../docs/performance-fast4.md).

Every factor is baseline time divided by candidate time. Values below one
mean slower. All timings are from the shared AMD EPYC 9V74 execution
container. Each of the two integrated comparisons below used six fixed
balanced pairs of complete public API calls, one complete warm-up per
implementation, and the separately compiled fast.3 baseline at
`9f3071ca2b7134a585f1727506a4e1472a7da299`. No observations were discarded.
The paired geometric summaries and ranges are descriptive, not confidence
intervals or predictions for other hardware.

## 1. Integrated eight-stream FMA

The first integrated candidate allowed bounded FMA with any worker count
and used eight-stream AVX-512 operations at all eligible lengths. This was
numerically validated but did not establish a consistent larger complete-
call PSD improvement. The eight-worker case raised a regression risk.

| Kind | Samples | Workers | fast.3 median | Candidate median | Ratio of medians | Paired geometric factor | Paired range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CSD | 1,000,000 | 1 | 1.181048 s | 0.946322 s | 1.2480x | 1.2574x | 1.1897–1.3759x | 6/6 |
| CSD | 10,000,000 | 8 | 4.569309 s | 3.075242 s | 1.4858x | 1.3896x | 0.9453–1.6802x | 5/6 |
| PSD | 1,000,000 | 1 | 0.623370 s | 0.595310 s | 1.0471x | 1.0172x | 0.7374–1.1343x | 5/6 |
| PSD | 10,000,000 | 1 | 7.371827 s | 7.263547 s | 1.0149x | 1.0167x | 0.9672–1.0821x | 3/6 |
| PSD | 10,000,000 | 8 | 1.438620 s | 1.893739 s | 0.7597x | 0.8900x | 0.6649–1.1888x | 3/6 |

[All calls and summaries](integrated-eight-way.json) and the
[lossless raw report](integrated-eight-way-raw.json.gz) retain the complete
series, including all per-frequency profiles. The CSD rows also confirm the
independent segment-sharing change; they do not establish an FMA CSD benefit,
because CSD never enters this FMA route. Earlier isolated CSD measurements
remain separate in the [CSD report](../../../docs/csd-segments.md).

The exact package snapshot is [integrated-eight-way.patch](integrated-eight-way.patch)
against `fc838aafa5de75fb1375574e0d8146756d4ab831`.

## 2. Native width screen and three-way API screen

The [width_screen archive](width_screen/README.md) contains six native
candidates, the subsequent middle-length candidate, all 324 native timed
observations and all 36 timed three-way API calls. Its strict control has
the integrated CSD implementation and disables Python's bounded route; it
is not the published fast.3 baseline. That distinction is retained in all
reports.

Four-stream passes for long vectors were slower and were rejected. The
middle-length 256-bit kernel repaired the eight-stream layout's weak
256–1023 interval. The source-only replay reconstructed all seven candidates
with all 175 expected source hashes verified. Its benchmark scripts retain
the original workloads and order; the portable replay was not rerun as a
new performance experiment. See its README for exact commands and limits.

## 3. Integrated four/eight-stream candidate before worker restriction

The mixed-width candidate was integrated, its invariant capability check
was moved before the segment timer, and the entire package was rebuilt.
At this stage FMA was still available to parallel frequency workers.

| Kind | Samples | Workers | fast.3 median | Candidate median | Ratio of medians | Paired geometric factor | Paired range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSD | 1,000,000 | 1 | 0.584540 s | 0.532444 s | 1.0978x | 1.0948x | 1.0261–1.2074x | 6/6 |
| PSD | 10,000,000 | 1 | 7.498051 s | 7.109991 s | 1.0546x | 1.0527x | 1.0160–1.0907x | 6/6 |
| PSD | 10,000,000 | 8 | 1.532320 s | 1.590088 s | 0.9637x | 0.9480x | 0.7691–1.1076x | 2/6 |

The one-worker cases improved in every pair. The eight-worker case did not
show a dependable advantage: two of six pairs improved, and both its median
and paired summary were slower. This was the concrete reason for restricting
the shipped Python dispatch to **one effective worker**. It was not treated
as a fixed universal regression percentage or concealed by selecting a more
favorable earlier session.

[All calls and summaries](unrestricted-four-eight.json) and the
[lossless raw report](unrestricted-four-eight-raw.json.gz) retain this series
and its two scalar-reference audits. The exact package snapshot is
[unrestricted-four-eight.patch](unrestricted-four-eight.patch), again against
`fc838aafa5de75fb1375574e0d8146756d4ab831`. Its reconstructed Python API and
every modified package source were verified against the hashes in the
recorded comparison before archiving.

The final implementation keeps this native kernel. Its Python selection
adds `workers == 1`. Two regression tests exercise both an explicit parallel
worker count and the resolved default count, requiring the ordinary native
entry point and unchanged output. The final wrapper's own complete-call
comparison is in the release report, not in the historical rows above.

## Timing and numerical scope

The common workload is resident seeded float64 pandas input, sample rate 1,
Kaiser PSLL 200, 1,000 requested frequencies, 100 averages, order-zero
detrending, `kernel="fast"`, PSD-only output, a 4096 MiB working-memory
budget and a 128 MiB window cache. Both versions receive identical inputs
and independently prepare each call. Input generation, imports, warm-up,
comparisons, hashes and file writes are outside the API wall timer.

All timed and profiled PSD/CSD values in these two integrated comparisons
are bit-identical to their recorded fast.3 reference, with matching grids
and dtypes. Repeating the same seeded input does not create independent
adversarial accuracy cases; those are covered separately by the regression
suites and scalar audits.

Every attached profile in these historical comparisons uses **one worker**,
including profiles stored beside eight-worker ordinary timings. They cannot
identify the cause of an eight-worker regression and are not extra timing
repetitions. Instrumented stage sums must not be added to complete-call
medians or interpreted as measured memory traffic.

## Replay the integrated snapshots

Use a separate checkout of the pinned base, apply one package snapshot,
and apply [replay-harness.patch](replay-harness.patch) to recover the two
then-modified benchmark helpers. The width-screen archive has its own
complete preparation script and does not need this separate harness patch.
For example:

```sh
git worktree add --detach /tmp/lpsd-fma-integrated fc838aafa5de75fb1375574e0d8146756d4ab831
git -C /tmp/lpsd-fma-integrated apply /path/to/archive/unrestricted-four-eight.patch
git -C /tmp/lpsd-fma-integrated apply /path/to/archive/replay-harness.patch
make -C /tmp/lpsd-fma-integrated compile
```

Run `benchmarks.bench_segment_core` from that prepared checkout, with its
`--baseline-root` pointing to a separately compiled fast.3 checkout. Choose
`--kind psd` or `--kind csd`, the exact sample count and worker count from
the table, and `--repeats 6`. Keep the compiler flags matched and serialize
all deliberate benchmark/build workloads. Exact hashes do not control
shared-host load, clocks or memory placement, so replayed times can differ.

The raw `.json.gz` files are lossless JSON archives. Their companion summary
files record both stored and uncompressed hashes. No native binaries are
committed; rebuild from the recorded sources and inspect the new build report.
