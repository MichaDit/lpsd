# Packed-real reconstruction and portable watchdog follow-up

This report describes the changes measured at fast.9 runtime
[`f4082f04f1fbfee166f40e9bcba48f73056eed40`](https://github.com/MichaDit/lpsd/commit/f4082f04f1fbfee166f40e9bcba48f73056eed40).
The comparison baseline is the reconstruction method from fast.8
[`b8c500217f7c0ad996f55b039bbf09d685800614`](https://github.com/MichaDit/lpsd/commit/b8c500217f7c0ad996f55b039bbf09d685800614).
Both sides use the already available packed-real disk FFT. The earlier
[full versus packed-real study](performance-platform-followup.md) is a
different experiment.

## Changes

The packed-real reconstruction used a Python loop to construct the phase
indices one column at a time. Eligible tiles now construct all column
offsets and broadcast the frequency indices into the existing tile in one
NumPy operation. The existing real scratch and index vector provide the
storage; no additional reconstruction array or larger budget is introduced.
Unusually wide tiles retain the previous bounded path.

These indices are integers no larger than the existing limit of 2^52. Their
intermediate additions and products are exactly representable in float64.
The subsequent complex arithmetic, original-N frequency grid, normalization
and output formats retain their existing behavior. The real scratch used
by the new path is included in the external-output alias checks.

Two independent resource-monitoring fixes accompany the optimization:

| Area | Observed problem | Correction |
|---|---|---|
| Windows private installation | The fast.8 post-merge portable job stopped while `uv` moved or removed an installation directory. PowerShell reported that it could not find a path. | Preserve the bytes already counted when enumeration is interrupted. Handle terminating and nonterminating missing-path exceptions, including native Windows error codes 2 and 3. Other error types remain fatal, and the settled installation is scanned again. |
| Live temporary-disk budget | The watcher counted root-level files but missed the files inside the exclusively owned `fft-work-*` directories introduced in fast.8. | Count the known input, FFT workspace and power files at their actual locations, including immediate owned FFT directories. Count logical lengths, including sparse files; reject symbolic links and Windows reparse points. |

The original Windows failure is retained in
[run 38219473997, Windows job 114711981308](https://github.com/MichaDit/lpsd/actions/runs/38219473997/job/114711981308).
Its short log does not identify the exception class. The native missing-path
classification is the supported explanation, backed by a regression that
fails on the old implementation and passes on the correction. The new
[Windows job 114772969610](https://github.com/MichaDit/lpsd/actions/runs/38237887942/job/114772969610)
completes the actual private installation and subsequent numerical checks.

The disk fix closes a monitoring gap; it does not establish that the earlier
ordinary runs exceeded their budgets. Their preflight allocation checks were
already present. Only known files in the private case directory are examined.
Files disappearing during cleanup are tolerated; unrelated metadata errors
stop and reap the owned worker.

## Complete-call comparison at 100 million samples

The [raw paired report](../benchmarks/results/packed-phase-100m-20261012.json)
contains four configurations, three alternating pairs per configuration,
and **24 complete PSD+NSD calls**. Its seven source-file hashes match the
measured runtime above. The input hash, FFTW binary hash, native binary hash,
all individual timings and comparison checks are retained.

This Linux x86-64 cohort uses Python 3.12.14, NumPy 2.3.5 and FFTW 3.3.10,
eight FFTW/native workers, 128 MiB FFT scratch and a 384 MiB native working
allowance. Both variants actually execute the N/2 packed transform, with
inner length 50,000,000 and factors 6250 × 8000. The original-N output grids
contain 704 and 1408 points.

| Method / actual points | Old median wall [s] | New median wall [s] | Reduction of wall medians | Old → new median CPU [s] |
|---|---:|---:|---:|---:|
| Simple FFTW / 704 | 9.653 | 7.800 | 19.19% | 10.438 → 8.582 |
| Simple FFTW / 1408 | 8.490 | 8.530 | −0.47% | 9.269 → 9.313 |
| Matched / 704 | 22.252 | 21.898 | 1.59% | 70.439 → 71.389 |
| Matched / 1408 | 33.120 | 30.061 | 9.24% | 124.092 → 121.517 |

Reduction means `1 - median(new) / median(old)`; a negative value denotes
a longer elapsed time. These medians describe this observed series and
should not be read as stable, causal improvement percentages.

| Method / actual points | Old wall range [s] | New wall range [s] | Median of the three individual pair reductions |
|---|---:|---:|---:|
| Simple FFTW / 704 | 9.465–10.304 | 7.279–10.214 | 17.58% |
| Simple FFTW / 1408 | 8.378–9.291 | 6.705–8.644 | 8.19% |
| Matched / 704 | 20.739–22.300 | 20.568–22.241 | 0.82% |
| Matched / 1408 | 30.489–34.059 | 29.658–34.312 | 1.40% |

All four old/new ranges overlap. The difference between the reduction of
separate medians and the median of paired reductions is substantial in
some configurations. Nine of twelve pairs have shorter new wall time;
six of twelve have shorter new CPU time. Every pair remains in the report.

The phase timings expose variation outside the changed code. In the first
simple-FFTW/704 pair, the full call improves by 3.025 seconds, but the
unchanged FFT itself accounts for 2.915 seconds of that difference. In the
best matched/1408 pair, the whole call improves while the reconstruction
phase is slower. Even the reconstruction phase varies enough that its
median does not improve in every configuration. This cohort therefore does
not establish a general 10% or 20% complete-call gain.

The small phase-construction experiment and the separate platform pairs
support retaining this bounded, exactly equivalent optimization. The large
run confirms numerical behavior at 100M and provides transparent timing
evidence, including the cases that do not improve. Files fit within the
host's 8-GiB cgroup ceiling. No cache flush or exclusive host-I/O control
was imposed; this is not an out-of-RAM or billion-sample measurement.

## Numerical evidence and timing scope

All **12,672 PSD and 12,672 NSD point comparisons** are bit-identical, and
all frequency grids agree. All pairs pass both the 1% gate and the stricter
3-ppm gate with zero absolute and relative error. The harness also verifies
the final input checksum. Equality is checked on every returned array byte;
raw spectra are not stored in this timing report.

The timer includes fresh FFTW planning, original-N windowing and
preprocessing, the FFT, reconstruction, energy checks, grouping, smoothing
and native low frequencies where applicable, both outputs, and cleanup.
Input generation, input hashes, imports, library loading and selection of
the reference class are outside the call timer.

The reference subclass replaces only `PackedRealDiskFFT._tile` with its
verbatim fast.8 implementation. The remaining code, scratch, transform,
input and resource settings are shared. The harness verifies the frozen
method's SHA-256 before starting:

```text
8bb729171b0bf4c6a2e300fc6e1882172b8f58b039e4f29a40424c099578f938
```

Comparisons retain the same tile shape on both sides. Different tile shapes
can already differ by a few ULP in fast.8 due to complex NumPy operations;
the tests do not introduce a false cross-tile byte-identity requirement.

## Platform and regression checks

The measured runtime passed all eight
[native test/build jobs](https://github.com/MichaDit/lpsd/actions/runs/38237887933)
and all four
[portable bootstrap jobs](https://github.com/MichaDit/lpsd/actions/runs/38237887942).
The portable cohort covers Linux x86-64, Linux ARM64, macOS ARM64 and
Windows x64. Each platform performs the new packed-phase comparison at
N=1,000,000, with both estimators, both actual output counts and three
alternating pairs. CI timings describe these particular shared runners.

The [platform evidence JSON](../benchmarks/results/packed-phase-platforms-20261012.json)
preserves all 16 configurations and 48 pairs, including CPU/phase timings,
ranges, job and artifact identities, archive digests, and all 28 source-file
checks. All **41,544 PSD and 41,544 NSD comparisons** are bit-identical.
The reductions of complete-call wall medians are:

| Platform | Simple / 577 points | Simple / 1154 | Matched / 577 | Matched / 1154 |
|---|---:|---:|---:|---:|
| Linux x86-64 | 8.28% | 8.09% | 1.26% | 0.51% |
| Linux ARM64 | 5.04% | 5.46% | 1.17% | 0.47% |
| macOS ARM64 | 11.96% | 6.66% | 0.51% | −1.64% |
| Windows x64 | 9.94% | 8.42% | 1.59% | 3.10% |

The simple-FFTW pairs support a useful improvement in these small cases on
all four tested runner platforms. Seven of the eight simple-FFTW old/new
ranges do not overlap; the macOS/1154 range does. Matched improvements are
smaller and mixed: the macOS/1154 median is 1.64% slower with nonoverlapping
ranges, and several other matched ranges overlap. No universal matched
speedup or physical-device result is inferred from these CI measurements.

The complete local regression run passed **2,137 tests**, with four skips
and two inherited expected failures. The Windows watchdog tests ran under
PowerShell 7.6.6 instead of being skipped. A subsequent focused run of the
final reference/harness and packed-real tests passed 103 tests; these
overlap the main suite and are not an additional 103 unique tests.
The measured packed-phase revision `f4082f04f1fbfee166f40e9bcba48f73056eed40`
is covered by the successful CI jobs above. The later integration has its
own [validation record](performance-fast9.md).

New disk-budget tests exercise actual termination and reaping of a child
that creates an oversized sparse FFT file. They also cover cleanup races,
metadata failures, root/nested symbolic links and preservation of an
outside sentinel. The Windows regression checks both forms of path loss,
partial byte totals, and continued rejection of access, storage and budget
errors.

The native CI's older A/B baseline is not fast.8. Its timing results are
not used to claim a new native-core speedup in this report.

## Reproduce

In a built checkout of the measured runtime:

```sh
python benchmarks/bench_disk_algorithms.py --compare-packed-phase --sizes 100000000 --threads 8 --fft-memory-mib 128 --native-working-mib 384 --repeats 3 --point-factors 1 2 --max-seconds 900 --output packed-phase-100m-pairs.json
```

For the small platform comparison, use `--sizes 1000000 --threads 2
--fft-memory-mib 64 --native-working-mib 128` and retain the three alternating
pairs. These are the CI allowances, rather than the 128/384-MiB profile of
the separate large run. The default harness mode without
`--compare-packed-phase` remains the separate full-versus-packed experiment.
The harness deadline is cooperative between complete calls. For supervised
campaign runs with process deadlines and live budget enforcement, use the
[portable runner](portable-benchmark.md).
