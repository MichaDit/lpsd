# Physical fast.11 analysis for the next optimization pass

Date: **21 October 2026**. This analysis uses the newly published
[physical fast.11 campaign](device-tests-fast11-20261021.md), collected from
source [`3d4c9e948d0563cdc0e03339f7f3a44d8609e986`](https://github.com/MichaDit/lpsd/commit/3d4c9e948d0563cdc0e03339f7f3a44d8609e986)
and published by
[`fa7052505b8a72a1c85d95f599cd188742a18421`](https://github.com/MichaDit/lpsd/commit/fa7052505b8a72a1c85d95f599cd188742a18421).
The measurements below are **observations of fast.11**, not measurements of
the subsequent fixes or optimizations. No device was contacted for this
analysis. New qualification takes place locally in the development environment.

The principal conclusions are:

- Native LPSD and the native subset of matched smoothing remain the largest
  arithmetic targets. In the instrument's packed-real 30M matched cases,
  the native subset and output stage consume **75.45% and 81.95%** of call time.
- Increasing native workers from one to four improved the recorded 1M calls
  by **3.452x on Windows** and **2.502x on the instrument**. FFTW's corresponding
  calls became slower. These observations do not justify one universal
  automatic thread setting.
- Packed-real completed the instrument's comparable 30M disk FFTW calls in
  **40.08% and 41.64% less time** than full-complex disk FFT. The existing
  explicit option is useful; this campaign does not identify a universal
  dispatch threshold.
- Two Windows discovery assertions still failed in the published suite.
  Their resolution and the tests of any fix must be reported separately
  from these preserved historical failures.

## Evidence audit and scope

All 13 report byte hashes and the regression-summary byte hash were checked
against the [manifest](../benchmarks/results/device-tests-fast11-20261021/manifest.json).
Every report has the manifest's source revision and completed state, and its
recorded and expected row counts agree. There are **164 completed rows**:
**152 final-attempt rows plus 12 initial-attempt smoke rows**. Across those
reports, **146 rows are comparison rows and 18 are calibration rows**. These
are neither 164 unique workloads nor 164 independent repetitions.

Every retained row has native version `1.0.6+fast.11` and passes all six saved
flags: finite output, nonnegative PSD, PSD/NSD conversion, unchanged input,
frequency-plan equality and resolved output-point count. These checks establish
the recorded contract; they do not make the FFTW, matched and native LPSD
estimators interchangeable. Native and matched use corrected statistics.

The [regression summary](../benchmarks/results/device-tests-fast11-20261021/regressions.json)
records:

| Device | Fast passes | Fast failures | Skips | Expected failures | Upstream passes |
|---|---:|---:|---:|---:|---:|
| Windows x64 | 2,820 | 2 | 45 | 1 | 18 |
| ARM64 instrument | 2,655 | 0 | 16 | 1 | 18 |

There are no recorded JUnit errors. The failed Windows cases are the `core`
and `thread` parameters of
`test_wheel_initialization_retries_only_the_same_failed_library_pair`.
Dependency and platform skips remain skips; module-level skips do not qualify
every individual test in those modules. The expected failure concerns uniform
relative accuracy near the tiny spectral response of an off-bin tone.

This is fresh physical evidence for Windows x64 and the instrument's existing
Debian ARM64 chroot. It is not a stock Android result. The Raspberry Pi was
excluded, and there is no fresh macOS or GPU performance campaign. Historical
SBC failures and the unexplained Windows billion-sample restart remain in
their original reports. Cross-day comparisons with fast.10 cannot isolate
the effects of fast.11 source changes.

## One-, two- and four-worker observations

The dedicated worker reports use white RAM input, **N=1,000,000**, point
factor one, 1,000 requested planner points and **577 actual output points**.
Only `case_role == "comparison"` rows are used here; the 100K calibration
rows are excluded. Times are the complete PSD+NSD call's `compute_wall_s`.
The ratio is one-worker time divided by four-worker time.

| Device / method | 1 worker, s | 2 workers, s | 4 workers, s | 1 / 4 ratio |
|---|---:|---:|---:|---:|
| Windows FFTW | 0.051274 | 0.051460 | 0.053300 | 0.962x |
| Windows matched | 0.444225 | 0.250959 | 0.168810 | 2.632x |
| Windows native LPSD | 0.925203 | 0.487843 | 0.268035 | 3.452x |
| Instrument FFTW | 0.486283 | 0.552650 | 0.580026 | 0.838x |
| Instrument matched | 10.693904 | 7.578212 | 6.176382 | 1.731x |
| Instrument native LPSD | 27.961721 | 18.063576 | 11.175177 | 2.502x |

Sources: `windows-x64-workers-{1,2,4}.json` and
`arm64-instrument-workers-{1,2,4}.json` in the
[campaign directory](../benchmarks/results/device-tests-fast11-20261021).
Input, frequency, PSD, NSD, native-library and FFTW-core hashes agree within
each device/method series. Each setting has one retained comparison call.
There is no alternating repeat series, controlled CPU-frequency experiment
or energy measurement.

The ratios describe the complete recorded configurations. Windows's effective
RAM admission changes with available memory, approximately 19,103–19,448 MiB
in these rows. The instrument remains at 768 MiB; its automatic native scratch
allowance falls from 501.37 to 498.37 to 492.37 MiB as worker reservations
increase. Matched smoothing's worker pool also follows the requested setting.
The separate Windows 16-thread main run and instrument two-thread main run
are not additional points in this dedicated series.

The saved `environment.logical_cpus_available` field contains the admitted
thread setting in this source, as assigned by
[`environment`](https://github.com/MichaDit/lpsd/blob/3d4c9e948d0563cdc0e03339f7f3a44d8609e986/lpsd_benchmark/runner.py).
Its values 1/2/4 are not evidence that physical CPU count or affinity changed.

## Comparable full-complex and packed-real disk calls

The new campaign has **four comparable full/packed pairs**: the instrument's
30M disk cases. Each pair matches input, output count, threads, memory
allowances, native and FFTW binaries, frequency, PSD and NSD hashes. The
combined source-closure hash differs because the packed route includes its
adapter. Both routes preserve the original sample length and frequency grid.

| Method | Point factor / actual points | Full disk, s | Packed-real disk, s | Full / packed | Less call time |
|---|---|---:|---:|---:|---:|
| FFTW | 1 / 678 | 80.934823 | 48.495600 | 1.669x | 40.08% |
| FFTW | 2 / 1,356 | 91.186519 | 53.212563 | 1.714x | 41.64% |
| Matched | 1 / 678 | 289.732854 | 257.410114 | 1.126x | 11.16% |
| Matched | 2 / 1,356 | 517.639283 | 507.836599 | 1.019x | 1.89% |

Sources: [instrument main](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-main.json)
and [instrument packed-real](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-packed-real.json).
`Less call time = 100 * (1 - packed_time / full_time)`. These are individual
observations. The 1.89% difference, especially, is not established as a
repeatable improvement.

All Windows main cases use RAM, whereas its packed-real report uses disk.
The instrument's smaller main cases also use RAM. Comparing those with the
packed-real disk report would mix storage, workspace strategy, diagnostics
and cleanup with the FFT algorithm. They are excluded from this table.

All eight instrument 30M FFTW/matched disk rows completed under the unchanged
805,306,368-byte anonymous-memory allowance. Their guard used `anonymous_rss`
with zero RSS-fallback samples. Total RSS includes mapped file pages and
can exceed 768 MiB. Completion under this policy does not prove a reduction
in algorithmic memory demand or invalidate earlier RSS-fallback stops.

## Phase evidence and optimization priorities

The packed-real reports retain phase timers. Each percentage below is the
saved phase divided by the same row's complete `compute_wall_s`; phases are
not imported from another observation.

| 30M packed-real matched | Point factor | Native subset + outputs | Frequency smoothing | Blocked FFT | Ordered powers |
|---|---:|---:|---:|---:|---:|
| Windows | 1 | 54.99% | 20.53% | 7.51% | 6.26% |
| Windows | 2 | 72.89% | 11.72% | 4.71% | 3.83% |
| Instrument | 1 | 75.45% | 10.25% | 6.50% | 5.15% |
| Instrument | 2 | 81.95% | 4.90% | 5.89% | 5.84% |

Sources: [Windows packed-real](../benchmarks/results/device-tests-fast11-20261021/windows-x64-packed-real.json)
and [instrument packed-real](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-packed-real.json).
The native column is `low_lpsd_and_outputs_s`, which includes output assembly;
it is not a direct measurement of C-kernel time alone. These data prioritize
the native calculation while explaining why an FFT-only gain has a smaller
effect on complete matched calls. They do not authorize replacing native
LPSD by the different FFTW estimator to satisfy a numerical error budget.

Two additional targets are visible:

1. In the Windows packed-real cases, `frequency_energy_diagnostic_s` is
   approximately **46–97 ms**. The
   [fast.11 disk pipeline](https://github.com/MichaDit/lpsd/blob/3d4c9e948d0563cdc0e03339f7f3a44d8609e986/benchmarks/bench_fftw_disk.py)
   calls `np.testing.assert_allclose` on two scalar energies within that
   phase. Removing avoidable testing-library initialization is a candidate
   for reducing complete-call overhead. The Parseval comparison and its
   numerical tolerance must remain enforced; a change needs local evidence
   before attributing any part of the observed phase to imports.
2. The instrument's 30M packed-real FFTW cleanup takes **9.651 and 8.183 s**,
   or **19.90% and 15.38%** of those calls. Investigate mapping teardown and
   temporary-file cleanup separately from FFT arithmetic. Required cleanup
   must remain in complete-call timing. The phase does not identify which
   operating-system or filesystem operation caused the time.

Automatic thread-count or packed-real dispatch changes need evidence across
the relevant workloads. The available observations support keeping explicit
controls and testing focused optimizations locally, without asserting that
one setting is optimal on all platforms.

## Whole-process overhead

For each main report, select completed RAM comparison rows with `N <= 1M`
and compute `process_wall_s - compute_wall_s`. Each selection has 12 rows:
two lengths, two point factors and three methods. These are different
workloads, not timing repetitions.

| Device | Median noncompute time | Sum process / compute time | Noncompute share of sum process |
|---|---:|---:|---:|
| Windows x64 | 0.58985 s | 8.43600 / 1.16473 s | 86.19% |
| ARM64 instrument | 8.81684 s | 191.18935 / 89.50287 s | 53.19% |

The final column is `100 * (1 - sum(compute_wall_s) / sum(process_wall_s))`.
Process time includes imports, input setup, hashing, validation and process
supervision. The remainder cannot be treated as fixed import overhead or
removed from actual elapsed time. A persistent application reusing a prepared
API has a different lifecycle from these isolated benchmark workers. Reducing
unnecessary work is useful; removing resource or correctness checks to lower
this measurement is not an optimization of the estimator.

The historical fast.10 campaign used additional signal types and different
coverage. Its aggregate overhead percentages are not a controlled baseline
for the white-only fast.11 selection above. Likewise, successful current
source qualification on a local Linux host does not replace physical Windows,
ARM or macOS validation of new source changes.
