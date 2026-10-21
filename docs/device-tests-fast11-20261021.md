# Physical fast.11 results — 21 October 2026

The ARM64 instrument's final regression suites passed **2,673 tests**, with
16 skips and one expected failure. Its targeted default/packed-real matrices
and dedicated worker series completed without benchmark failures or skips.
Windows passed **2,838 tests**, but **two FFTW-discovery regression tests
failed**; those failures remain unresolved. Its benchmark matrices completed.
**The Raspberry Pi was excluded and was not accessed or tested.**

Tested source:
[`3d4c9e948d0563cdc0e03339f7f3a44d8609e986`](https://github.com/MichaDit/lpsd/commit/3d4c9e948d0563cdc0e03339f7f3a44d8609e986),
native version **1.0.6+fast.11**, **corrected** statistics. This source contains
the [fast.11 implementation](performance-fast11.md) and the
[staged physical-check guide](device-rerun-fast11.md). The
[fast.10 results](device-tests-fast10-20261013.md) and
[subsequent historical review](platform-followup-20261020.md) retain their
own observations. Their failures and incomplete matrices are not rewritten.

## Runtime, budgets and completion

| Device class | Actual runtime / compiler | Requested baseline allowance | Main thread request |
|---|---|---|---:|
| windows-x64 | Windows x64, GCC, fresh optional CUDA bridge | 32 GiB memory, 16 GiB dataset; at most 100M samples | 16 |
| arm64-instrument | Existing offline Debian ARM64 chroot on Android, Zig 0.13.0 | 0.75 GiB case memory, 2 GiB dataset; at most 30M | 2 |

Each source archive contained 394 tracked files and was hash-checked before
builds. After execution, **188 relevant Python/C/header/config/build files
per host** were again checked against their pinned bytes. Fresh source and
reference/native imports were verified. Scientific Python dependencies were
reused in isolated environments; their actual versions and selected library
hashes remain in each report. Reused installed metadata says fast.4 on Windows
and omits an installed package version on the instrument. The pinned source,
verified imports, native build and per-case fast.11 identity identify this run.

The instrument was tested in the prepared Linux chroot, not stock Android
or Termux. It stayed offline. Device services, firmware, network settings and
the acquisition/web applications were not changed for this campaign. Its
webpage test owner confirmed the test interval. The original working checkout
and its pre-existing modified generated reference DLL were left untouched.

The instrument selected the matching FFTW runtime with **two configured
threads**, without the previous private library-path workaround. Configured
thread counts are not measurements of thread occupancy. Its private controller
and outer watchdog were detached; Windows used a separate background controller.
Both controllers have ended. Fresh live inspection confirmed the instrument's
terminal supervisor, **exit code 0**, `released=true`, no remaining own chroot
processes and no own proc/dev mounts. Windows ended with **exit code 1** because
of its two regression failures, despite completed benchmarks.

Each campaign had a 24-hour bound and the instrument an independent 25-hour
watchdog. Regression supervision used separate 8-GiB/1-GiB RSS ceilings for
Windows/instrument. Windows's large benchmark supervisor used its 32-GiB
allowance. The instrument benchmark's process-tree guard enforced the existing
0.75-GiB allowance using its recorded metric. An additional summed-RSS cutoff
was not imposed on its disk benchmarks: file-backed resident pages must not
replace the intended anonymous-memory check with a different limit.

## Final regression results

Counts were recalculated from the final JUnit testcases and checked against
the XML suite totals. Only the verified counts and relative nonpassing test
identifiers appear in the [regression summary](../benchmarks/results/device-tests-fast11-20261021/regressions.json).

| Device class | Fast passed | Failed | Skipped | Xfail | Upstream passed |
|---|---:|---:|---:|---:|---:|
| windows-x64 | 2820 | 2 | 45 | 1 | 18 |
| arm64-instrument | 2655 | 0 | 16 | 1 | 18 |

There were no JUnit errors. Skips and expected failures are not passes.
Module-level dependency skips do not qualify the individual tests inside
those skipped modules.
The Windows failures are:

- `test_fast/test_fftw_discovery.py::test_wheel_initialization_retries_only_the_same_failed_library_pair[core]`
- `test_fast/test_fftw_discovery.py::test_wheel_initialization_retries_only_the_same_failed_library_pair[thread]`

Both failed the test's assertion about the retried library pair. Their cause
was not established and this publication makes no source fix. Successful
actual FFTW preflights do not erase those failing discovery tests. Both current
parameterized variants of the instrument live-child-counter regression passed.
These are the current source's tests; that observation does not change the
saved failure of the earlier unparameterized fast.10 case.

Each upstream suite reported **three UserWarnings** from the uneven-time-step
and NaN-removal fixtures. The fast suites used legacy JUnit formatting, avoiding
the earlier xunit2 property warnings. The instrument's native build retained
**three loop-vectorization warnings**. The completed build does not establish
optimal vectorized machine code.

## Retained benchmark coverage

All **152 final-attempt rows** completed: 82 on Windows and 70 on the
instrument, including smoke, calibrations and repeated configurations.
A further **12 initial instrument smoke rows** are retained separately.
The resulting **13 reports / 164 rows** include **18 calibration rows**;
these totals are not counts of unique configurations or independent repeats.

All reports retain `state=completed`, with no benchmark failures or skips.
Every completed row passed its recorded finite/nonnegative PSD, exact PSD/NSD
conversion, unchanged-input, frequency-plan and resolved-point-count checks.
These checks apply to the recorded outputs, not a universal accuracy guarantee.

| Report | Attempt | Recorded and completed | Comparison | Calibration |
|---|---|---:|---:|---:|
| [windows-x64-smoke.json](../benchmarks/results/device-tests-fast11-20261021/windows-x64-smoke.json) | final | 12 | 12 | 0 |
| [windows-x64-main.json](../benchmarks/results/device-tests-fast11-20261021/windows-x64-main.json) | final | 36 | 36 | 0 |
| [windows-x64-packed-real.json](../benchmarks/results/device-tests-fast11-20261021/windows-x64-packed-real.json) | final | 16 | 16 | 0 |
| [windows-x64-workers-1.json](../benchmarks/results/device-tests-fast11-20261021/windows-x64-workers-1.json) | final | 6 | 3 | 3 |
| [windows-x64-workers-2.json](../benchmarks/results/device-tests-fast11-20261021/windows-x64-workers-2.json) | final | 6 | 3 | 3 |
| [windows-x64-workers-4.json](../benchmarks/results/device-tests-fast11-20261021/windows-x64-workers-4.json) | final | 6 | 3 | 3 |
| [arm64-instrument-smoke.json](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-smoke.json) | final | 12 | 12 | 0 |
| [arm64-instrument-main.json](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-main.json) | final | 24 | 24 | 0 |
| [arm64-instrument-packed-real.json](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-packed-real.json) | final | 16 | 16 | 0 |
| [arm64-instrument-workers-1.json](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-workers-1.json) | final | 6 | 3 | 3 |
| [arm64-instrument-workers-2.json](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-workers-2.json) | final | 6 | 3 | 3 |
| [arm64-instrument-workers-4.json](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-workers-4.json) | final | 6 | 3 | 3 |
| [arm64-instrument-initial-smoke.json](../benchmarks/results/device-tests-fast11-20261021/arm64-instrument-initial-smoke.json) | initial | 12 | 12 | 0 |

The main matrices use white-noise performance cases: instrument sizes 100K,
1M, 10M and 30M; Windows additionally uses 1,000,003 and 100M. Both point
factors and all three methods are retained. Packed-real uses FFTW/matched
disk cases at 100K/1M/10M/30M. These are the targeted fast.11 continuations,
not repetitions of the entire earlier multi-signal quality matrix. Full
regression suites remain separate. The unexplained historical Windows 1B
restart case was not repeated or declared resolved.

## Instrument: the eight 30M disk cases now completed

The earlier eight FFTW/matched 30M disk cases stopped under fast.10's sampled
`rss_fallback` metric. Here all eight completed at the unchanged
**805,306,368-byte (768-MiB) case allowance**, with the old-kernel guard reading
accessible anonymous counters and a complete process tree via a PPID scan.
These eight rows used **`anonymous_rss` with zero RSS-fallback samples**.

| Section | Method | Point factor | Actual points | Complete-call wall (s) | Peak guard anonymous bytes | Peak RSS bytes |
|---|---|---:|---:|---:|---:|---:|
| main | fftw | 1 | 678 | 80.9348 | 122269696 | 1001287680 |
| main | matched | 1 | 678 | 289.7329 | 228257792 | 1005240320 |
| main | fftw | 2 | 1356 | 91.1865 | 122601472 | 998584320 |
| main | matched | 2 | 1356 | 517.6393 | 238481408 | 1006075904 |
| packed-real | fftw | 1 | 678 | 48.4956 | 99639296 | 978509824 |
| packed-real | matched | 1 | 678 | 257.4101 | 224804864 | 983117824 |
| packed-real | fftw | 2 | 1356 | 53.2126 | 99926016 | 978747392 |
| packed-real | matched | 2 | 1356 | 507.8366 | 235466752 | 983601152 |

The guard's anonymous-byte values and total RSS measure different quantities.
The separate RSS peaks remain above the 768-MiB number because they also
include resident mapped file pages. These results demonstrate completed cases
under the recorded anonymous-memory policy, **not** that total RSS fell below
768 MiB or that the algorithm consumed less memory than the historical run.
The saved fast.10 RSS-fallback stops remain valid observations of that metric.

Packed-real FFTW calls were shorter than the corresponding full calls in these
single observations; matched gains were smaller and varied by point factor.
Both routes retain the same planned output count, but floating-point route
differences are covered by their own checks. These are observed timings, not
median speedups or proof of an optimum. The native LPSD RAM cases at 30M also
completed, on a different route from these disk methods.

## Worker observations at N=1,000,000

All nine dedicated comparison cases on each host completed. They use the same
white RAM input, point factor one, 1000 target points and **577 actual output
points**, with corrected statistics for native/matched. Within each method
and host, input, frequency, PSD, NSD and native hashes were checked for equality
across the 1/2/4 settings. The three 100K/469-output calibration rows in each
report remain separate. FFTW is a different estimator from native LPSD.

Each cell is **complete PSD+NSD call wall seconds / whole-process seconds**:

| Device class | Method | 1 thread | 2 threads | 4 threads |
|---|---|---:|---:|---:|
| windows-x64 | fftw | 0.0513 / 0.625 | 0.0515 / 0.610 | 0.0533 / 0.625 |
| windows-x64 | matched | 0.4442 / 1.109 | 0.2510 / 0.860 | 0.1688 / 0.735 |
| windows-x64 | lpsd | 0.9252 / 1.485 | 0.4878 / 1.109 | 0.2680 / 0.875 |
| arm64-instrument | fftw | 0.4863 / 7.413 | 0.5527 / 9.910 | 0.5800 / 9.383 |
| arm64-instrument | matched | 10.6939 / 17.764 | 7.5782 / 17.062 | 6.1764 / 15.691 |
| arm64-instrument | lpsd | 27.9617 / 34.425 | 18.0636 / 25.019 | 11.1752 / 18.269 |

Among these three settings, four threads gave the shortest observed native
LPSD and matched calls on both hosts; one thread gave the shortest FFTW calls.
The instrument's historical seven forecast skips were not repeated in this
series. This establishes completed observations, not a device-wide optimum,
energy efficiency or an improvement for every workload. The two-thread main
baseline and the separate Windows 16-thread main run are not additional
points in this controlled 1/2/4 series.

Process time includes imports, setup, input creation and checks. It cannot
be interpreted entirely as fixed import overhead. No paired physical startup
experiment or new GPU/EPYC comparison was performed here. The earlier Linux
x86-64 medians in the [fast.11 local report](performance-fast11.md) remain a
different experiment. Cross-day comparisons with saved fast.10 observations
are not controlled A/B measurements and do not isolate the cause of a change.

## Windows 100M continuation

All six targeted 100M configurations completed in the main report:

| Method | Point factor | Actual points | Complete-call wall (s) |
|---|---:|---:|---:|
| fftw | 1 | 704 | 4.6088 |
| matched | 1 | 704 | 20.6311 |
| lpsd | 1 | 704 | 38.9882 |
| fftw | 2 | 1408 | 4.6408 |
| matched | 2 | 1408 | 31.4009 |
| lpsd | 2 | 1408 | 80.3009 |

Different estimators and output counts are separate workloads. These are
single measurements with the recorded resources and libraries.

## Interrupted first attempt, collection recovery and public export

The first instrument attempt was deliberately stopped during `pytest-fast`
to correct the private supervisor's additional RSS policy before the large
disk cases. Its successful 12-row smoke report is published separately; its
partial regression has no final JUnit result and is not counted as a passing
suite. Raw status/logs remain private. The corrected attempt reused the fresh,
verified native builds and then ran both complete regression suites. No tested
public source or numerical assertion was changed for this restart.

The passive collector originally called an unavailable BusyBox `stat` applet.
It could retain terminal controller status before failing collection, leaving
a stale `unknown` state and unreleased local lease. The own collector was
stopped, corrected locally to the existing native system `stat` and restarted
without touching tests or device configuration. It is now finished. Final
files were also collected independently using base64 transport and checked
against device SHA-256. Build logs were recovered from the retained first
attempt because the final attempt reused those builds. Release was freshly
checked again after collection; stale sampler states were not used as proof.

The [manifest](../benchmarks/results/device-tests-fast11-20261021/manifest.json)
lists every report, byte SHA-256, attempt, role and count. Export used only
`lpsd_benchmark.privacy.sanitize`; equality to the retained already sanitized
report and idempotence were checked. Every case, status and admitted numeric
diagnostic is preserved. JUnit was reduced to verified counts and relative
identifiers. Raw XML/logs, launch scripts, credentials, private paths, hostnames,
network identifiers, device serials, arbitrary exception messages and spectra
remain private. Historical publications are preserved unchanged.
