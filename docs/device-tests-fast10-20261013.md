# Physical fast.10 campaign: final evidence, 13 October 2026

The Windows regression suites passed **2,695 tests**. The ARM64 SBC has
**five failing regressions**; the ARM64 instrument has **one**. The instrument
also has **eight memory-limit benchmark failures** across the main and
packed-real reports. The SBC's two large benchmark sections were stopped by
the private supervisor and remain incomplete. These failures are unresolved.

Tested source: [`d91e4cff8bfffc6206bb69785f9a1c81aa1bf639`](https://github.com/MichaDit/lpsd/commit/d91e4cff8bfffc6206bb69785f9a1c81aa1bf639),
native version **1.0.6+fast.10**, statistics **corrected**. This is a separate
cohort from the [older physical reports](device-tests-20261012.md), which retain
their original results and legacy statistics. Corrected and legacy are
different estimators; their spectra cannot be scored as identical estimators.
The [statistics contract](statistics.md), [implementation qualification](fast10-validation.md)
and [rerun guide](device-rerun-fast10.md) describe the code and intended campaign.

## Execution and scope

Each system used the pinned fresh source and fresh reference/native builds,
with imports and source hashes checked. The scientific Python dependencies
were reused in private isolated environments. On Windows, the optional CUDA
bridge was rebuilt before the regression suite. This campaign did not repeat
the optional GPU performance campaign or measure a new EPYC baseline.
Existing [accelerator measurements](accelerators.md) retain their own source,
hardware and estimator scope. A cuFFT periodogram is a different estimator
from LPSD, so its timing is not an LPSD speedup.

| Public device class | Runtime / build | Requested main budget | Main thread request |
|---|---|---|---:|
| windows-x64 | Windows x64, GCC, fresh CUDA bridge | 32 GiB RAM, 16 GiB disk, at most 100M samples | 16 |
| arm64-sbc | Raspberry Pi 4, Debian Linux ARM64, GCC | 0.375 GiB RAM, 2 GiB disk, at most 30M | 2 |
| arm64-instrument | Prepared Debian ARM64 chroot on Android, Zig 0.13.0 | 0.75 GiB RAM, 2 GiB disk, at most 30M | 2 |

The instrument run qualifies the prepared Linux chroot, not a stock Android
or Termux installation. It used cached dependencies with internet access
blocked. Acquisition/web applications stayed outside the test environment;
no device firmware, services or network configuration were changed.

The independent controllers and collector have ended. Final files were
collected again after completion. The instrument's supervisor reported
`finished`, exit code **1**, `released=true`; fresh live inspection confirmed
that its own test processes and rootfs mounts had ended. A terminal controller
is not a successful suite. A benchmark `state=completed` means the runner
processed its matrix, including failed or skipped rows. Reports left at
`state=running` are interrupted snapshots, not completed runs.

Both ARM environments needed a **private library-path workaround** to select
the matching core and OpenMP libraries from the same pyFFTW wheel. Configured
two-thread FFTW selection was verified in their main/smoke reports; this does
not measure thread occupancy. The unmodified public loader initially selected
one thread. The first SBC attempt failed the strict twelve-success smoke
gate because disk cases were not admitted. Its private continuation used an
admission-aware gate, retaining those skips. The instrument's first attempt
was interrupted before regressions to select the matching runtime. Initial
attempts remain private; the public source and tests were unchanged by these
workarounds.

The Windows report's installed distribution metadata still says fast.4;
ARM metadata may omit the installed package version. This reused metadata
does not identify the tested source. The pinned commit, per-case source/native
hashes, verified imports and `native_version=1.0.6+fast.10` do. The original
working checkout and its pre-existing modified generated reference DLL were
not used for publication or overwritten.

## Regression results

Counts were rederived from final JUnit testcases and checked against their
suite totals. The [regression summary](../benchmarks/results/device-tests-fast10-20261013/regressions.json)
retains counts and relative nonpassing test identifiers, not raw XML or messages.

| Device class | Fast suite passed | Failed | Skipped | Xfail | Upstream passed |
|---|---:|---:|---:|---:|---:|
| windows-x64 | 2677 | 0 | 31 | 1 | 18 |
| arm64-sbc | 2501 | 5 | 6 | 1 | 18 |
| arm64-instrument | 2505 | 1 | 6 | 1 | 18 |

No JUnit errors were recorded. Each fast suite reported **18 PytestWarnings**
about `record_property` with the xunit2 JUnit family. The SBC native build
logged an installed-version metadata warning; the instrument native build
logged **three loop-vectorization warnings**. Builds completed, but these
warnings are not suppressed or treated as proof of optimal machine code.
Each suite's one expected failure remains separate from skips and passes.

Failing relative test identifiers:

- **arm64-sbc**: `test_fast/test_disk_algorithm_comparison.py::test_direct_comparison_gates_actual_float32_outputs_at_3ppm[1e-06-True]`
- **arm64-sbc**: `test_fast/test_disk_algorithm_comparison.py::test_direct_comparison_gates_actual_float32_outputs_at_3ppm[1e-05-False]`
- **arm64-sbc**: `test_fast/test_disk_algorithm_comparison.py::test_direct_comparison_rejects_wrong_observed_route[abi]`
- **arm64-sbc**: `test_fast/test_disk_algorithm_comparison.py::test_direct_comparison_rejects_wrong_observed_route[half_n]`
- **arm64-sbc**: `test_fast/test_disk_algorithm_comparison.py::test_direct_comparison_rejects_wrong_observed_route[power_vector]`
- **arm64-instrument**: `test_fast/test_bootstrap_memory_tree.py::test_real_setup_live_child_counter_cannot_disappear_from_sum`

The SBC comparison failures include an unexpected nonzero comparison return,
a missing comparison-pairs key, and three missing expected route assertions.
The instrument's live-child-counter test received no child counter where it
expected one. The causes were not established, and no repair is claimed by
this results publication. Passing benchmark checks do not erase failing tests.

## Benchmark coverage and interruptions

All recorded completed cases passed their retained finite/nonnegative PSD,
PSD-to-NSD, unchanged-input, frequency-plan and resolved-point-count checks.
Those checks describe the recorded outputs, not a universal accuracy guarantee.
Unrecorded, failed and skipped configurations have no completed timing or
output validation.

| Report | Stored state | Recorded / planned | Completed | Failed | Skipped |
|---|---|---:|---:|---:|---:|
| [windows-x64-smoke.json](../benchmarks/results/device-tests-fast10-20261013/windows-x64-smoke.json) | completed | 12/12 | 12 | 0 | 0 |
| [windows-x64-main.json](../benchmarks/results/device-tests-fast10-20261013/windows-x64-main.json) | running | 121/126 | 121 | 0 | 0 |
| [windows-x64-packed-real.json](../benchmarks/results/device-tests-fast10-20261013/windows-x64-packed-real.json) | completed | 16/16 | 16 | 0 | 0 |
| [windows-x64-workers-1.json](../benchmarks/results/device-tests-fast10-20261013/windows-x64-workers-1.json) | completed | 6/6 | 6 | 0 | 0 |
| [windows-x64-workers-2.json](../benchmarks/results/device-tests-fast10-20261013/windows-x64-workers-2.json) | completed | 6/6 | 6 | 0 | 0 |
| [windows-x64-workers-4.json](../benchmarks/results/device-tests-fast10-20261013/windows-x64-workers-4.json) | completed | 6/6 | 6 | 0 | 0 |
| [arm64-sbc-smoke.json](../benchmarks/results/device-tests-fast10-20261013/arm64-sbc-smoke.json) | completed | 12/12 | 6 | 0 | 6 |
| [arm64-sbc-main.json](../benchmarks/results/device-tests-fast10-20261013/arm64-sbc-main.json) | running | 114/120 | 112 | 0 | 2 |
| [arm64-sbc-packed-real.json](../benchmarks/results/device-tests-fast10-20261013/arm64-sbc-packed-real.json) | running | 12/16 | 12 | 0 | 0 |
| [arm64-sbc-workers-1.json](../benchmarks/results/device-tests-fast10-20261013/arm64-sbc-workers-1.json) | completed | 6/6 | 6 | 0 | 0 |
| [arm64-sbc-workers-2.json](../benchmarks/results/device-tests-fast10-20261013/arm64-sbc-workers-2.json) | completed | 6/6 | 6 | 0 | 0 |
| [arm64-sbc-workers-4.json](../benchmarks/results/device-tests-fast10-20261013/arm64-sbc-workers-4.json) | completed | 6/6 | 6 | 0 | 0 |
| [arm64-instrument-smoke.json](../benchmarks/results/device-tests-fast10-20261013/arm64-instrument-smoke.json) | completed | 12/12 | 12 | 0 | 0 |
| [arm64-instrument-main.json](../benchmarks/results/device-tests-fast10-20261013/arm64-instrument-main.json) | completed | 120/120 | 116 | 4 | 0 |
| [arm64-instrument-packed-real.json](../benchmarks/results/device-tests-fast10-20261013/arm64-instrument-packed-real.json) | completed | 16/16 | 12 | 4 | 0 |
| [arm64-instrument-workers-1.json](../benchmarks/results/device-tests-fast10-20261013/arm64-instrument-workers-1.json) | completed | 6/6 | 4 | 0 | 2 |
| [arm64-instrument-workers-2.json](../benchmarks/results/device-tests-fast10-20261013/arm64-instrument-workers-2.json) | completed | 6/6 | 3 | 0 | 3 |
| [arm64-instrument-workers-4.json](../benchmarks/results/device-tests-fast10-20261013/arm64-instrument-workers-4.json) | completed | 6/6 | 4 | 0 | 2 |
| [windows-x64-continuation.json](../benchmarks/results/device-tests-fast10-20261013/windows-x64-continuation.json) | completed | 27/27 | 27 | 0 | 0 |
| [windows-x64-short-continuation-attempt.json](../benchmarks/results/device-tests-fast10-20261013/windows-x64-short-continuation-attempt.json) | completed | 9/9 | 3 | 0 | 6 |

### Windows: initial interruption and separate continuation

The private controller incorrectly reused the **8-GiB regression ceiling**
for the initial main benchmark. It stopped during a 100M matched case at a
sampled summed process-tree RSS of **9,033,617,408 bytes**. The retained report
has 121 completed rows and `state=running`; it does not contain a final result
for the interrupted case. This was a private supervisor stop, not numerical
failure or a demonstrated OS out-of-memory event.

A separate **32-GiB, 24-hour continuation** completed all six 100M method /
point-factor configurations. It retained 24 comparison rows, including
supporting 1M/10M/30M repeats, plus three 100K calibrations. Against the first
report, **19 comparisons repeat existing configurations and five are new**:
the combined coverage is **126 unique comparison configurations**, not 148.
The initial report already completed the first 100M FFTW case. Repeated rows
remain available in their own reports and are not pooled as independent
replicates or silently substituted for earlier times.

The earlier two-hour continuation attempt completed three calibration rows
and skipped all six 100M comparisons for `time_budget`. That attempt remains
published separately. An admission forecast is not a completed measurement.

The successful continuation's six observed complete-call times were:

| Method | Point factor | Actual output points | Compute wall time (s) |
|---|---:|---:|---:|
| FFTW | 1 | 704 | 4.1583 |
| Matched | 1 | 704 | 19.0137 |
| LPSD | 1 | 704 | 35.7052 |
| FFTW | 2 | 1408 | 4.1848 |
| Matched | 2 | 1408 | 30.7594 |
| LPSD | 2 | 1408 | 75.6488 |

These methods implement different estimators. The Windows 1B case was not
repeated because the earlier host crash and bugcheck remain unexplained;
these 100M results do not establish that the 1B failure is resolved.

### ARM64 SBC: supervisor stops and admission skips

The private controller applied its **0.6-GiB regression ceiling** to both
benchmark sections. It stopped the main section at **669,970,432 bytes** and
packed-real at **648,609,792 bytes** of sampled summed process-tree RSS.
Their last reports retain `state=running`: main has 114 of 120 planned rows
(112 completed, two memory-admission skips), and packed-real has 12 of 16
completed rows. **Six main and four packed-real configurations remain
unrecorded**, all at 30M. Even the controller's child exit code zero for these
stages does not make their supervisor-marked stops successful.

The two main admission skips were the 100K pink-noise matched and LPSD cases
at point factor one. The final twelve-case smoke retained six completed RAM
cases and six disk `memory_budget` skips: its effective budget was 374 MiB,
below the 384-MiB disk admission minimum. Skips did not become passes in the
private continuation. No OS OOM or numerical failure is inferred from these
supervisor stops or admission decisions.

### ARM64 instrument: eight retained memory-limit failures

All main and packed-real configurations were recorded. At 30M samples,
FFTW and matched disk cases failed for both point factors in both algorithms.
The case guard stopped them at its **805,306,368-byte (768-MiB) ceiling**.
The metric was **sampled process-tree `rss_fallback`**, using a PPID scan with
a complete final tree sample. It was not a heap-only measurement or proof of
an OS OOM event. The kernel did not provide the task-children proc file.

| Section | Method | Point factor | Limit (bytes) | Trigger (bytes) |
|---|---|---:|---:|---:|
| main | fftw | 1 | 805306368 | 812457984 |
| main | matched | 1 | 805306368 | 818733056 |
| main | fftw | 2 | 805306368 | 808501248 |
| main | matched | 2 | 805306368 | 808390656 |
| packed-real | fftw | 1 | 805306368 | 828518400 |
| packed-real | matched | 1 | 805306368 | 814284800 |
| packed-real | fftw | 2 | 805306368 | 840601600 |
| packed-real | matched | 2 | 805306368 | 826781696 |

The main 30M native LPSD RAM cases completed in observed **752.514 s** and
**1587.011 s** at point factors one and two. They are different execution
routes from the failed disk methods. The runner's completed matrix and the
private stage's zero exit code do not turn those eight failed rows into passes.

## Worker comparison: observations at N=1,000,000

The sequential dedicated worker reports request **1, 2 and 4 threads** with
the same white input, RAM storage, point factor one, 1000 target points and
**577 actual output points** for completed comparison rows. Statistics are
corrected for LPSD/matched and not applicable to FFTW. Each report also
contains three 100K calibration rows (469 output points), kept separate.
FFTW thread diagnostics and native smoothing controls remain in the JSON.

Each cell gives **compute wall seconds / process wall seconds**. Compute is
the complete PSD+NSD API call; process time includes startup, setup and checks.
These are single observed measurements, not medians or universal optima.
Skipped rows contain forecasts but no observed completed-call timing.

| Device class | Method | 1 thread | 2 threads | 4 threads |
|---|---|---:|---:|---:|
| windows-x64 | fftw | 0.0496 / 1.12 | 0.0496 / 1.12 | 0.0515 / 1.24 |
| windows-x64 | matched | 0.4326 / 1.59 | 0.2573 / 1.47 | 0.1702 / 1.36 |
| windows-x64 | lpsd | 0.9245 / 2.05 | 0.4972 / 1.70 | 0.2715 / 1.48 |
| arm64-sbc | fftw | 0.1737 / 4.36 | 0.1775 / 4.35 | 0.1821 / 4.36 |
| arm64-sbc | matched | 3.9216 / 8.04 | 2.8109 / 7.01 | 2.5159 / 6.67 |
| arm64-sbc | lpsd | 10.2416 / 14.32 | 6.0284 / 10.10 | 5.1979 / 9.34 |
| arm64-instrument | fftw | 0.5629 / 16.87 | skip: time budget | skip: time budget |
| arm64-instrument | matched | skip: time budget | skip: time budget | 5.4897 / 26.98 |
| arm64-instrument | lpsd | skip: time budget | skip: time budget | skip: time budget |

Four requested threads gave the lowest observed matched and LPSD complete-call
times among the tested 1/2/4 settings on Windows and the SBC. FFTW showed no
corresponding monotonic benefit at this size. This does not establish the
best setting for larger sizes, different output counts, concurrent jobs or
energy use. Windows's separate main campaign requests 16 threads and is not
an additional setting in this controlled 1/2/4 comparison.

The instrument worker reports retained **seven `time_budget` skips**: two
with one thread, three with two, and two with four. Only the one-thread FFTW
and four-thread matched 1M comparisons completed. Its 100K calibration rows
all completed, but cannot fill the missing 1M comparisons. **No optimum for
the instrument is established.**

## Public export and verification

The [manifest](../benchmarks/results/device-tests-fast10-20261013/manifest.json)
lists all 20 benchmark reports with byte SHA-256, roles, states, counts,
failure/skip categories and partial coverage. Each report was reconstructed
through the closed `lpsd_benchmark.privacy.sanitize` schema. Idempotence and
equality to the already sanitized retained report were checked, proving that
all cases, statuses and admitted numeric diagnostics survived export.

Seven targeted privacy/diagnostic regression checks passed against the identical
pinned, already built campaign source. Their first invocation in the separate
publication checkout stopped at fixture setup because that checkout had no
native fast library; none of those tests ran in that first invocation. The
successful repeat used the existing campaign build without changing source.

Only fixed synthetic input presets, public source/library hashes, numerical
diagnostics, counts and relative test identifiers are exported. Raw logs,
JUnit XML, launch scripts, credentials, private paths, hostnames, network
addresses, device serials, arbitrary exception messages and spectral arrays
remain private. This publication adds measured evidence; it does not change
the tested implementation, fix the failures or claim full ARM qualification.
