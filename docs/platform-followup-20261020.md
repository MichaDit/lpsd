# Physical campaign review and runtime forecasting, 20 October 2026

The retained fast.10 device results identify substantial process overhead in
small isolated cases and different useful thread settings for FFTW and native
LPSD. The forecast was scaling the entire process time, including startup,
with `N^1.35`. On the ARM64 instrument this rejected seven one-million-sample
worker comparisons despite their 30-minute campaign allowance.

The forecast now scales all noncompute time linearly with input length while
keeping the existing `N^1.35` scaling for the complete numerical call. It
retains the factor-four safety allowance, point-count and storage multipliers,
hard deadlines and memory limits. This is an admission correction, not a
measured spectral speedup. A forecast cannot establish that a previously
skipped case has now completed.

## Source and coverage

The current review starts from
[`8d5fab5f2ec027e1f59be5ce96725dc9108ad41f`](https://github.com/MichaDit/lpsd/commit/8d5fab5f2ec027e1f59be5ce96725dc9108ad41f).
The latest retained physical campaign tests
[`d91e4cff8bfffc6206bb69785f9a1c81aa1bf639`](https://github.com/MichaDit/lpsd/commit/d91e4cff8bfffc6206bb69785f9a1c81aa1bf639),
native version **1.0.6+fast.10**, corrected statistics. Its reports were
published in `d8f5fc0750b7139ff220ec2e45d876da49b67061`.

At this review, all nine remote branches were inspected. None contains newer
physical measurements, the repository has no GitHub Releases or Issues, and
the three PRs contain historical fast.8/fast.9 work. The October 15
[maintenance report](maintenance-20261015.md) records targeted local checks,
not a second physical campaign. This review therefore uses the existing
[20 sanitized physical reports](../benchmarks/results/device-tests-fast10-20261013/manifest.json),
without inventing new device observations.

All 20 byte SHA-256 digests, source commits, row counts, status counts and
stored report states were independently checked against that manifest.
There are **525 retained rows: 496 completed, 21 skipped and 8 failed**.
These totals include calibrations and repeated configurations; they are not
525 unique comparisons or 496 independent performance repetitions.

| Device class | Retained fast regression result | Upstream tests | Physical benchmark coverage |
|---|---|---:|---|
| Windows x64 | 2677 passed, 0 failed, 31 skipped, 1 expected failure | 18 passed | Initial main report stopped at 121/126; the separate continuation brings coverage to 126 unique main configurations. All 16 packed-real rows completed. |
| ARM64 SBC, Raspberry Pi 4 | 2501 passed, 5 failed, 6 skipped, 1 expected failure | 18 passed | Main retains 112 completed and 2 skipped rows out of 120 planned. Packed-real retains 12/16 completed rows. The missing 30M rows are interrupted, not passed. |
| ARM64 instrument, prepared Debian chroot on Android | 2505 passed, 1 failed, 6 skipped, 1 expected failure | 18 passed | Main retains 116 completed and 4 failed rows; packed-real retains 12 completed and 4 failed rows. The eight failures are 30M FFTW/matched memory-guard stops. |

Sources: [regression summary](../benchmarks/results/device-tests-fast10-20261013/regressions.json)
and the [original physical report](device-tests-fast10-20261013.md). The
instrument evidence applies to that prepared Linux environment, not a stock
Android or Termux installation. No new physical macOS or Linux x64 campaign
is included.

The five SBC regression failures and the instrument's regression failure
were subsequently addressed by the October 15 test/admission and optional
`NSpid` fixes. The old-kernel `smaps` fallback and FFTW wheel dependency
recovery were also added then. The saved failures above are not rewritten as
passes of that later source. Final qualification of those changes requires
the corresponding actual device runs.

## Process overhead in small cases

For completed RAM comparison rows at `N <= 1,000,000` in each main report,
subtract the complete-call `compute_wall_s` from `process_wall_s`:

| Device | Completed rows | Median noncompute process time | Total process / compute time | Noncompute share of total process time |
|---|---:|---:|---:|---:|
| Windows x64 | 72 | 1.1400 s | 91.03 / 7.01 s | 92.30% |
| ARM64 SBC | 70 | 4.2157 s | 491.79 / 180.75 s | 63.25% |
| ARM64 instrument | 72 | 19.6015 s | 2002.38 / 555.13 s | 72.28% |

Sources:
[Windows main](../benchmarks/results/device-tests-fast10-20261013/windows-x64-main.json),
[SBC main](../benchmarks/results/device-tests-fast10-20261013/arm64-sbc-main.json),
[instrument main](../benchmarks/results/device-tests-fast10-20261013/arm64-instrument-main.json).
These are aggregate descriptions of different signal, method and point-count
cases, not repeated timings of one workload. Noncompute time includes process
startup, imports, input setup, validation and supervision; the retained data
do not separate a fixed import cost from every size-dependent operation.

The reviewed fast.10 worker imports the native API and benchmark helpers in a
fresh process for each case. Its dependency chain includes pandas and SciPy
via the frozen upstream helpers. The observations justify profiling that startup
path. They do not justify subtracting the entire measured overhead as a fixed
constant, changing the frozen reference, or removing isolated-case memory
and timeout enforcement. Reusing a prepared API in a persistent application
also has a different timing scope from this supervised campaign.

## Useful thread settings from the completed comparisons

The dedicated worker reports use white RAM input, `N=1,000,000`, point
factor one and **577 actual output points**. Native/matched statistics are
corrected. Each number below is one observed complete PSD+NSD call, not a
median. The one-, two- and four-thread reports have identical input,
frequency, PSD and NSD hashes for each method on each device.

| Device and method | 1 thread, s | 2 threads, s | 4 threads, s | Observed ratio, 1 thread / 4 threads |
|---|---:|---:|---:|---:|
| Windows FFTW | 0.04958 | 0.04959 | 0.05152 | 0.962x |
| Windows matched | 0.43265 | 0.25728 | 0.17022 | 2.542x |
| Windows LPSD | 0.92452 | 0.49722 | 0.27146 | 3.406x |
| SBC FFTW | 0.17373 | 0.17753 | 0.18207 | 0.954x |
| SBC matched | 3.92164 | 2.81092 | 2.51589 | 1.559x |
| SBC LPSD | 10.24160 | 6.02843 | 5.19787 | 1.970x |

Sources: `windows-x64-workers-{1,2,4}.json` and
`arm64-sbc-workers-{1,2,4}.json` in the
[physical report directory](../benchmarks/results/device-tests-fast10-20261013).
The native and FFTW core digests also match across each series. A one-thread
FFTW report does not need the threaded companion's digest.

For these tested one-million-sample workloads, **one FFTW thread and four
native/matched workers are useful explicit settings on Windows and the SBC**.
This does not establish the best setting for other lengths, energy use or
concurrent applications. The separate Windows main campaign requested 16
threads and is not a fourth point in this dedicated comparison. The
instrument's seven missing worker cases prevent an equivalent recommendation
for that device.

## Full versus packed-real on the SBC

Only the SBC has completed same-size disk observations of both algorithms in
this campaign at 10M samples. The four pairs share the source revision,
input, native binary, FFTW core/thread binary, selected threads, memory
allowances, output counts and final frequency/PSD/NSD hashes.

| Method | Point factor | Full disk FFT, s | Packed-real disk FFT, s | Full / packed |
|---|---:|---:|---:|---:|
| FFTW | 1 | 24.59825 | 7.33301 | 3.355x |
| FFTW | 2 | 24.62265 | 8.02716 | 3.067x |
| Matched | 1 | 54.79231 | 38.12011 | 1.437x |
| Matched | 2 | 84.17581 | 63.90023 | 1.317x |

Sources:
[SBC main](../benchmarks/results/device-tests-fast10-20261013/arm64-sbc-main.json)
and [SBC packed-real](../benchmarks/results/device-tests-fast10-20261013/arm64-sbc-packed-real.json).
The combined `source_sha256` is intentionally different: the packed route
includes `_fftw_packed_real.py` in its source closure in
[`worker.run`](../lpsd_benchmark/worker.py).

These are four historical single pairs, with no alternating repeat series.
They support explicitly selecting `--disk-fft-algorithm packed-real` for
comparable SBC disk workloads. They do not establish a universal automatic
dispatch threshold. The Windows comparison would mix RAM with disk, while
the instrument's corresponding 30M disk rows failed in both algorithms;
neither supplies a completed equivalent disk pair here.

The SBC phase observations show that packed-real reduced the expensive
blocked FFT stage at 10M. In the packed matched cases the native low-frequency
stage remains approximately 65% and 84% of complete-call time for point
factors one and two. Further FFT-only improvement therefore has a smaller
effect on the complete matched estimator. These percentages are observations
of those cases, not universal workload fractions.

## Forecast correction

[`resources.predict_seconds`](../lpsd_benchmark/resources.py) previously used
the whole process time in the superlinear sample-count estimate. The revised
calibrated estimate for a larger target uses:

```text
P = max(1 second, observed process time)
C = observed complete-call compute time
r = target sample count / observed sample count, with r > 1

estimate = ((P - C) * r + C * r**1.35)
           * existing point-count multiplier
           * existing RAM-to-disk multiplier

forecast = max(15 seconds, 4 * minimum matching estimate)
```

Every noncompute second still grows linearly with input size. This retains
input generation, hashing, output checks and startup in the whole-process
forecast. The point multiplier remains `max(1, point_ratio**1.20)`, including
the existing preference for explicit point factors. The RAM-to-disk factor
remains eight. Same-size and smaller targets preserve the previous arithmetic
exactly, including its rule against scaling costs down.

The split is used only when both times are finite real numbers, process time
is positive, and `0 <= compute <= process`. Missing or inconsistent compute
telemetry retains the previous complete-process `N^1.35` estimate. A missing,
nonfinite or nonpositive process time cannot serve as calibration: a large
case remains uncalibrated if no usable observation is available. The existing
small-case bounded opportunity, estimator matching, algorithm matching,
working-memory matching and all runtime guards remain in force.

### Retained instrument skips

The dedicated instrument reports have a 1800-second limit but ended after
82.28, 60.67 and 86.43 seconds for one, two and four requested threads. Seven
comparison rows were skipped for `time_budget`. Replaying only each report's
three completed 100K calibrations gives:

| Threads | Skipped method | Saved forecast, s | Revised forecast, s |
|---:|---|---:|---:|
| 1 | Matched | 2041.88 | 981.59 |
| 1 | LPSD | 2030.89 | 1073.11 |
| 2 | FFTW | 1762.02 | 797.26 |
| 2 | Matched | 1744.44 | 829.48 |
| 2 | LPSD | 1863.01 | 928.34 |
| 4 | FFTW | 1792.35 | 809.68 |
| 4 | LPSD | 1753.29 | 852.21 |

Sources: `arm64-instrument-workers-{1,2,4}.json` in the
[physical report directory](../benchmarks/results/device-tests-fast10-20261013).
Each revised forecast is below the initial remaining campaign allowance
after calibration. The previously skipped rows still have no observed
compute/process time. Completion and the remaining allowance for later rows
must be measured in a new run; the replay does not fill their missing results.

### Replay and local contracts

The new [forecast regression suite](../test_fast/test_forecast_startup.py)
replays all 20 reports in their original row order without pooling devices
or reports. It forecasts each of the **394 completed cases with a preceding
matching calibration**. Every forecast exceeds the retained observed process
time; the smallest ratio is **2.4556x**. This is evidence on that fixed corpus,
not a mathematical upper bound for arbitrary devices or inputs.

The suite also checks the seven instrument skip forecasts, missing/malformed
telemetry, exact preservation for equal/smaller sizes, the compute-only case,
and the existing point, storage and safety multipliers. Run locally with:

```sh
python -m pytest -q test_fast/test_forecast_startup.py
```

The 29 focused tests passed during this change. Broader native validation and
any independently measured implementation speedups are recorded separately;
they must retain their own source and execution scope. No GitHub Actions were
run for this work.

## Remaining physical checks

The [fast.11 device guide](device-rerun-fast11.md) provides pinned staged
commands using the existing budgets. The
[fast.11 performance and local validation report](performance-fast11.md)
records the new implementation's separate evidence. Retain the current
source/native digests. The instrument needs the one-/two-/four-worker matrix
that was previously rejected by forecasts, followed by the affected 30M
memory cases with actual guard metrics. Both ARM installations need their
FFTW thread selection and earlier regression failures checked against the
maintenance changes. The SBC's interrupted 30M rows require a completed
continuation using the benchmark's intended outer supervision ceiling.

The historical Windows 1B restart/bugcheck remains unresolved and is not
reclassified by the successful 100M continuation. Packed-real and GPU results
retain their existing estimator, hardware and input-size scope. The evidence
supports targeted choices and fixes; it does not show that every platform is
fully optimized or newly qualified.
