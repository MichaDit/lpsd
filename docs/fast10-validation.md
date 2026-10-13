# fast.10: corrected statistics and portable diagnostics

Implementation source:
[`e0426c4bad9d7c5fccd54de935eea2d1ea1b2519`](https://github.com/MichaDit/lpsd/commit/e0426c4bad9d7c5fccd54de935eea2d1ea1b2519),
package version **1.0.6+fast.10**. Its parent is the published fast.9 source
`3a3aabce5a160d649770c01f2f84a31b7e448e3e`.

The [machine-readable local evidence](fast10-local-evidence.json) includes
184 implementation/test/configuration file hashes, the native build report,
every recorded revision-comparison pair, all 36 local portable cases and
the regression summary. Measurements were made in the modified checkout
before the source commit was created. The committed tree was checked against
those bytes; a subsequent documentation commit adds this report and the
pinned [device rerun instructions](device-rerun-fast10.md).

The later [physical fast.10 campaign](device-tests-fast10-20261013.md) tested
source `d91e4cff8bfffc6206bb69785f9a1c81aa1bf639` with fresh native builds.
It retains the final Windows results, five SBC and one instrument regression
failures, memory stops, incomplete coverage and dedicated worker comparisons.
Those device results are separate from the implementation evidence below.

## Changes

| Area | Resulting behavior |
|---|---|
| Segment mean | `lpsd_fast` defaults to `statistics="corrected"`, including every actual segment. Explicit `legacy` retains historical reproduction. |
| Variance and SEM | Welford's M2 accumulates all contributions. Complex CSD uses real squared complex distances. A single segment has a valid spectrum and `NaN` SEM. |
| Prepared and bounded paths | Prepared, rolling Boxcar, streaming, disk and the matched estimator's native low frequencies receive the same statistics selection. |
| Selected outputs | PSD/NSD-only calls skip second-moment updates. The corrected mean is still used. |
| Historical comparisons | Frozen references, CUDA experiments and revision comparisons request legacy explicitly. Calibration and comparison groups distinguish the statistics modes. |
| Failed result evidence | All five validity checks, output hashes and bounded diagnostic counts/maxima survive `invalid_result`. |
| FFTW threading | Discovery also tries the system OpenMP runtime, keeps automatic wheel companions with their core, and records selection/fallback causes. |
| Memory diagnosis | Reports retain the actual monitored metric, sample peak and abort trigger independently of the RSS high-water mark. Stale sample-coverage and RSS-source labels are corrected. |
| Bootstrap monitoring | Incomplete live process-tree readings are rejected. Linux uses a linear snapshot/traversal; the pre-Python probe caches a verified target PID and tolerates unrelated process exits. Windows verifies snapshot/read failures and confirmed exits. |
| Packaging and CI | Version is fast.10; the wheel command uses its setuptools implementation. CI tests corrected installed-wheel output and all 12 portable method/point/storage cases per profile. |

The original `lpsd/` sources and all pre-existing native C entry points remain
the frozen legacy reference. New corrected C entry points are additive.
Use `from lpsd_fast import lpsd, lcsd` for the corrected default.

### Statistical meaning

This is an estimator correction. Corrected and legacy spectra can differ
substantially, especially with few segments; that difference is not a
floating-point optimization tolerance. An exact two-segment fixture now
returns **PSD 5 and SEM 3**, while explicit legacy retains **PSD 8 and SEM 0**.

The deviation columns have a **nominal independent-segment SEM** meaning.
They do not correct for overlap, colored-noise correlation or nonstationarity
and are not general measurement confidence intervals. Complex CSD's SEM is
a real RMS complex error, not a complex pseudovariance square root.
See the [mathematical contract and examples](statistics.md).

## Completed local qualification

Environment: Linux x86-64, Python 3.12.14, NumPy 2.3.5, GCC 13.3.0-compatible
`cc`; portable target-clone build, `-fno-fast-math`, `-ffp-contract=off`.

| Check | Result |
|---|---|
| Entire upstream and fast suite | **2,524 passed, 0 failed, 0 errors, 6 skipped, 1 expected failure**, 207.14 seconds. |
| Independent corrected-statistics cases | **103 passed**, included in the full count. |
| Portable corrected mode | **12/12 completed**, all methods, RAM/disk and 469/938 actual points at N=100,000. |
| Portable legacy mode | **12/12 completed**, same coverage, explicitly labelled legacy. |
| Portable corrected bounded-window profile | **12/12 completed**, requested eight smoothing workers capped to the two admitted threads. RAM uses the one-shot bounded window; disk uses its established blocked window. |
| Output/diagnostic checks | All 36 portable cases pass the five invariants and retain valid memory evidence. FFTW/matched select the system pthread runtime with two configured threads here. |
| Legacy revision comparisons | 24 initial pairs plus 14 targeted follow-up pairs; **all 38 pairs have identical output fingerprints**. |
| Source preservation | Original upstream source/hash gates pass; code tree matches the published implementation tree. |

The independent statistics tests form segment periodograms directly and use
a separate two-pass variance reference. Coverage includes exact K=2/K=3
examples, K=1, complex CSD, channel conjugation/scaling, orders none/0/1,
multiple windows, scalar/auto/fast modes, selected/full outputs, multiple
workers, long batches, Prepared, disk and streaming execution.

The six local skips are the unavailable CUDA modules, native macOS/Windows
runtime checks and an FP64-only residual test in this wider-long-double
build. These do not represent completed tests on those platforms. Native
platform gates exercise their relevant counterparts.

### Remaining strict relative-null expectation

The one expected failure documents the historical uniform `rtol=2e-6,
atol=0` comparison for an off-bin pure tone. At extremely small spectral
values, relative error alone is not a useful uniform guarantee. The mandatory
absolute ASD forward-roundoff bounds remain unchanged and pass.

The current legacy-mode fixture gives:

| Kernel | Maximum relative PSD difference | Maximum absolute PSD difference | Points above 1% |
|---|---:|---:|---:|
| scalar | 0 | 0 | 0 |
| auto | 0.0058854% | 3.2817e-28 | 0 |
| fast | 0.0067726% | 5.0487e-28 | 0 |

The scalar legacy comparison is now an explicit bitwise regression test.
No general arbitrary-input 1% guarantee is inferred from this fixture, and
no automatic second computation was added for every frequency.

## Performance assessment

The local comparison uses the exact published fast.9 baseline, the same
current harness and dependencies, one small warmup, freshly started processes
and alternating baseline/current order. Both sides request legacy statistics;
this isolates implementation changes while retaining identical outputs.
The complete API call is timed; imports, input generation and the small
warmup are outside that timer. No timing threshold is used as a CI pass gate.

At N=1,000,000, the initial three-pair observations were:

| Kernel and outputs | Workers | fast.9 median, s | fast.10 legacy median, s | Wall-time change |
|---|---:|---:|---:|---:|
| auto, all | 1 | 1.51652 | 1.55649 | +2.64% |
| auto, all | 4 | 0.45990 | 0.43745 | -4.88% |
| fast, PSD + NSD | 1 | 1.25066 | 1.13893 | -8.93% |
| fast, PSD + NSD | 4 | 0.31731 | 0.29681 | -6.46% |

Two short N=100,000 cases initially showed +32.36% and +7.97% wall time.
They were rechecked with **seven further alternating pairs each**, unchanged
binaries and the original thread environment. No trials were discarded:

| Configuration | fast.9 median and range, ms | fast.10 median and range, ms | Median change |
|---|---:|---:|---:|
| auto/all, 4 workers | 124.570 (98.604–141.134) | 116.570 (106.246–138.433) | -6.42% |
| fast/PSD+NSD, 1 worker | 145.151 (108.016–182.267) | 117.761 (110.864–125.831) | -18.87% |

The suspected small-case slowdowns were **not reproduced**. The broad,
overlapping ranges on this shared executor do not support a universal speedup
claim either. The full first series and follow-up are retained together in
the evidence JSON. These are legacy regression checks, not physical-device
results or speed comparisons between the corrected and legacy estimators.

Concrete reductions in avoidable work are the linear bootstrap tree scan,
the verified target-PID cache and omitted variance work for selected outputs.
The expanded FFTW runtime search can enable an available multithreaded backend;
the physical ARM devices must establish whether that happens on their actual
installations and whether it improves their wall time. Existing packed-real
and direct-log options remain explicit, with their previous admission rules.

## Platform publication gates

The publication workflows are
[Native tests and wheel](../.github/workflows/tests.yml) and
[Portable benchmark bootstrap](../.github/workflows/portable-benchmark.yml).
Their [GitHub Actions history](https://github.com/MichaDit/lpsd/actions)
retains exact run/commit attribution and artifacts. The release is qualified
against the implementation and test hashes above; a documentation follow-up
does not introduce a different native implementation.

| Gate | Coverage |
|---|---|
| Native numerical matrix, 5 jobs | Linux x64 GCC/Python 3.10 and 3.12, Linux x64 Clang, Linux ARM64 GCC, macOS ARM64 Clang; full fast suite and 18 original tests. |
| Installed-wheel matrix, 3 jobs | Linux x64, Linux ARM64, macOS ARM64; builds and installs outside the checkout, checks legacy compatibility and the exact corrected PSD 5 / SEM 3 fixture. |
| Private bootstrap matrix, 4 jobs | Linux x64, Linux ARM64, macOS ARM64 and Windows x64; normal and bounded profiles each require 12 complete RAM/disk method/point cases and the new diagnostics. Existing packed/direct disk checks remain. |
| Additional Windows native checks | At least 103 statistics and 15 diagnostics cases must execute without skips; exact existing 24 frozen-estimator, 1 Bluestein and 28 CSD-fallback gates remain. |

Native revision comparisons now use the qualified fast.9 baseline on each
participating Linux x64, Linux ARM64 and macOS ARM64 runner. Library hashes,
compiler details and all pairs are retained in workflow artifacts. Different
runner CPUs are not pooled into a cross-platform speedup number.

## What still requires device evidence

The code and tests address the identified implementation defects. A new
physical campaign is still needed to confirm the instrument's 30M memory
behavior and ARM FFTW thread selection. The previous Windows billion-sample
case has no completed result, and its host restart has no confirmed cause.
No user-space fix or cloud CI result establishes that the kernel-level
restart is resolved. The [staged rerun guide](device-rerun-fast10.md) preserves
the existing budgets and keeps that case separate.

Experimental GPU adapters remain explicitly legacy and are not qualified
corrected-statistics backends. SEM correlation adjustment and an arbitrary
signal's uniform relative-null accuracy are not supplied by this release.
These limitations are explicit contracts rather than hidden numerical passes.
