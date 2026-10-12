# fast.9: platform follow-up, direct power reduction and recovered evidence

Version **1.0.6+fast.9** combines the packed-real phase optimization and
resource-guard repairs with the recovered direct logarithmic power reducer
and corrections prompted by the subsequently uploaded physical-device tests.
The fast.9 source qualification below is anchored to
[`1c345822f015f3df93d5acb60ace2170f99d0547`](https://github.com/MichaDit/lpsd/commit/1c345822f015f3df93d5acb60ace2170f99d0547).
The runtime integration `83da4d035258e718c4f0c8744b4c279a68092221` combines
both fast.9 branches and the published device reports, preserving all three
parent histories. The later `c417b9a` and `1c34582` commits change only test
fixtures: complete child readiness and a self-contained CSD source copy for
Windows checkouts whose temporary directory is on another drive.

The later [Linux memory-guard follow-up](linux-memory-guard.md) fixes a
separate process-tree accounting gap found while recovering the published
work. It retains these numerical sources and historical performance data.
Use the [portable guide](portable-benchmark.md#start-from-a-pinned-revision)
for the current qualified source pin and its separate validation.

The performance experiments retain their actual earlier source identities:
the phase change was measured at `f4082f04`, and the direct 100M comparison
at `ca2d26f9`. The later native CSD and process-supervision fixes are qualified
separately below. None of the historical measurements is relabelled as a
fresh measurement of the final commit.

## What improved

Two separately measured changes target different disk FFT routes. Their
percentages must not be added or their absolute timings pooled.

| Change and measured scope | Result | Decision |
|---|---|---|
| Packed-real phase construction, N=1,000,000, simple FFTW, three alternating pairs for each output count on each of four CI platforms | Wall-median reductions of 8.09–8.28% on Linux x86-64, 5.04–5.46% on Linux ARM64, 6.66–11.96% on macOS ARM64 and 8.42–9.94% on Windows x64 | Retain the bounded vectorized construction inside the explicit packed-real route |
| Same phase change, matched estimator, same small platform experiment | From 1.64% slower to 3.10% less wall time; several ranges overlap | Report mixed complete-call effects; no general matched speedup claim |
| Direct logarithmic power reduction, full FFT, N=100,000,000, Linux x86-64, three alternating pairs at 704 and 1,408 actual points | 6.89% and 10.08% less complete-call wall time; all six wall and CPU pairs faster | Integrate as the explicit `--disk-power-reduction direct-log` option |

The phase experiment compares the previous phase-construction method with
the new method while keeping the packed-real algorithm, tile geometry,
input, libraries, buffers and output grid fixed. All **41,544 PSD and 41,544
NSD comparisons** in its four-platform data are bit-identical. The separate
Linux 100M phase experiment also retains exact outputs, but its timing ranges
overlap and unchanged FFT phases explain much of the largest apparent gains.
It does not establish a stable general 10–20% improvement at that length.
The [phase report](performance-packed-phase.md) retains all medians, ranges,
paired variation and source hashes.

The new direct reducer avoids materializing and ordering the N-dependent
power vector. It retains the original-N full FFT and computes logarithmic
groups plus an independent complete nonnegative-bin energy check. All
**6,336 PSD and 6,336 NSD comparisons** in the integrated 100M run are
bit-identical, with zero recorded relative Parseval error. The acceptance
gate remains 3 ppm with zero absolute tolerance. The combined input and
complex workspace still occupy 24N bytes. See the
[direct profile and complete-call measurements](direct-log-power.md).

These observations cover the named lengths, routes, dependencies and hosts.
The 100M input/workspace fit inside this Linux environment's 8 GiB cgroup;
there is no matching direct-reducer 100M timing on Windows/macOS/ARM or a
completed integrated direct 1B pair. Automatic FFT selection, precision and
matched-worker defaults remain as previously qualified.

## Corrected failures and supervision

### Windows private installation

The fast.8 post-merge bootstrap could fail while `uv` renamed or removed
an installation path during the storage-budget scan. Both the scanner and
its PowerShell regression checks now distinguish a disappearing path from
access, storage or budget failures. Already counted bytes are retained, and
the settled installation is scanned again. Both terminating and
nonterminating path-loss errors are covered, including native Windows
error codes 2 and 3. The fix is exercised by actual Windows private-bootstrap
CI, as recorded in the [phase/platform report](performance-packed-phase.md).

The physical Windows suite also exposed an archive-name test failure. ZIP
validation now examines `ZipInfo.orig_filename`, before Python can normalize
Windows backslashes or truncate a name at NUL. The fixtures write those
actual invalid bytes into the archive and verify the original name on readback.
The existing archive policy then rejects the member before extraction.
This is separate from the external campaign controller's status-file
replacement collision and from the later host crash.

### Owned temporary files and unavailable measurements

Live disk accounting now includes the known workspace and power files in
immediate, exclusively owned `fft-work-*` directories. Logical file lengths
count sparse allocations without double-counting overlapping mappings.
The case root and nested workspace must be real directories; symbolic links,
Windows reparse points and nonregular data files are rejected. Cleanup races
can remove a file, but other metadata failures terminate and reap the owned
worker. Tests exercise an actual oversized sparse file and process cleanup.

Unavailable process-memory counters are represented as unavailable, never
as a successful zero-byte measurement. Complete-tree coverage is required.
If a necessary live measurement remains unavailable for one second, the
supervisor terminates and reaps the worker; a process that has already exited
is reaped normally. The separate direct-reducer calibration identity prevents
RAM, packed or materialized observations from qualifying its forecast.

### macOS RSS and USS

The final Darwin reader deliberately does not issue the optional privileged
USS query. It reads complete process-tree RSS and leaves private memory
unavailable. This also avoids a synchronous watchdog becoming blocked in
the optional query. An unreadable RSS value anywhere in the live tree
invalidates the whole sum; known parent bytes do not substitute for a missing
child measurement.

Earlier `ca2d26f` / `5f64570` CI recorded a real macOS child with readable RSS
and `AccessDenied` for USS. That implementation preserved RSS after the
exception. Its first portable cohort completed all twelve monitored macOS
RAM cases, with 96 valid live samples and no missing samples; its separate
disk correctness smoke passed. These are historical observations of that
reader. The final test records USS as `not_queried` and verifies the RSS-only
reader, with the new platform evidence linked below.

RAM cases use measured RSS. For disk cases on macOS only, complete positive
RSS can conservatively replace unavailable USS **against the same byte
limit**. RSS can include shared/file-backed pages and can count them more
than once across processes; this may reach the limit earlier. The report
increments `live_memory_rss_fallback_samples` for every such sample and does
not relabel RSS as private memory. The monitored portable cases above used
RAM, so their fallback count is zero; they do not constitute an observed
supervised macOS disk-fallback campaign.

The public fallback counter must be a bounded nonnegative integer no larger
than the number of valid live samples. Missing, partial or malformed counters
cannot claim successful monitoring. Linux and Windows keep their existing
anonymous/private-memory policies.

### Extreme CSD arithmetic

The instrument reported relative error `3.0936972401917152e-12` in
`test_csd_batches_preserve_native_exceptional_arithmetic[1e+155-2049]`, above
the required `2e-12`. The batching path now checks the four already computed
segment projections against a conservative exponent bound for the legacy
variance. Large or nonfinite projections disable batching and replay the
**entire segment sequence once** through the existing single-CSD path.
This resets all earlier batch contributions while reusing the prepared
coefficients. Ordinary inputs retain batching without another input-data scan.

The new route tests observe actual batch, single and replay counts in a
temporary instrumented copy. They cover early and late exceptional values,
the last single segment, NaN/Inf, unequal channel scales, a zero channel and
the output selection without variance. After fallback, finite outputs are
bit-identical to the same build's single reference and nonfinite masks match.
The frozen references, public ABI and existing numerical tolerances are
unchanged. Locally, the new and existing CSD tests passed **204/204** with
GCC; the **28** route tests also passed with Zig 0.13.0 and baseline x86-64.

A bounded normal-input control used seven alternating pairs of 64 native
single-frequency calls per observation. Each stored time is the block's
mean per call. Medians of those means rose **0.28%** at
segment length 2,049 and **1.50%** at 16,385; comparison outputs were
bit-identical. The initial shorter screen was noisy. The
[CSD evidence](../benchmarks/results/csd-extreme-fallback-20261012.json)
keeps the observations separate. This small control neither measures a full
estimator nor establishes the cost on other platforms.

The original instrument's discrepancy was not reproduced exactly on local
x86. The final platform tests qualify the guarded path on their own native
hardware, but a run of the corrected fast.9 source on that physical instrument
is still required for device confirmation. The guard covers the tested extreme projection range;
it is not a universal guarantee for every cancelling input.

### Portable regression fixtures

The frozen matched-FFTW comparison previously tried to obtain Git metadata
inside source archives. This blocked 25 physical ARM tests before numerical
comparison. Its fixture now verifies the frozen file's fixed SHA256 first,
then supplies explicit snapshot metadata and the four actual source digests,
with `commit=None`. The cache is restored after the module. Changed reference
bytes fail immediately. All **108** matched tests passed with the frozen
module's Git lookup deliberately unavailable, including two provenance tests.

The simulated Linux address-guard test uses a platform-neutral POSIX path
comparison on Windows. The POSIX deadline test waits for a completely written
PID matching its owned child within a bounded startup window, then shortens
the deadline to 150 ms and verifies termination/reaping. Production deadlines
are unchanged. The Windows portable workflow runs these archive-relevant
tests from its private source copy, using its actual private FFTW and native
compiler, and requires the frozen, Bluestein and CSD route cases to execute.
The instrumented CSD fixture copies its complete recursive include closure
with unchanged relative layout and verifies all source digests before adding
counters. Its compiler and provenance reader therefore work even when the
checkout and pytest directory occupy different Windows drives.

## Actual physical-device results

The later upload contains distinct physical Windows and ARM64 campaigns.
The [device report](device-tests-20261012.md) and its
[manifest](../benchmarks/results/device-tests-20261012/manifest.json) retain
the original successful, failed, skipped and interrupted records. These
fast.8 observations are separate from the new fast.9 CI qualification.

| Device, tested at `b8c5002` | Retained benchmark coverage | Interpretation |
|---|---|---|
| Windows x86-64, 16 admitted workers | **131 unique successful cases / 132** | The doubled-point 1B LPSD case was interrupted by a host crash; eight repeated support cases in the continuation are not counted twice |
| ARM64 SBC, two workers, 384 MiB RAM budget | **108 completed, zero failed, 24 skipped / 132** | Complete recorded benchmark; skips remain skips |
| ARM64 instrument, two workers, 768 MiB RAM budget | **132 recorded: 116 completed, three failed, 13 skipped** | Final fast.8 report recovered; three memory-limit failures remain, with a separate completed regression recheck |

The later [device evidence update](https://github.com/MichaDit/lpsd/commit/e193080134a9578f9dd0ca1b1b809e8f64e470e2)
recovered the instrument's complete report and post-benchmark regression run.
The latter passed 2,101 tests, failed the deadline and exceptional-CSD tests,
skipped six and retained two expected failures. All 25 frozen-reference
prerequisite failures were resolved. Cleanup and release were freshly verified
at 17:15 UTC on 12 October. These are recovered **fast.8** results; the new
fast.9 corrections still have their separate CI qualification.

Windows has approximately **64 GiB physical RAM** (64,727 MiB reported) and
used a **32 GiB benchmark ceiling**, with 16 GiB of admitted dataset disk.
It is not evidence from a physical 32 GiB machine. The SBC and instrument
each used a 2 GiB disk ceiling. The instrument ran a private Debian ARM64
chroot on an Android kernel, with ordinary acquisition/web processes present;
this does not qualify general Android or Termux installations.

The Windows fast.8 **one-billion-sample** API observations are:

| Estimator | 744 points, s | 1,488 points, s |
|---|---:|---:|
| Simple FFTW | 57.379 | 55.922 |
| LNSD-matched FFTW | 173.534 | 261.393 |
| Native LPSD | 410.560 | No completed result |

These are single complete-call timings, not paired medians or measurements of
equivalent estimators. Dependencies and source revisions differ across older
campaigns, so the tables do not establish an isolated cross-revision speedup.
Fixed worker counts alone do not establish an optimum. At small input sizes,
worker startup, setup and checks can dominate process time: the instrument's
100k simple FFTW observation is 0.126 s API versus 19.197 s process wall time.
No unmeasured startup improvement is claimed.

The recovered small-ARM reports also distinguish the two admitted workers
from the actual FFTW execution: the SBC and instrument report one FFTW thread,
including the matched estimator, while native LPSD reports two. Their FFTW
runtime is 3.3.5 and no threaded-library digest is retained. The loader tries
discovered threading libraries before its explicit one-thread fallback; there
is no ARM policy forcing one thread. The reports do not retain the load/init
error needed to distinguish an absent threading library from another runtime
failure. A working threaded runtime and an on-device one/two-thread comparison
would be needed to qualify an additional gain. The present data establish
neither a loader-code defect nor a measured two-thread improvement.

The control host recorded bugcheck **0x4E** on restarting at
`2026-10-12T09:32:56Z`; its cause remains unknown. The earlier `6246f0c`
Windows 1B/doubled LPSD `invalid_result` lacks an invariant breakdown or
retained spectra, so it is not diagnosed or relabelled as a valid timing.
Three instrument 30M disk cases hit their memory limit in both recorded
campaigns; this does not justify increasing device budgets. The corrected
source is ready for new device runs, while the original failed results remain
recorded.

## Validation and source identity

The complete local regression suite at the earlier `ca2d26f` runtime passed
**2,264 tests**, with **5 skips and 2 inherited expected failures**, in
58.88 seconds. PowerShell 7.6.6 was available and its watchdog regressions
ran. The skips concern optional GPU/platform capabilities; the two expected
failures remain the inherited low-energy relative-error and K=2 uncertainty
cases. After the test-fixture correction, all ten comparison-harness tests
passed on the correctly built Linux runtime. A missing-FFTW simulation kept
the five CLI tests active and skipped exactly the five runtime cases.

The first integrated PR cohort on `ca2d26f` had **4/4 portable jobs and
7/8 native/build jobs pass**. Its macOS native environment lacks optional
FFTW: five newly added harness tests failed their prerequisite. The separate
macOS portable environment included FFTW and passed the direct ABI3 and
all-point numerical smokes. The subsequent fixture-only correction handles
that optional dependency without hiding ABI or numerical failures.

The fixture-corrected `5f64570` cohort subsequently passed all **8 native/build
jobs and 4 portable jobs**:
[Native](https://github.com/MichaDit/lpsd/actions/runs/38241578686) and
[Portable](https://github.com/MichaDit/lpsd/actions/runs/38241578688).
Its actual PR checkout also contains the incoming main-branch device documents;
runtime file hashes, rather than a claim of whole-tree identity, bind those
observations to the source.

The final device-fix runtime `83da4d` passed the complete local suite:
**2,309 passed, 5 skipped, 2 expected failures in 69.35 s**, including the
PowerShell watchdog checks. The subsequent `c417b9a` change only tightens
the readiness fixture; its focused real-process test passed in 0.24 s.

At `c417b9a`, all eight native/build jobs and three portable jobs passed.
The Windows portable job recorded **381 passed, 2 POSIX skips and 28 setup
errors**. Every error came from the single instrumented CSD fixture: its
provenance reader could not express D-drive includes relative to a C-drive
temporary source. This blocked its source manifest before the actual CSD
probe compilation; only compiler discovery/feature probes had run. The
production CSD extreme test, all 24 frozen comparisons,
Bluestein, both provenance tests, all four small matched workers and the ZIP/path
regressions passed. All five native numerical jobs also passed the 28 CSD
route tests and the production extreme case, including ARM and macOS.

The test-only `1c34582` correction copies the recursive include closure onto
one temporary drive and verifies its original bytes before instrumentation.
Its local **28 route tests passed in 6.64 s**. No CI-only temp-directory
workaround or provenance exception is used. Production sources and numerical
assertions are unchanged from `83da4d`. The untouched upstream files and both
frozen reference snapshots retain their original hashes.

The final `1c34582` qualification passed all **8 native/build jobs and
4 portable jobs**:
[Native](https://github.com/MichaDit/lpsd/actions/runs/38246521516) and
[Portable](https://github.com/MichaDit/lpsd/actions/runs/38246521492).
The actual PR checkout has the same complete tree as this source revision.

The Windows private installation recorded **409 passed, 2 intentional POSIX
skips and no failures or setup errors**. All **63 required regression cases
actually executed and passed**, including the 24 frozen comparisons, Bluestein,
28 instrumented CSD routes, both source-provenance checks, four small matched
workers and the production extreme/archive/path cases. The two skips are
`test_deadline_stops_subprocess_group` and
`test_shell_help_and_invalid_revision_are_offline`.

All five native numerical jobs passed their 28 instrumented CSD cases and
the production `[1e+155-2049]` extreme case. The four Linux suites each
recorded **2,291 passed, 5 skipped and 2 expected failures**; macOS recorded
**1,890 passed, 406 skipped and 2 expected failures**. These CI suites and
the overlapping focused checks are not added together as distinct tests.

The real macOS owned-child probe used psutil 7.2.2 and confirmed RSS/tree-RSS
availability with `uss_probe=not_queried`. Across the four platforms, all
48 supervised RAM cases completed with observed memory: **663 valid samples
and 4 temporarily missing samples**, including **149 valid and 1 missing**
on macOS. These RAM runs recorded zero RSS-fallback samples; they do not
constitute a supervised macOS disk-fallback campaign.

Both independent artifact auditors found no consistency errors. The final
report binds all 12 jobs, nine archive digests and the source/test manifests
to the tested revision. Its paired numerical checks are bit-identical for
**41,544 packed-phase**, **75,312 full-versus-packed disk** and **1,648 direct
ABI3** values per PSD/NSD output type. These checks qualify the final source;
the performance comparisons above retain their original measured revisions.

The source-bound [integrated validation JSON](../benchmarks/results/fast9-integrated-validation-20261012.json)
records the workflow/job identities, tested trees, artifact digests, source
closure and exact test counts. The earlier failed cohort is retained
separately; timing observations from different revisions are not pooled.
The [final device-fix qualification](../benchmarks/results/fast9-device-fixes-validation-20261012.json)
is a separate record for the new CSD, archive and RSS-only source.

| Evidence | Measured source identity | Data |
|---|---|---|
| Four-platform 1M packed-phase comparison | `f4082f04f1fbfee166f40e9bcba48f73056eed40` | [48 pairs](../benchmarks/results/packed-phase-platforms-20261012.json) |
| Linux 100M packed-phase comparison | `f4082f04f1fbfee166f40e9bcba48f73056eed40` | [12 pairs](../benchmarks/results/packed-phase-100m-20261012.json) |
| Linux 100M direct-power comparison | `ca2d26f91ed94d295a2c4b497db8aa2775b3bcdc` | [6 pairs](../benchmarks/results/direct-power-100m-20261012.json) |
| Recovered external experiments | Each historical experiment's retained source hashes | [Follow-up archive summary](../benchmarks/results/recovered-followup-20261012.json) |
| Last external 1B attempt and initial PR #3 local checks | Historical E/external R3 attempt and separate `e6aa1603` local code freeze | [Final archive summary](../benchmarks/results/recovered-final-attempt-20261012.json) |

## What the recovered agent actually completed

The delayed experiment archives are now included in the evidence. Their
manifests and retained file hashes were independently verified. They supplement
the earlier recovered Linux campaign. The physical-device reports arrived
separately afterward and are summarized above, with their own source pins
and hardware conditions.

The external R3 study supplies useful small-case correctness evidence:
30 contract cases, 24 configurations, 48 complete calls and 18,828 PSD/NSD
comparisons per output type. Its source predates the integrated adapter and
its observer-instrumented times are not used as performance results.

The last external billion-sample attempt completed its E reference in
**1,153.282266 s wall / 322.839617 s CPU**, returning 744 points. Before the
R3 child started, the next fixed disk-space preflight was short by
**64,974,848 bytes** while preserving the existing 2 GiB reserve. Thus the
archive contains one reference observation, **zero completed pairs and no
R3 timing**. It does not qualify the integrated reducer at one billion
samples. The archived source pins, controller states and independent
checks of the stored reference values are preserved in the final summary.

A separate recovered 1B column experiment completed one ordered pair:
strided-default first took **784.942 s**, then auto took **1,276.722 s**.
Strided used 41.31% more process write bytes, and its factorization also
differed. This is a 38.52% smaller wall observation for that configuration,
not an isolated estimate of changing column policy. There is no automatic
policy change based on this pair. The G2 100M evidence does not establish a
reproducible general improvement, and its large doubled-point attempt has
no valid completed pair; that experimental change is not integrated.

## Reproduction

Use the full commit pinned in the [portable guide](portable-benchmark.md)
for fresh installations. The phase and direct reports each give an exact
paired command with their distinct FFT budgets and source identities.
Retain all complete-call repetitions, actual output counts, library/source
hashes, numerical gates and incomplete attempts. Use the supervised portable
runner, with a real local-disk work directory and the existing resource and
deadline limits, for future large-machine campaigns.
