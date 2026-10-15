# fast.10 maintenance: device findings and local-only validation

Date: 2026-10-15. Base: `5b1f7eb21affdb1a1614dd535d6b0226d107ced2`.
This is benchmark/runtime-discovery and test maintenance, not a new numerical
estimator. The fast.10 native code, corrected/legacy statistics, spectral
normalization and tolerances are unchanged. No global speed optimum is claimed.

## Recovery and provenance

The saved physical-device campaign was found in commit
`d8f5fc0750b7139ff220ec2e45d876da49b67061`, including the
[device report](device-tests-fast10-20261013.md) and 20 sanitized reports with
525 retained rows. Older October 8 and October 12 recovery archives were
inspected. No newer unpublished implementation from the interrupted agent
was recovered; the old agent's local working directory was not available.
The changes described here were developed anew against verified upstream
file blobs, rather than presented as recovered unsaved work.

Both GitHub Actions workflow definitions were removed from `main` in
`5b1f7eb21affdb1a1614dd535d6b0226d107ced2`. No new Actions run was dispatched.
Historical evidence and old branches were preserved. This is removal of
workflow definitions on main, not a claim that an account-wide administrative
Actions setting or historical storage was changed.

## What the physical campaign actually establishes

| Device | Saved regression outcome | Important limitation |
|---|---|---|
| Windows x64 | 2677 fast tests and 18 upstream tests passed | The successful 100M continuation is separate from the interrupted initial campaign. |
| ARM64 SBC | 2501 fast tests passed, 5 failed; 18 upstream passed | Five tiny comparison tests can stop at a conservative 512-MiB admission estimate before their intended numerical assertions. |
| ARM64 instrument | 2505 fast tests passed, 1 failed; 18 upstream passed | The live-child test incorrectly required the optional `NSpid` status field. |

The eight instrument benchmark failures are distinct from its one regression
test failure: four default and four packed-real 30M FFTW/matched cases stopped
at the unchanged 768-MiB guard, using **RSS fallback** rather than an anonymous
resident counter. They are not evidence of eight wrong spectra or eight OS
OOM events. The native LPSD RAM cases at 30M did complete: 752.514 s for the
base point count and 1587.011 s for the doubled count.

The SBC's incomplete main/packed matrices were also affected by a separate
outer supervisor retaining a regression-process ceiling. They must not be
counted as completed 30M comparisons. On Windows, the dedicated 100M
continuation completed with the intended larger allowance; those observations
must not be assigned to the previously interrupted process.

For details, exact counts, settings, skipped rows and evidence hashes, retain
the original linked report. No historical failed or skipped row was rewritten.
The previous Windows 1B restart/bugcheck remains unresolved and was not rerun.

## Implemented changes

### Lazy FFTW discovery and optional wheel dependency recovery

The loader stops searching as soon as a core/thread combination works.
A working system installation no longer causes an unnecessary wheel metadata
scan or OpenMP lookup after the pthread runtime already succeeded. Explicit
successful paths also avoid unnecessary automatic searches.

After an actual load error for a discovered wheel library, the loader may
initialize the installed `pyfftw` package once and retry the **same** failed
core or core/thread pair. This lets the package initialize bundled companion
dependencies. It neither downloads anything nor edits library search paths.
Automatic wheel companions remain with their own core family. Initialization
failures remain optional; failure diagnostics use the existing closed fields.
`main_candidates` now counts discovered candidates, not unsearched libraries.

The local ELF test demonstrates dependency recovery with real shared
libraries. The actual private wheel dependency failure on each ARM device was
not reproduced here, so the new loader is not proof that every device will
now configure two FFTW threads. Read back `fftw_threading` in the next run.

### Old-kernel anonymous resident memory

When `RssAnon` is absent, disk supervision first tries the owned process's
`smaps_rollup`, then `smaps`, summing only complete `Anonymous:` counters.
The parser bounds both line length and total bytes, validates mapping order
and mandatory counter pairs, and does not add PSS, Private_Dirty, swap or
AnonHugePages to the anonymous total. Denied, unavailable or malformed data
retains the same conservative RSS fallback, never a fabricated zero.

Modern `RssAnon` reads keep the existing fast path. RAM-only guards still use
total RSS and do not perform the otherwise unused page-table walk. The
existing RAM/disk admission limits, per-case byte ceiling, timeout and
failure policy remain in force. File-backed resident pages/page cache are
additional physical memory, subject to the OS/cgroup; this guard is not a
promise that total system RAM equals anonymous process residency.

Kernel field definitions and the non-atomic nature of procfs observations:
[Linux procfs documentation](https://docs.kernel.org/filesystems/proc.html).
`smaps` can be more expensive than status counters. There is no claim of
identical sampling cost or an atomic snapshot on older kernels.

### Correct the two test assumptions

The live-child regression now obtains the procfs-visible PID from the owned
child itself, verifies its parent relationship, and tests both with and
without `NSpid`. It does not guess that namespace PIDs equal proc directory
names. Denying a live child's memory reading still invalidates the sum.

The five fixed-size direct-power tests use a deterministic **test-only**
resource observation. A guard rejects any computation other than the
intended 4096-sample, one-worker, 128-MiB-allowance FFTW request before calling
the unchanged real pipeline. Numerical outputs, 3-ppm gates and corruption
assertions are not replaced or relaxed. A separate test keeps insufficient
resource admission a failure, not a numerical pass. Production benchmarks
and device resource ceilings are not overridden by this fixture.

### Local test entry point

[`scripts/validate_local.py`](../scripts/validate_local.py) builds both native
libraries and runs `test_fast` and `test`, using an existing environment.
It retains a fresh local report, source hashes, separate logs and JUnit
counts, with a timeout for each stage. Skips and expected failures are
reported separately. It performs no dependency installation, upload, Actions
dispatch or remote execution. See [local validation](local-validation.md).

## Measured benefit and settings

Eleven alternating local loader pairs, ten loads per observation, using the
same actual system FFTW runtime and two configured threads, gave medians:

| Measurement | Before | After |
|---|---:|---:|
| Loader-only time | 4.405 ms | 2.874 ms |

This is about 35% less **loader time**, or 1.531 ms saved per load in this
Linux x86-64 warm-process experiment with a narrow real-C-API adapter. It is
not a 35% improvement of a spectrum, a cold interpreter or another platform.
Complete-call benchmark timers exclude discovery, so their spectral compute
time should not be relabeled as accelerated by this change.

The saved device worker comparison already supports method-specific tuning
rather than raising threads indiscriminately. At 1M samples, 577 output
points, corrected statistics and white RAM input:

| Device and method | 1 thread | 2 threads | 4 threads |
|---|---:|---:|---:|
| Windows FFTW | 0.0496 s | 0.0496 s | 0.0515 s |
| Windows matched | 0.4326 s | 0.2573 s | 0.1702 s |
| Windows LPSD | 0.9245 s | 0.4972 s | 0.2715 s |
| SBC FFTW | 0.1737 s | 0.1775 s | 0.1821 s |
| SBC matched | 3.9216 s | 2.8109 s | 2.5159 s |
| SBC LPSD | 10.2416 s | 6.0284 s | 5.1979 s |

These are **single historical observations**, not new maintenance medians.
Four workers were useful for native LPSD/matched in that small matrix; FFTW
had no comparable benefit. No existing two-thread device request was silently
raised. The instrument's 1M worker matrix contains seven time-budget skips,
so it establishes no optimum. There is no new physical macOS measurement.
No speculative numerical kernel, GPU route or global dispatch change was
promoted without the corresponding hardware/performance evidence.

## Local validation actually performed

[Machine-readable evidence](maintenance-local-evidence-20261015.json):
**110 targeted tests passed**, zero failures/skips/expected failures on
Linux x86-64, Python 3.13.5. The breakdown is:

- 29 FFTW discovery tests, including real FFTW transforms for three lengths
  and 1/2/4 configured-thread requests, plus a real ELF dependency recovery.
- 40 smaps/supervision tests, including actual file mappings, anonymous
  allocations and enforced process termination; 9 local-launcher contracts;
  6 tiny-fixture isolation contracts.
- 26 Linux bootstrap tests run against the unchanged upstream memory-function
  excerpt, including both repaired live-child cases. This is not the entire
  bootstrap installation or a substitute numerical test suite.

An actual negative/positive guard control used the same 64-MiB limit, 16 MiB
of anonymous data and a 64-MiB file mapping. With the optional status counter
omitted, the baseline stopped on conservative RSS fallback; the candidate
completed with measured anonymous peak 21,475,328 bytes. A separate 128-MiB
anonymous allocation still triggered the unchanged limit. RAM-mode RSS
supervision also still rejects the mapped workload over its total-RSS cap.

**Not performed here:** a fresh full native LPSD regression suite, actual
Windows/ARM/macOS execution of this maintenance patch, or the instrument's
30M rerun. Direct runtime network access and the interrupted agent's complete
checkout were unavailable; exact changed files were read through the GitHub
connector, and tested helpers were executed from their actual source bodies.
The numerical kernels were not changed. The five direct-power numerical
pipeline cases were not rerun against a complete current native build here;
their fixture-isolation contracts were tested separately. Do not turn the
110 targeted passes into full multi-platform qualification.

## Remaining checks, without changing existing ceilings

Use a current complete checkout and the local validation command first.
Then rerun the existing small 12-case smoke and the affected cases with
corrected statistics and the same memory/disk limits. Retain actual source
and binary hashes. On the ARM devices verify selected FFTW threads; on the
instrument verify the guard metric and whether smaps is accessible before
interpreting a repeated 30M stop. An unavailable smaps interface correctly
retains RSS fallback and may still stop that case.

Keep the Windows 1B restart investigation separate. Do not reuse a small
regression supervisor's RSS ceiling around a larger admitted benchmark.
Until the physical checks exist, this is a locally tested maintenance patch,
not a claim that every outstanding hardware qualification has been completed.
