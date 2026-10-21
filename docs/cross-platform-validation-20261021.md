# Additional local platform validation (fast.14)

Date: **21 October 2026**. Baseline:
[`3317b77ca0aca40a9639f3b02da657da193a6bb8`](https://github.com/MichaDit/lpsd/commit/3317b77ca0aca40a9639f3b02da657da193a6bb8),
version **1.0.6+fast.13**. Candidate: **1.0.6+fast.14**.

Additional Python runtimes, QEMU, Wine and cross-compilers were installed in
the local Linux environment to execute the previously unavailable CPU/OS
paths. This follow-up found two real 32-bit FFT adapter defects, a Windows
build-provenance defect and a failure-reporting defect. Their fixes do not
change the native numerical C sources, the original reference, numerical
tolerances or FFT algorithms. An existing C ABI test fixture also needed a
portable exported symbol name.

The earlier [physical-device analysis](device-analysis-fast12-20261021.md)
and [measured disk-call improvements](performance-fast12.md) retain their
original scope. Emulated execution here checks functionality and ABI
compatibility; its elapsed times are not target-hardware performance results.
No GitHub Actions or remote-device jobs were started.

## Results and execution scope

| Target | Local execution/build | Result |
|---|---|---|
| Linux x86-64 | Native GCC 13.3, Python 3.12.14 | **2,725 fast-suite passes + 18 upstream passes**, 5 skips, 1 known expected failure; sdist/wheel build and isolated installed-wheel smoke pass. |
| Linux ARM64 | Actual AArch64 binaries under QEMU | Final affected FFT modules: **188/188 pass**; manifest checks: **5/5 pass**; platform ABI module: **14 pass, 1 skip**. Across four runs: **665 distinct passes, 2 skips**. |
| Linux ARMv7, 32-bit | Actual armhf binaries under QEMU | Final affected FFT modules: **188/188 pass**; manifest checks: **5/5 pass**; platform ABI module: **14 pass, 1 skip**. Across four runs: **666 distinct passes, 1 skip**, including the actual FP64 inline-mean path. |
| Windows x64 | Actual Windows Python and Clang under Wine 11.20 | Latest status across the full suite and two targeted follow-ups: **2,633 pass, 55 Wine file-identity failures, 42 skips, 1 expected failure**. All **18 upstream tests** and the isolated installed-wheel smoke pass. |
| macOS ARM64 + x86-64 | Cross-compile/link | Both native libraries build on each target; Mach-O architecture, exports and **9 ABI assertions per target** pass. No macOS runtime execution. |
| Windows ARM64 | Cross-compile/link | Both native libraries build; **7 ABI assertions** pass. No Windows ARM64 runtime execution. |
| PowerShell 7.6.6 | Actual Linux PowerShell runtime | Existing bootstrap module: **46/46 pass**, also included in the Linux fast-suite total. |

The ARM counts combine **527 broad cases, 188 final FFT cases, five
build-manifest cases and 15 platform ABI cases**, with **68 repeated IDs
counted once**. They are not a single final 667-case run.
Before the aggregation fix, the broad ARMv7 run had **519 passes and eight
failures**; the corresponding ARM64 run had **526 passes and one skip**.
All eight failures are covered by the final passing FFT run. Unchanged
broader C/statistics paths were not expensively repeated after a Python-only
FFT index fix. The last Windows path-key change was then checked separately
with the five manifest/include/hash cases on each guest, excluding two tests
that would rebuild unrelated C fixtures. The later test-symbol correction is
qualified by the complete platform ABI module on both guests. Its only skip
is the actual Darwin/Windows OS query on Linux. The final guest source
inventories match the host byte for byte, including the validator files
tested separately on Linux and Windows.

The Linux skips are two unavailable CUDA modules, the real macOS RSS/USS
query, the inapplicable FP64 inline-mean path and the actual Windows/macOS
memory-query test. The known expected failure is the documented uniform
relative-accuracy limit near an off-bin tone's spectral null. The full Linux
and Windows runs share the same **191-file** source/build inventory. The only
subsequent source change is the C symbol name in `test_platform_memory.py`,
checked with that entire module: **14 passes and one native-platform skip
on Linux**, **15 passes on Windows**. Those repeated cases do not increase
the Linux suite total. All production sources and both Linux native library
hashes remain identical to the completed full Linux run; the native library
hashes also match the fast.13 baseline.

The machine-readable [evidence summary](../benchmarks/results/cross-platform-20261021/summary.json)
records versions, commands, counts, exclusions, source fingerprints and native
library hashes. Full local logs and JUnit remain local because they include
environment paths and are not the portable benchmark's privacy allowlist.

## Defects found and the retained tests

### Buffer sizes on 32-bit hosts

FFTW's one-dimensional API accepts a signed 32-bit element count, but that
alone does not prove that the required byte buffers are addressable. The
ordinary real FFT also has a padded complex output; Bluestein has a larger
complex convolution buffer. Checking only the real input length misses both.

A real ARMv7 probe intercepted the allocator without allocating large arrays.
For `N = 2**29 + 2`, the old adapter passed only **16 and 32 bytes** after
conversion to 32-bit `size_t`; the in-place route passed **32 bytes**. An
explicit Bluestein convolution passed **16, 32, 48 and 48 bytes**. With the
fix, all three calls raise `ValueError` before any allocator call. Such a
truncated allocation cannot safely hold the requested transform.

The adapters now check the largest individual buffer against the platform's
NumPy index range before backend access. Bluestein applies this check even
when `max_working_mb=None`; disabling a policy cap cannot enlarge the address
space. This is an addressability check, not an estimate of available RAM.
`RealFFT` also converts integer-like lengths to a Python integer before byte
arithmetic. Otherwise a NumPy `int32` length can overflow under NumPy 2 even
on a 64-bit host.

Seven small regressions were added to existing modules: four padded-buffer
boundary cases covering both real layouts, one NumPy-integer allocation-size
case, and two Bluestein convolution boundaries. All seven failed before the
fix. They use a missing backend or a null allocator and never allocate the
multi-gigabyte buffers. The existing two billion-sample layout checks now
assert the expected rejection on a 32-bit runtime instead of asking it to
describe an unaddressable layout; their 64-bit checks are retained.

### NumPy logarithmic grouping on ARMv7

The first complete selected ARMv7 run exposed **eight failures** after
successful raw FFT execution. NumPy's `add.reduceat` requires indices that
fit `np.intp`; its default safe conversion rejects an `int64` index array
on a 32-bit runtime even when every value is small and valid. The shared
group boundaries were `int64` because the native C ABI requires that type.

Only the NumPy reduction indices are now converted to `np.intp`, after the
existing partition checks. The C boundary array stays `int64`. The prepared
binding caches its converted starts once, so repeated calls do not add an
index conversion or allocation. Three small cases exercise the standalone
aggregation helper, the unbound NumPy fallback and the prepared binding.
They also preserve the expected density and integrated band power.

The eight original failures and three new cases all pass on the actual
ARMv7 runtime after the fix. The three new regressions also failed before
the change on the 64-bit host using the same safe-casting rule. Complete
affected FFT modules are rerun on both ARM guests; the earlier broader
selection is recorded separately rather than counting repeated cases twice.

### Preserve the error when a mapping has already been closed

An early Windows/Wine run crashed while pytest was formatting a failed
assertion involving a memory-mapped array. A four-double example reproduced
the same problem on native Linux: a long traceback tried to display an array
after its `finally` block had unmapped the memory, and pytest exited with
`SIGSEGV` instead of retaining the original assertion and JUnit.

The local validator now uses `--tb=short`. The same probe then exits normally
with the expected assertion failure. One small integration test runs the
actual validator against a temporary checkout with no-op builds and checks
the error message, nonzero result and JUnit failure count. It does not turn
the intentional failure into a pass or suppress test assertions.

### Native builds with source and headers on different Windows volumes

The full Windows run also exposed **36 setup errors** from one independently
compiled segment-start fixture. Its temporary C file was on `C:`, while the
actual project header was on `Z:`. The build helper hashed recursive includes
using `os.path.relpath`, which raises `ValueError` across Windows volumes.
The same issue applies to a native Windows checkout on `D:` with `TEMP` on
`C:`; it is not specific to Wine or to a numerical kernel.

The source manifest now retains an absolute, resolved key when a relative
path cannot represent the header location. Ordinary relative keys, recursive
include discovery, cycle detection, byte hashes and the before/after build
comparison are preserved. One small regression exercises the real `ntpath`
different-volume error on every host and checks both recursive header hashes.
The actual Windows fixture is rebuilt in its original locations.

### Portable symbol name in the C ABI test fixture

The completed Windows suite found four failures in the existing memory-query
fixture. Its DLL defined a C function named `GlobalMemoryStatusEx`, but the
MinGW auto-export rules omitted that system API name from the PE export
table. The actual production Windows memory-query test passed.

The fixture now uses the private C name `lpsd_test_GlobalMemoryStatusEx` and
assigns the existing Python attribute to that loaded function pointer. All
structure-layout, return-value and above-4-GiB checks remain unchanged, as
do the other fixture exports. The complete existing 15-case module is checked
again on Windows and Linux and added to both ARM selections. No additional
test case or production-code change is needed for this correction.

No new dependency, emulator management layer, benchmark framework or timing
assertion is added to the project.

## Runtime details and limits

### Linux ARM64 and ARMv7

The guests use official Ubuntu Noble packages: Python **3.12.3**, NumPy
**1.26.4**, pandas **2.1.4**, SciPy **1.11.4**, FFTW **3.3.10** and GCC **13**.
QEMU user mode **8.2.2** executes their actual ELF binaries through PRoot
**5.5.0**, with Cortex-A53 and Cortex-A7 CPU models. Both the reference and
fast libraries were built for the target. A guest compiler also builds the
small C fixtures required by the tests. FFTW core and thread libraries are
found through a guest-local loader cache.

ARM64 uses 64-bit pointers/`long` and a 113-bit `long double` mantissa.
ARMv7 uses 32-bit pointers/`long` and a 53-bit `long double` mantissa. The
ARMv7 run therefore executes the actual FP64 inline-mean specialization that
the x86-64 host skips, including its independently compiled bitwise check.
The ARM64 skip of that specialization is expected.

The selected suite covers numerical statistics, API/resource boundaries,
prepared calls, FFTW, Bluestein, disk operation and platform ABI checks.
Twenty-four complete-spectrum high-point cases and eight longer independent
statistics-oracle cases were deselected to bound software-emulation cost.
The short independent statistical checks and native long-segment checks
remain. The initial interrupted exploratory runs are not counted as completed
qualification. QEMU user mode shares the Linux host kernel; these are not
physical Raspberry Pi, instrument, Android or complete guest-kernel results.

### Windows x64

Actual Windows CPython **3.12.10**, NumPy **2.3.5**, pandas **2.2.3**, SciPy
**1.17.0** and FFTW **3.3.5** run under Wine. The primary native build uses
the Windows executable of LLVM-MinGW **20260616 / Clang 22.1.8** with UCRT.
A separate GCC-MinGW **13.2** build checks a second compiler. Windows Python
uses 32-bit C `long` and 64-bit pointers; platform/resource tests exercise
these real Windows ABI rules.

Wine **9.0** initially exposed missing `CreateSymbolicLink`/`CopyFile2`
behavior, blocking link checks and wheel construction. A separately extracted
official WineHQ **11.20** runtime passes independent file-link,
directory-link and `shutil.copy2` probes. The final broad run uses that
runtime. Both runtimes require Microsoft's app-local UCRT for this full
numerical suite. Wine 9 lacks `crealf`; Wine 11.20 passes the initial NumPy
import and file probes with its built-in UCRT, but later aborts on the
unimplemented `cimagl` used by the independent complex-periodogram oracle.
The final run uses the official Microsoft Windows SDK **10.0.26100.1** UCRT.
The aborted exploratory run has no complete JUnit and is not counted as a
completed suite.

Both Wine versions still return zero device/inode identifiers for distinct
files while they are open as NumPy memmaps. Before mapping, their identities
differ; after mapping, `os.path.samefile` incorrectly reports equality. This
was reproduced outside LPSD on both the prefix's `C:` drive and the host's
`Z:` mapping. The resulting conservative input/output alias rejections are
reported as runtime limitations. Production alias protection and its tests
are not weakened to conceal this result.

The completed full Windows run records **2,615 passes, 59 failures, 56 skips
and one expected failure**, with no setup errors. The final symbol-name
correction resolves all four fixture failures in a separate **15/15** passing
module run. A further **23-case** selection repeats exactly the cases that
skipped because optional FFTW discovery was unavailable. Installing the
official **pyFFTW 0.15.1** Windows wheel, without changing dependencies or
source, gives **14 passes and nine preserved skips**. The wheel supplies the
byte-identical FFTW 3.3.5 double-precision DLL through the normal metadata
discovery path. The nine remaining cases have an earlier
`ctypes.util.find_library("fftw3")` prerequisite, which does not find the
wheel-private DLL here; their assertions are not claimed as executed.

Keeping the latest status per case across these three runs yields **2,633
passes, 55 remaining failures, 42 skips and one expected failure**. Repeated
IDs are counted once; this is not a second full-suite run. The evidence
summary retains both source fingerprints, the additional runtime's hash and
all three raw results. The other skips concern unavailable CUDA, system/OS
interfaces, floating-point environment support and the inapplicable inline
mean specialization.

All 55 remaining failures reach file-identity guards: 17 reject a power file
as the transform workspace, one rejects a packed-real power file, 36 reject
input/scratch overlap, and one reaches the same alias rejection before its
intended nonfinite-input check. Independent review found no other failure
category. Their failing assertions are retained; they are not converted to
skips or counted as passing numerical disk tests.

All **18 upstream tests** pass in a separate run. The Windows wheel builds,
installs into a separate target without dependency downloads, and passes the
isolated smoke check outside the checkout. That check covers the installed
native ABI and build hashes, exact legacy equality, PSD/NSD selection and
the independent corrected-statistics example. A separate earlier GCC-MinGW
selection has **391 passes and the same two mapped-disk identity failures**;
it is additional compiler evidence, not another complete final suite.

### macOS ARM64 and x86-64

Zig **0.15.2**, with Clang **20.1.2** and its Darwin headers/link stubs,
compiles and links both native libraries for **macOS 11.0 ARM64 and x86-64**.
All four outputs are genuine Mach-O dynamic libraries of the intended
architecture, with only `libSystem` as a dependency. The **50 fast and
14 reference project exports** agree with the Linux build. Nine compile-time
ABI assertions per architecture verify the resource-query layouts against
Darwin headers; the probe also links the four Mach resource symbols used by
the implementation. ARM64 compiles the 53-bit `long double` branch; x86-64
uses a 64-bit mantissa.

An actual startup of the official Darling **0.1.20260609** Noble package was
attempted. An isolated user namespace works, but Darling exits before
launching Mach-O programs with `Cannot mount overlay: Invalid argument`,
including with a RAM-backed prefix. No host kernel settings were changed.
These results qualify compilation, linkage, exports and ABI declarations;
they do not establish macOS runtime behavior or wheel execution.

### Windows ARM64 and PowerShell

Both native libraries also compile and link as Windows ARM64 PE binaries.
Seven static ABI assertions check LLP64 widths, the 53-bit/8-byte
`long double` and the Windows memory-status structure. No Windows ARM64
runtime is available, so this remains compile/link coverage.

The official PowerShell **7.6.6** Linux runtime executes the existing budget
check and the PowerShell-dependent bootstrap tests. All **46 tests** in
`test_bootstrap.py` pass on the native Linux host. This fills a previously
missing shell check; it is not evidence of a physical Windows PowerShell run.

## Reproducing the scope

With the described runtime/compiler and test dependencies installed, run the
ordinary complete host gate:

```sh
python scripts/validate_local.py --output-parent .validation/fast14 --stage-timeout 900
```

The recorded foreign-runtime commands and exact selected test paths are in
the evidence summary. For packaging, build the source distribution and its
platform wheel, install the wheel into a separate environment, then run
`test_fast/wheel_smoke.py` with `python -I` from outside the checkout.
The native C source inventory is unchanged from the baseline; foreign
binaries are kept separate from host build artifacts.
