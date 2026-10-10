# Native builds and architecture validation

The build uses a GCC- or Clang-compatible C compiler, selected with `CC`.
The original `lpsd/` sources are unchanged. Both native libraries are compiled
with `-O3`, `-fno-fast-math` and `-ffp-contract=off`, so the reference and the
optimized implementation use the same explicit contraction policy.

## Portable default and explicit local tuning

```bash
make compile CC=gcc
make compile CC=clang
python -m build
```

Default builds do not add a global host-specific ISA flag. AArch64's ordinary
Armv8-A target already includes floating-point and Advanced SIMD (NEON);
there is no unconditional SVE or SME requirement. See
[GCC's AArch64 options](https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html).

The fast builder probes OpenMP SIMD compilation, first with `-fopenmp-simd`
and then with Clang's frontend spelling. If neither works it reports a
warning and builds ordinary C loops. SIMD-only OpenMP does not require an
OpenMP runtime library or create worker threads; frequency parallelism remains
in Python. See the [Clang user manual](https://clang.llvm.org/docs/UsersManual.html)
for the SIMD-only and floating-point contraction options.

For an explicitly local build, `make compile-native` or
`python -m lpsd_fast.build --native` selects `-march=native` on x86 and
`-mcpu=native` on AArch64. Rebuild that library when moving to another CPU.
`LPSD_NATIVE=1` is allowed for editable installation but rejected during a
normal wheel build. Custom compiler arguments supplied through `CC` remain
the caller's responsibility.

On macOS the compiler and wheel metadata use the same deployment target,
11.0 by default. `MACOSX_DEPLOYMENT_TARGET` can select a newer version. The
native libraries retain their `.so` filenames for the existing ctypes loader,
but are linked as Mach-O dynamic libraries. The wheel tag uses the configured
compiler's target architecture; `lipo` verifies that both generated libraries
contain that single architecture. The native Apple CI passed with an `arm64`
wheel, including when Python's own configuration is universal2. Universal2
and cross-compilation are not validated by the supplied CI, and inconsistent
or fat library architectures are rejected during wheel construction.

### Optional x86 runtime dispatch

For x86 ELF targets with glibc 2.23 or newer, the fast builder compiles and
links a `target_clones("default", "avx2", "avx512f")` probe. Success defines
`LPSD_HAVE_TARGET_CLONES=1`, which enables the corresponding guarded native
annotations. The probe includes separate declarations, definitions and a call
between clones; declarations and definitions carry identical attributes,
as required by the validated Clang build. The native implementation applies
them to `fast_dft_impl`, the high-overlap `fast_dft_boxcar` implementation, and
the Kaiser, general-window and Fourier-coefficient generators. Their hot
private helpers are inlined into the clones, so the selected instruction set
applies to the actual computation loops. See the
[native declarations](../lpsd_fast/_native/fast_dft.h) and
[implementation](../lpsd_fast/_native/fast_dft.c).

The compiler keeps the default, AVX2 and AVX512F implementations and selects
a compatible clone through its runtime resolver. Selection is at the
frequency or generator call boundary; there is no resolver call per segment.
The default implementation preserves the library's baseline ISA requirement.
The build probe and the presence of clones do not establish a performance
benefit on a particular CPU.

Use `LPSD_TARGET_CLONES=0` to build without the macro, `auto` for the default
probe/fallback behavior, or `1` to require successful probing. Explicit
`--native` builds disable this portable dispatch option. This implementation
does not enable IFUNC dispatch on macOS, musl or AArch64.

The [GCC attribute documentation](https://gcc.gnu.org/onlinedocs/gcc/Common-Attributes.html)
and [Clang attribute documentation](https://clang.llvm.org/docs/AttributeReference.html#target-clones)
describe the target-clone resolver and the required default implementation.
GCC also notes that called functions do not automatically acquire clones;
annotations and inlining in the C implementation must be reviewed together.

## Build evidence and floating-point ABI

Each compilation writes a JSON sidecar beside its library:
`ltpda_dft.build.json` or `liblpsd_fast.build.json`. It records the compiler,
target, flags, SIMD/dispatch probe results, long-double format, deployment
target and library SHA-256. Generated reports are included in wheels, excluded
from source archives and not committed. A `target_clones: true` report means
the compiler capability and macro were enabled; it does not inventory which
C functions were annotated. The tests compare each report's hash with the
actual loaded-library file.

`long double` is not one portable precision or execution cost:

| Environment | Relevant precision |
|---|---|
| Native Linux x86-64/GCC CI | 64 significand bits, stored in 16 bytes |
| Native Linux AArch64/GCC CI | IEEE binary128, 113 significand bits, 16 bytes |
| Native Apple arm64 CI | Same representation and precision as `double`, 53 significand bits, 8 bytes |
| Local Windows x64/MinGW GCC 15.2 | C: 64 significand bits, 16 bytes; installed NumPy wheel: 53 bits, 8 bytes |

The [Arm AAPCS64 specification](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
defines the standard mapping; [Apple's arm64 documentation](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
documents its different long-double representation. Compiler-generated
quadruple-precision operations may use software helpers, so more precision
must not be confused with faster hardware arithmetic; see the
[GCC soft-float runtime interface](https://gcc.gnu.org/onlinedocs/gccint/Soft-float-library-routines.html).
The CI reports the actual compiler and NumPy formats instead of assuming them
from the CPU name. Detrending, projected coefficient preparation and reduction
rounding can therefore have different costs and numerical behavior on these
platforms. The [numerical limits](numerics.md) still apply.

### Opt-in compensated mean projection

For mean removal (`detrending_order=0`), `kernel="fast"` selects native mode 3
only when `native_long_double_mantissa_bits()` reports `LDBL_MANT_DIG > 64`.
That query returns the compiled C type's precision. It identifies neither
the processor's execution mechanism nor the cost of long-double operations;
the threshold is a selection rule, not proof of software arithmetic.
In particular, the ordinary x86-64/GCC and Apple arm64 formats in the table
above do not activate mode 3 through this rule.

Mode 3 prepares the same order-0 mean projector using compensated pairs of
FP64 values, then uses the existing anchored segment dot products. The
[compensated implementation](../lpsd_fast/_native/projected_dd.c) includes
two explicit `fma` operations per frequency to recover division residuals.
They are intentional parts of this arithmetic; general multiply-add
contraction remains disabled. Extreme coefficients can fall back to the
long-double preparation before any coefficient is modified.

This is a separate numerical path, with no guarantee of binary128's 113-bit
significand or bitwise equality to the long-double projector. Compensated
arithmetic assumes the usual IEEE rounding environment. The existing `auto`
mode uses mode 2 for order 0 when long double has more than 53 significand
bits, except on Windows; with at most 53 bits or on Windows it uses the
original residual arithmetic (mode 1).
The Apple runner exercises this fallback, including the DC-plus-nanovolt
regression case. Explicit `fast` still uses projected mode 2 on Apple, with
the separately documented numerical tolerances. For higher orders, `auto`
also retains the original residual detrending. The scalar path remains
available. Mode 3 does not extend the experimental order-1 projector.

An x86 build forced to use software binary128 is an arithmetic/build
surrogate. Its timings are not ARM measurements and do not establish an ARM
speedup. The native Linux AArch64 and Apple measurements below come from
their own runners, with numerical gates executed on those same architectures.

## Native CI matrix and small measurements

### Local Windows validation, 2026-10-10

The [local report](windows-accelerators.md) documents the actual Windows x64
MinGW build, complete test runs and an isolated installed wheel. Profiling
uses `QueryPerformanceCounter` on Windows, avoiding the POSIX clock's
additional `libwinpthread` dependency in the fast DLL. Compiler ABI tests
use an independently built C probe rather than assuming NumPy's long-double
format matches the C compiler. Windows `auto` selects residual detrending
after the projected order-0 DC regression failed; its numerical assertion
was not relaxed. These are local checks, not a new Windows CI job.

The [follow-up CUDA and FFT study](windows-optimization.md) adds 52 actual-device
regressions; portable/native local suites each pass 1,231 tests with two known
Xfails and no optional skips. The GPU code remains an optional benchmark
experiment, with explicit CPU fallbacks and separately measured estimators.

### Earlier native CI evidence

All eight jobs completed successfully in
[CI run 37803974329](https://github.com/MichaDit/lpsd/actions/runs/37803974329)
on 2026-10-08, for commit
[`0d80d10872fd0a6d632a272b31d356b28e15feb3`](https://github.com/MichaDit/lpsd/commit/0d80d10872fd0a6d632a272b31d356b28e15feb3).
Each of the five numerical jobs reported **300 passed and 2 expected failures**
for `test_fast`, plus **18 passed** for the unchanged upstream tests. All three
installed-wheel jobs also passed. These are completed native checks:

| Runner | Compiler | Python | Numerical tests | Installed wheel | Small benchmark |
|---|---|---|---|---|---|
| `ubuntu-24.04`, x86-64 | GCC 13.3 | 3.10, 3.12 | [3.10 passed](https://github.com/MichaDit/lpsd/actions/runs/37803974329/job/113403269825), [3.12 passed](https://github.com/MichaDit/lpsd/actions/runs/37803974329/job/113403268994) | [Passed](https://github.com/MichaDit/lpsd/actions/runs/37803974329/job/113403268914), 3.12 | Completed, 3.12 |
| `ubuntu-24.04`, x86-64 | Clang 18.1.3 | 3.12 | [Passed](https://github.com/MichaDit/lpsd/actions/runs/37803974329/job/113403268856) | Not in matrix | Not in matrix |
| `ubuntu-24.04-arm`, AArch64 | GCC 13.3 | 3.12 | [Passed](https://github.com/MichaDit/lpsd/actions/runs/37803974329/job/113403268796) | [Passed](https://github.com/MichaDit/lpsd/actions/runs/37803974329/job/113403268843) | Completed |
| `macos-15`, Apple Silicon | Apple Clang 17.0 | 3.12 | [Passed](https://github.com/MichaDit/lpsd/actions/runs/37803974329/job/113403268900) | [Passed](https://github.com/MichaDit/lpsd/actions/runs/37803974329/job/113403268634) | Completed |

These labels and architectures are documented in
[GitHub's hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
The workflow asserts the interpreter's architecture, builds both backends,
runs the existing numerical gates, and tests wheels installed outside the
checkout. No platform-specific relaxation of the numerical assertions is
introduced here. A changed expected-failure outcome requires review on the
actual platform.

The [earlier native run](https://github.com/MichaDit/lpsd/actions/runs/37801321576)
exposed the Clang declaration/definition attribute mismatch, the incorrect
macOS universal2 wheel tag and Apple's order-0 `auto` projection error on a
DC-plus-nanovolt input. The final run above passed after matching the clone
attributes, verifying native wheel architecture and selecting residual mode 1
for `auto` on the 53-bit long-double ABI. The numerical assertions were kept.

### Recorded wall times

The same run measured `N=100000` input samples and 469 actual output
frequencies, with the default Kaiser window, order-0 detrending and fixed
seed. Times below are seconds of ordinary public-API wall time. The original
API has one measured call; each optimized configuration is the median of
three calls. Every configuration has one small untimed warm-up call using
4096 samples and at most 100 requested frequencies.

| Native runner | Available workers W | Original, 1 worker | Auto, 1 worker | Auto, W workers | Fast PSD, 1 worker | Fast PSD, W workers |
|---|---:|---:|---:|---:|---:|---:|
| Linux x86-64/GCC | 4 | 1.651395 | 0.170180 | 0.119720 | 0.103736 | 0.100687 |
| Linux AArch64/GCC | 4 | 7.944584 | 0.391475 | 0.145769 | 0.172261 | 0.088587 |
| Apple arm64/Clang | 3 | 1.457533 | 0.442041 | 0.175523 | 0.126828 | 0.091993 |

The original API and `auto` return all seven output columns; `fast` requests
only PSD and skips variance accumulation. `fast` also uses blocked Fourier
coefficients and the positive-series Kaiser path with fallback. Its mode is
3 on Linux AArch64 and 2 on the other two runners. Auto uses mode 2 on the
Linux runners and mode 1 on Apple. Therefore an Auto/Fast API time ratio
includes several changes and cannot isolate one kernel's speedup.

In the separate one-worker Linux AArch64 profiles, projector preparation took
0.180995 s for Auto and 0.013152 s for Fast; segment work took 0.098275 s and
0.098642 s respectively. These native measurements support a reduced
preparation cost for the compensated path in this case, without claiming
binary128 precision for that path. The profiled calls are separate from the
uninstrumented wall-time medians above.

To reproduce the small native-runner measurements:

```bash
make compile
python benchmarks/ci_architecture.py --n 100000 --output-directory benchmark-results
```

This runs the original public API once, then `auto` with all outputs and
`fast` with selected PSD, each with one and the available number of workers
and three ordinary repetitions per configuration. Each optimized
configuration also has a separate profiled call. All use the same seeded
float64 Series and spectral parameters. The run's JSON artifacts contain wall and
CPU times, per-frequency phase data, build reports and the runner environment.
Input generation and result-file I/O are outside the API timers. There is no
hard timing gate and no saved signal array. Summed worker phase times overlap
and must not be interpreted as total wall time. These small shared-runner
measurements are evidence for that machine and commit, not a comparison of
processor families or a replacement for application-sized benchmarks.
