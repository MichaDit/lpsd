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
but are linked as Mach-O dynamic libraries. Universal2 and cross-compilation
are not validated by the supplied CI.

### Optional x86 runtime dispatch

For x86 ELF targets with glibc 2.23 or newer, the fast builder compiles and
links a `target_clones("default", "avx2", "avx512f")` probe. Success defines
`LPSD_HAVE_TARGET_CLONES=1`, which enables the corresponding guarded native
annotations. The native implementation applies them to `fast_dft_impl` and
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
| Local Linux x86-64/GCC build used to validate these build changes | 64 significand bits, stored in 16 bytes |
| AArch64 using the standard AAPCS64 mapping | IEEE binary128, 113 significand bits, 16 bytes |
| Apple arm64 | Same representation and precision as `double`, 53 significand bits, 8 bytes |

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
mode retains mode 2 for order-0 projection, and the original residual
detrending and scalar paths remain available. Mode 3 does not extend the
experimental order-1 projector.

An x86 build forced to use software binary128 is an arithmetic/build
surrogate. Its timings are not ARM measurements and do not establish an ARM
speedup. Native Linux AArch64 measurements must show whether preparation
improves, while the numerical gates check the resulting spectra. Apple
Silicon requires its own measurements and validation as well.

## Native CI matrix and small measurements

The workflow is configured to execute numerical tests on these native runners.
**The new Linux ARM64, Apple Silicon and Clang jobs are still awaiting their
first successful native CI run at this documentation update.** The table
describes the configured checks, not completed platform validation.

| Runner | Compiler | Python | Numerical tests | Installed wheel | Small benchmark |
|---|---|---|---|---|---|
| `ubuntu-24.04`, x86-64 | GCC | 3.10, 3.12 | Both | 3.12 | 3.12 |
| `ubuntu-24.04`, x86-64 | Clang | 3.12 | Yes | No | No |
| `ubuntu-24.04-arm`, AArch64 | GCC | 3.12 | Yes | Yes | Yes |
| `macos-15`, Apple Silicon | Apple Clang | 3.12 | Yes | Yes | Yes |

These labels and architectures are documented in
[GitHub's hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
The workflow asserts the interpreter's architecture, builds both backends,
runs the existing numerical gates, and tests wheels installed outside the
checkout. No platform-specific relaxation of the numerical assertions is
introduced here. A changed expected-failure outcome requires review on the
actual platform.

The new build path was locally checked on Linux x86-64 with GCC 13.3.0:
93 existing tests passed, two known numerical/statistical limits remained
expected failures, three new build-policy checks passed, and the sdist-built
wheel passed its isolated installation smoke test. This describes the build
change's local validation before dispatch, compensated preparation and other
feature branches were integrated. These historical counts do not validate
the later native changes or the additional architectures. Successful native
jobs and their artifacts are required before reporting those results.

To reproduce the small native-runner measurements:

```bash
make compile
python benchmarks/ci_architecture.py --n 100000 --output-directory benchmark-results
```

This runs the original public API once, then `auto` with one and the available
number of workers, three ordinary repetitions per configuration. Each fast
configuration also has a separate profiled call. All use the same seeded
float64 Series and spectral parameters. The JSON artifacts contain wall and
CPU times, per-frequency phase data, build reports and the runner environment.
Input generation and result-file I/O are outside the API timers. There is no
hard timing gate and no saved signal array. Summed worker phase times overlap
and must not be interpreted as total wall time. These small shared-runner
measurements are evidence for that machine and commit, not a comparison of
processor families or a replacement for application-sized benchmarks.
