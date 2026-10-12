# Cross-platform follow-up — fast.8

This follow-up builds on qualified revision
`8bfbf2b0648e6b328df04be434c376dbc8de71a4` (M). It combines recovered
measurement evidence with bounded implementation changes and new paired
measurements on native CI runners. The
[recovered large-input report](performance-memory-points.md) preserves the
older matrix and experimental cohorts separately.

## Changes selected from the profiles

| Area | Change | Scope |
|---|---|---|
| Apple arm64 mean detrending | Form order-0 residuals inside the native dot product when `long double` and `double` have the same precision and exponent range | Removes residual-vector writes/reads for eligible PSD/CSD; keeps the serial mean and residual rounding |
| Windows/macOS concurrency | Query actual available physical memory | Replaces the fixed 2-GiB assumption when the OS query succeeds; explicit working limits still apply |
| Even-length disk FFT | Optional original-N real-input packing into an N/2 complex FFT | `--disk-fft-algorithm packed-real`; full complex FFT remains the default |
| Native file-cache headroom | Optional cap on the existing working allowance | `--native-working-mib 384` is an explicit profile, not an automatic speed recommendation |
| Windows bootstrap | Tolerate a child path disappearing while a temporary directory is traversed | Permission, I/O, root-directory and resource-budget errors still fail |
| Disk diagnostics and cleanup | Handle zero spectral energy; use an exclusively owned workspace subdirectory | Zero/constant signals complete and pre-existing caller files survive success and partial setup failures |
| Disk power output | Reject independently mapped aliases and hardlinks to the transform workspace | Prevents an external power destination from corrupting unread transform data |

Linux AArch64 retains its already selected double-double preparation where
binary128 `long double` is expensive. Windows retains the residual path after
the earlier projected DC-plus-nanovolt failure. The public `auto` precision
choice is unchanged. See [platform arithmetic and memory details](platforms.md).

## Numerical qualification

The recovered packed-real prototype completed 48 main comparisons and 562
small contract checks. Under its original **3-ppm relative / zero absolute**
gate, 44 of 48 main comparisons passed; four simple-FFTW off-bin cases failed.
Those historical flags remain unchanged. Re-evaluation against the requested
**1% relative** limit passed all 37,656 PSD values and the same number of NSD
values. The largest relative differences were 0.151066% PSD and 0.075562% NSD,
with maximum absolute differences of 4.814825e-35 V²/Hz and
1.694066e-21 V/√Hz. There were no zero mismatches or nonfinite values.

The integrated adapter shares the existing complex disk FFT stages, retains
the original-N window/grid/normalization, and reconstructs powers in bounded
scratch. New tests compare actual transforms with NumPy, original-N Parseval,
all six canonical signals, odd-length fallback, tiny lengths, DC/Nyquist,
explicit inner factorizations, small reconstruction tiles, input/refill
lifetime, and overlapping or independently mapped output files. The existing
full-FFT numerical assertions are retained.

The active complex region is 8N bytes instead of 16N. The file remains 16N
bytes for original input/window staging, so input plus workspace remains
24N bytes. Large-length scalar admission checks are not large-run timings.

The separate Bluestein prototype also satisfies the requested 1% tolerance
in the recovered screen, with 0.101458% maximum relative PSD error and
0.050739% NSD error at very small nulls. Its existing optional RAM adapter
remains separate; this revision does not turn it into an automatic route.

## Validation before native CI

On the integration Linux x86-64 host, the complete `test_fast` plus upstream
`test` run passed **2,112 tests**, with five unavailable-platform/optional
skips and two documented legacy expected failures. This includes actual small
full/packed disk pipelines, both estimators, protected caller files, zero
energy, portable admission/forecast separation and sanitized report fields.

The inline residual arithmetic additionally passed a forced FP64-long-double
x86 build with 648 projection comparisons and complete PSD/CSD output checks.
That surrogate verifies arithmetic, not Apple hardware speed. Native Apple
execution is evaluated separately below.

## Native revision comparisons

The CI harness builds M and the candidate with the same compiler,
interpreter, dependencies and build policy. It alternates baseline/current,
current/baseline, baseline/current in fresh processes for N=100,000 and
1,000,000, workers 1 and up to 4, `auto` with all outputs and `fast` with
PSD+NSD. Each sample includes a complete public API call; imports, input
creation, a small 4,096-sample warmup and output hashing are excluded. Raw
reports bind the loaded binary to source/compiler hashes and compare every
output byte, including the frequency grid.

The native matrix covers Linux x86-64, Linux ARM64 and macOS ARM64. Small
shared-runner observations are reported by configuration and never pooled
across different CPUs. Three repetitions and their ranges identify obvious
variation; they do not establish a universal performance guarantee.

**Candidate measurements are pending the new CI run.** No native platform
speedup is claimed from a forced ABI build or an earlier runner's timing.

## Full versus packed-real comparisons

The portable CI uses its freshly bootstrapped Python/compiler/FFTW stack on
Linux x86-64, Linux ARM64, macOS ARM64 and Windows x64. On each platform,
`benchmarks/bench_disk_algorithms.py` alternates three full/packed pairs per
configuration at N=100,000 and 1,000,000, both FFTW estimators and both actual
output counts, with two admitted workers. Every complete PSD+NSD call includes
fresh FFTW planning, preprocessing, transform, bounded energy diagnostics,
output construction and workspace cleanup. Input creation, hashes and imports
are excluded. The FFTW binary, input and resource allowances are identical
within each pair. CPU and wall time are reported separately.

Every PSD/NSD point must satisfy the 1% gate; the report also preserves the
stricter 3-ppm result and maximum absolute/relative errors. Speed is not a CI
pass/fail condition. Input hashes are checked again after both algorithms.
Small file-backed CI cases mainly exercise algorithm and platform behavior;
they do not measure a billion-sample transform that exceeds RAM.

**Four-platform paired measurements are pending the new CI run.** The small
local development smoke is not used as a performance claim.

For an existing built checkout, reproduce a comparison with:

```sh
python benchmarks/bench_disk_algorithms.py --sizes 100000 1000000 --threads 2 --repeats 3 --point-factors 1 2 --max-seconds 600 --output disk-algorithm-pairs.json
```

For supervised large cases, use the
[portable runner and explicit disk profile](portable-benchmark.md#real-input-disk-fft).
Its process deadline and admission guards remain in force. Physical
RPi/RK3399/Android devices and an actual 32-GB host are not represented by
these CI runner measurements.
