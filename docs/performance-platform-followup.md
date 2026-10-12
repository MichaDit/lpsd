# Cross-platform follow-up — fast.8

This follow-up builds on qualified revision
`8bfbf2b0648e6b328df04be434c376dbc8de71a4` (M). It combines recovered
measurement evidence with bounded implementation changes and new paired
measurements on native CI runners. The
[recovered large-input report](performance-memory-points.md) preserves the
older matrix and experimental cohorts separately.

Validated fast.8 runtime: **`2e215275124d0ffa05e314d7eddf18ecab5a1ae1`**.
All eight [native test/build jobs](https://github.com/MichaDit/lpsd/actions/runs/38218181121)
and all four [portable platform jobs](https://github.com/MichaDit/lpsd/actions/runs/38218181042)
passed. The final documentation/evidence update changes no executable code.
[Full platform tables, ranges and sources](platform-comparison-20261012.md)
and [machine-readable evidence](../benchmarks/results/platform-comparison-20261012.json)
retain every measured configuration, repetition and artifact identity.

## Changes selected from the profiles

| Area | Change | Scope |
|---|---|---|
| Apple arm64 mean detrending | Form order-0 residuals inside the native dot product when `long double` and `double` have the same precision and exponent range | Removes residual-vector writes/reads for eligible PSD/CSD; keeps the serial mean and residual rounding |
| Windows/macOS concurrency | Query actual available physical memory | Replaces the fixed 2-GiB assumption when the OS query succeeds; explicit working limits still apply |
| Even-length disk FFT | Optional original-N real-input packing into an N/2 complex FFT | `--disk-fft-algorithm packed-real`; full complex FFT remains the default |
| Native file-cache headroom | Optional cap on the existing working allowance | `--native-working-mib 384` is an explicit profile, not an automatic speed recommendation |
| Windows bootstrap | Tolerate a child path disappearing while a temporary directory is traversed | Permission, I/O, root-directory and resource-budget errors still fail |
| Windows disk cleanup | Run optional allocator trimming only on Linux; share the helper with tuning | Fixes the first real Windows matched-disk call and preserves the conservative RAM fallback if the OS query fails |
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

## Validation

On the integration Linux x86-64 host, the complete `test_fast` plus upstream
`test` run passed **2,112 tests**, with five unavailable-platform/optional
skips and two documented legacy expected failures. This includes actual small
full/packed disk pipelines, both estimators, protected caller files, zero
energy, portable admission/forecast separation and sanitized report fields.

The inline residual arithmetic additionally passed a forced FP64-long-double
x86 build with 648 projection comparisons and complete PSD/CSD output checks.
That surrogate verifies arithmetic, not Apple hardware speed. Native Apple
execution is evaluated separately below.

The final Linux x64 and ARM64 CI suites each report **2,105 optimized tests
plus 18 upstream tests passed**, with six skipped/expected-failure entries.
The FFTW and disk subset gates overlap that optimized suite. macOS reports
1,728 optimized plus 18 upstream tests passed; its 383 skipped/expected-failure
entries mostly reflect the unavailable optional system FFTW library. The
portable macOS environment supplies FFTW and executes the disk comparisons.
Windows completes the real watchdog cases, the OS memory query without a
fallback, all three ordinary/bounded-window methods and both disk algorithms.
The two documented inherited numerical/statistical expected failures remain;
this work does not redefine the upstream uncertainty estimator.

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

All **24 native configurations and 72 baseline/current pairs** completed
with identical returned outputs. At N=1M, all Linux x64/ARM64 native median
changes are below 1%; these screens show no material regression.

The final macOS `auto` case with one worker has medians 4.458334 → 4.483215 s
and ranges 4.256393–4.957041 → 3.879336–5.081961 s. With three workers it is
1.963399 → 1.911285 s, again with overlapping ranges. The new inline residual
path is confirmed active. Two earlier separate runs observed one-worker
reductions of 14.06% and 8.08%, but the final cohort does not replicate a clear
percentage gain. The unchanged `fast` control also varies substantially.
These cohorts are not pooled. The removed residual storage and numerical
identity are established; a stable general macOS speedup is not.

## Complete 100-million-sample disk comparison

The [complete local paired report](../benchmarks/results/packed-real-disk-100m-20261012.json)
records four configurations and three alternating pairs each: **24 complete
PSD+NSD calls**. The six recorded source hashes match candidate
`2afaac85462542276b04c85ed5f9788a65f780b5`. The later Windows cleanup fix
does not change these successful Linux calculations. This cohort uses Linux
x86-64, system FFTW 3.3.10, eight FFTW/native workers, four smoothing workers,
128 MiB FFT scratch and 384 MiB native working allowance. The host has an
8-GiB cgroup ceiling and eight CPU equivalents. The test's files fit within
that memory ceiling; this is a file-backed comparison, **not an out-of-RAM
or one-billion-sample measurement**.

| Method / actual points | Full median wall [s] | Packed median wall [s] | Ratio of medians | Full → packed median CPU [s] |
|---|---:|---:|---:|---:|
| Simple FFTW / 704 | 24.506037 | 10.330516 | 2.372× | 25.651442 → 11.002871 |
| Simple FFTW / 1408 | 23.452197 | 10.371538 | 2.261× | 22.806063 → 11.131750 |
| Matched / 704 | 43.441363 | 23.875725 | 1.819× | 93.045462 → 80.253379 |
| Matched / 1408 | 53.618029 | 32.849665 | 1.632× | 148.064119 → 130.607350 |

The observed wall-time ranges were 23.232–25.626 versus 10.255–13.840 s,
22.629–24.080 versus 10.296–13.406 s, 42.635–68.562 versus 23.490–40.181 s,
and 50.268–55.209 versus 31.249–36.456 s, in table order. Every range is
preserved with its individual sample. The first matched/base pair was
noticeably slower for both algorithms; it remains in the report. No cache
flush or exclusive host-I/O control was imposed. The three-pair medians show
a useful gain in this configuration, not a guarantee for other storage,
lengths or machines. CPU savings differ from elapsed-time savings.

All **12,672 PSD and 12,672 NSD point pairs** are bit-identical, as are the
frequency grids. The original plans use J=1000 for 704 points and J=2130 for
1408 points. The report records the common input hash; successful completion
also requires the final input rehash to agree. Raw spectra are not included,
so the archived equality evidence comes from the source-hashed harness's
full `tobytes()` comparison, not a later reconstruction from rounded values.

The scope includes fresh FFTW plans, original-N window/preprocessing, FFT,
ordered powers, Parseval diagnostics, smoothing/native low frequencies where
applicable, both outputs and workspace cleanup. Input creation, hashing,
imports and library loading are outside each call. The complete command was:

```sh
python benchmarks/bench_disk_algorithms.py --sizes 100000000 --threads 8 --fft-memory-mib 128 --native-working-mib 384 --repeats 3 --point-factors 1 2 --max-seconds 900 --work-dir /path/on/local-drive --output packed-100m-pairs.json
```

The native cap is the same on both sides and is not independently evaluated
by this experiment. No claim is made for the packed-real route at 1B or on
a 32-GB host.

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

All **32 disk configurations and 96 full/packed pairs** completed. PSD and
NSD outputs are bit-identical, frequency grids agree and both the 1% and
stricter 3-ppm checks pass. N=1M ratios of full/packed wall-time medians are:

| Platform | Simple / 577 points | Simple / 1154 | Matched / 577 | Matched / 1154 |
|---|---:|---:|---:|---:|
| Linux x86-64 | 0.985× | 0.978× | 0.996× | 1.004× |
| Linux ARM64 | 1.063× | 1.031× | 1.003× | 1.002× |
| macOS ARM64 | 0.947× | 1.121× | 1.009× | 1.109× |
| Windows x64 | 0.994× | 1.021× | 1.032× | 0.979× |

A ratio below one means packed is slower in that small case. ARM64's simple
FFT shows a modest repeatable benefit; other small-case results are mixed or
close to the observed scatter. Windows ranges all overlap. macOS varies
between runs. The [complete table](platform-comparison-20261012.md) includes
every range and N100k result in its linked JSON. These results support the
explicit option and the separate large-case experiment, not an automatic
switch for every length and platform.

For an existing built checkout, reproduce a comparison with:

```sh
python benchmarks/bench_disk_algorithms.py --sizes 100000 1000000 --threads 2 --repeats 3 --point-factors 1 2 --max-seconds 600 --output disk-algorithm-pairs.json
```

For supervised large cases, use the
[portable runner and explicit disk profile](portable-benchmark.md#real-input-disk-fft).
Its process deadline and admission guards remain in force. Physical
RPi/RK3399/Android devices and an actual 32-GB host are not represented by
these CI runner measurements.
