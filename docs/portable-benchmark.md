# Portable LPSD / FFTW benchmark

The benchmark builds the native LPSD implementation for the current machine,
compares it with the direct FFTW and LNSD-matched FFTW estimators, and writes a
local JSON report. A public GitHub issue is optional. Contributors need no
repository push permission, fork or pull request to report measurements.

## Start from a pinned revision

The commands below pin the **fast.10 implementation**,
`e0426c4bad9d7c5fccd54de935eea2d1ea1b2519`. The
[fast.10 qualification](fast10-validation.md) records the statistics correction,
portable diagnostics, tests and performance comparisons. The
[device rerun guide](device-rerun-fast10.md) keeps the existing device budgets
and separates the small preflight from the previously problematic large cases.

Both the wrapper and the downloaded source use the same full commit. When
testing another revision, change both occurrences in the download command.
The publication's documentation follow-up does not change the implementation;
the qualification manifest binds its code and tests by SHA-256.
Historical [fast.9 performance results](performance-fast9.md),
[physical-device reports](device-tests-20261012.md) and the
[recovered evidence addendum](benchmark-final-evidence-20261012.md) keep their
original revisions and estimator modes.

Linux x86-64 / ARM64 and macOS, from a writable directory:

```sh
curl --proto '=https' --proto-redir '=https' --tlsv1.2 --connect-timeout 20 --max-time 60 -fsSL https://raw.githubusercontent.com/MichaDit/lpsd/e0426c4bad9d7c5fccd54de935eea2d1ea1b2519/scripts/run_benchmark.sh | sh -s -- --revision e0426c4bad9d7c5fccd54de935eea2d1ea1b2519
```

Windows, in Windows PowerShell 5.1 or PowerShell 7:

```powershell
& ([scriptblock]::Create((Invoke-RestMethod -TimeoutSec 60 'https://raw.githubusercontent.com/MichaDit/lpsd/e0426c4bad9d7c5fccd54de935eea2d1ea1b2519/scripts/run_benchmark.ps1'))) -Revision 'e0426c4bad9d7c5fccd54de935eea2d1ea1b2519'
```

The command creates a fresh `lpsd-benchmark...` directory under the current
directory. Set `LPSD_BENCHMARK_BASE_DIR` to another existing writable directory
when desired. It downloads a private Python, scientific packages and, if no
compatible compiler is available, a private C compiler. Administrator rights,
Git and a preinstalled Python are not required by these desktop commands.
No shell profile or global package environment is changed. Downloading and
executing the wrapper trusts the selected repository revision and HTTPS
distribution; inspect the script first if that does not match your workflow.

For an already downloaded checkout, the equivalent commands are:

```sh
sh scripts/run_benchmark.sh --revision e0426c4bad9d7c5fccd54de935eea2d1ea1b2519
```

```powershell
& ./scripts/run_benchmark.ps1 -Revision 'e0426c4bad9d7c5fccd54de935eea2d1ea1b2519'
```

## Select the statistics contract

The default is `--statistics corrected`. Native LPSD averages every actual
segment and accumulates its sample variance. Matched FFTW uses this mode for
its native low-frequency subset; its FFT smoothing is unchanged. Pure FFTW
has no native segment recurrence and records `statistics=not_applicable`.
All cases also record the CLI's `requested_statistics`.

Use `--statistics legacy` for direct comparisons with fast.1–fast.9 or the
frozen `lpsd` reference. The two modes define different estimators: changing
the mean can exceed a 1% numerical tolerance, especially with few segments.
Comparison groups and native/matched timing calibrations keep the modes
separate. Missing mode fields in old reports retain their historical legacy
meaning; old reports do not become corrected evidence.

The API's `ps_std` and `psd_std` use a nominal independent-segment SEM, are
undefined (`NaN`) for one segment, and do not compensate for overlap or time
correlation. The portable benchmark requests PSD and NSD only, so no variance
work is performed inside those complete-call timers. See the
[full statistics contract](statistics.md).

## Choose resource and time budgets

The default is at most **24 hours from wrapper start for setup plus the
benchmark**, with RAM, thread and temporary-disk budgets derived from the current
machine. The initial wrapper request has a separate 60-second client timeout
and precedes this setup/run budget. Appending
these arguments changes the run:

| Goal | Arguments |
|---|---|
| Current corrected native estimator (default) | `--statistics corrected` |
| Reproduce historical native statistics | `--statistics legacy` |
| Compare with a 32 GiB allocation ceiling | `--ram-gib 32` |
| Compare with a 32 decimal GB allocation ceiling | `--ram-gib 29.8023223876953125` |
| Small ARM machine, shorter run | `--ram-gib 0.75 --disk-gib 2 --threads 2 --max-hours 6` |
| Fast installation and numerical smoke test | `--sizes 100000 --profile performance --threads 2 --max-hours 0.25` |
| Baseline, double and quadruple output counts | `--point-factors 1 2 4` |
| RAM and disk paths, when both fit | `--storage both` |
| Explicit FFT buffer comparison | `--fft-layout in-place` or `--fft-layout out-of-place` |
| Explicit one-shot window memory comparison | `--window-storage full` or `--window-storage fft_input_then_blocked` |
| Explicit matched-FFTW smoothing parallelism | `--threads 8 --smoothing-workers 8` |
| Cap native LPSD working memory, including the hybrid's low frequencies | `--native-working-mib 384` |
| Compare the real-input disk FFT route | `--storage disk --disk-fft-algorithm packed-real` |
| Experimental direct logarithmic power reduction | `--methods fftw --storage disk --disk-fft-algorithm full --disk-power-reduction direct-log` |
| Preserve local spectra for plotting | `--save-spectra` |
| Put dataset files on a selected local drive | `--work-dir /path/on/local-drive/lpsd-work` |
| Use an installed compatible compiler only | `--compiler system` |
| Exercise the private compiler fallback | `--compiler zig` |

For example, the Unix command with a 32 GiB allocation ceiling is:

```sh
curl --proto '=https' --proto-redir '=https' --tlsv1.2 --connect-timeout 20 --max-time 60 -fsSL https://raw.githubusercontent.com/MichaDit/lpsd/e0426c4bad9d7c5fccd54de935eea2d1ea1b2519/scripts/run_benchmark.sh | sh -s -- --revision e0426c4bad9d7c5fccd54de935eea2d1ea1b2519 --ram-gib 32 --point-factors 1 2
```

For PowerShell, pass additional arguments as an array:

```powershell
& ([scriptblock]::Create((Invoke-RestMethod -TimeoutSec 60 'https://raw.githubusercontent.com/MichaDit/lpsd/e0426c4bad9d7c5fccd54de935eea2d1ea1b2519/scripts/run_benchmark.ps1'))) -Revision 'e0426c4bad9d7c5fccd54de935eea2d1ea1b2519' -BenchmarkArgs @('--ram-gib','32','--point-factors','1','2')
```

### What the limits mean

The experimental `direct-log` reduction is available in **fast.9**. It applies
to the simple full-complex disk FFT with the original input length, the fixed
50 Hz / K=100 / PSLL=200 profile and a 128 MiB FFT tile. It returns only the
owned logarithmic group arrays instead of materializing the full one-sided
power vector. Its independent energy traversal, original float32 PSD and
complex64-square-root NSD remain in the complete-call timer. The full FFT
matrix and the input still occupy **24N logical file bytes** together; this
option does not change the disk-space admission reserve. Small-plan or
insufficient-workspace cases remain ineligible. It does not select disk
storage by itself and does not change the matched or native estimators.

The default remains `materialized`. A separate Linux 100M paired comparison
at source `ca2d26f9` measured **6.89% / 10.08% less complete-call wall time**
at 704 / 1,408 points, with 6,336 PSD and 6,336 NSD comparisons bit-identical.
The two variants shared their FFT geometry, buffers and dependency versions.
This is a three-pair result on one host; the input/workspace fit in memory.
The acceptance gate remains `rtol=3e-6, atol=0`. See the
[direct profile, complete measurements and reproduction](direct-log-power.md).
The separate recovered R3 qualification passed the same all-point tolerance
on its small corpus. The last external one-billion-sample attempt completed
only its reference: the unchanged disk-space check refused R3 before it
started. Therefore **no one-billion Direct-log time or speedup is available**;
see the [historical evidence addendum](benchmark-final-evidence-20261012.md).

The supervisor counts known data files in owned immediate `fft-work-*`
directories as well as the input. It does not recursively scan the selected
drive. If budget-relevant live memory readings remain unavailable for one
second while a child is still running, it stops and reaps that child and
records `memory_measurement_unavailable`. A high-water mark collected after
exit is a separate observation; it does not stand in for live monitoring.

On macOS the live reader deliberately uses the complete process-tree RSS and
does not query USS through `memory_full_info()`. The [psutil changelog](https://github.com/giampaolo/psutil/blob/master/HISTORY.rst)
describes how the older USS query can require extra privileges or hang on
headless systems (issue 2885); the pinned reader avoids that query entirely.
For disk cases, the same RAM ceiling is therefore checked conservatively
against RSS, with `live_memory_rss_fallback_samples` recording the actual
number of such probes. RSS includes file mappings and can reject a large case
earlier than a private-memory policy would. Missing or incomplete RSS still
triggers the unchanged availability guard; no private value is invented.
Linux and Windows retain their existing anonymous/private-memory policies.

`--smoothing-workers` selects the worker limit for the `matched` method in both
RAM and disk mode. It accepts a positive integer below 2**63 and is capped by
the thread count admitted through affinity, CPU quota and `--threads`. For
example, requesting eight smoothing workers on a machine admitted for two
uses a limit of two. Other methods do not use this option. Without it, the
historical default remains **min(4, admitted threads)**; there is no automatic
search or change to that default.

The report distinguishes the explicit request, admitted smoothing limit,
configured executor capacity, generated task ranges and smoothed point count.
These are execution settings, not a measurement of how many threads ran at
the same instant. A serial RAM smoother has executor capacity zero and runs
its range on the caller thread. Existing `actual_threads` describes the FFTW
backend (or the native worker setting), not all live threads in the process.

Each smoothing worker above min(4, admitted threads) adds an **empirical
9 MiB incremental runtime/address reserve** to the resource model. Eight
instead of four therefore accounts for 36 MiB inside the existing budget.
This is extra headroom for thread mappings and runtime state; the shared
N-dependent FFT arrays are not multiplied by the number of workers. It is
neither a measured RSS increment nor a universal stack-size bound. The
`incremental_9mib_above_cap4_v1` model identifier and reserved byte count are
recorded explicitly. Larger platform-specific stack requirements can still
trigger a guard or allocation failure.

The same reserve is included when choosing storage, FFT layout and window
storage, and removed from the FFT/native working allowances before allocation.
It can make a previously marginal case choose a smaller layout, disk storage,
or skip for insufficient memory. RAM, cgroup and Linux address-space limits
are not raised, and the program does not silently switch back to four workers.
The complete-call timer and 24-hour setup/run deadline still cover the whole
selected computation. When comparing reports, include the smoothing limit
and reserve model along with the implementation and point counts.
A controlled 4/8-worker experiment with a fixed native working allowance
is a separate configuration: this option can change that allowance or the
selected layout, so the experiment's speedup does not predict every portable
case.

`--native-working-mib INTEGER` optionally caps the native working allowance
for `lpsd` and the native low-frequency part of `matched`, in RAM and disk
mode. It accepts at least **64 MiB**; one MiB is exactly 1,048,576 bytes.
Without this option, the existing automatic allowance is preserved.
The simple `fftw` method reports `not_applicable` and receives no native cap.

The effective allowance is the smallest of the requested cap, the existing
automatic allowance, and the remaining native headroom after input and
runtime reserves. A `384` request can therefore admit only 256 MiB on a
smaller machine. If the 64-MiB minimum or the actual planned buffers cannot
fit, the case is skipped before sample/scratch allocation. The cap is used
by both admission preview and the actual computation, and the capacity
returned by the native estimator is verified after the complete-call timer.

This option changes the native pool and potentially the selection of
streamed coefficients. It keeps the input samples, original frequency plan,
window/normalization definitions and output formats. A lower cap can improve
file-cache headroom, but it can also add coefficient regeneration and limit
concurrency; **a smaller cap is not automatically faster**. The cap is neither
the allocated scratch-pool size nor a total-process RSS limit. It does not
increase FFT buffers, admitted threads, total RAM/disk allowances, or OS
limits, and the existing conservative storage/layout/window selection stays
in force.

Reports distinguish `requested_native_working_mib`, the
`native_working_auto_limit_bytes` before the explicit cap, the admitted
`native_working_limit_bytes`, and `native_working_applied_bytes` read from a
completed native computation. An empty native subset does not invent an
applied capacity. `native_working_mb` retains its historical name but uses
MiB and agrees with the effective allowance. The closed policy is `auto`,
`explicit_cap`, or `not_applicable`. Older reports retain absent provenance.
Forecasts for an explicit cap use only completed cases with the same request
and admitted byte limit; auto/2048-MiB results cannot silently calibrate a
384-MiB profile. Keep separate reports when comparing caps, for example:

```sh
python -m lpsd_benchmark run --profile performance --methods lpsd matched --storage disk --native-working-mib 384 --point-factors 1 2 --output native384.json
```

The same argument passes through the Linux/macOS, PowerShell and prepared
Termux bootstrap routes. Existing setup/run deadlines and small calibration
cases still apply; no automatic cap search or extra large test is started.

### Real-input disk FFT

`--disk-fft-algorithm packed-real` explicitly selects a real-input transform
for the `fftw` and `matched` disk methods. For even N >= 4, adjacent real
samples form an N/2-point complex transform, followed by bounded reconstruction
of the original positive-frequency powers. The input samples, original-N
frequency grid, window sums, PSD/NSD normalization and matched smoothing are
preserved. Odd N and N=2 fall back to the established full complex transform.
RAM methods and native `lpsd` do not use this option. The default remains
`--disk-fft-algorithm full`.

The active complex region shrinks from 16N to 8N bytes. The backing file still
has **16N bytes** because it stages the original input and window, and the
input plus workspace still require **24N bytes**. Reconstruction reuses bounded
FFTW scratch, with no extra sample-sized anonymous array. Thus this option
reduces transform work and the active complex region; it does not halve the
total disk budget. The same conservative resource admission and process
supervision remain in force.

Floating-point rounding can differ, especially at tiny spectral nulls. The
original qualification requires every PSD and NSD value to satisfy
`rtol=3e-6, atol=0`; four off-bin comparisons in the earlier packed corpus
failed that gate. A separate existing platform report also evaluates a 1%
relative threshold. That alternate assessment does not authorize relaxing
the original limit or turn its failures into passes. The later controlled
100M white-noise pairs are bit-identical, but they do not cover those off-bin
failures or establish a one-billion-sample packed result. See the
[qualification and platform measurements](performance-platform-followup.md)
and the [final evidence addendum](benchmark-final-evidence-20261012.md).

Use small **disk** calibration cases before larger packed cases; full-FFT or
RAM timings do not calibrate this algorithm. For example:

```sh
python -m lpsd_benchmark run --profile performance --methods fftw matched --storage disk --sizes 100000 1000000 10000000 --disk-fft-algorithm packed-real --point-factors 1 2 --output packed-real.json
```

`disk_fft_algorithm` records the request, while
`disk_fft_effective_algorithm` and `disk_fft_inner_samples` are read back from
a completed disk computation. These fields make the odd-N fallback explicit.
The sanitized report accepts only the documented choices and integer lengths.
The same argument passes through the supported bootstrap wrappers.

### Allocation and disk budgets

`--ram-gib` is a conservative **allocation/work budget**, clamped to detected
physical memory, cgroup limits and currently available memory. It cannot make
32 GiB available on a smaller machine. One GiB is 1,073,741,824 bytes:
32 GiB is 34,359,738,368 bytes, whereas 32 decimal GB is 32,000,000,000 bytes
(29.8023223876953125 GiB). Neither flag reserves an exact physical partition.
The JSON records the admitted budget after OS reservations and rounding to
whole MiB, together with measured process memory.

Automatic RAM admission first tries the full window with separate FFT input
and output, then an in-place FFT with the full window. If neither fits, it
tries an in-place one-shot path: the existing FFT input initially holds the
window for unchanged normalization, then window coefficients are regenerated
in blocks of at most 262,144 samples while preparing the input. That removes
one separate 8N-byte window allocation (8 decimal GB at N=1 billion). The
original samples, frequency plan and complete-call timing scope are preserved.
This mode supports exactly one computation per prepared object. Normal reusable
estimator objects retain their full-window default.

RAM estimates also include an **empirical 8N-byte FFTW planning allowance**,
motivated by separate 10M/30M phase-memory measurements with FFTW 3.3.10. This
allowance is an admission estimate, not a proven upper bound for every FFTW
build or prime transform length. The JSON records the selected layout, window
mode, known window buffers and planning allowance separately from measured RSS.
The bounded-window scalar model admits both FFTW variants at one billion white
samples on an otherwise available 32-decimal-GB machine after the existing OS
reserve; actual execution still depends on measured free memory and guards.

The runner estimates the complete working set before allocating a case,
monitors workers, and uses an additional Linux address-space guard where
applicable. Setup also uses a conservative memory ceiling and watches the
installer/compiler process tree; package installation limits concurrent
downloads and unzip threads. A sampled process monitor is not a kernel-enforced physical RAM
partition. Mapped files, the OS page cache and aliased mappings require care
when comparing resident-memory figures. For experiments requiring an identical
physical 32 GiB limit, provide that limit through the operating system or a
container and record it separately. The program does not change OS memory or
swap settings.

`--disk-gib` limits the temporary dataset and FFT workspace files. By default,
the runner creates its own temporary work directory under the original OS/user
temporary directory, independently of the retained installation and results.
Each case's files are removed when that case finishes; the owned temporary work
directory is also removed when the runner exits, including handled deadlines,
interrupts and Python exceptions. An explicit `--work-dir` takes precedence;
only the case directories created by the runner are removed from that parent.

For large datasets, select a local SSD or another local drive with sufficient
free space through `--work-dir`. Avoid synchronized folders: their background
copies can consume additional storage and distort I/O timings. On Linux and
small ARM boards, `/tmp` may be a RAM filesystem. Detected `tmpfs` and `ramfs`
work directories receive a zero disk-workspace budget, so disk fallbacks are
skipped **before creating dataset files**; small RAM cases remain available.
The disk fallback needs storage on a real drive, not another allocation of the
same RAM. A custom RAM disk on another platform also needs a suitably chosen
local-drive `--work-dir`. See the kernel's
[tmpfs documentation](https://docs.kernel.org/filesystems/tmpfs.html).

The default private installation has a separate **2 GiB cap**, adjustable with
`--bootstrap-disk-gib`. The setup checks free space and archive expansion; an
unavailable package or a setup budget that is too small stops installation
with a local diagnostic. Installation consumes space before the runner derives
its scratch-disk budget. The retained environment and results can be removed by
deleting their own benchmark directory after saving any desired report.

An input of one billion float64 samples alone occupies 8,000,000,000 bytes.
Its disk-backed transform needs additional workspace; lowering RAM does not
make that disk cost disappear. Cases that do not fit are explicitly skipped.
`--storage auto` first considers RAM, including the smaller in-place FFT layout
when needed, then a bounded disk path. The selected path is included in each
result. An in-place FFT reuses its transform buffer while preserving the
original input samples.

For native LPSD with a file-backed input, an internally owned anonymous
coefficient pool of at least 512 MiB may shrink once during the call. The
adapter waits for the complete prefix of frequency points to finish, releases
the old pool and allocates the space required by the remaining points when
their largest concurrent stored lengths fit within 128 MiB. The same point
plan, streaming selection, gate capacity and arithmetic continue afterward.
Ordinary RAM inputs and caller-owned scratch mappings retain their lifecycle.
The wait, release and new allocation remain inside the complete-call timer.

The report records initial, final and peak pool capacity and whether and where
the replacement actually occurred. The original admission estimate is retained.
These byte counts describe allocations during the call; physical page return
depends on the allocator and OS. A smaller final pool does not itself establish
lower peak RSS or a speed gain. Compare completed times and observed memory
under the same input, storage and resource settings.

The downloaded shell or PowerShell wrapper sets an absolute deadline when it
starts. Installing Python, downloading packages, compiling, optional login,
importing libraries, generating data and checking results all consume that
same budget. The Python stages also use a
monotonic clock and terminate owned child processes at the deadline. Completed
cases are checkpointed. A clock or OS scheduling delay, device suspension,
external termination or forced power-off is not under application control.
The 24-hour setting is a maximum permitted run, not a promise that an Android
device or laptop will stay awake for that duration.

Before starting a larger case, the controller forecasts its whole-process
time from completed smaller cases, with a safety factor. An unknown machine
first receives a bounded small calibration case. Cases that are predicted to
exceed the remaining time are skipped; a forecast that proves too optimistic
is bounded by its own timeout and the global deadline. Forecasts, skipped rows
and timeouts are never presented as completed measurement times.

## What is compared

The default dataset lengths are:

```text
100,000; 1,000,000; 1,000,003; 10,000,000; 30,000,000; 100,000,000; 1,000,000,000
```

The non-round length exercises a different FFT factorization. Each admitted
case uses float64 synthetic input, the same frequency plan, a Kaiser-window
configuration with `psll=200`, `n_averages=100`, and float32 PSD/NSD outputs.
The preset sample rate is 50 Hz. The methods are:

| Method | Calculation |
|---|---|
| `lpsd` | Native bounded LPSD, with NSD derived from PSD |
| `fftw` | Direct FFTW periodogram with logarithmic aggregation |
| `matched` | FFTW with frequency-dependent smoothing and low-frequency LPSD, selected to approximate LNSD smoothness and peak width |

Matching smoothness and peak width does not make two different estimators
mathematically identical. The report therefore includes noise-normalized
roughness, peak-width measurements for tone presets, and spectral differences
relative to LPSD where comparable spectra are available. It also checks
nonnegative finite PSD, the PSD-to-NSD relation, frequency labels and unchanged
input data. See [the estimator comparison](fftw-comparison.md) for context.

The default `--point-factors 1 2 --base-points 1000` doubles the **actual number
of returned frequency points** relative to the native planner's baseline.
The native parameter `J=1000` often returns fewer than 1,000 points. The runner
resolves a new `J` for the requested output-count factor and records the
baseline count, requested count, selected `J` and actual count. If a planner
plateau prevents an exact count, the achieved count remains visible.
`--points 1000 2000` instead compares raw `J` targets and is mutually exclusive
with `--point-factors`; it does not promise twice as many outputs.

The `full` profile adds pink noise, brown noise, mixed tones, an off-bin tone
and a DC/nanovolt case to the white-noise case at the three historical quality
sizes, selected with `--quality-sizes 100000 1000000 1000003`. Large default
cases use white noise so the quality matrix does not multiply every expensive
billion-sample run. `--profile performance` uses white noise throughout.
The program reads no user measurement files.

### Read the timings correctly

| Field | Scope |
|---|---|
| `compute_wall_s` | One complete estimator invocation producing both PSD and NSD, including its planning, working arrays and output construction |
| `compute_cpu_s` | CPU time consumed by that invocation; multiple threads can make it exceed wall time |
| `input_setup_s` | Synthetic input creation and its initial integrity hash |
| `process_wall_s` | Isolated worker lifetime, including imports, setup, computation and checks |
| `phases` | Available finer timings, particularly for disk-backed calculations |
| `predicted_process_s` | Admission forecast only |

PSD and NSD share a calculation. Do not add their timings as if they were
independent complete runs, or compare these complete calls with a cached,
already-planned raw FFT timer. The portable matrix records one completed call
per case; repeat the campaign when estimating run-to-run variation.

The actual FFTW library version and binary hash are recorded independently
of `pyfftw`. A wrapper version is not an FFTW version: for example, the
verified Linux pyFFTW 0.15.1 wheel used during development contains FFTW 3.3.5.
The loader can also use an installed FFTW library. Use `--fftw-library` and
`--threads-library` to select explicit existing binaries. Results using
different library hashes or thread counts should remain identifiable.

### Failure, memory and thread diagnostics

An `invalid_result` now retains all five checks: finite spectrum, nonnegative
PSD, exact PSD-to-NSD relation, unchanged input and the expected frequency
plan. The `validation` object contains bounded error counts and finite error
maxima. Hashes identify the result without publishing its arrays.

`fftw_threading` distinguishes the request from the selected backend's thread
count. It records the selected sources/runtime and closed load, symbol and
initialization failure categories. Discovery tries system pthread and OpenMP
runtimes and compatible wheel companions before a one-thread fallback. A
selected thread count is configuration, not a measurement of simultaneous
CPU activity; the new discovery does not establish a speedup on an untested
device.

`memory_guard` records the actual RSS or anonymous/private metric compared to
the budget, its last/peak sample, and the value and metric causing termination.
It also records available process-tree coverage and source information. This
can differ from `peak_rss_bytes`, which retains its separate RSS meaning.
Incomplete live reads are not silently counted as zero. The setup guard also
rejects incomplete process-tree reads; its pre-Python Linux probe caches only
a verified target PID and revalidates it on subsequent samples. These remain
sampled observations, with the existing budgets, intervals and deadlines.

The same bounded diagnostic fields survive the public export; paths, raw
exceptions and spectrum arrays do not. See the [export schema](benchmark-privacy.md).

## Platforms and dependency integrity

The automatic path pins Python 3.12.14, uv 0.13.0, scientific wheel versions
and optional Zig 0.15.2. It verifies tool-archive SHA256 values and installs
only hashes admitted by `scripts/benchmark-requirements.lock`, without
unbounded source builds of NumPy, pandas or SciPy. Source downloads identify
one full commit. The local `bootstrap-local.json` records the source-archive
hash, lock hash, setup time, setup size and sampled setup memory; it is not an upload attachment.

| Environment | Route and limits |
|---|---|
| Linux x86-64, glibc 2.28+ | Automatic private Python/wheels/compiler; exercised with the Linux integration smoke |
| Linux ARM64, glibc 2.28+ | Same automatic route; suitable in principle for a 64-bit Raspberry Pi or RK3399 Linux installation; board-specific performance and thermal behavior need device results |
| macOS Intel, 13+ | Automatic route based on the selected wheel tags; CI is the platform gate |
| macOS Apple Silicon, 14+ | Native ARM64 automatic route based on the selected wheel tags; CI is the platform gate |
| Windows x64 | Automatic route; private compiler when GCC/Clang is absent; Windows CI is the platform gate |
| Windows 11 ARM64 | Deliberate **x64 emulation** for the pinned package set; host and process architecture are reported separately |
| Android/Termux ARM64 | Experimental prepared-environment route below; no claim of desktop-wheel or stock-Android support |
| 32-bit Python/OS, Windows 10 ARM64, older macOS, other libc/ABI | Outside this automatic distribution's validated package set; do not silently substitute another architecture |

The [uv platform policy](https://docs.astral.sh/uv/reference/policies/platforms/)
describes its own support tiers. The
[pyFFTW wheel matrix](https://github.com/pyFFTW/pyFFTW#wheels) and
[published wheel files](https://pypi.org/project/pyFFTW/0.15.1/#files) determine
the narrower scientific-package requirements. In particular, the selected
pyFFTW/pandas set has no native Windows ARM64 route. Microsoft's
[ARM emulation documentation](https://learn.microsoft.com/en-us/windows/arm/apps-on-arm-x86-emulation)
distinguishes Windows 11 x64 emulation from Windows 10 ARM support.

### Android / Termux

Python's [Android documentation](https://docs.python.org/3.14/using/android.html)
explains why Android is not an ordinary desktop Python target. Use a genuine
Termux environment installed according to its
[official installation instructions](https://github.com/termux/termux-app#installation).
Android uses a different C-library ABI from glibc desktop Linux; downloading
the Linux ARM64 wheels does not establish compatibility.

This wrapper uses Termux's **existing** Python and system-site packages and
checks that `numpy`, `pandas`, `scipy` and `psutil` import successfully. A
compatible Termux `clang` and FFTW must already be available. NumPy, SciPy and
psutil have official Termux package definitions; pandas availability must be
resolved for that device before running this path. The bootstrap does not
silently launch a large scientific source build or install OS packages.

With that prepared environment, the Unix command works with small limits,
for example `--ram-gib 0.75 --disk-gib 2 --max-hours 6 --threads 2`. The CLI
skips large cases that cannot fit. Choose a local, non-synchronized filesystem
for `--work-dir` when the OS temporary directory is RAM-backed or too small.
Missing Termux `gh` leaves a browser-submittable
issue draft. Android may terminate long-running or CPU-heavy processes; the
Termux project documents additional Android 12+ limitations. Native Linux on
the same ARM board is the more directly supported benchmark target.

## Optional public result submission

Read [the privacy notice](benchmark-privacy.md) before publishing. Add both
`--upload-github --accept-public-upload` to opt into a public issue on
`MichaDit/lpsd`. The bootstrap reuses an installed `gh`, or downloads a
SHA-pinned private GitHub CLI when missing. An existing login is reused.

To request browser/device login explicitly during setup, also add
`--github-login`. This step is interactive and uses your own GitHub account;
the benchmark cannot complete GitHub consent on your behalf. Login time
counts against the total deadline. The login's Git configuration is redirected
to the private benchmark directory, and no SSH keys are generated or uploaded.
The CLI uses the OS credential store when available; GitHub documents a
plaintext configuration-file fallback when it is not. Treat that private
directory as potentially containing credentials and do not upload it.

Without successful authentication or connectivity, the benchmark keeps its
results and prepares `results.github-issue.md`. Open
[a new issue](https://github.com/MichaDit/lpsd/issues/new), review that file and
paste its contents into the issue body. Use a title such as
`Benchmark: Linux aarch64`. This needs a GitHub account, not push permission.

To review before deciding to upload, run locally first. In the retained source
directory, use its private Python:

```sh
/path/to/lpsd-benchmark/venv/bin/python -m lpsd_benchmark submit /path/to/results.json
```

With `--save-spectra`, each checkpoint also writes a compact `results.public.json`
containing exactly the allowlisted report, without spectra. Automatic submission
uses that explicit file. To submit it manually, select it yourself:

```sh
/path/to/lpsd-benchmark/venv/bin/python -m lpsd_benchmark submit /path/to/results.public.json
```

The submit reader retains its **16 MiB input-file limit**. A local report with
saved spectra can exceed it; use the generated `.public.json` instead. The CLI
never silently substitutes a sibling file for the path you selected. Both
`results.json` and `results.public.json` prepare `results.github-issue.md`.
The compact file still passes through the sanitizer and the unchanged issue-size
limit. Creating it is local-only and does not select public upload.

That command prepares the exact public content without posting. Once you choose
to publish, use a logged-in `gh` and append `--accept-public-upload`, or post
the prepared body in your browser. On Windows the private interpreter is
`venv\Scripts\python.exe`. If a submission attempt times out without a
verified issue URL, check existing issues before retrying to avoid duplicates.

### Large reports: one issue and its comments

The existing single-body format is retained whenever the complete sanitized
report fits, including its existing gzip/base64 fallback. If that still exceeds
**60,000 bytes**, submission prepares one issue body plus numbered comments.
Every part stays within the same 60,000-byte limit. There are at most **16 parts
total: one issue and up to 15 comments**. The 16 MiB input-file and 512-case
schema limits remain unchanged. The uploader rejects a report that cannot fit
those limits before its CLI discovery, noninteractive authentication check or
result transfer; no cases are dropped.

All parts contain one SHA256 for the **entire exact uncompressed JSON payload**
and their own index/total. The gzip stream is created once, encoded as base64,
then split into chunks. A chunk alone is not a standalone gzip file.

For a three-part report, local preparation produces:

- `results.github-issue.md` — part 1, used as the issue body.
- `results.github-issue.part-02-of-03.md` — part 2, used as the first comment.
- `results.github-issue.part-03-of-03.md` — part 3, used as the second comment.

The printed ordered list is authoritative; unrelated older files are not
deleted. The additional `*.request.json` files contain only the corresponding
public comment body in the REST request wrapper. The uploader prepares every
Markdown body and request file before its noninteractive authentication check
or any result transfer. Installation downloads and an explicitly requested
`--github-login` are separate earlier bootstrap steps.
The same explicit upload flags authorize this public report, including its
comments. Creating these local files alone does not authorize publication.

Automatic submission creates the issue first and verifies its repository URL.
Comments use the documented
[`gh api --input`](https://cli.github.com/manual/gh_api) route to
[create issue comments](https://docs.github.com/en/rest/issues/comments#create-an-issue-comment).
Each returned JSON receipt must contain a positive comment ID, that comment's
exact URL in the confirmed issue, the matching API issue URL, and the exact
prepared body. Requests are serial, with a pause between POSTs charged to the
same upload deadline, following
[GitHub's guidance](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api#pause-between-mutative-requests).
Unverified response fields and logs are not included in the report.

**An issue and its comments are not an atomic transaction.** The program reports
the confirmed prefix, the last part whose acceptance is uncertain, and parts
not attempted. A local `*.submission.json` receipt is claimed exclusively before
the first POST and checkpointed before/after subsequent POSTs. It is keyed to the
entire payload SHA256. Another submit of that same payload reads the saved state;
it does not recreate the issue, resend a part, or automatically continue.
Incomplete/unreadable receipts also stop publication instead of being removed.

For manual publication, paste part 1 into one new issue and the remaining parts
as comments in that issue. If any upload was attempted, first inspect the
existing issue and its SHA256/index markers: confirmed parts are already public,
and an unconfirmed part may also have reached GitHub. Add only genuinely missing
parts to that same issue. When the first issue receipt itself is uncertain,
check the repository's issues before creating anything. The program never
automatically retries an uncertain POST.

### Offline reconstruction for collection operators

Save the issue body and matching part comments as local UTF-8 Markdown files.
Use only parts with the same payload SHA256 and total, collect exactly each
index once, order their base64 chunks by index, concatenate, decode base64 once,
and decompress the single gzip stream. Verify SHA256 over the resulting **exact
JSON bytes before parsing or reformatting them**.

The following optional offline example accepts only explicitly named local
files. It reads at most 60,001 bytes per input, accepts at most 16 parts, and
limits decompression to 16 MiB plus one byte for overflow detection. It rejects
missing/duplicate/mixed parts and trailing or multiple gzip streams. It does not
follow any URL or execute any text from a report. Save it locally as
`reconstruct_parts.py`; it is not another benchmark command or network collector.

```python
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import zlib

output, *inputs = sys.argv[1:]
if not 2 <= len(inputs) <= 16:
    raise ValueError("Supply exactly the local files for one multipart report")
chunks, digests, totals = {}, set(), set()
for name in inputs:
    with Path(name).open("rb") as stream:
        data = stream.read(60_001)
    if len(data) > 60_000:
        raise ValueError("Part exceeds 60,000 bytes")
    text = data.decode("utf-8").replace("\r\n", "\n")
    hashes = re.findall(r"^Payload SHA-256: `([0-9a-f]{64})`$", text, re.M)
    labels = re.findall(r"^Payload part: `([0-9]{1,2})/([0-9]{1,2})`$", text, re.M)
    blocks = re.findall(r"```text\n([A-Za-z0-9+/=]+)\n```", text)
    if len(hashes) != 1 or len(labels) != 1 or len(blocks) != 1:
        raise ValueError("Invalid multipart framing")
    index, total = map(int, labels[0])
    if not 1 <= index <= total <= 16 or index in chunks:
        raise ValueError("Invalid or duplicate part index")
    chunks[index] = blocks[0]
    digests.add(hashes[0])
    totals.add(total)
if totals != {len(inputs)} or len(digests) != 1:
    raise ValueError("Missing or mixed payload parts")
if set(chunks) != set(range(1, len(inputs) + 1)):
    raise ValueError("Missing part index")
encoded = "".join(chunks[i] for i in range(1, len(inputs) + 1))
compressed = base64.b64decode(encoded, validate=True)
decoder = zlib.decompressobj(wbits=31)
payload = decoder.decompress(compressed, 16 * 1024 ** 2 + 1)
if (len(payload) > 16 * 1024 ** 2 or not decoder.eof
        or decoder.unused_data or decoder.unconsumed_tail):
    raise ValueError("Oversized, incomplete, or multiple gzip streams")
if hashlib.sha256(payload).hexdigest() != next(iter(digests)):
    raise ValueError("Payload SHA256 mismatch")
json.loads(payload)  # Check JSON syntax; do not reserialize before hashing.
with Path(output).open("xb") as stream:
    stream.write(payload)
```

For a three-part report:

```sh
python reconstruct_parts.py reconstructed.public.json results.github-issue.md results.github-issue.part-02-of-03.md results.github-issue.part-03-of-03.md
```

The example refuses to overwrite an existing output file. Reconstruction checks
completeness and byte integrity; numerical/schema validation and trust in a
contributor's claimed measurements remain separate.

## Development and CI

`--source-dir /path/to/checkout` explicitly copies a local tree into the private
environment. It is for PR/CI validation before the commit is downloadable and
does not claim that uncommitted source is a clean remote revision. Normal
contributors should use the commit-pinned download route.

```sh
sh scripts/run_benchmark.sh --revision e0426c4bad9d7c5fccd54de935eea2d1ea1b2519 --source-dir /path/to/checkout --compiler zig --sizes 100000 --profile performance --threads 2 --max-hours 0.25
```

The bootstrap unit tests cover archive traversal/link rejection, checksum and
size failures, private installer environment, deadline termination, consent
gating and mocked GitHub setup. Tests never create public benchmark issues.
Windows/macOS/ARM compatibility should be reported from the corresponding CI
or physical-device run, not inferred from a successful Linux import.
