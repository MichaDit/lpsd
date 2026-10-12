# Portable LPSD / FFTW benchmark

The benchmark builds the native LPSD implementation for the current machine,
compares it with the direct FFTW and LNSD-matched FFTW estimators, and writes a
local JSON report. A public GitHub issue is optional. Contributors need no
repository push permission, fork or pull request to report measurements.

## Start from a pinned revision

Use the **full 40-character commit from the benchmark release or reviewed PR**
for `COMMIT` below. The wrapper and downloaded source must use the same commit.
These are command templates; `COMMIT` is deliberately not a moving branch name.
For a published benchmark campaign, its announcement should supply the complete
command with that value already filled in.

Linux x86-64 / ARM64 and macOS, from a writable directory:

```sh
curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/MichaDit/lpsd/COMMIT/scripts/run_benchmark.sh | sh -s -- --revision COMMIT
```

Windows, in Windows PowerShell 5.1 or PowerShell 7:

```powershell
& ([scriptblock]::Create((Invoke-RestMethod 'https://raw.githubusercontent.com/MichaDit/lpsd/COMMIT/scripts/run_benchmark.ps1'))) -Revision 'COMMIT'
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
sh scripts/run_benchmark.sh --revision COMMIT
```

```powershell
& ./scripts/run_benchmark.ps1 -Revision 'COMMIT'
```

## Choose resource and time budgets

The default is at most **24 hours for setup plus the benchmark**, with RAM,
thread and temporary-disk budgets derived from the current machine. Appending
these arguments changes the run:

| Goal | Arguments |
|---|---|
| Compare with a 32 GiB allocation ceiling | `--ram-gib 32` |
| Compare with a 32 decimal GB allocation ceiling | `--ram-gib 29.8023223876953125` |
| Small ARM machine, shorter run | `--ram-gib 0.75 --disk-gib 2 --threads 2 --max-hours 6` |
| Fast installation and numerical smoke test | `--sizes 100000 --profile performance --threads 2 --max-hours 0.25` |
| Baseline, double and quadruple output counts | `--point-factors 1 2 4` |
| RAM and disk paths, when both fit | `--storage both` |
| Explicit FFT buffer comparison | `--fft-layout in-place` or `--fft-layout out-of-place` |
| Preserve local spectra for plotting | `--save-spectra` |
| Put dataset files on a selected local drive | `--work-dir /path/on/local-drive/lpsd-work` |
| Use an installed compatible compiler only | `--compiler system` |
| Exercise the private compiler fallback | `--compiler zig` |

For example, the Unix command with a 32 GiB allocation ceiling is:

```sh
curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/MichaDit/lpsd/COMMIT/scripts/run_benchmark.sh | sh -s -- --revision COMMIT --ram-gib 32 --point-factors 1 2
```

For PowerShell, pass additional arguments as an array:

```powershell
& ([scriptblock]::Create((Invoke-RestMethod 'https://raw.githubusercontent.com/MichaDit/lpsd/COMMIT/scripts/run_benchmark.ps1'))) -Revision 'COMMIT' -BenchmarkArgs @('--ram-gib','32','--point-factors','1','2')
```

### What the limits mean

`--ram-gib` is a conservative **allocation/work budget**, clamped to detected
physical memory, cgroup limits and currently available memory. It cannot make
32 GiB available on a smaller machine. One GiB is 1,073,741,824 bytes:
32 GiB is 34,359,738,368 bytes, whereas 32 decimal GB is 32,000,000,000 bytes
(29.8023223876953125 GiB). Neither flag reserves an exact physical partition.
The JSON records the admitted budget after OS reservations and rounding to
whole MiB, together with measured process memory.

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

The outer bootstrap sets an absolute deadline; installing Python, downloading
packages, compiling, optional login, importing libraries, generating data and
checking results all consume that same budget. The Python stages also use a
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

That command prepares the exact issue body without posting. Once you choose
to publish, use a logged-in `gh` and append `--accept-public-upload`, or post
the prepared body in your browser. On Windows the private interpreter is
`venv\Scripts\python.exe`. If a submission attempt times out without a
verified issue URL, check existing issues before retrying to avoid duplicates.

## Development and CI

`--source-dir /path/to/checkout` explicitly copies a local tree into the private
environment. It is for PR/CI validation before the commit is downloadable and
does not claim that uncommitted source is a clean remote revision. Normal
contributors should use the commit-pinned download route.

```sh
sh scripts/run_benchmark.sh --revision COMMIT --source-dir /path/to/checkout --compiler zig --sizes 100000 --profile performance --threads 2 --max-hours 0.25
```

The bootstrap unit tests cover archive traversal/link rejection, checksum and
size failures, private installer environment, deadline termination, consent
gating and mocked GitHub setup. Tests never create public benchmark issues.
Windows/macOS/ARM compatibility should be reported from the corresponding CI
or physical-device run, not inferred from a successful Linux import.
