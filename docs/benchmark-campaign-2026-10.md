# LPSD / FFTW benchmark campaign — October 2026

Help compare native **LPSD/LNSD**, direct **FFTW**, and **FFTW matched to LNSD smoothness and peak width** on your hardware. The benchmark generates synthetic signals, records complete PSD + NSD calculation times and numerical checks, and keeps results local by default. You can optionally share a sanitized report through a GitHub issue; **no repository push permission, fork, or pull request is needed to contribute measurements**.

This campaign uses commit [`6246f0c6061b5f72dba8e235e640fb0b71281a20`](https://github.com/MichaDit/lpsd/commit/6246f0c6061b5f72dba8e235e640fb0b71281a20) from **[draft PR #1](https://github.com/MichaDit/lpsd/pull/1)**. The PR remains unmerged. The commands below download that exact commit and work before the PR is merged.

## Run with one command

Start in a writable directory. On the supported desktop/Linux routes, the command downloads private, pinned dependencies and builds the native implementation. It needs no administrator rights, Git installation, or preinstalled Python. The installation and results stay in the printed `lpsd-benchmark...` directory.

**Linux x86-64 / ARM64 and macOS:**

```sh
curl --proto '=https' --proto-redir '=https' --tlsv1.2 --connect-timeout 20 --max-time 60 -fsSL https://raw.githubusercontent.com/MichaDit/lpsd/6246f0c6061b5f72dba8e235e640fb0b71281a20/scripts/run_benchmark.sh | sh -s -- --revision 6246f0c6061b5f72dba8e235e640fb0b71281a20 --point-factors 1 2
```

**Windows PowerShell 5.1 or PowerShell 7:**

```powershell
& ([scriptblock]::Create((Invoke-RestMethod -TimeoutSec 60 'https://raw.githubusercontent.com/MichaDit/lpsd/6246f0c6061b5f72dba8e235e640fb0b71281a20/scripts/run_benchmark.ps1'))) -Revision '6246f0c6061b5f72dba8e235e640fb0b71281a20' -BenchmarkArgs @('--point-factors','1','2')
```

The wrapper and source use the same full commit. Review the [pinned instructions and scripts](https://github.com/MichaDit/lpsd/blob/6246f0c6061b5f72dba8e235e640fb0b71281a20/docs/portable-benchmark.md) before executing downloaded code if desired.

## What the campaign measures

| Method | Estimator |
|---|---|
| `lpsd` | Native bounded LPSD, with LNSD derived from PSD |
| `fftw` | FFTW periodogram with logarithmic aggregation |
| `matched` | FFTW with frequency-dependent smoothing and low-frequency LPSD to approximate LNSD smoothness and peak width |

The default sizes are **100,000; 1,000,000; 1,000,003; 10,000,000; 30,000,000; 100,000,000; and 1,000,000,000 samples**. The full profile checks six signal presets at the three smaller sizes and white noise at the larger sizes. Cases run serially, using the same frequency plan, float64 synthetic input, and float32 PSD/NSD outputs.

`--point-factors 1 2` requests the baseline and **twice the actual number of returned frequency points**. It resolves the native planner parameter `J` separately and records achieved counts; a planner plateau remains visible. Simply changing `J` from 1,000 to 2,000 does not ensure twice as many outputs. To add a further useful comparison, replace the argument with `--point-factors 1 2 4`.

Each `compute_wall_s` / `compute_cpu_s` pair describes **one complete call producing both PSD and NSD**, including planning and working arrays. `input_setup_s` includes input generation and its initial checksum. `process_wall_s` covers the full worker lifetime, including imports, setup, computation, and final checks. Integrity outcomes are recorded as flags, hashes, and quality metrics; there is no separate final-check timer. The matrix takes one observation per case; repeat a campaign to assess variation. Forecasts, skips, and timeouts are labelled separately from completed measurements. The report also identifies the actual FFTW version and binary hash.

## Choose a 32 GB or smaller budget

The default derives RAM, disk, and thread limits from the machine. An explicit RAM setting is an **allocation/work ceiling**, reduced when physical memory, cgroup limits, available memory, or the OS reserve require it.

| Requested ceiling | Argument | Bytes before reserves |
|---|---|---:|
| **32 decimal GB** | `--ram-gib 29.8023223876953125` | 32,000,000,000 |
| **32 GiB** | `--ram-gib 32` | 34,359,738,368 |
| Small ARM system | `--ram-gib 0.75 --disk-gib 2 --threads 2 --max-hours 6` | Budgets are checked on that device |

**32-decimal-GB campaign, including local spectra for plotting — Linux/macOS:**

```sh
curl --proto '=https' --proto-redir '=https' --tlsv1.2 --connect-timeout 20 --max-time 60 -fsSL https://raw.githubusercontent.com/MichaDit/lpsd/6246f0c6061b5f72dba8e235e640fb0b71281a20/scripts/run_benchmark.sh | sh -s -- --revision 6246f0c6061b5f72dba8e235e640fb0b71281a20 --ram-gib 29.8023223876953125 --point-factors 1 2 --save-spectra
```

**The same 32-decimal-GB campaign — Windows:**

```powershell
& ([scriptblock]::Create((Invoke-RestMethod -TimeoutSec 60 'https://raw.githubusercontent.com/MichaDit/lpsd/6246f0c6061b5f72dba8e235e640fb0b71281a20/scripts/run_benchmark.ps1'))) -Revision '6246f0c6061b5f72dba8e235e640fb0b71281a20' -BenchmarkArgs @('--ram-gib','29.8023223876953125','--point-factors','1','2','--save-spectra')
```

For **32 GiB**, replace the value after `--ram-gib` with `32`. The admitted budget and measured process memory in your report describe the actual experiment. These commands configure new measurements; this announcement contains no measured 32-GB performance claim.

### Memory, disk, and time handling

With automatic settings, RAM admission tries a **full window with separate FFT input/output**, then a **full window with an in-place FFT**, then **an in-place FFT with a bounded window**. The last mode regenerates window coefficients in blocks of at most 262,144 samples and avoids one separate 8N-byte window allocation—8 decimal GB at one billion samples. The original samples, frequency plan, and normalization are preserved. This window mode is for one complete computation per prepared object; reusable estimators retain their full-window default.

The buffer estimates include an empirical FFTW planning allowance. **They are admission estimates, not guaranteed peak RSS.** The selected FFT layout, window mode, estimates, and measured memory are recorded. If RAM admission fails, the runner considers a bounded disk path; cases that cannot fit are skipped.

Temporary datasets use a private directory under the original OS/user temporary directory. For large cases, choose a local SSD or another local drive with `--work-dir /path/on/local-drive/lpsd-work`. Avoid synchronized folders, whose background copies can consume space and alter timings. Detected Linux `tmpfs`/`ramfs` workdirs disable the disk fallback before dataset files are created; RAM cases remain available. The private installation has a separate default **2 GiB setup cap**.

Once the downloaded wrapper starts, the maximum is **24 hours for bootstrap setup plus the benchmark**; `--max-hours` can shorten it. The initial wrapper request has a separate 60-second client timeout and precedes this setup/run budget. Installation, compilation, optional login, data generation, computation, and checks consume that budget. Larger cases are admitted using conservative forecasts from smaller completed cases, and owned worker processes have timeouts. Completed results are checkpointed. Long-running cases print the elapsed process time and effective process limit once per minute; these are not completed compute timings. The deadline cannot keep a suspended or powered-off device running.

## Platform coverage

| Environment | Route |
|---|---|
| Linux x86-64 / ARM64, glibc 2.28+ | Automatic private installation; both architectures have portable CI coverage |
| Raspberry Pi / Rockchip RK3399 | Use a compatible **64-bit Linux** installation and suitable RAM/disk limits; board-specific performance still needs device measurements |
| macOS Apple Silicon, 14+ | Native ARM64 automatic route with portable CI coverage |
| macOS Intel, 13+ | Automatic route based on the pinned package tags; not covered by this campaign's four portable CI jobs |
| Windows x64 | Automatic private installation with portable CI coverage |
| Windows 11 ARM64 | Deliberate x64 emulation for this dependency set; host and process architectures are reported separately |
| Android / Termux ARM64 | **Experimental prepared-environment route:** existing compatible Python, NumPy, pandas, SciPy, psutil, Clang, and FFTW are required |

Termux does not use the desktop Linux wheels or automatically build the scientific stack. Follow the [pinned Termux instructions](https://github.com/MichaDit/lpsd/blob/6246f0c6061b5f72dba8e235e640fb0b71281a20/docs/portable-benchmark.md#android--termux); stock Android and 32-bit systems are outside this automatic route. Android may terminate long-running processes.

For this commit, the [native test/build workflow](https://github.com/MichaDit/lpsd/actions/runs/38197747608) and [portable bootstrap workflow](https://github.com/MichaDit/lpsd/actions/runs/38197747490) passed all 12 jobs. The portable jobs cover Linux x64, Linux ARM64, macOS ARM64, and Windows x64, including numerical smoke runs with the bounded window. CI does not constitute a billion-sample run on every platform.

## Share a result without repository push access

Read the [pinned privacy notice](https://github.com/MichaDit/lpsd/blob/6246f0c6061b5f72dba8e235e640fb0b71281a20/docs/benchmark-privacy.md) first. To opt into a public GitHub issue, add **both** `--upload-github --accept-public-upload`.

**Automatic submission with an existing GitHub CLI login — Linux/macOS:**

```sh
curl --proto '=https' --proto-redir '=https' --tlsv1.2 --connect-timeout 20 --max-time 60 -fsSL https://raw.githubusercontent.com/MichaDit/lpsd/6246f0c6061b5f72dba8e235e640fb0b71281a20/scripts/run_benchmark.sh | sh -s -- --revision 6246f0c6061b5f72dba8e235e640fb0b71281a20 --point-factors 1 2 --upload-github --accept-public-upload
```

**Windows:**

```powershell
& ([scriptblock]::Create((Invoke-RestMethod -TimeoutSec 60 'https://raw.githubusercontent.com/MichaDit/lpsd/6246f0c6061b5f72dba8e235e640fb0b71281a20/scripts/run_benchmark.ps1'))) -Revision '6246f0c6061b5f72dba8e235e640fb0b71281a20' -BenchmarkArgs @('--point-factors','1','2','--upload-github','--accept-public-upload')
```

The bootstrap reuses an installed `gh`, or obtains a pinned private copy when available. To explicitly request a new browser/device login, also add `--github-login`—as another shell argument, or as `'--github-login'` in the PowerShell array. You complete that authorization using your own GitHub account. Login time counts against the deadline.

If authentication or connectivity is unavailable, results and prepared public bodies remain local. You can review `results.github-issue.md` and paste it into [a new GitHub issue](https://github.com/MichaDit/lpsd/issues/new). To prepare public content after a local-only run, use the retained private Python from the retained source directory with `-m lpsd_benchmark submit /path/to/results.json`; this command alone does not publish.

When `--save-spectra` is selected, every checkpoint also writes a compact **`results.public.json`** containing the exact allowlisted report without spectra. Automatic submission uses this file. Select it explicitly for later manual submission if the larger local spectra report exceeds the unchanged 16 MiB reader limit. The uploader does not silently substitute a different input file.

Large reports are compressed and, when necessary, divided into **one issue and numbered comments on that same issue**, with at most 16 parts of 60,000 bytes each. No cases are silently omitted. Every part identifies the same complete-payload SHA256; the documented offline procedure reconstructs the exact public JSON. The uploader prepares all public bodies and comment requests before its noninteractive authentication check and any result upload. An explicitly requested `--github-login` is a separate earlier bootstrap step; it can run before the measurements and report preparation.

A multipart receipt records confirmed and uncertain parts. If a response is lost, inspect the existing issue and its digest/index markers before manually adding any missing parts. Repeating the same local multipart submission does not automatically recreate, retry, or continue it. A completed local calculation does not depend on successful publication.

**A public issue is associated with your GitHub account.** The allowlisted payload excludes hostnames, usernames, paths, network identifiers, tokens, logs, raw signals, and saved spectra. This reduces disclosed data; it does not make the issue anonymous or provide a blanket GDPR/DSGVO guarantee. GitHub also processes account and connection data. The privacy notice explains the remaining collection-operator responsibilities. Publish only the prepared report, not the installation directory or terminal logs.

The benchmark issue form is included in PR #1 and will appear after it reaches the repository's default branch. **Ordinary issues and the explicit CLI submission work now.** Code changes remain subject to PR review; participants can contribute measurements without receiving repository write permission.
