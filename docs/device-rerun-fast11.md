# Staged device checks for fast.11

Use **`d63457f81183c68c0759c6e3999b07e72bcc82cc`** for both the downloaded wrapper and source. This
guide prepares new physical runs; no fast.11 device result is claimed here.
See the [fast.11 changes and local validation](performance-fast11.md) and the
[review of the retained device data](platform-followup-20261020.md). The
[fast.10 campaign](device-tests-fast10-20261013.md) retains its original source,
failures, skips and measurements.

## 1. Regression validation and the existing budgets

From a complete checkout of the pinned source, with the test dependencies and
supported compiler available, first run:

```sh
python scripts/validate_local.py
```

This builds both native libraries and runs the fast and upstream suites
locally. The same command works in a configured Windows Python environment.
Review failures, skips and expected failures separately; see
[local validation](local-validation.md).

| Device | RAM ceiling | Dataset/disk ceiling | Baseline requested threads |
|---|---:|---:|---:|
| Windows x64 | 32 GiB | 16 GiB | 16 |
| ARM64 SBC, Raspberry Pi 4 | 0.375 GiB | 2 GiB | 2 |
| ARM64 instrument, existing prepared Debian chroot on Android | 0.75 GiB | 2 GiB | 2 |

The detected resources can reduce these ceilings. Use the instrument's
existing prepared Linux environment; these commands do not install a stock
Android or Termux environment. Dataset files need a real local filesystem;
add `--work-dir` for an existing local-drive directory if the default temporary
directory is RAM-backed. An outer supervisor must use the intended benchmark
allowance, not an unrelated smaller regression ceiling.

## 2. Small preflight: 12 planned comparison cases

The commands below keep reports local and create a fresh private benchmark
environment for each invocation. Setup and execution share each requested
time budget. In the Unix example, select **0.375 for the SBC or 0.75 for the
instrument** before running the commands:

```sh
lpsd_revision=d63457f81183c68c0759c6e3999b07e72bcc82cc
lpsd_ram_gib=0.375
lpsd_run() {
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 --connect-timeout 20 --max-time 60 -fsSL \
        "https://raw.githubusercontent.com/MichaDit/lpsd/$lpsd_revision/scripts/run_benchmark.sh" |
        sh -s -- --revision "$lpsd_revision" --statistics corrected \
            --ram-gib "$lpsd_ram_gib" --disk-gib 2 "$@"
}
lpsd_run --sizes 100000 --point-factors 1 2 --storage both \
    --profile performance --threads 2 --max-hours 0.5
```

Windows PowerShell 5.1 or 7:

```powershell
$lpsdRevision = 'd63457f81183c68c0759c6e3999b07e72bcc82cc'
$lpsdWrapper = [scriptblock]::Create((Invoke-RestMethod -TimeoutSec 60 "https://raw.githubusercontent.com/MichaDit/lpsd/$lpsdRevision/scripts/run_benchmark.ps1"))
$lpsdCommon = @('--statistics','corrected','--ram-gib','32','--disk-gib','16')
& $lpsdWrapper -Revision $lpsdRevision -BenchmarkArgs ($lpsdCommon + @('--sizes','100000','--point-factors','1','2','--storage','both','--profile','performance','--threads','16','--max-hours','0.5'))
```

The smoke matrix has three methods, RAM/disk and two point factors:
**12 planned cases**. Require `state=completed`, a record for each planned
case, and successful completion of every admitted case. The six RAM cases
must complete before proceeding. Examine all retained validity checks and
selected FFTW threads.

The SBC's effective budget can fall below the **384-MiB disk admission
minimum**. Six disk `memory_budget` skips can then be expected; retain them
as skips and record that disk execution was not qualified. Do not require
12 successful cases by raising the existing budget or relabelling skipped
rows. Dependency skips, failed cases and other incomplete coverage require
their own diagnosis. A report's `completed` state alone is insufficient.

## 3. Repeat the one-/two-/four-worker series

Use the same shell/session definitions above. Each invocation uses white RAM
input at N=1,000,000, point factor one, and automatically adds its small
calibrations. Keep the three reports separate.

```sh
for lpsd_threads in 1 2 4; do
    lpsd_run --sizes 1000000 --point-factors 1 --storage ram \
        --profile performance --threads "$lpsd_threads" --max-hours 0.5
done
```

```powershell
foreach ($lpsdThreads in @(1,2,4)) {
    & $lpsdWrapper -Revision $lpsdRevision -BenchmarkArgs ($lpsdCommon + @('--sizes','1000000','--point-factors','1','--storage','ram','--profile','performance','--threads',"$lpsdThreads",'--max-hours','0.5'))
}
```

This directly revisits the instrument's seven forecast-based skips. A
smaller forecast does not itself count as completed work: retain actual
process and compute times, output counts, hashes and any remaining skips.
The 1/2/4 requests are explicit experimental settings; the two-thread ARM
baseline remains unchanged for the following continuation.

## 4. Targeted larger continuation

For the SBC and instrument, these two separate runs revisit the interrupted
or failed 30M cases using the default full algorithm and then packed-real:

```sh
lpsd_run --sizes 100000 1000000 10000000 30000000 --point-factors 1 2 \
    --methods fftw matched lpsd --storage auto --disk-fft-algorithm full \
    --profile performance --threads 2 --max-hours 24
lpsd_run --sizes 100000 1000000 10000000 30000000 --point-factors 1 2 \
    --methods fftw matched --storage disk --disk-fft-algorithm packed-real \
    --profile performance --threads 2 --max-hours 24
```

The instrument's eight historical stops were 30M FFTW/matched disk cases.
Inspect the actual storage route, selected FFTW runtime and memory guard
metric/trigger. Accessible old-kernel `smaps` counters can avoid conservative
RSS fallback; inaccessible counters retain that fallback and may still stop
a case at the existing limit. A RAM case and a disk case are different
performance configurations.

For Windows, the routine continuation ends at **100M samples**:

```powershell
& $lpsdWrapper -Revision $lpsdRevision -BenchmarkArgs ($lpsdCommon + @('--sizes','100000','1000000','1000003','10000000','30000000','100000000','--point-factors','1','2','--storage','auto','--profile','performance','--threads','16','--max-hours','24'))
```

These are targeted performance continuations, not repetitions of every
quality-signal matrix. The separate regression suites exercise numerical
contracts. The historical Windows 1B restart/bugcheck remains unresolved and
outside this routine continuation.

## 5. Evidence to return

Return each run's already sanitized **`results.json`**, or
**`results.public.json`** when `--save-spectra` was used. Check the recorded
source commit, `native_version=1.0.6+fast.11`, native/library hashes, corrected
statistics for native/matched, actual output count, thread selection,
complete/failed/skipped rows and guard evidence. Installed distribution
metadata alone does not identify a tested native binary.

Raw regression logs and JUnit are local diagnostic material, outside the
portable public-report schema. No upload or GitHub Actions run is part of
these commands. Keep results distinct by device, run, method, point count,
storage, libraries, threads and budgets; a new completion does not alter a
historical failure.
