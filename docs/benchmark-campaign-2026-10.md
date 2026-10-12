# LPSD / FFTW benchmark campaign — October 2026

The recovered campaign contains **132 completed cases out of 132**, including
one billion samples and genuinely doubled output point counts. The
[performance and memory report](performance-memory-points.md) explains the
results; the [compact evidence JSON](../benchmarks/results/recovered-campaign-20261012.json)
records exact measurements, historical source revisions and artifact SHA256
identities. The historical measurements use E/G (`1.0.6+fast.7`). This document
also describes explicit controls retained in `1.0.6+fast.9`;
those controls do not turn the older observations into measurements of the
new revision.

The campaign compares native **LPSD/LNSD**, direct **FFTW**, and **FFTW matched
to LNSD smoothness and peak width**. It generates synthetic signals and keeps
results local by default. Participants can optionally share a sanitized
GitHub issue without repository push permission, a fork or a pull request.

## Read the completed results

All entries below are from the recovered Linux x86-64 environment with eight
CPU equivalents and an 8 GiB cgroup ceiling. They are not measurements on all
supported platforms or a physical 32 GB machine.

| Experiment | Completed observation | What it establishes |
|---|---|---|
| Main matrix | 132/132 cases; 42 white-noise cases across seven lengths | Complete joint PSD + NSD timings, numerical checks and actual point counts; one observation per case |
| Fresh native 1B E→G pair | 5233.983 s → 1570.826 s wall; identical spectra | Smaller retained native pool and lower I/O pressure; the 3.332× wall ratio is not an isolated causal speedup |
| Matched smoothing, four → eight workers | Complete-call median reductions of 14.25–18.19% at 30M/100M | A useful explicit configuration on this machine, with numerical identity; no universal worker default |
| Prime length 1,000,003 | White-noise medians: direct FFTW 218.224 → 123.357 ms; matched 325.038 → 262.702 ms | A qualified Bluestein candidate for this length and build |
| 1B disk columns, same factors | Auto 1094.373 s → strided 847.729 s; exact spectra | One complete ordered pair, 22.537% less wall time, but 3.529× as many process write bytes |

The native main matrix includes an anomalously faster doubled-point 1B case.
The separate E/G 100M counterpair also shows order-dependent variation. The
report retains these observations and their CPU/I/O evidence; it does not
conclude that more output points are intrinsically faster or attribute the
entire E/G wall ratio to releasing one buffer pool.

The historical strict numerical harness used `rtol=3e-6, atol=0`. Its failed
pure-tone gates for some Bluestein and Packed Real cases remain recorded as
failed. Reassessment against the campaign's approved **less-than-1%**
tolerance qualifies the stored errors: maxima are 0.101458% PSD / 0.050739%
NSD for Bluestein and 0.151066% PSD / 0.075562% NSD for Packed Real, with very
small absolute errors. These are distinct acceptance criteria. The original
Packed Real numerical experiment provides no performance measurement.

## Run the campaign

The current fast.9 source pin, including the Linux process-tree memory fix, is
**`09d5385f262988885478a4d47151fcd1dfd39e9c`**. The
[portable guide](portable-benchmark.md#start-from-a-pinned-revision) supplies
complete launch commands using that pin. The
[fast.9 platform qualification and paired performance results](performance-fast9.md)
include the later device-driven corrections. The subsequent
[Linux memory-guard qualification](linux-memory-guard.md) is separate from
those performance measurements. The
[physical Windows and ARM64 reports](device-tests-20261012.md) retain their
separate fast.8 source, successful cases and interruptions. Neither set is
pooled with the historical E/G table above.

Use the [portable benchmark instructions](portable-benchmark.md) for the
platform-specific launcher and pinned dependencies. Select the full
40-character commit from the published release or reviewed PR, and use that
same revision for both launcher and source. A revision identifies a
reproducible campaign more precisely than the package version alone.

In an already prepared checkout of the selected revision, using its prepared
Python environment, the local campaign command is:

```sh
python -m lpsd_benchmark run --point-factors 1 2 --save-spectra --output results.json
```

The same Python arguments work in PowerShell. With the portable PowerShell
launcher, pass them in `-BenchmarkArgs`; the shell launcher accepts them as
ordinary arguments. Bootstrap installation and platform requirements are
maintained in the portable guide so that this announcement does not pin a
new feature to an older runtime.

| Method argument | Estimator |
|---|---|
| `lpsd` | Native bounded LPSD, with LNSD derived from PSD |
| `fftw` | FFTW periodogram with logarithmic aggregation |
| `matched` | FFTW with frequency-dependent smoothing and low-frequency LPSD to approximate LNSD smoothness and peak width |

The default lengths are **100,000; 1,000,000; 1,000,003; 10,000,000;
30,000,000; 100,000,000; and 1,000,000,000**. The full profile checks six
signal presets at the three smaller lengths and white noise at the four
larger lengths. Cases run serially, using the same frequency plan, float64
synthetic input and float32 PSD/NSD outputs. `--profile performance` selects
white noise only.

`--point-factors 1 2` resolves the planner parameter J separately to obtain
baseline and **twice the actual returned frequency count**. The main
campaign's pairs were 469/938, 577/1154, 577/1154, 651/1302, 678/1356,
704/1408 and 744/1488, in the length order above. `--points 1000 2000`
requests raw J values and does not promise that doubling. Reports retain
achieved counts and any planner plateau.

Each `compute_wall_s` / `compute_cpu_s` pair describes **one complete call
producing both PSD and NSD**, including planning and working arrays.
`input_setup_s` covers input generation and the initial checksum;
`process_wall_s` covers the entire worker lifetime, including imports,
setup and final checks. CPU time can exceed wall time when threads run in
parallel. Forecasts, skips and timeouts are separate from completed timings.

## Compare explicit controls

Run configurations serially, retain separate reports and repeat in both
orders when assessing speed. Keep the signal, length, output count, source,
FFTW build, storage and resource limits comparable.

| Comparison | Arguments added to the campaign command |
|---|---|
| Smoothing default at an eight-thread ceiling | `--threads 8` |
| Explicit eight-worker matched smoother | `--threads 8 --smoothing-workers 8` |
| Cap native LPSD working memory, including matched low frequencies | `--native-working-mib 384` |
| Full disk FFT reference | `--storage disk --disk-fft-algorithm full` |
| Packed Real disk FFT comparison | `--storage disk --disk-fft-algorithm packed-real` |
| Explicit RAM window comparison | `--storage ram --window-storage full` or `--storage ram --window-storage fft_input_then_blocked` |

Smoothing retains its default of **min(4, admitted threads)**. An explicit
request is capped by CPU affinity, quota and `--threads`. Each admitted
worker above that default adds an empirical 9 MiB runtime/address reserve
inside the existing allocation budget: eight instead of four reserves
36 MiB. This may change storage or layout admission and is not a measured
RSS increment.

The optional native cap only reduces the existing native working allowance;
it does not increase the overall budget or give the released space to a
larger FFT. An explicit 384 MiB setting is therefore a separate configuration,
not a claim that 384 MiB is fastest. The recovered G384-v2 gate has 233 small
cases and 685 streaming comparisons, but no completed billion-sample timing
comparison. Reports record the requested, admitted and applied byte limits.

Packed Real is an explicit disk choice. The historical prototype retained
24N bytes of large files, so halving its internal FFT length did not halve
the complete file footprint. Current integration and platform follow-up
results must be assessed using their own revision and measurements.

## Resource limits and a 32 GB experiment

| Goal | Arguments |
|---|---|
| 32 decimal GB requested ceiling | `--ram-gib 29.8023223876953125` |
| 32 GiB requested ceiling | `--ram-gib 32` |
| Smaller ARM device | `--ram-gib 0.75 --disk-gib 2 --threads 2 --max-hours 6` |
| Select a local dataset drive | `--work-dir /path/on/local-drive/lpsd-work` |

GB is 10⁹ bytes and GiB is 2³⁰ bytes. The requested RAM ceiling is bounded
by physical/cgroup limits, available memory and the OS reserve. A physical
32 decimal GB machine therefore cannot provide a 32 GB working budget.
The [memory report](performance-memory-points.md) gives the exact conservative
1B admission estimates; they are models, not measured 32 GB timings or
promises of peak RSS.

Automatic RAM admission considers a full window with separate FFT buffers,
then in-place FFT, then in-place FFT with a bounded window. The bounded
window avoids a separate 8N-byte vector: 8 decimal GB at one billion samples.
If RAM admission fails, a bounded disk route is considered. Cases exceeding
resource or time limits are skipped. Detected Linux `tmpfs`/`ramfs` workdirs
disable disk fallback before dataset creation. A local drive avoids
synchronization traffic during measurements.

The portable wrapper allows at most 24 hours for setup plus the run;
`--max-hours` shortens this. The initial wrapper download has a separate
client timeout. Compilation, optional login, generation, computation and
checks consume the setup/run budget. Completed cases are checkpointed,
larger cases use conservative forecasts, and owned workers have timeouts.
Each separate invocation has its own budget.

Linux x86-64/ARM64, macOS and Windows routes, Windows ARM64's x64 process,
and the experimental prepared Termux route are described in the
[platform guide](portable-benchmark.md). CI smoke tests establish portability
and numerical behavior for their tested cases. They do not establish
billion-sample performance on each platform, Raspberry Pi, RK3399 or Android.

## Share a result

Read the [privacy notice](benchmark-privacy.md) before publishing. A run
uploads only when **both** `--upload-github --accept-public-upload` are
selected. `--github-login` on the bootstrap explicitly requests an
interactive login using the participant's GitHub account. Without successful
submission, calculations and prepared public bodies remain local.

To prepare an existing report for inspection without publishing:

```sh
python -m lpsd_benchmark submit results.public.json
```

To publish that selected report using an authenticated GitHub CLI, add
`--accept-public-upload`. `--save-spectra` keeps spectra only in the local
report and also writes the compact `results.public.json`; automatic upload
uses that public file. Select it explicitly for later manual submission if
the local spectra report exceeds the 16 MiB reader limit.

Large public payloads are compressed and may be split into one issue and
numbered comments, with digest markers and a transfer receipt. The
[portable guide](portable-benchmark.md) describes reconstruction and handling
an uncertain response before any manual retry. The uploader does not
silently discard cases or switch to another selected input file.

A public issue is associated with the participant's GitHub account. The
allowlist omits hostnames, usernames, absolute paths, network identifiers,
tokens, logs, raw signals and saved spectra; this does not make the issue
anonymous. Publish the prepared report rather than installation directories
or terminal logs. Numerical work and optional report publication remain
separate, and participants need no repository write permission.
