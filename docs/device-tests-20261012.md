# Physical-device test results, 12 October 2026

The completed first portable benchmark campaign is published below. A second
campaign tested the merged fast.8 source, including the full regression suites.
It is incomplete on two machines. Failed tests and interrupted work are retained.

Publication snapshot: **2026-10-12T11:06:42.790025+00:00**. Latest local collection:
**2026-10-12T11:06:13.696356+00:00**. Reports contain synthetic fixed-presets only.
Machine labels are generic; no hostnames, addresses, credentials, serials,
personal paths, source spectra or arbitrary logs are published.

## Completed campaign: 6246f0c

Source: [`6246f0c`](https://github.com/MichaDit/lpsd/commit/6246f0c6061b5f72dba8e235e640fb0b71281a20).
Each matrix has 132 FFTW, LNSD-matched FFTW and LPSD cases, with baseline/doubled
output counts. Six signal presets are tested at the three smaller quality sizes;
larger sizes use white noise. Skips reflect the admitted memory, disk or time
limits and are not successful tests.

| Platform | Completed | Failed | Skipped |
|---|---:|---:|---:|
| Windows x86-64 | 127 | 1 | 4 |
| ARM64 SBC | 108 | 0 | 24 |
| ARM64 instrument | 116 | 3 | 13 |

Windows failed the billion-sample LPSD case with doubled output points:
`invalid_result`. That report does not identify which invariant failed; no
root-cause or acceptable-result claim is made. The instrument failed three
30-million-sample disk cases at `memory_limit`: FFTW at both point factors,
and matched FFTW at baseline points.

Representative white-noise results at **N=1,000,000**, baseline points:

| Platform | Method | Actual points | API wall, s | Process wall, s |
|---|---|---:|---:|---:|
| Windows x86-64 | fftw | 577 | 0.061583 | 1.281000 |
| Windows x86-64 | matched | 577 | 0.152836 | 1.281000 |
| Windows x86-64 | lpsd | 577 | 0.105096 | 1.281000 |
| ARM64 SBC | fftw | 577 | 0.177122 | 4.251557 |
| ARM64 SBC | matched | 577 | 2.796865 | 6.898490 |
| ARM64 SBC | lpsd | 577 | 5.941102 | 10.053157 |
| ARM64 instrument | fftw | 577 | 0.554663 | 20.268357 |
| ARM64 instrument | matched | 577 | 8.706935 | 28.804876 |
| ARM64 instrument | lpsd | 577 | 20.340191 | 39.728885 |

## fast.8 campaign: b8c5002

Source: [`b8c5002`](https://github.com/MichaDit/lpsd/commit/b8c500217f7c0ad996f55b039bbf09d685800614).

| Platform | Benchmark evidence | Current limitation |
|---|---|---|
| Windows x86-64 | 131 unique successful cases out of 132 | Outstanding doubled-point billion-sample LPSD case interrupted by host crash |
| ARM64 SBC | 132 recorded: 108 completed, zero failed, 24 skipped | Benchmark complete; regression coverage comes from separate runs |
| ARM64 instrument | 6 recorded at the last retained checkpoint | Fresh status unavailable after the control host restarted; final results not claimed |

The first Windows segment stopped after 130 successful cases because a campaign
status-file replacement collided with a Windows reader handle. The wrapper was
repaired and a calibrated continuation started. The local host subsequently
restarted at **2026-10-12T09:32:56Z**; system events recorded bugcheck **0x4E**.
Its cause is unconfirmed and is not attributed to LPSD or a hardware component.
The continuation retained the doubled-point billion-sample matched result before
the restart. Its eight smaller supporting/calibration cases are excluded from
the unique 132-case count. The unfinished LPSD timing is not reported as a result.

The instrument used detached on-device supervisors; the host restart does not
establish that those device processes stopped. Their subsequent liveness and
completion have not been freshly verified because the ADB transport is offline.
The final post-regression/cleanup release remains unverified. The retained
checkpoint is historical evidence, not a current running/finished assertion.

Representative fast.8 white-noise results at **N=100,000**, baseline points:

| Platform | Method | Actual points | API wall, s | Process wall, s |
|---|---|---:|---:|---:|
| Windows x86-64 | fftw | 469 | 0.019216 | 1.109000 |
| Windows x86-64 | matched | 469 | 0.039490 | 1.125000 |
| Windows x86-64 | lpsd | 469 | 0.039816 | 1.125000 |
| ARM64 SBC | fftw | 469 | 0.038662 | 4.043637 |
| ARM64 SBC | matched | 469 | 0.311121 | 4.774580 |
| ARM64 SBC | lpsd | 469 | 0.573499 | 4.552017 |
| ARM64 instrument | fftw | 469 | 0.125578 | 19.196576 |
| ARM64 instrument | matched | 469 | 0.942455 | 19.692130 |
| ARM64 instrument | lpsd | 469 | 1.714765 | 20.492503 |

### Regression results

| Run | Passed | Failed | Errors | Skipped | Expected failures |
|---|---:|---:|---:|---:|---:|
| Windows fast, corrected prerequisites | 2297 | 2 | 0 | 6 | 2 |
| Windows upstream | 18 | 0 | 0 | 0 | 0 |
| SBC first full fast run | 1735 | 0 | 0 | 374 | 2 |
| SBC matched-smoothing recheck | 106 | 0 | 0 | 0 | 0 |
| SBC upstream | 18 | 0 | 0 | 0 | 0 |
| Instrument first full fast run | 2076 | 27 | 0 | 6 | 2 |
| Instrument upstream | 18 | 0 | 0 | 0 | 0 |

The SBC dependency-enabled full rerun first passed 2,078 tests and failed 25,
with six skips and two expected failures. All 25 failures occurred at the frozen
reference's Git provenance lookup, before numerical comparison. Supplying real
shallow metadata for the pinned public commit restored that prerequisite;
the subsequent 106-test matched-smoothing run passed completely. Reference
source bytes and numerical assertions were unchanged. These separate runs
provide combined coverage; a single clean all-suite rerun is not claimed.

The two Windows failures remain: backslash ZIP-name rejection and a simulated
Linux address-guard path assertion. They are not designated expected failures.
The first preparation errors from a missing CUDA bridge and cross-drive test
temporary paths were repaired before the reported corrected rerun.

The instrument's 27 initial failures include the same 25 Git prerequisite
failures, a deadline test whose child PID file was not created in time, and
an exceptional CSD comparison at input amplitude `1e155`. The latter observed
**3.0936972401917152e-12**, exceeding the mandatory **2e-12** limit. Three compiler
warnings also report smoothing loops that were not vectorized. These results
are unresolved. A complete fast regression was queued after the benchmark with
the restored Git prerequisite; its outcome is not yet verified.

### Conditions and interpretation

- Windows admitted 16 workers, 32 GiB benchmark RAM and 16 GiB dataset disk.
  The SBC admitted two workers, 0.375 GiB RAM and 2 GiB disk; the instrument
  admitted two workers, 0.75 GiB RAM and 2 GiB disk.
- The instrument ran a private Debian ARM64 chroot on the Android kernel,
  with its ordinary acquisition/web processes preserved. This is not a stock
  Android/Termux test or an isolated idle-device measurement. The first campaign
  used portable libraries compiled on the SBC; fast.8 used fresh instrument
  builds with Zig 0.13.0 after Zig 0.15.2 failed a preprocessor query.
- `compute_wall_s` covers the complete PSD/NSD API call. `process_wall_s` also
  includes worker startup/setup/checks. Each is an observed individual execution,
  not a median or confidence interval. No unfinished process heartbeat is used
  as a completed timing.
- Software versions differ between the Windows environment and pinned ARM64
  environments. Installed package-version metadata can lag the copied source;
  the explicit source commit and freshly built library hashes identify the
  tested code. Cross-revision tables are separate cohorts, not paired speedup
  evidence. These fixed worker limits do not establish optimal parallelization.
- FFTW, matched FFTW and LPSD are different estimators. Do not interpret their
  timing ratios as equivalent-estimator acceleration or pool total durations
  across matrices with different skipped cases. The finiteness, nonnegative
  PSD, PSD/NSD relationship and input-immutability checks are not a universal
  signal-accuracy guarantee.

The earlier local CUDA/cuFFT/VkFFT and worker measurements remain in
[Windows optimization](windows-optimization.md). CI's separate paired platform
evidence is in [platform comparison](platform-comparison-20261012.md).

## Published evidence

The [manifest](../benchmarks/results/device-tests-20261012/manifest.json) records
source commits, observed lifecycles, case totals and SHA256 for each report.
The original `state` inside an interrupted checkpoint can remain `running`;
the manifest and interruption notes specify the observed process lifecycle.
Benchmark exports use the repository's closed `privacy.sanitize` schema.
The [regression summary](../benchmarks/results/device-tests-20261012/regressions.json)
contains reviewed counts and relative test identifiers, without raw logs.

- [legacy-windows.json](../benchmarks/results/device-tests-20261012/legacy-windows.json)
- [legacy-arm64-board.json](../benchmarks/results/device-tests-20261012/legacy-arm64-board.json)
- [legacy-arm64-instrument.json](../benchmarks/results/device-tests-20261012/legacy-arm64-instrument.json)
- [fast8-windows-initial.json](../benchmarks/results/device-tests-20261012/fast8-windows-initial.json)
- [fast8-windows-continuation.json](../benchmarks/results/device-tests-20261012/fast8-windows-continuation.json)
- [fast8-arm64-board.json](../benchmarks/results/device-tests-20261012/fast8-arm64-board.json)
- [fast8-arm64-instrument-snapshot.json](../benchmarks/results/device-tests-20261012/fast8-arm64-instrument-snapshot.json)

## Follow-up in fast.9

The [fast.9 follow-up](performance-fast9.md) evaluates these records and
qualifies corrections for ZIP member names, Windows path assertions,
archive-based frozen-reference metadata, slow-child deadline fixtures and
extreme CSD projections. Its local and CI results are new source cohorts;
they do not replace the failures, skips or interrupted cases in this
publication snapshot. The physical instrument rerun and Windows doubled-point
1B LPSD case remain unconfirmed. Original report bytes and manifest hashes
are preserved.
