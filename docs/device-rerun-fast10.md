# Device rerun for fast.10

Use the implementation commit
**`e0426c4bad9d7c5fccd54de935eea2d1ea1b2519`** for the wrapper and downloaded source.
The new default is `statistics=corrected`; the old physical-device reports
used legacy statistics. The [qualification](fast10-validation.md) and
[statistics contract](statistics.md) explain the change. These instructions
prepare a new campaign; they do not report a completed device rerun.

## 1. Small preflight on each device

The preflight requests all three methods, RAM and disk, both point factors,
and N=100,000. Check `state=completed` and all **12 comparison cases** before
starting a larger campaign. Admission skips remain skips. A disk case needs
a real local filesystem; select an existing local-drive directory with
`--work-dir` if the system temporary directory is backed by RAM.

Linux x86-64, Linux ARM64 or macOS, in a writable directory:

```sh
curl --proto '=https' --proto-redir '=https' --tlsv1.2 --connect-timeout 20 --max-time 60 -fsSL https://raw.githubusercontent.com/MichaDit/lpsd/e0426c4bad9d7c5fccd54de935eea2d1ea1b2519/scripts/run_benchmark.sh | sh -s -- --revision e0426c4bad9d7c5fccd54de935eea2d1ea1b2519 --statistics corrected --sizes 100000 --point-factors 1 2 --storage both --profile performance --threads 2 --max-hours 0.5
```

Keep the previous explicit budgets when running on the small ARM devices:

| Device | Append to the Unix command |
|---|---|
| ARM64 SBC | `--ram-gib 0.375 --disk-gib 2` |
| ARM64 instrument | `--ram-gib 0.75 --disk-gib 2` |

For the instrument, use the same prepared Linux environment as the previous
campaign. The desktop bootstrap is not a stock Android installation method.
The separate prepared Termux route and its prerequisites remain documented
in [the portable guide](portable-benchmark.md#android--termux).

Windows x64, preserving the prior 32-GiB allocation ceiling, 16-GiB dataset
budget and 16-thread request:

```powershell
& ([scriptblock]::Create((Invoke-RestMethod -TimeoutSec 60 'https://raw.githubusercontent.com/MichaDit/lpsd/e0426c4bad9d7c5fccd54de935eea2d1ea1b2519/scripts/run_benchmark.ps1'))) -Revision 'e0426c4bad9d7c5fccd54de935eea2d1ea1b2519' -BenchmarkArgs @('--statistics','corrected','--sizes','100000','--point-factors','1','2','--storage','both','--profile','performance','--threads','16','--ram-gib','32','--disk-gib','16','--max-hours','0.5')
```

All ceilings remain bounded by the detected resources. Each wrapper invocation
creates its own retained environment and reports. Setup and the test share
the stated time limit.

## 2. Targeted continuation at the existing budgets

After a successful preflight, use the same pinned wrapper command with the
following benchmark arguments. Change only the options shown; the revision
and statistics mode stay fixed.

| Device | Continuation arguments | Main question |
|---|---|---|
| ARM64 SBC | `--ram-gib 0.375 --disk-gib 2 --threads 2 --sizes 100000 1000000 1000003 10000000 30000000 --point-factors 1 2 --storage auto --profile full --max-hours 24` | Confirm the corrected estimator and the selected FFTW runtime within the existing admission limits. |
| ARM64 instrument | `--ram-gib 0.75 --disk-gib 2 --threads 2 --sizes 100000 1000000 1000003 10000000 30000000 --point-factors 1 2 --storage auto --profile full --max-hours 24` | Revisit the three 30M disk memory stops with the actual guard metric and trigger retained. |
| Windows x64 | `--ram-gib 32 --disk-gib 16 --threads 16 --sizes 100000 1000000 1000003 10000000 30000000 100000000 --point-factors 1 2 --storage auto --profile full --max-hours 24` | Confirm the corrected statistics, archive/path repairs and complete small-to-100M behavior. |

Keep `--statistics corrected` in each continuation. The full profile adds the
historical signal presets at the small quality sizes; larger cases use white
noise. The instrument's former 30M failures were memory-guard terminations,
not proof of an OS out-of-memory event. A fresh stop must be evaluated from
`memory_guard`, not inferred from a different peak-RSS field. These limits
are not increased to turn an inadmissible case into a pass.

The previous Windows doubled-point billion-sample native case coincided with
a host restart and bugcheck 0x4E; its cause remains unconfirmed. Keep that
case separate until the smaller continuation and the new diagnostics have
been evaluated. A user-space statistics or reporting fix does not establish
that a kernel-level restart has been resolved. There is still no completed
timing for that failed historical case. See the
[original device evidence](device-tests-20261012.md#fast8-campaign-b8c5002).

## 3. Existing optional performance routes

The packed-real disk FFT remains an explicit comparison. After the main
continuation, use the same source, statistics mode and budgets with:

```text
--methods fftw matched --storage disk --disk-fft-algorithm packed-real --sizes 100000 1000000 10000000 30000000 --point-factors 1 2 --profile performance
```

Keep its report separate from the default full-complex FFT. The active
transform region is smaller, but the conservative input-plus-workspace disk
admission remains 24N bytes. Compare actual storage, point count, library
hashes, native working allowance and configured threads before interpreting
timings. Past fast.9 measurements do not become measurements of this rerun.

The direct-log reduction requires a 128-MiB FFT tile. The existing 384-MiB
and 768-MiB ARM RAM budgets admit smaller FFT work allowances, so those
configurations do not qualify for that route. The bounds and algorithm
requirements remain documented in [the portable guide](portable-benchmark.md).

For comparison with historical fast.1–fast.9 spectra or timings, repeat only
the relevant cases in a separate report with `--statistics legacy`. A change
from legacy to corrected statistics is a change of estimator and must not
be scored as numerical optimization error. The historical comparison harness
uses legacy explicitly; new portable runs default to corrected.

## 4. Evidence to retain

The commands above produce an already sanitized **`results.json`**; return
that file for analysis. If a separate run used `--save-spectra`, retain its
`results.json` locally and return only **`results.public.json`**. The public
report contains the revision/library hashes, statistics mode,
all result checks, finite diagnostic maxima, requested/selected FFTW threads
and memory-guard evidence. It does not contain the input/spectrum arrays,
personal paths or raw exception text. A setup failure has a separate local
bootstrap report; keep it for diagnosis rather than treating it as a
completed numerical campaign.

The acceptance questions are concrete:

- Did every admitted comparison complete, or does a retained check explain
  a failure? A skip is never counted as completion.
- Did FFTW configure two threads on the ARM devices? If it selected one,
  which runtime was tried and which closed failure category was recorded?
- Did a 30M instrument case finish within the unchanged ceiling? If not,
  which sampled metric and byte value actually crossed it?
- Are the expected frequency plan, nonnegative finite PSD, exact PSD-to-NSD
  conversion and unchanged input confirmed in each completed case?
- Are timing comparisons using the same statistics, point count, storage,
  libraries, threads and resource settings?

No new physical-device measurements are claimed by the fast.10 code change
or by successful cloud CI alone.
