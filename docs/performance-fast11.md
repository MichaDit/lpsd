# fast.11: lower startup and frequency-dispatch overhead

Date: **20 October 2026**. Baseline:
[`8d5fab5f2ec027e1f59be5ce96725dc9108ad41f`](https://github.com/MichaDit/lpsd/commit/8d5fab5f2ec027e1f59be5ce96725dc9108ad41f),
version **1.0.6+fast.10**. The measured candidate is **1.0.6+fast.11**.
Its published implementation commit is
[`d63457f81183c68c0759c6e3999b07e72bcc82cc`](https://github.com/MichaDit/lpsd/commit/d63457f81183c68c0759c6e3999b07e72bcc82cc).

The changes reduce unnecessary imports and Python task management around the
existing numerical implementation. In the final local measurements, the
fresh-process first call fell from **1.746511 to 0.313531 seconds**. Several
small parallel API configurations also improved. Larger warmed calls have
mixed results, including the two slower medians retained below. There is no
claim that every input, worker count or platform becomes faster.

## Evidence and source identity

The measured implementation is identified by file hashes, not by assigning
the baseline's Git HEAD to an uncommitted candidate. The
[final qualification and common source/native manifest](../benchmarks/results/fast11-20261020/qualification.json)
binds the released candidate to the locally validated source bytes.
The [complete API measurements](../benchmarks/results/fast11-20261020/api-overhead.json)
retain both implementations' Python sources, recursive native source
manifests, actual shared reference modules, build reports, binary hashes,
input/output hashes and every timed observation. The script verified that
its implementation evidence and measurement sources were unchanged at the
end of the run.

| Measured identity | SHA256 |
|---|---|
| Native library, identical for baseline and candidate | `5bf5c0f4b1e194adf3573f6e1aa248aaede07d1b89f0cd8b768f0b80d8ac9745` |
| Candidate `lpsd_fast/api.py` | `922bd9db061d2978020fbc294ce6f2514b6279566122054d8f27bc713cb6b6a9` |
| Candidate `lpsd_fast/planning.py` | `dab6943619da46903753194a55a9861867294a05b54ea05f5b8f509fc876a48c` |

The [startup observations](../benchmarks/results/fast11-20261020/startup.json)
record their own script hash and the top-level `lpsd_fast` Python file hashes.
Those hashes agree with the final API run. That startup JSON is a narrower
provenance record; its additional benchmark dependencies and native binary
are covered by the common qualification manifest, rather than claimed as a
complete standalone source closure.

Historical physical-device results remain attributed to their actual tested
revisions. The [physical campaign review](platform-followup-20261020.md)
explains those measurements, unfinished cases and method-specific worker
comparisons. They are separate evidence from the new local observations here.

## Changes

### Dynamic frequency dispatch

The public parallel API now submits one long-lived task per admitted worker.
Each worker claims the next frequency index under a short lock, then performs
that frequency's existing calculation outside the lock. This avoids a
separate `Future` and completion notification for every spectral point while
retaining dynamic distribution at single-frequency granularity.

Results still occupy their original indices. Each frequency acquires and
releases its own existing memory reservation. The index lock is released
before a worker waits for memory or enters native code; workers do not hold
it throughout a calculation. The executor joins active tasks before an
exception returns, so input arrays are no longer being read when control
returns to the caller.

The final implementation uses no frequency-cost model, fixed frequency
blocks or input-size threshold. Initial contiguous grouping experiments
showed configuration-dependent load imbalance and were not selected. The
single-worker path retains its existing sequential loop.

### Imports needed for the requested operation

`lpsd_fast` package metadata and build tooling load without NumPy, pandas or
SciPy. Requesting a public numerical function resolves the actual function
object, retaining its signature. Common NumPy/Kaiser computations load the
native API without importing the frozen reference wrapper and its SciPy
detrending dependency.

Five planning/normalization function bodies are copied verbatim from the
pinned `lpsd/_helpers.py` into the NumPy-only `lpsd_fast/planning.py`:
`_kaiser_alpha`, `_myround`, `_kaiser_rov`, `_ltf_plan` and `_asdrms`.
An AST-equivalence regression compares all five definitions with the pinned
reference, including their arithmetic expressions. The original reference
files remain byte-identical. See [source provenance](provenance.md).

Flat-top registry resolution retains the actual original function identities
and coefficients. Flat-top callbacks and signal presets that require extra
dependencies load them when used. The FFTW/matched white-signal import path
also uses the lightweight planning subset. Benchmark helpers explicitly
import aliased API modules before accessing their native bindings, including
when the baseline checkout itself has lazy package initialization.

### Native initialization and FFTW runtime isolation

The native API publishes its cached library handle only after every ctypes
binding has been initialized. A missing later symbol therefore cannot leave
a partially initialized handle that a subsequent call mistakes for a
complete binding.

The Python FFTW wrapper also tracks initialized thread runtimes by their
loaded FFTW core handle. A one-thread plan following a threaded plan resets
its own core's thread count. It no longer uses the most recently initialized
thread companion from an unrelated system or wheel core. The same lookup
is used by ordinary real FFT, Bluestein and disk planning. The targeted
[thread-runtime tests](../test_fast/test_fftw_runtime_isolation.py) cover this
wrapper defect; the warmed native-API timings below are not measurements of
an FFTW speedup.

### Forecasts for isolated benchmark cases

The portable forecast separates spectral compute time from the remaining
process time before extrapolating to a larger input. Noncompute time scales
linearly with input length, while the method's existing work model scales
compute time. This avoids amplifying a small-case startup cost with the
native computation exponent. The existing guard factor, deadlines, memory
ceilings and matching-calibration requirements remain in effect.

The [campaign and forecast report](platform-followup-20261020.md#forecast-correction)
contains the equation, historical validation and affected instrument cases.
A less pessimistic forecast permits an actual measurement; it does not turn
a historical skip into a completed result.

## Warmed complete-API results

Environment: **Linux x86-64, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3**,
nine visible logical CPUs. Both native builds used
GCC 13.3.0, portable x86 target clones, `-fno-fast-math` and
`-ffp-contract=off`. Their native binary hashes are identical.

Each configuration uses one full-problem warmup per implementation followed
by **seven alternating baseline/candidate and candidate/baseline pairs**.
All **154 timed calls in 77 pairs** are retained, alongside the 22 untimed
warmup calls. There were no concurrent benchmark runs. Timers enclose the
complete public API, including input validation, planning, fresh windows and
coefficients, native computation and output assembly. Imports, input creation,
hashes, comparisons and report writes are outside the timer.

Inputs are resident float64 white-noise arrays from PCG64 seed 20610 at
50 samples/s. Calls request 1,000 frequencies and 100 averages, corrected
statistics, a 512-MiB concurrent working allowance and a 128-MiB window cache.
The actual number of points is shown below. Kaiser uses PSLL 200 and its
recommended overlap; Boxcar uses overlap 0.8. Unrelated BLAS/OpenMP numerical
pools are limited to one thread; the native API uses the stated worker count.

A speed ratio is baseline median divided by candidate median. “Less wall
time” is `100 * (1 - candidate / baseline)`; a negative value means slower.

| Samples | Workers | Kernel / outputs / window | Actual points | fast.10 median, s | fast.11 median, s | Speed ratio | Less wall time | Faster pairs |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 10,000 | 1 | fast / PSD + NSD / Kaiser | 297 | 0.013681 | 0.013403 | 1.021 | 2.03% | 3/7 |
| 10,000 | 2 | fast / PSD + NSD / Kaiser | 297 | 0.028628 | 0.018518 | 1.546 | 35.31% | 6/7 |
| 10,000 | 4 | fast / PSD + NSD / Kaiser | 297 | 0.029983 | 0.023180 | 1.293 | 22.69% | 7/7 |
| 100,000 | 1 | fast / PSD + NSD / Kaiser | 469 | 0.062807 | 0.062440 | 1.006 | 0.58% | 3/7 |
| 100,000 | 2 | fast / PSD + NSD / Kaiser | 469 | 0.060305 | 0.048450 | 1.245 | 19.66% | 7/7 |
| 100,000 | 4 | fast / PSD + NSD / Kaiser | 469 | 0.055250 | 0.048136 | 1.148 | 12.88% | 6/7 |
| 1,000,000 | 1 | fast / PSD + NSD / Kaiser | 577 | 0.509189 | 0.516660 | 0.986 | -1.47% | 2/7 |
| 1,000,000 | 2 | fast / PSD + NSD / Kaiser | 577 | 0.333474 | 0.309203 | 1.078 | 7.28% | 6/7 |
| 1,000,000 | 4 | fast / PSD + NSD / Kaiser | 577 | 0.200059 | 0.205373 | 0.974 | -2.66% | 3/7 |
| 100,000 | 4 | auto / all / Kaiser | 469 | 0.076675 | 0.061185 | 1.253 | 20.20% | 7/7 |
| 100,000 | 4 | fast / PSD + NSD / Boxcar | 482 | 0.050753 | 0.044450 | 1.142 | 12.42% | 7/7 |

Every timed spectrum has **identical frequency indices, output dtypes and
output bytes** to its baseline reference, including the `auto/all` columns.
The candidate warmup outputs also passed the same comparison. Each of the
11 input arrays was unchanged at the end of its case. Source/native evidence
was rechecked after the final call.

The strongest consistent improvements are in the small parallel cases,
where Python task management represents more of the API time. Extra workers
still cost time for very small inputs: at N=10,000, the candidate's one-worker
13.403 ms remains faster than its two- and four-worker calls. No automatic
worker-count change is inferred from this one-host matrix.

The N=1,000,000/four-worker median is 2.66% slower and only three pairs favor
the candidate. Its full ranges overlap: baseline **0.179112–0.210242 s**,
candidate **0.172468–0.225670 s**. The single-worker medians are also mixed,
although that scheduling path and the native binary are unchanged. These
observations remain part of the result rather than being discarded or
converted into a universal speedup claim. All ranges, individual pair
ratios and geometric pair summaries are in the raw report. Shared-host
scheduling and uncontrolled CPU clocks limit causal interpretation of small
percentage changes.

## Fresh-process startup and memory

These are a separate series of **seven pairs per scenario**. Each observation
starts a new child process from a small standard-library-only parent. The
reported wall time includes process launch, the requested imports or work,
and process exit. Filesystem caches were not flushed. The “first call” case
also creates a seeded 10,000-sample array and performs a four-worker
fast/Kaiser corrected PSD+NSD calculation.

Memory columns are **median current RSS** from Linux `VmRSS` at the end of the
requested work, before child exit. They are not inferred allocation sizes or
peak working-memory requirements. The raw report separately retains process
high-water counters; those counters are not substituted for current RSS.

| Scenario | fast.10 process median, s | fast.11 process median, s | Speed ratio | fast.10 current RSS, MiB | fast.11 current RSS, MiB |
|---|---:|---:|---:|---:|---:|
| Package metadata and builder import | 1.812422 | 0.039593 | 45.776 | 121.40 | 11.18 |
| Public numerical API import | 1.738767 | 0.283286 | 6.138 | 121.39 | 53.38 |
| FFTW/matched imports and tiny white-signal construction | 1.887566 | 0.297727 | 6.340 | 121.56 | 54.45 |
| Imports, input construction and first API call | 1.746511 | 0.313531 | 5.570 | 125.01 | 57.23 |

All four baseline scenarios loaded SciPy and the reference package; none of
the corresponding candidate observations did. The 14 first-call output
fingerprints agree. Its baseline process times range from
**1.684470–1.865532 s**, compared with **0.300556–0.356138 s** for the
candidate. The visible reduction is primarily initialization work avoided
for these operations. It is separate from the warmed API ratios and must
not be multiplied by them or attributed to faster native arithmetic.

## Local qualification

The complete final local validation ran fresh reference and fast builds,
the full fast regression suite and the upstream suite:

| Gate | Final outcome |
|---|---|
| Fast regression suite | **2,665 passed, 0 failed/errors, 6 skipped, 1 expected failure** |
| Original upstream suite | **18 passed, 0 failed/errors** |
| Combined passes | **2,683** |
| Same-host baseline | **2,610 passes**: 2,592 fast and 18 upstream |
| Additional focused tests | **73** beyond the baseline |

The final local log records 1.120 s for the reference build, 6.338 s for the
fast build, 98.848 s for fast regressions and 15.261 s for upstream tests.
The qualification manifest retains the actual source hashes and gate results.
The skips and expected failure remain distinct from passing tests. Local
Linux success does not stand in for native Windows, ARM or macOS execution.

The source distribution and portable Linux x86-64 wheel built successfully.
The wheel was installed without dependency downloads into a separate virtual
environment outside the checkout and tested with an isolated interpreter.
The installed smoke checks passed native-library/source binding, portable
wheel flags, direct-log ABI/endpoints, exact scalar legacy equality, selected
PSD/NSD consistency and the independent corrected two-segment mean/SEM gate.
The artifact and smoke-script digests are retained in the qualification record.

The portable integration matrix uses N=100,000, three estimators, RAM/disk,
two output factors (469/938 actual points), two threads and 1-GiB RAM/disk
budgets. The [baseline run](../benchmarks/results/fast11-20261020/portable-baseline.json)
and [verified fast.11 run](../benchmarks/results/fast11-20261020/portable-fast11.json)
each completed all 12 cases. Every corresponding input/frequency/PSD/NSD and
native-binary hash is identical; all five validity checks passed. FFTW and
matched cases retained the same FFTW core/thread binaries and enabled two
threads. These single integration runs do not establish a timing ratio.
The [comparison record](../benchmarks/results/fast11-20261020/portable-comparison.json)
retains their actual timings and final-file checks.

An initial candidate invocation returned success and printed 12 completed
cases, while its saved checkpoint retained only 11 rows and `state=running`.
That [incomplete checkpoint](../benchmarks/results/fast11-20261020/portable-fast11-incomplete.json)
is preserved and excluded from qualification; its missing row is not filled
from console text. The cause was not established. An absolute-output rerun
and a separate [relative-output recheck](../benchmarks/results/fast11-20261020/portable-fast11-relative-recheck.json)
each saved all 12 successful cases. Their parent process read and hashed the
complete file immediately after child exit. Both sets have identical
numerical hashes to the baseline. The file/state discrepancy was not
reproduced, and no unproven persistence fix is claimed.

## Reproduce the measurements locally

Use two complete checkouts, with the baseline at the exact commit above.
Build both native libraries in each checkout using the same Python environment
and compiler; [local validation](local-validation.md) describes the build and
regression entry point. From the candidate checkout, with the built baseline
at `../lpsd-baseline`, run the two measurement series sequentially:

```sh
python -m benchmarks.bench_api_overhead \
  --baseline-root ../lpsd-baseline --candidate-root . \
  --n 10000 100000 1000000 --workers 1 2 4 --repeats 7 \
  --extra-cases --output benchmark-results/fast11/api-overhead.json

python -m benchmarks.bench_startup \
  --baseline-root ../lpsd-baseline --candidate-root . --repeats 7 \
  --output benchmark-results/fast11/startup.json
```

The warmed-API command requires actual native build reports and verifies
source/binary attribution before computing. Both commands retain every pair.
They do not install dependencies, change OS resource limits or dispatch
GitHub Actions. Keep other CPU benchmarks stopped during these runs.

## Remaining physical checks

The [fast.11 device rerun guide](device-rerun-fast11.md) keeps the next physical
checks explicit. Windows, the ARM64 SBC and the ARM64 instrument need fresh
fast.11 reports to determine their actual startup savings, worker scaling,
FFTW runtime selection and outcomes of the previously incomplete cases.
There is no new physical macOS result. Existing memory and disk ceilings
continue to apply, and the previously unexplained Windows billion-sample
restart remains a separate unresolved case. Neither this local matrix nor
revised forecasts establish its cause or resolution.
