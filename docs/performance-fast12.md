# fast.12: disk-call overhead and platform regression fixes

Date: **21 October 2026**. Baseline:
[`fa7052505b8a72a1c85d95f599cd188742a18421`](https://github.com/MichaDit/lpsd/commit/fa7052505b8a72a1c85d95f599cd188742a18421),
version **1.0.6+fast.11**. Candidate: **1.0.6+fast.12**.

The new release removes avoidable initialization from the disk FFT's scalar
energy check, repairs the Windows-specific assertions exposed by the physical
campaign, and reports CPU capacity independently of the requested worker cap.
The native numerical library is unchanged. A proposed native optimization was
measured and rejected because complete-call results were mixed.

In seven paired fresh-process observations per configuration, the complete
disk FFTW call used **27.53–34.46% less time at 100K samples** and
**11.36–13.85% less time at 1M samples** on this Linux x86-64 host. Every
frequency/output byte matched its baseline. These are first-call disk-pipeline
results, not native-LPSD kernel gains or new physical Windows/ARM measurements.

The [physical-results analysis](device-analysis-fast12-20261021.md) separately
audits all 13 newly uploaded reports and their 164 rows. Historical results
retain their source revisions, failures, skipped coverage and single-observation
limitations. The analysis supports native/matched parallelism and the existing
explicit packed-real disk option, but not a universal new worker or FFT default.

## Changes and their scope

### Scalar disk energy diagnostic

The disk pipeline previously called `np.testing.assert_allclose` on two scalar
Parseval energies inside its complete-call timer. In an isolated benchmark
worker, that call first loaded NumPy's testing framework. The replacement uses
an ordinary scalar comparison with the same finite-energy acceptance condition:

```text
abs(fft_energy - time_energy) <= 1e-10 * abs(time_energy)
```

Zero reference energy still requires exactly zero computed energy. Signed
zero remains valid. Nonfinite or negative energy now fails explicitly, so
matching infinities or NaNs cannot make this diagnostic pass. The time-domain
and frequency-domain energy passes remain inside the timer. Nothing is removed
from the FFT, window, power normalization, output assembly or required cleanup.

The change applies to the disk FFTW and matched pipelines, including their
full/packed routes and the existing full direct-log energy check. The measured
comparison below isolates FFTW calls. It does not claim the same percentages
for matched calls, which spend much more time in their native subset, or for
persistent applications that already loaded `numpy.testing`.

Tests retain the zero-energy cases, probe the tolerance and nonfinite boundary,
and deliberately corrupt the measured energy through real full/packed FFTW
and matched calls. Failed diagnostics still release the owned workspace.
An independent review also compared 1,916 finite scalar pairs against the
old reference-relative condition, including tolerance neighbours and subnormals.

### Windows discovery assertions

The physical Windows failures were test-path assumptions. The loader used
the native paths returned by the wheel locator and retried the same failed
core/thread pair correctly; the assertions required literal POSIX separators.
The corrected tests now exercise both `PurePosixPath` and `PureWindowsPath`
on every host. The family-isolation check also compares complete paths instead
of a POSIX-only prefix, preventing that check from becoming vacuous on Windows.

The FFTW discovery implementation and library selection were not changed to
make these tests pass. Local path-semantics tests do not constitute fresh
execution on physical Windows hardware. The two failures remain visible in
the saved fast.11 campaign.

### CPU-capacity metadata

`environment.logical_cpus_available` previously repeated the admitted
requested thread setting. It now uses the affinity/quota-limited CPU capacity,
rounded down with a minimum of one serial worker. The selected worker limit
remains in `settings.threads`; `environment.cpu_quota` retains its fractional
capacity. The availability field is not a count of physical CPU cores.

Sixteen regressions exercise detection, environment generation and the public
sanitizer together: requested limits of 1/4/16 or automatic, affinity limits,
6.5-CPU and 0.5-CPU quotas, and no quota. Resource admission and public-schema
allowlists are unchanged; historical JSON is not rewritten.

## Controlled complete disk-call measurements

Source and all observations:
[benchmark script](../benchmarks/bench_disk_call_overhead.py) and
[raw paired report](../benchmarks/results/fast12-20261021/disk-call-overhead.json).

Environment: Linux x86-64, Intel Xeon Platinum 8573C, Python 3.12.14,
NumPy 2.3.5 and pandas 2.2.3. The native build uses portable GCC target clones,
`-fno-fast-math` and `-ffp-contract=off`. Both implementations loaded the same
native binary:

```text
5bf5c0f4b1e194adf3573f6e1aa248aaede07d1b89f0cd8b768f0b80d8ac9745
```

All 56 observations also use identical FFTW core and thread binaries, with
two configured FFTW threads. Their hashes are retained in every observation.
The benchmark verifies each native binary against its build/source manifest,
checks its identity throughout the run, and verifies the source inventory
again at the end.

Each configuration has seven alternating baseline/candidate and
candidate/baseline pairs. Every observation starts a fresh interpreter;
there is no prior complete-call warmup. The timer encloses the complete
`disk_pipeline` call, including both energy passes and workspace cleanup.
Initial imports, library discovery, synthetic input creation, input/output
hashes and evidence collection are outside it. **Lazy imports performed by
the measured call itself remain inside it.** Filesystem caches are warm;
cache flushing and CPU-clock control are not claimed. There were no concurrent
deliberate benchmark or test runs during this comparison.

Inputs are deterministic white-noise float64 files, 50 samples/s, seed
20261021, Kaiser PSLL 200, 1,000 requested frequencies and 100 requested
averages. Both routes use a 16-MiB FFT workspace allowance and workspace power
storage. The existing full/packed algorithm selection is explicit in each row.

| Samples | Algorithm | Actual points | fast.11 median, ms | fast.12 median, ms | Speed ratio | Less call time | Faster pairs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100,000 | Full | 469 | 25.881 | 18.756 | 1.380x | 27.53% | 6/7 |
| 100,000 | Packed-real | 469 | 23.261 | 15.244 | 1.526x | 34.46% | 7/7 |
| 1,000,000 | Full | 577 | 103.385 | 91.640 | 1.128x | 11.36% | 6/7 |
| 1,000,000 | Packed-real | 577 | 85.868 | 73.971 | 1.161x | 13.85% | 7/7 |

`Speed ratio = baseline median / candidate median` and
`Less call time = 100 * (1 - candidate median / baseline median)`.
All observations, ranges and paired geometric ratios remain in the report,
including the two individual pairs that did not favor the candidate.

All 28 baseline calls loaded `numpy.testing` during the call; none of the
28 candidate calls did. The median frequency-energy diagnostic fell from
approximately 6.84–8.71 ms to 0.082–0.441 ms across these configurations.
The phase also includes the unchanged frequency-energy sum, so it is not a
standalone import timer. Shared-host variability means the difference of
whole-call medians should not be interpreted as an exact import cost.

Every observation retained unchanged input, finite/nonnegative outputs,
exact PSD-to-NSD conversion and completed workspace cleanup. Input, frequency
and spectrum hashes match within every configuration. No spectra, averages,
frequencies or precision were discarded for the speedup.

## Rejected native statistics specialization

The [experiment and retained evidence](../benchmarks/experiments/psd_statistics/README.md)
test a real-auto-spectrum specialization of corrected mean/variance updates.
All 640 native comparisons agreed, and some short-segment timings improved.
However, complete 1M API-call baseline/candidate ratios were **0.915, 1.111,
0.892 and 0.994** for one/four workers and PSD/full output. The single-worker
full-output improvement did not offset the mixed results and slower PSD medians.

The candidate is preserved as a patch for reproducibility, not compiled into
the release. The planned 10M screen was stopped after the completed 1M groups;
its uncompleted stage is not a test pass or a timing result. No new architecture
dispatch heuristic was inferred from this single host.

## Local qualification

Fresh reference and fast builds and the complete test suites passed locally:

| Gate | Outcome |
|---|---|
| Fast regression suite | 2,703 passed, 0 failed/errors, 6 skipped, 1 expected failure |
| Original upstream suite | 18 passed, 0 failed/errors |
| Combined passes | **2,721** |
| Focused disk/Parseval/direct-log checks | 116 passed |
| Windows discovery and runtime isolation | 41 passed |
| CPU-capacity and portable-controller checks | 80 passed |
| Portable full FFT / native integration | 12/12 completed; all six validity checks passed |
| Portable packed-real integration | 8/8 completed; all six validity checks passed |
| Source distribution and portable Linux wheel | Built successfully; installed wheel passed isolated smoke checks outside the checkout |

The focused counts overlap the complete suite and are not added to 2,721.
The six skips concern unavailable CUDA dependencies, native macOS/Windows
queries, PowerShell and an inapplicable FP64-long-double path. The expected
failure remains the documented strict relative accuracy limit near a pure-tone
spectral null; absolute-error checks remain mandatory and pass.

The [full integration report](../benchmarks/results/fast12-20261021/portable-full.json)
covers 100K samples, FFTW/matched/native LPSD, RAM/disk and both output-count
factors (469/938 points). The [packed-real report](../benchmarks/results/fast12-20261021/portable-packed.json)
covers FFTW/matched disk calls at 100K/1M and both factors (469/938 and
577/1,154 points). Both use two workers and 1-GiB memory/dataset allowances.
The final saved JSON was read back independently: both files are `completed`,
with all requested rows and no failure or skip. Their individual timings are
integration evidence, not a paired speed comparison. CPU availability now
correctly reports eight while `settings.threads` remains two on this host.

The [qualification manifest](../benchmarks/results/fast12-20261021/qualification.json)
binds the tested candidate to its source and native hashes and records the
packaging gate. Tests ran before committing the candidate. Accordingly the
portable reports retain the baseline Git HEAD with `source_dirty=true`;
their actual candidate bytes are identified by hashes, not attributed to the
baseline commit. Raw logs/JUnit remain local; only counts, relative nonpassing
test identifiers and reproducibility metadata are published.

The wheel installation used a separate virtual environment without dependency
downloads. Its isolated interpreter verified the original native backend,
portable compiler flags and binary manifests, installed package version,
direct-log ABI/endpoints, exact legacy scalar equality, PSD/NSD selection and
the independent corrected two-segment mean/SEM example. Both distribution
artifact digests and the smoke-script digest are in the qualification manifest.

No GitHub Actions workflow was used. Current workflow definitions remain
absent; active test documentation now describes local validation consistently.
New source qualification is local Linux x86-64 only. The older physical
Windows/ARM measurements inform the analysis but do not qualify these new
source bytes on those devices or on macOS/Raspberry Pi.

## Reproduction

Build the pinned baseline and the candidate in separate local checkouts with
the same interpreter and compiler, then run the measurement serially:

```sh
python -m benchmarks.bench_disk_call_overhead \
  --baseline-root ../lpsd-baseline --candidate-root . \
  --n 100000 1000000 --threads 2 --repeats 7 \
  --output benchmark-results/fast12/disk-call-overhead.json

python scripts/validate_local.py
```

The comparison requires identifiable native and FFTW binary hashes. It saves
incomplete/error state if a comparison fails, rather than treating a missing
row as a successful observation. The existing native LPSD, matched and FFTW
estimators retain their distinct statistical meanings.
