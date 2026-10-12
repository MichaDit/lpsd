# Direct logarithmic power reduction

The optional `--disk-power-reduction direct-log` route computes the simple
FFTW estimator's logarithmic power groups directly from the original-N
complex FFT matrix. It avoids constructing and revisiting the N-dependent
one-sided power vector. The standard selection is `materialized`.

This option applies to the simple FFTW disk pipeline. The matched estimator
needs frequency-dependent smoothing, so it continues to use its existing
power representation. Packed-real reconstruction is a separate explicit
FFT choice, documented in the [platform follow-up](performance-platform-followup.md)
and [phase-construction comparison](performance-packed-phase.md).

## Supported profile

The first integrated profile keeps the conditions of the reviewed R3
qualification:

| Setting | Required value |
|---|---|
| Estimator and storage | Simple `fftw`, disk |
| FFT algorithm | `full`, original input length |
| Input length | 2 through 1,000,000,000, subject to the existing admission checks |
| Sample rate | 50 Hz |
| Native frequency-plan averaging target | 100 |
| Window | Existing Kaiser window with 200 dB PSLL target |
| FFT working allowance | 128 MiB |
| Workspace layout | Existing workspace-power-compatible geometry |
| Returned frequency groups | 1 through min(65,536, N/2) |
| Native reducer ABI | 3, verified before the large workspace is created |

These are admission limits, not measured coverage of every length or
machine. The portable runner reports unsupported combinations instead of
substituting a different calculation. The reduction is ignored for methods
or RAM paths to which it does not apply, with `not_applicable` recorded.

In a prepared checkout, an explicit campaign uses:

```sh
python -m lpsd_benchmark run --methods fftw --storage disk --disk-fft-algorithm full --disk-power-reduction direct-log --point-factors 1 2 --save-spectra --output direct-log-results.json
```

Use a separate run with `--disk-power-reduction materialized` for the
reference. Retain the input, revision, FFTW build, thread count, working
allowances and output counts, and alternate comparison order. The runner's
deadline and memory/disk guards remain active. An explicit setting must fit
the available budget; it does not raise that budget.
The portable runner derives the FFT allowance from available working memory.
If it cannot admit the required 128 MiB, this direct profile is unsupported;
there is no separate portable `--fft-memory-mib` override.

## Arithmetic and storage

For the stored transform matrix, element `(r, c)` represents original
frequency bin `k = r + A*c`. The native reducer traverses the intersections
of the requested logarithmic intervals with each matrix row. It computes
the same squared complex amplitudes and normalization as the materialized
power path, with the existing DC and even-length Nyquist endpoint weights.
Bounded sums use compensation; a different summation order can change
floating-point rounding.

A second, independent traversal covers every nonnegative FFT bin for the
energy check, including DC even when it is outside the requested output
groups. The existing Parseval tolerance is `rtol=1e-10, atol=0`. The final
PSD conversion and the NSD square root retain the existing output types.
Each result verifies group/bin counts, endpoint counts and ABI metadata.

The transform matrix is read-only during reduction. Ordinary owned arrays
hold the group result, so it remains usable after the FFT workspace closes.
The reducer's reported array state is **48J + 48 bytes** for J groups:
33,840 bytes at 704 groups and 67,632 bytes at 1408 groups. It creates no
N-sized power array, extra power file or additional worker threads.

The input file still occupies 8N bytes and the full complex workspace 16N
bytes. Their combined logical size is **24N bytes**. The direct route avoids
writing, ordering and rereading the power vector and its additional virtual
mapping. It still traverses the complex matrix for the groups and the
independent energy check. It does not reduce the two large files or turn a
disk-space failure into an admitted run. The original conservative allocation
limits remain in force.

The result records `disk_power_reduction`, `direct_log_power_abi`,
`direct_log_power_state_bytes` and `power_vector_created`. Public reports
retain these through a closed allowlist. The direct route has a separate
calibration identity: an earlier RAM or materialized observation cannot
qualify its large-case forecast.

## Integrated 100-million-sample comparison

The [complete paired measurements](../benchmarks/results/direct-power-100m-20261012.json)
were collected on 12 October 2026 at integrated source revision
[`ca2d26f91ed94d295a2c4b497db8aa2775b3bcdc`](https://github.com/MichaDit/lpsd/commit/ca2d26f91ed94d295a2c4b497db8aa2775b3bcdc).
All nine recorded source SHA256 values match that commit. The subsequent
`5f64570f8a8028a0ac8fa81d65d523fdeae0328e` commit changes only the optional
FFTW test fixture; the measured implementation and harness are unchanged.
The final `1c34582` release qualification also includes later CSD and
supervision corrections. Its native source closure is therefore a new
identity, even though the direct reducer and FFT operations used here are
unchanged. This table remains a measurement of `ca2d26f9`; the
[fast.9 report](performance-fast9.md) separates the final validation.

This is a Linux x86-64 comparison with Python 3.12.14, NumPy 2.3.5,
FFTW 3.3.10 and eight admitted FFTW threads, in an 8 GiB cgroup with eight
CPU equivalents. Both variants use the same synthetic white-noise input,
original-N full transform, 10,000 by 10,000 factorization, strided columns,
128 MiB FFT allowance and 384 MiB native allowance. For each actual output
count, three pairs alternate materialized/direct, direct/materialized,
materialized/direct. All twelve complete calls include planning, windowing,
preprocessing, FFT, reduction, diagnostics, output construction and cleanup.
Input generation/hashing and library loading are outside the timer.

| Returned points | Materialized wall median (range), s | Direct wall median (range), s | Less wall time | Materialized / direct CPU medians, s |
|---|---:|---:|---:|---:|
| 704 | 20.539 (19.664–20.652) | 19.123 (18.141–19.569) | 6.89% | 21.758 / 20.364 |
| 1,408 | 22.244 (20.786–23.867) | 20.002 (19.172–20.076) | 10.08% | 23.625 / 21.250 |

The reduction column is `1 - median(direct) / median(materialized)`.
All six pairs have lower direct wall and CPU times. Within-pair wall
reductions range from 5.24% to 7.75% at 704 points and 7.76% to 16.19%
at 1,408 points. The respective medians of the paired reductions are
6.89% and 9.75%, which differ from the ratio of separate medians above.
The old/new wall ranges do not overlap in either configuration. CPU median
reductions are 6.41% and 10.05%.

For the comparable post-FFT subtotal, each call first sums its measured
power/group/energy stages and then enters the median. That subtotal falls
from **2.345 to 0.479 s** at 704 points and from **2.761 to 0.651 s** at
1,408 points; it is smaller in all six pairs. Unchanged preparation and FFT
phases also vary. In particular, the largest 16.19% complete-call pair
includes about 1.915 s less unchanged input preprocessing time, so that entire
percentage cannot be attributed to the reduction code alone.

All **6,336 PSD and 6,336 NSD point comparisons** are bit-identical, with
identical frequency grids. All twelve calls have zero recorded relative
Parseval error and the expected Float32 NSD square-root relationship.
Read-back metadata confirms full-N FFT execution, materialized storage for
the reference, and ABI3/no power vector/O(J) state for the direct route.
Bit identity is an observed result for this input and build; the acceptance
gate remains 3 ppm with zero absolute tolerance for other inputs.

These three-pair results support an improvement for the measured profile.
They do not establish billion-sample or out-of-RAM performance: the combined
logical input/workspace size is 2.4 GB here, below the host's memory limit.
There was no deliberate cache flush or exclusive-host reservation. The
separate packed-phase experiment uses a different FFT algorithm and timing
cohort; its absolute times are not a controlled comparison with this table.
The direct option remains explicit, and other platforms have correctness
smokes rather than matching 100M performance pairs. See the
[integrated platform validation](performance-fast9.md).

In a built checkout of the measured revision, reproduce with:

```sh
python benchmarks/bench_disk_algorithms.py --compare-direct-power --sizes 100000000 --methods fftw --fft-memory-mib 128 --native-working-mib 384 --threads 8 --repeats 3 --point-factors 1 2 --max-seconds 600 --output direct-power-100m-pairs.json
```

This comparison mode validates the actual route and every returned value
after each complete-call timer. Its deadline is cooperative between calls;
use the portable runner for supervised campaigns with process deadlines.

## Recovered qualification and its limits

The independently recovered R3 experiment completed 30 small contract
cases and 24 configurations / 48 complete calls at N=100,000 and 1,000,000,
with six signals and two actual output counts. All **18,828 PSD and 18,828
NSD comparisons** were bit-identical. The maximum recorded relative
Parseval error was approximately 1.17e-15. Exact counts, errors and original
source pins are retained in the
[recovered follow-up evidence](../benchmarks/results/recovered-followup-20261012.json).
That source revision predates the integrated fast.9 adapter, and its
observer-instrumented timings do not establish a performance gain.

The later recovered
[one-billion-sample attempt](../benchmarks/results/recovered-final-attempt-20261012.json)
used the historical E source and an external frozen R3 helper. It stopped
after E completed, before R3 started: the fixed disk-space preflight was
short by 64,974,848 bytes. The E call took 1,153.282266 seconds of wall time
and 322.839617 seconds of CPU time, producing 744 PSD/NSD points. Those
are one completed reference observation, with no matching R3 observation.
This attempt did not run the integrated fast.9 adapter.

The recovered archive binds the separate local fast.9 correctness checks
to [PR #3's initial code](https://github.com/MichaDit/lpsd/commit/e6aa1603d01ef81b9cfb6cc03643029768712bd0).
No completed R3 billion-sample result or speedup follows from either set of
records. Native loops and repeated bin-power calculations can offset the
saved materialization, so the optional route requires its own measured
comparison rather than an automatic performance assumption.
