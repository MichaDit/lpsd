# Large data sets: LPSD/LNSD and FFTW under an 8 GiB memory limit

This report extends the [optimized FFTW comparison](performance-fftw-optimized.md)
to 100 million and one billion samples under the same **8 GiB physical-memory
limit**. The selected file-backed simple FFTW calculation completed in
**10.79 minutes at 1B samples**; the matched FFTW hybrid took **37.36 minutes**.
The corrected native LPSD/LNSD calculation took **36.86 minutes**. All six
selected calls at 100M and 1B samples completed and passed the final output
validation. At 1B, the hybrid took only **1.338% longer** than native LPSD/LNSD;
one observation per method does not establish a reliable ranking between
such close times.

The table contains the selected **complete file-backed calls**, each producing
PSD and NSD together. These are single observations, not repeated-call medians
or RAM timings. Wall times are rounded to milliseconds; the detailed sections
retain the recorded values and timer boundaries.

| Samples | Method | Joint PSD+NSD wall time (s) | Measurement status |
|---:|---|---:|---|
| 100,000,000 | Simple FFTW | 22.685 | Completed; validated |
| 100,000,000 | Matched FFTW hybrid | 101.851 | Completed; validated |
| 100,000,000 | Native LPSD/LNSD, corrected endpoint | 70.043 | Completed; exact comparison with earlier outputs passed |
| 1,000,000,000 | Simple FFTW | 647.514 | Completed; validated |
| 1,000,000,000 | Matched FFTW hybrid | 2241.305 | Completed; validated |
| 1,000,000,000 | Native LPSD/LNSD, corrected endpoint | 2211.715 | Completed; validated |

**Version and comparison scope:** the selected FFTW and hybrid observations
at both sizes use commit `5d5f5d8564928cd27135fdad63bfd042c9be5853`, with native
version **1.0.6+fast.5** and binary hash beginning `c1511109`. The selected
native LPSD/LNSD series uses the endpoint correction at commit
`68542b32c180ec26fe2821a6604ae436f1c9446b`, version **1.0.6+fast.6**, with binary
hash beginning `6e2c621a`. Its completed 100M outputs match the earlier RAM and
file-backed results exactly. The existing 10M/100M RAM observations retain
the original `cfa8a5a3` baseline and **fast.5**; they have not been remeasured
with fast.6. Full source identities and the narrowly scoped correction are
documented below.

The simple FFTW periodogram, the matched hybrid and native LPSD/LNSD are
different spectral estimators. The hybrid approximates the noise waviness and
frequency response of LPSD/LNSD. In the completed 1B calls, memory and file
access dominate the simple FFTW pipeline; the hybrid spends most of its time
on its 64 native low-frequency points, all of which run serially under the
2 GiB working-memory gate. These large-N measurements use white noise and
do not measure tone widths or leakage. Earlier adapter trials, timeouts and
the original failed native 1B attempt remain separately documented as history;
none supplies a timing in the selected table.

## Evidence and measurement identity

The selected results and their final checks are:

- `results_disk_final.json`: exactly one completed joint PSD+NSD observation
  for each of three methods at each of two sizes. It preserves the selected
  raw records and their explicit source cohorts; earlier attempts remain in
  its history and are not pooled into the six times.
- `disk_final_validation.json`: passed stored-output checks for both sizes,
  bound by SHA-256 to the exact final result bytes. It confirms unchanged
  common inputs and grids, finite nonnegative Float32 PSD/NSD, square-root
  consistency, matching native plans and bit-identical hybrid low points.
- `disk_final_summary.json`: the validated timings, within-method scaling,
  noise summaries and scoped phase/resource profiles. It records hashes of
  both the final result and its passed validator.
- `validation_large_n.json`: the consolidated numerical and provenance
  checks, including the explicit native transition and the 100M comparisons
  with older RAM and workspace observations. Its overall status is passed.

The supporting RAM and historical raw records are:

- `results_large_n_final.json`: the consolidated RAM results. Completed
  simple FFTW, cache-free hybrid and native LPSD jobs come from
  `results_large_n.json`; the completed 512 MiB prepared-hybrid jobs come from
  `results_matched_512.json`. The consolidation retains original observations,
  source hashes and job files. It does not average timings across these runs.
- `results_disk_blocked.json`: the V2 measurement series for the checked
  row-blocked I/O revision, using `/tmp/lpsd_disk_blocked_20261010`. Its three
  100M method records and its 1B simple FFTW record are complete. Its 1B
  hybrid and native LPSD records have process-timeout status. Their absence
  of completed call timings is retained.
- `disk_blocked_100m_validation.json`: the independent comparison of the
  completed row-blocked 100M outputs with their recorded RAM counterparts.
- `results_disk_local.json`: the first separate file-backed run using
  `/tmp/lpsd_disk_benchmark_20261010`. Its completed 100M observations are the
  preliminary control values reported here. Its 1B FFTW attempt was
  interrupted; it supplies no completed 1B runtime. These observations must
  remain attached to the recorded adapter source hashes after the I/O
  revision.
- `results_disk.json`: an earlier file-backed run in another working
  directory. Its 100M jobs completed, but the 1B storage preflight rejected
  execution. Those earlier timings are retained as history and are not pooled
  with the local-file control run.

The workspace-storage series is a separate revision and set of records:
`results_disk_workspace_100m.json` for the new 100M control and
`results_disk_extended.json` for the subsequent 1B series. All three 100M
controls are complete. `disk_workspace_100m_cross_validation.json` and
`disk_workspace_100m_metadata_validation.json` both report passed checks.
The 1B series started only after these controls passed. Its simple FFTW
record is complete, and `disk_workspace_1b_fftw_validation.json` reports
passed checks against the V2 result, including the shared input hash, grid
and exact PSD/NSD values. The hybrid record is also complete;
`disk_workspace_1b_matched_validation.json` reports passed checks of that
job's finite Float32 outputs, square-root relation and noise summaries.
The original native 1B failure and its raw logs remain unchanged in that
series. `results_disk_endpoint_fixed.json` records the separate fresh native
100M/1B run. Both records are complete and separately checked by
`disk_endpoint_fixed_100m_validation.json` and
`disk_endpoint_fixed_1b_validation.json`. The final consolidation selects
these corrected native observations alongside the original completed FFTW
and hybrid observations. No old observation is silently relabeled as a
measurement of the corrected implementation.

The endpoint change has explicit evidence in
`native_endpoint_fix_provenance.json`,
`native_endpoint_fix_validation.json` and
`native_endpoint_fix_plan_validation.json`. The selected comparison uses
these source cohorts:

| Measurement set | Version | Measured source revision | Native library SHA-256 prefix |
| --- | --- | --- | --- |
| Existing 10M/100M RAM observations | 1.0.6+fast.5 | Original `cfa8a5a3` baseline and recorded RAM drivers | Recorded in original RAM reports |
| File-backed FFTW and hybrid, 100M and 1B | 1.0.6+fast.5 | `5d5f5d8564928cd27135fdad63bfd042c9be5853` | `c15111091b1d8eef` |
| Fresh file-backed full native LPSD/LNSD, 100M and 1B | 1.0.6+fast.6 | `68542b32c180ec26fe2821a6604ae436f1c9446b` | `6e2c621a8d611ad14` |

The full native library identities are
`c15111091b1d8eef7ae73902d4833619533943243c72b89bd6036a1a0d07fd4b`
before the correction and
`6e2c621a8d611ad146abbda48bff4ccefee48f09acc58f124ea9655892f3f10f`
after it. The earlier completed native 100M disk control remains historical
fast.5 evidence. **The RAM measurements have not been repeated with fast.6.**
Binary identity is checked within each cohort; the declared native
before/after transition is an explicit, validated exception between cohorts,
not a removal of provenance checks for the FFTW/hybrid records.

The executable measurement definitions are
[bench_fftw_large_n.py](../benchmarks/bench_fftw_large_n.py) and
[bench_fftw_disk.py](../benchmarks/bench_fftw_disk.py). They record source and
native-binary hashes, library identity, input hashes, timing observations,
memory information, spectra and per-stage profiles. The first RAM harness
source and the original raw jobs were retained when the prepared-hybrid
cache setting was reduced.

The recorded environment is Linux x86-64 on an AMD EPYC 9V74 virtual host,
Python 3.12.14, NumPy 2.3.5 and actual FFTW 3.3.11, with SSE2, AVX, AVX2 and
AVX-512 variants available in that FFTW build. There are nine CPUs in the
affinity mask and an eight-CPU-equivalent cgroup quota. FFTW uses eight
threads, native LPSD uses up to eight frequency workers, and hybrid smoothing
uses four workers. The drivers restrict other numerical-library thread pools
to one and serialize method processes. The host's CPU frequency, shared-host
activity and operating-system cache state were not controlled.

The hard cgroup memory limit is **8,589,934,592 bytes = 8 GiB**. The system has
no swap capacity. A file mapping does not remove its resident pages from that
memory accounting. The benchmark does not change the cgroup limit or install
swap.

## Estimators, data and output units

| Method | Spectral calculation | Relationship to LPSD |
| --- | --- | --- |
| Native LPSD/LNSD | Original frequency-dependent segment lengths and starts, periodic Kaiser windows, order-zero local detrending and inherited power recurrence | Native reference using `kernel="fast"` |
| Simple FFTW | One full-record periodic Kaiser window, global mean removal, one-sided periodogram and logarithmic groups of adjacent Fourier-bin powers | Same output labels, different temporal and spectral averaging |
| Matched FFTW hybrid | A full-record periodic Tukey window with total ramp fraction 0.05, analytical Kaiser power-response weights, and an explicit native LPSD subset | Approximation intended to reproduce similar noise waviness and line widths; not the same finite-record estimator |

All cases use `sample_rate=50`, `psll=200`, `n_frequencies=1000`,
`n_averages=100` and detrending order zero. The Kaiser-derived overlap is
`0.7658465180619339`. In the hybrid, `psll=200` describes the target Kaiser
response and the native subset; it is not a claim that the full-record Tukey
window has 200 dB sidelobe suppression.

The input is one Gaussian white-noise realization per length, with one-sided
source NSD **10 nV/sqrt(Hz)** and source PSD **1e-16 V^2/Hz**, equivalently
**100 nV^2/Hz**. The seed sequence is `[20261010, 0, N]`. Thus methods at one
length receive the same samples, while different lengths use different
realizations, not a common prefix. Repeating a timing on the same samples does
not create an independent noise realization.

For samples expressed in volts, PSD has units V^2/Hz and NSD has units
V/sqrt(Hz). Samples, FFT values and power arithmetic are Float64/Complex128.
The prepared hybrid may store its already normalized smoothing weights in
Float32; this does not convert its input or FFT to Float32. Public PSD output
is rounded to Float32. NSD follows the existing Complex64 square-root path
**after** power averaging and that PSD rounding. Requesting both outputs
reuses one spectral calculation.

## RAM timing scopes

Each N/method combination runs in a separate child process. The two output
requests, PSD and NSD, are shuffled within each repetition. The table records
three observations per output and scope, following two prepared warm-ups or
one complete-call warm-up. Minimum–maximum ranges are observed ranges, not
confidence intervals.

| Scope | Included in the timed call | Excluded from that call |
| --- | --- | --- |
| Reuse | Complete data-dependent `compute`, including its input checks, FFT or segment work, normalization, smoothing and output construction | Object preparation, FFTW planning, retained window/weight/q preparation and object destruction |
| Fresh class | Forget FFTW wisdom; construct the configured object; compute; close | Process start, imports, native-library loading and signal creation |
| Hybrid single call | `estimate_once`: construct without persistent weight/q caches, compute using deferred uncached normalization, close | Process start, imports, library loading and signal creation |
| Native complete | Public `lpsd_fast.lpsd` call, including input validation, frequency planning, windows, coefficients, original segments and output construction | Process start, imports, library loading and signal creation |

Signal generation, hashes, independent numerical/energy checks, plotting and
the additional phase-profile calls are outside ordinary RAM timing
observations. Operating-system caches are not cleared. A fresh object is
therefore not a claim of a cold operating-system cache.

The simple pipeline uses `operations="native_grouped"`, logarithmic power
aggregation, Float32 output and `ESTIMATE`. The prepared hybrid uses native
pre/postprocessing and smoothing, Float32 cached weights, and **512 MiB total
persistent cache**. Its requested weight cap of 1536 MiB is capped by that
total. Any bytes left after actual weight allocation are available to native
q coefficients. Its temporary concurrency budget is 2048 MiB. `estimate_once`
forces both persistent caches to zero. The native complete call uses a
4096 MiB concurrency budget and a 128 MiB window cache. All are subject to the
same hard 8 GiB cgroup limit; the individual budgets are not RSS limits.

This is a new large-N profile. The earlier optimization study used 2048 MiB
total hybrid cache and sometimes a different FFTW thread count. Its timings
must not be appended to this series as though the configuration were fixed.
An initial 1024 MiB prepared-hybrid attempt at 100M failed under the RAM
process's address-space guard; the historical failure remains in the raw
record rather than being replaced by a fabricated timing.

### Completed RAM observations

Every entry is **median [minimum–maximum], seconds**, for three calls. PSD and
NSD are separate complete spectral evaluations, not a timed square root of a
previous result.

| N | Method and scope | PSD, s | NSD, s |
| ---: | --- | ---: | ---: |
| 10,000,000 | Simple FFTW, reuse | 0.053699 [0.053218–0.056191] | 0.049658 [0.049309–0.051958] |
| 10,000,000 | Simple FFTW, fresh | 0.219275 [0.216013–0.220356] | 0.225674 [0.224022–0.231748] |
| 10,000,000 | Hybrid, reuse | 0.714329 [0.597821–1.141111] | 0.833569 [0.569662–1.019936] |
| 10,000,000 | Hybrid, fresh class | 1.697971 [1.498385–1.774439] | 1.712397 [1.617466–1.918323] |
| 10,000,000 | Hybrid, single call | 1.342879 [1.341075–1.401322] | 1.356437 [1.275856–1.757958] |
| 10,000,000 | Native LPSD/LNSD, complete | 1.304308 [1.286673–1.524655] | 1.519635 [1.487462–2.106361] |
| 100,000,000 | Simple FFTW, reuse | 0.594474 [0.592895–0.599250] | 0.589512 [0.566966–0.597641] |
| 100,000,000 | Simple FFTW, fresh | 1.819993 [1.812112–1.831387] | 1.882835 [1.836648–1.923112] |
| 100,000,000 | Hybrid, reuse | 18.828544 [18.229045–19.923824] | 19.130901 [18.104525–19.376179] |
| 100,000,000 | Hybrid, fresh class | 26.195586 [24.075962–26.823930] | 27.413293 [26.856321–27.601964] |
| 100,000,000 | Hybrid, single call | 21.307377 [20.663402–21.653321] | 19.799917 [19.088660–19.828554] |
| 100,000,000 | Native LPSD/LNSD, complete | 20.130438 [19.889806–20.827437] | 22.645526 [20.735201–23.456251] |

PSD and NSD share almost all computation. Differences between their separately
sampled medians do not show that taking a square root changes the fundamental
cost of the estimator. Likewise, a reuse-to-complete comparison describes two
usage patterns with different preparation scopes; it does not isolate an
algorithmic speedup.

### Process memory

These are recorded **process-lifetime peak RSS values**, including setup,
warm-ups, timed calls and independent validation. They are not per-call peaks
and are not measurements of heap storage alone.

| Method process | 10M peak RSS, GiB | 100M peak RSS, GiB |
| --- | ---: | ---: |
| Simple FFTW | 0.479 | 3.655 |
| Prepared hybrid, 512 MiB total cache | 2.218 | 6.759 |
| Cache-free hybrid single call | 1.669 | 6.278 |
| Native LPSD/LNSD | 1.611 | 4.976 |

The RAM driver also sets a Linux virtual-address-space guard of
8,053,063,680 bytes, or 7.5 GiB. This is an additional process guard, distinct
from RSS and from the cgroup's physical-memory accounting.

## What actually scales with N

`n_frequencies=1000` and `n_averages=100` are planning targets. They do not mean
that precisely 1000 frequencies or 100 segments are computed. The unchanged
[planner](../lpsd/_helpers.py) determines the actual frequency spacing,
segment lengths and segment counts from the record length.

| N | Actual frequencies | Lowest output frequency, Hz | Segment lengths L | Segment counts K | Sum of L*K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000,000 | 651 | 5e-6 | 129–10,000,000 | 1–331,059 | 27,072,536,884 |
| 100,000,000 | 704 | 5e-7 | 113–100,000,000 | 1–3,779,380 | 294,316,951,682 |
| 1,000,000,000 | 744 | 5e-8 | 100–1,000,000,000 | 1–42,707,028 | 3,122,351,444,856 |

All three completed 1B methods use the same **744 frequencies** on the
original planner's grid. The final native result confirms the lengths,
counts and logical coverage in the last row; two points have `K=1`.

`sum(L*K)` describes logical segment-sample coverage of the original direct
calculation. It is neither a byte count read from storage nor a hardware
instruction count. SIMD batches, overlapping segments and filesystem caching
can share work or reads.

The planner uses `logfact = (N/2)**(1/1000) - 1`. Away from its low-frequency
adjustments, the target spacing is approximately `f*logfact`, with
`L ~= sample_rate / spacing`. Increasing N extends the range to lower
frequencies, but with the same requested J it also changes the spacing in a
fixed higher-frequency band. For example, the planning factor increases from
0.0178856 at 100M to 0.0202321 at 1B, about 13.1%. This is an analytical
property of the planner, **not a measured 1B peak width or runtime**. Ten times
as many samples therefore does not imply ten times narrower output peaks at
every frequency in this logarithmic estimator.

### Why the memory-limited hybrid loses much of its earlier advantage

At 10M, the 512 MiB profile caches all 568 smoothing kernels and two of its
83 native LPSD points. At 100M it caches 525 of 631 smoothing kernels, leaving
106 to be computed without stored weights. Only 6,857,264 bytes remain for
the native coefficients, too little for any of its 73 long LPSD points. The
actual weight cache occupies 530,013,648 bytes; there is no retained native
q vector at 100M.

An additional 100M profile call took 20.595 s, including approximately
15.649 s for the native LPSD subset, 4.360 s for frequency smoothing and
0.357 s for FFT execution. These are subdivisions of that separate profile
call, **not a decomposition of a median** and not replacements for the median
timing observations. Optimizing only
the FFT cannot remove the dominant long-point preparation and segment work
in this configuration.

The full native profile similarly retains substantial segment work. Summed
per-worker elapsed times overlap and must not be added as if they were call
wall time. The different temporary budgets and the caches that actually fit
also affect concurrency. Neither the RAM series nor the measured file-backed
size pairs establish a universal runtime law beyond these observations.

## What the white-noise spectra establish

The 100M noise summary covers **428 output points from 0.01 to 20 Hz** out of
the full 704-point grid. Define `u = NSD / source_NSD`. The following
descriptive values come from the one stored white-noise realization.

| Method | Mean u | Sample standard deviation of u, % | RMS adjacent difference / sqrt(2), % |
| --- | ---: | ---: | ---: |
| Native LNSD | 1.000321 | 0.408696 | 0.224380 |
| Matched FFTW hybrid | 1.000491 | 0.383816 | 0.168459 |
| Simple FFTW | 1.000599 | 1.900484 | 1.865855 |

The hybrid's broad-band waviness is close to LNSD in this record, while its
adjacent-point roughness is somewhat lower. The simple logarithmically
grouped periodogram remains visibly rougher. Frequency points are correlated;
these are not standard errors, confidence intervals, or a Monte Carlo
estimate of estimator variance. Different point densities also matter when
comparing summaries between lengths: the same 0.01–20 Hz interval contains
493 points at 10M.

The source PSD/NSD is a model reference, not an exact expectation for every
finite-window, detrended low-frequency output. The present large-N input
contains **no injected tones**. Consequently this run measures no tone peak
height, PSD/NSD half-maximum width, two-tone resolution or far-out leakage
floor. The earlier tone/response experiments remain separate evidence; this
white-only run does not extend their numerical guarantees to 100M or 1B.
Output agreement between RAM and disk adapters validates those execution
paths on the tested inputs, not equality of the simple FFTW or hybrid
estimator to LPSD for arbitrary signals.

### The inherited segment recurrence is deliberately retained

The native code preserves the upstream mean update indexed from zero:

```text
M_0 = P_0
M_i = M_(i-1) + (P_i - M_(i-1)) / i, for i >= 1
```

The divisor is `i`, not `i+1`. With K >= 2, exact arithmetic therefore leaves
the mean of segments 1 through K-1; segment zero's contribution cancels in
the second update. At K=1 the first segment is used. In floating-point
arithmetic, that cancellation can itself retain rounding effects, especially
for a very large initial transient. All planned segment calculations still
run and use the same recurrence; this benchmark does not silently correct
the estimator in one path.

The inherited variance accumulator also overwrites its update rather than
forming a conventional accumulated second moment. `psd_std` is not requested
or presented as a validated uncertainty estimate here. The descriptive
frequency-domain waviness above is computed directly from the stored NSD
outputs and is separate from that legacy statistic. See
[fast_dft.c](../lpsd_fast/_native/fast_dft.c) and the
[numerical notes](numerics.md).

## One billion samples: RAM feasibility and file-backed execution

The existing RAM constructors are rejected before input allocation at 1B.
Known arrays alone exceed the hard limit:

| Existing RAM path | Known simultaneously required bytes including Float64 input | At 1B, GiB |
| --- | --- | ---: |
| Simple FFTW | `32*N + 16` for even N | 29.80 |
| Matched FFTW | `36*N + 24` for even N, before caches and native low-point work | 33.53 |
| Native LPSD | `32*N` at the longest point, L=N | 29.80 |

FFTW plans and other temporary allocations can add to these lower bounds.
Reducing a cache cap cannot make those mandatory arrays disappear. The
corresponding RAM status is `not_executed_memory_limit`, with **no execution
time**, not a zero-second measurement or a failed accuracy result.

### A separate execution strategy

The file-backed driver changes storage and FFT factorization, while retaining
each method's spectral definition. It does not remove input samples,
frequencies, segment lengths or averages.

[DiskFFT](../benchmarks/_fftw_disk.py) evaluates the same N-point forward DFT
as a two-factor Cooley–Tukey decomposition, `N=A*B`, using bounded RAM tiles
and FFTW `ESTIMATE` batch plans of lengths A and B. The work file contains N
Complex128 values. Before execution, its two real halves stage the real FFT
input and the window. During execution those values are overwritten. The
output matrix has coordinates `[k2, k1]`, corresponding to original bin
`k = k2 + A*k1`; powers are restored to original frequency order before
aggregation. This avoids creating one enormous full-length FFTW plan and its
potentially large anonymous twiddle tables. Its operation order can change
floating-point rounding relative to a full-record R2C transform.

The checked I/O revision gathers and scatters bounded, contiguous portions
of file rows into the existing Complex128 twiddle scratch. It performs the
required transpose between that staging area and the FFT tile in RAM. This
avoids the previous column-wise copy order across the large mapping. The
twiddle calculation fully rewrites the scratch before using it again. The
factors, FFTW batch plans, twiddle arithmetic, frequency-bin order and
normalization are unchanged; this revision adds no large data buffer.

Optional `madvise` hints request random access during the column and power
reordering phases, sequential access during the row phase and the new
workspace-power packing phase, and normal access otherwise. These are
best-effort filesystem hints, not cache eviction, extra memory capacity or a
correctness requirement. Metadata records whether the hints were available.
Bounded progress callbacks report phase counters during long operations;
they do not change the input or transform.

[DiskLPSDSubset](../benchmarks/_lpsd_disk.py) gives each active frequency of
length L a disjoint region of L Complex128 slots in the same mapped workspace.
Its window aliases the real coefficient region before coefficient generation.
It uses the existing native normalization, projection and segment functions,
the original mode 2/3 selection, and ordinary segment arithmetic without the
optional bounded FMA path. A shared scratch pool and the existing
`24*L+8192` concurrency reservations control simultaneous points; an oversized
point runs alone. Windows and coefficients are not persistently cached.

The hybrid first computes and smooths the file-backed Fourier powers, then
closes FFTW plans and mappings and reopens the workspace for the native
subset. It retains no smoothing-weight cache. The file requirements depend
on the selected power-storage mode:

| File-backed path | Maximum large files used together | At 1B, GiB |
| --- | --- | ---: |
| Simple FFTW or hybrid, V2/external powers | `8*N` input + `16*N` complex workspace + `4*N+8` positive powers | 26.08 |
| Simple FFTW or hybrid, new workspace powers | `8*N` input + `16*N` reused workspace | 22.35 |
| Native LPSD | `8*N` input + `16*N` coefficient workspace | 22.35 |

The storage preflight adds a 512 MiB reserve to the selected maximum file
requirement: `28*N+8` bytes with external powers, or `24*N` with workspace
powers/native-only LPSD. This counts file storage separately from virtual
mappings and physical memory.

### Reusing the workspace for Fourier powers

`--power-storage workspace` removes the separate positive-power file. It
requires an even B factor; unsupported factorizations are rejected before
destructive power extraction, while the external-power path remains
available. At 1B, the existing factors `A=31,250`, `B=32,000` satisfy this
condition. The algorithm retains the same Fourier values and Float64
square/add/scale operations:

1. Save the Nyquist value `Y[0,B/2]` before overwriting its source.
2. Copy all positive Fourier values of a bounded row batch to the existing
   RAM tile. Only then write that batch's powers compactly at the start of
   the workspace. After r consumed rows, the compact destination ends at
   byte `4*B*r`, while the next unread FFT row starts at byte `16*B*r`.
3. Transpose the packed powers from `[0,4*N)` into the disjoint range
   `[8*N,12*N+8)`, restoring the original positive-frequency order. Apply
   the usual DC/Nyquist factors exactly once.

The returned Float64 power array is a separate mapping of the same file,
starting at byte `8*N`. Thus **physical file lengths total 24N bytes**, but
input, FFT workspace and the additional power view can still occupy
**28N+8 virtual mapping bytes** together. The conservative FFT-path address
guard remains those mapping sizes plus 2 GiB. Both FFT and power mappings
close before the hybrid reopens the same workspace for native coefficients;
the workspace file is removed only after its final consumer.

The cgroup's **8 GiB physical-memory limit remains unchanged**. File pages
remain normally pageable filesystem data, with no swap or copy-on-write
substitute. Process peak RSS can count the same physical file page more than
once through aliased mappings. It therefore does not represent unique
physical memory or total cgroup memory usage.
The configured 128 MiB FFT buffer budget covers explicit anonymous adapter arrays,
not all FFTW internals or resident file pages. The new packing and transpose
work remains inside the complete-call timer. The completed 100M and 1B calls
of all three methods are reported below; their measured times are separate
from the reduction in required file capacity.

At preparation for the new series, the working filesystem reported
25,275,891,712 bytes free, exceeding the 1B workspace-storage preflight of
24,536,870,912 bytes (`24*N + 512 MiB`). This is a recorded capacity check,
not a prediction of future free space or execution time. The cgroup limit
and swap configuration were not changed; its OOM and OOM-kill counters were
both zero at that checkpoint and after all selected calls completed. The
corrected native 1B process reached a peak RSS of 8,260,460,544 bytes
(7.693 GiB); its complete resource observations are reported below.

### Why all 64 native hybrid low points run serially at 1B

The native concurrency gate reserves `24*L+8192` bytes per frequency point
against the configured 2048 MiB budget. This is a scheduling reservation,
not the point's measured resident memory and not an assertion that all of
its mapped data fits in 2 GiB.

At 1B, the completed hybrid's final plan contains 64 native low points, and
**every one has `L > N/16 = 62,500,000`**. Their lengths range from
63,214,728 to 1,000,000,000 samples, with segment counts from 1 to 64. There
are no short `L < 32` fallback points in this plan. Each selected point
reserves more than 1,500,008,192 bytes before
clamping. Two such reservations would exceed 3,000,016,384 bytes and cannot
fit within the 2,147,483,648-byte gate together. At
`L >= 89,478,144`, a single reservation fills the whole gate; still larger
requests are treated as an oversized point that runs alone. Consequently,
**all 64 native points must run serially** under this gate despite eight
configured workers. Increasing the worker count alone cannot admit a second
one. The full native estimator can admit shorter points concurrently when
their combined reservations fit; the statement about all 64 points applies
specifically to the hybrid subset.

Each point must also generate its mapped window/coefficients and revisit
input and scratch pages under the physical-memory limit. Live CPU and I/O
activity during the V2 timeout attempts is consistent with this continuing
work, rather than evidence of a resource-pool deadlock. Neither the gate
arithmetic nor a partial progress record supplies a completed runtime.

The completed workspace hybrid reports all 64 native points finished. At
the first point, with one segment (`K=1`), its elapsed times were approximately:

| Selected stage of the first completed native hybrid point | Elapsed time, s |
| --- | ---: |
| Kaiser window construction | 14.105 |
| Complex coefficient generation | 41.576 |
| Native projected-coefficient preparation | 43.207 |
| Native segment calculation | 3.9996 |
| Complete point wall time | 106.054 |

The preparation stage has substantive work even when only one segment is
required. The existing `prepare_projected_coefficients` function in
[fast_dft.c](../lpsd_fast/_native/fast_dft.c) traverses the length-L real and
imaginary coefficient arrays twice: first accumulating their moments in
`long double`, then removing the fitted order-zero component using
`long double` arithmetic and storing Float64 coefficients. The uncached
file-backed path repeats that work for every native point. This explains
why the cost of a long point cannot be represented by its final segment
calculation alone.

The selected stage times are elapsed times, including any waiting within
those stages; they are not a CPU-versus-I/O decomposition or a complete
partition of point wall time. Observed cgroup CPU usage during the early
part of the subset was near one core, consistent with serial admission.
This single-point profile does not predict the other points or replace the
separately measured complete hybrid call reported below.

The new optional native callback reports a point only after its calculation,
normalization and both resource releases have completed. Its records contain
copied scalar metadata. The workspace series sets `--lpsd-progress-interval-s 0`
to retain a checkpoint for every finished point. This is not a heartbeat
inside a still-running point. `lpsd_points_completed` means the frequency
points are finished; output assembly and finalization follow. Callback time
is included in complete wall time and can overlap other workers, so it is
not subtracted as an independent correction.

### Preliminary local-file control at 100M: first implementation

These are **one observed joint PSD+NSD call per method**, not medians and not
separate PSD/NSD measurements. The values come from `results_disk_local.json`
and describe the first file-backed adapter, before the strided-I/O revision.
They are retained as historical measurements, not as final results for the
revised adapter. The completed control for the same implementation as the
successful 1B FFTW call appears in the next section.

| Method | Joint complete wall time, s | Process-lifetime peak RSS, GiB |
| --- | ---: | ---: |
| Simple FFTW | 21.856467385 | 2.804 |
| Matched FFTW hybrid | 100.987869507 | 2.804 |
| Native LPSD/LNSD | 64.563498167 | 2.347 |

The disk timer includes working-file construction, estimator preparation,
input checks, required paging, transformation/segments, outputs, bounded
energy diagnostics, progress checkpoints and removal of the disposable work
files performed by the pipeline. Signal creation, input hashes, process start,
imports and library loading are excluded. Work files are temporary, not
durable deliverables; no final `fsync` is forced. Operating-system caches are
not deliberately cleared, and input hashing can affect their state.
In the revised adapter, complete call/FFT times also include the internal
progress callbacks. Individual gather, FFTW and scatter phase times do not
capture all callback costs; their sum is not a substitute for measured total
wall time.

For example, the completed hybrid control includes 23.292 s in the blocked
FFT stage, 5.143 s in frequency smoothing, and 65.561 s in the native subset
and output stage. These are elapsed stages of that single observed run; the
timing does not isolate SSD bandwidth or computational throughput alone.
This control shows the cost of the separate storage strategy at a length
where the ordinary RAM paths also fit. Do not divide a joint call by two,
compare it with a reuse median as a like-for-like speedup, or treat the earlier
working-directory run as an additional identical repetition.

### V2 row-blocked series: completed observations and process timeouts

The V2 run uses the checked row-blocked adapter at commit
`ee4b4a7bde9c88c17d8d7cbe2ff1a05eaed438db`,
`/tmp/lpsd_disk_blocked_20261010` and the raw output
`results_disk_blocked.json`. The 100M controls and 1B simple FFTW call below
have completed. Every time is **one joint complete PSD+NSD call**, not a median
or separate timing for each output. The two other 1B processes timed out;
their rows deliberately contain no completed call time.

| Method at N=100,000,000, checked row-blocked adapter | Status | Joint complete wall time, s | Process-lifetime peak RSS, GiB |
| --- | --- | ---: | ---: |
| Simple FFTW, file-backed | Completed | 20.725447026 | 2.804 |
| Matched FFTW hybrid, file-backed | Completed | 102.654059468 | 2.805 |
| Native LPSD/LNSD, file-backed | Completed | 64.056014300 | 2.347 |

| Method at N=1,000,000,000 | Status | Joint complete wall time, s | Process-lifetime peak RSS, GiB |
| --- | --- | ---: | ---: |
| Simple FFTW, file-backed | Completed | 717.431748803 | 7.720 |
| Matched FFTW hybrid, file-backed | Process timeout, 2400 s limit | — | — |
| Native LPSD/LNSD, file-backed | Process timeout, 2400 s limit | — | — |

The timeout applies to the **entire child process**, including imports and
native-library startup, the input hashes before and after the timed call,
computation and report finalization. It is not a 2400 s call timing, and it
does not establish that the timed calculation itself ran for at least 2400 s.
A stale `running` child checkpoint is retained evidence of the last update;
the controller's timeout status is authoritative for that attempt.

#### What the completed 1B FFTW call measures

The simple FFTW call completed in about **11.96 minutes**, with 251.432182559 s
of process CPU time summed across threads. Its recorded peak RSS was
8,289,214,464 bytes under the unchanged 8 GiB cgroup limit. The transform used
the exact factorization `1,000,000,000 = 31,250 * 32,000` and produced 744
logarithmic PSD/NSD output points.

The measured bottleneck is file access and data movement. The two gather
stages alone account for about 76% of the full call's wall time:

| Selected subphase in the completed 1B call | Elapsed time, s |
| --- | ---: |
| Column gather for the first FFT stage | 362.570514483 |
| Gather restoring positive powers to original frequency order | 183.004541072 |
| FFTW execution in the column stage | 2.840725363 |
| FFTW execution in the row stage | 2.609139774 |
| Twiddle calculation and multiplication | 21.170907522 |

The **5.449865137 s** obtained by adding the two FFTW execution rows is only
time inside those FFTW subtransform calls. It excludes twiddles, gathers,
scatters, real-to-complex expansion, input validation and preprocessing,
power extraction, output construction and other timed work. It is **not a
hypothetical in-RAM complete time**, an alternative 1B measurement, or a
basis for comparing with complete LPSD/LNSD calls. The selected subphases are
not a complete wall-time partition.

The process I/O counters increased by 97,396,449,280 read bytes and
82,619,879,424 write bytes during the monitored pipeline: about 97.40 and
82.62 decimal GB, respectively. These counters describe process-attributed
storage activity; they are not the logical file sizes, a direct SSD-bandwidth
measurement or a promise that every temporary write was made durable. The
large difference between the 100M and 1B calls includes the changed paging
and storage behavior, so it does not establish a general scaling law.

The included Parseval diagnostic reports relative energy disagreement
`1.5778795068414844e-16`. This checks the weighted time-domain and complete
Fourier-power normalization. It does not constitute a same-input 1B RAM
comparison, a per-bin error bound, or a tone-resolution test.

#### Earlier incomplete attempts

The first local-file 1B FFTW attempt reached window construction, input
preprocessing and a bounded energy diagnostic. It then spent several minutes
in column-wise file gathering, with heavy read I/O and little write progress.
The controller was interrupted with exit status 130. This is an **incomplete
attempt**, not a completed transform or a runtime observation for the table.
The subsequent hybrid and native 1B jobs in that attempt did not run. The
checked row-blocked access revision responds to this observed I/O problem;
the completed records above measure that revised implementation.

The earlier `results_disk.json` run did not execute a 1B transform: its disk
capacity check failed before input/estimator execution. That historical
storage result must not be described as a 1B runtime or a 1B out-of-memory
crash. The completed simple FFTW call supplies no runtime for the timed-out
hybrid or native method. The separate workspace-storage measurements belong
in their own series. A timeout or other incomplete job remains a status entry
with no completed runtime. No scaling projection is a substitute for
measurement.

### Workspace-storage observations and the corrected native cohort

The new series uses `--power-storage workspace`, a **10800 s whole-process
limit**, eight FFTW/native workers, four smoothing workers, the existing
128 MiB FFT buffer budget, a 2048 MiB native concurrency budget, and
`--lpsd-progress-interval-s 0`. The initial workspace revision changed
storage and reporting; the later native cohort additionally applies the
narrow endpoint correction described below. Its observations are not pooled
with V2 as if the execution strategy were unchanged.
The original 100M control completed from the frozen
`5d5f5d8564928cd27135fdad63bfd042c9be5853` sources and passed both independent
validators. Those same frozen sources were used for the initial 1B series,
including its failed full-native attempt. The corrected native 100M/1B
cohort is separately recorded under `68542b32c180ec26fe2821a6604ae436f1c9446b`.
The completed entries below are single joint PSD+NSD calls; they are not
medians or separate PSD/NSD timings. Both corrected native measurements and
the final three-method comparison have completed and passed validation.

| N | Method and source cohort | Status | Joint complete wall time, s | Process-lifetime peak RSS, GiB |
| ---: | --- | --- | ---: | ---: |
| 100,000,000 | Simple FFTW, `5d5f5d8` / fast.5 | Completed; validated | 22.684656378 | 2.805 |
| 100,000,000 | Hybrid, `5d5f5d8` / fast.5 | Completed; validated | 101.850774360 | 2.805 |
| 100,000,000 | Native LPSD/LNSD, `5d5f5d8` / fast.5 | Historical completed control; validated | 67.440235931 | 2.358 |
| 100,000,000 | Native LPSD/LNSD, `68542b3` / fast.6 | Completed; exact transition checks passed | 70.042952008 | 2.358 |
| 1,000,000,000 | Simple FFTW, `5d5f5d8` / fast.5 | Completed; validated against V2 | 647.514033935 | 8.990 |
| 1,000,000,000 | Hybrid, `5d5f5d8` / fast.5 | Completed; final validation passed | 2241.304918831 | 9.258 |
| 1,000,000,000 | Native LPSD/LNSD, `5d5f5d8` / fast.5 | Failed: unsafe segment bounds; 732 point reports | — | — |
| 1,000,000,000 | Native LPSD/LNSD, `68542b3` / fast.6 | Completed; final validation passed | 2211.714508946 | 7.693 |

All three original fast.5 100M PSD/NSD outputs are bit-identical to their V2 disk counterparts.
Against the recorded RAM results, simple FFTW and native LPSD are also exact;
the hybrid retains the previously observed maximum relative differences of
`6.855234516544525e-8` in PSD and `8.946151414865158e-8` in NSD. Therefore the
100M noise-waviness values reported earlier are unchanged by this storage
revision. This establishes output preservation on the checked white-noise
record; it does not make the different estimators equivalent.

The fresh fast.6 native 100M call completed in **70.042952008 s**, with
216.529188805 s of summed process CPU time. All 704 PSD and 704 NSD values
are exactly equal to both its old RAM and old workspace counterparts under
the declared native-binary transition. The common input, grid and window
plan checks passed, and all 73 native low points of the unchanged 100M
hybrid match this corrected full-native output bitwise. These are the checks
in `disk_endpoint_fixed_100m_validation.json`; they do not reuse an old time
as a measurement of the corrected version. The two native 100M observations
are single calls, so their time difference does not isolate the cost of the
endpoint guard.

The selected size pairs have these wall and summed process-CPU observations:

| Method | 100M wall, s | 1B wall, s | Wall ratio, 1B / 100M | 100M CPU, s | 1B CPU, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple FFTW, fast.5 | 22.684656378 | 647.514033935 | 28.544141165 | 24.203905580 | 237.070318646 |
| Matched FFTW hybrid, fast.5 | 101.850774360 | 2241.304918831 | 22.005772002 | 149.806270527 | 1888.609009123 |
| Native LPSD/LNSD, fast.6 | 70.042952008 | 2211.714508946 | 31.576546184 | 216.529188805 | 3411.382031178 |

Each ratio uses the same measured source/binary cohort and computational
settings within its method. The tenfold increase in sample count changes
the original frequency plan and the paging regime; the observed factors
are not asymptotic complexity estimates or predictions for another host.
The corresponding CPU-time factors are 9.794713414, 12.607009056 and
15.754836796. CPU time is summed across threads and cannot replace elapsed
wall time. Filesystem cache state and other host I/O were not controlled
as independent cold/warm experiments.

At 1B, simple FFTW took about 29.3% of native LPSD/LNSD's wall time, while
the hybrid took **1.337894641% longer** than native. The first comparison
also changes the estimator and its measured noise waviness. The small
hybrid/native difference is from one complete call per method, so it does
not establish a stable speed ranking or the cost of the endpoint correction.

#### Completed 1B simple FFTW with workspace powers

The 1B workspace call took **647.514033935 s**, about **10.79 minutes**, with
237.070318646 s of process CPU time summed across threads. The whole call is
28.544141165 times the corresponding 100M workspace call. This is an observed
ratio between two sizes under the memory and storage constraints of this
host, not a general runtime scaling law. Both timings include the same
joint PSD+NSD output scope.

The principal pipeline phases were:

| Phase in the completed 1B workspace call | Elapsed time, s |
| --- | ---: |
| Window construction and normalization | 19.084409592 |
| Input validation, copy, mean and window application | 46.059252358 |
| Complete blocked Fourier transform phase | 470.515328891 |
| Extraction and ordering of one-sided powers | 108.041109246 |

These selected phases do not form a complete wall-time partition: the
complete call also includes planning, energy diagnostics, logarithmic
grouping, output construction, callbacks and cleanup. Within the transform
phase, column gathering alone took 376.019606503 s. The two FFTW execution
subphases took 3.007068714 s and 2.236405259 s, a sum of **5.243473973 s**.
That sum is only the time inside the two families of FFTW subtransform calls.
It is neither the 470.515328891 s complete transform phase nor the
647.514033935 s complete spectral-estimation call, and it supplies **no 1B
RAM runtime**. File access and data movement remain the dominant measured
costs.

The internal power-extraction timer decreased from 193.848463454 s in V2 to
108.040506576 s in this workspace run. That comparison has the same internal
timer scope in both revisions; it does not establish a general reduction in
storage traffic. Process-attributed I/O counters in the new call increased
by **108,815,646,720 read bytes** and **141,007,237,120 write bytes**, about
108.82 and 141.01 decimal GB. Both counters are higher than in the V2
observation, despite the smaller required file capacity and shorter
observed total call. These are single calls on a shared host with
uncontrolled filesystem cache state, not repeated I/O performance trials.
The counters also do not assert that every temporary write was made durable.

Process-lifetime peak RSS was **9,653,334,016 bytes**, about **8.990 GiB**.
The aliased power view can cause process RSS to count the same physical
file page through more than one mapping. This value therefore does not
measure unique physical pages or the cgroup's total memory usage. The cgroup
limit remained **8,589,934,592 bytes**, swap remained unavailable, and the
recorded OOM and OOM-kill counters remained zero. No resource limit was
increased for this call.

The saved input remained unchanged. All **744 PSD and 744 NSD values** and
their frequency grid are exactly equal to the completed V2 disk outputs;
the validator also confirms the same input and native-binary hashes. The
Parseval diagnostic again gives relative disagreement
`1.5778795068414844e-16`. These checks establish output preservation for
the measured white-noise input and the workspace-storage change. They are
not a 1B RAM comparison or a peak-resolution measurement.

#### Completed 1B hybrid with workspace powers

The selected hybrid completed in **2241.304918831 s**, about **37.36 minutes**,
with 1888.609009123 s of process CPU time summed across threads. Its joint
complete PSD+NSD call took **22.005772002 times** its same-source 100M control
of 101.850774360 s, and **3.461399756 times** the completed 1B simple FFTW
call. These ratios compare the recorded single complete calls. They are not
median speedups, a general scaling law, or a claim that the simple and
hybrid estimators produce equivalent spectra.

| Phase in the completed 1B hybrid call | Elapsed time, s |
| --- | ---: |
| Window construction and normalization | 25.306865042 |
| Input validation, copy, mean and window application | 53.333806483 |
| Complete blocked Fourier transform phase | 440.441444980 |
| Extraction and ordering of one-sided powers | 101.708376518 |
| Matched frequency smoothing | 45.071969871 |
| Native low-frequency calculation and output assembly | 1572.408377376 |

The selected phases are measurements inside this complete call, with the
remaining planning, diagnostics, callbacks and cleanup also included in its
total. The native low-frequency phase accounts for about **70.16%** of the
total wall time; frequency smoothing accounts for about **2.01%**. Thus the
main measured cost of this hybrid configuration is its retained native
low-frequency work. The 440.441444980 s transform phase includes file access
and movement as well as FFTW execution.

The final metadata confirm **64 native points and 680 smoothed points** on
the 744-point output grid. The native subset spans `5e-8` to
`2.5574987528287625e-5 Hz` and finished all 64 points. All satisfy the length
condition proved above, so they ran serially under the 2 GiB gate. The
recorded `effective_workers=8` is the configured pool size, not eight active
points at once. Their logical segment coverage sums to **230,671,322,343
sample iterations** (`sum(L*K)`); that is not a count of physical disk reads.

Across these points, the stored elapsed-time sums are 588.091013883 s for
native projected-coefficient preparation, 535.337463326 s for complex
coefficient generation, 203.042211914 s for window generation,
20.547578861 s for window sums and 223.868190523 s for native segment
calculation. The preparation and segment timers subdivide the C kernel;
they must not be added again to its enclosing timer. These are elapsed
durations that include any file-page or scheduling waits within the calls,
not a measured division between CPU work and I/O time. They show why the
long uncached coefficient preparation matters even though only 64 of the
744 output points use the native path.

Process-lifetime peak RSS was **9,940,348,928 bytes**, about **9.258 GiB**;
the same alias-counting qualification applies as in the simple workspace
call. The recorded hard cgroup limit remains 8 GiB. Whole-pipeline I/O
counters increased by **1,015,059,873,792 read bytes** and
**596,736,704,512 write bytes**, about 1.015 decimal TB and 596.74 decimal GB.
Those counters are not file sizes, durable-write totals or an attribution
of I/O costs to individual native stages.

The completed hybrid's individual validator reports finite, nonnegative
Float32 PSD and NSD at all 744 frequencies, with **zero ULP difference**
between every NSD output and the Float32-rounded square root of its stored
PSD. Its input hash is unchanged and matches the completed simple FFTW
input. Recomputed noise summaries agree with the stored values. The
Fourier-stage Parseval diagnostic gives relative disagreement
`1.5778169082785702e-16`; it checks the complete Fourier powers before
frequency smoothing and substitution of native low points. The comparison
with the completed corrected native result passes: all **64 native low
outputs are bit-identical in both PSD and NSD**, with matching indices,
lengths, counts and window normalization.

#### Completed 1B native LPSD/LNSD

The corrected native calculation completed all **744 frequencies** in
**2211.714508946 s**, about **36.86 minutes**, with **3411.382031178 s** of
summed process CPU time. This is a fresh joint PSD+NSD call, including
window/coefficient preparation, validation, the original segment estimator,
per-frequency reporting, output assembly and workspace cleanup. It does not
resume or reuse the earlier 732-point failed attempt.

The final native plan spans `5e-8` to `24.685157819345033 Hz`, with
`L=100..1,000,000,000`, `K=1..42,707,028` and
`sum(L*K)=3,122,351,444,856`. Every point completed, including j724, j733 and
j740, whose otherwise invalid final starts require the endpoint repair.
These are logical segment-sample counts, not storage-byte counts or
independent statistical averages.

The native inner call took 2211.635865925 s. Its recorded per-point elapsed
times sum to 4895.255250594 s because frequency work can overlap after
shorter points fit through the gate. The summed stage times include
3348.199241908 s in segment evaluation, 644.534005038 s in native projected
coefficient preparation, 607.375828709 s in complex coefficient generation
and 270.526953972 s in window generation. These elapsed aggregates are not
an additive decomposition of the complete wall time and do not separate
CPU execution from paging waits. Native preparation and segments are nested
inside the C-kernel timer. The 746 progress callbacks recorded
13.410736638 s in aggregate; their time is included and may overlap other
workers, so it is not subtracted from the result.

Process-lifetime peak RSS was **8,260,460,544 bytes**, or **7.693 GiB**.
The cgroup stayed at 8 GiB with no swap and zero recorded OOM/OOM-kill
events. Process-attributed I/O increased by **1,222,935,777,280 read bytes**
and **577,043,632,128 write bytes**, approximately 1.223 decimal TB and
577.04 decimal GB. Repeated paging makes these totals much larger than
the 24 GB of large input/workspace files; they are not unique-byte counts
or a device-throughput benchmark. The workspace mapping closes and its
scratch file is removed inside the complete-call scope.

`disk_endpoint_fixed_1b_validation.json` confirms the successful controller,
the completed native timing and the three-method output checks. All 30
protected source hashes, the measured Git HEAD and the native binary match
their pre-measurement identities. The 1B spectra therefore belong to the
declared fast.6 cohort throughout the complete calculation.

#### Completed 1B white-noise comparison

For the same `0.01..20 Hz` band and definitions used above, the completed
1B spectra contain **379 frequency points** in the band. The saved summaries
are:

| Method at 1B | Mean NSD / source NSD | SD of NSD / source NSD, % | Adjacent-difference RMS / sqrt(2), % |
| --- | ---: | ---: | ---: |
| Simple FFTW | 0.999711678 | 0.556430 | 0.595436 |
| Matched FFTW hybrid | 1.000188860 | 0.154570 | 0.071630 |
| Native LNSD | 1.000209694 | 0.155739 | 0.075449 |

The hybrid and native LNSD have similar measured waviness on this white-noise
record, while the simple FFTW result is rougher. The source NSD is
`1e-8 V/sqrt(Hz)`, so the respective standard deviations are approximately
`1.545700433e-11`, `1.557387510e-11` and `5.564299166e-11 V/sqrt(Hz)`.
The adjacent-difference statistic distinguishes local roughness from the
standard deviation across the whole band; both measures are reported.

This band contains the hybrid's smoothed outputs, entirely above its native
low-frequency subset. At each of its 379 frequencies, define the relative
NSD difference as `(method_NSD - native_NSD) / native_NSD`. No reference
point is zero. The recorded pointwise comparisons are:

| Method relative to native LNSD at 1B | RMS relative NSD difference, % | Maximum absolute relative NSD difference, % |
| --- | ---: | ---: |
| Simple FFTW | 0.548206025 | 2.998878334 |
| Matched FFTW hybrid | 0.034927512 | 0.172675857 |

The hybrid's absolute RMS difference is `3.494990616e-12 V/sqrt(Hz)`.
Native LNSD is the comparison reference, not known pointwise ground truth.
The separate exact low-point check establishes agreement where the hybrid
actually calls the native estimator. The high-band waviness and difference
statistics are descriptive values across correlated frequencies in one
realization, not confidence intervals, universal accuracy bounds or
peak-width/leakage measurements. The 100M version of this band contains
428 points; that changed grid also matters when comparing summaries
between record lengths.

#### Original native 1B failure and the endpoint correction

The original full-native 1B attempt at `5d5f5d8` failed with
`LPSD segment evaluation: unsafe segment bounds`. Its raw controller,
child JSON and log remain unchanged. It reported **732 successfully
computed frequencies**, but did not return a complete 744-point PSD/NSD
result or a completed-call time. Parallel point reports are not an ordered
prefix: the first missing source index was j724, while other workers later
reported successful points through j732.

This was not a different segment count supplied by the disk adapter. Both
the original planner and C kernel give `K=29,251,388` at j724 (`L=146`).
The inherited repeated Binary64 operation `start += shift` accumulated a
final start of `999,999,854.5296893`. Rounding that value selected
`999,999,855`, one sample beyond the legal endpoint `N-L=999,999,854`.
The optimized native guard rejected that out-of-bounds segment; the older
unprotected upstream routine would have attempted an invalid read.

Commit `68542b32c180ec26fe2821a6604ae436f1c9446b` adds a shared
[endpoint helper](../lpsd_fast/_native/segment_starts.h) for the native direct
and rolling paths. It changes **only a finite, otherwise invalid last
segment start** to the exact endpoint `N-L`. Every previously valid start,
including a final start below that endpoint, remains unchanged. Negative,
nonfinite and nonfinal out-of-bounds starts are still rejected. Bounds are
checked before converting to C `long`. K, shift, repeated addition, windows,
coefficients, local anchors, power normalization and the legacy statistics
recursion are retained.

The separate constant-memory index gate checked **all 744 original 1B plan
points and all 2,151,164,196 segment starts** using the actual helper. It
found exactly three formerly invalid starts, each the last segment of its
frequency and each rounded one sample beyond the valid endpoint:

| Source frequency index, zero-based | Segment length L | Segment count K | Terminal index repair |
| ---: | ---: | ---: | --- |
| 724 | 146 | 29,251,388 | 999,999,855 to 999,999,854 |
| 733 | 122 | 35,005,760 | 999,999,879 to 999,999,878 |
| 740 | 106 | 40,289,649 | 999,999,895 to 999,999,894 |

After correction, the gate found **zero invalid starts, zero nonfinal
repairs and zero changed previously valid starts**, with unchanged K at
every point. The grid and the 732 previously completed point plans agree
with the original records. This evidence is in
`native_endpoint_fix_plan_validation.json`. It evaluates index arithmetic
only: no signal array, DFT, projected coefficients or complete PSD/NSD call
was computed in that gate, so it supplies no native 1B spectrum or runtime.

The new native measurement is a fresh complete calculation from the
corrected version. It does not resume or splice the 732 earlier point
reports. Its validated 100M and 1B results are listed above. The completed
1B hybrid retains its original source and binary identity. Its 64 native
low points are outside the three
affected high-frequency positions and match the corrected full-native 1B
output bitwise in both PSD and NSD.

The final stored-output checks pass for both sizes: the three methods have
identical input hashes and frequency grids, finite nonnegative Float32
PSD/NSD and **zero ULP difference** from the Float32-rounded square-root
reference at every point. All 73 hybrid low points at 100M and all 64 at
1B match full LPSD in output bits, selected indices, lengths, counts and
normalization. White-noise waviness is summarized separately; it is not
used as a pointwise-equivalence test between different estimators.

## Numerical validation and its applicable revision

All **81 small tests** passed in the centralized serialized test run for the
I/O revision, recorded in `disk_validation_v2.xml` and
`disk_validation_v2.log`. There were no numerical failures. The gate covers the
[blocked FFT](../test_fast/test_fftw_disk.py),
[mapped LPSD](../test_fast/test_lpsd_disk.py), and
[complete pipelines](../test_fast/test_disk_pipelines.py). Coverage includes
odd/even lengths, factorization and bin order, real-to-complex expansion,
off-bin tones, DC plus nanovolt noise, shared-window aliasing, short native
fallbacks, workers 1/8, scratch release after failure, file-region overlap,
read-only workspaces, chunked input validation, Tukey construction and chunked
random-number generation. The added checks cover bounded file-side copy
views, unavailable or disabled memory advice, and scalar progress callbacks.
This extends the earlier 77-test gate and applies to the V2 adapter used in
`results_disk_blocked.json` and preserved at commit `ee4b4a7bde9c88c17d8d7cbe2ff1a05eaed438db`.

A new same-input 100M disk-versus-RAM audit of the **checked row-blocked
adapter** passed, recorded in `disk_blocked_100m_validation.json`. All three
methods used the same input and frequency grid as their RAM counterparts.
Simple FFTW and native LPSD outputs were bit-identical in both PSD and NSD.
The hybrid's maximum relative differences were `6.855234516544525e-8` in PSD
and `8.946151414865158e-8` in NSD, both below `9e-8`. These findings repeat the
successful earlier adapter audit on the revised implementation.

These observations validate the adapters and their normalization on the
tested cases. They are not a universal relative-error bound at spectral
nulls, a tone-quality measurement of the 100M white record, or a same-input
1B disk-versus-RAM audit. At 1B the completed simple FFTW call supplies the
separate energy diagnostic described above.

The new workspace-storage and per-frequency-reporting changes passed their
own central regression gate: **121 passed, zero failures, errors or skips**,
recorded in `disk_validation_workspace.xml`,
`disk_validation_workspace.log` and `disk_validation_workspace_summary.json`.
This includes the 92 prior cases after the reporting changes and 29
additional workspace/preflight/pipeline cases. Those tested source files
remained frozen throughout the original workspace 100M/1B series;
native, original-reference and FFTW library hashes were unchanged within
that fast.5 cohort. The later native correction has the separate identity
and gates described below.

The additional cases check bitwise external/workspace power agreement,
DC/Nyquist, asymmetric matrices and partial tiles, destructive extraction
state, refill, independent mapping lifetime, odd-B rejection, physical
file-budget accounting, complete estimators and hybrid cleanup. They
complement the static alias and resource-release review.

The new large-N control is independently checked in
`disk_workspace_100m_cross_validation.json`: all PSD and NSD values match V2
exactly, while the RAM comparison passes with the exact/under-`9e-8` results
listed above. The separate
`disk_workspace_100m_metadata_validation.json` also passes. It confirms the
shared input hash and all 704 grid values, finite nonnegative Float32
outputs, and exact agreement of every stored NSD value with the
Float32-rounded square-root reference (zero ULP difference for all 704
points of each method). All 73 native hybrid low points match full native
LPSD bitwise in PSD and NSD, with identical indices, L/K, window sums and
ENBW. Recomputed noise summaries agree with the stored summaries. These
checks use saved outputs outside the measured calls.

The completed 1B simple FFTW workspace output also passes the separate
`disk_workspace_1b_fftw_validation.json` check. Its unchanged input hash,
744 frequency values and all PSD/NSD values agree exactly with the V2 disk
record, with the same native-binary identity and Parseval result.

The completed hybrid independently passes
`disk_workspace_1b_matched_validation.json`. The record covers all 744 finite,
nonnegative Float32 PSD/NSD values, exact agreement with the Float32-rounded
square-root reference at every point, unchanged input hash and recomputed
noise summaries. Its scope is the completed hybrid job. Full native 1B
outputs and the final three-method audit are now independently checked by
`disk_endpoint_fixed_1b_validation.json` and `disk_final_validation.json`.
The latter covers all six selected jobs: 704 frequencies per method at
100M and 744 per method at 1B. Every output is finite and nonnegative;
all six NSD arrays agree exactly with the Float32-rounded square-root
reference. The unchanged common inputs and grids are confirmed, along
with 73/64 bit-identical hybrid low points at 100M/1B and their native
plan/window metadata. These checks do not require the smoothed high
frequencies of different estimators to be bit-identical.

GitHub Actions run `38095019106` passed **all eight CI jobs** for the original
`5d5f5d8` revision. Its Linux test job reports **1379 fast-suite tests and
18 upstream tests**. This broader CI result is distinct from the focused
121-case numerical gate and the completed large-N output checks.

The endpoint correction's complete central regression is recorded in
`native_endpoint_fix_validation.json`, `.log` and `.xml`: **1433 passed,
1 skipped and 2 expected failures**, out of 1436 cases, with zero unexpected
failures or errors. Its 36 new endpoint tests are included in that total.
The native binary and checked source hashes remained unchanged during the
gate, and the benchmark-native compiler flags match the previous build.
The separate 744-point index gate is described above; it is not another
full spectrum measurement.

`disk_endpoint_fixed_100m_validation.json` passes the large-N transition
check: all 704 corrected native PSD/NSD values exactly match the older RAM
and workspace values, while all 73 native low points of the unchanged
hybrid match the corrected full-native result. The input, grid and window
plan checks also pass. For this comparison, an explicit provenance file
must name the before/after native SHA-256 hashes and the measured corrected
commit. Only the native method may use that declared binary transition,
and exact PSD **and** NSD equality is required even where the ordinary RAM
comparison otherwise allows a numerical tolerance. FFTW and hybrid retain
their same-binary requirements.

The final provenance check in `validation_large_n.json` was performed after
both corrected native measurements completed and before any later
metadata-only Git update. All **30 protected source hashes**, the measured
commit `68542b32c180ec26fe2821a6604ae436f1c9446b` and native SHA-256
`6e2c621a8d611ad146abbda48bff4ccefee48f09acc58f124ea9655892f3f10f`
were unchanged. Both selected native jobs and their controller completed
with exit status zero. The final result's completed status requires six
selected complete jobs; it does not rewrite the terminal failure of the
earlier unselected native attempt.

The endpoint revision's initial remote CI status is preserved honestly in
`native_endpoint_fix_ci_summary.json`: **five numerical jobs passed and
three wheel jobs failed** at the version-metadata check, where
`pyproject.toml` still declared fast.5 while the Python module declared
fast.6. A separate remote metadata-only commit,
`d9eb84b682e33cb8b1e335ce133b9651b752b843`, changes only that version entry in
`pyproject.toml`. Its CI run **38101284887 passed all eight jobs**, including
the five numerical jobs and three native wheel smoke jobs. The Linux
GCC/Python 3.12 job reports **1415 fast-suite passes, 2 skips and 2 expected
failures**, plus **18 upstream passes**.

This follow-up is recorded separately in
`native_endpoint_metadata_provenance.json` and
`native_endpoint_metadata_ci_summary.json`. It did not change the local
measured checkout or native binary: the measured native cohort remained
`68542b32` with the `6e2c621a` library identity. The original CI record is
not rewritten as successful, and the metadata follow-up is not presented
as a new performance measurement. The completed native results and their
passed final output checks retain the measured cohort's original identity.

## Reproduction

Run from the repository root with the native LPSD libraries already built
for the stated source cohort and the same actual FFTW 3.3.11 build available.
Use separate checkouts/builds to reproduce the original fast.5 observations
and the corrected fast.6 native series. The test fixture also requires
the original reference C backend. See [build provenance](provenance.md) for
the native build identity and [the earlier FFTW comparison](fftw-comparison.md)
for library setup. Set the two paths to your installation; these shell
variables do not select NumPy's FFT implementation.

```bash
export LPSD_FFTW_LIB=/absolute/path/to/libfftw3.so.3
export LPSD_FFTW_THREADS_LIB=/absolute/path/to/libfftw3_threads.so.3

python benchmarks/bench_fftw_large_n.py \
  --sizes 10000000 100000000 1000000000 \
  --methods fftw matched matched_once lpsd \
  --outputs psd nsd --repeats 3 \
  --fftw-library "$LPSD_FFTW_LIB" \
  --threads-library "$LPSD_FFTW_THREADS_LIB" \
  --fftw-threads 8 --workers 8 --smoothing-workers 4 \
  --total-cache-mb 512 --sample-rate 50 --seed 20261010 \
  --output benchmark-output/results_large_n.json
```

These RAM commands describe the recorded fast.5 baseline; no fast.6 RAM
remeasurement is reported here. The RAM driver reads the actual memory limit and rejects known infeasible
jobs before allocation. Do not pass a fabricated larger memory limit to
force the 1B RAM cases past that check. The command reproduces the final
configuration; wall times on another host need not equal this table.

The separate file-backed experiment requires a regular file-backed working
directory with enough free capacity. A memory-backed temporary filesystem
is not a substitute when RAM capacity is the limiting resource. To reproduce
V2, use its immutable `ee4b4a7b...` sources and recorded external-power/2400 s
settings; the next two commands describe the original
`5d5f5d8564928cd27135fdad63bfd042c9be5853` workspace-storage revision. Their
working-directory path can be adapted to available regular file storage.

First run **only the 100M controls**:

```bash
python benchmarks/bench_fftw_disk.py \
  --sizes 100000000 --methods fftw matched lpsd \
  --fftw-library "$LPSD_FFTW_LIB" \
  --threads-library "$LPSD_FFTW_THREADS_LIB" \
  --fftw-threads 8 --workers 8 --smoothing-workers 4 \
  --fft-memory-mb 128 --low-working-mb 2048 \
  --sample-rate 50 --seed 20261010 \
  --power-storage workspace --lpsd-progress-interval-s 0 \
  --work-dir /tmp/lpsd_disk_workspace_control_20261010 \
  --timeout-seconds 10800 \
  --output benchmark-output/results_disk_workspace_100m.json
```

Compare these completed outputs with the recorded 100M RAM outputs on the
same input and grid. The selected **1B FFTW/hybrid observations** retain
that same source cohort:

```bash
python benchmarks/bench_fftw_disk.py \
  --sizes 1000000000 --methods fftw matched \
  --fftw-library "$LPSD_FFTW_LIB" \
  --threads-library "$LPSD_FFTW_THREADS_LIB" \
  --fftw-threads 8 --workers 8 --smoothing-workers 4 \
  --fft-memory-mb 128 --low-working-mb 2048 \
  --sample-rate 50 --seed 20261010 \
  --power-storage workspace --lpsd-progress-interval-s 0 \
  --work-dir /tmp/lpsd_disk_workspace_20261010 \
  --timeout-seconds 10800 \
  --output benchmark-output/results_disk_extended.json
```

The original full-native 1B attempt also ran at that revision and failed;
its historical record is retained, not used as a completed runtime. For
the **corrected native 100M/1B cohort**, use commit
`68542b32c180ec26fe2821a6604ae436f1c9446b` and its rebuilt native library:

```bash
python benchmarks/bench_fftw_disk.py \
  --sizes 100000000 1000000000 --methods lpsd \
  --fftw-library "$LPSD_FFTW_LIB" \
  --threads-library "$LPSD_FFTW_THREADS_LIB" \
  --fftw-threads 8 --workers 8 --smoothing-workers 4 \
  --fft-memory-mb 128 --low-working-mb 2048 \
  --sample-rate 50 --seed 20261010 \
  --power-storage workspace --lpsd-progress-interval-s 0 \
  --work-dir /tmp/lpsd_disk_endpoint_fixed_20261011 \
  --timeout-seconds 10800 \
  --output benchmark-output/results_disk_endpoint_fixed.json
```

The working-directory paths can be adapted to available regular file
storage. Preserve each cohort's source hashes, native library hash and
unmodified raw reports. The native transition comparison additionally
uses `native_endpoint_fix_provenance.json`; it requires the same input/grid
and exact PSD/NSD output equality for the checked 100M records.

The 10800 s timeout is a limit for each entire method process, not a runtime
prediction or a measured-call lower bound. It includes startup and input
hashing outside the calculation timer. The input generator and method jobs
are separate phases. The driver archives previous records and marks
interrupted checkpoints with the controller's terminal status; it does not
invent a call duration from that limit. Keep raw statuses and logs even if a
method cannot finish. Do not run tests, plots or another benchmark concurrently
with these timing jobs.

To rerun the small numerical gates separately from performance measurement:

```bash
LPSD_TEST_FFTW_LIBRARY="$LPSD_FFTW_LIB" \
python -m pytest -q test_fast/test_fftw_disk.py \
  test_fast/test_lpsd_disk.py test_fast/test_disk_pipelines.py \
  test_fast/test_fftw_large_n_reporting.py
```

The corrected native cohort additionally includes
`test_fast/test_segment_starts.py`. Run regression and index checks outside
all timed measurement processes; the full central regression count above
is broader than this focused example command.

The optional report renderer can read the validated final consolidated
records without rerunning any estimator. These are the selected records
in the evidence bundle; the individual historical controller files retain
their original statuses and are not replacements for this consolidation:

```bash
python benchmarks/render_fftw_large_n.py \
  --results benchmark-output/results_large_n_final.json \
  --disk-results benchmark-output/results_disk_final.json \
  --output-dir benchmark-output/figures
```
