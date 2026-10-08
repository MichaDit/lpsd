# Performance and reproducible measurements

The optimized implementation reduces the wall time of the existing direct-DFT
LPSD calculation. It retains the frequency-dependent segment lengths, overlap,
window and normalization. It does not replace LPSD with one FFT followed by
logarithmic binning. The original `lpsd` API remains available; opt in through
`from lpsd_fast import lpsd`.

## Recorded result

For ten million real samples, the unchanged `lpsd` 1.0.6 public API took
**151.412 seconds** in one measured call. `lpsd_fast`, using `kernel="auto"`
and eight workers, took **2.937 seconds**, the median of three measured calls.
The ratio is approximately **51.6**, on the machine and input described below.
This combines faster per-frequency computation with parallel frequencies.

| Samples | Actual frequencies | Implementation | Workers | Measured calls | Wall time, seconds |
|---:|---:|---|---:|---:|---:|
| 1,000,000 | 577 | `auto` | 1 | 3 | 1.740 median |
| 1,000,000 | 577 | `auto` | 8 | 3 | 0.423 median |
| 10,000,000 | 651 | Original C core | 1 | 1 | 151.412 |
| 10,000,000 | 651 | `auto` | 1 | 1 | 15.476 |
| 10,000,000 | 651 | `auto` | 8 | 3 | 2.937 median |
| 10,000,000 | 651 | `scalar` | 8 | 1 | 16.184 |
| 30,000,000 | 678 | `auto` | 8 | 1 | 12.506 |

The three ten-million-sample `auto`/eight-worker wall times were 3.715, 2.937
and 2.926 seconds. The median call used 23.134 seconds of process CPU time:
CPU time includes work on all threads and is not wall time. At this size the
single-worker optimization alone was approximately 9.8 times faster than the
original; eight workers then reduced the optimized wall time by another factor
of approximately 5.3. There was no original baseline at thirty million samples,
so that row has no measured speedup.

[The curated JSON](../benchmarks/results.json) contains the individual timings,
CPU times, process memory high-water marks, conditions and numerical summaries.
These are historical measurements of the optimized implementation before its
integration into this repository, not fresh measurements of every subsequent
commit. The original source baseline is the `lpsd` code at `2fd15da`, corresponding
to the unchanged 1.0.6 implementation used in the comparison.

### Conditions and limits

The measurements used a single Linux x86-64 machine reporting an AMD EPYC 9V74
80-Core Processor, with an eight-CPU cgroup quota and an 8 GiB memory limit,
Python 3.12.14, NumPy 2.3.5 and pandas 2.2.3. The saved build environment records
GCC 13.3.0. The optimized C build used `-O3`, `-fopenmp-simd`,
`-ffp-contract=off` and `-march=native`; it did not use global `-ffast-math`.
The eight-CPU quota governed available CPU time. CPU clock rates were not
controlled.

The input was float64 standard-normal noise generated with seed `20261008`.
Parameters were `sample_rate=1`, `n_frequencies=1000`, `n_averages=100`,
`detrending_order=0`, and the default Kaiser window/overlap with `psll=200`.
Each process made one small warm-up call using 4,096 samples and a target of
100 frequencies before the reported calls. This is not a full-size warm-up.
Input generation, imports, comparisons and result-file output were outside
the measured API call. Planning, input conversion, window/coefficient
generation, native computation and output construction were inside it.
The original baseline and the eight-worker headline timings were uninstrumented.
The historical single-worker `auto` calls also collected phase profiles, so
their API times include profiling probes. Each JSON measurement row records
whether profiling was enabled.

The historical original call received a pandas Series and the historical fast
calls received the same numerical samples as a NumPy array. Both public API
paths included their own conversion costs. The reproduction script below uses
a Series for **both** implementations to align this interface detail. Times
also depend on compiler, CPU, cache state, memory pressure and other processes.
Three repetitions expose some variation but are not a confidence interval.

At ten million samples, the eight-worker `auto` process reached approximately
1.96 GiB RSS, versus approximately 0.51 GiB for one worker. These are process
lifetime high-water marks, not isolated allocations of a particular call.
The concurrency memory budget controls active work reservations; it is not
a hard cap on total process RSS.

All seven output columns and the frequency index agreed bitwise with the
saved original result for the benchmark noise inputs that had a reference.
That observation does not extend to arbitrary signals: see
[numerical compatibility and known limitations](numerics.md).

## Reproduce an API wall-time comparison

Install this repository, including both native cores, using the instructions
in the root README. A CPU-specific optimized build can be selected explicitly:

```bash
python -m lpsd_fast.build --native
```

Run configurations sequentially, preferably without competing CPU workloads.
Each command uses its own process and writes a compact JSON report; input
samples and output spectra are not saved.

```bash
python benchmarks/bench_lpsd.py --backend original --n 10000000 --repeats 1 \
  --output benchmark-results/original-10m.json
python benchmarks/bench_lpsd.py --backend fast --n 10000000 --workers 1 \
  --kernel auto --repeats 3 --output benchmark-results/auto-1-10m.json
python benchmarks/bench_lpsd.py --backend fast --n 10000000 --workers 8 \
  --kernel auto --repeats 3 --output benchmark-results/auto-8-10m.json
python benchmarks/bench_lpsd.py --backend fast --n 10000000 --workers 8 \
  --kernel scalar --repeats 3 --output benchmark-results/scalar-8-10m.json
```

The script checks that the original C core is available rather than silently
timing its Python fallback. Both backends receive the same seeded Series and
spectral parameters. Build/load warm-up and input generation are reported
separately; file output is outside the timer. Output fingerprints allow
repeated runs to be associated with their result but do not replace the
numerical validation suite.

## Profile the remaining work

```bash
python benchmarks/bench_lpsd.py --backend fast --n 1000000 --workers 1 \
  --kernel auto --repeats 1 --profile \
  --output benchmark-results/profile-auto-1-1m.json
```

`--profile` performs an **additional** instrumented API call after the normal
timed calls. Its separate report includes planning and input conversion,
per-frequency window/normalization time, Fourier-coefficient time, native
preparation and native segment processing, as well as `L`, `K` and `K*L`.
Use one worker when identifying sequential bottlenecks. With multiple workers,
per-frequency elapsed times overlap and must not be added to estimate total
wall time. Native preparation/segment times are subdivisions of native-kernel
time, not additional work to add to that time.

### Recorded ten-million-sample phase breakdown

These tables describe **two separate one-worker profiling runs**. The first
uses a granular Python profiler around the unchanged original C binary;
its 151.3754-second total is separate from the 151.4123-second uninstrumented
original API call above. The second is the final historical `auto` build with
profiling enabled, whose externally measured API time was 15.4761 seconds.
The uninstrumented eight-worker result remains a separate measurement.
The selected raw phase sums and source report names are retained under
`recorded_phase_profiles` in [results.json](../benchmarks/results.json).

| Original implementation: phase | Seconds |
|---|---:|
| Native C kernel | 109.3126 |
| Kaiser window generation | 12.1393 |
| Python `sum(w)` | 11.8696 |
| Python `sum(w**2)` | 11.9339 |
| Complex Fourier coefficients | 4.7947 |
| Other preparation, copies, output and profiling overhead | 1.3252 |
| **Profiled API total** | **151.3754** |

| Final `auto`, one worker: phase | Seconds |
|---|---:|
| Window generation and sequential normalization | 2.0161 |
| Fourier coefficients | 3.4700 |
| Native coefficient-projection preparation | 1.1203 |
| Native segment projections and inherited statistics | 8.7104 |
| Other allocation, API and profiling overhead | 0.1593 |
| **Profiled API total** | **15.4761** |

The phase boundaries differ: the original coefficient row excludes phase-vector
construction and real/imaginary copies, which are included in its other work.
The optimized window row includes normalization. Native preparation and segment
processing are subdivisions of the optimized C kernel and are counted once.
These are one-worker sums; no elapsed times from overlapping workers are added.

The original profile motivated three complementary changes: remove redundant
PSD dot products and copies, move Python window-sum loops into C, and avoid
repeating mean detrending for every segment through a stabilized coefficient
projection. Native Kaiser/complex-coefficient generation and independent
frequency workers address the overhead exposed by the faster segment loop.

## Algorithmic work and scaling

Let `N` be the number of input samples and `M` the **actual** output frequency
count. At frequency `j`, let `L_j` be the segment length and `K_j` the number of
overlapping segments. For a fixed supported detrending degree, the direct
implementation performs work of order

\[
T = \Theta\!\left(N + \sum_{j=1}^{M} L_j
                       + \sum_{j=1}^{M} K_j L_j\right).
\]

For fixed valid overlap below one and the package's segment plan,
`K_j * L_j` is of order `N`, giving **Theta(N M)** work. SIMD, fewer passes
and threads reduce constants and wall time; they do not change that work bound.
With fixed logarithmic point density from roughly `fs/N` to `fs/2`,
`M = Theta(log N)` and the work is `Theta(N log N)`. A bounded actual `M`
instead gives `Theta(N)`. The requested `n_frequencies` is a planning target,
so substituting it for actual `M` without inspecting the plan can mislead.

For the measured defaults, the segment iteration counts were 2,377,782,853
at one million samples and 27,072,536,884 at ten million. These count visits
to segment samples, not CPU instructions or floating-point operations.
A full FFT and LPSD may share an asymptotic `N log N` description under the
fixed-density assumption while computing different estimators and doing very
different amounts of real work. This benchmark makes no FFTW speed comparison.
