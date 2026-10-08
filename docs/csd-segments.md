# Sharing coefficient loads between CSD segments

The previous projected cross-spectrum path processed one segment at a time.
Each segment read both coefficient vectors again for its two input channels.
The new path shares those coefficient loads between two CSD segments, or four
on a supported AVX-512 target. It reuses the existing four/eight input-stream
projection helpers; no separate transform or changed estimator is introduced.

## Complete API measurements

These are comparisons against fast.3 commit
`9f3071ca2b7134a585f1727506a4e1472a7da299`, with the same portable GCC build
flags, on the exposed AMD EPYC 9V74 host. The container exposes nine logical
CPUs but has an eight-CPU cgroup quota. Each row contains six balanced pairs
of complete public API calls after one full warm-up per implementation.
All observations are retained in
[results_csd_batches.json](../benchmarks/results_csd_batches.json); the
[lossless raw report](../benchmarks/results_csd_batches_raw.json.gz) also
retains every per-frequency diagnostic profile.

| Samples | Workers | Actual frequencies | fast.3 median | CSD batch median | Ratio | Less wall time | Candidate faster |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 1 | 577 | 1.254964 s | 0.954886 s | 1.3143x | 23.91% | 6/6 pairs |
| 10,000,000 | 8 | 651 | 3.687593 s | 3.604374 s | 1.0231x | 2.26% | 5/6 pairs |

The eight-worker result is small compared with the observed variation:
baseline calls ranged from 3.342753 to 4.480481 s, candidate calls from
3.159973 to 4.095996 s. It does not establish a substantial multithreaded
speedup. The single-worker result is the clear measured benefit in this
series. Every ordinary and profiled output comparison retained bit-identical
complex64 PSD values, frequency grids and dtypes for the seeded inputs.

The separate one-worker diagnostic profiles show where work was saved:

| Samples | fast.3 segment time | CSD batch segment time | fast.3 full profiled call | CSD batch full profiled call |
| ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 0.970541 s | 0.781740 s | 1.232733 s | 1.083765 s |
| 10,000,000 | 13.918730 s | 10.108759 s | 15.867244 s | 12.261329 s |

These instrumented calls are independent of the ordinary timing samples.
They are not extra repetitions, and their stage times must not be added to
the complete-call medians. The results above isolate the CSD candidate before
subsequent PSD FMA integration; the final integrated build has its own report.

## Semantics and selection

Four CSD segments use sixteen real/imaginary accumulators: one pair per channel
per segment. This route requires x86-64 AVX-512, at least four remaining
segments, and lengths 128..255 or at least 1024. Two-segment batches use the
existing eight-accumulator helper at lengths of at least 2048. Other lengths,
remainders and nonprojected modes retain their previous calculation.

The first two CSD segments always retain their original single-segment
projection. This matters because the inherited mean update for the second
periodogram is `P0 + (P1 - P0)`: a tiny rounding change in a very large first
periodogram can otherwise become a substantial error after it is subtracted.
All later mean updates retain their original order. The legacy deviation
formula is preserved as well, including its previously documented limitations.

Every channel and segment has its own local input anchor. CSD phase and the
conjugation relationship on exchanging the input channels are preserved.
There is no frequency rounding, signal decimation, window substitution,
different segment spacing or change of detrending order.

The additional tests compare with the retained independent-segment native
export and exercise fractional starts, batch remainders, DC plus nanovolt
signals, phase-shifted tones, constants, channel exchange, large first
transients, nonfinite and overflow behavior, all returned columns and selected
PSD output. The isolated candidate passed 172 targeted cases in native
AVX-512 and in an AVX2 build without target clones. These checks establish
the tested cases; they are not a universal relative-error guarantee at a
spectral or cross-spectral zero.

## Repeat the comparison

Build both checkouts freshly before running, and do not rebuild or edit their
sources during measurement. For the archived portable comparisons, use
`make compile` in each checkout. Run from the candidate checkout:

```sh
python -m benchmarks.bench_segment_core \
  --baseline-root /path/to/fast3 --candidate-root /path/to/candidate \
  --kind csd --n 1000000 --workers 1 --repeats 6 \
  --output benchmark-results/csd-1000000-w1.json
```

Repeat with `--n 10000000 --workers 8` for the second row. Both use resident
float64 pandas input, PCG64 seed 20261008, channels `x` and
`0.7*x + 0.6*independent_noise`, sample rate 1, Kaiser/PSLL 200, requested
1000 frequencies and 100 averages, detrending order zero, `kernel="fast"`,
PSD-only output and a 4096 MiB working-memory target. Input creation and
verification remain outside the API timer. Different sampling or planning
parameters change the workload and must be reported separately.
