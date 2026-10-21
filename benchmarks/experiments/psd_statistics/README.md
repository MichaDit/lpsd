# Finite real-PSD statistics specialization — rejected

This 21 October 2026 experiment tested a specialization of the corrected
native auto-spectrum mean and variance updates. **It was rejected and is not
used by production code.** No reliable general complete-call improvement
was established. The native C implementation remains unchanged.

## Candidate and numerical scope

For a real auto spectrum, finite power implies an exactly zero imaginary
periodogram. The candidate keeps the original real power and Welford
operation order while omitting redundant complex products, divisions and
finite-difference guards. It uses the original generic helper whenever a
power or preceding mean is nonfinite. Legacy statistics and cross spectra
continue through their original helper. The patch also applies the same
specialization to rolling-boxcar and streaming PSD updates.

The [candidate patch](candidate.patch) reconstructs the tested source against
`fa7052505b8a72a1c85d95f599cd188742a18421`. The
[manifest](manifest.json) records both build manifests, compiler flags,
source and native binary hashes, measured-script hashes and completion state.
No native binaries are committed.

The native screen completed **640 comparisons**, spanning scalar/SIMD/
projected/compensated modes, selected/full outputs, corrected/legacy
statistics and lengths 16, 129, 256 and 2048. Signals include ordinary noise,
10 V DC plus nanovolt noise, an initial large transient, constants, zero,
subnormal-scale power, overflow-scale inputs, NaN and infinity. Array values
and segment counts matched with `assert_array_equal`, treating NaNs as NaNs;
this is not a separate proof of NaN payload or signed-zero bit identity.
These checks do not qualify every native ABI or another architecture.

## Complete API result

The host was an Intel Xeon Platinum 8573C, x86-64, with nine CPUs in its
allowed affinity mask and an eight-CPU quota. Both builds used the same
portable GCC flags and runtime-dispatched kernels. No other deliberate
benchmark or build ran during the measured screen; other shared-host load
and CPU frequency were not controlled.

Each configuration used the same one-million-sample float64 noise input,
seed 20261021, sample rate 1, Kaiser PSLL 200, 1000 requested frequencies,
100 requested averages, order-zero detrending, `kernel='auto'`, corrected
statistics and a 4096 MiB working budget. Each library received one warm-up,
then six paired calls alternating baseline/candidate and candidate/baseline.
Each timed call includes the complete API operation and its preparation;
imports, input creation, checks and JSON writes are outside the timer.
The common wrapper was the baseline implementation, isolating the C change.

| Workers | Outputs | Baseline median (s) | Candidate median (s) | Baseline / candidate | Faster candidate pairs |
|---:|---|---:|---:|---:|---:|
| 1 | PSD | 1.7080 | 1.8666 | 0.9150 | 4/6 |
| 1 | All seven columns | 2.1509 | 1.9362 | 1.1109 | 6/6 |
| 4 | PSD | 0.5892 | 0.6601 | 0.8925 | 2/6 |
| 4 | All seven columns | 0.6109 | 0.6145 | 0.9941 | 2/6 |

A ratio above one means a shorter candidate time. Single-worker full output
improved in these observations, but PSD-only medians regressed and parallel
full output showed no useful gain. The spread within the paired series is
visible in [all retained API calls](api-screen.json). It does not justify a
universal slowdown percentage, a new architecture dispatch rule, or a claim
about Windows or ARM performance. All saved final output values matched,
and their output-array SHA-256 values were identical in each configuration.

The separate [native screen](native-screen.json) retains six pairs for each
of 11 segment lengths and both variance settings at two million samples.
Some short-length configurations improved substantially; other lengths
regressed. These are isolated native-frequency timings, not whole-spectrum
measurements, and did not override the mixed complete-call result.

**The API screen was deliberately interrupted with exit code 130 after the
four complete one-million-sample groups.** Its planned ten-million-sample
stage has no complete retained group and is not a measured result. Neither
the interrupted stage nor unexecuted architectures are counted as passes.

## Reproduce in disposable local checkouts

Create two local worktrees at the pinned commit. Apply `candidate.patch`
only to the candidate worktree. Build both with the same Python environment
and ordinary portable settings:

```sh
python -m lpsd_fast.build
```

Copy their generated libraries to an otherwise fresh artifact directory as
`baseline.so` and `candidate.so`; this experiment's replay targets a Linux
host. The build sidecars document compiler/source identity and should also
be retained. Set `PYTHONPATH` to the pinned baseline checkout and use its
Python runtime dependencies when invoking these scripts:

```sh
python /path/to/native_screen.py /path/to/fresh-artifacts
python /path/to/api_screen.py /path/to/fresh-artifacts
```

Run the two screens sequentially, with other deliberate benchmark/build
work stopped and `OPENBLAS_NUM_THREADS=1` plus `OMP_NUM_THREADS=1`.
`native_screen.py` writes `native-screen-replay.json`; `api_screen.py` writes
`api-screen-replay.json`. These are new measurements, not replacements for
the archived observations. The replay scripts generalize the original
artifact directory, and the API replay includes only the four completed 1M
configurations. The manifest retains the hashes of the original measured
scripts and records this adaptation.
