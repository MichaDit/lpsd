# Fast implementation regression gates

Run from the repository root, with GCC available:

```sh
python -m pip install '.[test]'
make compile
python -m pytest -o addopts='' -q -ra test_fast
python -m pytest -o addopts='' -q -ra test
```

The small fast suite uses at most 2,049 samples per signal. It is a numerical
regression suite; performance measurements belong in `benchmarks/` and are not
timing assertions in shared CI. GitHub Actions runs the suite on Linux with
Python 3.10 and 3.12 and portable GCC code generation. Dependency versions are
resolved from package requirements, without pinning a particular NumPy build.

## Reference and numerical contract

The reference is the unchanged `lpsd` package shipped in this repository, with
its original **C** backend explicitly required. Missing native libraries fail
the suite instead of selecting the Python fallback. No separate bundle path,
network download, mutable release tag, or worktree is needed.
`reference_sources.json` records SHA-256 hashes of nine original Python/C/header
files from upstream commit `2fd15da6930d19f5978f7e37b7b0785ce560f7d3`.
`test_reference_sources_match_pinned_upstream` checks those bytes, including the
planning/normalization helpers and `polyreg.c`. It deliberately does not compare
compiled-library hashes: compiler and platform changes produce different binaries.
Changing the reference requires an explicit provenance and numerical review;
updating hashes merely to make a failure disappear is not a valid fix.

The scalar kernel must match all seven columns, frequency indices, dtypes, and
every output bit for 26 deterministic cases. These cover noise, pink spectra,
DC plus nanovolt noise, a 20 V ramp plus nanovolt noise, detrending orders
None/0/1/2, short/odd inputs, custom windows, overlap and planning settings,
integer/float32 data, pandas containers, inferred sampling and complex CSD.
This is a same-build Linux regression gate, not a cross-platform bitwise promise.

The auto kernel must match the five primary outputs (`ps`, `psd`, `enbw`, `asd`,
`asdrms`) at relative tolerance `2e-6`, with zero absolute tolerance, for the
21 ordinary cases. This allowance accounts for reordered double-precision work
before float32/complex64 output conversion. It is not a uniform relative-error
claim at a spectral zero. Zero and constant inputs have separate exact-zero
residual checks. All three pure-tone cases retain mandatory finite/nonnegative,
frequency/dtype and absolute ASD roundoff gates. The latter use window norms,
input magnitude and the usual floating-point dot-product error factor; they
are not a confidence interval or a statistical accuracy claim.

Exactly two narrow `xfail(strict=True, raises=AssertionError)` tests expose known
limitations. They are visible with `-ra`, and an unexpected pass fails CI so that
changed behavior is reviewed:

1. An off-bin pure tone does not satisfy a uniform relative PSD tolerance of
   `2e-6` in its vanishing sidelobes. Its absolute ASD error is independently
   checked by a mandatory test; the whole tone suite is never marked xfail.
2. The original C recurrence gives zero standard error for a constructed pair
   of periodograms `[2, 8]`, whose usual sample standard error is 3. A separate
   passing test requires fast and original outputs to preserve this exact
   legacy behavior. Statistical corrections are a separate estimator change.

The general auto comparison excludes `ps_std` and `psd_std`: the inherited
recurrence has cancellation-sensitive artifacts, especially at two segments.
Those columns remain gated bitwise in scalar and parallel determinism tests;
the analytical fixture documents the statistical limitation explicitly.
The large-ramp case also requires auto to keep the stable linear detrending
path. Reversed, strided, unaligned and read-only arrays are passed directly to
the candidate without a layout-erasing copy and checked for input mutation.
Parallel results must match one worker bitwise for both scalar and auto.

## Packaging gate

The separate CI job builds an sdist and then a platform wheel, installs the
wheel in a clean virtual environment, and runs:

```sh
/path/to/clean-venv/bin/python -I /path/to/checkout/test_fast/wheel_smoke.py
```

Run it from outside the checkout. The script rejects source-tree imports,
requires the original native backend, compares scalar outputs, exercises auto
with a NumPy input and two workers, and checks distribution/module versions.
This catches missing native libraries and accidental success from local build
artifacts. CI does not publish distributions or use write credentials.

## Selected outputs and window acceleration

The additional tests in `test_outputs.py` compare requested columns with the
full result from the **same kernel**, bit for bit. They cover individual names,
requested order, the `nsd` alias, cumulative RMS and deviation dependencies,
parallel workers, multichannel data, and difficult DC/ramp/tone inputs. The
unchanged original compatibility tests remain the independent estimator gate.
NSD must follow the existing rounded complex64 PSD-to-ASD path, including its
float32 real output, rather than taking a double-precision square root and
rounding afterward. Complex and exactly anticorrelated CSD retain the legacy
`asd` behavior; `nsd` and `lnsd` reject CSD. Explicit `outputs` on `lnsd`, even
`None`, is an error instead of being silently discarded.

Tests also require PSD/NSD-only calls to skip cumulative RMS and ENBW-only calls
to avoid Fourier coefficients and the DFT kernel entirely. These are structural
work-saving checks, not fragile timing thresholds. Scalar/auto standard-window
tests distinguish NumPy's symmetric Hann, Hamming, Blackman and Bartlett
conventions from the original periodic Kaiser. Opaque callbacks remain opaque,
including a callback whose name matches a standard window and one that reuses
a noncontiguous buffer while frequencies run concurrently.

The benchmark CLI accepts output and window controls without changing the
default full-output configuration:

```sh
python benchmarks/bench_lpsd.py --backend fast --n 1000000 --workers 1 \
  --outputs psd --output benchmark-results/psd-only.json
python benchmarks/bench_lpsd.py --backend fast --n 1000000 --workers 1 \
  --entry-point lnsd --output benchmark-results/nsd-direct.json
python benchmarks/bench_lpsd.py --backend fast --n 1000000 --workers 1 \
  --outputs psd nsd --window hann --overlap 0.5 --profile \
  --output benchmark-results/hann-profile.json
```

`--outputs` preserves the requested order; `--output` still names the JSON
report file. `--psll` selects the Kaiser parameter. Every non-Kaiser window
requires explicit overlap. The `hft248d` option provides an existing, more
expensive arbitrary-callback comparison in addition to the recognized NumPy
windows. The original backend always computes its full outputs and rejects
selected-output requests. Optional profiling is an additional API call and
now summarizes window generation, window sums and output assembly separately
when those fields are available.

For wall-time comparisons, run each configuration in a separate process and
run the processes sequentially. A focused matrix is full/PSD-only/NSD-only at
one and eight workers, followed by Kaiser/Hann/Hamming/Blackman/Bartlett/boxcar
with the same requested output. Hold sample count, seed, frequency planning,
overlap, detrending and compiler flags fixed within each comparison; changing
overlap changes the estimator's work. Use a small input with a dense requested
grid to expose Python/output overhead, then large inputs for the segment and
memory costs. Do not present the sum of concurrent worker timers as wall time.
