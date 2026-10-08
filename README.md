# LPSD with a parallel native backend

This fork adds `lpsd_fast` to the Python/C [LPSD project](https://gitlab.com/uhh-gwd/lpsd).
It accelerates logarithmic power and cross-spectral density estimation with
native preparation, fewer memory passes, SIMD and parallel output frequencies.
The original `lpsd` implementation is included as the reference API.

The fork starts from **upstream v1.0.6**, commit
`2fd15da6930d19f5978f7e37b7b0785ce560f7d3`, with its Git history preserved.
Its package version is `1.0.6+fast.1`; this is not an official upstream release.

For the recorded ten-million-sample noise input, the original public API took
151.412 s and the optimized eight-worker API took a median of 2.937 s across
three calls: approximately **51.6 times faster** on that machine. The
single-worker optimized call, with profiling enabled, took 15.476 s. These are historical measurements
on an eight-CPU-quota Linux environment, not a speed guarantee for other
signals or hardware. See [measurements and reproduction](docs/performance.md)
and [numerical behavior](docs/numerics.md).

## Install from source

The tested target is **Linux x86-64 with GCC**. Python 3.10 or newer, NumPy,
pandas and SciPy are required; pip installs the Python dependencies. No FFTW,
Python development headers or NumPy C headers are required by the native build.
Other operating systems and toolchains need their own build validation.

```bash
git clone https://github.com/MichaDit/lpsd.git
cd lpsd
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

The editable install builds both shared libraries in the checkout. To optimize
the fast core for the CPU on which it will run:

```bash
python -m lpsd_fast.build --native
```

Rebuild that CPU-specific library before moving it to a different CPU. Ordinary
builds omit `-march=native`. Native files are generated locally and are not new
Git-tracked artifacts. Compiler failures stop installation instead of leaving
a silent Python fallback.

A regular non-editable installation is also supported with `python -m pip
install .`. Run that installed package from outside the checkout, or build the
source-tree libraries with `make compile` before importing from the checkout.

## Compute a spectrum

```python
import numpy as np
from lpsd_fast import lpsd

samples = np.random.default_rng(0).normal(size=1_000_000)
spectrum = lpsd(
    samples,
    sample_rate=1_000.0,
    n_frequencies=1_000,
    n_averages=100,
    workers=8,
    kernel="auto",
    max_working_mb=2_048,
)

print(spectrum[["psd", "asd"]])
```

The result is a pandas DataFrame indexed by frequency. It contains the
original `ps`, `psd`, `ps_std`, `psd_std`, `enbw`, `asd` and `asdrms` columns
and their original dtypes. For large files, a one-dimensional float64
`np.load("signal.npy", mmap_mode="r")` input can avoid making an initial
full copy. Noncontiguous or differently typed inputs may still require
conversion. Input arrays are not modified.

Pandas Series and DataFrames are supported. A two-column DataFrame can be
passed to `from lpsd_fast import lcsd` for cross-spectral density. Supply
`sample_rate` explicitly for NumPy inputs; the pandas path can infer it from
the time index. Spectral options retain the original meanings, including
window, overlap, detrending degree, frequency target and averaging target.
The requested frequency count is a planning target; inspect the result length
for the actual number of frequency points.

### Performance controls

| Option | Meaning |
| --- | --- |
| `workers=None` | Select workers using CPU affinity and, on Linux, the cgroup v2 CPU quota |
| `workers=1` | Sequential frequencies, useful for isolating computational bottlenecks |
| `max_working_mb` | Budget for concurrent temporary work; not a hard limit on total process RSS |
| `window_cache_mb=128` | Bounded cache for repeated window lengths |
| `profile=True` | Add per-frequency preparation and segment timings to `result.attrs["lpsd_profile"]` |

Input storage, caches, allocator overhead and an individual frequency larger
than the concurrency budget can take process RSS above `max_working_mb`.

### Kernel choice and numerical limits

| Kernel | Behavior |
| --- | --- |
| `auto` | Projected mean removal for detrending order 0; original residual detrending plus SIMD for other orders |
| `scalar` | Original window/coefficient route and serial Fourier accumulation; closest original-C compatibility |
| `simd` | Original long-double polynomial detrending with vectorized Fourier reductions |
| `projected` | Explicit projection for orders 0 and 1; order 1 is experimental for large ramps with tiny residuals |

The fast modes keep the LPSD plan and intended calculation, but they change
floating-point evaluation. Tests found relative differences near deep tone
nulls despite tiny absolute errors. `scalar` matched all 350 output-column
arrays in the recorded 49-case suite bitwise; this is not a universal
cross-platform equality guarantee.

**Inherited statistics:** all modes deliberately preserve two defects in the
original mean/second-moment recurrence. In particular, `ps_std` and `psd_std`
are not validated statistical uncertainty estimates. Their correction would
be a separate estimator change. Read [the numerical notes](docs/numerics.md)
before interpreting these columns or extremely small sidelobes.

## Original API

Existing code can continue to use `from lpsd import lpsd, lcsd, lpsd_trad`.
The `lpsd/` source files and the original 18-test suite are retained. Frequency
planning, normalization and polynomial routines are shared by both paths.
The original module also provides its Python backend; `lpsd_fast` requires
its C backend.

## Tests and benchmarks

```bash
python -m pip install -e '.[test,dev]'
python -m pytest -o addopts='' test test_fast
python benchmarks/bench_lpsd.py --backend fast --n 1000000 --workers 1 \
  --repeats 1 --profile --output benchmark-results/profile.json
```

The tests compare against pinned original sources, check numerical behavior,
input handling and worker determinism. Known numerical limitations are
reported separately from ordinary passing regression checks. The CI workflow
also builds an sdist/wheel and imports the installed wheel from outside the
checkout. See [test details](test_fast/README.md).

The benchmark script times public API calls with identical seeded Series
inputs for both backends. Input generation, warm-up and file output are
separate. Profiling uses an additional instrumented call; per-worker elapsed
times overlap and must not be summed as wall time.

- [Performance, bottlenecks and complexity](docs/performance.md)
- [Recorded timings and validation summaries](benchmarks/results.json)
- [Numerical analysis and compatibility](docs/numerics.md)
- [Source provenance and licenses](docs/provenance.md)

## References and license

- Michael Troebs and Gerhard Heinzel,
  [Improved spectrum estimation from digitized time series on a logarithmic frequency axis](https://doi.org/10.1016/j.measurement.2005.10.010).
- Gerhard Heinzel, Albrecht Ruediger and Roland Schilling,
  [Spectrum and spectral density estimation by the Discrete Fourier transform](http://hdl.handle.net/11858/00-001M-0000-0013-557A-5).
- [MATLAB Toolbox LTPDA](https://www.elisascience.org/ltpda/).

The original authors and attribution remain in the sources. This derivative
is distributed under the repository's [GPL license](LICENSE). The native
NumPy/Cephes Kaiser port also retains its [BSD notice](lpsd_fast/_native/LICENSE.numpy.txt).
