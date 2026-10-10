# LPSD with a parallel native backend

This fork adds `lpsd_fast` to the Python/C [LPSD project](https://gitlab.com/uhh-gwd/lpsd).
It accelerates logarithmic power and cross-spectral density estimation with
native preparation, fewer memory passes, SIMD and parallel output frequencies.
The original `lpsd` implementation is included as the reference API.

The fork starts from **upstream v1.0.6**, commit
`2fd15da6930d19f5978f7e37b7b0785ce560f7d3`, with its Git history preserved.
Its package version is `1.0.6+fast.4`; this is not an official upstream release.

Version `fast.4` adds bounded FP64 FMA segment projections to single-worker
`kernel="fast"` calls on supported AVX-512 CPUs, plus shared coefficient
loads for projected CSD segments. The initial projections and inherited
statistics remain unchanged; scalar, automatic and parallel PSD paths
retain their established arithmetic. See the
[fused segment implementation](docs/fused-segments.md),
[CSD comparison](docs/csd-segments.md) and the
[complete fast.4 measurements](docs/performance-fast4.md).
The build and benchmark reports now bind native binaries to their project
source hashes, including recursively included C files and headers.

Version `fast.3` accelerates the segment calculation itself. Supported
64-bit AVX-512 CPUs share projected coefficients across eight auto-spectrum
segments. High-overlap Boxcar spectra can reuse outgoing and incoming
segment boundaries, with periodic direct rebuilds and numerical fallbacks.
The frequency plan, requested samples, window and inherited statistics are
retained. See the [new segment performance report](docs/performance-fast3.md)
for matched complete-call timings, profiles and reproducible experiments.
For the final ten-million-sample Kaiser comparison, all twelve retained
eight-worker pairs give medians of **1.684 s for fast.2 and 1.457 s for
fast.3**, about **13.5% less wall time**. The two component series differ
substantially and are shown separately in the report. At 95% Boxcar overlap,
reuse adds **2.01×** over the already adaptive CPU kernel on this host.

Selected PSD/NSD outputs, direct `lnsd`, native built-in and flat-top windows,
the SIMD Kaiser series and portable SIMD dispatch from `fast.2` remain
available. The [fast.2 performance report](docs/performance.md) and
[measurements](benchmarks/results_fast2_final.json) retain that release's
separate comparison with `fast.1`. Read the [numerical notes](docs/numerics.md)
and [platform validation](docs/platforms.md) for the tested conditions and
limits.

The additional [FFTW comparison and headroom report](docs/fftw-comparison.md)
measures the official FFTW 3.3.11 testbench and complete spectral pipelines on
the same host. It separates plan reuse, fresh setup and awkward transform
lengths, and documents why FFT-based logarithmic power averaging estimates a
different spectrum from segmented LPSD. Reproducible scripts and all timing
repetitions are included; FFTW remains an optional benchmark dependency.

The [GPU and accelerator assessment](docs/accelerators.md) records available
hardware, exact proposed offload payloads and a reproducible comparison
procedure. No unmeasured GPU backend is enabled.

The [local Windows acceleration measurements](docs/windows-accelerators.md)
test a Ryzen 9 7945HX, RTX 4070 Laptop CUDA FP64 and Radeon 610M OpenCL FP64.
The measured GPU prototypes are slower than the optimized CPU. CPU worker
counts plateau around 16–24; native compilation and reusable CPU plans show
measured benefits. Windows `auto` retains residual detrending after the
projected DC-plus-nanovolt path failed the existing numerical gate.

## Install from source

Python 3.10 or newer, NumPy, pandas and SciPy are required; pip installs
the Python dependencies. The native build needs a C11 compiler and `libm`;
it does not require FFTW, Python development headers or NumPy C headers.
The build supports GCC/Clang on Linux x86-64 and AArch64, Apple Clang on
Apple Silicon, and locally tested MinGW GCC on Windows x64.
See [platform validation and build options](docs/platforms.md)
for the actual CI evidence and remaining limits.

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
builds omit host-only ISA flags and use runtime AVX2/AVX512F dispatch on
supported x86 ELF systems. Native files are generated locally and are not new
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
| `kernel="fast"` | Enable additional coefficient/window acceleration with extra floating-point rounding |
| `outputs="psd"` | Return only PSD; skip variance updates and unused derived columns |
| `outputs="nsd"` | Return only noise spectral density, with the same square root and rounding as ASD |
| `profile=True` | Add per-frequency preparation and segment timings to `result.attrs["lpsd_profile"]` |

Input storage, caches, allocator overhead and an individual frequency larger
than the concurrency budget can take process RSS above `max_working_mb`.

### Request only the data you need

```python
from lpsd_fast import lpsd, lnsd

psd = lpsd(samples, sample_rate=1_000.0, kernel="fast", outputs="psd")
nsd = lnsd(samples, sample_rate=1_000.0, kernel="fast")
both = lpsd(samples, sample_rate=1_000.0, outputs=("psd", "nsd"))
```

`lnsd` returns a frequency-indexed DataFrame with one `nsd` column. If the
samples have units V and the sample rate is in Hz, PSD has units V²/Hz and
NSD has units V/√Hz. NSD is the existing `asd` calculation under an explicit
name: the square root is taken after the inherited complex64 PSD rounding.
It is available for auto spectra; `lcsd` retains the existing complex `asd`
behavior instead.

`outputs=None` or `outputs="all"` keeps the original seven-column result.
Any ordered selection of `ps`, `psd`, `ps_std`, `psd_std`, `enbw`, `asd`,
`asdrms` and `nsd` is accepted. Names must be unique. When both calls use
direct segment projections, each selected column matches the same column
in that kernel's full-output result. For Boxcar overlap reuse, requesting
deviation columns can trigger a stricter overflow fallback; at extreme input
scales, changing the output selection can therefore change final rounding.
Within a call, `nsd` and `asd` use the same square root of the same
already-rounded PSD.
Omitting both deviation columns skips native variance updates. Requesting
only `enbw` needs no DFT at all. The frequency plan, input samples and
number of segments used for an actual spectrum are unchanged.

For a Boxcar spectrum that already requires high overlap, `kernel="fast"`
automatically selects overlap reuse when its length and segment-count checks
permit it. For example:

```python
boxcar_psd = lpsd(
    samples, sample_rate=1_000.0, window_function="boxcar", overlap=0.9,
    detrending_order=0, kernel="fast", outputs="psd", workers=8,
)
```

This also works with NSD and the other auto-spectrum outputs. Select the
window and overlap for the required estimator; replacing Kaiser with Boxcar
changes the spectral result. See [overlap reuse](docs/rolling-boxcar.md) for
eligibility and the direct fallbacks. In profiles, `sample_iterations` is
logical `L*K` coverage, not the executed sample visits of the rolling method.

### Standard and custom windows

```python
hann_psd = lpsd(
    samples, sample_rate=1_000.0,
    window_function="hann", overlap=0.5, outputs="psd",
)
```

Known names are `kaiser`, `hann`/`hanning`, `hamming`, `blackman`, `bartlett`
and `boxcar`. The corresponding NumPy functions (`np.kaiser`, `np.hanning`,
`np.hamming`, `np.blackman`, `np.bartlett`, `np.ones`) are accepted too.
Fast kernels generate these windows in native code without serializing
Python callbacks. Kaiser keeps the original periodic definition
`np.kaiser(L + 1, beta)[:-1]`; Hann, Hamming, Blackman and Bartlett match
NumPy's symmetric length-L definitions. Boxcar contains ones. Supply an
explicit overlap for every non-Kaiser window.

All 19 periodic functions from `lpsd.flattop` also have a native path,
including `SFT3F`–`SFT5F`, `SFT3M`–`SFT5M`, `FTNI`, `FTHP`, `FTSR`,
`Matlab` and the `HFT` family through `HFT248D`. Pass the original function
or its name, for example `window_function="HFT248D", overlap=0.841`.
Their original cosine-series coefficients and periodic sample grid are
retained. Fast generation computes harmonics with a SIMD Chebyshev
recurrence from one cosine, which changes rounding but avoids separate
large temporary arrays for every harmonic. The scalar kernel continues
to call the original implementation.

Arbitrary window callables remain supported. The implementation calls the
function with the requested segment length and copies its result under a
lock, so a callback that reuses its own work buffer remains safe. Wrapping
a known NumPy function in a custom callable keeps that generic path. This
also preserves user-defined periodic/symmetric conventions.

### Kernel choice and numerical limits

| Kernel | Behavior |
| --- | --- |
| `fast` | Blockwise Fourier coefficients and SIMD Kaiser series with direct fallbacks; compensated order-0 preparation above 64 long-double mantissa bits; additional rounding differences |
| `auto` | Projected mean removal when C long double has more than 53 bits; original residual detrending plus SIMD on 53-bit targets and for other orders |
| `scalar` | Original window/coefficient route and serial Fourier accumulation; closest original-C compatibility |
| `simd` | Original long-double polynomial detrending with vectorized Fourier reductions |
| `projected` | Explicit projection for orders 0 and 1; order 1 is experimental for large ramps with tiny residuals |

The fast modes keep the LPSD plan and intended calculation, but they change
floating-point evaluation. Tests found relative differences near deep tone
nulls despite tiny absolute errors. `scalar` matched all 350 output-column
arrays in the recorded 49-case suite bitwise; this is not a universal
cross-platform equality guarantee. `fast` does not enforce an error tolerance
at runtime. The audit can apply an explicit relative limit and separate
PSD/NSD absolute limits, while keeping all relative violations visible.
No arbitrary-signal 1% guarantee is implied.

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

- [Current segment performance and bottlenecks](docs/performance-fast3.md)
- [High-overlap Boxcar reuse](docs/rolling-boxcar.md)
- [GPU and other accelerator assessment](docs/accelerators.md)
- [Earlier performance and direct-kernel complexity](docs/performance.md)
- [Final fast.2 timings and accuracy](benchmarks/results_fast2_final.json)
- [First fast.2 integration measurements](benchmarks/results_fast2.json)
- [Historical original-versus-fast measurements](benchmarks/results.json)
- [Native platforms and installed-wheel validation](docs/platforms.md)
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
