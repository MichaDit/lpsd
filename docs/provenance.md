# Source provenance and redistribution

The optimization is an addition to the existing `lpsd` project. The original
module is retained, and its source at commit `2fd15da` is the integration
baseline. The performance and compatibility work used the unchanged PyPI
`lpsd` 1.0.6 Python/C implementation as its reference. The upstream project
metadata identifies [uhh-gwd/lpsd](https://gitlab.com/uhh-gwd/lpsd) as its
repository; the project's original authors and notices remain in the source.

The underlying logarithmic spectral-estimation method is described by
M. Troebs and G. Heinzel in
[Improved spectrum estimation from digitized time series on a logarithmic frequency axis](https://doi.org/10.1016/j.measurement.2005.10.010).
The new code accelerates the direct per-frequency implementation. It does
not claim a new asymptotic algorithm or substitute a logarithmically rebinned
FFT for the original estimator.

## Native sources and notices

| Component | Provenance | Notice retained |
|---|---|---|
| Original Python/C `lpsd` | Existing upstream project, version 1.0.6 reference | Project GPL version 3 license and original author information |
| `lpsd_fast/_native/fast_dft.c` | Optimized derivative of `ltpda_dft.c`; original DFT by M. Hewitson, CSD modifications by Artem Basalaev | `GPL-3.0-or-later` source header and attribution |
| `lpsd/c_sources/polyreg.c` | Original polynomial detrending by Gerhard Heinzel, AEI; directly included by the fast native core | One shared, unchanged upstream source |
| `lpsd_fast/_native/numpy_kaiser.c` | C adaptation of NumPy 2.3.5 `numpy/lib/_function_base_impl.py`; NumPy identifies the `i0` approximation as originating in Cephes | NumPy copyright and BSD-3-Clause text in the source |

The Kaiser port retains the NumPy approximation coefficients and Chebyshev
recurrence. Its changes concern C evaluation, symmetry and bounded temporary
storage. The NumPy notice is additional to the repository's GPL license; it
must remain with this component. The full NumPy license is included as
[`LICENSE.numpy.txt`](../lpsd_fast/_native/LICENSE.numpy.txt), and the project
license remains in [`LICENSE`](../LICENSE).

The fast package also reuses `_ltf_plan`, `_asdrms`, `_kaiser_alpha` and
`_kaiser_rov` from `lpsd._helpers`. Planning and polynomial detrending are
shared with the original module rather than maintained as duplicate source
copies. The combined distribution is named `lpsd`; `lpsd_fast` is the opt-in
module within that distribution.

## Reproducibility materials

[Benchmark results](../benchmarks/results.json) contain selected raw timing
records and validation summaries for synthetic inputs. Historical source and
binary fingerprints identify the implementation that was measured; they are
not a claim that a locally rebuilt binary has the same hash. The benchmark
tool imports the installed public APIs and generates its input from a recorded
seed. It does not download data or contact any remote service.

New build products such as shared libraries, object files, caches and generated
benchmark outputs should remain untracked. The existing upstream tree may
already contain binary files; their presence does not make newly generated
CPU-specific binaries suitable source artifacts. Distribute source, build
instructions and the required notices, and rebuild `-march=native` products
on the target CPU. No large sample arrays or saved spectral archives are
needed to reproduce the reported workflow.
