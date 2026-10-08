# Global FMA contraction experiment — rejected

This directory records an experiment against `fast.2` source commit `f97b8b8`.
It does **not** change the production compiler policy or any production kernel.
The candidate replaced `-ffp-contract=off` with `-ffp-contract=fast` in an
otherwise identical Linux/x86 native GCC 13.3 build. `-fno-fast-math` remained
enabled. Both libraries already used AVX-512 vectors on the measured host.

The global flag was rejected because real auto spectra acquired small nonzero
imaginary components and changed from `float32` to `complex64`. The stable
single-worker result showed only about a 1% difference; the additional profile
showed effectively identical segment time. No targeted production FMA change
was justified by this experiment.

## Recorded evidence

[`results.json`](results.json) contains all twelve individual timing results,
the additional profile summaries, compiler/binary provenance and both aborted
accuracy-audit attempts. [`semantics.json`](semantics.json) retains the complete
small complex-valued comparison, including dtypes and nonfinite counts.

The test used ten million seeded float64 samples, the default Kaiser/PSLL 200
configuration, `kernel="fast"`, PSD-only output, 651 resulting frequencies, a
4096 MiB working budget and one warm-up per process. Three separate process
pairs were run in baseline/FMA, FMA/baseline, baseline/FMA order. Each process
contained one ordinary timing call. The pair-2 W1 processes also made separate
profiled calls after their ordinary timing calls.

| Workers | Baseline median [range], s | FMA median [range], s | Ratio of medians |
|---|---:|---:|---:|
| 1 | 8.24343 [8.20290, 8.30251] | 8.15882 [7.88119, 8.19835] | 1.0104 |
| 8 | 2.69681 [2.17179, 2.99967] | 1.94674 [1.58915, 2.84022] | 1.3853 |

The W8 paired ratios were 1.116, 1.056 and 1.697. Three broadly varying pairs do
not establish a stable 38.5% gain or a confidence interval. The fresh additional
W1 profiles were 8.18162 s for baseline and 8.21277 s for FMA; their segment
phases were 6.29529 s and 6.29285 s respectively. Profile times are not
subdivisions of the ordinary timing medians.

The 12-case semantic probe used 4097 samples, amplitudes 1, 1e6 and 1e9, white
noise, an off-bin tone, DC plus nanovolt noise, and phase-shifted CSD. It found:

- 27 dtype-mismatch columns, covering all three selected columns in all nine
  PSD cases; all three CSD cases retained their complex dtype.
- 255 newly nonzero imaginary PSD points and no nonfinite values.
- For amplitude-1 white noise, imaginary PSD reached approximately 8.30e-19;
  at amplitude 1e9 it reached 0.3442, still only approximately 3.30e-17 of the
  reference at the worst relative point. Small relative error does not remove
  the return-type/API change.

The attempted 19-case audits at N=32769 and N=131073 both aborted with
`ValueError: This audit is for real auto spectra, not complex CSD.` Their exit
code 1 is **not** a completed numerical rejection at 1%, and it is not a pass.
The real-spectrum audit correctly refused to discard the unexpected imaginary
components. The semantic probe computes complex errors without a float cast.

The arithmetic explanation is that the statistics expression
`r * (-i) + i * r` cancels exactly when both products receive the same rounding,
provided those products are finite. Contracting one multiplication with the
addition can leave the residual of the independently rounded other product.
The production API only converts an auto spectrum to real dtype when its
imaginary PSD entries are exactly zero. This is a concrete compatibility issue
even when the residual is numerically small.

## Reproduction

Use an installed isolated checkout with the ordinary original and fast native
libraries already built. Its fast build report supplies the compiler and flags:

```bash
python -m lpsd_fast.build --native
python benchmarks/experiments/fp_contract/prepare.py
```

`prepare.py` captures the current native C/H sources into a private directory,
preserving the relative include layout. It builds **both** the strict reference
and FMA candidate from that one snapshot, then keeps the snapshot and both
binaries in the ignored `artifacts/` directory. It verifies the existing build
sidecar, compiler version/target, and retained strict floating-point flags.
Both private build reports record identical source hashes, a snapshot-manifest
hash and their exact compiler commands. The existing production binary supplies
compiler/flags provenance only; it is not used as the new strict reference.
Thus editing source files after the installed build cannot silently compare an
old reference against a newly compiled candidate. The loader verifies both
binary hashes and the retained source snapshot. No production binary is changed.
`--replace` is only for rebuilding private artifacts when no process has them
loaded. Do not rebuild during measurements.

Run each timing in a separate process and serialize it with all other CPU work:

```bash
python benchmarks/experiments/fp_contract/replay.py baseline bench -- --backend fast --kernel fast --outputs psd --n 10000000 --workers 1 --repeats 1 --warmups 1 --max-working-mb 4096 --output benchmark-results/fma/baseline-w1.json
python benchmarks/experiments/fp_contract/replay.py fma bench -- --backend fast --kernel fast --outputs psd --n 10000000 --workers 1 --repeats 1 --warmups 1 --max-working-mb 4096 --output benchmark-results/fma/fma-w1.json
python benchmarks/experiments/fp_contract/probe_semantics.py --output benchmark-results/fma/semantics.json
```

Repeat the timing commands with `--workers 8`, and alternate pair order as
described above. The wrapper records the library actually loaded. Both variants
use the same public `kernel="fast"` API. `probe_semantics.py` measures no times.

To reproduce the audit's expected refusal of complex auto spectra:

```bash
python benchmarks/experiments/fp_contract/replay.py fma audit -- --n 32769 --kernel fast --workers 1 --relative-limit .01 --psd-absolute-limit 1e-24 --nsd-absolute-limit 1e-12 --fail-on-limit --output benchmark-results/fma/audit-32769.json
```

Scalar reference calls always use the separate baseline library. The wrapper
switches handles only between complete synchronous API calls; it is not an
interface for concurrently launching benchmark and audit calls from one process.
The global experiment also contracts generator/statistics arithmetic and does
not isolate dot-loop FMA. Its outcome says nothing universal about a future
carefully scoped FMA implementation, another CPU, or arbitrary input signals.

The archived scripts were moved from a private experiment directory and their
loading paths were generalized during curation. Their original execution
hashes and measured library hashes remain in the evidence. The reproduction
recipe was subsequently strengthened to build a fresh strict/FMA pair from a
shared source snapshot; this does not alter the historical results. No native
binaries are committed. New reproduction calls must be kept separate from the
archived measurements.

The corrected recipe was validated with two private builds and one 12-case
semantic smoke. It reproduced the original strict binary SHA256
`b0ab2e54a6231bd54b8d01c9c364efadb924c8e41dcc0398ef637f29d3e9e623`
and FMA binary SHA256
`4ec423046695c19d8b78ac6516982b97f2dd5e156b8cc78390dc2b0a726a2d0b`
from the same eight-file snapshot. The smoke again found 27 dtype-mismatch
columns, 255 new imaginary PSD points and zero nonfinite values. Installed
production hashes were checked before and after and remained unchanged. No
performance measurement or full accuracy audit was repeated for this fix.
