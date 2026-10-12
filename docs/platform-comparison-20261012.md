# Platform comparison for revision 2e21527

Candidate: `2e215275124d0ffa05e314d7eddf18ecab5a1ae1`. Baseline: `8bfbf2b0648e6b328df04be434c376dbc8de71a4`.

Primary pull-request cohort only. Three alternating pairs per configuration; speed factor is the ratio of wall-time medians. Ranges are observed minimum–maximum, not confidence intervals.

The tested PR merge `89274daee27bee1c21f122201692b39d861df7a0` has exactly the candidate tree `b68bdca9d60ad5ac4bb9df8248ebe7866524d3a3` (verified through the GitHub commit API).

## Findings

- All eight native and all four portable jobs completed successfully at the final head, including the real Windows bootstrap watchdog and memory query.
- At N1M the final Linux native medians differ by less than 1% in every configuration. No material Linux regression is evident in these small screens.
- The final macOS native auto comparison is noisy: 0.994x with one worker and 1.027x with three. Earlier separate 2afaac cohorts observed 1.164x and 1.088x with one worker, but the final cohort does not establish a stable percentage gain. The unchanged fast control also has wide scatter.
- At N1M the packed FFT is consistently modestly faster for simple FFT on Linux ARM64 (1.063x and 1.031x). Other platform/method cases are mixed or close to measurement scatter. Retain explicit algorithm selection; do not advertise a universal small-N gain.
- Windows N1M full/packed medians differ by about -2% to +3%; all ranges overlap. The clear Windows result is repaired execution and verified numerical agreement.

Across the final cohort, all 24 native cases (72 pairs) and 32 disk cases (96 pairs) completed. Every returned output comparison is bit-identical. Every disk frequency-grid, all-point 1% limit and stricter 3ppm check passed.

Earlier macOS observations are from [PR native run 38217313104](https://github.com/MichaDit/lpsd/actions/runs/38217313104) and [separate push run 38217309458](https://github.com/MichaDit/lpsd/actions/runs/38217309458); they are not pooled with the final cohort.

## Native complete API calls at N = 1,000,000

`auto` returns all outputs; `fast` returns PSD and NSD. Times in seconds.

| Platform | Kernel | Workers | Baseline median [range] | Current median [range] | Factor | Returned outputs |
|---|---|---:|---:|---:|---:|---|
| linux-arm64-gcc | auto | 1 | 3.475439 [3.473437–3.478124] | 3.464214 [3.460155–3.467381] | 1.003× | identical |
| linux-arm64-gcc | auto | 4 | 0.928140 [0.927172–0.932166] | 0.920848 [0.919813–0.926431] | 1.008× | identical |
| linux-arm64-gcc | fast | 1 | 1.709189 [1.706391–1.712249] | 1.694852 [1.690299–1.694868] | 1.008× | identical |
| linux-arm64-gcc | fast | 4 | 0.474236 [0.472766–0.476281] | 0.470333 [0.469031–0.479370] | 1.008× | identical |
| linux-x64-gcc | auto | 1 | 1.022500 [1.019468–1.029612] | 1.019495 [1.017453–1.020039] | 1.003× | identical |
| linux-x64-gcc | auto | 4 | 0.464618 [0.461815–0.465773] | 0.465856 [0.464563–0.468916] | 0.997× | identical |
| linux-x64-gcc | fast | 1 | 0.507110 [0.502619–0.514281] | 0.510130 [0.504282–0.511021] | 0.994× | identical |
| linux-x64-gcc | fast | 4 | 0.276384 [0.275056–0.279260] | 0.277232 [0.276007–0.278828] | 0.997× | identical |
| macos-arm64-clang | auto | 1 | 4.458334 [4.256393–4.957041] | 4.483215 [3.879336–5.081961] | 0.994× | identical |
| macos-arm64-clang | auto | 3 | 1.963399 [1.888276–2.079500] | 1.911285 [1.751346–2.064510] | 1.027× | identical |
| macos-arm64-clang | fast | 1 | 1.380734 [1.048451–1.382396] | 1.150885 [1.044940–1.174284] | 1.200× | identical |
| macos-arm64-clang | fast | 3 | 0.439827 [0.309603–0.547039] | 0.474257 [0.301452–0.479742] | 0.927× | identical |

## Full versus packed real FFT at N = 1,000,000

Two FFTW threads and two native workers. Both PSD and NSD returned. Times in milliseconds; 577 and 1,154 are the actual output point counts.

| Platform | Method | Points | Full median [range] | Packed median [range] | Factor | Returned outputs |
|---|---|---:|---:|---:|---:|---|
| linux-arm64 | fftw | 577 | 67.253 [65.834–68.304] | 63.244 [62.083–63.639] | 1.063× | identical |
| linux-arm64 | fftw | 1154 | 69.506 [69.217–71.066] | 67.386 [65.858–67.425] | 1.031× | identical |
| linux-arm64 | matched | 577 | 396.414 [395.554–397.788] | 395.352 [395.262–399.028] | 1.003× | identical |
| linux-arm64 | matched | 1154 | 672.161 [671.134–673.359] | 670.583 [670.319–671.226] | 1.002× | identical |
| linux-x64 | fftw | 577 | 60.762 [59.695–66.330] | 61.694 [60.995–63.799] | 0.985× | identical |
| linux-x64 | fftw | 1154 | 63.432 [63.001–64.486] | 64.882 [63.322–65.424] | 0.978× | identical |
| linux-x64 | matched | 577 | 335.344 [331.136–338.030] | 336.713 [334.326–337.730] | 0.996× | identical |
| linux-x64 | matched | 1154 | 564.245 [563.215–565.790] | 561.727 [547.300–586.192] | 1.004× | identical |
| macos-arm64 | fftw | 577 | 47.636 [47.498–48.701] | 50.309 [48.334–60.833] | 0.947× | identical |
| macos-arm64 | fftw | 1154 | 55.872 [52.373–57.108] | 49.860 [49.738–54.994] | 1.121× | identical |
| macos-arm64 | matched | 577 | 340.006 [337.305–360.450] | 336.909 [315.417–338.140] | 1.009× | identical |
| macos-arm64 | matched | 1154 | 592.383 [532.682–613.272] | 534.319 [503.118–567.788] | 1.109× | identical |
| windows-x64 | fftw | 577 | 104.910 [104.111–112.993] | 105.539 [104.831–115.259] | 0.994× | identical |
| windows-x64 | fftw | 1154 | 113.908 [110.005–119.819] | 111.512 [109.670–111.519] | 1.021× | identical |
| windows-x64 | matched | 577 | 422.639 [412.914–448.803] | 409.490 [400.286–469.247] | 1.032× | identical |
| windows-x64 | matched | 1154 | 646.028 [615.491–648.566] | 659.774 [628.866–673.627] | 0.979× | identical |

All-point PSD/NSD numerical checks and exact frequency-grid checks are retained in [the evidence JSON](../benchmarks/results/platform-comparison-20261012.json), together with N = 100,000 results and every raw repeat time. Float32 output identity does not replace the separate stricter raw-transform tests.

## Sources

- [Native workflow 38218181121](https://github.com/MichaDit/lpsd/actions/runs/38218181121)
- [Portable workflow 38218181042](https://github.com/MichaDit/lpsd/actions/runs/38218181042)

| Artifact | ID | Relative path inside artifact |
|---|---:|---|
| [results-linux-arm64-gcc-python-3.12](https://github.com/MichaDit/lpsd/actions/runs/38218181121/artifacts/11732812594) | 11732812594 | `benchmark-results/revision-comparison.json` |
| [results-linux-x64-gcc-python-3.12](https://github.com/MichaDit/lpsd/actions/runs/38218181121/artifacts/11733386322) | 11733386322 | `benchmark-results/revision-comparison.json` |
| [results-macos-arm64-clang-python-3.12](https://github.com/MichaDit/lpsd/actions/runs/38218181121/artifacts/11732642844) | 11732642844 | `benchmark-results/revision-comparison.json` |
| [portable-smoke-linux-arm64](https://github.com/MichaDit/lpsd/actions/runs/38218181042/artifacts/11733456036) | 11733456036 | `lpsd-benchmark.vRrFAHGP/disk-algorithm-pairs.json` |
| [portable-smoke-linux-x64](https://github.com/MichaDit/lpsd/actions/runs/38218181042/artifacts/11733161837) | 11733161837 | `lpsd-benchmark.dsoWpufe/disk-algorithm-pairs.json` |
| [portable-smoke-macos-arm64](https://github.com/MichaDit/lpsd/actions/runs/38218181042/artifacts/11732732070) | 11732732070 | `lpsd-benchmark.8iMyHjsR/disk-algorithm-pairs.json` |
| [portable-smoke-windows-x64](https://github.com/MichaDit/lpsd/actions/runs/38218181042/artifacts/11732597764) | 11732597764 | `lpsd-benchmark-cad7da3a64ea/disk-algorithm-pairs.json` |

## Validation scope

- Native comparison uses three alternating revision pairs per configuration, shared Python and compiler, verified loaded build reports.
- Disk comparison uses three alternating full/packed-real pairs with the same FFTW binary, input, grid, and two threads.
- Native timings cover complete API calls; disk timings include preparation, plans, diagnostics, PSD+NSD and cleanup. Imports, input creation and output hashes are excluded.
- Returned spectra are float32; identical returned output hashes are distinct from the separate strict raw-FFT numerical tests.
- CI files fit in RAM at these sizes. These timings do not measure out-of-RAM storage performance or promise cross-machine speedups.
- Previous head 2afaac and its secondary push cohort are preserved separately.

### Windows runtime checks

[Windows job 114707745784](https://github.com/MichaDit/lpsd/actions/runs/38218181042/job/114707745784) completed the actual PowerShell disappearance/access/storage/budget/deadline/root tests, the native available-memory query without fallback, both ordinary and bounded-window three-method comparisons, and all eight disk algorithm cases.

The preceding failed cohort exposed an unsupported `ctypes.CDLL(None)` call in optional allocator trimming; the final run executes that formerly failing matched/full path successfully.

Native test counts include skips and expected failures as separate non-passing entries; disk/FFTW subset jobs overlap the main numerical suite and must not be added as unique tests.

| Artifact | Suite | Passed | Skipped/xfail | Failures/errors |
|---|---|---:|---:|---:|
| results-linux-arm64-gcc-python-3.12 | fast.xml | 2105 | 6 | 0 |
| results-linux-arm64-gcc-python-3.12 | upstream.xml | 18 | 0 | 0 |
| results-linux-x64-gcc-python-3.12 | disk.xml | 204 | 0 | 0 |
| results-linux-x64-gcc-python-3.12 | fast.xml | 2105 | 6 | 0 |
| results-linux-x64-gcc-python-3.12 | fftw.xml | 210 | 0 | 0 |
| results-linux-x64-gcc-python-3.12 | upstream.xml | 18 | 0 | 0 |
| results-macos-arm64-clang-python-3.12 | fast.xml | 1728 | 383 | 0 |
| results-macos-arm64-clang-python-3.12 | upstream.xml | 18 | 0 | 0 |
