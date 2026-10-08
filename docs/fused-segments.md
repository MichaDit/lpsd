# Bounded FP64 fused segment projections

The optional fast auto-spectrum path can now use four- and eight-stream
FP64 projections with explicit fused multiply-add instructions. It retains every
frequency, Fourier/window coefficient, segment start and sample. Each
accumulator belongs to one segment, and every inherited mean update still
runs in the original order. The existing scalar, SIMD and selected native
entry points retain their prior arithmetic.

This changes rounding within eligible projections. It does not change the
spectral estimator or repair the inherited 1.0.6 mean/variance defects.

## Kernel and dispatch

The eight-stream kernel keeps sixteen real/imaginary vector accumulators.
Explicit AVX-512 intrinsics permit short tree reductions without
the large stack workspace generated for the previous sixteen-variable
OpenMP reduction. The eight input vectors keep distinct segment anchors.
Fewer than eight remaining samples use ordinary FP64 multiply/add; there
are no reads beyond the supplied segment.

For `256 <= L < 1024`, the kernel instead executes two four-stream passes
using 256-bit AVX/FMA vectors. Each pass keeps eight real/imaginary vector
accumulators, shares both coefficient loads between four anchored input
streams, and handles its final `L % 4` samples without vector overreads.
This reduces the register and instruction pressure of the eight-stream
layout in that interval. The additional AVX/FMA capability check runs once
per frequency. The outer group still represents eight original segments.
Longer vectors retain the eight-stream implementation: experiments with
four-stream passes there were slower and were rejected.

The Python API selects the additive bounded entry point only for single-worker projected
`kernel="fast"` auto spectra on a supported build and CPU. Calls with multiple frequency workers retain the ordinary selected path,
because the complete-call comparisons did not establish a repeatable FMA
benefit there. Other kernels and CSD retain their existing entry points. The current architecture-specific
implementation requires 64-bit x86 with AVX-512F; portable function-clone
builds check the executing CPU. Builds without this implementation use the
existing direct paths. This is a capability statement, not a performance
claim for other processors.

The new route requires segment length at least 128 and enough remaining
segments for an eight-segment group after the preserved initial projections.
It therefore needs at least ten total segments for lengths 256 through 1023,
and at least sixteen otherwise. Checking these counts before the coefficient
range avoids scanning long coefficient arrays when no fused group can run.
The existing nonfused four-way and single-segment kernels handle remainders.

The first batch is deliberately preserved:

| Previous initial route | Projections kept on that route | Reason |
| --- | ---: | --- |
| Eight-way selected kernel | First eight segments | Retain its exact initial reduction order |
| Single-segment selected kernel | First two segments | Retain the cancellation-sensitive first mean reset |

The inherited second mean update is `P0 + (P1 - P0)`. If the first
periodogram is enormous compared with the next, changing an ulp of its
projection can create a significant final error. Preserving these initial
projections avoids introducing that sensitivity into the fused tail. The
subsequent mean update order, the final overwriting variance assignment and
output rounding points are unchanged.

## Range checks and overflow

An unconditional FMA replacement can avoid an intermediate overflow that
separate multiplication and addition would produce. A range check therefore
precedes this optimization.

The Python caller obtains the largest absolute input value using one minimum
and one maximum reduction per complete channel. These reductions require no
input-sized `abs` allocation. Input finiteness has already been checked.
The native preparation then verifies the projected real and imaginary
coefficients. Both the input peak and coefficient magnitudes must be at most
`2^100`. Larger values and nonfinite coefficients use the ordinary selected
kernel.

These deliberately generous limits cover ordinary measurement units while
leaving a large margin for every native intermediate:

| Quantity | Conservative magnitude bound |
| --- | ---: |
| Input sample or projected coefficient | `2^100` |
| Anchored input difference | `2^101` |
| One product | `2^201` |
| Real or imaginary dot, given `L < 2^31`, including rounding reserve | Less than `2^233` |
| Auto-spectrum periodogram, including rounding reserve | Less than `2^467` |
| Inherited fourth-order native variance intermediates | Less than `2^940` |

Binary64 overflow begins near `2^1024`. Thus fused arithmetic cannot conceal
an overflow of the corresponding nonfused projection or native statistics
inside the accepted range. This does not promise a universal relative-error
bound at spectral nulls, nor alter the public float32/complex64 output
precision. Very small differences from underflow and ordinary rounding are
still assessed using relative and application-specific absolute tolerances.

The coefficient guard uses a branch-free bitwise OR of two negated `<=`
comparisons. Both comparisons always run, which allows packed vector
comparisons without relaxing the compiler's floating-point settings. A NaN
still makes its negated comparison true and disables FMA. The whole build
continues to use `-ffp-contract=off`; only explicit intrinsics fuse operations.

## Native interface

`fast_dft_selected_bounded` adds these arguments after the existing selected
arguments:

| Argument | Contract |
| --- | --- |
| `double input_peak` | Independently checked upper bound on every absolute input sample; infinity requests the ordinary kernel |
| `double *preparation_seconds` | Optional preparation timer; includes coefficient-range checking |
| `double *segments_seconds` | Optional segment timer; supplied together with the preparation timer |
| `long *fused_batches` | Optional count of logical eight-segment groups that used FMA, including groups executed as two four-stream passes |

Native callers must keep samples unchanged after obtaining the bound and
throughout the call. Negative or NaN bounds are rejected. The projected
coefficient range is independently checked by the native implementation.
`native_segment_fma_supported()` reports whether this build and CPU can use
the fused implementation. Existing entry points do not enable it.

Profiled Python results report `input_bound_s` separately, plus
`fused_segment_batches` per frequency and in aggregate. The segment method
is `direct_fma` when fused batches ran. `L*K` continues to count logical
sample coverage; coefficient reuse and SIMD do not remove samples.

## Validation and measurement scope

The focused suite in `test_fast/test_fused_segments.py` covers 229 cases:
fractional starts, batch and remainder boundaries, modes 2/3, order-zero
and order-one native projections, ordinary noise, tones, DC plus nanovolt
noise, constants, extreme input/coefficient ranges, first-periodogram
cancellation, selected statistics and the complete profiled Python API.
Together with the 209 existing segment-batching regressions, these make 438
focused tests. They cover every vector tail, exact and adjacent input and
coefficient bounds, and both the supported and ordinary fallback paths.

The first integrated all-eight-stream candidate did not demonstrate a
consistent larger PSD improvement across complete API calls. In particular,
its eight-worker timings exposed an unresolved regression risk on the shared
host. That entire series is retained alongside the subsequent four/eight-
stream screening in the [dispatch experiment](../benchmarks/experiments/fma_dispatch/README.md).
The [fast.4 report](performance-fast4.md) separates those candidates from the
final built version and states the timing variation. Native microbenchmarks
and instrumented profiles are not substituted for complete-call results.

Reproduce the focused numerical check after building both libraries:

```sh
make compile
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  test_fast/test_fused_segments.py test_fast/test_segment_batches.py
```

Use the existing complete-version benchmark harness for performance
comparisons. Keep input creation outside the timer, include input-bound
checking and preparation inside complete-call wall time, retain every
repetition, and serialize intentional workloads on a shared host.
