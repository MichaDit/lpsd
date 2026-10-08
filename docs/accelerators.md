# GPU and other accelerator assessment

Assessment date: 2026-10-08. The workload counts below use the unchanged
LPSD frequency planner and Kaiser PSLL 200 defaults. The hardware inventory
was collected in the same CPU-only execution environment used for this
optimization work.

## Result and implemented scope

A GPU implementation of the segment dot products is a plausible next step
for large, repeated workloads. **No GPU speedup has been measured here.**
This environment exposes no NVIDIA, DRM render, ROCm, Linux accelerator,
or WSL GPU device nodes, no display/processing PCI devices, and no relevant
GPU compiler, inventory command, or Python GPU runtime package. This states
what the container exposes; it does not establish the physical host's
hardware inventory. See the captured
[inventory and payload report](../benchmarks/results_accelerators.json).

The implemented addition is
[`benchmarks/probe_accelerators.py`](../benchmarks/probe_accelerators.py):
it records accelerator prerequisites and computes the LPSD payload sizes
without allocating a signal or performing a spectral calculation. It keeps
package/device visibility separate from runtime and FP64 execution, which
remain explicitly `not_tested`. Optional installed inventory commands have
bounded execution time. It introduces no GPU dependency into `lpsd_fast`.

A production GPU backend is not enabled without an actual end-to-end
speed and numerical comparison on an accessible device. A port that
compiles, or a large advertised tensor throughput, would not establish the
user's requested wall-time improvement.

### Independent runtime check for the next segment revision

The 2026-10-08 follow-up environment again reports AMD EPYC 9V74, nine
visible logical CPUs and an eight-CPU cgroup quota (`800000 100000`).
CPU feature enumeration includes AVX2, FMA and AVX-512F. No NVIDIA, DRM
render or ROCm device nodes are exposed, and CUDA/ROCm compilers and device
inventory commands are absent. In addition to checking files, a direct
`ctypes` call to the installed `libOpenCL.so.1` loader returned
`clGetPlatformIDs = -1001` with zero platforms; `/etc/OpenCL/vendors`
contains no vendor ICDs. The loader alone is therefore not a usable
OpenCL device in this session. This is a container-access finding, not a
claim about the physical host.

The system-wide BLAS lookup resolves to reference BLAS 3.12.0; NumPy's
own configuration separately reports bundled OpenBLAS 0.3.30 with dynamic
CPU dispatch and 64-bit BLAS indices. Those facts must not be collapsed
into a claim that OpenBLAS is unavailable. The existing C segment kernel
does not call BLAS, and this inventory establishes no BLAS speed advantage.

No device kernel, FP64 device computation, host/device transfer benchmark
or complete GPU LPSD comparison can be measured through the exposed
runtime. A GPU implementation still needs all four, including first-call
costs and comparison with the newly optimized CPU segment code on the
same machine. Consequently this revision adds CPU segment work only; it
does not enable an unmeasured GPU backend or advertise a GPU speedup.

## The kernel worth moving

The existing projected order-0 kernel computes two real dot products per
segment, after subtracting that segment's first input value. Frequencies
and segments expose substantial parallel work. The segment lengths and
Fourier coefficients vary with frequency; in the recorded 10-million
sample case, 651 output frequencies use 649 different lengths. These counts
come from the [native profile](../benchmarks/results_fftw.json), independently
checked against the planning probe.

A first GPU experiment should keep the same input samples, frequency plan,
segment starts, window, detrending projector, normalization, and PSD-only
output contract. A suitable candidate has the following design:

1. Copy the input once and retain it on the device across frequencies.
2. Initially prepare projected FP64 coefficients on the CPU. This isolates
   GPU dot-product validation from changes to window/projector arithmetic.
   Stream or cache coefficients according to available device memory.
3. Generate exact start indices using the inherited repeated-addition and
   rounding convention. Replacing this with `round(segment * shift)` can
   select a different sample at rounding boundaries. Reuse the resulting
   index plan across records with identical parameters.
4. Group similar lengths. Assign short segments to warps/subgroups and
   multiple segments to a block so that coefficients can be reused. For
   long segments, use chunked parallel reductions with bounded scratch
   memory. Keep reads within a segment contiguous.
5. Initially return segment powers for the existing host mean update.
   Move aggregation to the device only after its numerical contract is
   independently checked. The inherited averaging recurrence has observable
   behavior; blindly replacing it with an ordinary average of all segments
   changes the estimator.
6. Reuse device buffers and prepared plans for repeated records. Consider
   graph capture after measuring launch overhead. CUDA graphs can amortize
   repeated work submission, but graph construction/instantiation belongs
   in the first-call timing. See NVIDIA's
   [CUDA Graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

This is a proposed implementation strategy, not a measured GPU algorithm.
The data-residency and contiguous-read choices follow NVIDIA's documented
[data-transfer and memory-access guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#what-runs-on-a-cuda-enabled-device).
They still need measurement for this workload.

### Transfer and allocation costs

The following are **exact payload counts for a proposed FP64 PSD offload**,
not measured PCIe traffic, device allocation, or runtime predictions. MB
and GB in this table are decimal. The machine-readable report retains
integer bytes.

| Payload | 10 million samples | 30 million samples |
| --- | ---: | ---: |
| Input once, FP64 | 80.000 MB | 240.000 MB |
| All frequency coefficient pairs, FP64 | 3.576 GB | 10.036 GB |
| All segment starts, int32 | 86.500 MB | 257.941 MB |
| Host-prepared input + coefficients + starts, H2D total | 3.742 GB | 10.534 GB |
| Segment powers returned for host aggregation, FP64 | 172.999 MB | 515.881 MB |
| Final PSD returned after device aggregation, FP64 | 5,208 bytes | 5,424 bytes |
| Input plus one frequency's coefficients, starts, powers: peak payload | 240.000 MB | 720.000 MB |
| Materializing every overlapping segment as FP64: avoid | 216.580 GB | 678.567 GB |

The per-frequency peak excludes runtime/allocator overhead, reduction and
window scratch, double buffering, and pinned host buffers. Consequently it
is not a recommendation to allocate a device with exactly that memory.
The total coefficient payload differs from the peak live payload: processing
one frequency at a time bounds memory, but still transfers every frequency's
coefficients if they are all prepared on the host. Coefficients at equal
lengths are generally different because their frequencies differ.

The input transfer alone therefore understates the first-call offload cost.
Keeping prepared coefficients resident can remove most of that repeated
transfer, while requiring several GB of device memory for these cases.
Generating/projecting coefficients on the GPU would remove the corresponding
H2D payload, but adds a separate numerical and performance task. A producer/
consumer pipeline with concurrent CPU preparation and GPU execution is
another candidate to measure.

The expanded-segment figures also explain why a straightforward dense
matrix conversion is unattractive. A view can avoid a Python allocation,
but a library may still pack it, and the read/compute work remains. Only a
small number of frequencies share a common length here; applying one large
GEMM or batched FFT does not automatically reuse this variable-length work.
Multiple channels with the same plan could offer a more favorable matrix
layout and deserve a separate measurement if that use case matters.

Tensor cores are not categorically excluded by FP64: current cuBLAS 13.4
documents [FP64 fixed-point emulation based on the Ozaki schemes](https://docs.nvidia.com/cuda/cublas/index.html#fixed-point).
Its dynamic precision mode selects a representation from the input data
and can fall back to native FP64. This is a further candidate for suitably
shaped matrix work, especially many channels with a common plan. It does
not remove LPSD's different coefficient vectors, irregular segment starts,
matrix packing, or emulation workspace costs. A single-frequency reduction
with only two coefficient columns should not be assigned the library's
large-matrix throughput without measurement. Lower fixed precision would
also require the same spectral error checks as any other approximate path.

## Which hardware is a realistic target?

The CPU comparator is already a SIMD and multicore implementation. A fair
offload comparison must use the current optimized CPU checkout, including
its segment improvements, rather than only the original scalar package.
Replacing each pair of dot products with individual BLAS calls would keep
the overlapping scans and add a call boundary per segment. BLAS or dedicated
CPU matrix instructions become more interesting if several records/channels
can share a packed coefficient problem; that layout and its packing cost
must be measured instead of inferred from a large GEMM benchmark.

| Target | LPSD-specific assessment |
| --- | --- |
| NVIDIA CUDA GPU with strong FP64 throughput | A sensible first implementation target when accessible. The custom segmented reduction, its memory behavior, and all host overhead still decide wall time. H100 specifications distinguish ordinary FP64 from FP64 Tensor Core throughput; the latter cannot be substituted into a dot-product runtime estimate. |
| AMD Instinct / ROCm | Also a credible target for the same FP64 strategy, using HIP and device-specific subgroup tuning. AMD lists both FP64 and memory capacity/bandwidth explicitly for MI355X. This is capability evidence, not an LPSD benchmark. |
| Consumer GPU | Evaluate its actual FP64 rate and runtime support. CUDA's documented arithmetic throughput varies sharply by compute capability. FP32/FP16 headline performance does not establish useful FP64 LPSD performance. |
| Intel GPU / SYCL or OpenCL | Query the selected device and driver for FP64 support before compiling a numerical candidate. Intel's architecture guide distinguishes native double support across product families; a vendor name alone is insufficient. |
| Apple GPU / Metal | Metal Shading Language 4.1 does not expose `double` or `long double`. A direct FP64 kernel port is unavailable through this language; a mixed-precision or emulated implementation would need its own accuracy and timing study. The existing ARM CPU path remains the validated option. |
| NPU | The inspected Intel OpenVINO NPU path exposes F32/F16 model precisions with FP16 hardware computation. That is not an immediate fit for an arbitrary-signal FP64 spectral reduction. No NPU implementation is justified by current evidence. |
| FPGA | A custom streaming or fixed-configuration engine is technically possible; Vitis HLS supports mapping double-precision arithmetic to FPGA resources. No board, synthesis result, or end-to-end measurement is available here. Changing variable lengths, resource use, external-memory traffic, and numerical behavior require a distinct hardware design effort. |

Primary sources: [NVIDIA H100 specifications](https://www.nvidia.com/en-eu/data-center/h100/),
[NVIDIA arithmetic throughput by compute capability](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions),
[AMD MI355X specifications](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html),
[Intel Xe architecture guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html),
[Apple MSL 4.1, section 2.1, p. 25](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
[OpenVINO 2026 NPU data types](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html),
and [AMD Vitis HLS operator binding](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Bind_op).
All were inspected on 2026-10-08. The Apple PDF is dated 2026-06-04,
14,564,497 bytes, SHA-256
`41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5`.

### Precision is a constraint on the complete computation

An FP32 output column does not imply that the input or dot products can be
converted to FP32 safely. For example, rounding a signal with a large DC
offset to FP32 *before* centering can remove fluctuations that were present
in its FP64 samples. Scaling the physical units can make the lost PSD
material in absolute terms. Pre-centering helps some cases, but does not
provide a uniform relative-error guarantee near spectral cancellations.

A GPU implementation must test DC plus small residuals, high dynamic range,
off-bin tones and deep nulls, constant/zero signals, varied input scales,
short/long segments, and supported windows. The existing
[`check_accuracy.py`](../benchmarks/check_accuracy.py) signal suite is a
starting point; a GPU adapter must expose results to those same comparisons.
Report absolute and relative PSD **and NSD** differences separately. The
allowed small absolute error needs a physical scale; the existing synthetic
audit floors are not universal unit-independent tolerances.

Changing reduction order and using fused multiply-add changes rounding even
with FP64. GPU FP64 also does not reproduce x86 extended-precision projector
preparation automatically. NVIDIA's
[floating-point guide](https://docs.nvidia.com/cuda/floating-point/index.html)
explains these distinctions. A private compensated projector or validated
host preparation is preferable to assuming identical arithmetic. Preserve
the real PSD output convention and check overflow/nonfinite behavior as
well as finite-input differences; the archived
[global FMA experiment](../benchmarks/experiments/fp_contract/README.md)
shows why dtype checks matter.

A single full-record cuFFT/rocFFT followed by logarithmic power grouping is
a different estimator. The [FFTW comparison](fftw-comparison.md) already
demonstrates pointwise differences much larger than one percent. GPU FFT
libraries do not resolve that mathematical difference. They are appropriate
for a separately named estimator or where several requested coefficients
actually share the same windowed transform.

## Reproduce the inventory and establish a GPU-machine baseline

Run from the repository root. The first command also works without any
LPSD dependencies when invoked as a file; the second uses the installed
Python/NumPy/SciPy dependencies and unchanged planner.

```bash
python benchmarks/probe_accelerators.py --inventory-only --probe-commands > accelerator-inventory.json
python -m benchmarks.probe_accelerators --probe-commands --sizes 10000000 30000000 > accelerator-payloads.json
```

On the **same machine and exact checkout** used for a future GPU candidate,
build the normal native libraries and collect the CPU comparator. Choose
the worker count for the actual CPU quota; eight is the historical case,
not a universal best setting.

```bash
python build_native.py
python -m lpsd_fast.build --native
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.bench_lpsd \
  --backend fast --kernel fast --outputs psd --n 10000000 --workers 8 \
  --n-frequencies 1000 --n-averages 100 --window kaiser --psll 200 --order 0 \
  --max-working-mb 4096 --full-length-warmup --warmups 1 --repeats 7 \
  --profile --output cpu-current-10m.json
```

Verify the GPU's transfers separately using the manufacturer's tool after
installing/building it for the actual driver/toolkit. For an installed
[NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth), the following
records host-to-device and device-to-host copy-engine throughput with data
verification enabled. Its buffer size is in MiB.

```bash
nvbandwidth --version
nvbandwidth -b 256 -i 7 --format json \
  -t host_to_device_memcpy_ce device_to_host_memcpy_ce > nvbandwidth-h2d-d2h.json
```

For an installed [AMD TransferBench](https://rocm.docs.amd.com/projects/TransferBench/en/latest/),
the following exercises unidirectional CPU/GPU transfer pairs with DMA for
GPU executors. Read the CPU-to-GPU and GPU-to-CPU entries, not a GPU-to-GPU
or aggregate bidirectional number. Record the memory types and CPU NUMA
locations printed by the tool.

```bash
P2P_MODE=1 USE_GPU_DMA=1 TransferBench p2p > transferbench-p2p.txt
```

These are bandwidth prechecks, not LPSD speed measurements. Consult
[TransferBench's timing scopes](https://rocm.docs.amd.com/projects/TransferBench/en/latest/conceptual/transferbench-timing.html)
when interpreting CPU wall time versus device events. No transfer command
above was run in this CPU-only environment.

## Required timing scopes for an actual GPU candidate

No GPU LPSD CLI currently exists in this repository. Once a candidate is
available, record these scopes separately, using the CPU run above as the
matched comparator:

| Scope | Inside the timer |
| --- | --- |
| First complete host call | Plan construction, relevant runtime/JIT initialization, allocations, input and coefficient transfers, all kernels, result transfer, normalization, output conversion, and final device synchronization. If imports or process launch are excluded, state that explicitly. |
| Warm complete host call | A fresh input record already in host memory; the same defined plan/buffer reuse policy; all required H2D/D2H work and output assembly; final synchronization. |
| Prepared resident call | Input, plan and coefficients already on device; execution and final device synchronization. Measure result transfer separately and state whether it is included. |
| Device kernel only | Device event timing of segmented dots/reduction. Diagnostic only; never compare this directly against complete CPU API time as a speedup claim. |

For asynchronous APIs, synchronize before starting the host timer and after
the last dependent operation. Synchronize every stream that contributes to
the result, or wait on one completion event that depends on all of them.
Otherwise a Python timer can measure only work submission. Profile a
separate call to resolve H2D, coefficient preparation, segment dots, power
aggregation, D2H and synchronization costs without mixing profiling
overhead into ordinary repetitions.

Use identical input arrays, seeds, lengths, output frequency labels,
window/overlap, detrending, precision/output dtype, and number of segments.
Alternate CPU and GPU measurements where practical, report every repetition
and medians/ranges, and include the realistic number of records/channels.
Record the GPU model, driver/runtime, selected device, FP64 capability,
power/clock policy, memory budget, transfer mode, compiler flags, and source
commit. A useful production backend must pass the numerical contract and
show a repeatable advantage in the user's relevant **complete-call** scope;
no numerical speedup threshold is assumed by this report.
