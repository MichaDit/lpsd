# SPDX-License-Identifier: GPL-3.0-or-later
"""Inventory accelerator prerequisites and size an unchanged LPSD workload.

This is a capability/planning report, not a GPU benchmark or a GPU backend.
No input signal, coefficient vectors, or expanded segments are allocated.
The inventory-only mode needs only Python's standard library.
"""

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess


DEVICE_PATTERNS = (
    "/dev/nvidia*", "/dev/dri/renderD*", "/dev/kfd",
    "/dev/accel/accel*", "/dev/dxg",
)
MODULES = ("cupy", "torch", "numba", "dpctl", "dpnp", "jax", "pyopencl", "mlx")
COMMANDS = ("nvidia-smi", "nvcc", "rocminfo", "amd-smi", "rocm-smi", "hipcc",
            "sycl-ls", "clinfo", "system_profiler")
PROBE_ARGUMENTS = {
    "nvidia-smi": ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    "rocminfo": [],
    "sycl-ls": [],
    "clinfo": ["--list"],
    "system_profiler": ["SPDisplaysDataType", "-json"],
}


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def pci_accelerators(root=Path("/sys/bus/pci/devices")):
    """Record display/processing PCI classes; presence does not imply access."""
    result = []
    try:
        devices = sorted(root.iterdir())
    except OSError:
        return result
    for device in devices:
        class_text = read_text(device / "class")
        try:
            base_class = int(class_text, 16) >> 16
        except (ValueError, TypeError):
            continue
        if base_class not in (0x03, 0x12):
            continue
        driver = device / "driver"
        result.append({
            "address": device.name, "class": class_text,
            "vendor": read_text(device / "vendor"),
            "device": read_text(device / "device"),
            "driver": driver.resolve().name if driver.is_symlink() else None,
        })
    return result


def command_probe(command, arguments, timeout):
    """Use read-only inventory commands; failure is not proof of no hardware."""
    argv = [command, *arguments]
    result = {"argv": argv}
    try:
        process = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                 errors="replace", check=False)
        result.update(returncode=process.returncode,
                      status="completed" if process.returncode == 0 else "failed",
                      stdout=process.stdout[:16384], stderr=process.stderr[:16384],
                      output_truncated=(len(process.stdout) > 16384 or
                                        len(process.stderr) > 16384))
    except subprocess.TimeoutExpired:
        result.update(status="timed_out", timeout_s=timeout)
    except OSError as error:
        result.update(status="failed_to_start", error=str(error))
    return result


def inventory(probe_commands=False, timeout=5.0):
    commands = {name: shutil.which(name) for name in COMMANDS}
    modules = {}
    for name in MODULES:
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            modules[name] = None
    nodes = {pattern: sorted(glob.glob(pattern)) for pattern in DEVICE_PATTERNS}
    result = {
        "platform": platform.system(), "machine": platform.machine(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_max": read_text("/sys/fs/cgroup/cpu.max"),
        "memory_max": read_text("/sys/fs/cgroup/memory.max"),
        "device_nodes": nodes, "pci_display_or_processing_devices": pci_accelerators(),
        "commands": commands, "python_modules_present_not_imported": modules,
        "runtime_computation": "not_tested",
        "fp64_computation": "not_tested",
        "notes": [
            "Device nodes, PCI devices, executables, and packages are prerequisite evidence only.",
            "No CUDA, HIP, SYCL, OpenCL, Metal, or NPU kernel is executed by this probe.",
            "Absent container devices do not prove the physical host has no accelerator.",
            "Check the selected runtime device's FP64 support before running a numerical benchmark.",
        ],
    }
    if probe_commands:
        result["command_probes"] = {
            name: command_probe(commands[name], arguments, timeout)
            for name, arguments in PROBE_ARGUMENTS.items() if commands[name]
        }
    return result


def workload_storage(n, lengths, counts):
    """Exact payload counts for a proposed single-channel FP64 PSD offload.

    The streamed payload holds the signal and ONE frequency's Cr/Ci, starts,
    and segment powers. It excludes allocator/runtime overhead, window and
    reduction scratch, double buffering, and pinned host buffers.
    """
    if not 1 <= n <= 2147483647 or len(lengths) != len(counts) or not lengths:
        raise ValueError("Require a nonempty matching plan and positive data length.")
    if any(not 1 <= length <= n or count < 1 for length, count in zip(lengths, counts)):
        raise ValueError("Invalid segment length or count.")
    total_lengths = sum(lengths)
    total_segments = sum(counts)
    visits = sum(length * count for length, count in zip(lengths, counts))
    return {
        "frequencies": len(lengths), "distinct_lengths": len(set(lengths)),
        "sum_lengths": total_lengths, "sum_segments": total_segments,
        "sample_iterations": visits,
        "input_fp64_bytes": 8 * n,
        "all_frequency_coefficients_fp64_bytes": 16 * total_lengths,
        "all_segment_starts_int32_bytes": 4 * total_segments,
        "all_segment_powers_fp64_bytes": 8 * total_segments,
        "materialized_segment_samples_fp64_bytes_avoid": 8 * visits,
        "one_frequency_streamed_payload_peak_bytes": 8 * n + max(
            16 * length + 12 * count for length, count in zip(lengths, counts)),
        "host_prepared_h2d_payload_bytes": 8 * n + 16 * total_lengths + 4 * total_segments,
        "host_legacy_reduction_d2h_segment_powers_bytes": 8 * total_segments,
        "device_reduced_d2h_psd_fp64_bytes": 8 * len(lengths),
        "notes": [
            "Payload counts are not measured transfers, peak device allocation, or runtime predictions.",
            "Coefficients differ by frequency even where lengths match.",
            "Starts may use int32 because the current API limits N to INT_MAX.",
            "Keeping starts/coefficients on device across calls changes the transfer budget.",
            "Device reduction must preserve the inherited estimator's mean and special-value behavior.",
        ],
    }


def plan_workload(n, frequencies=1000, averages=100, psll=200.0, overlap=None):
    # Deferred imports keep --inventory-only useful on a machine without lpsd.
    from lpsd_fast.api import _frequency_plan
    from lpsd_fast.planning import _kaiser_alpha, _kaiser_rov

    if overlap is None:
        overlap = float(_kaiser_rov(_kaiser_alpha(psll)))
    plan = _frequency_plan(n, 1.0, overlap, 1, 0, frequencies, averages)
    lengths = [int(value) for value in plan[3]]
    # Match native arithmetic, including Python's olap * 100 API conversion.
    native_ovfact = 1.0 / (1.0 - (overlap * 100.0) / 100.0)
    counts = [math.floor(((n - length) * native_ovfact) / length + 1.0 + 0.5)
              for length in lengths]
    return {
        "parameters": {"n": n, "sample_rate": 1.0, "n_frequencies": frequencies,
                       "n_averages": averages, "psll": psll, "overlap": overlap,
                       "n_min_bins": 1, "min_segment_length": 0},
        "source_sha256": {
            "lpsd_fast.api": hashlib.sha256(Path(inspect.getsourcefile(_frequency_plan)).read_bytes()).hexdigest(),
            "lpsd._helpers": hashlib.sha256(Path(inspect.getsourcefile(_kaiser_rov)).read_bytes()).hexdigest(),
        },
        "native_vs_planner_segment_count_differences": sum(
            count != int(planned) for count, planned in zip(counts, plan[4])),
        "storage": workload_storage(n, lengths, counts),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--probe-commands", action="store_true",
                        help="Run installed read-only device inventory commands.")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Per inventory command timeout, in seconds (default: 5).")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10000000, 30000000])
    parser.add_argument("--frequencies", type=int, default=1000)
    parser.add_argument("--averages", type=int, default=100)
    parser.add_argument("--psll", type=float, default=200.0)
    parser.add_argument("--overlap", type=float)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 60:
        parser.error("--timeout must be in (0, 60] seconds.")
    if not math.isfinite(args.psll):
        parser.error("--psll must be finite.")
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "accelerator_inventory_and_lpsd_payload_model_not_performance",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inventory": inventory(args.probe_commands, args.timeout),
        "workloads": [] if args.inventory_only else [
            plan_workload(n, args.frequencies, args.averages, args.psll, args.overlap)
            for n in args.sizes
        ],
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
