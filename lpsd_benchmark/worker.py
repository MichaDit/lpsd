"""One isolated numerical case. Never called in the controller process.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import ctypes.util
import errno
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

from .privacy import integer
from .resources import MIB, smoothing_controls


def _write(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_white(n, sample_rate, seed):
    """Historical seeded noise with only one N-element RAM allocation."""
    import numpy as np
    values = np.empty(n, dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, n]))
    scale = 10e-9 * np.sqrt(sample_rate / 2)
    for first in range(0, n, 262144):
        part = values[first:first + 262144]
        rng.standard_normal(len(part), out=part)
        part *= scale
    return values


def noise_line_mask(frequencies, lengths, sample_rate, tones, provider, beta):
    """The historical +/-3 native-ENBW exclusion, including output rounding.

    Calculate actual native window sums rather than an asymptotic ENBW
    constant. Only one window is live, and repeated lengths reuse a scalar.
    This quality check runs after the complete-call timer. Tone-containing
    presets already passed their larger coloured-synthesis memory admission.
    """
    import numpy as np
    frequencies = np.asarray(frequencies, dtype=np.float64)
    bandwidths = np.empty(len(frequencies), dtype=np.float32)
    by_length = {}
    for index, length in enumerate(lengths):
        length = int(length)
        if length not in by_length:
            window = provider.generate(length, beta)
            s1, s2 = provider.sums(window)
            del window
            # The original public enbw column is complex64. Its real part
            # and the historical multiplication by three are both float32.
            by_length[length] = np.float32(sample_rate * s2 / (s1 * s1))
        bandwidths[index] = by_length[length]
    excluded = np.zeros(len(frequencies), dtype=bool)
    for tone in tones:
        excluded |= np.abs(frequencies - tone["frequency_hz"]) <= 3 * bandwidths
    return excluded


def isolated_tone_metrics(signal_name, frequencies, psd, tones):
    """Report widths only for the isolated off-bin line, as in the harness.

    The mixed preset's low-frequency noise can exceed its first line; a
    broad argmax there is not a defensible line-width measurement.
    """
    if signal_name != "offbin_tone" or not tones:
        return {}
    import numpy as np
    from benchmarks.fftw_comparison_metrics import tone_metrics
    tone = tones[0]
    if np.count_nonzero(np.abs(np.asarray(frequencies) - tone["frequency_hz"]) <= .6) < 3:
        return {}
    return tone_metrics(frequencies, psd, tone["frequency_hz"], tone["rms_v"])


def _load_fftw(bf, requested_threads, explicit=None, explicit_threads=None):
    """Load genuine FFTW symbols, from system libraries or a pyFFTW wheel.

    Loading a wheel does not replace FFTW by NumPy's FFT. Some wheels bundle
    the threading symbols in the main DLL, others in a separate library.
    A missing threading runtime is recorded as a single-thread execution.
    """
    candidates = []
    if explicit:
        candidates.append(str(explicit))
    else:
        system = ctypes.util.find_library("fftw3")
        if system:
            candidates.append(system)
        try:
            distribution = metadata.distribution("pyfftw")
            for entry in distribution.files or ():
                name = Path(entry).name.lower()
                if (re.match(r"(?:lib)?fftw3[-._]", name)
                        and any(marker in name for marker in (".so", ".dylib", ".dll"))):
                    candidates.append(str(distribution.locate_file(entry)))
        except metadata.PackageNotFoundError:
            pass
    candidates = list(dict.fromkeys(candidates))
    system_threads = ctypes.util.find_library("fftw3_threads")
    thread_candidates = [str(explicit_threads)] if explicit_threads else [system_threads]
    thread_candidates += [name for name in candidates if "threads" in Path(name).name or "omp" in Path(name).name]
    primary = [name for name in candidates if "threads" not in Path(name).name and "omp" not in Path(name).name]
    for name in primary:
        for thread_library in list(dict.fromkeys(thread_candidates + [name])):
            if not thread_library:
                continue
            try:
                return bf.FFTWLibrary(name, thread_library, requested_threads)
            except (OSError, RuntimeError, AttributeError, ValueError):
                continue
    for name in primary:
        try:
            return bf.FFTWLibrary(name, None, 1)
        except (OSError, RuntimeError, AttributeError, ValueError):
            pass
    raise ImportError("FFTW double-precision runtime unavailable")


def resolve_point_plan(api, n, fs, base_points, factor, explicit_points, averages):
    """Reuse the native planner; find the smallest J giving the requested count."""
    overlap = api._kaiser_rov(api._kaiser_alpha(200.0))
    plans = {}

    def plan(j):
        if j not in plans:
            plans[j] = api._frequency_plan(n, fs, overlap, 1, 0, j, averages)
        return plans[j]

    base = plan(base_points)
    if factor is None:
        actual_j = explicit_points
        requested = len(plan(actual_j)[0])
    else:
        requested = max(2, int(round(len(base[0]) * factor)))
        low, high = 2, max(2, base_points)
        while len(plan(high)[0]) < requested and high < 65536:
            high = min(65536, high * 2)
        if len(plan(high)[0]) < requested:
            raise ValueError("Requested output count cannot be reached within J<=65536")
        while low < high:
            middle = (low + high) // 2
            if len(plan(middle)[0]) >= requested:
                high = middle
            else:
                low = middle + 1
        actual_j = low
        # Preserve the historical base J for factor one: a count plateau can
        # start earlier but have different frequency labels at a smaller J.
        if factor == 1:
            actual_j = base_points
    selected = plan(actual_j)
    return selected, {
        "base_target_points": base_points, "base_actual_points": len(base[0]),
        "target_points": actual_j, "requested_output_points": requested,
        "actual_points": len(selected[0]), "point_count_resolved": True,
    }


def _case_smoothing_controls(case):
    """Validate the scalar request again before importing numerical modules.

    Older private cases without control fields retain the historical default.
    An explicitly supplied limit/reserve may not disagree with the resolver.
    """
    controls = smoothing_controls(case.get("method", "lpsd"), case["requested_threads"],
                                  case.get("requested_smoothing_workers"))
    for key, value in controls.items():
        if key == "requested_smoothing_workers" and case.get("method", "lpsd") != "matched":
            continue  # A validated global CLI request has no effect on other methods.
        if key in case and (type(case[key]) is not type(value) or case[key] != value):
            raise ValueError("Inconsistent smoothing control or reserve")
    return controls


def _smoothing_result_fields(case, metadata=None, disk_details=None):
    """Report configured capacity and work ranges, not sampled active threads."""
    controls = _case_smoothing_controls(case)
    if case["method"] != "matched":
        return {"smoothing_pool_capacity": 0, "smoothing_task_ranges": 0,
                "smoothed_frequency_points": 0, "smoothing_backend": "not_applicable"}
    if case["storage"] == "disk":
        details = (disk_details or {}).get("smoothing", {})
        supplied = details.get("workers")
        capacity = details.get("pool_capacity")
        ranges = details.get("task_ranges")
        points = details.get("high_points")
        backend = details.get("backend")
    else:
        details = metadata or {}
        supplied = details.get("requested_smoothing_workers")
        capacity = details.get("smoothing_pool_capacity")
        ranges = details.get("smoothing_task_ranges")
        points = details.get("smoothed_frequency_points")
        backend = details.get("smoothing_backend")
    limit = controls["smoothing_workers_limit"]
    if (integer(supplied, 1) != limit or integer(capacity, 0, limit) is None
            or integer(points, 0, 65536) is None or integer(ranges, 0, 2 * limit) is None
            or ranges > points or backend != "native"):
        raise ValueError("Measured smoothing configuration differs from admission")
    return {"smoothing_pool_capacity": capacity, "smoothing_task_ranges": ranges,
            "smoothed_frequency_points": points, "smoothing_backend": backend}


def _native_working_budget(case):
    """Return the bounded-native allowance in MiB and available buffer bytes.

    RAM cases retain the original float64 input and reserve interpreter and
    worker overhead before assigning the remaining configured budget. The
    hybrid releases its FFT buffers before this native phase. Disk cases
    retain their separately configured working allowance and headroom model.

    The adapter has a 64-MiB minimum capacity. Return no allowance when that
    cannot fit, so preflight can skip before constructing it or the input.
    This scalar calculation does not allocate numerical arrays.
    """
    controls = _case_smoothing_controls(case)
    headroom = case["ram_budget_bytes"] - 256 * MIB - controls["smoothing_runtime_reserve_bytes"]
    if case["storage"] == "ram":
        headroom -= 8 * case["n"] + case["requested_threads"] * 3 * MIB
        working_mb = headroom / MIB
    else:
        working_mb = case["low_working_mb"]
    if headroom < 64 * MIB:
        return None, headroom
    return working_mb, headroom


def _native_scratch_fields(method, storage, attrs, disk_details=None):
    """Scalar pool capacities from completed-call metadata, never RSS.

    Read the returned metadata, not the constructor's admission preview.
    Missing lifecycle data remains absent for older adapters and FFTW-only
    calls. Planned or failed replacements do not publish retirement indices.
    """
    if method not in ("lpsd", "matched"):
        return {}
    if storage == "disk":
        source = (disk_details.get("lpsd" if method == "lpsd" else "low_lpsd")
                  if isinstance(disk_details, dict) else None)
    elif isinstance(attrs, dict):
        source = attrs.get("lpsd_disk" if method == "lpsd" else "matched_smoothing_once")
        if method == "matched":
            source = source.get("low_frequency_preparation") if isinstance(source, dict) else None
    else:
        source = None
    lifecycle = source.get("scratch_pool_lifecycle") if isinstance(source, dict) else None
    if not isinstance(lifecycle, dict):
        return {}
    fields = {}
    for key, public_key in (("initial_bytes", "native_scratch_initial_bytes"),
                            ("final_bytes", "native_scratch_final_bytes"),
                            ("peak_allocated_bytes", "native_scratch_peak_bytes")):
        value = integer(lifecycle.get(key))
        if value is not None:
            fields[public_key] = value
    state = lifecycle.get("state")
    retired = isinstance(state, str) and state == "retired"
    fields["native_scratch_retired"] = retired
    if retired:
        for key in ("retirement_index", "retirement_source_index"):
            value = integer(lifecycle.get(key))
            if value is not None:
                fields["native_scratch_" + key] = value
    return fields


def _address_guard(case):
    """Additional Linux guard on allocations plus explicitly allowed mappings.

    Existing library mappings are accounted separately. RLIMIT_AS is virtual
    address space, not resident RAM; public results do not conflate the two.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import resource
        status = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            fields = line.split()
            if fields[0] in ("VmSize:", "VmRSS:"):
                status[fields[0]] = int(fields[1]) * 1024
        base = status["VmSize:"] - status["VmRSS:"]
        maps = ((8 if case["method"] == "lpsd" else 28) * case["n"] + 16) if case["storage"] == "disk" else 0
        desired = base + maps + int(case["ram_budget_bytes"])
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            desired = min(desired, hard)
        if soft != resource.RLIM_INFINITY:
            desired = min(desired, soft)
        resource.setrlimit(resource.RLIMIT_AS, (desired, hard))
    except (ImportError, OSError, KeyError, ValueError):
        pass


def run(case, directory):
    controls = _case_smoothing_controls(case)
    case.update(controls)
    import numpy as np
    from benchmarks import bench_fftw as bf
    from benchmarks.fftw_comparison_signals import signal, theory
    from benchmarks.fftw_comparison_metrics import summarize_noise
    from benchmarks.matched_smoothing import estimate_once
    from lpsd_fast import api

    try:
        api._native()
    except (RuntimeError, OSError, AttributeError) as exc:
        raise ImportError("Native runtime unavailable") from exc
    plan, resolved = resolve_point_plan(api, case["n"], case["sample_rate_hz"],
                                        case["base_target_points"], case.get("point_factor"),
                                        case["target_points"], case["n_averages"])
    case.update(resolved)
    n, method = case["n"], case["method"]
    result = {**resolved, **controls}
    import lpsd_fast
    result["native_version"] = lpsd_fast.__version__
    result["native_sha256"] = _file_hash(api._LIB._name)
    source = hashlib.sha256()
    from benchmarks import matched_smoothing
    source_paths = [Path(api.__file__), Path(bf.__file__), Path(matched_smoothing.__file__), Path(__file__),
                    Path(__file__).with_name("resources.py")]
    if method != "lpsd":
        source_paths.append(Path(bf.__file__).parent / "_fftw_native.py")
        if case.get("window_storage") == "fft_input_then_blocked":
            source_paths.append(Path(bf.__file__).parent / "_bounded_window.py")
    if case["storage"] == "disk" or method in ("lpsd", "matched"):
        source_paths.extend(Path(bf.__file__).parent / name for name in ("bench_fftw_disk.py", "_fftw_disk.py", "_lpsd_disk.py"))
    for path in source_paths:
        source.update(path.name.encode("ascii") + b"\0")
        source.update(path.read_bytes())
    result["source_sha256"] = source.hexdigest()
    backend = None
    if method != "lpsd":
        backend = _load_fftw(bf, case["requested_threads"], case.get("fftw_library"),
                             case.get("threads_library"))
        result["fftw_sha256"] = backend.evidence.get("binary_sha256")
        found_version = re.search(r"fftw-([0-9]+(?:\.[0-9]+){1,3})", backend.version)
        if found_version:
            result["fftw_version"] = found_version.group(1)
        if backend.threads_evidence:
            result["fftw_threads_sha256"] = backend.threads_evidence.get("binary_sha256")
    result["actual_threads"] = backend.threads if backend else case["requested_threads"]
    bounded_options = None
    if method in ("lpsd", "matched"):
        working_mb, headroom = _native_working_budget(case)
        if working_mb is None:
            return {**result, "status": "skipped", "reason": "memory_budget"}
        from benchmarks._lpsd_disk import DiskLPSDSubset
        bounded_options = dict(sample_rate=case["sample_rate_hz"], n_frequencies=case["target_points"],
                               n_averages=case["n_averages"], psll=200., workers=case["requested_threads"],
                               max_working_mb=working_mb, streaming="auto")
        if method == "matched":
            lengths = np.asarray(plan[3])
            preview = DiskLPSDSubset(n, **bounded_options, plan=plan,
                                     plan_mask=(lengths > n / 16) | (lengths < 32))
        else:
            preview = DiskLPSDSubset(n, **bounded_options)
        description = preview.metadata
        model = description["bounded_memory_model"]
        scratch_bytes = model["scratch_bytes"]
        streaming = description["streaming"]
        stream_bytes = model["streaming_bytes_upper_bound"]
        result.update(native_backend="bounded", native_working_mb=working_mb,
                      native_scratch_bytes=scratch_bytes, native_streaming_points=streaming["eligible_points"],
                      native_streaming_buffer_bytes=stream_bytes,
                      native_working_bytes_upper_bound=model["working_bytes_upper_bound"])
        del preview
        if not model["fits_working_budget"] or model["working_bytes_upper_bound"] > headroom:
            return {**result, "status": "skipped", "reason": "memory_budget"}
    _address_guard(case)
    started = time.perf_counter()
    tones = []
    values = None
    path = directory / "input.f64"
    if case["storage"] == "disk":
        from benchmarks.bench_fftw_disk import input_file
        from benchmarks._fftw_disk import DiskFFT
        if method != "lpsd":
            try:
                estimate = DiskFFT.memory_estimate(n, case["fft_memory_mb"])
            except (ValueError, MemoryError):
                return {**result, "status": "skipped", "reason": "unsupported"}
            disk_needed = 24 * n + (0 if estimate["workspace_power_supported"] else 4 * n + 8) + 16 * MIB
        else:
            disk_needed = 8 * n + 16 * MIB
        result["disk_required_bytes"] = disk_needed
        if disk_needed > case["disk_budget_bytes"]:
            return {**result, "status": "skipped", "reason": "disk_budget"}
        if case["signal"] == "white":
            input_file(path, n, case["sample_rate_hz"], case["seed"])
        else:
            generated, tones = signal(case["signal"], n, case["sample_rate_hz"], case["seed"])
            destination = np.memmap(path, dtype=np.float64, mode="w+", shape=(n,))
            destination[:] = generated
            del generated
            destination.flush()
            destination._mmap.close()
            del destination
        values = np.memmap(path, dtype=np.float64, mode="r", shape=(n,))
    elif case["signal"] == "white":
        # Match the historical fixed generator, but avoid an extra N-element
        # allocation for scaling. This matters for bounded native RAM at 1B.
        values = generate_white(n, case["sample_rate_hz"], case["seed"])
    else:
        values, tones = signal(case["signal"], n, case["sample_rate_hz"], case["seed"])
    try:
        fingerprint = bf.digest_array(values)
        result["input_setup_s"] = time.perf_counter() - started
        result["input_sha256"] = fingerprint
        common = dict(sample_rate=case["sample_rate_hz"], psll=200.,
                      n_frequencies=case["target_points"], n_averages=case["n_averages"])
        # The one-shot hybrid has no persistent cache and releases FFT buffers
        # before its bounded native subset. Admission checked the actual pool.
        in_place = case.get("fft_layout") == "in-place"
        window_storage = case.get("window_storage", "full")
        result.setdefault("native_working_mb", (case["low_working_mb"] if case["storage"] == "disk"
                                               else 0.))
        result.setdefault("native_backend", "not_applicable")
        result["cache_bytes"] = 0
        disk_details = None
        wall, cpu = time.perf_counter(), time.process_time()
        phases = {}
        window_metadata = None
        if case["storage"] == "disk":
            from benchmarks.bench_fftw_disk import disk_pipeline
            power_storage = ("workspace" if method == "lpsd" or estimate["workspace_power_supported"] else "external")
            options = SimpleNamespace(sample_rate=case["sample_rate_hz"],
                                      n_frequencies=case["target_points"], n_averages=case["n_averages"],
                                      fft_memory_mb=case["fft_memory_mb"], power_storage=power_storage,
                                      workers=case["requested_threads"],
                                      smoothing_workers=controls["smoothing_workers_limit"],
                                      low_working_mb=case["low_working_mb"],
                                      lpsd_progress_interval_s=60.)
            frame, disk_details = disk_pipeline(method, values, backend, directory, options)
            phases = disk_details.get("phases", {})
        elif method == "lpsd":
            # A fresh constructor keeps planning and workspace preparation
            # inside the complete-call timer despite the earlier preflight.
            estimator = DiskLPSDSubset(n, **bounded_options)
            frame = estimator.compute(values, workspace=None, outputs=("psd", "nsd"))
        elif method == "fftw":
            provider = bf.KaiserWindow("native", Path(api._LIB._name))
            backend.forget_wisdom()
            with bf.Periodogram(backend, n, **common, window_provider=provider,
                                operations="native_grouped", operations_library=api._LIB,
                                aggregation="log", output_dtype="float32", planner="estimate",
                                fft_in_place=in_place, window_storage=window_storage) as estimator:
                frame = estimator.compute(values, outputs=("psd", "nsd"))
                window_metadata = estimator.metadata
        else:
            provider = bf.KaiserWindow("native", Path(api._LIB._name))
            backend.forget_wisdom()
            frame = estimate_once(backend, values, outputs=("psd", "nsd"), **common,
                                  window_provider=provider, workers=case["requested_threads"],
                                  max_working_mb=bounded_options["max_working_mb"], total_cache_mb=0,
                                  max_kernel_cache_mb=0, window_cache_mb=0,
                                  smoothing_workers=controls["smoothing_workers_limit"],
                                  weight_dtype="float32", operations_backend="native",
                                  smoothing_backend="native", planner="estimate", fft_in_place=in_place,
                                  low_frequency_backend="bounded", window_storage=window_storage)
            window_metadata = frame.attrs["matched_smoothing_once"]
        result.update(compute_wall_s=time.perf_counter() - wall, compute_cpu_s=time.process_time() - cpu)
        result.update(_smoothing_result_fields(case, window_metadata, disk_details))
        result.update(_native_scratch_fields(method, case["storage"], frame.attrs, disk_details))
        if window_metadata is not None:
            for key in ("window_storage", "window_buffer_bytes", "window_scratch_bytes", "one_shot_window"):
                result[key] = window_metadata[key]
            details = window_metadata.get("window_storage_details")
            if details:
                result["window_block_samples"] = details["block_samples"]
        f = frame.index.to_numpy(dtype=np.float64)
        psd, nsd = frame.psd.to_numpy(), frame.nsd.to_numpy()
        expected_f = np.asarray(plan[0], dtype=np.float64)
        finite = bool(np.isfinite(psd).all() and np.isfinite(nsd).all())
        nonnegative = bool((psd >= 0).all())
        sqrt_matches = bool(np.array_equal(nsd, np.sqrt(psd.astype(np.complex64)).real))
        input_unchanged = bf.digest_array(values) == fingerprint
        if not (finite and nonnegative and sqrt_matches and input_unchanged and np.array_equal(f, expected_f)):
            return {**result, "status": "failed", "reason": "invalid_result"}
        source_psd = theory(case["signal"], f, case["sample_rate_hz"])
        excluded = None
        if source_psd is not None and tones:
            metric_window = bf.KaiserWindow("native", Path(api._LIB._name))
            excluded = noise_line_mask(f, plan[3], case["sample_rate_hz"], tones,
                                       metric_window, api._kaiser_alpha(200.) * np.pi)
        noise = (summarize_noise(f, nsd, source_psd, exclude=excluded)["overall"]
                 if source_psd is not None else {})
        tone = isolated_tone_metrics(case["signal"], f, psd, tones)
        result.update(status="completed", reason="", finite=finite, psd_nonnegative=nonnegative,
                      nsd_matches_sqrt_psd=sqrt_matches, input_unchanged=input_unchanged,
                      actual_points=len(f), frequency_sha256=bf.digest_array(f),
                      psd_sha256=bf.digest_array(psd), nsd_sha256=bf.digest_array(nsd),
                      minimum_frequency_hz=float(f[0]), maximum_frequency_hz=float(f[-1]),
                      sample_iterations=sum(int(length) * int(count) for length, count in zip(plan[3], plan[4])),
                      noise=noise, tone=tone, phases=phases,
                      peak_rss_bytes=bf.max_rss_bytes(),
                      _spectrum={"frequency_hz": f.tolist(), "psd": psd.tolist(), "nsd": nsd.tolist()})
        return result
    finally:
        if isinstance(values, np.memmap):
            values._mmap.close()
        values = None
        path.unlink(missing_ok=True)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        return 2
    case_path, result_path = map(Path, args)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    try:
        report = run(case, case_path.parent)
    except (ImportError, ModuleNotFoundError):
        report = {"status": "skipped", "reason": "dependency_unavailable"}
    except OSError as exc:
        report = {"status": "failed", "reason": "disk_limit" if exc.errno == errno.ENOSPC else "worker_failed"}
    except MemoryError:
        report = {"status": "failed", "reason": "memory_limit"}
    except Exception:
        # Exception text can contain paths, identities and environment values.
        # The transport exposes only this closed failure category.
        report = {"status": "failed", "reason": "worker_failed"}
    _write(result_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
