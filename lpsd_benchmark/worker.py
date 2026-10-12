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

from .privacy import integer, number
from .resources import MIB, native_working_budget, smoothing_controls


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


class _FFTWUnavailable(ImportError):
    def __init__(self, diagnostics):
        super().__init__("FFTW double-precision runtime unavailable")
        self.diagnostics = diagnostics


def _load_fftw(bf, requested_threads, explicit=None, explicit_threads=None):
    """Load genuine FFTW symbols, from system libraries or a pyFFTW wheel.

    Loading a wheel does not replace FFTW by NumPy's FFT. Some wheels bundle
    the threading symbols in the main DLL, others in a separate library.
    Keep failures as closed categories, never exception text or library paths.
    Search pthread and OpenMP runtimes, and keep wheel companions with their
    own core instead of combining independent system and wheel FFTW builds.
    """
    candidates = []
    if explicit:
        candidates.append((str(explicit), "explicit"))
    else:
        system = ctypes.util.find_library("fftw3")
        if system:
            candidates.append((system, "system"))
        try:
            distribution = metadata.distribution("pyfftw")
            for entry in distribution.files or ():
                name = Path(entry).name.lower()
                if (re.match(r"(?:lib)?fftw3[-._]", name)
                        and any(marker in name for marker in (".so", ".dylib", ".dll"))):
                    candidates.append((str(distribution.locate_file(entry)), "wheel"))
        except metadata.PackageNotFoundError:
            pass
    candidates = list(dict.fromkeys(candidates))
    def is_thread_library(name):
        base = Path(name).name.lower()
        return "threads" in base or "omp" in base

    # An explicit core path can have any filename, including "compiled".
    # Its exported double-precision symbols decide whether it is usable.
    primary = [(name, source) for name, source in candidates
               if source == "explicit" or not is_thread_library(name)]
    companions = [(name, source) for name, source in candidates if is_thread_library(name)]
    system_threads = []
    if requested_threads > 1:
        for runtime in ("fftw3_threads", "fftw3_omp"):
            found = ctypes.util.find_library(runtime)
            if found:
                system_threads.append((found, "system"))
    failures = {}
    diagnostics = {"requested": requested_threads, "actual": 0, "status": "unavailable",
                   "main_candidates": len(primary), "thread_attempts": 0, "failures": failures}
    fallback = None

    def fail(category):
        failures[category] = failures.get(category, 0) + 1

    def select(backend, source, thread_source, status, runtime="none"):
        backend.threading_diagnostics = {
            **diagnostics, "actual": backend.threads, "status": status,
            "main_source": source, "thread_source": thread_source, "thread_runtime": runtime,
            "failures": dict(failures)}
        return backend

    for name, source in primary:
        # Validate the core once so a later missing symbol or init failure can
        # be attributed to the threading attempt, without parsing exceptions.
        try:
            core = bf.FFTWLibrary(name, None, 1)
        except OSError:
            fail("main_load_failed")
            continue
        except AttributeError:
            fail("main_symbols_missing")
            continue
        except (RuntimeError, ValueError):
            fail("main_runtime_rejected")
            continue
        if requested_threads == 1:
            return select(core, source, "none", "single_thread_requested")
        if fallback is None:
            fallback = core, source
        thread_candidates = [(str(explicit_threads), "explicit")] if explicit_threads else []
        if source == "wheel":
            thread_candidates.extend((library, origin) for library, origin in companions
                                     if Path(library).parent == Path(name).parent)
        elif source == "system":
            thread_candidates.extend(system_threads)
        # Integrated threading symbols are common in Windows wheels. For an
        # explicit core, prefer its own runtime before system companions.
        thread_candidates.append((name, "main_library"))
        if source == "explicit":
            thread_candidates.extend(system_threads)
        for thread_library, thread_source in dict.fromkeys(thread_candidates):
            diagnostics["thread_attempts"] += 1
            try:
                backend = bf.FFTWLibrary(name, thread_library, requested_threads)
            except OSError:
                fail("thread_load_failed")
            except AttributeError:
                fail("thread_symbols_missing")
            except RuntimeError:
                fail("thread_init_failed")
            except ValueError:
                fail("thread_runtime_rejected")
            else:
                runtime = ("integrated" if thread_library == name else "openmp"
                           if "omp" in Path(thread_library).name.lower() else "pthreads"
                           if "threads" in Path(thread_library).name.lower() else "unspecified")
                return select(backend, source, thread_source, "enabled", runtime)
    if fallback is not None:
        return select(*fallback, "none", "fallback_single_thread")
    raise _FFTWUnavailable(diagnostics)


def _result_validation(bf, values, fingerprint, frequencies, psd, nsd, expected_f):
    """Retain every output check and bounded aggregate diagnostics on failure."""
    import numpy as np
    with np.errstate(all="ignore"):
        expected_nsd = np.sqrt(psd.astype(np.complex64)).real
        fields = {
            "finite": bool(np.isfinite(psd).all() and np.isfinite(nsd).all()),
            "psd_nonnegative": bool((psd >= 0).all()),
            "nsd_matches_sqrt_psd": bool(np.array_equal(nsd, expected_nsd)),
            "frequency_matches_plan": bool(np.array_equal(frequencies, expected_f)),
            "input_unchanged": bf.digest_array(values) == fingerprint,
        }
        details = {
            "expected_frequency_points": int(expected_f.size),
            "frequency_points": int(frequencies.size), "psd_points": int(psd.size),
            "nsd_points": int(nsd.size),
            "nonfinite_psd_points": int(np.count_nonzero(~np.isfinite(psd))),
            "nonfinite_nsd_points": int(np.count_nonzero(~np.isfinite(nsd))),
            "negative_psd_points": int(np.count_nonzero(psd < 0)),
        }
        for label, actual, expected in (("nsd_sqrt", nsd, expected_nsd),
                                        ("frequency", frequencies, expected_f)):
            if actual.shape != expected.shape:
                continue
            details[label + "_mismatch_points"] = int(np.count_nonzero(actual != expected))
            finite_pairs = np.isfinite(actual) & np.isfinite(expected)
            if not finite_pairs.any():
                continue
            difference = np.abs(actual[finite_pairs].astype(np.float64)
                                - expected[finite_pairs].astype(np.float64))
            maximum = number(float(np.max(difference)))
            if maximum is not None:
                details["maximum_" + label + "_absolute_error"] = maximum
            if label == "nsd_sqrt":
                nonzero = expected[finite_pairs] != 0
                if nonzero.any():
                    relative = difference[nonzero] / np.abs(expected[finite_pairs][nonzero])
                    maximum = number(float(np.max(relative)))
                    if maximum is not None:
                        details["maximum_nsd_sqrt_relative_error"] = maximum
    return {**fields, "validation": details}


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
    """Compatibility helper returning the allowance and remaining headroom."""
    working_mb, headroom, _ = _case_native_working_controls(case)
    return working_mb, headroom


def _case_native_working_controls(case):
    """Revalidate request and effective bytes before numerical imports."""
    _case_smoothing_controls(case)
    working_mb, headroom, fields = native_working_budget(case)
    for key, value in fields.items():
        if key == "requested_native_working_mib" and case.get("method") == "fftw":
            continue
        if key in case and (type(case[key]) is not type(value) or case[key] != value):
            raise ValueError("Inconsistent native working control or allowance")
    return working_mb, headroom, fields


def _native_result_source(method, storage, attrs, disk_details=None):
    """Locate completed native metadata for the four supported call routes."""
    if method not in ("lpsd", "matched"):
        return None
    if storage == "disk":
        source = (disk_details.get("lpsd" if method == "lpsd" else "low_lpsd")
                  if isinstance(disk_details, dict) else None)
    elif isinstance(attrs, dict):
        source = attrs.get("lpsd_disk" if method == "lpsd" else "matched_smoothing_once")
        if method == "matched":
            source = source.get("low_frequency_preparation") if isinstance(source, dict) else None
    else:
        source = None
    return source if isinstance(source, dict) else None


def _native_working_result_fields(case, attrs, disk_details=None):
    """Read the actual native capacity after timing, without inventing work."""
    if case["method"] not in ("lpsd", "matched"):
        return {}
    source = _native_result_source(case["method"], case["storage"], attrs, disk_details)
    if (source is not None and type(source.get("frequencies")) is int
            and source["frequencies"] == 0):
        return {}  # An empty native subset executed no frequency work.
    if source is None and case["method"] == "matched" and case["storage"] == "ram":
        metadata = attrs.get("matched_smoothing_once", {}) if isinstance(attrs, dict) else {}
        if type(metadata.get("exact_low_frequency_points")) is int and metadata["exact_low_frequency_points"] == 0:
            return {}
    _, _, controls = _case_native_working_controls(case)
    applied = integer(source.get("concurrency_budget_bytes"), 64 * MIB) if source else None
    if applied is None and controls["native_working_policy"] == "auto":
        return {}  # Older private adapters need not invent new provenance.
    if applied is None or applied != controls["native_working_limit_bytes"]:
        raise ValueError("Measured native working capacity differs from admission")
    return {"native_working_applied_bytes": applied}


def _native_scratch_fields(method, storage, attrs, disk_details=None):
    """Scalar pool capacities from completed-call metadata, never RSS.

    Read the returned metadata, not the constructor's admission preview.
    Missing lifecycle data remains absent for older adapters and FFTW-only
    calls. Planned or failed replacements do not publish retirement indices.
    """
    source = _native_result_source(method, storage, attrs, disk_details)
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


def _disk_fft_result_fields(case, details):
    """Read back the actual transform length, including odd-N fallback."""
    if case["storage"] != "disk" or case["method"] == "lpsd":
        return {}
    algorithm = case.get("disk_fft_algorithm", "full")
    description = details.get("fft", {}) if isinstance(details, dict) else {}
    if not description:
        # Existing automatic test doubles/legacy adapters have no readback.
        if algorithm == "full":
            return {}
        raise ValueError("Packed FFT execution metadata is missing")
    n = description.get("n")
    inner = description.get("inner_n", n)
    expected = case["n"] // 2 if algorithm == "packed-real" and case["n"] % 2 == 0 and case["n"] > 2 else case["n"]
    if type(n) is not int or n != case["n"] or type(inner) is not int or inner != expected:
        raise ValueError("Inconsistent original-N FFT execution metadata")
    return {"disk_fft_inner_samples": inner,
            "disk_fft_effective_algorithm": "packed-real" if inner < n else "full"}


def _case_disk_power_reduction(case):
    """Validate selection before imports; only simple disk FFTW applies it."""
    selection = case.get("disk_power_reduction", "materialized")
    if not isinstance(selection, str) or selection not in ("materialized", "direct-log", "not_applicable"):
        raise ValueError("Invalid disk power reduction")
    if case["storage"] != "disk" or case["method"] != "fftw":
        return "not_applicable"
    if selection == "not_applicable":
        raise ValueError("Simple disk FFTW requires a power reduction")
    return selection


def _direct_power_profile_supported(case):
    """Keep the first explicitly qualified full-disk profile unchanged."""
    return (type(case.get("n")) is int and 2 <= case["n"] <= 1_000_000_000
            and case.get("disk_fft_algorithm", "full") == "full"
            and case.get("sample_rate_hz") == 50.0
            and case.get("n_averages") == 100 and case.get("psll_db", 200.0) == 200.0
            and case.get("fft_memory_mb") == 128.0)


def _disk_power_result_fields(case, details):
    """Validate completed actual reduction after both timer-end readings."""
    selection = _case_disk_power_reduction(case)
    if selection == "not_applicable":
        return {"disk_power_reduction": selection}
    if not isinstance(details, dict) or "disk_power_reduction" not in details:
        if selection == "materialized":
            return {"disk_power_reduction": selection}  # Older private adapters.
        raise ValueError("Direct log-power execution metadata is missing")
    if details["disk_power_reduction"] != selection:
        raise ValueError("Measured disk power reduction differs from admission")
    fields = {"disk_power_reduction": selection}
    if selection == "materialized":
        if details.get("power_vector_created") is not True:
            raise ValueError("Materialized power metadata is inconsistent")
        return {**fields, "power_vector_created": True}
    source = details.get("direct_group_reduction", {})
    storage = details.get("power_storage", {})
    fft = details.get("fft", {})
    points = integer(case.get("actual_points"), 1, min(65536, case["n"] // 2))
    if not all(isinstance(value, dict) for value in (source, storage, fft)) or points is None:
        raise ValueError("Direct log-power execution metadata is invalid")
    expected = {"n": case["n"], "groups": points, "abi_version": 3,
                "energy_bins": case["n"] // 2 + 1, "dc_bins": 1,
                "nyquist_bins": int(case["n"] % 2 == 0),
                "array_state_bytes": 48 * points + 48, "additional_workers": 0}
    if (any(type(source.get(key)) is not int or source[key] != value for key, value in expected.items())
            or source.get("power_vector_created") is not False
            or source.get("matrix_written") is not False
            or source.get("ordinary_result_arrays") is not True
            or details.get("power_vector_created") is not False
            or storage.get("mode") != "direct_log_groups"
            or type(storage.get("extra_power_file_bytes")) is not int or storage["extra_power_file_bytes"] != 0
            or type(storage.get("maximum_large_file_bytes")) is not int
            or storage["maximum_large_file_bytes"] != 24 * case["n"]
            or storage.get("workspace_power_offset_bytes") is not None
            or fft.get("power_storage") != "direct_log_groups"
            or type(fft.get("n")) is not int or fft["n"] != case["n"]):
        raise ValueError("Direct log-power counts, ABI or storage metadata differ from execution")
    return {**fields, "direct_log_power_abi": source["abi_version"],
            "direct_log_power_state_bytes": source["array_state_bytes"],
            "power_vector_created": False}


def run(case, directory):
    requested_statistics = case.get("requested_statistics", case.get("statistics", "legacy"))
    if requested_statistics == "not_applicable" and case["method"] == "fftw":
        requested_statistics = "legacy"
    if requested_statistics not in ("corrected", "legacy"):
        raise ValueError("Invalid statistics mode")
    statistics = (case.get("statistics", requested_statistics)
                  if case["method"] != "fftw" else "not_applicable")
    if case["method"] != "fftw" and statistics not in ("corrected", "legacy"):
        raise ValueError("Invalid statistics mode")
    statistics_fields = {"statistics": statistics, "requested_statistics": requested_statistics}
    algorithm = case.get("disk_fft_algorithm", "full")
    if algorithm not in ("full", "packed-real", "not_applicable"):
        raise ValueError("Invalid disk FFT algorithm")
    needs_fft = case["storage"] == "disk" and case["method"] != "lpsd"
    if needs_fft and algorithm == "not_applicable":
        raise ValueError("Disk FFT cases require a transform algorithm")
    applied_algorithm = algorithm if needs_fft else "not_applicable"
    reduction = _case_disk_power_reduction(case)
    if reduction == "direct-log" and not _direct_power_profile_supported(case):
        return {**statistics_fields, "status": "skipped", "reason": "unsupported",
                "disk_fft_algorithm": applied_algorithm, "disk_power_reduction": reduction}
    controls = _case_smoothing_controls(case)
    case.update(controls)
    working_mb, headroom, native_controls = _case_native_working_controls(case)
    case.update(native_controls)
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
    result = {**resolved, **controls, **native_controls, **statistics_fields,
              "disk_fft_algorithm": applied_algorithm, "disk_power_reduction": reduction}
    if reduction == "direct-log":
        from benchmarks._fftw_direct_power import DirectLibrary, MAX_GROUPS
        if not 1 <= resolved["actual_points"] <= min(MAX_GROUPS, n // 2):
            return {**result, "status": "skipped", "reason": "unsupported"}
        try:
            DirectLibrary(api._LIB)
        except RuntimeError as error:
            raise ImportError("Direct log-power ABI3 is unavailable") from error
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
    if needs_fft and algorithm == "packed-real":
        source_paths.append(Path(bf.__file__).parent / "_fftw_packed_real.py")
    if reduction == "direct-log":
        source_paths.extend((Path(bf.__file__).parent / "_fftw_direct_power.py",
                             Path(api.__file__).parent / "_native" / "fftw_direct_log_power.c"))
    for path in source_paths:
        source.update(path.name.encode("ascii") + b"\0")
        source.update(path.read_bytes())
    result["source_sha256"] = source.hexdigest()
    backend = None
    if method != "lpsd":
        try:
            backend = _load_fftw(bf, case["requested_threads"], case.get("fftw_library"),
                                 case.get("threads_library"))
        except _FFTWUnavailable as error:
            return {**result, "status": "skipped", "reason": "dependency_unavailable",
                    "fftw_threading": error.diagnostics}
        if hasattr(backend, "threading_diagnostics"):
            result["fftw_threading"] = backend.threading_diagnostics
        result["fftw_sha256"] = backend.evidence.get("binary_sha256")
        found_version = re.search(r"fftw-([0-9]+(?:\.[0-9]+){1,3})", backend.version)
        if found_version:
            result["fftw_version"] = found_version.group(1)
        if backend.threads_evidence:
            result["fftw_threads_sha256"] = backend.threads_evidence.get("binary_sha256")
    result["actual_threads"] = backend.threads if backend else case["requested_threads"]
    bounded_options = None
    if method in ("lpsd", "matched"):
        if working_mb is None:
            return {**result, "status": "skipped", "reason": "memory_budget"}
        from benchmarks._lpsd_disk import DiskLPSDSubset
        bounded_options = dict(sample_rate=case["sample_rate_hz"], n_frequencies=case["target_points"],
                               n_averages=case["n_averages"], psll=200., workers=case["requested_threads"],
                               max_working_mb=working_mb, streaming="auto", statistics=statistics)
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
        if (not model["fits_working_budget"] or model["working_bytes_upper_bound"] > headroom
                or model["working_bytes_upper_bound"] > native_controls["native_working_limit_bytes"]):
            return {**result, "status": "skipped", "reason": "memory_budget"}
    _address_guard(case)
    started = time.perf_counter()
    tones = []
    values = None
    path = directory / "input.f64"
    if case["storage"] == "disk":
        from benchmarks.bench_fftw_disk import disk_fft_class, input_file
        if method != "lpsd":
            try:
                estimate = disk_fft_class(algorithm).memory_estimate(n, case["fft_memory_mb"])
            except (ValueError, MemoryError):
                return {**result, "status": "skipped", "reason": "unsupported"}
            if reduction == "direct-log" and not estimate["workspace_power_supported"]:
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
                                      disk_fft_algorithm=algorithm if needs_fft else "full",
                                      disk_power_reduction=reduction if method == "fftw" else "materialized",
                                      statistics=statistics,
                                      workers=case["requested_threads"],
                                      smoothing_workers=controls["smoothing_workers_limit"],
                                      low_working_mb=(working_mb if bounded_options is not None
                                                      else case["low_working_mb"]),
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
                                  low_frequency_backend="bounded", window_storage=window_storage,
                                  statistics=statistics)
            window_metadata = frame.attrs["matched_smoothing_once"]
        result.update(compute_wall_s=time.perf_counter() - wall, compute_cpu_s=time.process_time() - cpu)
        result.update(_smoothing_result_fields(case, window_metadata, disk_details))
        result.update(_native_working_result_fields(case, frame.attrs, disk_details))
        result.update(_disk_fft_result_fields(case, disk_details))
        result.update(_disk_power_result_fields(case, disk_details))
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
        checks = _result_validation(bf, values, fingerprint, f, psd, nsd, expected_f)
        result.update(checks)
        result.update(actual_points=len(f), frequency_sha256=bf.digest_array(f),
                      psd_sha256=bf.digest_array(psd), nsd_sha256=bf.digest_array(nsd),
                      phases=phases, peak_rss_bytes=bf.max_rss_bytes())
        if not all(checks[key] for key in ("finite", "psd_nonnegative", "nsd_matches_sqrt_psd",
                                           "input_unchanged", "frequency_matches_plan")):
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
        result.update(status="completed", reason="",
                      minimum_frequency_hz=float(f[0]), maximum_frequency_hz=float(f[-1]),
                      sample_iterations=sum(int(length) * int(count) for length, count in zip(plan[3], plan[4])),
                      noise=noise, tone=tone,
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
