"""Portable controller: bounded serial children, conservative admission, opt-in upload.

Run from a pinned source checkout with ``python -m lpsd_benchmark run``.
Numerical dependencies are loaded only after a case has passed admission.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import ctypes
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import tempfile
import time

from . import DEFAULT_SIZES, MAX_SECONDS, METHODS, SCHEMA, SIGNALS
from .privacy import MAX_CASES, MAX_LOCAL_BYTES, sanitize, submit
from .resources import (Deadline, MIB, calibration_matches, choose_fft_layout, choose_window_storage,
                        detect_resources, fftw_plan_reserve, native_working_budget,
                        native_working_request, predict_seconds, select_storage, smoothing_controls)

ROOT = Path(__file__).resolve().parent.parent
_MEMORY_MEASUREMENT_GRACE_S = 1.0


def _arch(value):
    return {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64",
            "aarch64": "aarch64", "armv7l": "armv7", "armv7": "armv7",
            "i386": "x86", "i686": "x86", "x86": "x86"}.get(str(value).lower(), "other")


def _windows_architecture(process_arch, host_arch):
    """Windows ARM x64 emulation needs IsWow64Process2, not just machine()."""
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        function = kernel.IsWow64Process2
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]
        function.restype = ctypes.c_int
        process, native = ctypes.c_ushort(), ctypes.c_ushort()
        if function(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)):
            names = {0x8664: "x86_64", 0xaa64: "aarch64", 0x14c: "x86", 0x1c4: "armv7"}
            host_arch = names.get(native.value, "other")
            process_arch = names.get(process.value, host_arch) if process.value else host_arch
    except (OSError, AttributeError):
        pass
    return process_arch, host_arch


def environment(resources):
    system = platform.system()
    if sys.platform == "android" or (system == "Linux" and hasattr(sys, "getandroidapilevel")):
        system = "Android"
    if system not in ("Linux", "Windows", "Darwin", "Android"):
        system = "Other"
    process_arch = _arch(platform.machine())
    host_arch = process_arch
    if os.name == "nt":
        # Windows reports process architecture in an emulated Python. Only
        # exact architecture tokens from the OS are allowed into the report.
        host_arch = _arch(os.environ.get("PROCESSOR_ARCHITEW6432", os.environ.get("PROCESSOR_ARCHITECTURE", platform.machine())))
        process_arch = _arch(os.environ.get("PROCESSOR_ARCHITECTURE", platform.machine()))
        process_arch, host_arch = _windows_architecture(process_arch, host_arch)
    elif system == "Darwin" and process_arch == "x86_64":
        try:
            translated = subprocess.run(["sysctl", "-in", "sysctl.proc_translated"],
                                        capture_output=True, text=True, timeout=2, check=False)
            if translated.returncode == 0 and translated.stdout.strip() == "1":
                host_arch = "aarch64"
        except (OSError, subprocess.SubprocessError):
            pass
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "pyfftw", "lpsd"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"os": system, "process_arch": process_arch, "host_arch": host_arch,
            "logical_cpus_available": resources.threads, "cpu_quota": resources.cpu_capacity,
            "ram_total_mib": resources.total_ram // MIB, "versions": versions}


def _source_commit():
    supplied = os.environ.get("LPSD_BENCHMARK_SOURCE_COMMIT")
    if supplied is not None:
        return supplied if supplied != "local" else None
    try:
        response = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, timeout=2, check=True)
        return response.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        # Archive bootstraps do not need a Git executable or a .git directory.
        return None


def _source_dirty():
    supplied = os.environ.get("LPSD_BENCHMARK_SOURCE_COMMIT")
    if supplied is not None:
        return supplied == "local"
    try:
        result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                                cwd=ROOT, capture_output=True, text=True, timeout=2, check=True)
        return bool(result.stdout)
    except (OSError, subprocess.SubprocessError):
        return None


def write_report(path, report, save_spectra=False):
    clean = sanitize(report)
    submission_path = path
    if save_spectra:
        # Serialize the allowlist before adding local-only spectra. Keep this
        # compact checkpoint within the existing submit reader's size limit.
        submission_path = path.with_suffix(".public.json")
        public_bytes = (json.dumps(clean, ensure_ascii=True, allow_nan=False,
                                   separators=(",", ":")) + "\n").encode("ascii")
        if len(public_bytes) > MAX_LOCAL_BYTES:
            raise ValueError("Public benchmark report exceeds the local input size limit")
        public_temporary = submission_path.with_suffix(submission_path.suffix + ".tmp")
        public_temporary.write_bytes(public_bytes)
        public_temporary.replace(submission_path)
        # Only internally generated synthetic spectra enter this local file.
        # sanitize() deliberately never admits this field to an upload.
        clean["spectra"] = report.get("spectra", [])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(clean, stream, allow_nan=False, indent=2)
        stream.write("\n")
    temporary.replace(path)
    return submission_path


def terminate_tree(process):
    """End the owned process group, including compiler/BLAS helper children."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=3, check=False)
        except (OSError, subprocess.SubprocessError):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except (PermissionError, OSError):
            process.kill()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        # No portable user-space API can make an uninterruptible kernel I/O
        # request return. Do not turn cleanup into an unbounded wait.
        pass


def _linux_status(path):
    """Read only numeric process relationships and memory counters."""
    try:
        fields = {}
        for line in path.read_text().splitlines():
            items = line.split()
            if len(items) < 2:
                continue
            if items[0] in ("Pid:", "PPid:"):
                fields[items[0][:-1]] = int(items[1])
            elif items[0] == "NSpid:":
                fields["NSpid"] = tuple(map(int, items[1:]))
            elif items[0] in ("VmRSS:", "RssAnon:"):
                fields[items[0][:-1]] = int(items[1]) * 1024
        return fields
    except (OSError, ValueError):
        return {}


def _linux_children(proc_root, visible_pid):
    """Children are expressed in the PID namespace of the procfs mount."""
    found = set()
    try:
        for task in (proc_root / str(visible_pid) / "task").iterdir():
            try:
                found.update(int(value) for value in (task / "children").read_text().split())
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return found


class _LinuxMemoryReader:
    """Resolve a namespace PID through its controller, then read its tree.

    Containers can expose a procfs mount from an ancestor PID namespace.
    Popen.pid must therefore not be used as a /proc directory name without
    checking NSpid and PPid. Matching only the innermost NSpid is ambiguous
    across unrelated namespace trees.
    """
    def __init__(self, pid, proc_root=Path("/proc")):
        self.pid = pid
        self.proc_root = proc_root
        parent = _linux_status(proc_root / "self" / "status")
        self.parent_pid = parent.get("Pid")
        self.depth = len(parent.get("NSpid", ()))
        self.same_namespace = self.parent_pid == os.getpid()
        self.visible_pid = None

    def _resolve(self):
        if self.parent_pid is None:
            return
        candidates = _linux_children(self.proc_root, self.parent_pid)
        if not candidates:
            # Older kernels may not expose task/children. Restrict fallback
            # candidates to a verified direct child of this controller.
            try:
                candidates = {int(path.name) for path in self.proc_root.iterdir() if path.name.isdecimal()}
            except OSError:
                return
        for candidate in candidates:
            fields = _linux_status(self.proc_root / str(candidate) / "status")
            if fields.get("PPid") != self.parent_pid:
                continue
            nested = fields.get("NSpid", ())
            if ((self.depth and len(nested) >= self.depth and nested[self.depth - 1] == self.pid)
                    or (not self.depth and self.same_namespace and candidate == self.pid)):
                self.visible_pid = candidate
                return

    def read(self):
        if self.visible_pid is None:
            self._resolve()
        if self.visible_pid is None:
            return 0, 0
        pending = [(self.visible_pid, self.parent_pid)]
        visited = set()
        rss, anonymous = 0, 0
        while pending:
            pid, parent = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            fields = _linux_status(self.proc_root / str(pid) / "status")
            if fields.get("PPid") != parent:
                continue
            resident = fields.get("VmRSS", 0)
            rss += resident
            # Old kernels without RssAnon use a conservative RSS fallback.
            anonymous += fields.get("RssAnon", resident)
            pending.extend((child, pid) for child in _linux_children(self.proc_root, pid))
        return rss, anonymous


def _memory_usage(pid):
    """Return RSS and anonymous bytes, or None for an unavailable private counter."""
    if sys.platform.startswith("linux"):
        return _LinuxMemoryReader(pid).read()
    try:
        import psutil
        owner = psutil.Process(pid)
        processes = [owner] + owner.children(recursive=True)
        rss, private, private_available = 0, 0, True
        for proc in processes:
            if not proc.is_running():
                continue
            info = proc.memory_info()
            resident = info.rss
            if type(resident) is not int or resident < 0:
                return 0, 0
            rss += resident
            if sys.platform == "darwin":
                # Older psutil USS queries need a privileged task port and can
                # hang on headless macOS. Do not issue that optional query in
                # the synchronous watchdog; preserve the complete RSS instead.
                value = None
            elif os.name == "nt":
                value = getattr(info, "private", None)
                if value is None:
                    value = getattr(info, "pagefile", None)
            else:
                value = getattr(proc.memory_full_info(), "uss", None)
            if type(value) is int and value >= 0:
                private += value
            else:
                # An absent counter is different from a measured zero. Do
                # not turn partial tree coverage into an anonymous reading.
                private_available = False
        return rss, private if private_available else None
    except (ImportError, OSError, ProcessLookupError):
        return 0, 0
    except Exception:
        # psutil's platform-specific NoSuchProcess/AccessDenied exceptions do
        # not all inherit OSError. A disappearing child is not a failure.
        return 0, 0


def _owned_case_file_bytes(directory):
    """Logical lengths of only this private case's known data filenames.

    FFT calls own an immediate fft-work-* child. No arbitrary descendants,
    symlink targets, file contents or mapped-page/RSS estimates are read.
    Disappearance during normal cleanup is harmless; other errors propagate
    so execute_case still terminates and reaps its owned process.
    """
    def checked(path, *, folder=False):
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        expected = stat.S_ISDIR if folder else stat.S_ISREG
        if stat.S_ISLNK(info.st_mode) or reparse or not expected(info.st_mode):
            raise ValueError("Unexpected owned case file layout")
        return info

    total = 0
    for name in ("input.f64", "workspace.c128", "powers.f64"):
        info = checked(directory / name)
        if info is not None:
            total += info.st_size
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.name.startswith("fft-work-"):
                    continue
                child = directory / entry.name
                if checked(child, folder=True) is None:
                    continue
                for name in ("workspace.c128", "powers.f64"):
                    info = checked(child / name)
                    if info is not None:
                        total += info.st_size
    except FileNotFoundError:
        pass
    return total


def execute_case(case, directory, deadline, timeout, *, command=None, interval=.1):
    """Whole-process timeout includes imports, input setup and final checks."""
    case_path, result_path = directory / "case.json", directory / "result.json"
    case_path.write_text(json.dumps(case, allow_nan=False), encoding="utf-8")
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    env["MALLOC_ARENA_MAX"] = "2"
    kwargs = {"cwd": ROOT, "env": env, "stdin": subprocess.DEVNULL,
              "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    arguments = command or [sys.executable, "-m", "lpsd_benchmark.worker", str(case_path), str(result_path)]
    wall = time.monotonic()
    case_end = min(deadline.end, wall + timeout)
    next_heartbeat = wall + 60.
    process = subprocess.Popen(arguments, **kwargs)
    memory_reader = (_LinuxMemoryReader(process.pid).read if sys.platform.startswith("linux")
                     else lambda: _memory_usage(process.pid))
    status, reason, peak_rss = None, "", 0
    live_samples, missing_samples, rss_observed = 0, 0, False
    rss_fallback_samples = 0
    missing_since = wall
    try:
        while process.poll() is None:
            now = time.monotonic()
            if now >= case_end:
                status, reason = "timeout", "time_limit"
                terminate_tree(process)
                break
            rss, anonymous = memory_reader()
            sampled_at = time.monotonic()
            rss_available = type(rss) is int and rss >= 0
            anonymous_available = type(anonymous) is int and anonymous >= 0
            if rss_available:
                peak_rss = max(peak_rss, rss)
                rss_observed = rss_observed or rss > 0
            rss_fallback = (case["storage"] == "disk" and sys.platform == "darwin"
                            and rss_available and rss > 0 and anonymous is None)
            if case["storage"] == "disk":
                available = ((rss_available and anonymous_available and (rss > 0 or anonymous > 0))
                             or rss_fallback)
            else:
                available = rss_available and rss > 0
            if available:
                live_samples += 1
                missing_since = None
                rss_fallback_samples += int(rss_fallback)
                # On Linux disk paths, use anonymous resident bytes: the same
                # file pages can be counted twice by overlapping mmap views.
                # Only macOS may substitute known RSS when USS is unavailable.
                # RSS includes shared/file pages, so this is a stricter bound.
                measured = anonymous if case["storage"] == "disk" and not rss_fallback else rss
                if measured > case["ram_budget_bytes"]:
                    status, reason = "failed", "memory_limit"
                    terminate_tree(process)
                    break
            else:
                missing_samples += 1
                if missing_since is None:
                    missing_since = sampled_at
                if sampled_at - missing_since >= _MEMORY_MEASUREMENT_GRACE_S:
                    # A child may exit while its final memory read disappears.
                    # Reap that exit normally instead of inventing a live loss.
                    if process.poll() is not None:
                        break
                    status, reason = "failed", "memory_measurement_unavailable"
                    terminate_tree(process)
                    break
            # Only known owned file names are examined, not a recursive scan
            # of the user's drive. Logical lengths also bound sparse files.
            logical_bytes = _owned_case_file_bytes(directory)
            if logical_bytes > case["disk_budget_bytes"]:
                status, reason = "failed", "disk_limit"
                terminate_tree(process)
                break
            if now >= next_heartbeat:
                next_heartbeat = now + 60.
                try:
                    print(f"[running] N={case.get('n', '?')} {case.get('method', '?')} "
                          f"requested J={case.get('target_points', '?')}, "
                          f"point factor={case.get('point_factor') or 'explicit'}; "
                          f"process elapsed {now - wall:.0f} s; limit {case_end - wall:.0f} s "
                          "(includes setup/checks; not a completed timing)", flush=True)
                except (OSError, ValueError):
                    pass  # An unavailable output stream must not stop the child.
            sleep_at = time.monotonic()
            pause = min(interval, max(0., case_end - sleep_at))
            if missing_since is not None:
                pause = min(pause, max(0., missing_since + _MEMORY_MEASUREMENT_GRACE_S - sleep_at))
            time.sleep(pause)
    except BaseException:
        terminate_tree(process)
        raise
    elapsed = time.monotonic() - wall
    if status is None:
        if process.returncode == 0 and result_path.is_file() and result_path.stat().st_size <= MAX_LOCAL_BYTES:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(result, dict):
                    raise ValueError("Invalid worker result")
            except (ValueError, OSError):
                result = {"status": "failed", "reason": "invalid_result"}
        else:
            result = {"status": "failed", "reason": "worker_failed"}
    else:
        result = {"status": status, "reason": reason}
    high_water = result.get("peak_rss_bytes")
    if type(high_water) is int and high_water >= 0:
        peak_rss = max(peak_rss, high_water)
        result["rss_measurement"] = "process_high_water_mark"
    else:
        result["rss_measurement"] = "sampled" if rss_observed else "unavailable"
    result.update(process_wall_s=elapsed, peak_rss_bytes=peak_rss, time_limit_s=timeout,
                  live_memory_observed=live_samples > 0, live_memory_samples=live_samples,
                  live_memory_missing_samples=missing_samples,
                  live_memory_rss_fallback_samples=rss_fallback_samples)
    return result


def _matrix(args):
    sizes = sorted(set(args.sizes))
    factors = args.point_factors if args.points is None else [None] * len(args.points)
    points = [args.base_points] * len(factors) if args.points is None else args.points
    specifications = list(zip(points, factors))
    modes = ("ram", "disk") if args.storage == "both" else (args.storage,)
    matrix = []
    quality_sizes = set(args.quality_sizes) & set(sizes) if args.profile == "full" else set()
    # Explicit large-only requests still need measured calibration evidence.
    calibration_signals = ["white"] if sizes[0] > 100_000 else []
    if quality_sizes and min(quality_sizes) > 100_000:
        calibration_signals.extend(SIGNALS[1:])
    if calibration_signals:
        calibration_n = min(100_000, sizes[0])
        for name in calibration_signals:
            for method in args.methods:
                for storage in modes:
                    matrix.append(dict(n=calibration_n, target_points=args.base_points, point_factor=1.,
                                       method=method, signal=name, requested_storage=storage,
                                       case_role="calibration"))
    for n in sizes:
        signals = list(SIGNALS) if n in quality_sizes else ["white"]
        for signal_name in signals:
            for points_target, factor in specifications:
                for method in args.methods:
                    for storage in modes:
                        matrix.append(dict(n=n, target_points=points_target, point_factor=factor,
                                           method=method, signal=signal_name, requested_storage=storage,
                                           case_role="comparison"))
    return matrix


def _compare_group(report, spectra, row, spectrum):
    """Small Python-only comparison; no NumPy dependency in the parent."""
    key = (row["n"], row.get("point_factor"), row["target_points"], row["signal"], row["storage"])
    group = spectra.setdefault(key, {})
    group[row["method"]] = (row, spectrum)
    if "lpsd" not in group:
        return
    _, reference = group["lpsd"]
    for method, (current, data) in group.items():
        if method == "lpsd" or data["frequency_hz"] != reference["frequency_hz"]:
            continue
        differences = [(a / r - 1) for f, a, r in zip(data["frequency_hz"], data["nsd"], reference["nsd"])
                       if .01 <= f <= 20 and r > 0 and math.isfinite(a / r)]
        if differences:
            current["comparison"] = {
                "points": len(differences),
                "rms_relative_nsd_difference": math.sqrt(sum(x * x for x in differences) / len(differences)),
                "maximum_relative_nsd_difference": max(abs(x) for x in differences),
            }


def run(args):
    native_request = native_working_request(getattr(args, "native_working_mib", None))
    disk_algorithm = getattr(args, "disk_fft_algorithm", "full")
    if disk_algorithm not in ("full", "packed-real"):
        raise ValueError("disk-fft-algorithm must be full or packed-real")
    disk_reduction = getattr(args, "disk_power_reduction", "materialized")
    if disk_reduction not in ("materialized", "direct-log"):
        raise ValueError("disk-power-reduction must be materialized or direct-log")
    deadline = Deadline(args.max_hours * 3600, os.environ.get("LPSD_BENCHMARK_DEADLINE_EPOCH"))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resources = detect_resources(args.work_dir, args.ram_gib, args.disk_gib, args.threads)
    smoothing_request = getattr(args, "smoothing_workers", None)
    smoothing = smoothing_controls("matched", resources.threads, smoothing_request)
    matrix = _matrix(args)
    if len(matrix) > MAX_CASES:
        raise ValueError(f"Matrix exceeds {MAX_CASES} cases; use fewer sizes, point choices or storage modes")
    report = {"schema": SCHEMA, "state": "running", "source_commit": _source_commit(), "source_dirty": _source_dirty(),
              "environment": environment(resources), "cases": [], "elapsed_seconds": deadline.elapsed(),
              "settings": {"max_seconds": args.max_hours * 3600,
                           "bootstrap_remaining_seconds": deadline.seconds,
                           "ram_budget_mib": resources.ram_budget // MIB,
                           "disk_budget_mib": resources.disk_budget // MIB,
                           "threads": resources.threads, "sample_rate_hz": 50., "seed": 20261010,
                           "smoothing_worker_policy": smoothing["smoothing_worker_policy"],
                           "requested_smoothing_workers": smoothing_request,
                           "smoothing_runtime_reserve_model": smoothing["smoothing_runtime_reserve_model"],
                           "requested_native_working_mib": native_request,
                           "native_working_policy": "auto" if native_request is None else "explicit_cap",
                           "n_averages": 100, "psll_db": 200., "prediction_safety_factor": 4.,
                           "storage": args.storage, "fft_layout": args.fft_layout,
                           "window_storage": args.window_storage,
                           "disk_fft_algorithm": disk_algorithm,
                           "disk_power_reduction": disk_reduction,
                           "sizes": args.sizes, "points": args.points or [],
                           "quality_sizes": sorted(set(args.quality_sizes) & set(args.sizes)) if args.profile == "full" else [],
                           "base_points": args.base_points,
                           "point_factors": args.point_factors if args.points is None else []}}
    spectra = {}
    print(f"Budget: RAM {resources.ram_budget / (1024 ** 3):.2f} GiB, "
          f"disk {resources.disk_budget / (1024 ** 3):.2f} GiB, "
          f"threads {resources.threads}, remaining {deadline.remaining() / 3600:.2f} h.", flush=True)
    if smoothing_request is not None and "matched" in args.methods:
        print(f"Matched smoothing: requested {smoothing_request}, admitted limit "
              f"{smoothing['smoothing_workers_limit']}; extra runtime/address reserve "
              f"{smoothing['smoothing_runtime_reserve_bytes'] / MIB:.0f} MiB within the RAM budget.", flush=True)
    if native_request is not None and any(method in ("lpsd", "matched") for method in args.methods):
        print(f"Native working cap: requested {native_request} MiB; each case may admit less "
              "within its existing RAM allowance. Default FFT buffers and resource limits are retained.", flush=True)
    print("Only synthetic signals are used. Results stay local unless public GitHub upload was explicitly selected.", flush=True)
    if args.upload_github:
        print("Public upload selected: the issue will be associated with your GitHub account.", flush=True)
    write_report(args.output, report, args.save_spectra)
    reserve = min(10., max(.1, deadline.seconds * .01))
    unavailable = set()
    try:
        for position, specification in enumerate(matrix):
            row = {key: value for key, value in specification.items() if key != "requested_storage"}
            row.update(base_target_points=args.base_points, requested_threads=resources.threads,
                       point_count_resolved=False)
            row.update(smoothing_controls(row["method"], resources.threads, smoothing_request))
            smoothing_reserve = row["smoothing_runtime_reserve_bytes"]
            refreshed = detect_resources(args.work_dir, args.ram_gib, args.disk_gib, args.threads)
            # A run never grows its original budget just because another
            # process exits. Reduced free resources are honoured immediately.
            refreshed = type(resources)(resources.total_ram, refreshed.available_ram, refreshed.disk_free,
                                        min(resources.ram_budget, refreshed.ram_budget),
                                        min(resources.disk_budget, refreshed.disk_budget),
                                        resources.threads, resources.cpu_capacity)
            fft_layout = choose_fft_layout(row["n"], row["method"], row["signal"], args.fft_layout,
                                           refreshed.ram_budget, args.window_storage,
                                           smoothing_runtime_reserve_bytes=smoothing_reserve)
            window_storage = choose_window_storage(row["n"], row["method"], row["signal"],
                                                    args.window_storage, fft_layout, refreshed.ram_budget,
                                                    smoothing_runtime_reserve_bytes=smoothing_reserve)
            storage, ram_need, disk_need, skip = select_storage(
                row["n"], row["method"], row["signal"], specification["requested_storage"],
                refreshed, fft_layout, window_storage,
                smoothing_runtime_reserve_bytes=smoothing_reserve)
            working_ram = max(0, refreshed.ram_budget - smoothing_reserve)
            row.update(storage=storage, ram_required_bytes=ram_need, disk_required_bytes=disk_need,
                       disk_fft_algorithm=(disk_algorithm if storage == "disk" and row["method"] != "lpsd"
                                           else "not_applicable"),
                       disk_power_reduction=(disk_reduction if storage == "disk" and row["method"] == "fftw"
                                             else "not_applicable"),
                       fft_layout="blocked" if storage == "disk" and row["method"] != "lpsd" else fft_layout,
                       window_storage="blocked" if storage == "disk" and row["method"] != "lpsd" else window_storage,
                       fftw_plan_reserve_bytes=fftw_plan_reserve(row["n"], row["method"]) if storage == "ram" else 0,
                       ram_budget_mib=refreshed.ram_budget // MIB, disk_budget_mib=refreshed.disk_budget // MIB,
                       fft_memory_mb=max(1., min(128., working_ram / MIB / 8)),
                       low_working_mb=max(1., min(2048., working_ram / MIB / 3)))
            allowance, _, native_controls = native_working_budget({
                **row, "ram_budget_bytes": refreshed.ram_budget,
                "requested_native_working_mib": native_request})
            row.update(native_controls)
            if row["method"] in ("lpsd", "matched") and allowance is None and not skip:
                skip = "memory_budget"
            if (row["disk_power_reduction"] == "direct-log" and not skip
                    and (disk_algorithm != "full" or row["n"] > 1_000_000_000
                         or row["fft_memory_mb"] != 128.0)):
                skip = "unsupported"
            prediction = predict_seconds(row, report["cases"])
            calibration = (row["n"] <= 100_000 and not any(
                calibration_matches(row, old) for old in report["cases"]))
            if prediction is not None and not calibration:
                row["predicted_process_s"] = prediction
            if calibration:
                # The first small case is a strictly time-bounded probe, not
                # a forecast. It may use a shorter window in a quick run.
                prediction = min(180., max(0., deadline.remaining() - reserve))
            dependency_key = ((row["method"], "direct-log")
                              if row["disk_power_reduction"] == "direct-log" else row["method"])
            if dependency_key in unavailable:
                skip = "dependency_unavailable"
            if deadline.remaining() <= reserve:
                skip = "time_budget"
            elif not skip and prediction is None:
                skip = "uncalibrated"
            elif not skip and prediction > deadline.remaining() - reserve and not calibration:
                skip = "time_budget"
            if skip:
                row.update(status="skipped", reason=skip)
            else:
                timeout = min(prediction, deadline.remaining() - reserve)
                case = {**row, "ram_budget_bytes": refreshed.ram_budget,
                        "disk_budget_bytes": refreshed.disk_budget,
                        "sample_rate_hz": 50., "seed": 20261010, "n_averages": 100,
                        "fftw_library": str(args.fftw_library) if args.fftw_library else None,
                        "threads_library": str(args.threads_library) if args.threads_library else None}
                with tempfile.TemporaryDirectory(prefix="lpsd-case-", dir=args.work_dir) as name:
                    result = execute_case(case, Path(name), deadline, timeout)
                spectrum = result.pop("_spectrum", None)
                row.update(result)
                if row.get("reason") == "dependency_unavailable":
                    unavailable.add(dependency_key)
                if row["status"] == "completed" and spectrum:
                    _compare_group(report, spectra, row, spectrum)
                    if args.save_spectra:
                        report.setdefault("spectra", []).append({"n": row["n"], "method": row["method"],
                                                                  "signal": row["signal"], "storage": storage,
                                                                  "disk_power_reduction": row["disk_power_reduction"],
                                                                  "target_points": row["target_points"], **spectrum})
            report["cases"].append(row)
            report["elapsed_seconds"] = deadline.elapsed()
            write_report(args.output, report, args.save_spectra)
            point_label = f"J={row['target_points']} / {row['actual_points']} output points" if row.get("point_count_resolved") else f"point factor {row.get('point_factor') or 'explicit'}"
            detail = f"{row['compute_wall_s']:.6f} s" if row["status"] == "completed" else row.get("reason", "")
            if row["native_working_policy"] == "explicit_cap":
                detail += f"; native cap {row['native_working_limit_bytes'] / MIB:g} MiB"
            print(f"[{position + 1}/{len(matrix)}] N={row['n']} {row['method']} {storage} "
                  f"{row['signal']} {point_label}: {row['status']} {detail}", flush=True)
        report["state"] = "deadline" if deadline.remaining() <= reserve else "completed"
    except KeyboardInterrupt:
        report["state"] = "cancelled"
        print("Cancelled; completed results are preserved and owned child processes were stopped.", flush=True)
    finally:
        report["elapsed_seconds"] = deadline.elapsed()
        submission_path = write_report(args.output, report, args.save_spectra)
    print(f"Results: {args.output}", flush=True)
    if submission_path != args.output:
        print(f"Public report for optional submission: {submission_path}", flush=True)
    if args.upload_github:
        result = submit(submission_path, accepted=args.accept_public_upload, timeout=deadline.remaining())
        _print_submission(result)
    return 0


def _print_multipart_submission(result):
    confirmed, total = result["confirmed_parts"], result["total_parts"]
    print(f"Public report transfer: {confirmed}/{total} parts have confirmed receipts.", flush=True)
    if result.get("url"):
        print(f"GitHub issue: {result['url']}", flush=True)
    if result["uncertain_part"] is not None:
        print(f"Part {result['uncertain_part']} has no verified receipt and may already be public. "
              "Inspect the issue and its comments before posting any remaining part.", flush=True)
    elif confirmed < total:
        print(f"Parts {confirmed + 1} through {total} were not sent by this transfer "
              f"({result['reason']}).", flush=True)
    if result["reason"] == "existing_transfer":
        print("The saved transfer receipt was reused; no new upload or continuation was attempted.", flush=True)
    if result["reason"] == "local_receipt_unavailable":
        print("The local transfer receipt could not be updated. Review the issue before submitting again.", flush=True)
    if confirmed < total:
        print("No automatic retry or continuation is performed. Use the prepared files for manual completion "
              "of this same issue after reviewing confirmed and unconfirmed parts.", flush=True)
    print("Prepared public parts, in order:", flush=True)
    for index, path in enumerate(result["body_paths"], 1):
        state = ("confirmed" if index <= confirmed else "unconfirmed; inspect GitHub"
                 if index == result["uncertain_part"] else "not attempted")
        print(f"  Part {index}/{total}: {path} ({state})", flush=True)
    if result["receipt_path"].exists():
        print(f"Local transfer receipt: {result['receipt_path']}", flush=True)
    if result["reason"] in ("github_cli_unavailable", "github_login_unavailable"):
        print("Install GitHub CLI if needed and authenticate with "
              "`gh auth login --hostname github.com` before explicitly submitting.", flush=True)


def _print_submission(result):
    if result.get("total_parts", 1) > 1:
        _print_multipart_submission(result)
        return
    if result["status"] == "submitted":
        print(f"Public GitHub issue: {result['url']}", flush=True)
    elif result["status"] == "uncertain":
        print("GitHub submission was attempted, but no verified issue receipt was received. "
              "Check https://github.com/MichaDit/lpsd/issues before retrying. "
              f"Prepared body: {result['body_path']}", flush=True)
    else:
        print(f"No upload ({result['reason']}). Prepared public issue body: {result['body_path']}", flush=True)
        if result["reason"] in ("github_cli_unavailable", "github_login_unavailable"):
            print("Install GitHub CLI if needed, run `gh auth login --hostname github.com`, "
                  "then repeat `python -m lpsd_benchmark submit RESULTS.json --accept-public-upload`.", flush=True)


def _positive_smoothing_workers(value):
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("smoothing-workers must be a positive integer") from error
    if not 1 <= count <= 2 ** 63 - 1:
        raise argparse.ArgumentTypeError("smoothing-workers must be a positive integer below 2**63")
    return count


def _native_working_mib(value):
    try:
        return native_working_request(int(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run serial synthetic benchmarks within resource and time budgets")
    run_parser.add_argument("--max-hours", type=float, default=24., help="Setup-inclusive maximum; never more than 24 hours")
    run_parser.add_argument("--ram-gib", default="auto", help="Allocation budget in GiB, bounded by available RAM")
    run_parser.add_argument("--disk-gib", default="auto", help="Temporary dataset-file budget in GiB, bounded by free disk")
    run_parser.add_argument("--threads", type=int, help="Maximum threads, capped by affinity and CPU quota")
    run_parser.add_argument("--smoothing-workers", type=_positive_smoothing_workers,
                            help="Matched FFTW smoothing worker limit, capped by admitted threads; default min(4, threads)")
    run_parser.add_argument("--native-working-mib", type=_native_working_mib,
                            help="Optional native LPSD/low-frequency working cap in MiB (integer >=64); only lowers existing budgets")
    run_parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    points = run_parser.add_mutually_exclusive_group()
    points.add_argument("--points", nargs="+", type=int, help="Raw native planner J targets, e.g. 1000 2000")
    points.add_argument("--point-factors", nargs="+", type=float, default=[1., 2.],
                        help="Desired output-count factors relative to --base-points; default 1 2")
    run_parser.add_argument("--base-points", type=int, default=1000)
    run_parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    run_parser.add_argument("--storage", choices=("auto", "ram", "disk", "both"), default="auto")
    run_parser.add_argument("--disk-fft-algorithm", choices=("full", "packed-real"), default="full",
                            help="Disk FFT: full complex transform or original-N real packing; odd N uses full fallback")
    run_parser.add_argument("--disk-power-reduction", choices=("materialized", "direct-log"), default="materialized",
                            help="Explicit simple/full disk FFTW reduction; direct-log requires the original "
                                 "fs50/K100/Kaiser200 profile, 128 MiB FFT allowance and workspace admission")
    run_parser.add_argument("--fft-layout", choices=("auto", "out-of-place", "in-place"), default="auto",
                            help="auto first tries out-of-place RAM, then the smaller in-place allocation")
    run_parser.add_argument("--window-storage", choices=("auto", "full", "fft_input_then_blocked"), default="auto",
                            help="auto keeps the full window if it fits, otherwise regenerates bounded blocks for a one-shot call")
    run_parser.add_argument("--profile", choices=("full", "performance"), default="full",
                            help="full adds the five historical signal presets at --quality-sizes")
    run_parser.add_argument("--quality-sizes", nargs="+", type=int, default=[100_000, 1_000_000, 1_000_003],
                            help="Selected dataset sizes receiving all six historical signal presets")
    run_parser.add_argument("--work-dir", type=Path,
                            help="Dataset/workspace parent; default is a private directory in the OS temporary directory")
    run_parser.add_argument("--output", type=Path, default=Path.cwd() / "lpsd-benchmark-results.json")
    run_parser.add_argument("--fftw-library", type=Path)
    run_parser.add_argument("--threads-library", type=Path)
    run_parser.add_argument("--save-spectra", action="store_true", help="Keep synthetic spectra locally; never upload them")
    run_parser.add_argument("--upload-github", action="store_true", help="Submit a public issue using the current gh login")
    run_parser.add_argument("--accept-public-upload", action="store_true",
                            help="Explicitly accept a public GitHub issue linked to your account")
    submit_parser = commands.add_parser("submit", help="Prepare/submit the allowlisted public report after a local run")
    submit_parser.add_argument("path", type=Path)
    submit_parser.add_argument("--accept-public-upload", action="store_true")
    return root


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "submit":
        try:
            _print_submission(submit(args.path, args.accept_public_upload))
            return 0
        except (OSError, ValueError):
            print("Cannot prepare submission: unsupported, unreadable or oversized benchmark report "
                  "(limits: 16 MiB per input file; 60,000 bytes per issue/comment; 16 parts). "
                  "If --save-spectra created a .public.json report, "
                  "explicitly submit that file.", file=sys.stderr)
            return 2
    if (not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 24
            or args.threads is not None and args.threads < 1
            or not 2 <= args.base_points <= 8192
            or len(args.sizes) > 32 or any(not 16 <= n <= 2_147_483_647 for n in args.sizes)
            or len(args.quality_sizes) > 32 or any(not 16 <= n <= 2_147_483_647 for n in args.quality_sizes)
            or args.points and (len(args.points) > 8 or any(not 2 <= j <= 8192 for j in args.points))
            or len(args.point_factors) > 8 or any(not math.isfinite(f) or not .25 <= f <= 8 for f in args.point_factors)):
        cli.error("Require <=24 h, positive threads, 16<=N<=INT_MAX, 2<=J<=8192 and .25<=point factors<=8")
    if args.upload_github and not args.accept_public_upload:
        cli.error("--upload-github requires --accept-public-upload: the issue is public and linked to your GitHub account")
    try:
        if args.work_dir is None:
            # Large dataset files must not inherit a checkout/results directory
            # that may be synchronized to cloud storage. Own only this fresh
            # temporary directory; explicit work parents remain user-owned.
            with tempfile.TemporaryDirectory(prefix="lpsd-benchmark-work-") as directory:
                args.work_dir = Path(directory)
                return run(args)
        return run(args)
    except (OSError, ValueError):
        # Do not print exception text containing a personal filesystem path.
        print("Benchmark setup failed: invalid budget, inaccessible output/work directory or invalid configuration.", file=sys.stderr)
        return 2
