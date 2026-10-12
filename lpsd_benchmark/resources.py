"""Conservative admission estimates; an allocation budget is not an OS sandbox.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

from . import MAX_SECONDS

MIB = 1024 ** 2
GIB = 1024 ** 3


def _read(path):
    try:
        return Path(path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return ""


def _positive_int(text):
    try:
        value = int(text)
        return value if 0 < value < 2 ** 60 else None
    except (TypeError, ValueError):
        return None


def _cgroup_directories(controller):
    """Inspect nested v1/v2 limits as well as their visible ancestors."""
    root = Path("/sys/fs/cgroup")
    found = {root, root / controller}
    for line in _read("/proc/self/cgroup").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if controllers and controller not in controllers.split(","):
            continue
        bases = [root] if not controllers else [root / controller, root / controllers]
        for base in bases:
            candidate = base / relative.lstrip("/")
            # cgroup paths describe the running process, never user input. Do
            # not traverse outside a known mount in unusual namespaces.
            try:
                candidate.relative_to(base)
            except ValueError:
                continue
            while candidate != base and base in candidate.parents:
                found.add(candidate)
                candidate = candidate.parent
            found.add(base)
    return sorted(found)


def _clean_file_cache(stats, *, legacy=False):
    """Conservative clean, evictable file bytes in either cgroup layout.

    Active file pages can also be reclaimed. Bound both file LRU lists by
    file/cache minus shared memory, then subtract dirty, writeback and
    unevictable pages. Some exclusions may overlap; subtracting twice is
    deliberately conservative. v1 hierarchical usage needs total_* stats.
    """
    def value(name):
        return stats.get("total_" + name, stats.get(name, 0)) if legacy else stats.get(name, 0)
    file_bytes = max(0, value("cache" if legacy else "file") - value("shmem"))
    file_lru = value("active_file") + value("inactive_file")
    unavailable = (value("dirty" if legacy else "file_dirty")
                   + value("writeback" if legacy else "file_writeback")
                   + value("unevictable"))
    return max(0, min(file_bytes, file_lru) - unavailable)


def system_memory():
    """Return total and currently available bytes without host identifiers."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        total, available = int(memory.total), int(memory.available)
    except (ImportError, OSError, AttributeError):
        values = {}
        for line in _read("/proc/meminfo").splitlines():
            fields = line.split()
            if len(fields) >= 2:
                value = _positive_int(fields[1])
                if value is not None:
                    values[fields[0].rstrip(":")] = value * 1024
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", values.get("MemFree", 0))
        if not total and os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        "total_physical", "available_physical", "total_page",
                        "available_page", "total_virtual", "available_virtual", "extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total, available = status.total_physical, status.available_physical
        if not total and platform.system() == "Darwin":
            try:
                result = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                        capture_output=True, text=True, timeout=2, check=True)
                total = int(result.stdout.strip())
                # A deliberately conservative fallback when psutil is absent.
                available = total // 2
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
    if not total or not available:
        # Unknown RAM is not permission to allocate a large numerical problem.
        return 0, 0
    for directory in _cgroup_directories("memory"):
        limit = _positive_int(_read(directory / "memory.max"))
        current = _positive_int(_read(directory / "memory.current")) or 0
        stat_name = "memory.stat"
        legacy = limit is None
        if limit is None:
            limit = _positive_int(_read(directory / "memory.limit_in_bytes"))
            current = _positive_int(_read(directory / "memory.usage_in_bytes")) or 0
        if limit is None:
            continue
        stats = {}
        for line in _read(directory / stat_name).splitlines():
            fields = line.split()
            if len(fields) == 2:
                stats[fields[0]] = _positive_int(fields[1]) or 0
        # Count clean cache from both file LRUs. A recent disk run can leave
        # gigabytes on active_file without making those pages unevictable.
        # Do not count tmpfs/shared memory or pending filesystem writes.
        reclaimable = min(current, _clean_file_cache(stats, legacy=legacy))
        total = min(total, limit)
        available = min(available, max(0, limit - current + reclaimable))
    return int(total), max(0, min(int(total), int(available)))


def cpu_capacity():
    count = max(1, os.cpu_count() or 1)
    if hasattr(os, "sched_getaffinity"):
        try:
            count = min(count, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    quota = float(count)
    for directory in _cgroup_directories("cpu"):
        fields = _read(directory / "cpu.max").split()
        if len(fields) == 2:
            maximum, period = _positive_int(fields[0]), _positive_int(fields[1])
        else:
            maximum = _positive_int(_read(directory / "cpu.cfs_quota_us"))
            period = _positive_int(_read(directory / "cpu.cfs_period_us"))
        if maximum and period:
            quota = min(quota, maximum / period)
    return max(1, min(count, math.floor(quota))), quota


@dataclass(frozen=True)
class Resources:
    total_ram: int
    available_ram: int
    disk_free: int
    ram_budget: int
    disk_budget: int
    threads: int
    cpu_capacity: float


def budget_bytes(value):
    if value == "auto" or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Budget must be 'auto' or a positive number of GiB") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Budget must be 'auto' or a positive number of GiB")
    return int(number * GIB)


def ram_backed_work_directory(directory, *, mountinfo=Path("/proc/self/mountinfo")):
    """Identify Linux tmpfs/ramfs without counting their capacity as disk.

    Resolve symlinks and prefer the most specific visible mount, including
    bind mounts. Mountinfo escapes whitespace and backslashes as octal bytes.
    No path or mount metadata is added to the public benchmark report.
    """
    if platform.system() != "Linux":
        return False
    try:
        directory = Path(directory).resolve()
        lines = Path(mountinfo).read_text(encoding="utf-8", errors="surrogateescape").splitlines()
    except OSError:
        return False
    filesystem, matched_parts = None, -1
    for line in lines:
        before, separator, after = line.partition(" - ")
        fields, extra = before.split(), after.split()
        if not separator or len(fields) < 5 or not extra:
            continue
        name = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])
        mount = Path(name)
        if not mount.is_absolute() or not directory.is_relative_to(mount):
            continue
        if len(mount.parts) >= matched_parts:
            filesystem, matched_parts = extra[0], len(mount.parts)
    return filesystem in ("tmpfs", "ramfs")


def detect_resources(directory, ram="auto", disk="auto", threads=None):
    total, available = system_memory()
    free = shutil.disk_usage(directory).free
    ram_requested, disk_requested = budget_bytes(ram), budget_bytes(disk)
    # Auto is deliberately cautious. An explicit ceiling may use more of
    # currently available resources, but still leaves a fixed OS/controller
    # reserve and never bypasses physical, cgroup or free-space limits.
    if ram_requested is None:
        ram_budget = max(0, min(int(total * .90), int(available * .80)))
    else:
        reserve = max(256 * MIB, min(2 * GIB, int(total * .10)))
        ram_budget = max(0, min(ram_requested, total - reserve, available - reserve))
    if disk_requested is None:
        disk_budget = max(0, min(int(free * .80), free - 256 * MIB))
    else:
        disk_budget = max(0, min(disk_requested, free - 512 * MIB))
    # tmpfs/ramfs file pages consume RAM and are not included in the disk
    # worker's anonymous-RSS guard. Do not admit a disk fallback there even
    # when its logical filesystem capacity exceeds the available RAM.
    if ram_backed_work_directory(directory):
        disk_budget = 0
    cpu_count, capacity = cpu_capacity()
    used_threads = cpu_count if threads is None else min(cpu_count, int(threads))
    return Resources(total, available, free, ram_budget // MIB * MIB,
                     disk_budget // MIB * MIB, max(1, used_threads), capacity)


def required_ram(n, method, signal="white", fft_layout="out-of-place"):
    """Peak live arrays, bounded native scratch and interpreter overhead.

    This models the complete-call paths used by worker.py, not the much
    larger persistent-cache configuration from earlier reuse experiments.
    FFTW may allocate planner-dependent additional scratch; the monitor and
    address-space guard remain active, including for prime-length transforms.
    """
    per_sample = {"fftw": 32, "matched": 36, "lpsd": 8}[method]
    if fft_layout == "in-place" and method != "lpsd":
        per_sample -= 8
    source_peak = 72 if signal in ("pink", "brown", "mixed_tones") else 24
    if signal == "white":
        source_peak = 8
    # The first case also includes NumPy/pandas/SciPy/native import overhead.
    arrays = per_sample * n
    if method == "lpsd":
        # Minimum useful bounded scratch allowance. The worker resolves the
        # real native plan before creating data, then verifies its exact pool
        # and streaming-buffer bounds against the admitted available budget.
        arrays += min(16 * n, max(128 * MIB, math.ceil(n / 3)))
    elif method == "matched":
        # estimate_once releases FFT/window/powers before bounded low LPSD.
        low_peak = 8 * n + min(16 * n, max(128 * MIB, math.ceil(n / 3)))
        arrays = max(arrays, low_peak)
    return max(arrays, source_peak * n) + 256 * MIB


def required_disk(n, method):
    # Native disk mode stores only the input; its bounded scratch is RAM.
    # FFTW workspace powers need 24N. The worker checks the exact factorization
    # before creating data; odd factors can require another 4N+8 bytes.
    return (8 if method == "lpsd" else 24) * n + 16 * MIB


def choose_fft_layout(n, method, signal, requested, ram_budget):
    if method == "lpsd":
        return "not_applicable"
    if requested != "auto":
        return requested
    if required_ram(n, method, signal, "out-of-place") <= ram_budget:
        return "out-of-place"
    return "in-place"


def select_storage(n, method, signal, requested, resources, fft_layout="out-of-place"):
    ram_need = required_ram(n, method, signal, fft_layout)
    if requested in ("auto", "ram") and ram_need <= resources.ram_budget:
        return "ram", ram_need, 0, None
    if requested == "ram":
        return "ram", ram_need, 0, "memory_budget"
    if signal != "white" and 72 * n + 256 * MIB > resources.ram_budget:
        # Coloured Fourier synthesis itself needs RAM. Small historical
        # signal cases can still use the disk estimator after synthesis.
        return "disk", 72 * n + 256 * MIB, required_disk(n, method), "memory_budget"
    disk_need = required_disk(n, method)
    minimum_ram = 384 * MIB
    if resources.ram_budget < minimum_ram:
        return "disk", minimum_ram, disk_need, "memory_budget"
    if disk_need > resources.disk_budget:
        return "disk", minimum_ram, disk_need, "disk_budget"
    # Current blocked FFT is a factorization, not a prime-length substitute.
    # Large prime cases admitted to RAM can use FFTW normally. For disk, the
    # worker's factor preflight makes the final decision before allocating.
    return "disk", minimum_ram, disk_need, None


class Deadline:
    """Monotonic budget, also capped by the bootstrap's setup-inclusive end."""
    def __init__(self, seconds, external_epoch=None, *, monotonic=time.monotonic,
                 wallclock=time.time):
        if not math.isfinite(seconds) or not 0 < seconds <= MAX_SECONDS:
            raise ValueError("Total budget must be positive and at most 24 hours")
        self._clock = monotonic
        self.started = monotonic()
        remaining = seconds
        if external_epoch is not None:
            end = float(external_epoch)
            if not math.isfinite(end):
                raise ValueError("Invalid bootstrap deadline")
            remaining = min(remaining, max(0., end - wallclock()))
        self.seconds = remaining
        self.end = self.started + remaining

    def remaining(self):
        return max(0., self.end - self._clock())

    def elapsed(self):
        return max(0., self._clock() - self.started)


def predict_seconds(case, completed, safety_factor=4.0):
    """Conservative whole-process forecast; never report this as a timing."""
    matching = [row for row in completed if row.get("status") == "completed"
                and row.get("method") == case["method"]
                and row.get("signal") == case["signal"]]
    if not matching:
        # Small calibration cases get a bounded opportunity on an unknown
        # machine. They are the only starts without measured scaling evidence.
        return 180.0 if case["n"] <= 100_000 else None
    estimates = []
    for row in matching:
        size_ratio = case["n"] / row["n"]
        if case.get("point_factor") and row.get("point_factor"):
            point_ratio = case["point_factor"] / row["point_factor"]
        else:
            point_ratio = case["target_points"] / row["target_points"]
        size_cost = max(1., size_ratio ** 1.35)
        point_cost = max(1., point_ratio ** 1.20)
        storage_cost = 8. if row["storage"] == "ram" and case["storage"] == "disk" else 1.
        # Process setup is included, which intentionally overestimates small
        # cases and avoids optimistic extrapolation from raw FFT-only time.
        estimates.append(max(1., row["process_wall_s"]) * size_cost * point_cost * storage_cost)
    return max(15.0, min(estimates) * safety_factor)
