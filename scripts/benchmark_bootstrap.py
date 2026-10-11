#!/usr/bin/env python3
"""Install a pinned, private benchmark environment and run its bounded CLI.

The shell/PowerShell entry points first provide a private uv and Python. This
stdlib-only second stage owns the source archive, wheel lock and compiler. It
does not install an OS package, edit a profile, or submit a result itself.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile


GIB = 1024 ** 3
REPOSITORY = "MichaDit/lpsd"
MAX_ARCHIVE_BYTES = 160 * 1024 ** 2
MAX_SOURCE_BYTES = 256 * 1024 ** 2
MAX_TOOL_BYTES = 1024 ** 3
MAX_FILES = 50_000
MAX_HOURS = 24.0
DOWNLOAD_HOSTS = frozenset({
    "github.com", "codeload.github.com", "raw.githubusercontent.com",
    "release-assets.githubusercontent.com", "objects.githubusercontent.com",
    "ziglang.org", "releases.astral.sh",
})


class BootstrapError(RuntimeError):
    """A short local diagnostic, never a raw network or subprocess response."""


def positive_hours(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("hours must be a number") from error
    if not math.isfinite(value) or not 0 < value <= MAX_HOURS:
        raise argparse.ArgumentTypeError("hours must be greater than 0 and at most 24")
    return value


def revision_value(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("revision must be a full 40-character Git commit")
    return value.lower()


def directory_bytes(root):
    """Count only files owned by this fresh directory, without following links."""
    total = 0
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [name for name in names
                    if not (Path(directory) / name).is_symlink()]
        for name in files:
            try:
                entry = (Path(directory) / name).lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(entry.st_mode):
                total += entry.st_size
    return total


def stop_process(process):
    """Stop the active setup process and descendants; never kill unrelated jobs."""
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass


def setup_process_memory():
    """Resident memory of this bootstrap and its descendants; no identifiers saved."""
    entries = {}
    own_pid = os.getpid()
    if os.name == "nt":
        from ctypes import wintypes
        class Entry(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                        ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                        ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                        ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                        ("flags", wintypes.DWORD), ("executable", wintypes.WCHAR * 260)]
        class Counters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                    "nonpaged", "pagefile", "peak_pagefile")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            return None
        try:
            entry = Entry(); entry.size = ctypes.sizeof(entry)
            valid = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while valid:
                entries[int(entry.pid)] = [int(entry.parent), 0]
                valid = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel.CloseHandle(snapshot)
    elif sys.platform.startswith("linux"):
        # /proc can be mounted in an ancestor PID namespace. Its visible
        # numeric directories then differ from os.getpid()/Popen.pid.
        try:
            own_status = Path("/proc/self/status").read_text().splitlines()
            own_pid = next(int(line.split()[1]) for line in own_status if line.startswith("Pid:"))
        except (OSError, ValueError, StopIteration):
            return None
        for directory in Path("/proc").glob("[0-9]*"):
            try:
                fields = {}
                for line in (directory / "status").read_text().splitlines():
                    parts = line.split()
                    if parts and parts[0] in ("PPid:", "VmRSS:"):
                        fields[parts[0]] = int(parts[1])
                entries[int(directory.name)] = [fields.get("PPid:", 0), fields.get("VmRSS:", 0) * 1024]
            except (OSError, ValueError):
                continue
    else:
        try:
            probe = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="],
                                   capture_output=True, text=True, timeout=2, check=True)
            for line in probe.stdout.splitlines():
                pid, parent, rss = map(int, line.split())
                entries[pid] = [parent, rss * 1024]
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    if own_pid not in entries:
        return None
    descendants = {own_pid}
    while True:
        expanded = descendants | {pid for pid, (parent, _) in entries.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    if os.name == "nt":
        for pid in descendants:
            handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
            if not handle:
                continue  # A process may have exited since the snapshot.
            try:
                counters = Counters(); counters.size = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.size):
                    entries[pid][1] = int(counters.rss)
            finally:
                kernel.CloseHandle(handle)
    return sum(entries[pid][1] for pid in descendants)


class Budget:
    def __init__(self, root, hours, setup_bytes, *, inherited_deadline=None):
        self.root = Path(root)
        self.setup_bytes = int(setup_bytes)
        started = time.time()
        deadline = started + positive_hours(hours) * 3600
        if inherited_deadline is not None:
            try:
                inherited = float(inherited_deadline)
            except (TypeError, ValueError) as error:
                raise BootstrapError("Invalid inherited deadline") from error
            if not math.isfinite(inherited):
                raise BootstrapError("Invalid inherited deadline")
            deadline = min(deadline, inherited)
        self.deadline_epoch = deadline
        self.deadline_monotonic = time.monotonic() + max(0, deadline - started)
        self.process = None
        self.finished = threading.Event()
        self.setup_ram_bytes = None
        self.peak_setup_rss_bytes = 0
        self._last_memory_probe = -math.inf

    def remaining(self):
        # A wall-clock correction backwards cannot extend the original budget.
        return min(self.deadline_epoch - time.time(),
                   self.deadline_monotonic - time.monotonic())

    def check(self, additional=0, *, disk=True):
        if self.remaining() <= 0:
            raise BootstrapError("Setup/run deadline reached")
        if disk and directory_bytes(self.root) + additional > self.setup_bytes:
            raise BootstrapError("Private setup disk budget exceeded")
        if disk and shutil.disk_usage(self.root).free < additional + 32 * 1024 ** 2:
            raise BootstrapError("Insufficient free disk space; 32 MiB reserve retained")
        if disk and self.setup_ram_bytes is not None and time.monotonic() - self._last_memory_probe >= .25:
            self._last_memory_probe = time.monotonic()
            measured = setup_process_memory()
            if measured is None:
                raise BootstrapError("Cannot monitor private setup RAM on this platform")
            self.peak_setup_rss_bytes = max(self.peak_setup_rss_bytes, measured)
            if measured > self.setup_ram_bytes:
                raise BootstrapError("Private setup RAM budget exceeded")

    def start_watchdog(self):
        def watch():
            while not self.finished.wait(min(0.25, max(0.001, self.remaining()))):
                if self.remaining() <= 0:
                    try:
                        stop_process(self.process)
                    finally:
                        # Also bounds DNS, decompression and blocked download APIs.
                        os.write(2, b"Benchmark setup/run deadline reached.\n")
                        os._exit(124)
        threading.Thread(target=watch, daemon=True,
                         name="lpsd-bootstrap-deadline").start()


class RestrictedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        checked_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def checked_url(url):
    parts = urllib.parse.urlsplit(url)
    if (parts.scheme != "https" or parts.hostname not in DOWNLOAD_HOSTS
            or parts.username or parts.password or parts.port not in (None, 443)):
        raise BootstrapError("Download URL is outside the official HTTPS allowlist")
    return url


def download(url, destination, budget, *, expected_sha256=None,
             max_bytes=MAX_ARCHIVE_BYTES):
    checked_url(url)
    destination = Path(destination)
    budget.check()
    digest = hashlib.sha256()
    received = 0
    try:
        opener = urllib.request.build_opener(RestrictedRedirect())
        request = urllib.request.Request(url, headers={"User-Agent": "lpsd-benchmark-bootstrap/1"})
        with opener.open(request, timeout=min(30, max(0.1, budget.remaining()))) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise BootstrapError("Download exceeds its bounded file size")
            with destination.open("xb") as target:
                while True:
                    data = response.read(1024 * 1024)
                    if not data:
                        break
                    received += len(data)
                    if received > max_bytes:
                        raise BootstrapError("Download exceeds its bounded file size")
                    budget.check(additional=len(data))
                    digest.update(data)
                    target.write(data)
    except Exception as error:
        destination.unlink(missing_ok=True)
        if isinstance(error, BootstrapError):
            raise
        raise BootstrapError("Official HTTPS download failed; no response or credentials logged") from error
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        destination.unlink(missing_ok=True)
        raise BootstrapError("Downloaded archive SHA256 does not match the pinned value")
    return actual


def archive_path(name):
    if not name or "\\" in name or ":" in name or any(ord(c) < 32 for c in name):
        raise BootstrapError("Archive contains an unsafe path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise BootstrapError("Archive contains an unsafe path")
    return path


def extract_archive(archive, destination, budget, *, max_bytes, expected_root=None,
                    single_root=True):
    """Extract regular files/directories only, with traversal/link/size checks."""
    destination = Path(destination)
    destination.mkdir()
    extracted = 0
    seen = set()
    setup_before = directory_bytes(budget.root)
    free_before = shutil.disk_usage(budget.root).free

    def validate(name, size, is_directory, is_regular):
        nonlocal extracted
        path = archive_path(name)
        if expected_root is not None and path.parts[0] != expected_root:
            raise BootstrapError("Source archive does not identify the pinned revision")
        if not is_directory and not is_regular:
            raise BootstrapError("Archive links and special files are not permitted")
        normalized = path.as_posix().rstrip("/")
        if normalized in seen:
            raise BootstrapError("Archive contains duplicate file names")
        seen.add(normalized)
        if len(seen) > MAX_FILES or size < 0:
            raise BootstrapError("Archive contains too many entries or an invalid size")
        extracted += size
        if extracted > max_bytes:
            raise BootstrapError("Expanded archive exceeds its bounded size")
        if setup_before + extracted > budget.setup_bytes:
            raise BootstrapError("Expanded archive exceeds the private setup disk budget")
        if extracted + 32 * 1024 ** 2 > free_before:
            raise BootstrapError("Insufficient free space for the expanded archive")
        budget.check(disk=False)
        target = destination.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def copy(source, target, size, mode):
        copied = 0
        with target.open("xb") as output:
            while True:
                budget.check(disk=False)
                block = source.read(min(1024 * 1024, size - copied + 1))
                if not block:
                    break
                copied += len(block)
                if copied > size:
                    raise BootstrapError("Archive entry has an invalid expanded size")
                output.write(block)
        if copied != size:
            raise BootstrapError("Archive entry is truncated")
        target.chmod(0o755 if mode & 0o111 else 0o644)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                mode = entry.external_attr >> 16
                directory = entry.is_dir()
                regular = not stat.S_ISLNK(mode) and (not stat.S_IFMT(mode) or stat.S_ISREG(mode))
                target = validate(entry.filename, entry.file_size, directory, regular)
                if directory:
                    target.mkdir(exist_ok=True)
                else:
                    with bundle.open(entry) as source:
                        copy(source, target, entry.file_size, mode)
    else:
        with tarfile.open(archive, "r:*") as bundle:
            for entry in bundle:
                target = validate(entry.name, entry.size, entry.isdir(), entry.isfile())
                if entry.isdir():
                    target.mkdir(exist_ok=True)
                else:
                    with bundle.extractfile(entry) as source:
                        copy(source, target, entry.size, entry.mode)
    budget.check()
    if not single_root:
        return destination
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise BootstrapError("Expected one archive root directory")
    return roots[0]


def child_environment(root, deadline):
    # Configured indexes, inherited Python paths and installer settings must not
    # silently replace the pinned public dependency set. Keep OS/proxy settings.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("UV_", "PIP_", "PYTHONPATH", "PYTHONHOME"))}
    environment.update({
        "UV_NO_CONFIG": "1", "UV_NO_PROGRESS": "1",
        "UV_CONCURRENT_DOWNLOADS": "2", "UV_CONCURRENT_INSTALLS": "2",
        "UV_CONCURRENT_BUILDS": "1",
        "UV_CACHE_DIR": str(root / "cache"),
        "UV_PYTHON_INSTALL_DIR": str(root / "python"),
        "UV_TOOL_DIR": str(root / "tools"),
        "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
        "LPSD_BENCHMARK_DEADLINE_EPOCH": str(deadline),
        "ZIG_GLOBAL_CACHE_DIR": str(root / "zig-cache"),
        "ZIG_LOCAL_CACHE_DIR": str(root / "zig-local-cache"),
        "TMPDIR": str(root / "temporary"),
        "TMP": str(root / "temporary"), "TEMP": str(root / "temporary"),
        "GH_TELEMETRY": "0", "DO_NOT_TRACK": "1",
        "GH_NO_UPDATE_NOTIFIER": "1", "GH_NO_EXTENSION_UPDATE_NOTIFIER": "1",
    })
    environment.pop("GH_DEBUG", None)
    (root / "temporary").mkdir(exist_ok=True)
    return environment


def run_checked(command, budget, *, cwd, environment, monitor_setup=True):
    budget.check(disk=monitor_setup)
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    # Build/install messages can contain local paths. They remain only here and
    # are never included in the runner's sanitized issue payload.
    process = subprocess.Popen(command, cwd=cwd, env=environment, **options)
    budget.process = process
    try:
        while process.poll() is None:
            budget.check(disk=monitor_setup)
            try:
                process.wait(timeout=min(1, max(0.001, budget.remaining())))
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise BootstrapError("A setup or benchmark stage failed (exit %d)" % process.returncode)
    except BaseException:
        stop_process(process)
        raise
    finally:
        budget.process = None


def compiler_command(pins, root, budget, environment, choice):
    if choice != "zig":
        requested = os.environ.get("CC")
        candidates = [shlex.split(requested)] if requested else [[name] for name in (
            ("gcc", "clang") if os.name == "nt" else ("cc", "gcc", "clang"))]
        for candidate in candidates:
            if candidate and shutil.which(candidate[0]):
                # Avoid Apple's tool stub opening an installation dialog.
                if sys.platform == "darwin" and candidate[0] in ("cc", "gcc", "clang"):
                    probe = subprocess.run(["xcode-select", "-p"], stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, timeout=5, check=False)
                    if probe.returncode:
                        continue
                return shlex.join(candidate), "existing"
        if choice == "system":
            raise BootstrapError("No GCC/Clang-compatible compiler available")
    if "TERMUX_VERSION" in os.environ:
        raise BootstrapError("Termux needs its own clang compiler; desktop Zig is not used on Android")
    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system())
    machine = {"AMD64": "x86_64", "arm64": "aarch64"}.get(platform.machine(), platform.machine())
    key = "%s-%s" % (machine, system)
    if key not in pins["zig"]:
        raise BootstrapError("No pinned portable compiler for this platform")
    item = pins["zig"][key]
    archive = root / ("zig.zip" if system == "windows" else "zig.tar.xz")
    print("Downloading a private C compiler (Zig %s)." % pins["zig_version"], flush=True)
    download(item["url"], archive, budget, expected_sha256=item["sha256"], max_bytes=item["size"])
    tool = extract_archive(archive, root / "compiler", budget, max_bytes=MAX_TOOL_BYTES)
    archive.unlink()
    command = [str(tool / ("zig.exe" if system == "windows" else "zig")), "cc"]
    if system == "windows":
        command += ["-target", machine + "-windows-gnu"]
    return shlex.join(command), "zig-" + pins["zig_version"]


def prepare_github(pins, root, budget, environment, *, upload, accepted, login):
    """Provide optional gh, with public upload and interactive login independent.

    This function never posts an issue. Authentication is an explicit user
    choice; a missing login leaves the numerical benchmark usable locally.
    """
    if login and not (upload and accepted):
        raise BootstrapError("--github-login requires --upload-github --accept-public-upload")
    if not (upload and accepted):
        return "not-requested"
    executable = shutil.which("gh", path=environment.get("PATH"))
    origin = "existing"
    if executable is None:
        if "TERMUX_VERSION" in os.environ:
            print("Termux: install its gh package or submit the prepared issue in your browser.", flush=True)
            return "unavailable"
        system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system())
        machine = {"AMD64": "x86_64", "arm64": "aarch64"}.get(platform.machine(), platform.machine())
        item = pins.get("gh", {}).get("%s-%s" % (machine, system))
        if item is None:
            print("No pinned GitHub CLI for this platform; the issue can be submitted in your browser.", flush=True)
            return "unavailable"
        suffix = ".tar.gz" if item["url"].endswith(".tar.gz") else ".zip"
        archive = root / ("gh" + suffix)
        try:
            print("Downloading private GitHub CLI %s for the requested upload." % pins["gh_version"], flush=True)
            download(item["url"], archive, budget, expected_sha256=item["sha256"], max_bytes=item["size"])
            tool_root = extract_archive(archive, root / "github-cli", budget,
                                        max_bytes=128 * 1024 ** 2, single_root=False)
            name = "gh.exe" if os.name == "nt" else "gh"
            matches = [path for path in tool_root.rglob(name) if path.is_file()]
            if len(matches) != 1:
                raise BootstrapError("Expected exactly one private GitHub CLI executable")
            executable = str(matches[0])
        except (BootstrapError, OSError):
            archive.unlink(missing_ok=True)
            shutil.rmtree(root / "github-cli", ignore_errors=True)
            budget.check()
            print("GitHub CLI setup unavailable; numerical results and an issue draft will stay local.", flush=True)
            return "unavailable"
        archive.unlink()
        environment["PATH"] = str(Path(executable).parent) + os.pathsep + environment.get("PATH", "")
        origin = "private-" + pins["gh_version"]
    if origin.startswith("private-") or login:
        configuration = root / "github-config"
        configuration.mkdir(mode=0o700, exist_ok=True)
        environment["GH_CONFIG_DIR"] = str(configuration)
    if login:
        if environment.get("GH_TOKEN") or environment.get("GITHUB_TOKEN"):
            print("An existing GitHub authentication environment is in use; browser login is unnecessary.", flush=True)
        else:
            login_environment = environment.copy()
            login_environment.pop("GH_PROMPT_DISABLED", None)
            # gh may offer Git authentication during browser login. Any Git
            # configuration it writes belongs to this private directory.
            login_environment.update({
                "GH_HOST": "github.com", "GIT_CONFIG_GLOBAL": str(root / "github-git-config"),
                "GIT_CONFIG_SYSTEM": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
            })
            print("Explicit GitHub browser/device login requested. Do not paste tokens into benchmark arguments.", flush=True)
            try:
                run_checked([executable, "auth", "login", "--hostname", "github.com",
                             "--git-protocol", "https", "--skip-ssh-key", "--web"],
                            budget, cwd=root, environment=login_environment)
            except BootstrapError:
                budget.check()
                print("GitHub login did not complete; the benchmark will keep a local issue draft if upload fails.", flush=True)
    return origin


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--revision", required=True, type=revision_value)
    parser.add_argument("--bootstrap-root", type=Path,
                        help="private directory created by the platform entry point")
    parser.add_argument("--bootstrap-disk-gib", type=float, default=2.0,
                        help="maximum private setup allocation, default 2 GiB")
    parser.add_argument("--compiler", choices=("auto", "system", "zig"), default="auto")
    parser.add_argument("--uv-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-dir", type=Path,
                        help="developer/CI: copy this local source instead of downloading")
    parser.add_argument("--github-login", action="store_true",
                        help="explicitly open gh browser/device login for a requested public upload")
    parser.add_argument("--max-hours", type=positive_hours, default=24.0)
    parser.add_argument("--ram-gib", default="auto", help="setup and numerical RAM allocation budget")
    return parser.parse_known_args(arguments)


def main(arguments=None):
    options, remaining = parse_args(arguments)
    if not math.isfinite(options.bootstrap_disk_gib) or options.bootstrap_disk_gib <= 0:
        raise BootstrapError("Setup disk budget must be positive and finite")
    if sys.version_info[:2] != (3, 12) and "TERMUX_VERSION" not in os.environ:
        raise BootstrapError("Use the platform entry point: this lock targets Python 3.12")
    if remaining[:1] == ["--"]:
        remaining.pop(0)
    if options.github_login and not {"--upload-github", "--accept-public-upload"}.issubset(remaining):
        raise BootstrapError("--github-login requires --upload-github --accept-public-upload")
    root = options.bootstrap_root
    if root is None:
        root = Path(tempfile.mkdtemp(prefix="lpsd-benchmark-", dir=Path.cwd()))
    root = root.absolute()
    if root.is_symlink():
        raise BootstrapError("Setup directory must not be a symlink")
    root.mkdir(exist_ok=True)
    budget = Budget(root, options.max_hours, options.bootstrap_disk_gib * GIB,
                    inherited_deadline=os.environ.get("LPSD_BENCHMARK_DEADLINE_EPOCH"))
    budget.check()
    budget.start_watchdog()
    environment = child_environment(root, budget.deadline_epoch)
    print("Private benchmark directory:", root, flush=True)
    try:
        if options.source_dir is None:
            archive = root / "source.zip"
            source_hash = download("https://codeload.github.com/" + REPOSITORY + "/zip/" + options.revision,
                                   archive, budget)
            source = extract_archive(archive, root / "source", budget,
                                     max_bytes=MAX_SOURCE_BYTES, expected_root="lpsd-" + options.revision)
            archive.unlink()
        else:
            source = root / "source"
            if options.source_dir.resolve() == root.resolve():
                raise BootstrapError("Local source and private setup directory must differ")
            pattern_ignore = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc",
                                                    "lpsd-benchmark.*", "lpsd-benchmark-*")

            def ignored(directory, names):
                hidden = set(pattern_ignore(directory, names))
                hidden.update(name for name in names
                              if (Path(directory) / name).resolve() == root.resolve())
                return hidden

            # The local/CI path is explicit, but must not recursively copy its
            # own freshly created setup directory when launched from a checkout.
            source_size = 0
            for directory, names, files in os.walk(options.source_dir, followlinks=False):
                names[:] = [name for name in names if name not in ignored(directory, names)]
                for name in files:
                    path = Path(directory) / name
                    if name not in ignored(directory, files) and not path.is_symlink():
                        source_size += path.stat().st_size
            if source_size > MAX_SOURCE_BYTES:
                raise BootstrapError("Local source exceeds the bounded source size")
            budget.check(additional=source_size)
            shutil.copytree(options.source_dir, source, ignore=ignored, symlinks=True)
            source_hash = "local-developer-source"
        pins = json.loads((source / "scripts/benchmark-bootstrap-pins.json").read_text())
        # This resource module is stdlib-only and does not import numerical
        # packages. Apply the same physical/cgroup/available-memory policy to
        # package installation and native compilation as to the later run.
        sys.path.insert(0, str(source))
        try:
            from lpsd_benchmark.resources import detect_resources
            resources = detect_resources(root, ram=options.ram_gib)
        finally:
            sys.path.pop(0)
        budget.setup_ram_bytes = resources.ram_budget
        if resources.ram_budget < 128 * 1024 ** 2:
            raise BootstrapError("Private setup needs at least 128 MiB of admitted RAM")
        budget.check()
        uv = options.uv_path or Path(shutil.which("uv") or "")
        termux = "TERMUX_VERSION" in os.environ
        python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if termux:
            # Android's Bionic ABI cannot consume these desktop Linux wheels.
            run_checked([sys.executable, "-m", "venv", "--system-site-packages", str(root / "venv")],
                        budget, cwd=root, environment=environment)
            run_checked([str(python), "-c", "import numpy, pandas, scipy, psutil"],
                        budget, cwd=root, environment=environment)
        else:
            if not uv.is_file():
                raise BootstrapError("Private uv missing; use run_benchmark.sh or run_benchmark.ps1")
            run_checked([str(uv), "--no-config", "venv", "--python", sys.executable,
                         str(root / "venv")], budget, cwd=root, environment=environment)
            run_checked([str(uv), "--no-config", "--no-cache", "pip", "install", "--python", str(python),
                         "--index-url", "https://pypi.org/simple", "--no-deps", "--require-hashes",
                         "--only-binary", ":all:", "-r", str(source / "scripts/benchmark-requirements.lock")],
                        budget, cwd=root, environment=environment)
        compiler, compiler_origin = compiler_command(pins, root, budget, environment, options.compiler)
        environment["CC"] = compiler
        environment["LPSD_BENCHMARK_SOURCE_COMMIT"] = options.revision
        if options.source_dir is not None:
            environment["LPSD_BENCHMARK_SOURCE_COMMIT"] = "local"
        run_checked([str(python), "build_native.py"], budget, cwd=source, environment=environment)
        run_checked([str(python), "lpsd_fast/build.py", "--native"], budget, cwd=source, environment=environment)
        github_origin = prepare_github(pins, root, budget, environment,
                                       upload="--upload-github" in remaining,
                                       accepted="--accept-public-upload" in remaining,
                                       login=options.github_login)
        for name in ("cache", "zig-cache", "zig-local-cache", "temporary"):
            shutil.rmtree(root / name, ignore_errors=True)
        (root / "temporary").mkdir(exist_ok=True)
        provenance = {
            "schema": "lpsd-bootstrap-v1", "repository": REPOSITORY,
            "requested_revision": options.revision, "source_archive_sha256": source_hash,
            "python": platform.python_version(), "compiler_origin": compiler_origin,
            "github_cli_origin": github_origin,
            "dependency_lock_sha256": hashlib.sha256((source / "scripts/benchmark-requirements.lock").read_bytes()).hexdigest(),
            "setup_wall_seconds": max(0, time.time() - float(os.environ.get(
                "LPSD_BENCHMARK_STARTED_EPOCH", str(budget.deadline_epoch - options.max_hours * 3600)))),
            "setup_bytes": directory_bytes(root), "termux_system_packages": termux,
            "setup_ram_budget_bytes": budget.setup_ram_bytes,
            "peak_setup_rss_bytes": budget.peak_setup_rss_bytes,
        }
        (root / "bootstrap-local.json").write_text(json.dumps(provenance, indent=2) + "\n")
        command = [str(python), "-m", "lpsd_benchmark", "run", "--max-hours", str(options.max_hours),
                   "--ram-gib", str(options.ram_gib),
                   "--work-dir", str(root / "work"), "--output", str(root / "results.json"), *remaining]
        print("Starting the benchmark; remaining total budget %.1f seconds." % budget.remaining(), flush=True)
        run_checked(command, budget, cwd=source, environment=environment, monitor_setup=False)
    finally:
        budget.finished.set()
    print("Finished. Local results:", root / "results.json", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BootstrapError, KeyboardInterrupt) as error:
        print("Bootstrap stopped:", str(error) if isinstance(error, BootstrapError) else "interrupted", file=sys.stderr)
        raise SystemExit(2)
