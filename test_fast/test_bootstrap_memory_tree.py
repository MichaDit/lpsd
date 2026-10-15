"""Complete standalone setup memory probes, including read and exit races."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from .test_bootstrap import bootstrap, ROOT


def _proc_tree(root):
    parents = {2000: 1, 3000: 2000, 3100: 3000, 4000: 1}
    for pid, parent in parents.items():
        path = root / str(pid)
        path.mkdir()
        rss = {2000: 100, 3000: 200, 3100: 300, 4000: 9000}[pid]
        local = os.getpid() if pid == 2000 else pid // 1000 + 5
        (path / "status").write_text(
            f"Pid:\t{pid}\nPPid:\t{parent}\nNSpid:\t{pid} {local}\n"
            f"State:\tS (sleeping)\nThreads:\t1\nVmRSS:\t{rss} kB\n")
    (root / "self").mkdir()
    (root / "self" / "status").write_bytes((root / "2000" / "status").read_bytes())
    return root


def test_linux_setup_counts_self_and_all_descendants_with_one_inventory(tmp_path, monkeypatch):
    root = _proc_tree(tmp_path)
    scans, reads = [], []
    original_iterdir, original_read = Path.iterdir, Path.read_text

    def inventory(path):
        if path == root:
            scans.append(path)
        return original_iterdir(path)

    def read(path, *args, **kwargs):
        reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", inventory)
    monkeypatch.setattr(Path, "read_text", read)
    for _ in range(2):
        scans.clear(); reads.clear()
        assert bootstrap._linux_setup_memory(root) == 600 * 1024
        assert len(scans) == 1
        assert all(reads.count(root / str(pid) / "status") == 1 for pid in (2000, 3000, 3100))


@pytest.mark.parametrize("pid", [2000, 3000, 3100, 4000])
def test_linux_setup_cannot_accept_an_incomplete_live_inventory(tmp_path, monkeypatch, pid):
    root = _proc_tree(tmp_path)
    denied = root / str(pid) / "status"
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == denied:
            raise PermissionError("controlled live-process denial")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert bootstrap._linux_setup_memory(root) is None
    assert denied.is_file()


@pytest.mark.parametrize("replacement", ["", "VmRSS:\t-1 kB\n", "VmRSS:\tbad kB\n",
                                          "VmRSS:\t300 unknown\n", "VmRSS:\n"])
def test_linux_setup_missing_or_invalid_live_rss_is_not_zero(tmp_path, replacement):
    root = _proc_tree(tmp_path)
    status = root / "3100" / "status"
    status.write_text(status.read_text().replace("VmRSS:\t300 kB\n", replacement))
    assert bootstrap._linux_setup_memory(root) is None


@pytest.mark.parametrize("before, after", [
    ("Pid:\t3100\n", "Pid:\t9999\n"),
    ("PPid:\t3000\n", ""),
    ("PPid:\t3000\n", "PPid:\t-1\n"),
    ("PPid:\t3000\n", "PPid:\t3100\n"),
    ("NSpid:\t3100 8\n", "NSpid:\t3100\n"),
])
def test_linux_setup_rejects_invalid_relationships_and_namespace_depth(tmp_path, before, after):
    root = _proc_tree(tmp_path)
    status = root / "3100" / "status"
    assert before in status.read_text()
    status.write_text(status.read_text().replace(before, after))
    assert bootstrap._linux_setup_memory(root) is None


def test_linux_setup_missing_status_in_existing_process_is_unknown(tmp_path):
    root = _proc_tree(tmp_path)
    (root / "3100" / "status").unlink()
    assert bootstrap._linux_setup_memory(root) is None


def test_linux_setup_confirmed_disappearance_during_read_is_an_exit(tmp_path, monkeypatch):
    root = _proc_tree(tmp_path)
    disappearing = root / "3100" / "status"
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == disappearing:
            shutil.rmtree(path.parent)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert bootstrap._linux_setup_memory(root) == 300 * 1024


@pytest.mark.parametrize("threads, expected", [("1", 300 * 1024), ("2", None),
                                                (None, None), ("0", None), ("bad", None)])
def test_linux_setup_zombie_leader_requires_verified_single_thread(tmp_path, threads, expected):
    root = _proc_tree(tmp_path)
    thread_line = "" if threads is None else f"Threads:\t{threads}\n"
    (root / "3100" / "status").write_text(
        "Pid:\t3100\nPPid:\t3000\nNSpid:\t3100 8\nState:\tZ (zombie)\n" + thread_line)
    assert bootstrap._linux_setup_memory(root) == expected


def test_linux_setup_accepts_old_status_without_nspid_and_non_utf8_names(tmp_path):
    root = _proc_tree(tmp_path)
    for path in root.glob("*/status"):
        text = "\n".join(line for line in path.read_text().splitlines() if not line.startswith("NSpid:"))
        path.write_bytes(b"Name:\tprocess-\xff\xfe\n" + text.encode())
    assert bootstrap._linux_setup_memory(root) == 600 * 1024


def test_linux_setup_unavailable_measurement_trips_unchanged_budget(tmp_path, monkeypatch):
    root = _proc_tree(tmp_path)
    (root / "3100" / "status").unlink()
    monkeypatch.setattr(bootstrap, "setup_process_memory", lambda: bootstrap._linux_setup_memory(root))
    budget = bootstrap.Budget(root, .1, 16 * 1024 ** 2)
    budget.setup_ram_bytes = 128 * 1024 ** 2
    with pytest.raises(bootstrap.BootstrapError, match="Cannot monitor"):
        budget.check()
    assert budget.peak_setup_rss_bytes == 0


class _WinFunction:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


class _WindowsProbe:
    """Exercise actual ctypes control flow without pretending to run Windows."""
    def __init__(self, monkeypatch):
        self.owner = os.getpid()
        self.child, self.grandchild, self.other = (self.owner + offset for offset in (10001, 10002, 10003))
        self.parents = {self.owner: 1, self.child: self.owner,
                        self.grandchild: self.child, self.other: 1}
        self.memory = {self.owner: 100, self.child: 200, self.grandchild: 300, self.other: 9000}
        self.snapshots = [self.parents]
        self.snapshot_count = 0
        self.snapshot_fail = set()
        self.end_errors = {}
        self.enumerations = {}
        self.denied_open, self.denied_memory, self.denied_exit = set(), set(), set()
        self.exit_codes, self.closed, self.opened, self.last_error = {}, [], [], 0
        kernel = SimpleNamespace(**{name: _WinFunction(getattr(self, name)) for name in (
            "CreateToolhelp32Snapshot", "Process32FirstW", "Process32NextW", "OpenProcess",
            "CloseHandle", "GetExitCodeProcess")})
        psapi = SimpleNamespace(GetProcessMemoryInfo=_WinFunction(self.GetProcessMemoryInfo))
        monkeypatch.setattr(bootstrap.ctypes, "WinDLL", lambda name, **kwargs:
                            kernel if name == "kernel32" else psapi, raising=False)
        monkeypatch.setattr(bootstrap.ctypes, "set_last_error", self.set_last_error, raising=False)
        monkeypatch.setattr(bootstrap.ctypes, "get_last_error", lambda: self.last_error, raising=False)

    def set_last_error(self, value):
        self.last_error = value

    def CreateToolhelp32Snapshot(self, flags, pid):
        assert (flags, pid) == (2, 0)
        self.snapshot_count += 1
        if self.snapshot_count in self.snapshot_fail:
            return bootstrap.ctypes.c_void_p(-1).value
        handle = self.snapshot_count
        self.enumerations[handle] = iter(self.snapshots[min(handle - 1, len(self.snapshots) - 1)].items())
        return handle

    def Process32FirstW(self, snapshot, pointer):
        return self.Process32NextW(snapshot, pointer)

    def Process32NextW(self, snapshot, pointer):
        pair = next(self.enumerations[snapshot], None)
        if pair is None:
            self.last_error = self.end_errors.get(snapshot, 18)
            return 0
        pointer._obj.pid, pointer._obj.parent = pair
        return 1

    def OpenProcess(self, rights, inherit, pid):
        assert rights == 0x1000 and inherit is False
        if pid in self.denied_open:
            self.last_error = 5
            return 0
        handle = pid + 10_000_000
        self.opened.append(handle)
        return handle

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1

    def GetProcessMemoryInfo(self, handle, pointer, size):
        pid = handle - 10_000_000
        if pid in self.denied_memory:
            return 0
        pointer._obj.rss = self.memory[pid]
        return 1

    def GetExitCodeProcess(self, handle, pointer):
        pid = handle - 10_000_000
        if pid in self.denied_exit:
            return 0
        pointer._obj.value = self.exit_codes.get(pid, 259)
        return 1

    def read(self):
        result = bootstrap._windows_setup_memory()
        assert all(handle in self.closed for handle in self.opened)
        return result


def test_windows_setup_counts_complete_tree_and_closes_handles(monkeypatch):
    probe = _WindowsProbe(monkeypatch)
    assert probe.read() == 600
    assert probe.snapshot_count == 1
    assert 1 in probe.closed


@pytest.mark.parametrize("which", ["owner", "child", "grandchild"])
def test_windows_setup_live_open_failure_invalidates_positive_parent_sum(monkeypatch, which):
    probe = _WindowsProbe(monkeypatch)
    probe.denied_open.add(getattr(probe, which))
    assert probe.read() is None
    assert probe.snapshot_count == (1 if which == "owner" else 2)


def test_windows_setup_open_failure_can_skip_only_a_confirmed_exit(monkeypatch):
    probe = _WindowsProbe(monkeypatch)
    probe.denied_open = {probe.child, probe.grandchild}
    probe.snapshots.append({probe.owner: 1, probe.other: 1})
    assert probe.read() == 100
    assert probe.snapshot_count == 2  # One shared recheck, not one scan per failure.


@pytest.mark.parametrize("exit_code", [0, 37, 259, None])
def test_windows_setup_memory_failure_checks_exit_without_assuming_it(monkeypatch, exit_code):
    probe = _WindowsProbe(monkeypatch)
    probe.denied_memory.add(probe.child)
    if exit_code is None:
        probe.denied_exit.add(probe.child)
    else:
        probe.exit_codes[probe.child] = exit_code
    assert probe.read() == (400 if exit_code in (0, 37) else None)
    assert probe.snapshot_count == 1


@pytest.mark.parametrize("error", [0, 5, 87])
def test_windows_setup_incomplete_enumeration_is_not_end_of_list(monkeypatch, error):
    probe = _WindowsProbe(monkeypatch)
    probe.end_errors[1] = error
    assert probe.read() is None
    assert probe.opened == []
    assert 1 in probe.closed


@pytest.mark.parametrize("stage", [1, 2])
def test_windows_setup_snapshot_failure_is_unknown(monkeypatch, stage):
    probe = _WindowsProbe(monkeypatch)
    probe.snapshot_fail.add(stage)
    if stage == 2:
        probe.denied_open.add(probe.child)
    assert probe.read() is None


def test_descendant_walk_does_not_rescan_inventory_for_each_generation():
    class Parents(dict):
        scans = 0

        def items(self):
            self.scans += 1
            return super().items()

    parents = Parents({pid: pid - 1 for pid in range(1, 3001)})
    assert bootstrap._setup_descendants(parents, 1) == set(range(1, 3001))
    assert parents.scans == 1


def _shell_functions():
    source = (ROOT / "scripts/run_benchmark.sh").read_text()
    return source[source.index("linux_child_memory() {"):source.index('\nif ! command -v curl')]


@pytest.mark.skipif(os.name == "nt" or not shutil.which("sh") or not shutil.which("awk"),
                    reason="POSIX bootstrap uses shell and awk")
@pytest.mark.parametrize("rss, threads, state, expected", [
    ("123 kB", "1", "S", "123"), ("0 kB", "1", "S", "0"),
    (None, "1", "S", None), ("-1 kB", "1", "S", None),
    ("bad kB", "1", "S", None), ("123 unknown", "1", "S", None),
    (None, "1", "Z", "0"), (None, "2", "Z", None), (None, "0", "Z", None),
])
def test_pre_python_linux_probe_rejects_missing_live_memory(tmp_path, rss, threads, state, expected):
    status = tmp_path / "status"
    status.write_text("Pid:\t3000\nPPid:\t2000\nNSpid:\t3000 6\n" +
                      f"State:\t{state}\nThreads:\t{threads}\n" +
                      ("" if rss is None else f"VmRSS:\t{rss}\n"))
    result = subprocess.run(["sh", "-c", _shell_functions() + '\nlinux_child_memory "$@"',
                             "probe", "2000", "6", str(status)], capture_output=True, text=True, timeout=5)
    assert result.returncode == (0 if expected is not None else 1)
    if expected is not None:
        assert result.stdout.strip() == "3000 " + expected


@pytest.mark.skipif(os.name == "nt" or not shutil.which("sh") or not shutil.which("awk"),
                    reason="POSIX bootstrap uses shell and awk")
@pytest.mark.parametrize("missing_first", [False, True])
def test_pre_python_probe_ignores_foreign_exit_but_requires_readable_target(tmp_path, missing_first):
    target, foreign = tmp_path / "target", tmp_path / "foreign"
    target.write_text("Pid:\t3000\nPPid:\t2000\nNSpid:\t3000 6\n"
                      "State:\tS (sleeping)\nThreads:\t1\nVmRSS:\t123 kB\n")
    command = ["sh", "-c", _shell_functions() + '\nlinux_child_memory "$@"', "probe", "2000", "6"]
    paths = [foreign, target] if missing_first else [target, foreign]
    result = subprocess.run(command + list(map(str, paths)), capture_output=True, text=True, timeout=5)
    assert result.returncode == 0 and result.stdout.strip() == "3000 123"
    target.unlink()
    result = subprocess.run(command + list(map(str, paths)), capture_output=True, text=True, timeout=5)
    assert result.returncode == 1 and not result.stdout


@pytest.mark.skipif(os.name == "nt" or not shutil.which("sh") or not shutil.which("awk"),
                    reason="POSIX bootstrap uses shell and awk")
@pytest.mark.parametrize("reading", ["return 1", "printf '\\n'", "printf '1\\n'", "printf '3000 1\\n'"])
def test_pre_python_guard_stops_live_installer_when_probe_is_unknown(tmp_path, reading):
    script = _shell_functions() + f"\nlinux_child_memory() {{ {reading}; }}\n" + '''
lpsd_root=$1; shift
lpsd_system=Linux; lpsd_proc_pid=2000; lpsd_setup_kib=2000000; lpsd_ram_kib=1000000
LPSD_BENCHMARK_DEADLINE_EPOCH=$(($(date +%s) + 8))
bounded_python_install "$@"
'''
    duration = ".25" if reading == "printf '3000 1\\n'" else "10"
    started = time.monotonic()
    result = subprocess.run(["sh", "-c", script, "guard", str(tmp_path), sys.executable,
                             "-c", f"import time; time.sleep({duration})"],
                            capture_output=True, text=True, timeout=5)
    assert (result.returncode == 0) == (duration == ".25")
    assert time.monotonic() - started < 5


@pytest.mark.skipif(os.name == "nt" or not shutil.which("sh") or not shutil.which("awk"),
                    reason="POSIX bootstrap uses shell and awk")
def test_pre_python_guard_rechecks_only_its_verified_visible_pid(tmp_path):
    script = _shell_functions() + '''
linux_child_memory() {
    shift 2
    if [ "$#" -eq 1 ] && [ "$1" = /proc/3000/status ]; then
        printf 'direct\n' >> "$lpsd_root/probes"
    else
        printf 'inventory\n' >> "$lpsd_root/probes"
    fi
    printf '3000 1\n'
}
lpsd_root=$1; shift
lpsd_system=Linux; lpsd_proc_pid=2000; lpsd_setup_kib=2000000; lpsd_ram_kib=1000000
LPSD_BENCHMARK_DEADLINE_EPOCH=$(($(date +%s) + 8))
bounded_python_install "$@"
'''
    subprocess.run(["sh", "-c", script, "guard", str(tmp_path), sys.executable,
                    "-c", "import time; time.sleep(1.25)"], check=True,
                   capture_output=True, text=True, timeout=5)
    probes = (tmp_path / "probes").read_text().splitlines()
    assert probes[0] == "inventory" and len(probes) >= 2
    assert all(probe == "direct" for probe in probes[1:])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Live Linux procfs memory")
@pytest.mark.parametrize("without_nspid", [False, True])
def test_real_setup_live_child_counter_cannot_disappear_from_sum(monkeypatch, without_nspid):
    import select
    # Read the child's own procfs-visible PID. It need not equal Popen.pid
    # when procfs was mounted by an ancestor namespace, and old kernels need
    # not expose NSpid at all. No PID guessing or global PID scan is needed.
    child = """
import time
from pathlib import Path
data = bytearray(16*1024**2)
visible = next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
               if line.startswith('Pid:'))
print(visible, flush=True)
time.sleep(20)
"""
    process = subprocess.Popen([sys.executable, "-I", "-S", "-c", child],
                               stdout=subprocess.PIPE, text=True, start_new_session=True)
    original = Path.read_text
    if without_nspid:
        def old_status(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            return "\n".join(line for line in text.splitlines() if not line.startswith("NSpid:")) + "\n"
        monkeypatch.setattr(Path, "read_text", old_status)
    try:
        assert select.select([process.stdout], [], [], 5.)[0], "Child did not become ready"
        visible = int(process.stdout.readline().strip())
        assert visible > 0 and process.poll() is None
        root = Path("/proc")
        owner = bootstrap._linux_setup_status(root / "self" / "status")["Pid"]
        child_status = root / str(visible) / "status"
        fields = bootstrap._linux_setup_status(child_status)
        assert fields["Pid"] == visible and fields["PPid"] == owner
        assert fields["VmRSS"] >= 16 * 1024 ** 2
        assert bootstrap.setup_process_memory() >= 16 * 1024 ** 2
        current_read = Path.read_text

        def read(path, *args, **kwargs):
            if path == child_status:
                raise PermissionError("controlled live-child denial")
            return current_read(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", read)
            assert bootstrap.setup_process_memory() is None
            assert process.poll() is None
        assert bootstrap.setup_process_memory() >= 16 * 1024 ** 2
    finally:
        process.kill()
        process.wait(timeout=3)
        process.stdout.close()
