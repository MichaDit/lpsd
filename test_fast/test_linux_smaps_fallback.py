"""Old-kernel anonymous-RSS fallback, with real owned-process guard tests.

All procfs parsing is read-only. Synthetic proc trees exercise failures;
Linux integration tests allocate bounded anonymous and mapped file memory.
No ARM, Android, Windows or full numerical campaign is simulated as a pass.
"""
from __future__ import annotations

import ast
import ctypes
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 ** 2


def reader_namespace(path=None):
    path = path or ROOT / "lpsd_benchmark/runner.py"
    names = {"_linux_status", "_linux_path_missing", "_linux_children", "_linux_process_snapshot",
             "_read_smaps_anonymous", "_linux_smaps_anonymous", "_MemorySample", "_LinuxMemoryReader",
             "_memory_usage", "_owned_case_file_bytes", "terminate_tree", "execute_case"}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    # Baseline readers deliberately have no smaps helpers.
    assert len(selected) in (len(names), len(names) - 2)
    namespace = dict(Path=Path, os=os, sys=sys, ctypes=ctypes, stat=stat,
                     subprocess=subprocess, signal=signal, time=time, json=json,
                     ROOT=ROOT, MAX_LOCAL_BYTES=16 * MIB,
                     _LINUX_CHILDREN_UNSUPPORTED=object(), _MEMORY_MEASUREMENT_GRACE_S=1.)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def reader():
    return reader_namespace()


def smaps(rss=120, anonymous=80, first=0x100000, stop=0x400000):
    return (f"{first:x}-{stop:x} rw-p 00000000 00:00 0 /PRIVATE/path\n"
            f"Rss: {rss} kB\nPss: 1 kB\nPrivate_Dirty: 2 kB\n"
            f"Anonymous: {anonymous} kB\nAnonHugePages: {anonymous} kB\n"
            "Swap: 8192 kB\nVmFlags: rd wr mr mw me ac\n").encode()


def test_smaps_uses_anonymous_not_pss_dirty_or_hugepage_double_count(reader, tmp_path):
    path = tmp_path / "smaps"
    path.write_bytes(smaps() + smaps(300, 200, 0x800000, 0x1000000))
    assert reader["_read_smaps_anonymous"](path) == 280 * 1024


@pytest.mark.parametrize("payload", [
    b"", b"Anonymous: 0 kB\n", smaps().replace(b"Anonymous: 80 kB\n", b""),
    smaps().replace(b"Rss: 120 kB\n", b""),
    smaps().replace(b"Anonymous: 80 kB\n", b"Anonymous: 80 kB\nAnonymous: 80 kB\n"),
    smaps().replace(b"Rss: 120 kB\n", b"Rss: 120 kB\nRss: 120 kB\n"),
    smaps().replace(b"Anonymous: 80 kB", b"Anonymous: -1 kB"),
    smaps().replace(b"Anonymous: 80 kB", b"Anonymous: nan kB"),
    smaps().replace(b"Anonymous: 80 kB", b"Anonymous: 80 MB"),
    smaps().replace(b"Anonymous: 80 kB", b"Anonymous: 121 kB"),
    smaps().replace(b"Anonymous: 80 kB", b"Anonymous: 1125899906842624 kB"),
    smaps() + smaps(), smaps().replace(b"100000-400000", b"400000-100000"),
    smaps().replace(b"100000-400000", b"broken-400000"), smaps()[:-1],
    smaps() + b"2000000-3000000 rw-p 00000000 00:00 0\nRss: 0 kB\n",
])
def test_incomplete_or_malformed_optional_counter_is_unknown(reader, tmp_path, payload):
    path = tmp_path / "smaps"
    path.write_bytes(payload)
    assert reader["_read_smaps_anonymous"](path) is None


@pytest.mark.parametrize("limit", [0, -1, 1, 64, None, 3.5, True])
def test_optional_reader_has_a_hard_input_byte_budget(reader, tmp_path, limit):
    path = tmp_path / "smaps"
    path.write_bytes(smaps())
    assert reader["_read_smaps_anonymous"](path, max_bytes=limit) is None


def test_zero_is_measured_only_in_a_complete_mapping(reader, tmp_path):
    path = tmp_path / "smaps"
    path.write_bytes(smaps(0, 0))
    assert reader["_read_smaps_anonymous"](path) == 0


def test_overlong_line_does_not_allocate_the_rest_of_the_file(reader, tmp_path, monkeypatch):
    reads = []
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def readline(self, size):
            assert size == 8193
            reads.append(size)
            return b"x" * size  # A virtual oversized line, not a large allocation.
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: Stream())
    assert reader["_read_smaps_anonymous"](tmp_path / "smaps") is None
    assert reads == [8193]


def test_rollup_is_preferred_and_full_smaps_is_not_opened(reader, tmp_path, monkeypatch):
    (tmp_path / "smaps_rollup").write_bytes(smaps(200, 100))
    original = Path.open
    def opened(path, *args, **kwargs):
        assert path.name != "smaps"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    assert reader["_linux_smaps_anonymous"](tmp_path) == 100 * 1024


@pytest.mark.parametrize("mode", ["missing", "denied", "malformed"])
def test_full_smaps_recovers_an_unusable_rollup(reader, tmp_path, monkeypatch, mode):
    (tmp_path / "smaps").write_bytes(smaps(200, 100))
    if mode == "malformed":
        (tmp_path / "smaps_rollup").write_bytes(b"Anonymous: 0 kB\n")
    if mode == "denied":
        original = Path.open
        def opened(path, *args, **kwargs):
            if path.name == "smaps_rollup": raise PermissionError("PRIVATE")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", opened)
    assert reader["_linux_smaps_anonymous"](tmp_path) == 100 * 1024


def proc_tree(root, *, children_files=True, anonymous=True):
    for pid, ppid, rss, local in ((2000, 1, 100, os.getpid()), (3000, 2000, 200, 8),
                                   (3100, 3000, 300, 9), (4000, 1, 9000, 10)):
        directory = root / str(pid)
        task = directory / "task" / str(pid)
        task.mkdir(parents=True)
        if children_files:
            (task / "children").write_text({2000: "3000", 3000: "3100"}.get(pid, ""))
        text = (f"Pid: {pid}\nPPid: {ppid}\nNSpid: {pid} {local}\nState: S (sleeping)\n"
                f"Threads: 1\nVmRSS: {rss} kB\n")
        if anonymous:
            text += f"RssAnon: {rss // 2} kB\n"
        (directory / "status").write_text(text)
    (root / "self").mkdir()
    (root / "self/status").write_bytes((root / "2000/status").read_bytes())


@pytest.mark.parametrize("children_files", [False, True])
def test_old_kernel_tree_reads_only_owned_smaps(reader, tmp_path, monkeypatch, children_files):
    proc_tree(tmp_path, children_files=children_files, anonymous=False)
    for pid, rss, anon in ((3000, 200, 40), (3100, 300, 60)):
        (tmp_path / str(pid) / "smaps").write_bytes(smaps(rss, anon))
    original = Path.open
    def opened(path, *args, **kwargs):
        if path.name.startswith("smaps"):
            assert path.parent.name in ("3000", "3100")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    sample = reader["_LinuxMemoryReader"](8, tmp_path).read()
    assert sample == (500 * 1024, 100 * 1024)
    assert sample.diagnostics["rss_fallback_processes"] == 0
    assert sample.diagnostics["private_metric"] == "anonymous_rss"
    assert sample.diagnostics["processes"] == 2
    assert "PRIVATE" not in json.dumps(sample.diagnostics)


def test_status_counter_fast_path_never_reads_smaps(reader, tmp_path):
    proc_tree(tmp_path)
    reader["_linux_smaps_anonymous"] = lambda path: pytest.fail("unnecessary page-table walk")
    assert reader["_LinuxMemoryReader"](8, tmp_path).read() == (500 * 1024, 250 * 1024)


@pytest.mark.parametrize("mixed", [False, True])
def test_missing_optional_counters_preserve_conservative_rss_fallback(reader, tmp_path, mixed):
    proc_tree(tmp_path, anonymous=False)
    if mixed:
        (tmp_path / "3000/smaps").write_bytes(smaps(200, 40))
    sample = reader["_LinuxMemoryReader"](8, tmp_path).read()
    assert sample == (500 * 1024, (340 if mixed else 500) * 1024)
    assert sample.diagnostics["rss_fallback_processes"] == (1 if mixed else 2)
    assert sample.diagnostics["private_metric"] == ("anonymous_rss_with_rss_fallback" if mixed else "rss_fallback")


def test_optional_counter_does_not_hide_an_unreadable_owned_status(reader, tmp_path):
    proc_tree(tmp_path, anonymous=False)
    (tmp_path / "3100/status").unlink()
    assert reader["_LinuxMemoryReader"](8, tmp_path).read() == (None, None)


_GUARD_CHILD = r'''
import json, mmap, sys, time
from pathlib import Path
MiB=1024**2
work, result, anon_mib = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
a = bytearray(anon_mib * MiB)
for offset in range(0, len(a), 4096): a[offset] = 1
with work.open("w+b") as stream:
    stream.truncate(64*MiB)
    with mmap.mmap(stream.fileno(), 64*MiB) as mapped:
        for offset in range(0, len(mapped), 4096): mapped[offset] = 1
        time.sleep(.6)
        result.write_text(json.dumps({"status":"completed", "reason":""}))
'''


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Real Linux owned-process memory guard")
@pytest.mark.parametrize("storage,anon_mib,expected", [("disk", 16, "completed"), ("disk", 128, "failed"),
                                                       ("ram", 16, "failed")])
def test_real_old_status_guard_distinguishes_mappings_from_anonymous_limit(
        reader, tmp_path, monkeypatch, storage, anon_mib, expected):
    # Omit only the optional status field; read real owned-process smaps and
    # execute the actual supervisor/termination code, with the unchanged cap.
    original = Path.read_text
    def text(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path.name == "status":
            value = "\n".join(line for line in value.splitlines() if not line.startswith("RssAnon:")) + "\n"
        return value
    monkeypatch.setattr(Path, "read_text", text)
    if storage == "disk" and reader["_linux_smaps_anonymous"](Path("/proc/self")) is None:
        pytest.skip("Optional owned-process smaps interface is denied on this host")
    if storage == "ram":
        reader["_linux_smaps_anonymous"] = lambda path: pytest.fail("RAM guard does not need anonymous RSS")
    case = {"n": 4096, "method": "fftw", "storage": storage,
            "ram_budget_bytes": 64 * MIB, "disk_budget_bytes": 96 * MIB}
    result = reader["execute_case"](case, tmp_path, SimpleNamespace(end=time.monotonic()+10), 8,
        command=[sys.executable, "-I", "-S", "-c", _GUARD_CHILD, str(tmp_path / "input.f64"),
                 str(tmp_path / "result.json"), str(anon_mib)], interval=.02)
    assert result["status"] == expected, result
    guard = result["memory_guard"]
    assert guard["limit_bytes"] == 64 * MIB
    assert guard["last_metric"] == ("anonymous_rss" if storage == "disk" else "rss")
    assert guard["rss_fallback_samples"] == 0
    assert guard["last_sample_tree_complete"] is True
    if expected == "completed":
        assert result["peak_rss_bytes"] > 64 * MIB
        assert guard["peak_bytes"] <= 64 * MIB and guard["termination"] == "none"
    else:
        assert result["reason"] == "memory_limit"
        assert guard["trigger_bytes"] > 64 * MIB
        assert guard["trigger_metric"] == ("anonymous_rss" if storage == "disk" else "rss")
        assert guard["termination"] == "limit_exceeded"


def test_reader_can_disable_unused_page_table_walk(reader, tmp_path):
    proc_tree(tmp_path, anonymous=False)
    instance = reader["_LinuxMemoryReader"](8, tmp_path)
    instance.smaps_fallback = False
    reader["_linux_smaps_anonymous"] = lambda path: pytest.fail("unused anonymous counter")
    sample = instance.read()
    assert sample == (500 * 1024, 500 * 1024)
    assert sample.diagnostics["private_metric"] == "rss_fallback"
