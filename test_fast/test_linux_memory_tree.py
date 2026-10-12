"""Linux process-tree probes with real missing/unreadable procfs evidence."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from lpsd_benchmark import resources, runner


def _proc_tree(tmp_path, *, children_files=True):
    relationships = {2000: 1, 3000: 2000, 3100: 3000, 3200: 3100, 4000: 1}
    memory = {2000: (0, 0), 3000: (100, 60), 3100: (200, 120),
              3200: (300, 180), 4000: (9000, 8000)}
    inner = {2000: 5, 3000: 6, 3100: 7, 3200: 8, 4000: 99}
    for pid, parent in relationships.items():
        directory = tmp_path / str(pid)
        task = directory / "task" / str(pid)
        task.mkdir(parents=True)
        rss, private = memory[pid]
        (directory / "status").write_text(
            f"Pid:\t{pid}\nPPid:\t{parent}\nNSpid:\t{pid} {inner[pid]}\n"
            f"State:\tS (sleeping)\nVmRSS:\t{rss} kB\nRssAnon:\t{private} kB\n")
        if children_files:
            children = [child for child, owner in relationships.items() if owner == pid]
            (task / "children").write_text(" ".join(map(str, children)))
    (tmp_path / "self").mkdir()
    (tmp_path / "self" / "status").write_bytes((tmp_path / "2000" / "status").read_bytes())
    return runner._LinuxMemoryReader(6, proc_root=tmp_path)


@pytest.mark.parametrize("children_files", [True, False])
def test_complete_nested_tree_uses_at_most_one_fallback_inventory_per_probe(
        tmp_path, monkeypatch, children_files):
    reader = _proc_tree(tmp_path, children_files=children_files)
    inventories, status_reads = [], []
    original_snapshot = runner._linux_process_snapshot
    original_status = runner._linux_status

    def inventory(root):
        inventories.append(root)
        return original_snapshot(root)

    def status(path):
        status_reads.append(path)
        return original_status(path)

    monkeypatch.setattr(runner, "_linux_process_snapshot", inventory)
    monkeypatch.setattr(runner, "_linux_status", status)
    for _ in range(2):
        inventories.clear()
        status_reads.clear()
        assert reader.read() == (600 * 1024, 360 * 1024)
        assert reader.visible_pid == 3000
        assert len(inventories) == (0 if children_files else 1)
        if not children_files:
            # Descendants reuse the inventory instead of starting another
            # scan/read per process. The already-resolved owner can have one
            # initial status read before discovering its missing children file.
            assert status_reads.count(tmp_path / "3000" / "status") <= 2
            for pid in (3100, 3200):
                assert status_reads.count(tmp_path / str(pid) / "status") == 1


@pytest.mark.parametrize("children_files", [True, False])
def test_existing_unreadable_child_invalidates_known_parent_bytes(tmp_path, monkeypatch, children_files):
    reader = _proc_tree(tmp_path, children_files=children_files)
    assert reader.read() == (600 * 1024, 360 * 1024)
    denied = tmp_path / "3100" / "status"
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == denied:
            raise PermissionError("controlled child-status denial")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", read)
        assert reader.read() == (None, None)
        assert denied.is_file()
    assert reader.read() == (600 * 1024, 360 * 1024)


@pytest.mark.parametrize("children_files", [True, False])
@pytest.mark.parametrize("replacement", [None, "-1", "unreadable"])
def test_live_child_missing_or_invalid_rss_is_not_zero(tmp_path, children_files, replacement):
    reader = _proc_tree(tmp_path, children_files=children_files)
    path = tmp_path / "3100" / "status"
    text = path.read_text()
    line = "VmRSS:\t200 kB\n"
    path.write_text(text.replace(line, "" if replacement is None else f"VmRSS:\t{replacement} kB\n"))
    assert reader.read() == (None, None)


def test_partial_task_children_read_is_unavailable_without_global_scan(tmp_path, monkeypatch):
    reader = _proc_tree(tmp_path)
    denied = tmp_path / "3000" / "task" / "3001" / "children"
    denied.parent.mkdir()
    denied.write_text("3100")
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == denied:
            raise PermissionError("controlled second-task denial")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(runner, "_linux_process_snapshot",
                        lambda root: pytest.fail("A denied children read must not become an unsupported-kernel fallback"))
    assert reader.read() == (None, None)


@pytest.mark.parametrize("children_files", [True, False])
def test_missing_status_in_existing_process_is_not_an_exit(tmp_path, children_files):
    reader = _proc_tree(tmp_path, children_files=children_files)
    (tmp_path / "3100" / "status").unlink()
    assert reader.read() == (None, None)


@pytest.mark.parametrize("children_files", [True, False])
def test_leaf_disappearing_during_probe_is_a_verified_exit_race(tmp_path, monkeypatch, children_files):
    reader = _proc_tree(tmp_path, children_files=children_files)
    vanished = tmp_path / "3200" / "status"
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == vanished:
            shutil.rmtree(vanished.parent)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert reader.read() == (300 * 1024, 180 * 1024)


@pytest.mark.parametrize("children_files", [True, False])
def test_verified_zombie_without_memory_counters_is_not_a_live_gap(tmp_path, children_files):
    reader = _proc_tree(tmp_path, children_files=children_files)
    (tmp_path / "3200" / "status").write_text(
        "Pid:\t3200\nPPid:\t3100\nNSpid:\t3200 8\nState:\tZ (zombie)\nThreads:\t1\n")
    assert reader.read() == (300 * 1024, 180 * 1024)


@pytest.mark.parametrize("children_files", [True, False])
@pytest.mark.parametrize("threads", [None, "2", "0", "invalid"])
def test_zombie_group_leader_cannot_hide_live_or_unverified_threads(tmp_path, children_files, threads):
    reader = _proc_tree(tmp_path, children_files=children_files)
    thread_line = "" if threads is None else f"Threads:\t{threads}\n"
    (tmp_path / "3200" / "status").write_text(
        "Pid:\t3200\nPPid:\t3100\nNSpid:\t3200 8\nState:\tZ (zombie)\n" + thread_line)
    assert reader.read() == (None, None)


def test_disappearing_thread_does_not_discard_other_task_children(tmp_path, monkeypatch):
    reader = _proc_tree(tmp_path)
    vanished = tmp_path / "3000" / "task" / "3001" / "children"
    vanished.parent.mkdir()
    vanished.write_text("")
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == vanished:
            shutil.rmtree(vanished.parent)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert reader.read() == (600 * 1024, 360 * 1024)


def test_older_kernel_without_anonymous_counter_keeps_conservative_rss(tmp_path):
    reader = _proc_tree(tmp_path, children_files=False)
    path = tmp_path / "3100" / "status"
    path.write_text(path.read_text().replace("RssAnon:\t120 kB\n", ""))
    assert reader.read() == (600 * 1024, 440 * 1024)


def test_fallback_ignores_non_utf8_bytes_in_unrelated_process_name(tmp_path):
    reader = _proc_tree(tmp_path, children_files=False)
    path = tmp_path / "4000" / "status"
    path.write_bytes(b"Name:\tprocess-\xff\xfe\n" + path.read_bytes())
    assert reader.read() == (600 * 1024, 360 * 1024)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Live Linux procfs process tree")
def test_real_live_grandchild_is_counted_and_unreadable_status_fails_closed(tmp_path, monkeypatch):
    ready = tmp_path / "child-ready"
    child_script = ("import os,time,pathlib; memory=bytearray(64*1024**2); "
                    f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)")
    parent_script = ("import subprocess,sys,time; "
                     f"child=subprocess.Popen([sys.executable,'-c',{child_script!r}]); time.sleep(30)")
    process = subprocess.Popen([sys.executable, "-c", parent_script], start_new_session=True)
    try:
        deadline = time.monotonic() + 8.
        while not ready.is_file() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.is_file(), "The owned allocation child did not become ready"
        reader = runner._LinuxMemoryReader(process.pid)
        full = reader.read()
        assert full[0] is not None and full[0] >= 64 * resources.MIB
        # Find the actual visible procfs PID using the verified relationship,
        # independently of the namespace-local PID returned by Popen.
        snapshot = runner._linux_process_snapshot(Path("/proc"))
        assert snapshot is not None
        children = snapshot[1][reader.visible_pid]
        assert len(children) == 1
        child_path = Path("/proc") / str(next(iter(children))) / "status"
        before = runner._linux_status(child_path)
        assert before["VmRSS"] >= 64 * resources.MIB
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path == child_path:
                raise PermissionError("controlled live-child denial")
            return original(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", read)
            assert reader.read() == (None, None)
            assert process.poll() is None
        after = runner._linux_status(child_path)
        assert after["VmRSS"] >= 64 * resources.MIB
        assert reader.read()[0] >= 64 * resources.MIB
    finally:
        runner.terminate_tree(process)
    assert process.returncode is not None
