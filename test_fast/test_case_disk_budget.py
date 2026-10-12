"""Regression for the live budget of privately owned nested FFT files."""
from pathlib import Path
import shutil
import sys

import pytest

from lpsd_benchmark import resources, runner


def test_inventory_counts_nested_and_legacy_files_only(tmp_path):
    (tmp_path / "input.f64").write_bytes(b"i" * 8)
    work = tmp_path / "fft-work-test"
    work.mkdir()
    (work / "workspace.c128").write_bytes(b"w" * 16)
    (work / "powers.f64").write_bytes(b"p" * 4)
    (work / "unrelated").write_bytes(b"x" * 128)
    other = tmp_path / "unrelated"
    other.mkdir()
    (other / "workspace.c128").write_bytes(b"x" * 128)
    assert runner._owned_case_file_bytes(tmp_path) == 28
    (work / "powers.f64").unlink()
    assert runner._owned_case_file_bytes(tmp_path) == 24
    shutil.rmtree(work)
    assert runner._owned_case_file_bytes(tmp_path) == 8
    (tmp_path / "workspace.c128").write_bytes(b"w" * 16)
    (tmp_path / "powers.f64").write_bytes(b"p" * 4)
    assert runner._owned_case_file_bytes(tmp_path) == 28


def test_nested_sparse_file_enforces_disk_limit_and_reaps_child(tmp_path, monkeypatch):
    processes = []
    popen = runner.subprocess.Popen

    def track(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", track)
    monkeypatch.setattr(runner._LinuxMemoryReader, "read", lambda self: (0, 0))
    monkeypatch.setattr(runner, "_memory_usage", lambda pid: (0, 0))
    script = (
        "from pathlib import Path; import time; "
        f"p = Path({str(tmp_path)!r}) / 'fft-work-child'; p.mkdir(); "
        "f = (p / 'workspace.c128').open('wb'); f.truncate(65536); f.close(); "
        "time.sleep(30)"
    )
    case = {"storage": "disk", "ram_budget_bytes": 1024 ** 3, "disk_budget_bytes": 1024}
    result = runner.execute_case(case, tmp_path, resources.Deadline(5.), 4.,
                                 command=[sys.executable, "-c", script], interval=.02)
    assert result["status"] == "failed"
    assert result["reason"] == "disk_limit"
    assert processes[0].returncode is not None


@pytest.mark.parametrize("location", ["root_file", "nested_file", "fft_directory"])
def test_counter_rejects_links_without_following_them(tmp_path, location):
    case = tmp_path / "case"
    case.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "workspace.c128"
    sentinel.write_bytes(b"outside sentinel")
    if location == "root_file":
        link, target = case / "workspace.c128", sentinel
    elif location == "nested_file":
        work = case / "fft-work-test"
        work.mkdir()
        link, target = work / "powers.f64", sentinel
    else:
        link, target = case / "fft-work-test", outside
    try:
        link.symlink_to(target, target_is_directory=location == "fft_directory")
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links unavailable on this platform")
    with pytest.raises(OSError, match="Expected a (regular|real) owned"):
        runner._owned_case_file_bytes(case)
    assert sentinel.read_bytes() == b"outside sentinel"


def test_counter_tolerates_disappearing_fft_child(tmp_path, monkeypatch):
    (tmp_path / "input.f64").write_bytes(b"i" * 8)
    work = tmp_path / "fft-work-test"
    work.mkdir()
    (work / "workspace.c128").write_bytes(b"w" * 16)
    original = Path.lstat

    def disappear(path, *args, **kwargs):
        if path == work / "workspace.c128" and work.exists():
            shutil.rmtree(work)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", disappear)
    assert runner._owned_case_file_bytes(tmp_path) == 8


def test_counter_preserves_metadata_errors(tmp_path, monkeypatch):
    original = Path.lstat

    def denied(path, *args, **kwargs):
        if path == tmp_path / "input.f64":
            raise PermissionError("metadata unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(PermissionError, match="metadata unavailable"):
        runner._owned_case_file_bytes(tmp_path)


def test_metadata_error_reaps_child_before_propagating(tmp_path, monkeypatch):
    case_directory = tmp_path / "case"
    case_directory.mkdir()
    sentinel = tmp_path / "outside"
    sentinel.write_bytes(b"unchanged")
    processes = []
    popen = runner.subprocess.Popen

    def track(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    def denied(directory):
        raise PermissionError("metadata unavailable")

    monkeypatch.setattr(runner.subprocess, "Popen", track)
    monkeypatch.setattr(runner, "_owned_case_file_bytes", denied)
    monkeypatch.setattr(runner._LinuxMemoryReader, "read", lambda self: (0, 0))
    monkeypatch.setattr(runner, "_memory_usage", lambda pid: (0, 0))
    case = {"storage": "disk", "ram_budget_bytes": 1024 ** 3, "disk_budget_bytes": 1024}
    with pytest.raises(PermissionError, match="metadata unavailable"):
        runner.execute_case(case, case_directory, resources.Deadline(5.), 4.,
                            command=[sys.executable, "-c", "import time; time.sleep(30)"])
    assert processes[0].returncode is not None
    assert sentinel.read_bytes() == b"unchanged"
