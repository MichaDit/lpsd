"""Small stdlib-only regression gates for owned FFT workspace accounting."""
from pathlib import Path
import sys

import pytest

from lpsd_benchmark import resources, runner


def _file(path, size):
    path.write_bytes(b"x" * size)


def test_counts_only_root_and_immediate_owned_fft_files(tmp_path):
    child = tmp_path / "fft-work-test"
    child.mkdir()
    _file(tmp_path / "input.f64", 8)
    _file(child / "workspace.c128", 16)
    _file(child / "powers.f64", 4)
    _file(tmp_path / "unrelated.bin", 100)
    other = tmp_path / "unrelated"
    other.mkdir()
    _file(other / "workspace.c128", 100)
    nested = child / "fft-work-deeper"
    nested.mkdir()
    _file(nested / "workspace.c128", 100)
    assert runner._owned_case_file_bytes(tmp_path) == 28
    (child / "powers.f64").unlink()
    assert runner._owned_case_file_bytes(tmp_path) == 24
    (child / "workspace.c128").unlink()
    assert runner._owned_case_file_bytes(tmp_path) == 8
    _file(tmp_path / "workspace.c128", 16)
    _file(tmp_path / "powers.f64", 4)
    assert runner._owned_case_file_bytes(tmp_path) == 28


@pytest.mark.parametrize("folder", [False, True])
def test_owned_names_do_not_follow_links(tmp_path, folder):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "workspace.c128"
    sentinel.write_bytes(b"outside must remain unchanged")
    case = tmp_path / "case"
    case.mkdir()
    link = case / ("fft-work-link" if folder else "workspace.c128")
    try:
        link.symlink_to(outside if folder else sentinel, target_is_directory=folder)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Link creation unavailable: {type(error).__name__}")
    with pytest.raises(ValueError, match="owned case"):
        runner._owned_case_file_bytes(case)
    assert sentinel.read_bytes() == b"outside must remain unchanged"
    assert link.is_symlink()


def test_disappearance_is_tolerated_but_other_stat_errors_propagate(tmp_path, monkeypatch):
    child = tmp_path / "fft-work-race"
    child.mkdir()
    target = child / "workspace.c128"
    _file(target, 16)
    original = Path.lstat

    def disappeared(path, *args, **kwargs):
        if path == target:
            raise FileNotFoundError("simulated cleanup")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", disappeared)
    assert runner._owned_case_file_bytes(tmp_path) == 0

    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError("simulated metadata failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(PermissionError):
        runner._owned_case_file_bytes(tmp_path)


def test_nested_workspace_triggers_disk_limit_and_child_is_reaped(tmp_path, monkeypatch):
    # The child performs no numerical work and allocates only a tiny file.
    # Its long wait is interrupted by the actual live budget check.
    script = (
        "from pathlib import Path; import sys,time; "
        "p=Path(sys.argv[1])/'fft-work-child'; p.mkdir(); "
        "(p/'workspace.c128').write_bytes(b'x'*128); time.sleep(30)"
    )
    spawned = []
    original = runner.subprocess.Popen

    def record_process(*args, **kwargs):
        process = original(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", record_process)
    monkeypatch.setattr(runner._LinuxMemoryReader, "read", lambda self: (1, 1))
    monkeypatch.setattr(runner, "_memory_usage", lambda pid: (1, 1))
    case = {"storage": "disk", "ram_budget_bytes": 1024 ** 3, "disk_budget_bytes": 64,
            "n": 16, "method": "fftw", "target_points": 2}
    report = runner.execute_case(case, tmp_path, resources.Deadline(8), 5,
                                 command=[sys.executable, "-c", script, str(tmp_path)], interval=.02)
    assert report["status"] == "failed" and report["reason"] == "disk_limit"
    assert spawned and spawned[0].returncode is not None
    assert spawned[0].poll() is not None


def test_inventory_failure_still_terminates_owned_child(tmp_path, monkeypatch):
    terminated = []
    original = runner.terminate_tree

    def record_termination(process):
        original(process)
        terminated.append(process)

    def denied(directory):
        raise PermissionError("simulated metadata failure")

    monkeypatch.setattr(runner, "terminate_tree", record_termination)
    monkeypatch.setattr(runner, "_owned_case_file_bytes", denied)
    monkeypatch.setattr(runner._LinuxMemoryReader, "read", lambda self: (1, 1))
    monkeypatch.setattr(runner, "_memory_usage", lambda pid: (1, 1))
    case = {"storage": "disk", "ram_budget_bytes": 1024 ** 3, "disk_budget_bytes": 64}
    with pytest.raises(PermissionError):
        runner.execute_case(case, tmp_path, resources.Deadline(8), 5,
                            command=[sys.executable, "-c", "import time; time.sleep(30)"], interval=.02)
    assert len(terminated) == 1 and terminated[0].returncode is not None
