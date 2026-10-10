# SPDX-License-Identifier: GPL-3.0-or-later
"""Parent-process reporting gates; no FFTW plan or signal array is created."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmarks import bench_fftw_disk as driver


def test_input_setup_failure_finishes_parent_report_and_cleans_partial_input(tmp_path, monkeypatch):
    work = tmp_path / "work"
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "bench_fftw_disk.py", "--sizes", "17", "--methods", "fftw",
        "--fftw-library", str(tmp_path / "unused_fftw"),
        "--threads-library", str(tmp_path / "unused_fftw_threads"),
        "--work-dir", str(work), "--output", str(output),
    ])
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda path: SimpleNamespace(free=1 << 40))
    monkeypatch.setattr(driver, "read_file", lambda path: "memory events fixture")

    def fail_input(path, *args):
        path.write_bytes(b"partial input")
        raise RuntimeError("input setup failed")

    def forbid_child(*args, **kwargs):
        pytest.fail("Input failure must not launch a child or FFTW calculation")

    monkeypatch.setattr(driver, "input_file", fail_input)
    monkeypatch.setattr(driver.subprocess, "run", forbid_child)
    with pytest.raises(RuntimeError, match="input setup failed"):
        driver.main()

    report = json.loads(output.read_text())
    assert report["status"] == "incomplete"
    assert report["parent_error"] == {"error_type": "RuntimeError", "error": "input setup failed"}
    assert report["finished_utc"] >= report["started_utc"]
    assert report["memory_events_after"] == "memory events fixture"
    assert report["inputs"] == [] and report["jobs"] == []
    assert not (work / "input_17.f64").exists()



@pytest.mark.parametrize("checkpoint_kind", ("running", "missing", "invalid"))
def test_parent_timeout_updates_child_preserves_history_and_forwards_budget(
        tmp_path, monkeypatch, checkpoint_kind):
    work = tmp_path / "work"
    output = tmp_path / "report.json"
    jobs = tmp_path / "report_jobs"
    jobs.mkdir()
    job_path = jobs / "lpsd_17.json"
    previous_parent = b'{"historical_parent": true}\n'
    previous_job = b'{"historical_child": true}\n'
    previous_log = b'historical child log\n'
    output.write_bytes(previous_parent)
    job_path.write_bytes(previous_job)
    job_path.with_suffix(".log").write_bytes(previous_log)
    monkeypatch.setattr(sys, "argv", [
        "bench_fftw_disk.py", "--sizes", "17", "--methods", "lpsd",
        "--fftw-library", str(tmp_path / "unused_fftw"),
        "--threads-library", str(tmp_path / "unused_fftw_threads"),
        "--work-dir", str(work), "--output", str(output),
        "--timeout-seconds", "15000", "--lpsd-progress-interval-s", "7",
        "--power-storage", "workspace",
    ])
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda path: SimpleNamespace(free=1 << 40))
    monkeypatch.setattr(driver, "read_file", lambda path: "memory events fixture")

    def make_input(path, *args):
        path.write_bytes(b"small input fixture")
        return {"sha256": "fixture-sha", "samples": 17}

    raw_checkpoint = None

    def timeout_child(command, **kwargs):
        nonlocal raw_checkpoint
        assert kwargs["timeout"] == 15000
        assert float(command[command.index("--timeout-seconds") + 1]) == 15000
        assert float(command[command.index("--lpsd-progress-interval-s") + 1]) == 7
        assert command[command.index("--power-storage") + 1] == "workspace"
        child_output = Path(command[command.index("--output") + 1])
        assert not child_output.exists()
        if checkpoint_kind == "running":
            raw_checkpoint = (json.dumps({
                "status": "running", "n": 17, "method": "lpsd",
                "settings": {"timeout_seconds": 15000, "lpsd_progress_interval_s": 7},
                "progress": [{"phase": "workspace_and_plan_s", "wall_s": .01}],
                "lpsd_completed_frequencies": [{"j": 0, "L": 17, "K": 1,
                                                 "c_kernel_s": .02}],
            }) + "\n").encode()
        elif checkpoint_kind == "invalid":
            raw_checkpoint = b'{"status": "running", interrupted'
        if raw_checkpoint is not None:
            child_output.write_bytes(raw_checkpoint)
        abandoned = work / "disk_lpsd_17_abandoned"
        abandoned.mkdir()
        (abandoned / "workspace.c128").write_bytes(b"small abandoned fixture")
        kwargs["stdout"].write("timeout fixture; no numerical calculation\n")
        raise driver.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(driver, "input_file", make_input)
    monkeypatch.setattr(driver.subprocess, "run", timeout_child)
    with pytest.raises(SystemExit) as caught:
        driver.main()
    assert caught.value.code == 1
    report = json.loads(output.read_text())
    saved_job = json.loads(job_path.read_text())
    assert report["status"] == "incomplete"
    assert report["jobs"] == [saved_job]
    assert saved_job["status"] == "timeout"
    assert saved_job["timeout_seconds"] == saved_job["controller_timeout_seconds"] == 15000
    assert saved_job["controller_process_elapsed_wall_s"] >= 0
    assert saved_job["controller_finished_utc"]
    assert saved_job["child_exit_code_is_timeout_sentinel"]
    assert "input hashes" in saved_job["timeout_scope"]
    assert "not a completed-call timing" in saved_job["timeout_scope"]
    assert "timing" not in saved_job  # Never invent a call duration from the timeout.
    assert Path(report["previous_report"]["path"]).read_bytes() == previous_parent
    assert {Path(row["path"]).read_bytes() for row in saved_job["previous_artifacts"]} == {
        previous_job, previous_log}
    archived = saved_job["interrupted_checkpoint_archive"]
    if raw_checkpoint is None:
        assert archived is None
    else:
        assert Path(archived["path"]).read_bytes() == raw_checkpoint
        assert driver.sha256(archived["path"]) == archived["sha256"]
    if checkpoint_kind == "running":
        assert saved_job["checkpoint_status_before_timeout"] == "running"
        assert saved_job["settings"]["timeout_seconds"] == 15000
        assert saved_job["lpsd_completed_frequencies"][0]["c_kernel_s"] == .02
    elif checkpoint_kind == "invalid":
        assert saved_job["checkpoint_read_error"]
    assert not (work / "input_17.f64").exists()
    assert not list(work.glob("disk_lpsd_17_*"))


@pytest.mark.parametrize("methods,power_storage,expected", [
    (("lpsd",), "external", 24_000_000_000),
    (("lpsd",), "workspace", 24_000_000_000),
    (("fftw", "matched", "lpsd"), "workspace", 24_000_000_000),
    (("fftw", "matched", "lpsd"), "external", 28_000_000_008),
])
def test_billion_sample_file_preflight_counts_aliases_once(methods, power_storage, expected):
    assert driver.large_file_bytes(1_000_000_000, methods, power_storage) == expected
    if power_storage == "workspace" or methods == ("lpsd",):
        assert expected + 512 * 1024**2 < 25_000_000_000
