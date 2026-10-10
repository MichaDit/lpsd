# SPDX-License-Identifier: GPL-3.0-or-later
"""Parent-process reporting gates; no FFTW plan or signal array is created."""
import json
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
