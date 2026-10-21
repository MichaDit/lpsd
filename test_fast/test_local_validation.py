"""Local launcher contracts; these tests do not stand in for a native suite."""
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

SPEC = importlib.util.spec_from_file_location("local_validation", Path(__file__).resolve().parents[1]/"scripts/validate_local.py")
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def test_junit_does_not_count_skips_or_xfail_as_passes(tmp_path):
    report = tmp_path / "test.xml"
    report.write_text('<testsuites><testsuite><testcase/><testcase><failure/></testcase>'
        '<testcase><error/></testcase><testcase><skipped/></testcase>'
        '<testcase><skipped type="pytest.xfail"/></testcase></testsuite></testsuites>')
    assert validation.junit_counts(report) == dict(tests=5, passed=1, failed=1, errors=1, skipped=1, xfailed=1)


def test_empty_and_malformed_junit(tmp_path):
    report = tmp_path / "test.xml"
    report.write_text('<testsuites/>')
    assert validation.junit_counts(report)["tests"] == 0
    report.write_text('<broken')
    with pytest.raises(ET.ParseError):
        validation.junit_counts(report)


@pytest.mark.parametrize("code", [0, 7])
def test_stage_exit_is_not_reinterpreted(tmp_path, code):
    log = tmp_path / "stage.log"
    result = validation.run_stage([sys.executable, "-I", "-S", "-c", f"print('local'); raise SystemExit({code})"],
                                   tmp_path, log, 5)
    assert result["returncode"] == code and not result["timed_out"]
    assert result["status"] == ("passed" if code == 0 else "failed")
    assert log.read_text().strip() == "local"


def test_stage_timeout_is_a_failure_and_log_is_not_overwritten(tmp_path):
    log = tmp_path / "stage.log"
    command = [sys.executable, "-I", "-S", "-c", "import time; print('started',flush=True); time.sleep(10)"]
    result = validation.run_stage(command, tmp_path, log, .2)
    assert result["timed_out"] and result["status"] == "failed"
    assert result["returncode"] != 0 and result["wall_seconds"] < 5
    previous = log.read_bytes()
    with pytest.raises(FileExistsError):
        validation.run_stage(command, tmp_path, log, .2)
    assert log.read_bytes() == previous


@pytest.mark.parametrize("timeout", ["0", "-1", "86401"])
def test_bad_deadline_is_rejected_before_work(timeout):
    with pytest.raises(SystemExit) as error:
        validation.main(["--stage-timeout", timeout])
    assert error.value.code == 2


def test_patch_only_directory_is_not_called_a_full_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as error:
        validation.main(["--output-parent", str(tmp_path / "reports")])
    assert error.value.code == 2 and not list(tmp_path.iterdir())


def test_closed_mapping_failure_keeps_original_error_and_junit(tmp_path, monkeypatch):
    # Exercise the real launcher/reporting path with tiny no-op builds.
    files = {
        "build_native.py": "", "lpsd_fast/__init__.py": "", "lpsd_fast/build.py": "",
        "test_fast/conftest.py": "", "test/test_lpsd.py": "",
        "test_fast/test_mapping.py": '''import numpy as np
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass

def failing_reader(values):
    try:
        raise AssertionError("deliberate error before cleanup")
    finally:
        values._mmap.close()

def test_mapping(tmp_path):
    values = np.memmap(tmp_path / "input.f64", mode="w+", dtype=np.float64, shape=(4,))
    failing_reader(values)
''',
    }
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    assert validation.main(["--output-parent", str(reports), "--stage-timeout", "30"]) == 1
    summary_path, = reports.glob("*/summary.json")
    summary = json.loads(summary_path.read_text())
    stage = summary["stages"][-1]
    assert summary["status"] == "failed" and stage["name"] == "regressions"
    assert stage["returncode"] == 1 and not stage["timed_out"]
    assert stage["counts"] == dict(tests=1, passed=0, failed=1, errors=0, skipped=0, xfailed=0)
    assert "deliberate error before cleanup" in (summary_path.parent / "regressions.log").read_text()
