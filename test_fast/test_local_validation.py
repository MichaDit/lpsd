"""Local launcher contracts; these tests do not stand in for a native suite."""
import importlib.util
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
