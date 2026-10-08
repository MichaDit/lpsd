"""Official-bench wrapper checks with synthetic process output, never FFTW."""
from pathlib import Path
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    return runpy.run_path(str(ROOT / "benchmarks" / "bench_fftw_official.py"))


def fake_process(monkeypatch, runner, stdout, stderr="", returncode=0):
    class Process:
        def communicate(self, timeout=None):
            return stdout, stderr

    process = Process()
    process.returncode = returncode
    monkeypatch.setattr(runner["subprocess"], "Popen", lambda *args, **kwargs: process)


def test_records_official_minimum_and_preserves_raw_output(runner, monkeypatch, tmp_path):
    stdout = "NTHREADS = 8\nplanner time: 5.1 s\n(fft-plan)\n1500 0.0125 5.1\n"
    fake_process(monkeypatch, runner, stdout)
    row = runner["run_case"](["bench"], tmp_path, {}, 1, "case", tmp_path)
    assert row["ok"]
    assert row["reported_thread_counts"] == [8]
    assert row["official_report"] == {
        "normalized_mflops_not_actual_flop_count": 1500,
        "minimum_execution_seconds_per_fft": .0125,
        "setup_seconds": 5.1,
    }
    assert (tmp_path / row["stdout_file"]).read_text() == stdout
    assert (tmp_path / row["stderr_file"]).read_text() == ""


@pytest.mark.parametrize("stdout", (
    "1500 .01 5\n1600 .009 5\n",  # Ambiguous: two reported problems.
    "1500 0 5\n",
    "1500 -1 5\n",
    "1500 .01 -5\n",
    "inf .01 5\n",
))
def test_rejects_ambiguous_or_invalid_timing(runner, monkeypatch, tmp_path, stdout):
    fake_process(monkeypatch, runner, stdout)
    row = runner["run_case"](["bench"], tmp_path, {}, 1, "case", tmp_path)
    assert "official_report" not in row


def test_unknown_option_is_not_a_success(runner, monkeypatch, tmp_path):
    fake_process(monkeypatch, runner, "1500 .01 5\n",
                 stderr="unknown user option: time-limit=5. Ignoring.\n")
    row = runner["run_case"](["bench"], tmp_path, {}, 1, "case", tmp_path)
    assert not row["ok"]
    assert row["option_warning"]


def test_timeout_stops_child_and_retains_partial_log(runner, monkeypatch, tmp_path):
    class Process:
        pid = 12345
        returncode = -9

        def communicate(self, timeout=None):
            if timeout is not None:
                raise runner["subprocess"].TimeoutExpired("bench", timeout)
            return "partial stdout\n", "partial stderr\n"

    process = Process()
    stopped = []
    monkeypatch.setattr(runner["subprocess"], "Popen", lambda *args, **kwargs: process)
    # runpy returns a copy; patch the function's actual globals, not the copy.
    monkeypatch.setitem(runner["run_case"].__globals__, "stop_process", stopped.append)
    row = runner["run_case"](["bench"], tmp_path, {}, 1, "case", tmp_path)
    assert stopped == [process]
    assert row["timed_out"] and not row["ok"]
    assert (tmp_path / row["stdout_file"]).read_text() == "partial stdout\n"


def test_default_is_a_dry_run(runner, monkeypatch, tmp_path, capsys):
    output = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", ["bench_fftw_official.py", "--source", "unused",
                                     "--output-directory", str(output),
                                     "--sizes", "1000000", "--threads", "1"])
    monkeypatch.setattr(runner["subprocess"], "Popen", lambda *args, **kwargs:
                        pytest.fail("Dry run must not launch FFTW"))
    runner["main"]()
    commands = capsys.readouterr().out.splitlines()
    assert len(commands) == 2
    assert "-otimelimit=5.0" in commands[1]
    assert not output.exists()
