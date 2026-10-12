"""Bounded failure evidence, thread selection and actual watchdog counters."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lpsd_benchmark import SCHEMA, privacy, runner, worker
from test_fast.test_benchmark_memory_measurement import _fake_case
from test_fast.test_linux_memory_tree import _proc_tree
from test_fast.test_portable_direct_power import _direct_case, _fake_worker


def _fftw_discovery(monkeypatch, libraries, wheel=()):
    monkeypatch.setattr(worker.ctypes.util, "find_library", lambda key: libraries.get(key))
    monkeypatch.setattr(worker.metadata, "distribution", lambda name: SimpleNamespace(
        files=list(wheel), locate_file=lambda entry: Path("/PRIVATE-WHEEL") / entry))


def test_fftw_uses_available_openmp_runtime_after_pthread_load_failure(monkeypatch):
    _fftw_discovery(monkeypatch, {"fftw3": "core", "fftw3_threads": "libfftw3_threads.so",
                                 "fftw3_omp": "libfftw3_omp.so"})
    calls = []

    def load(core, threads, count):
        calls.append((core, threads, count))
        if threads == "libfftw3_threads.so":
            raise OSError("PRIVATE dependency path")
        return SimpleNamespace(threads=count)

    backend = worker._load_fftw(SimpleNamespace(FFTWLibrary=load), 2)
    assert calls == [("core", None, 1), ("core", "libfftw3_threads.so", 2),
                     ("core", "libfftw3_omp.so", 2)]
    details = backend.threading_diagnostics
    assert details["requested"] == details["actual"] == 2
    assert details["status"] == "enabled" and details["thread_runtime"] == "openmp"
    assert details["failures"] == {"thread_load_failed": 1}
    assert "PRIVATE" not in json.dumps(details)


def test_fftw_tries_working_wheel_before_fallback_without_crossing_core_families(monkeypatch):
    wheel = ("libs/libfftw3-a1.so", "libs/libfftw3_threads-b2.so")
    _fftw_discovery(monkeypatch, {"fftw3": "system-core", "fftw3_threads": "system-threads"}, wheel)
    calls = []

    def load(core, threads, count):
        calls.append((core, threads, count))
        if threads is not None and core == "system-core":
            raise AttributeError("PRIVATE thread symbol")
        return SimpleNamespace(threads=count)

    backend = worker._load_fftw(SimpleNamespace(FFTWLibrary=load), 3)
    assert backend.threads == 3 and backend.threading_diagnostics["main_source"] == "wheel"
    assert backend.threading_diagnostics["thread_source"] == "wheel"
    assert not any(core == "system-core" and thread and "PRIVATE-WHEEL" in thread
                   or "PRIVATE-WHEEL" in core and thread == "system-threads"
                   for core, thread, count in calls)
    assert calls[-1] == (str(Path("/PRIVATE-WHEEL") / wheel[0]), str(Path("/PRIVATE-WHEEL") / wheel[1]), 3)


def test_fftw_records_load_symbol_and_init_failures_before_single_thread_fallback(monkeypatch):
    _fftw_discovery(monkeypatch, {"fftw3_threads": "pthread", "fftw3_omp": "openmp"})
    errors = {"PRIVATE-thread": OSError, "PRIVATE-core": AttributeError,
              "pthread": RuntimeError, "openmp": ValueError}

    def load(core, threads, count):
        if threads is not None:
            raise errors[threads]("PRIVATE path and environment")
        return SimpleNamespace(threads=count)

    details = worker._load_fftw(SimpleNamespace(FFTWLibrary=load), 2,
                               "PRIVATE-core", "PRIVATE-thread").threading_diagnostics
    assert details["status"] == "fallback_single_thread" and details["actual"] == 1
    assert details["requested"] == 2 and details["thread_attempts"] == 4
    assert details["failures"] == {"thread_load_failed": 1, "thread_symbols_missing": 1,
                                   "thread_init_failed": 1, "thread_runtime_rejected": 1}
    assert "PRIVATE" not in json.dumps(details)
    assert privacy._fftw_threading_fields(details) == details


def test_fftw_single_thread_request_does_not_initialize_optional_thread_runtimes(monkeypatch):
    _fftw_discovery(monkeypatch, {"fftw3": "core"})
    calls = []

    def load(core, threads, count):
        calls.append((core, threads, count))
        assert threads is None and count == 1
        return SimpleNamespace(threads=count)

    details = worker._load_fftw(SimpleNamespace(FFTWLibrary=load), 1).threading_diagnostics
    assert calls == [("core", None, 1)]
    assert details["status"] == "single_thread_requested" and details["thread_attempts"] == 0


def test_fftw_explicit_core_is_validated_even_when_its_filename_contains_omp(monkeypatch):
    _fftw_discovery(monkeypatch, {})
    calls = []

    def load(core, threads, count):
        calls.append((core, threads, count))
        return SimpleNamespace(threads=count)

    result = worker._load_fftw(SimpleNamespace(FFTWLibrary=load), 2, explicit="PRIVATE-compiled-core")
    assert calls == [("PRIVATE-compiled-core", None, 1), ("PRIVATE-compiled-core", "PRIVATE-compiled-core", 2)]
    assert result.threading_diagnostics["status"] == "enabled"
    assert result.threading_diagnostics["main_source"] == "explicit"
    assert result.threading_diagnostics["thread_runtime"] == "integrated"


def test_missing_fftw_keeps_structured_dependency_diagnosis(monkeypatch, tmp_path):
    _fftw_discovery(monkeypatch, {"fftw3": "PRIVATE-unloadable"})

    def missing(*args):
        raise OSError("PRIVATE transitive dependency")

    with pytest.raises(worker._FFTWUnavailable) as caught:
        worker._load_fftw(SimpleNamespace(FFTWLibrary=missing), 2)
    evidence = caught.value.diagnostics
    assert evidence["actual"] == 0 and evidence["failures"] == {"main_load_failed": 1}
    _fake_worker(monkeypatch)

    def unavailable(*args):
        raise worker._FFTWUnavailable(evidence)

    monkeypatch.setattr(worker, "_load_fftw", unavailable)
    result = worker.run(_direct_case(), tmp_path)
    assert result["status"] == "skipped" and result["reason"] == "dependency_unavailable"
    assert result["fftw_threading"] == evidence
    assert "PRIVATE" not in json.dumps(result) and not list(tmp_path.iterdir())


@pytest.mark.parametrize("check", ["finite", "psd_nonnegative", "nsd_matches_sqrt_psd",
                                   "frequency_matches_plan", "input_unchanged"])
def test_invalid_result_preserves_all_checks_and_bounded_evidence(tmp_path, monkeypatch, check):
    from benchmarks import bench_fftw_disk as disk
    _fake_worker(monkeypatch)
    pipeline = disk.disk_pipeline

    def altered(*args):
        frame, metadata = pipeline(*args)
        if check == "finite":
            frame.iloc[0, 0] = np.inf
        elif check == "psd_nonnegative":
            frame.iloc[0, 0] = -frame.iloc[0, 0]
            frame.iloc[0, 1] = 0.
        elif check == "nsd_matches_sqrt_psd":
            frame.iloc[0, 1] *= 2
        elif check == "frequency_matches_plan":
            labels = frame.index.to_numpy().copy()
            labels[0] += .001
            frame.index = labels
        else:
            values = args[1]
            changed = np.memmap(values.filename, dtype=np.float64, mode="r+", shape=values.shape)
            changed[0] += 1
            changed.flush()
            changed._mmap.close()
        return frame, metadata

    monkeypatch.setattr(disk, "disk_pipeline", altered)
    case = _direct_case()
    result = worker.run(case, tmp_path)
    assert result["status"] == "failed" and result["reason"] == "invalid_result"
    for key in ("finite", "psd_nonnegative", "nsd_matches_sqrt_psd", "frequency_matches_plan", "input_unchanged"):
        assert type(result[key]) is bool
    assert result[check] is False and result["actual_points"] == 4
    assert result["compute_wall_s"] == 11. and result["phases"]
    assert "_spectrum" not in result
    assert all(len(result[key]) == 64 for key in ("frequency_sha256", "psd_sha256", "nsd_sha256"))
    row = privacy.sanitize({"schema": SCHEMA, "cases": [{**case, **result}]})["cases"][0]
    assert row[check] is False and row["validation"] == result["validation"]
    json.dumps(row, allow_nan=False)
    assert not list(tmp_path.iterdir())


def test_validation_counts_shape_mismatch_without_serializing_arrays():
    from benchmarks import bench_fftw as bf
    values = np.zeros(16)
    f = np.array([1., 2.])
    psd = np.array([1., 4.], dtype=np.float32)
    result = worker._result_validation(bf, values, bf.digest_array(values), f,
                                       psd, np.array([1.], dtype=np.float32), f[:1])
    assert result["finite"] and result["input_unchanged"]
    assert not result["frequency_matches_plan"] and not result["nsd_matches_sqrt_psd"]
    assert result["validation"]["frequency_points"] == 2
    assert result["validation"]["expected_frequency_points"] == result["validation"]["nsd_points"] == 1
    assert "frequency_mismatch_points" not in result["validation"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("children_files", [False, True])
def test_linux_memory_sample_records_complete_tree_and_partial_rss_fallback(tmp_path, children_files):
    reader = _proc_tree(tmp_path, children_files=children_files)
    sample = reader.read()
    assert sample == (600 * 1024, 360 * 1024)
    assert sample.diagnostics == {"source": "proc_children" if children_files else "proc_ppid_scan",
                                  "processes": 3, "private_metric": "anonymous_rss",
                                  "tree_complete": True, "rss_fallback_processes": 0}
    path = tmp_path / "3200" / "status"
    path.write_text(path.read_text().replace("RssAnon:\t180 kB\n", ""))
    sample = reader.read()
    assert sample == (600 * 1024, 480 * 1024)
    assert sample.diagnostics["private_metric"] == "anonymous_rss_with_rss_fallback"
    assert sample.diagnostics["rss_fallback_processes"] == 1


def _sample(rss, private, *, source="proc_children", processes=2, fallback=0, metric="anonymous_rss"):
    return runner._MemorySample(rss, private, source=source, processes=processes,
                                private_metric=metric, rss_fallback_processes=fallback)


def test_guard_reports_actual_anonymous_trigger_not_larger_rss(tmp_path, monkeypatch):
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: _sample(800, 51), budget=50)
    guard = result["memory_guard"]
    assert result["reason"] == "memory_limit" and stopped == [(process, 0.)]
    assert result["peak_rss_bytes"] == 800
    assert guard["last_bytes"] == guard["peak_bytes"] == guard["trigger_bytes"] == 51
    assert guard["limit_bytes"] == 50 and guard["trigger_metric"] == "anonymous_rss"
    assert guard["termination"] == "limit_exceeded" and guard["complete_tree_samples"] == 1
    assert guard["last_sample_tree_complete"] and guard["last_processes"] == guard["peak_processes"] == 2


@pytest.mark.parametrize("high_water, measurement", [(2000, "process_high_water_mark"),
                                                      (900, "process_high_water_mark"), (700, "sampled")])
def test_guard_peak_stays_separate_from_worker_rss_high_water(tmp_path, monkeypatch, high_water, measurement):
    samples = [(100, 20), (900, 40), (600, 30)]
    result, *_ = _fake_case(tmp_path, monkeypatch,
        lambda clock, child: _sample(*samples[int(clock[0] / .25)]), exit_at=.75,
        worker_result={"status": "completed", "reason": "", "peak_rss_bytes": high_water})
    guard = result["memory_guard"]
    assert result["status"] == "completed" and result["peak_rss_bytes"] == max(900, high_water)
    assert result["rss_measurement"] == measurement
    assert guard["peak_bytes"] == 40 and guard["last_bytes"] == 30
    assert guard["peak_metric"] == guard["last_metric"] == "anonymous_rss"
    assert guard["complete_tree_samples"] == 3 and guard["termination"] == "none"
    assert "trigger_bytes" not in guard


def test_guard_does_not_reuse_previous_tree_evidence_for_a_new_plain_tuple(tmp_path, monkeypatch):
    result, *_ = _fake_case(tmp_path, monkeypatch,
        lambda clock, child: _sample(100, 20) if clock[0] == 0 else (90, 25), exit_at=.5)
    guard = result["memory_guard"]
    assert guard["last_bytes"] == guard["peak_bytes"] == 25
    assert guard["complete_tree_samples"] == 1 and guard["peak_processes"] == 2
    assert not any(key in guard for key in ("last_sample_tree_complete", "last_processes", "last_source"))
    assert privacy._memory_guard_fields(guard, result) == guard


@pytest.mark.parametrize("platform,private,metric", [
    ("linux", 60, "anonymous_rss_with_rss_fallback"), ("darwin", None, "rss_fallback")])
def test_guard_identifies_linux_and_macos_rss_fallback_without_relaxing_limit(
        tmp_path, monkeypatch, platform, private, metric):
    sample = _sample(80, private, source="psutil" if platform == "darwin" else "proc_ppid_scan",
                     fallback=1 if platform == "linux" else 0, metric=metric)
    result, *_ = _fake_case(tmp_path, monkeypatch, lambda clock, child: sample,
                            platform=platform, budget=50)
    assert result["reason"] == "memory_limit"
    assert result["live_memory_rss_fallback_samples"] == 1
    guard = result["memory_guard"]
    assert guard["trigger_bytes"] == (60 if platform == "linux" else 80)
    assert guard["trigger_metric"] == metric and guard["limit_bytes"] == 50
    assert guard["rss_fallback_samples"] == guard["complete_tree_samples"] == 1


def test_guard_does_not_accept_worker_claims_when_no_live_probe_occurred(tmp_path, monkeypatch):
    result, *_ = _fake_case(tmp_path, monkeypatch, lambda clock, child: pytest.fail("No live child"),
        exit_at=0., worker_result={"status": "completed", "memory_guard": {
            "peak_bytes": 999, "last_bytes": 999, "limit_bytes": 999, "last_source": "PRIVATE"}})
    assert result["memory_guard"] == {"scope": "process_tree", "measurement": "sampled",
        "limit_bytes": 100, "termination": "none", "complete_tree_samples": 0, "rss_fallback_samples": 0}


def test_diagnostic_schema_and_issue_payload_are_closed_and_idempotent(tmp_path, monkeypatch):
    result, *_ = _fake_case(tmp_path, monkeypatch, lambda clock, child: _sample(80, 51), budget=50)
    row = {"n": 16, "target_points": 2, "method": "matched", "signal": "white", "storage": "disk",
           "statistics": "corrected", "requested_statistics": "corrected", "frequency_matches_plan": False,
           "validation": {"frequency_points": 2, "frequency_mismatch_points": 1,
                          "nsd_sqrt_mismatch_points": True, "nonfinite_psd_points": 65537,
                          "maximum_nsd_sqrt_absolute_error": float("inf"), "input": "PRIVATE"},
           "fftw_threading": {"requested": 2, "actual": 1, "status": "fallback_single_thread",
                              "thread_runtime": "PRIVATE", "exception": "PRIVATE",
                              "failures": {"thread_load_failed": 2, "PRIVATE": 1}}, **result}
    row["memory_guard"]["path"] = "PRIVATE"
    row["memory_guard"]["pid"] = 12345
    report = {"schema": SCHEMA, "state": "completed", "cases": [row]}
    clean = privacy.sanitize(report)
    assert privacy.sanitize(clean) == clean
    actual = clean["cases"][0]
    assert actual["memory_guard"]["trigger_bytes"] == 51
    assert actual["validation"] == {"frequency_points": 2, "frequency_mismatch_points": 1}
    assert actual["fftw_threading"]["failures"] == {"thread_load_failed": 2}
    body = privacy.issue_body(clean)
    assert "PRIVATE" not in body and '"pid"' not in body and '"input"' not in body
    decoded = json.loads(body.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert decoded == clean


def test_statistics_default_old_reports_and_comparison_groups_are_explicit():
    assert runner.parser().parse_args(["run"]).statistics == "corrected"
    assert runner.parser().parse_args(["run", "--statistics", "legacy"]).statistics == "legacy"
    base = {"n": 16, "target_points": 2, "signal": "white", "storage": "ram", "status": "completed"}
    old = privacy.sanitize({"schema": SCHEMA, "cases": [{**base, "method": "lpsd"},
                                                       {**base, "method": "fftw"}]})
    assert old["settings"]["requested_statistics"] == "legacy"
    assert old["cases"][0]["statistics"] == "legacy"
    assert old["cases"][1]["statistics"] == "not_applicable"
    assert "frequency_matches_plan" not in old["cases"][0]
    spectrum = {"frequency_hz": [1., 2.], "nsd": [1., 2.]}
    report, spectra = {}, {}
    legacy = {**base, "method": "lpsd"}
    corrected = {**base, "method": "matched", "statistics": "corrected", "requested_statistics": "corrected"}
    fft = {**base, "method": "fftw", "statistics": "not_applicable", "requested_statistics": "corrected"}
    for row in (legacy, corrected, fft):
        runner._compare_group(report, spectra, row, spectrum)
    assert "comparison" not in corrected and "comparison" not in fft
    reference = {**base, "method": "lpsd", "statistics": "corrected", "requested_statistics": "corrected"}
    runner._compare_group(report, spectra, reference, spectrum)
    assert corrected["comparison"]["reference_statistics"] == "corrected"
    assert fft["comparison"]["reference_statistics"] == "corrected"
    assert corrected["comparison"]["maximum_relative_nsd_difference"] == 0
