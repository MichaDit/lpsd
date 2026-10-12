"""Tiny watchdog contracts; no spectral arrays or performance measurements."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from lpsd_benchmark import SCHEMA, privacy, resources, runner


def _fake_case(tmp_path, monkeypatch, probe, *, storage="disk", exit_at=None,
               interval=.25, budget=100, worker_result=None, platform=None):
    clock, reads, sleeps, stopped = [0.], [], [], []

    class Process:
        pid = 12345
        returncode = None

        def poll(self):
            if self.returncode is None and exit_at is not None and clock[0] >= exit_at:
                self.returncode = 0
            return self.returncode

    process = Process()

    def memory():
        reads.append(clock[0])
        return probe(clock, process)

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    def terminate(child):
        stopped.append((child, clock[0]))
        child.returncode = -9

    (tmp_path / "result.json").write_text(json.dumps(
        worker_result if worker_result is not None else {"status": "completed", "reason": ""}
    ), encoding="utf-8")
    if platform is not None:
        monkeypatch.setattr(runner, "sys", SimpleNamespace(platform=platform, executable=sys.executable))
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, "_LinuxMemoryReader", lambda pid: SimpleNamespace(read=memory))
    monkeypatch.setattr(runner, "_memory_usage", lambda pid: memory())
    monkeypatch.setattr(runner, "terminate_tree", terminate)
    case = {"storage": storage, "ram_budget_bytes": budget, "disk_budget_bytes": 1024}
    result = runner.execute_case(case, tmp_path, SimpleNamespace(end=5.), 5.,
                                 command=["synthetic-watchdog-child"], interval=interval)
    return result, process, reads, sleeps, stopped


def test_missing_startup_probe_fails_closed_at_grace_even_with_long_poll_interval(tmp_path, monkeypatch):
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (0, 0), interval=3.)
    assert result["status"] == "failed"
    assert result["reason"] == "memory_measurement_unavailable"
    assert result["process_wall_s"] == 1.
    assert reads == [0., 1.] and sleeps == [1.]
    assert stopped == [(process, 1.)] and process.returncode is not None
    assert result["live_memory_observed"] is False
    assert result["live_memory_samples"] == 0
    assert result["live_memory_missing_samples"] == 2
    assert result["rss_measurement"] == "unavailable"
    assert "compute_wall_s" not in result


@pytest.mark.parametrize("storage", ["ram", "disk"])
def test_recovery_within_grace_accepts_small_usage_and_measured_zero_private(tmp_path, monkeypatch, storage):
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (0, 0) if clock[0] < .75 else (1, 0),
        storage=storage, exit_at=2.)
    assert result["status"] == "completed" and not stopped
    assert result["process_wall_s"] == 2. and process.returncode == 0
    assert result["live_memory_observed"] is True
    assert result["live_memory_samples"] == 5
    assert result["live_memory_missing_samples"] == 3
    assert result["peak_rss_bytes"] == 1 and result["rss_measurement"] == "sampled"


def test_later_persistent_gap_also_fails_closed(tmp_path, monkeypatch):
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (7, 3) if clock[0] == 0. else (0, 0))
    assert result["status"] == "failed" and result["reason"] == "memory_measurement_unavailable"
    assert result["process_wall_s"] == 1.25
    assert stopped == [(process, 1.25)]
    assert result["live_memory_observed"] is True
    assert result["live_memory_samples"] == 1
    assert result["live_memory_missing_samples"] == 5
    assert result["peak_rss_bytes"] == 7


def test_each_recovery_resets_only_the_missing_probe_grace(tmp_path, monkeypatch):
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch,
        lambda clock, child: (1, 1) if clock[0] in (0., .75, 1.5) else (0, 0), exit_at=2.)
    assert result["status"] == "completed" and not stopped
    assert result["process_wall_s"] == 2.
    assert result["live_memory_samples"] == 3
    assert result["live_memory_missing_samples"] == 5


@pytest.mark.parametrize("storage,expected", [("disk", "failed"), ("ram", "completed")])
@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_missing_private_counter_does_not_masquerade_as_a_disk_probe(tmp_path, monkeypatch, storage, expected, platform):
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform=platform, executable=sys.executable))
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (7, None), storage=storage, exit_at=1.5)
    assert result["status"] == expected
    assert result["peak_rss_bytes"] == 7 and result["rss_measurement"] == "sampled"
    assert result["live_memory_rss_fallback_samples"] == 0
    if storage == "disk":
        assert result["reason"] == "memory_measurement_unavailable"
        assert result["live_memory_observed"] is False and result["live_memory_samples"] == 0
        assert len(stopped) == 1
    else:
        assert result["live_memory_observed"] is True and result["live_memory_missing_samples"] == 0
        assert not stopped


def test_ram_requires_its_own_positive_rss_probe(tmp_path, monkeypatch):
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (0, 3), storage="ram")
    assert result["status"] == "failed" and result["reason"] == "memory_measurement_unavailable"
    assert result["process_wall_s"] == 1. and stopped == [(process, 1.)]
    assert result["live_memory_observed"] is False and result["live_memory_samples"] == 0
    assert result["rss_measurement"] == "unavailable"


@pytest.mark.parametrize("storage,budget,expected", [
    ("ram", 6, "failed"), ("disk", 6, "completed"), ("disk", 2, "failed")])
def test_valid_rss_and_private_budget_policies_are_unchanged(tmp_path, monkeypatch, storage, budget, expected):
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (7, 3), storage=storage,
        budget=budget, exit_at=.5)
    assert result["status"] == expected
    assert result["live_memory_observed"] is True
    assert result["live_memory_missing_samples"] == 0
    if expected == "failed":
        assert result["reason"] == "memory_limit" and len(stopped) == 1
    else:
        assert not stopped


@pytest.mark.parametrize("high_water", [None, 123])
def test_child_finished_before_first_probe_is_not_reported_as_live_observed(tmp_path, monkeypatch, high_water):
    worker = {"status": "completed", "reason": "", "live_memory_observed": True,
              "live_memory_samples": 999, "live_memory_missing_samples": 999,
              "live_memory_rss_fallback_samples": 999}
    if high_water is not None:
        worker["peak_rss_bytes"] = high_water

    def forbidden(clock, child):
        pytest.fail("A child already reaped by poll must not be probed")

    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, forbidden, exit_at=0., worker_result=worker)
    assert result["status"] == "completed" and not reads and not sleeps and not stopped
    assert result["live_memory_observed"] is False
    assert result["live_memory_samples"] == result["live_memory_missing_samples"] == 0
    assert result["live_memory_rss_fallback_samples"] == 0
    assert result["rss_measurement"] == ("unavailable" if high_water is None else "process_high_water_mark")


def test_exit_during_missing_read_is_reaped_before_failure_classification(tmp_path, monkeypatch):
    def disappeared(clock, child):
        clock[0] = 1.
        return 0, 0

    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, disappeared, exit_at=.5)
    assert result["status"] == "completed" and process.returncode == 0
    assert not stopped and not sleeps
    assert result["live_memory_observed"] is False
    assert result["live_memory_samples"] == 0 and result["live_memory_missing_samples"] == 1


def test_missing_probe_uses_actual_termination_and_reaps_tiny_child(tmp_path, monkeypatch):
    spawned = []
    original = runner.subprocess.Popen

    def record_process(*args, **kwargs):
        child = original(*args, **kwargs)
        spawned.append(child)
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", record_process)
    monkeypatch.setattr(runner, "_LinuxMemoryReader", lambda pid: SimpleNamespace(read=lambda: (0, 0)))
    monkeypatch.setattr(runner, "_memory_usage", lambda pid: (0, 0))
    monkeypatch.setattr(runner, "_MEMORY_MEASUREMENT_GRACE_S", .05)
    case = {"storage": "disk", "ram_budget_bytes": resources.MIB, "disk_budget_bytes": resources.MIB}
    started = time.monotonic()
    result = runner.execute_case(case, tmp_path, resources.Deadline(8.), 5., interval=.01,
                                 command=[sys.executable, "-c", "import time; time.sleep(30)"])
    assert result["status"] == "failed" and result["reason"] == "memory_measurement_unavailable"
    assert spawned and spawned[0].returncode is not None and spawned[0].poll() is not None
    assert result["live_memory_observed"] is False and result["live_memory_samples"] == 0
    assert result["live_memory_missing_samples"] >= 1
    assert time.monotonic() - started < 8.


@pytest.mark.parametrize("system,memory_info,full_info,expected", [
    ("posix", {}, {}, (9, None)),
    ("posix", {}, {"uss": 0}, (9, 0)),
    ("nt", {"private": 0, "pagefile": 7}, {}, (9, 0)),
    ("nt", {"pagefile": 0}, {}, (9, 0)),
    ("nt", {}, {}, (9, None)),
])
def test_platform_reader_distinguishes_missing_counter_from_measured_zero(
        monkeypatch, system, memory_info, full_info, expected):
    process = SimpleNamespace(
        children=lambda recursive: [], is_running=lambda: True,
        memory_info=lambda: SimpleNamespace(rss=9, **memory_info),
        memory_full_info=lambda: SimpleNamespace(**full_info))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda pid: process))
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="non-linux"))
    monkeypatch.setattr(runner, "os", SimpleNamespace(name=system))
    assert runner._memory_usage(12345) == expected


def test_partial_private_tree_counter_is_unavailable(monkeypatch):
    child = SimpleNamespace(is_running=lambda: True, memory_info=lambda: SimpleNamespace(rss=3),
                            memory_full_info=lambda: SimpleNamespace())
    owner = SimpleNamespace(children=lambda recursive: [child], is_running=lambda: True,
                            memory_info=lambda: SimpleNamespace(rss=9),
                            memory_full_info=lambda: SimpleNamespace(uss=2))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda pid: owner))
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="non-linux"))
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix"))
    assert runner._memory_usage(12345) == (12, None)


@pytest.mark.parametrize("error_kind", ["access", "exited", "unsupported", "oserror"])
def test_other_posix_optional_uss_error_preserves_valid_rss(monkeypatch, error_kind):
    import psutil

    errors = {"access": psutil.AccessDenied(12345), "exited": psutil.NoSuchProcess(12345),
              "unsupported": NotImplementedError(), "oserror": OSError()}

    def unavailable():
        raise errors[error_kind]

    process = SimpleNamespace(children=lambda recursive: [], is_running=lambda: True,
                              memory_info=lambda: SimpleNamespace(rss=9),
                              memory_full_info=unavailable)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda pid: process, AccessDenied=psutil.AccessDenied, NoSuchProcess=psutil.NoSuchProcess))
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="other-posix"))
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix"))
    assert runner._memory_usage(12345) == (9, None)


@pytest.mark.parametrize("failure", ["children", "child_rss", "missing_child_rss"])
def test_incomplete_rss_tree_stays_unavailable(monkeypatch, failure):
    import psutil

    def denied(*args, **kwargs):
        raise psutil.AccessDenied(12345)

    child = SimpleNamespace(is_running=lambda: True,
                            memory_info=denied if failure == "child_rss" else lambda: SimpleNamespace(rss=None),
                            memory_full_info=lambda: SimpleNamespace(uss=1))
    owner = SimpleNamespace(children=denied if failure == "children" else lambda recursive: [child],
                            is_running=lambda: True, memory_info=lambda: SimpleNamespace(rss=9),
                            memory_full_info=lambda: SimpleNamespace(uss=2))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda pid: owner, AccessDenied=psutil.AccessDenied, NoSuchProcess=psutil.NoSuchProcess))
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix"))
    assert runner._memory_usage(12345) == (None, None)


def test_unavailable_psutil_is_not_reported_as_zero_memory(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="darwin"))
    assert runner._memory_usage(12345) == (None, None)


@pytest.mark.parametrize("storage,budget,expected,fallbacks", [
    ("disk", 7, "completed", 2), ("disk", 6, "failed", 1),
    ("ram", 7, "completed", 0), ("ram", 6, "failed", 0)])
def test_macos_rss_fallback_uses_unchanged_budget_and_reports_its_samples(
        tmp_path, monkeypatch, storage, budget, expected, fallbacks):
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="darwin", executable=sys.executable))
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (7, None),
        storage=storage, budget=budget, exit_at=.5)
    assert result["status"] == expected
    assert result["live_memory_rss_fallback_samples"] == fallbacks
    assert result["live_memory_missing_samples"] == 0 and result["live_memory_observed"] is True
    if expected == "failed":
        assert result["reason"] == "memory_limit" and stopped == [(process, 0.)]
    else:
        assert not stopped


@pytest.mark.parametrize("rss", [None, 0, True, -1])
def test_macos_fallback_requires_valid_positive_rss(tmp_path, monkeypatch, rss):
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="darwin", executable=sys.executable))
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (rss, None))
    assert result["status"] == "failed" and result["reason"] == "memory_measurement_unavailable"
    assert result["live_memory_rss_fallback_samples"] == 0
    assert result["process_wall_s"] == 1. and stopped == [(process, 1.)]


def test_macos_measured_zero_uss_does_not_trigger_conservative_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="darwin", executable=sys.executable))
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (7, 0), budget=1, exit_at=.5)
    assert result["status"] == "completed" and not stopped
    assert result["live_memory_rss_fallback_samples"] == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="Real macOS RSS/USS API qualification")
def test_real_macos_owned_child_memory_probe(capsys):
    """Record the real counter/privilege result without logging a PID or path."""
    import psutil

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; data=bytearray(1024*1024); print('ready', flush=True); time.sleep(10)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        process = psutil.Process(child.pid)
        rss = process.memory_info().rss
        assert type(rss) is int and rss > 0
        private_state = "not_queried"
        tree_rss, tree_private = runner._memory_usage(child.pid)
        observed = {"psutil_version": psutil.__version__, "rss_available": True,
                    "uss_probe": private_state,
                    "tree_rss_available": type(tree_rss) is int and tree_rss > 0,
                    "tree_private_available": type(tree_private) is int and tree_private >= 0}
        with capsys.disabled():
            print("macOS owned-child memory probe: " + json.dumps(observed, sort_keys=True), flush=True)
        assert observed["tree_rss_available"] and tree_private is None, observed
    finally:
        child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)
        child.stdout.close()


def _darwin_reader_with_forbidden_uss(monkeypatch):
    rss_reads, uss_reads = [], []

    def process(label, resident):
        def memory_info():
            rss_reads.append(label)
            return SimpleNamespace(rss=resident)

        def memory_full_info():
            uss_reads.append(label)
            raise PermissionError("The privileged USS API must not be called")

        return SimpleNamespace(is_running=lambda: True, memory_info=memory_info,
                               memory_full_info=memory_full_info)

    owner, child = process("owner", 9), process("child", 3)
    owner.children = lambda recursive: [child]
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda pid: owner))
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="darwin", executable=sys.executable))
    return runner._memory_usage, owner, child, rss_reads, uss_reads


def test_darwin_reads_complete_rss_tree_without_issuing_privileged_uss_query(monkeypatch):
    read, owner, child, rss_reads, uss_reads = _darwin_reader_with_forbidden_uss(monkeypatch)
    assert read(12345) == (12, None)
    assert rss_reads == ["owner", "child"]
    assert not uss_reads


@pytest.mark.parametrize("storage,budget,expected,fallbacks", [
    ("ram", 20, "completed", 0), ("ram", 10, "failed", 0),
    ("disk", 20, "completed", 2), ("disk", 10, "failed", 1),
])
def test_darwin_actual_reader_keeps_ram_rss_and_enforces_conservative_disk_bound(
        tmp_path, monkeypatch, storage, budget, expected, fallbacks):
    read, owner, child, rss_reads, uss_reads = _darwin_reader_with_forbidden_uss(monkeypatch)
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: read(child.pid), storage=storage,
        budget=budget, exit_at=.5, platform="darwin")
    assert result["status"] == expected and not uss_reads
    assert result["peak_rss_bytes"] == 12 and result["rss_measurement"] == "sampled"
    assert result["live_memory_observed"] is True and result["live_memory_missing_samples"] == 0
    assert result["live_memory_rss_fallback_samples"] == fallbacks
    assert len(rss_reads) == 2 * result["live_memory_samples"]
    if expected == "failed":
        assert result["reason"] == "memory_limit" and stopped == [(process, 0.)]
    else:
        assert not stopped


@pytest.mark.parametrize("failure", ["exception", "absent", "negative", "boolean"])
def test_darwin_partial_or_invalid_rss_does_not_pass_using_known_owner_bytes(
        tmp_path, monkeypatch, failure):
    read, owner, child, rss_reads, uss_reads = _darwin_reader_with_forbidden_uss(monkeypatch)

    def bad_child_info():
        if failure == "exception":
            raise PermissionError("Child RSS unavailable")
        return SimpleNamespace(**({} if failure == "absent" else {"rss": -1 if failure == "negative" else True}))

    child.memory_info = bad_child_info
    assert read(12345) == (None, None)
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: read(child.pid), platform="darwin")
    assert result["status"] == "failed" and result["reason"] == "memory_measurement_unavailable"
    assert result["process_wall_s"] == 1. and stopped == [(process, 1.)]
    assert result["live_memory_samples"] == result["live_memory_rss_fallback_samples"] == 0
    assert not uss_reads


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_private_counter_absence_does_not_enable_disk_rss_fallback_on_other_platforms(
        tmp_path, monkeypatch, platform):
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (7, None), platform=platform)
    assert result["status"] == "failed" and result["reason"] == "memory_measurement_unavailable"
    assert result["process_wall_s"] == 1. and stopped == [(process, 1.)]
    assert result["live_memory_samples"] == result["live_memory_rss_fallback_samples"] == 0


@pytest.mark.parametrize("private", [0, 3])
def test_darwin_known_private_counter_is_not_replaced_by_larger_rss(tmp_path, monkeypatch, private):
    result, process, reads, sleeps, stopped = _fake_case(
        tmp_path, monkeypatch, lambda clock, child: (7, private), budget=6,
        exit_at=.5, platform="darwin")
    assert result["status"] == "completed" and not stopped
    assert result["live_memory_samples"] == 2 and result["live_memory_rss_fallback_samples"] == 0


def test_rss_fallback_public_counter_is_bounded_and_cannot_claim_more_than_live_samples():
    row = {"n": 16, "target_points": 2, "method": "fftw", "signal": "white", "storage": "disk",
           "status": "completed", "live_memory_observed": True, "live_memory_samples": 3,
           "live_memory_missing_samples": 0, "live_memory_rss_fallback_samples": 2}
    report = {"schema": SCHEMA, "state": "completed", "cases": [row]}
    clean = privacy.sanitize(report)
    assert clean["cases"][0]["live_memory_rss_fallback_samples"] == 2
    assert privacy.sanitize(clean) == clean
    for bad in (True, False, -1, 1.5, 4, 2 ** 63, "personal-hostname"):
        row["live_memory_rss_fallback_samples"] = bad
        assert "live_memory_rss_fallback_samples" not in privacy.sanitize(report)["cases"][0]
    row["live_memory_rss_fallback_samples"] = 0
    del row["live_memory_samples"]
    assert "live_memory_rss_fallback_samples" not in privacy.sanitize(report)["cases"][0]


def test_live_probe_report_fields_are_closed_and_strictly_typed():
    row = {"n": 16, "target_points": 2, "method": "fftw", "signal": "white", "storage": "disk",
           "status": "failed", "reason": "memory_measurement_unavailable", "rss_measurement": "unavailable",
           "live_memory_observed": False, "live_memory_samples": 0, "live_memory_missing_samples": 2,
           "live_memory_rss_fallback_samples": 0,
           "memory_probe_error": "/private/owner/path", "memory_reader": "personal-hostname"}
    report = {"schema": SCHEMA, "state": "completed", "cases": [row]}
    clean = privacy.sanitize(report)
    actual = clean["cases"][0]
    assert actual["reason"] == "memory_measurement_unavailable"
    assert actual["rss_measurement"] == "unavailable"
    assert actual["live_memory_observed"] is False
    assert actual["live_memory_samples"] == 0 and actual["live_memory_missing_samples"] == 2
    assert actual["live_memory_rss_fallback_samples"] == 0
    assert "private" not in json.dumps(clean) and "personal" not in json.dumps(clean)
    assert privacy.sanitize(clean) == clean
    assert "memory_measurement_unavailable" in privacy.issue_body(report)

    row.update(live_memory_observed=1, live_memory_samples=True, live_memory_missing_samples=1.5,
               live_memory_rss_fallback_samples=True,
               rss_measurement="personal-hostname", reason="private-reader-error")
    actual = privacy.sanitize(report)["cases"][0]
    assert not any(key.startswith("live_memory_") for key in actual)
    assert "rss_measurement" not in actual and actual["reason"] == "worker_failed"
