"""Report admitted CPU capacity separately from the requested worker cap."""
from types import SimpleNamespace

import pytest

from lpsd_benchmark import SCHEMA, privacy, resources, runner


@pytest.mark.parametrize("affinity,quota,expected_available,expected_quota", [
    (8, 6.5, 6, 6.5),
    (3, 6.5, 3, 3.0),
    (8, 0.5, 1, 0.5),
    (8, None, 8, 8.0),
])
@pytest.mark.parametrize("requested", [None, 1, 4, 16])
def test_environment_capacity_does_not_change_with_requested_workers(
        tmp_path, monkeypatch, affinity, quota, expected_available, expected_quota, requested):
    controller = tmp_path / "cpu"
    controller.mkdir()
    maximum = "max" if quota is None else str(int(quota * 100_000))
    (controller / "cpu.max").write_text(f"{maximum} 100000\n", encoding="ascii")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda pid: set(range(affinity)),
                        raising=False)
    monkeypatch.setattr(resources, "_cgroup_directories", lambda kind: [controller])
    monkeypatch.setattr(resources, "system_memory", lambda: (8 * resources.GIB, 6 * resources.GIB))
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda path: SimpleNamespace(free=8 * resources.GIB))
    monkeypatch.setattr(resources, "ram_backed_work_directory", lambda path: False)

    detected = resources.detect_resources(tmp_path, threads=requested)
    expected_workers = expected_available if requested is None else min(requested, expected_available)
    assert detected.threads == expected_workers
    report = privacy.sanitize({
        "schema": SCHEMA, "state": "completed", "cases": [],
        "environment": runner.environment(detected),
        "settings": {"threads": detected.threads},
    })
    assert report["environment"]["logical_cpus_available"] == expected_available
    assert report["environment"]["cpu_quota"] == expected_quota
    assert report["settings"]["threads"] == expected_workers
