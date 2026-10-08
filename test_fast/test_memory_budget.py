"""Keep reclaimable page cache from accidentally serializing frequency work."""
from pathlib import Path

import pytest

from lpsd_fast import api


@pytest.mark.parametrize("stat,expected", (
    ("inactive_file 7000\nfile_dirty 200\nfile_writeback 100\n", 7700),
    ("inactive_file 7000\nfile_dirty 8000\n", 1000),
    ("anon 7000\nshmem 1000\nactive_file 1000\n", 1000),
    ("inactive_file 99999\n", 10000),
    ("inactive_file invalid\n", 1000),
    (None, 1000),
))
def test_cgroup_budget_counts_only_clean_inactive_file_cache(monkeypatch, stat, expected):
    files = {
        '/sys/fs/cgroup/memory.max': '10000',
        '/sys/fs/cgroup/memory.current': '9000',
        '/sys/fs/cgroup/memory.stat': stat,
        '/proc/meminfo': 'MemAvailable: 1000 kB\n',
    }

    def read_text(path, *args, **kwargs):
        value = files.get(str(path))
        if value is None:
            raise FileNotFoundError(str(path))
        return value

    monkeypatch.setattr(Path, 'read_text', read_text)
    assert api._available_memory() == expected


def test_host_available_memory_still_caps_cgroup_estimate(monkeypatch):
    files = {
        '/sys/fs/cgroup/memory.max': '10000',
        '/sys/fs/cgroup/memory.current': '9000',
        '/sys/fs/cgroup/memory.stat': 'inactive_file 7000\n',
        '/proc/meminfo': 'MemAvailable: 2 kB\n',
    }
    monkeypatch.setattr(Path, 'read_text', lambda path: files[str(path)])
    assert api._available_memory() == 2048
