"""A downloaded source archive must run without Git or a checkout marker."""
import subprocess
from pathlib import Path

import pytest

from benchmarks import matched_smoothing as module
from lpsd_fast import api


@pytest.mark.parametrize("supplied", [None, "local", "a" * 40, "invalid-personal-text"])
def test_archive_provenance_keeps_hashes_without_requiring_git(monkeypatch, supplied):
    api._native()
    monkeypatch.setattr(module, "_SOURCE_METADATA", None)
    if supplied is None:
        monkeypatch.delenv("LPSD_BENCHMARK_SOURCE_COMMIT", raising=False)
    else:
        monkeypatch.setenv("LPSD_BENCHMARK_SOURCE_COMMIT", supplied)

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(module.subprocess, "check_output", unavailable)
    result = module._source_metadata()
    assert result["commit"] == ("a" * 40 if supplied == "a" * 40 else None)
    assert result["commit_origin"] == ("pinned_archive" if supplied == "a" * 40 else "unavailable")
    assert result["source_sha256"]
    assert all(len(digest) == 64 for digest in result["source_sha256"].values())
    assert str(Path(api.__file__).with_name("planning.py").resolve()) in result["source_sha256"]
    assert len(result["native_binary_sha256"]) == 64
