"""Offline multipart protocol and mocked publication; no numerical workers."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from lpsd_benchmark import SCHEMA, privacy, runner


def report_fixture(count=198):
    # A schema-valid entropy stress case, not claimed device measurements.
    report = {"schema": SCHEMA, "state": "completed", "elapsed_seconds": 1.,
              "environment": {"os": "Linux", "process_arch": "aarch64",
                              "username": "IDENTITY_CANARY"},
              "settings": {"point_factors": [1., 2., 4.]}, "cases": [],
              "hostname": "IDENTITY_CANARY", "spectra": ["SPECTRA_CANARY"]}
    for index in range(count):
        report["cases"].append({
            "n": 100_000, "target_points": 1000, "method": "fftw", "signal": "white",
            "storage": "ram", "status": "completed", "compute_wall_s": .1 + index / 10000,
            **{name: hashlib.sha256(f"{index}/{name}".encode()).hexdigest() for name in privacy.HASHES},
            "untrusted_path": "PATH_CANARY"})
    return report


def reconstruct(parts):
    """Independent bounded reconstruction for these generated test fixtures."""
    assert 2 <= len(parts) <= 16
    chunks, digests, totals = {}, set(), set()
    for body in parts:
        assert len(body.encode("utf-8")) <= 60_000
        digest, = re.findall(r"^Payload SHA-256: `([0-9a-f]{64})`$", body, re.M)
        (index, total), = re.findall(r"^Payload part: `([0-9]+)/([0-9]+)`$", body, re.M)
        chunk, = re.findall(r"```text\n([A-Za-z0-9+/=]+)\n```", body)
        index, total = int(index), int(total)
        assert index not in chunks
        chunks[index] = chunk
        digests.add(digest)
        totals.add(total)
    assert totals == {len(parts)} and len(digests) == 1
    assert set(chunks) == set(range(1, len(parts) + 1))
    compressed = base64.b64decode("".join(chunks[i] for i in range(1, len(parts) + 1)), validate=True)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        payload = stream.read(privacy.MAX_LOCAL_BYTES + 1)
    assert len(payload) <= privacy.MAX_LOCAL_BYTES
    assert hashlib.sha256(payload).hexdigest() == next(iter(digests))
    return payload


class Clock:
    def __init__(self):
        self.now = 0.
        self.pauses = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.pauses.append(seconds)
        self.now += seconds


class GitHub:
    """Only fabricated receipts; all files must precede even the auth probe."""
    issue_url = "https://github.com/MichaDit/lpsd/issues/123"
    api_issue = "https://api.github.com/repos/MichaDit/lpsd/issues/123"

    def __init__(self, directory, parts, clock, failure=None, expire_after=None):
        self.directory, self.parts, self.clock = directory, parts, clock
        self.failure, self.expire_after = failure, expire_after
        self.calls, self.posts = [], []
        self.paths = [directory / "results.github-issue.md"] + [
            directory / f"results.github-issue.part-{i:02d}-of-{len(parts):02d}.md"
            for i in range(2, len(parts) + 1)]

    def find_gh(self, name):
        assert name == "gh"
        for index, (path, body) in enumerate(zip(self.paths, self.parts)):
            assert path.read_text(encoding="utf-8") == body
            if index:
                assert json.loads(path.with_suffix(".request.json").read_text()) == {"body": body}
        return "gh-mock"

    def __call__(self, command, **kwargs):
        self.find_gh("gh")
        self.calls.append(command)
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["env"]["GH_HOST"] == "github.com"
        assert kwargs["env"]["GH_TELEMETRY"] == "0"
        assert "GH_DEBUG" not in kwargs["env"]
        assert 0 < kwargs["timeout"] <= 45.
        self.clock.now += .05
        if command[1:3] == ["auth", "status"]:
            return SimpleNamespace(returncode=0, stdout="PRIVATE_AUTH_CANARY", stderr="")
        self.posts.append(command)
        index = len(self.posts)
        receipt_path, = self.directory.glob("*.submission.json")
        receipt = json.loads(receipt_path.read_text())
        assert receipt["uncertain_part"] == index
        assert len(receipt["part_urls"]) == index - 1
        if index == 1:
            assert command[1:3] == ["issue", "create"]
            assert command[command.index("--body-file") + 1] == str(self.paths[0])
            return SimpleNamespace(returncode=0, stdout=self.issue_url + "\n", stderr="")
        assert command[1:] == [
            "api", "--hostname", "github.com", "--method", "POST",
            "repos/MichaDit/lpsd/issues/123/comments", "--input",
            str(self.paths[index - 1].with_suffix(".request.json"))]
        reply = {"id": 1000 + index, "html_url": f"{self.issue_url}#issuecomment-{1000 + index}",
                 "issue_url": self.api_issue, "body": self.parts[index - 1],
                 "user": {"login": "PRIVATE_RESPONSE_CANARY"}}
        if self.failure and index == 2:
            if self.failure == "timeout":
                raise subprocess.TimeoutExpired(command, 1, output="PRIVATE_RESPONSE_CANARY")
            if self.failure == "invalid_json":
                return SimpleNamespace(returncode=0, stdout="PRIVATE_RESPONSE_CANARY", stderr="")
            if self.failure == "other_issue":
                reply["issue_url"] = self.api_issue + "4"
            if self.failure == "different_body":
                reply["body"] = "PRIVATE_RESPONSE_CANARY"
            if self.failure == "boolean_id":
                reply["id"] = True
            if self.failure == "missing_url":
                reply.pop("html_url")
        if self.expire_after == index:
            self.clock.now = 44.75
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr="PRIVATE_STDERR_CANARY")


def prepare(tmp_path, monkeypatch, count=198, **kwargs):
    report = report_fixture(count)
    path = tmp_path / "results.public.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    parts = privacy.issue_parts(report)
    clock = Clock()
    monkeypatch.setattr(privacy, "time", clock)
    return path, report, GitHub(tmp_path, parts, clock, **kwargs)


@pytest.mark.parametrize("repeated", [False, True])
def test_issue_parts_preserves_exact_existing_single_body(repeated):
    report = report_fixture(1)
    if repeated:
        report["cases"] *= 300
    body = privacy.issue_body(report)
    assert privacy.issue_parts(report) == [body]
    assert len(body.encode("utf-8")) <= privacy.MAX_ISSUE_BYTES


@pytest.mark.parametrize("count", [198, 396])
def test_multipart_reconstructs_exact_allowlisted_payload_and_sha256(count):
    report = report_fixture(count)
    with pytest.raises(ValueError, match="too large for one GitHub issue"):
        privacy.issue_body(report)
    parts = privacy.issue_parts(report)
    expected = json.dumps(privacy.sanitize(report), ensure_ascii=True, allow_nan=False,
                          separators=(",", ":")).encode("ascii")
    assert reconstruct(list(reversed(parts))) == expected
    assert json.loads(expected) == privacy.sanitize(report)
    assert b"CANARY" not in expected and "spectra" not in json.loads(expected)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "digest", "total"])
def test_multipart_reconstruction_rejects_incomplete_or_mixed_parts(damage):
    parts = privacy.issue_parts(report_fixture())
    if damage == "missing":
        parts.pop()
    elif damage == "duplicate":
        parts[-1] = parts[0]
    elif damage == "digest":
        parts[0] = re.sub(r"(?<=Payload SHA-256: `)[0-9a-f]{64}", "0" * 64, parts[0])
    else:
        parts[0] = parts[0].replace(f"Payload part: `1/{len(parts)}`",
                                    f"Payload part: `1/{len(parts) + 1}`")
    with pytest.raises(AssertionError):
        reconstruct(parts)


def test_multipart_hard_limit_rejects_seventeenth_part_before_transport(tmp_path, monkeypatch):
    assert privacy.MAX_ISSUE_PARTS == 16
    assert privacy.MAX_LOCAL_BYTES == 16 * 1024 ** 2
    assert privacy.MAX_ISSUE_BYTES == 60_000 and privacy.MAX_CASES == 512
    path = tmp_path / "results.json"
    path.write_text(json.dumps(report_fixture(198)))
    # Fault-inject an excessive compressed size without generating an
    # oversized numerical matrix. No decoding or transport may follow.
    monkeypatch.setattr(privacy.gzip, "compress", lambda data, mtime: b"x" * 672_001)
    def forbidden(*args, **kwargs):
        pytest.fail("All multipart limits must be checked before auth or transport")
    with pytest.raises(ValueError, match="multipart GitHub issue limit"):
        privacy.submit(path, True, find_gh=forbidden, transport=forbidden)
    assert not list(tmp_path.glob("*.github-issue*"))


def test_multipart_prepares_all_manual_files_without_login_or_consent(tmp_path, monkeypatch):
    path, report, service = prepare(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("No login, CLI discovery or transport is authorized")
    result = privacy.submit(path, False, find_gh=forbidden, transport=forbidden)
    assert result["status"] == "prepared" and result["confirmed_parts"] == 0
    assert result["body_paths"] == service.paths
    assert reconstruct([p.read_text() for p in result["body_paths"]]) == json.dumps(
        privacy.sanitize(report), ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    service.find_gh("gh")  # Checks only already-prepared local files.
    assert not result["receipt_path"].exists()


def test_multipart_posts_one_issue_then_comments_once_and_reuses_receipt(tmp_path, monkeypatch):
    path, _, service = prepare(tmp_path, monkeypatch, count=396)
    result = privacy.submit(path, True, find_gh=service.find_gh, transport=service)
    assert result["status"] == "submitted"
    assert result["confirmed_parts"] == result["total_parts"] == len(service.parts)
    assert len(service.posts) == len(service.parts)
    assert len(service.clock.pauses) == len(service.parts) - 1
    assert service.clock.pauses == pytest.approx([1.] * (len(service.parts) - 1))
    assert "CANARY" not in str(result)
    def forbidden(*args, **kwargs):
        pytest.fail("An existing transfer must never be recreated or automatically resumed")
    again = privacy.submit(path, True, find_gh=forbidden, transport=forbidden)
    assert again["status"] == "submitted" and again["reason"] == "existing_transfer"
    assert again["part_urls"] == result["part_urls"]


@pytest.mark.parametrize("failure", [
    "timeout", "invalid_json", "other_issue", "different_body", "boolean_id", "missing_url"])
def test_lost_or_invalid_comment_receipt_stops_without_retry(tmp_path, monkeypatch, capsys, failure):
    path, _, service = prepare(tmp_path, monkeypatch, count=396, failure=failure)
    result = privacy.submit(path, True, find_gh=service.find_gh, transport=service)
    assert len(service.parts) >= 3
    assert result["status"] == "uncertain"
    assert result["confirmed_parts"] == 1 and result["uncertain_part"] == 2
    assert len(service.posts) == 2  # Issue and one attempted comment only.
    assert result["part_urls"] == [service.issue_url]
    assert all(path.exists() for path in result["body_paths"])
    assert "CANARY" not in str(result)
    receipt = json.loads(result["receipt_path"].read_text())
    assert receipt["uncertain_part"] == 2 and receipt["part_urls"] == [service.issue_url]
    def forbidden(*args, **kwargs):
        pytest.fail("Do not retry an unconfirmed POST or start later comments")
    again = privacy.submit(path, True, find_gh=forbidden, transport=forbidden)
    assert again["status"] == "uncertain" and again["uncertain_part"] == 2
    runner._print_submission(result)
    shown = capsys.readouterr().out
    assert "Part 2 has no verified receipt" in shown
    assert "not attempted" in shown and "CANARY" not in shown


def test_multipart_deadline_keeps_confirmed_prefix_without_next_post(tmp_path, monkeypatch, capsys):
    path, _, service = prepare(tmp_path, monkeypatch, count=396, expire_after=2)
    result = privacy.submit(path, True, find_gh=service.find_gh, transport=service)
    assert result["status"] == "partial" and result["reason"] == "time_budget"
    assert result["confirmed_parts"] == 2 and result["uncertain_part"] is None
    assert len(service.posts) == 2
    runner._print_submission(result)
    assert "Parts 3 through" in capsys.readouterr().out


def test_multipart_slow_first_receipt_does_not_start_post_after_deadline(tmp_path, monkeypatch):
    path, _, service = prepare(tmp_path, monkeypatch)
    path_open = Path.open
    def slow_claim(current, *args, **kwargs):
        stream = path_open(current, *args, **kwargs)
        if args and args[0] == "x":
            service.clock.now = 46.
        return stream
    monkeypatch.setattr(Path, "open", slow_claim)
    result = privacy.submit(path, True, find_gh=service.find_gh, transport=service)
    assert result["status"] == "prepared" and result["reason"] == "time_budget"
    assert not service.posts and result["uncertain_part"] is None


def test_multipart_concurrent_first_claim_prevents_duplicate_issue(tmp_path, monkeypatch):
    path, report, service = prepare(tmp_path, monkeypatch)
    payload = json.dumps(privacy.sanitize(report), ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    receipt_path = tmp_path / f"results.github-issue.{digest}.submission.json"
    def transport(command, **kwargs):
        reply = service(command, **kwargs)
        if command[1:3] == ["auth", "status"]:
            # Another process claimed the same payload after the first check.
            receipt_path.write_text(json.dumps({
                "schema": "lpsd-issue-transfer-v1", "payload_sha256": digest,
                "total_parts": len(service.parts), "part_urls": [], "uncertain_part": 1}))
        return reply
    result = privacy.submit(path, True, find_gh=service.find_gh, transport=transport)
    assert result["reason"] == "local_receipt_unavailable" and not service.posts
    assert json.loads(receipt_path.read_text())["uncertain_part"] == 1
