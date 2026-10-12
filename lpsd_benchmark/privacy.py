"""Small public result schema and an explicit, account-associated issue upload.

No arbitrary strings, paths, environment values, logs or input data cross this
boundary. Publishing is still public and associated with the submitting GitHub
account; data minimisation is not a legal promise of GDPR compliance.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from . import METHODS, SCHEMA, SIGNALS

REPOSITORY = "MichaDit/lpsd"
MAX_LOCAL_BYTES = 16 * 1024 ** 2
MAX_CASES = 512
MAX_ISSUE_BYTES = 60_000
MAX_ISSUE_PARTS = 16
STATES = ("completed", "skipped", "timeout", "failed", "cancelled")
REASONS = ("memory_budget", "disk_budget", "time_budget", "uncalibrated",
           "dependency_unavailable", "time_limit", "memory_limit", "disk_limit",
           "worker_failed", "invalid_result", "cancelled", "unsupported", "")
ARCHITECTURES = ("x86_64", "aarch64", "armv7", "x86", "other")
OPERATING_SYSTEMS = ("Linux", "Windows", "Darwin", "Android", "Other")
VERSIONS = ("python", "numpy", "pandas", "scipy", "pyfftw", "lpsd")
HASHES = ("input_sha256", "frequency_sha256", "psd_sha256", "nsd_sha256",
          "native_sha256", "fftw_sha256", "fftw_threads_sha256", "source_sha256")


def number(value, low=0., high=1e20):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and low <= value <= high else None


def integer(value, low=0, high=2 ** 63 - 1):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if low <= value <= high else None


def _hash(value, length=64):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) else None


def _enum(value, choices, fallback):
    return value if isinstance(value, str) and value in choices else fallback


def _numbers(source, names):
    if not isinstance(source, dict):
        return {}
    return {key: number(source[key]) for key in names
            if key in source and number(source[key]) is not None}


def sanitize(report):
    """Construct a new report from a closed schema, never delete-list redact.

    Even values inside admitted fields are type/range/enum checked. For
    example, a hostname placed in an OS/version/status field is not retained.
    Input records are synthetic fixed presets; no external samples are read.
    """
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError("Unsupported benchmark report schema")
    cases = report.get("cases", [])
    if not isinstance(cases, list) or len(cases) > MAX_CASES:
        raise ValueError("Invalid or oversized case list")
    environment = report.get("environment", {})
    environment = environment if isinstance(environment, dict) else {}
    settings = report.get("settings", {})
    settings = settings if isinstance(settings, dict) else {}
    clean = {
        "schema": SCHEMA,
        "privacy_schema": 1,
        "synthetic_input_only": True,
        "state": _enum(report.get("state"), ("running", "completed", "deadline", "cancelled"), "cancelled"),
        "environment": {
            "os": _enum(environment.get("os"), OPERATING_SYSTEMS, "Other"),
            "process_arch": _enum(environment.get("process_arch"), ARCHITECTURES, "other"),
            "host_arch": _enum(environment.get("host_arch"), ARCHITECTURES, "other"),
            **_numbers(environment, ("logical_cpus_available", "cpu_quota", "ram_total_mib")),
        },
        "settings": {
            **_numbers(settings, ("max_seconds", "bootstrap_remaining_seconds", "ram_budget_mib",
                                  "disk_budget_mib", "threads", "sample_rate_hz", "seed",
                                  "n_averages", "psll_db", "prediction_safety_factor", "base_points")),
            "storage": _enum(settings.get("storage"), ("auto", "ram", "disk", "both"), "auto"),
            "fft_layout": _enum(settings.get("fft_layout"), ("auto", "out-of-place", "in-place"), "auto"),
            "window_storage": _enum(settings.get("window_storage"), ("auto", "full", "fft_input_then_blocked"), "auto"),
            "timing_scope": "complete_call_psd_and_nsd",
            "input_dtype": "float64",
            "output_dtype": "float32",
        },
        "elapsed_seconds": number(report.get("elapsed_seconds")),
        "cases": [],
    }
    commit = _hash(report.get("source_commit"), 40)
    if commit:
        clean["source_commit"] = commit
    if isinstance(report.get("source_dirty"), bool):
        clean["source_dirty"] = report["source_dirty"]
    for key in ("points", "sizes", "quality_sizes"):
        supplied = settings.get(key, [])
        if isinstance(supplied, list):
            clean["settings"][key] = [value for value in supplied[:32]
                                      if integer(value, 2, 2_147_483_647) is not None]
    factors = settings.get("point_factors", [])
    if isinstance(factors, list):
        clean["settings"]["point_factors"] = [value for value in factors[:8]
                                             if number(value, .25, 8.) is not None]
    versions = environment.get("versions", {})
    if isinstance(versions, dict):
        clean["environment"]["versions"] = {
            name: versions[name] for name in VERSIONS if isinstance(versions.get(name), str)
            and re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{1,4}){1,3}(?:(?:a|b|rc)[0-9]{1,3})?(?:\+fast\.[0-9]{1,3})?", versions[name])
        }
    for row in cases:
        if not isinstance(row, dict):
            continue
        if (row.get("method") not in METHODS or row.get("signal") not in SIGNALS
                or integer(row.get("n"), 16, 2_147_483_647) is None
                or integer(row.get("target_points"), 2, 65536) is None):
            continue
        result = {
            "n": row["n"], "target_points": row["target_points"],
            "method": row["method"], "signal": row["signal"],
            "storage": _enum(row.get("storage"), ("ram", "disk"), "ram"),
            "fft_layout": _enum(row.get("fft_layout"), ("out-of-place", "in-place", "blocked", "not_applicable"), "not_applicable"),
            "window_storage": _enum(row.get("window_storage"), ("full", "fft_input_then_blocked", "blocked", "not_applicable"), "not_applicable"),
            "status": _enum(row.get("status"), STATES, "failed"),
            "reason": _enum(row.get("reason", ""), REASONS, "worker_failed"),
            "case_role": _enum(row.get("case_role"), ("calibration", "comparison"), "comparison"),
            **_numbers(row, ("actual_points", "point_factor", "base_target_points", "base_actual_points",
                             "requested_output_points", "requested_threads", "actual_threads",
                             "ram_budget_mib", "disk_budget_mib", "fft_memory_mb", "low_working_mb",
                             "native_working_mb", "cache_bytes",
                             "native_scratch_bytes", "native_streaming_points", "native_streaming_buffer_bytes",
                             "native_working_bytes_upper_bound",
                             "fftw_plan_reserve_bytes", "window_buffer_bytes", "window_scratch_bytes", "window_block_samples",
                             "ram_required_bytes", "disk_required_bytes", "predicted_process_s",
                             "time_limit_s", "process_wall_s", "compute_wall_s", "compute_cpu_s",
                             "input_setup_s", "peak_rss_bytes", "sample_iterations",
                             "minimum_frequency_hz", "maximum_frequency_hz")),
        }
        # Pool capacities are observed allocations, not resident-memory
        # measurements. Keep integer bytes and only actual retirement indices.
        for key in ("native_scratch_initial_bytes", "native_scratch_final_bytes", "native_scratch_peak_bytes"):
            value = integer(row.get(key))
            if value is not None:
                result[key] = value
        if isinstance(row.get("native_scratch_retired"), bool):
            result["native_scratch_retired"] = row["native_scratch_retired"]
            if row["native_scratch_retired"]:
                for key in ("native_scratch_retirement_index", "native_scratch_retirement_source_index"):
                    value = integer(row.get(key))
                    if value is not None:
                        result[key] = value
        for key in HASHES:
            value = _hash(row.get(key))
            if value:
                result[key] = value
        for key in ("fftw_version", "native_version"):
            value = row.get(key)
            if isinstance(value, str) and re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{1,4}){1,3}(?:\+fast\.[0-9]{1,3})?", value):
                result[key] = value
        for key in ("finite", "psd_nonnegative", "nsd_matches_sqrt_psd", "input_unchanged", "point_count_resolved", "one_shot_window"):
            if isinstance(row.get(key), bool):
                result[key] = row[key]
        if row.get("rss_measurement") in ("process_high_water_mark", "sampled"):
            result["rss_measurement"] = row["rss_measurement"]
        if row.get("native_backend") in ("bounded", "legacy", "not_applicable"):
            result["native_backend"] = row["native_backend"]
        result["noise"] = _numbers(row.get("noise"), (
            "points", "mean_nsd_over_source", "normalized_nsd_std",
            "adjacent_difference_rms_over_sqrt2"))
        result["tone"] = _numbers(row.get("tone"), (
            "center_hz", "peak_psd", "peak_nsd", "peak_label_hz", "psd_fwhm_hz", "nsd_fwhm_hz"))
        result["comparison"] = _numbers(row.get("comparison"), (
            "points", "rms_relative_nsd_difference", "maximum_relative_nsd_difference"))
        result["phases"] = _numbers(row.get("phases"), (
            "workspace_and_plan_s", "workspace_and_fftw_planning_s", "window_and_normalization_s",
            "input_validation_copy_mean_window_s", "blocked_fftw_s", "ordered_one_sided_power_s",
            "time_energy_diagnostic_s", "frequency_energy_diagnostic_s", "logarithmic_grouping_and_outputs_s",
            "matched_frequency_smoothing_s", "low_lpsd_and_outputs_s", "lpsd_complete_s",
            "release_fft_before_low_lpsd_s", "workspace_cleanup_s"))
        clean["cases"].append(result)
    return clean


def issue_body(report):
    clean = sanitize(report)
    payload = json.dumps(clean, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    completed = sum(row["status"] == "completed" for row in clean["cases"])
    intro = (
        "## Portable LPSD / FFTW benchmark\n\n"
        "This report contains synthetic-data benchmark measurements. It is public and "
        "associated with the submitting GitHub account. The allowlisted payload excludes "
        "hostnames, usernames, file paths, network identifiers, environment variables, logs "
        "and user-supplied measurement data. GitHub processes account and connection data "
        "separately under its own privacy statement.\n\n"
        f"Completed cases: **{completed}/{len(clean['cases'])}**. "
        "Times are complete PSD+NSD calls; process totals also include setup and checks. "
        "Admission forecasts and timeouts are not completed measurement times.\n\n"
        f"Payload SHA-256: `{digest}`\n\n"
    )
    body = intro + "```json\n" + payload + "\n```\n"
    if len(body.encode("utf-8")) > MAX_ISSUE_BYTES:
        # A complete matrix can exceed GitHub's issue-body size. Compression
        # keeps all rows, instead of silently dropping failed/slow cases.
        compressed = base64.b64encode(gzip.compress(payload.encode("ascii"), mtime=0)).decode("ascii")
        body = intro + "Encoding: gzip + base64; decompress to the JSON schema above.\n\n```text\n" + compressed + "\n```\n"
    if len(body.encode("utf-8")) > MAX_ISSUE_BYTES:
        raise ValueError("Sanitized report is too large for one GitHub issue")
    return body


def issue_parts(report):
    """Keep the existing single-body format, or split one lossless payload.

    Multipart transport is one issue followed by comments. Every part carries
    the SHA-256 of the same uncompressed ASCII JSON, not a per-chunk digest.
    """
    clean = sanitize(report)
    try:
        return [issue_body(clean)]
    except ValueError as error:
        if str(error) != "Sanitized report is too large for one GitHub issue":
            raise
    payload = json.dumps(clean, ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(gzip.compress(payload, mtime=0)).decode("ascii")
    chunk_chars = 56_000  # Multiple of four; leaves room below the body limit.
    count = (len(encoded) + chunk_chars - 1) // chunk_chars
    if not 2 <= count <= MAX_ISSUE_PARTS:
        raise ValueError("Sanitized report exceeds the multipart GitHub issue limit")
    completed = sum(row["status"] == "completed" for row in clean["cases"])
    intro = (
        "## Portable LPSD / FFTW benchmark\n\n"
        "This multipart synthetic-data report is public and associated with the "
        "submitting GitHub account. It contains the same allowlisted fields as "
        "a single-body report; no spectra, paths, hostnames, tokens or logs are "
        "added. GitHub processes account and connection data separately.\n\n"
        f"Completed cases: **{completed}/{len(clean['cases'])}**. Times are "
        "complete PSD+NSD calls; forecasts and timeouts are not measured times.\n\n"
        f"Payload SHA-256: `{digest}`\n\n"
    )
    parts = []
    for index in range(count):
        body = (
            intro + f"Payload part: `{index + 1}/{count}`\n\n"
            "Encoding: gzip + base64 chunks of ONE stream. Collect exactly "
            "parts 1 through the stated total, all with the same SHA-256. "
            "Concatenate their text blocks in index order, decode base64 once, "
            "then decompress once and verify the uncompressed JSON SHA-256. "
            "Part 1 belongs in the issue; later parts belong in its comments.\n\n"
            "```text\n" + encoded[index * chunk_chars:(index + 1) * chunk_chars] + "\n```\n"
        )
        if len(body.encode("utf-8")) > MAX_ISSUE_BYTES:
            raise ValueError("Sanitized report exceeds the multipart GitHub issue limit")
        parts.append(body)
    return parts


def _atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value.encode("utf-8"))
    temporary.replace(path)


def _valid_issue_url(value):
    return (isinstance(value, str) and len(value) <= 256
            and re.fullmatch(r"https://github\.com/MichaDit/lpsd/issues/[1-9][0-9]*", value))


def _load_submission_receipt(path, digest, count):
    """Read only bounded local transfer state; never resume a previous POST."""
    state = load_report(path)
    if (not isinstance(state, dict) or state.get("schema") != "lpsd-issue-transfer-v1"
            or state.get("payload_sha256") != digest or state.get("total_parts") != count):
        raise ValueError("Invalid local multipart submission receipt")
    urls, uncertain = state.get("part_urls"), state.get("uncertain_part")
    if not isinstance(urls, list) or len(urls) > count:
        raise ValueError("Invalid local multipart submission receipt")
    if urls and (not _valid_issue_url(urls[0]) or any(
            not isinstance(url, str) or len(url) > 256
            or not re.fullmatch(re.escape(urls[0]) + r"#issuecomment-[1-9][0-9]*", url)
            for url in urls[1:])):
        raise ValueError("Invalid local multipart submission receipt")
    if uncertain is not None and (integer(uncertain, 1, count) is None or uncertain != len(urls) + 1):
        raise ValueError("Invalid local multipart submission receipt")
    if len(set(urls)) != len(urls):
        raise ValueError("Invalid local multipart submission receipt")
    return list(urls), uncertain


def _submit_parts(report, parts, body_path, accepted, expires, transport, find_gh):
    """Prepare everything first, then at most one POST per part; never retry."""
    clean = sanitize(report)
    payload = json.dumps(clean, ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).encode("ascii")
    digest, count = hashlib.sha256(payload).hexdigest(), len(parts)
    paths = [body_path] + [
        body_path.with_name(f"{body_path.stem}.part-{index:02d}-of-{count:02d}.md")
        for index in range(2, count + 1)]
    request_paths = [None] + [path.with_suffix(".request.json") for path in paths[1:]]
    # This loop and all limits complete before find_gh, auth, or any transport.
    for index, (path, body) in enumerate(zip(paths, parts)):
        _atomic_text(path, body)
        if index:
            request = json.dumps({"body": body}, ensure_ascii=True, separators=(",", ":")) + "\n"
            _atomic_text(request_paths[index], request)
    receipt_path = body_path.with_name(f"{body_path.stem}.{digest}.submission.json")
    urls, uncertain = [], None
    existing_receipt = receipt_path.exists()
    if existing_receipt:
        urls, uncertain = _load_submission_receipt(receipt_path, digest, count)

    def result(reason):
        status = ("submitted" if len(urls) == count else "uncertain" if uncertain is not None
                  else "partial" if urls else "prepared")
        return {"status": status, "reason": reason, "body_path": paths[0], "body_paths": paths,
                "total_parts": count, "confirmed_parts": len(urls), "uncertain_part": uncertain,
                "part_urls": list(urls), "receipt_path": receipt_path,
                **({"url": urls[0]} if urls else {})}

    def checkpoint(initial=False):
        state = {"schema": "lpsd-issue-transfer-v1", "payload_sha256": digest,
                 "total_parts": count, "part_urls": list(urls), "uncertain_part": uncertain}
        try:
            text = json.dumps(state, separators=(",", ":")) + "\n"
            if initial:
                # An exclusive first claim prevents two concurrent submits
                # from creating duplicate issues. Partial claims fail closed.
                with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(text)
            else:
                _atomic_text(receipt_path, text)
        except OSError:
            return False
        return True

    # The receipt is keyed by the exact payload. A repeated explicit submit
    # reports the old transfer; it does not recreate, resume or resend it.
    if existing_receipt:
        return result("existing_transfer")
    if not accepted:
        return result("public_upload_not_accepted")
    gh = find_gh("gh")
    if not gh:
        return result("github_cli_unavailable")
    if time.monotonic() >= expires:
        return result("time_budget")
    environment = os.environ.copy()
    environment.update(GH_HOST="github.com", GH_PROMPT_DISABLED="1", GH_PAGER="cat",
                       GH_TELEMETRY="0", DO_NOT_TRACK="1", GH_NO_UPDATE_NOTIFIER="1",
                       GH_NO_EXTENSION_UPDATE_NOTIFIER="1", NO_COLOR="1")
    environment.pop("GH_DEBUG", None)
    environment.pop("GH_FORCE_TTY", None)
    try:
        authentication = transport([gh, "auth", "status", "--hostname", "github.com"],
                                   env=environment, capture_output=True, text=True,
                                   stdin=subprocess.DEVNULL,
                                   timeout=min(10., max(.01, expires - time.monotonic())), check=False)
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
        return result("github_login_unavailable")
    if authentication.returncode:
        return result("github_login_unavailable")
    env = clean["environment"]
    title = f"Benchmark: {env['os']} {env['process_arch']} ({len(clean['cases'])} cases)"
    next_post = 0.
    for index, body in enumerate(parts):
        # GitHub recommends a pause between mutating requests. It consumes
        # the existing shared deadline, never an additional upload budget.
        delay = max(0., next_post - time.monotonic())
        if expires - time.monotonic() <= delay:
            return result("time_budget")
        if delay:
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                return result("cancelled")
        remaining = expires - time.monotonic()
        if remaining <= 0:
            return result("time_budget")
        if index == 0:
            command = [gh, "issue", "create", "--repo", REPOSITORY,
                       "--title", title, "--body-file", str(paths[0])]
        else:
            issue_number = urls[0].rsplit("/", 1)[1]
            command = [gh, "api", "--hostname", "github.com", "--method", "POST",
                       f"repos/{REPOSITORY}/issues/{issue_number}/comments",
                       "--input", str(request_paths[index])]
        # Persist before starting the process. A hard interruption must leave
        # this part unconfirmed, even if GitHub accepted its POST.
        uncertain = index + 1
        if not checkpoint(initial=index == 0):
            uncertain = None  # This POST has definitely not started.
            return result("local_receipt_unavailable")
        remaining = expires - time.monotonic()
        if remaining <= 0:
            uncertain = None
            saved = checkpoint()
            return result("time_budget" if saved else "local_receipt_unavailable")
        try:
            response = transport(command, env=environment, capture_output=True, text=True,
                                 stdin=subprocess.DEVNULL, timeout=remaining, check=False)
        except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
            return result("github_upload_receipt_unavailable")
        if response.returncode or not isinstance(response.stdout, str):
            return result("github_upload_receipt_unavailable")
        if index == 0:
            url = response.stdout.strip()
            if not _valid_issue_url(url):
                return result("github_upload_receipt_unavailable")
        else:
            # Do not depend on gh issue comment's human-facing stdout.
            # gh api exposes the documented REST JSON creation response.
            try:
                if len(response.stdout.encode("utf-8")) > MAX_LOCAL_BYTES:
                    raise ValueError("Oversized comment response")
                reply = json.loads(response.stdout)
            except (ValueError, UnicodeError):
                return result("github_upload_receipt_unavailable")
            if not isinstance(reply, dict) or integer(reply.get("id"), 1) is None:
                return result("github_upload_receipt_unavailable")
            url = f"{urls[0]}#issuecomment-{reply['id']}"
            if (reply.get("html_url") != url or reply.get("body") != body
                    or reply.get("issue_url") != f"https://api.github.com/repos/{REPOSITORY}/issues/{issue_number}"):
                return result("github_upload_receipt_unavailable")
        if url in urls:
            return result("github_upload_receipt_unavailable")
        urls.append(url)
        uncertain = None
        if not checkpoint():
            return result("local_receipt_unavailable")
        next_post = time.monotonic() + 1.
    return result("")


def load_report(path):
    path = Path(path)
    if path.stat().st_size > MAX_LOCAL_BYTES:
        raise ValueError("Benchmark report exceeds the local input size limit")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def submit(path, accepted=False, *, timeout=45., transport=subprocess.run,
           find_gh=shutil.which):
    """Prepare exact public content first; the flag explicitly opts into upload.

    A mocked transport is used in tests. No real issue is posted by test code.
    Transport stdout is accepted only if it is the expected repository URL;
    neither command stderr nor arbitrary transport output is returned.
    """
    expires = time.monotonic() + max(0., min(45., timeout))
    path = Path(path)
    report = load_report(path)
    # A public sidecar is an explicitly selected input, never an implicit
    # fallback. Preserve results.github-issue.md for results.public.json.
    body_source = path.with_suffix("") if path.name.endswith(".public.json") else path
    body_path = body_source.with_suffix(".github-issue.md")
    parts = issue_parts(report)
    if len(parts) > 1:
        return _submit_parts(report, parts, body_path, accepted, expires, transport, find_gh)
    body_path.write_text(parts[0], encoding="utf-8")
    if not accepted:
        return {"status": "prepared", "reason": "public_upload_not_accepted", "body_path": body_path}
    gh = find_gh("gh")
    if not gh:
        return {"status": "prepared", "reason": "github_cli_unavailable", "body_path": body_path}
    if time.monotonic() >= expires:
        return {"status": "prepared", "reason": "time_budget", "body_path": body_path}
    environment = os.environ.copy()
    environment.update(GH_HOST="github.com", GH_PROMPT_DISABLED="1", GH_PAGER="cat",
                       GH_TELEMETRY="0", DO_NOT_TRACK="1", GH_NO_UPDATE_NOTIFIER="1",
                       GH_NO_EXTENSION_UPDATE_NOTIFIER="1", NO_COLOR="1")
    environment.pop("GH_DEBUG", None)
    environment.pop("GH_FORCE_TTY", None)
    clean = sanitize(report)
    env = clean["environment"]
    title = f"Benchmark: {env['os']} {env['process_arch']} ({len(clean['cases'])} cases)"
    creation_attempted = False
    try:
        # Probe the current login without opening a browser, printing account
        # details or collecting them in the payload. Login itself is a
        # separate, explicitly requested interactive bootstrap step.
        authentication = transport([gh, "auth", "status", "--hostname", "github.com"],
                                   env=environment, capture_output=True, text=True,
                                   stdin=subprocess.DEVNULL,
                                   timeout=min(10., max(.01, expires - time.monotonic())), check=False)
        if authentication.returncode:
            return {"status": "prepared", "reason": "github_login_unavailable", "body_path": body_path}
        remaining = expires - time.monotonic()
        if remaining <= 0:
            return {"status": "prepared", "reason": "time_budget", "body_path": body_path}
        creation_attempted = True
        response = transport([gh, "issue", "create", "--repo", REPOSITORY,
                              "--title", title, "--body-file", str(body_path)],
                             env=environment, capture_output=True, text=True,
                             stdin=subprocess.DEVNULL, timeout=remaining, check=False)
    except (OSError, subprocess.SubprocessError):
        return {"status": "uncertain" if creation_attempted else "prepared",
                "reason": "github_upload_receipt_unavailable" if creation_attempted else "github_login_unavailable",
                "body_path": body_path}
    url = response.stdout.strip() if isinstance(response.stdout, str) else ""
    if response.returncode or not re.fullmatch(r"https://github\.com/MichaDit/lpsd/issues/[1-9][0-9]*", url):
        # A lost response can occur after GitHub created the issue. Never
        # retry automatically or incorrectly promise that nothing was sent.
        return {"status": "uncertain", "reason": "github_upload_receipt_unavailable", "body_path": body_path}
    return {"status": "submitted", "url": url, "body_path": body_path}
