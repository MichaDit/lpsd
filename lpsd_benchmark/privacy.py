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
    body_path = path.with_suffix(".github-issue.md")
    body_path.write_text(issue_body(report), encoding="utf-8")
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
