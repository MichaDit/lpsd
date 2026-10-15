#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build and test this checkout locally; never install, upload or dispatch CI.

Use the existing Python environment with the project's test dependencies
already installed. Logs and reports are retained in a fresh local directory.
A successful run is qualification of this host only; skips are not passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def junit_counts(path):
    """Count executed cases, distinguishing expected failures from skips."""
    cases = list(ET.parse(path).iter("testcase"))
    counts = dict(tests=len(cases), passed=0, failed=0, errors=0, skipped=0, xfailed=0)
    for case in cases:
        if case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("skipped") is not None:
            kind = case.find("skipped").get("type", "")
            counts["xfailed" if kind == "pytest.xfail" else "skipped"] += 1
        else:
            counts["passed"] += 1
    return counts


def stop_owned_group(process):
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass  # Kernel I/O can be uninterruptible; never wait without a bound.


def run_stage(command, root, log, timeout):
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    options = (dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt"
               else dict(start_new_session=True))
    start = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, **options)
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_owned_group(process)
            returncode = process.returncode
        except BaseException:
            stop_owned_group(process)
            raise
    return dict(returncode=returncode, timed_out=timed_out,
                wall_seconds=time.monotonic()-start,
                status="passed" if returncode == 0 and not timed_out else "failed")


def source_hashes(root):
    result = {}
    for name in ("lpsd", "lpsd_fast", "lpsd_benchmark", "benchmarks", "scripts", "test", "test_fast"):
        for path in sorted((root / name).rglob("*")):
            if (path.suffix in (".py", ".c", ".h") and path.is_file() and not path.is_symlink()
                    and "__pycache__" not in path.parts):
                result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("build_native.py", "setup.py", "setup.cfg", "pyproject.toml", "Makefile"):
        path = root / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-parent", type=Path, default=Path(tempfile.gettempdir()),
                     help="Parent of a fresh retained local result directory")
    cli.add_argument("--stage-timeout", type=int, default=3600,
                     help="Maximum seconds per build/test stage, 1..86400")
    args = cli.parse_args(argv)
    if not 1 <= args.stage_timeout <= 86400:
        cli.error("stage-timeout must be between 1 and 86400 seconds")
    required = ("build_native.py", "lpsd_fast/build.py", "test_fast/conftest.py", "test/test_lpsd.py")
    if any(not (ROOT / name).is_file() for name in required):
        cli.error("A complete source checkout is required, not a patch-only directory")
    args.output_parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="lpsd-local-validation-", dir=args.output_parent))
    report = dict(schema="lpsd-local-validation-v1", status="running",
                  host=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version()),
                  scope="Current host only; no package installs, uploads or GitHub Actions; skips remain explicit",
                  source_sha256=source_hashes(ROOT), stages=[])
    def save():
        target = directory / "summary.json"
        temporary = directory / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(target)
    stages = [("build-reference", [sys.executable, "build_native.py"], None),
              ("build-fast", [sys.executable, "-m", "lpsd_fast.build"], None)]
    for name, suite in (("regressions", "test_fast"), ("upstream", "test")):
        xml = directory / (name+".xml")
        stages.append((name, [sys.executable, "-m", "pytest", "-o", "addopts=", "-o", "junit_family=legacy",
                              "-q", "-ra", suite, "--junitxml", str(xml)], xml))
    save()
    print(f"Local validation directory: {directory}", flush=True)
    try:
        for name, command, xml in stages:
            print(f"Running {name}", flush=True)
            row = dict(name=name, **run_stage(command, ROOT, directory/(name+".log"), args.stage_timeout))
            if xml is not None:
                try:
                    row["counts"] = junit_counts(xml)
                    if not row["counts"]["tests"] or row["counts"]["failed"] or row["counts"]["errors"]:
                        row["status"] = "failed"
                except (OSError, ET.ParseError):
                    row.update(status="failed", junit="unavailable")
            report["stages"].append(row)
            save()
            if row["status"] != "passed":
                report["status"] = "failed"
                return 1
        report["status"] = "passed"
        return 0
    except KeyboardInterrupt:
        report["status"] = "cancelled"
        return 130
    except OSError:
        report["status"] = "failed"
        return 1
    finally:
        save()
        print(f"Validation {report['status']}; retained summary: {directory/'summary.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
