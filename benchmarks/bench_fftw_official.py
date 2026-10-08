#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Record the unmodified FFTW tests/bench; print commands unless --run is given.

The official report is a MINIMUM over repeated execution batches, not an API
median. PATIENT with a finite planning budget is explicitly labelled capped.
Each case uses a fresh process and neither imports nor exports FFTW wisdom.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import statistics
import subprocess
import time


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
REPORT = re.compile(rf"^\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*$")


def digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stop_process(process):
    """Stop the wrapper and its benchmark child on timeout or interruption."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass  # The child may finish between communicate() and termination.


def run_case(command, cwd, env, timeout, stem, output):
    started = time.perf_counter()
    row = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=os.name == "posix")
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            row["timed_out"] = False
        except subprocess.TimeoutExpired:
            # Also stop a child launched by the libtool wrapper, rather than
            # leaving benchmark work running during the next case.
            stop_process(process)
            stdout, stderr = process.communicate()
            row["timed_out"] = True
        except BaseException:
            stop_process(process)
            process.communicate()
            raise
        row["returncode"] = process.returncode
    except OSError as error:
        stdout, stderr = "", str(error)
        row.update(returncode=None, timed_out=False, launch_error=str(error))
    row["process_wall_s"] = time.perf_counter() - started
    row["stdout_file"], row["stderr_file"] = stem + ".stdout.txt", stem + ".stderr.txt"
    (output / row["stdout_file"]).write_text(stdout)
    (output / row["stderr_file"]).write_text(stderr)
    row["option_warning"] = bool(re.search(
        r"unknown user option|threads not supported|ignoring threads_callback",
        stdout + stderr, flags=re.IGNORECASE))
    row["ok"] = row["returncode"] == 0 and not row["option_warning"] and not row["timed_out"]
    row["reported_thread_counts"] = sorted({
        int(value) for value in re.findall(r"^NTHREADS = (\d+)\s*$", stdout, re.MULTILINE)
    })
    records = [tuple(float(value) for value in match.groups())
               for line in stdout.splitlines() if (match := REPORT.match(line))]
    if (len(records) == 1 and all(math.isfinite(value) for value in records[0])
            and records[0][0] > 0 and records[0][1] > 0 and records[0][2] >= 0):
        mflops, execution, setup = records[0]
        row["official_report"] = {
            "normalized_mflops_not_actual_flop_count": mflops,
            "minimum_execution_seconds_per_fft": execution,
            "setup_seconds": setup,
        }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Configured Unix FFTW release build containing tests/bench.")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--release-url", help="Official archive URL for provenance; no download is performed.")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000_000, 10_000_000, 30_000_000])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--planners", nargs="+", choices=["estimate", "patient"],
                        default=["estimate", "patient"])
    parser.add_argument("--planning-seconds", type=float, default=5.0,
                        help="PATIENT time limit; -1 means uncapped, explicitly recorded.")
    parser.add_argument("--batch-min-seconds", type=float, default=0.1)
    parser.add_argument("--batch-repeats", type=int, default=8)
    parser.add_argument("--process-repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.)
    parser.add_argument("--verify", action="store_true",
                        help="Also verify each size/thread pair, with ESTIMATE and three rounds.")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if (any(value < 1 for value in args.sizes + args.threads)
            or args.batch_repeats < 1 or args.process_repeats < 1
            or not all(math.isfinite(value) for value in
                       (args.planning_seconds, args.batch_min_seconds, args.timeout))
            or args.batch_min_seconds <= 0 or args.timeout <= 0
            or (args.planning_seconds < 0 and args.planning_seconds != -1)):
        parser.error("Require positive sizes/threads/repetitions/intervals; planning budget is >=0 or -1.")
    source = args.source.resolve()
    bench = source / "tests" / "bench"  # Libtool wrapper selects the local built libraries.
    cases = []
    if args.verify:
        for size in args.sizes:
            for threads in args.threads:
                cases.append(({"kind": "verify", "n": size, "threads": threads},
                              [str(bench), "-oestimate", f"-onthreads={threads}",
                               "--verify-rounds=3", "-v2", "-y", f"orf{size}"]))
    for repetition in range(args.process_repeats):
        group = []
        for size in args.sizes:
            for threads in args.threads:
                for planner in args.planners:
                    label = planner + ("_capped" if planner == "patient"
                                       and args.planning_seconds >= 0 else "")
                    command = [str(bench), f"-o{planner}", f"-onthreads={threads}"]
                    if planner == "patient":
                        command.append(f"-otimelimit={args.planning_seconds}")
                    command += [f"--time-min={args.batch_min_seconds}",
                                f"--time-repeat={args.batch_repeats}", "-v2",
                                "--report-benchmark", "-s", f"orf{size}"]
                    group.append(({"kind": "speed", "n": size, "threads": threads,
                                   "planner": label, "process_repetition": repetition}, command))
        cases += group if repetition % 2 == 0 else list(reversed(group))
    if not args.run:
        for _, command in cases:
            print(shlex.join(command))
        return
    if not bench.is_file():
        parser.error(f"Build the official FFTW release first: missing {bench}")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    library_path = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
    env[library_path] = os.pathsep.join(
        [str(source / ".libs"), str(source / "threads" / ".libs")]
        + ([env[library_path]] if env.get(library_path) else []))
    archive = args.archive or source.parent / (source.name + ".tar.gz")
    provenance_paths = [bench, source / "tests" / ".libs" / "bench",
                        source / ".libs" / "libfftw3.so",
                        source / "threads" / ".libs" / "libfftw3_threads.so",
                        source / ".libs" / "libfftw3.dylib",
                        source / "threads" / ".libs" / "libfftw3_threads.dylib",
                        source / "threads" / ".libs" / "libfftw3_omp.so",
                        source / "threads" / ".libs" / "libfftw3_omp.dylib",
                        source / "tests" / "fftw-bench.c",
                        source / "libbench2" / "speed.c", source / "libbench2" / "report.c"]
    report = {
        "schema_version": 1,
        "purpose": "Unmodified official FFTW R2C kernel benchmark; not an LPSD estimator.",
        "source_directory_name": source.name,
        "release_url": args.release_url,
        "archive_sha256": digest(archive),
        "runner_sha256": digest(__file__),
        "files_sha256": {str(path.relative_to(source)): digest(path) for path in provenance_paths},
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "options": {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()},
        "measurement_definition": {
            "transform": "double precision, real forward, out of place, exact requested length",
            "input": "FFTW bench uses zero-filled arrays for speed; numerical verification is separate.",
            "planning": "Fresh process per case; no wisdom import/export. PATIENT budget is approximate.",
            "execution": "Minimum per-FFT time over official batches; individual batch times are not printed.",
            "setup": "Official setup excludes the preliminary ESTIMATE capability plan, allocation, "
                     "thread initialization, executable startup and API postprocessing.",
            "process_wall": "Whole benchmark child-process wall time, including planning and repeated FFTs.",
            "mflops": "Conventional 2.5*N*log2(N)/time scaling, not measured FLOP count.",
        },
        "results": [],
    }
    report["bench_info"] = run_case([str(bench), "--info-all", "--print-precision"],
                                     source, env, args.timeout, "bench-info", output)
    info_lines = [line.strip() for line in
                  (output / report["bench_info"]["stdout_file"]).read_text().splitlines()
                  if line.strip()]
    report["reported_precision"] = info_lines[-1] if info_lines else None
    if not report["bench_info"]["ok"] or report["reported_precision"] != "double":
        report["all_commands_succeeded"] = False
        report["error"] = "Expected a working double-precision FFTW bench executable."
        save(output / "report.json", report)
        raise SystemExit(1)
    save(output / "report.json", report)
    for index, (case, command) in enumerate(cases):
        label = f"{index:03d}-{case['kind']}-n{case['n']}-w{case['threads']}"
        result = dict(case, **run_case(command, source, env, args.timeout, label, output))
        if case["kind"] == "speed" and "official_report" not in result:
            result["ok"] = False
        if case["threads"] > 1 and result["reported_thread_counts"] != [case["threads"]]:
            result["ok"] = False
            result["thread_validation_error"] = "Requested thread count was not confirmed by FFTW's plan output."
        report["results"].append(result)
        save(output / "report.json", report)
        print(json.dumps(result), flush=True)
    groups = {}
    for row in report["results"]:
        if row["kind"] == "speed" and row["ok"]:
            key = (row["n"], row["threads"], row["planner"])
            groups.setdefault(key, []).append(row["official_report"])
    report["summary"] = [
        {"n": size, "threads": threads, "planner": planner, "processes": len(rows),
         "statistic": "single_process_official_minimum" if len(rows) == 1
                      else "median_of_process_minima",
         "median_of_official_minimum_execution_seconds": statistics.median(
             row["minimum_execution_seconds_per_fft"] for row in rows),
         "execution_minima_seconds": [row["minimum_execution_seconds_per_fft"] for row in rows],
         "setup_seconds": [row["setup_seconds"] for row in rows]}
        for (size, threads, planner), rows in groups.items()]
    report["all_commands_succeeded"] = report["bench_info"]["ok"] and all(
        row["ok"] for row in report["results"])
    save(output / "report.json", report)
    if not report["all_commands_succeeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
