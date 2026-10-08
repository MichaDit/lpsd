# SPDX-License-Identifier: GPL-3.0-or-later
"""Small native-runner comparison; writes JSON evidence, never a timing gate.

Run after ``make compile``. The ordinary API repetitions and the additional
profiled call are measured separately by bench_lpsd.py. No signal files are saved.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--output-directory", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    if args.n < 16:
        parser.error("--n must be at least 16")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    import lpsd
    import lpsd_fast
    from lpsd_fast.api import available_workers

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workers = available_workers()
    reports = [
        Path(lpsd.__file__).parent / "ltpda_dft.build.json",
        Path(lpsd_fast.__file__).parent / "_native" / "liblpsd_fast.build.json",
    ]
    environment = {
        "schema_version": 1,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "available_workers": workers,
        "numpy_long_double": {
            "bytes": np.dtype(np.longdouble).itemsize,
            "mantissa_bits_including_leading_bit": np.finfo(np.longdouble).nmant + 1,
            "epsilon": str(np.finfo(np.longdouble).eps),
        },
        "github": {key: os.environ.get(key) for key in (
            "GITHUB_SHA", "GITHUB_RUN_ID", "RUNNER_OS", "RUNNER_ARCH", "ImageOS", "ImageVersion"
        )},
        "build_reports": [json.loads(path.read_text()) for path in reports],
        "note": "Shared CI hardware and small inputs; no cross-machine speedup or hard timing claim.",
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    # Force both benchmark backends to this checkout, matching the source-tree
    # numerical tests and the build reports above, even with another installation.
    child_environment = dict(os.environ, PYTHONPATH=str(root))
    common = [sys.executable, str(root / "benchmarks" / "bench_lpsd.py"), "--n", str(args.n)]
    commands = [common + ["--backend", "original", "--repeats", "1",
                          "--output", str(output / "original-w1.json")]]
    for count in sorted({1, workers}):
        commands.append(common + [
            "--backend", "fast", "--kernel", "auto", "--workers", str(count),
            "--repeats", "3", "--profile", "--output", str(output / f"fast-auto-w{count}.json"),
        ])
    for command in commands:
        subprocess.run(command, cwd=root, env=child_environment, check=True)


if __name__ == "__main__":
    main()
