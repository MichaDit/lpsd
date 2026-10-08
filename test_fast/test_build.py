"""Build policy checks against the actual compiled reference and fast libraries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative", (
    "lpsd/ltpda_dft.build.json", "lpsd_fast/_native/liblpsd_fast.build.json",
))
def test_compiled_library_has_matching_fp_and_build_report(relative):
    path = ROOT / relative
    report = json.loads(path.read_text())
    library = path.parent / report["library"]
    assert hashlib.sha256(library.read_bytes()).hexdigest() == report["binary_sha256"]
    assert "-ffp-contract=off" in report["flags"]
    assert "-ffast-math" not in report["flags"]
    assert report["floating_point"]["long_double_mant_dig"] == np.finfo(np.longdouble).nmant + 1
    assert report["floating_point"]["long_double_bytes"] == np.dtype(np.longdouble).itemsize


def test_native_only_wheel_build_is_rejected_before_compilation(tmp_path):
    environment = dict(os.environ, LPSD_NATIVE="1", CC="lpsd-deliberately-missing-compiler")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_py", "--build-lib", str(tmp_path)],
        cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert result.returncode != 0
    assert "LPSD_NATIVE=1 is only for editable/local builds" in result.stdout
