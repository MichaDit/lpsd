#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Copy a built native baseline and compile an isolated contraction experiment.

No production file is changed. This Linux/x86 experiment deliberately uses the
same compiler, source and flags as the existing --native baseline, except for
changing -ffp-contract=off to fast in its private candidate library.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = HERE / "artifacts"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true",
                        help="Replace prior private artifacts; never run while they are loaded.")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("This archived experiment recipe targets Linux.")
    source = ROOT / "lpsd_fast/_native/fast_dft.c"
    sidecar = source.parent / "liblpsd_fast.build.json"
    baseline = source.parent / "liblpsd_fast.so"
    if not baseline.is_file() or not sidecar.is_file():
        parser.error("Build the ordinary native package first: python -m lpsd_fast.build --native")
    report = json.loads(sidecar.read_text())
    if sha256(baseline) != report["binary_sha256"]:
        raise RuntimeError("The baseline library does not match its build report.")
    if not report.get("native") or report.get("target_clones"):
        parser.error("Use an ordinary --native baseline without portable target clones.")
    if report["flags"].count("-ffp-contract=off") != 1:
        parser.error("The baseline must use exactly one -ffp-contract=off flag.")
    if "-fno-fast-math" not in report["flags"] or "-ffast-math" in report["flags"]:
        parser.error("The baseline must retain -fno-fast-math.")
    compiler = report["compiler"]
    version = subprocess.check_output(compiler + ["--version"], text=True).splitlines()[0]
    target = subprocess.check_output(compiler + ["-dumpmachine"], text=True).strip()
    if version != report["compiler_version"] or target != report["compiler_target"]:
        parser.error("Compiler changed since baseline build; rebuild the baseline first.")
    if not target.startswith(("x86_64-", "i686-", "i386-")):
        parser.error("This archived native recipe was measured only on Linux/x86.")
    if ARTIFACTS.exists() and not args.replace:
        parser.error("Private artifacts already exist; use --replace only when no experiment is running.")
    files = sorted(source.parent.glob("*.c")) + sorted(source.parent.glob("*.h"))
    files += [ROOT / "lpsd/c_sources/polyreg.c"]
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    flags = ["-ffp-contract=fast" if flag == "-ffp-contract=off" else flag
             for flag in report["flags"]]
    with tempfile.TemporaryDirectory(prefix="fp-contract-", dir=HERE) as temporary:
        temporary = Path(temporary)
        copied = temporary / "baseline-final.so"
        shutil.copy2(baseline, copied)
        candidate = temporary / "experimental-fma.so"
        command = compiler + flags + [str(source), "-o", str(candidate)] + report["link_flags"]
        subprocess.run(command, check=True)
        baseline_report = dict(report, library=copied.name)
        candidate_report = dict(report, library=candidate.name, flags=flags,
            binary_sha256=sha256(candidate), baseline_sha256=sha256(copied),
            experimental=True, source_sha256=sources,
            compile_command=command)
        for name, content in (("baseline-final", baseline_report),
                              ("experimental-fma", candidate_report)):
            (temporary / (name + ".build.json")).write_text(json.dumps(content, indent=2)+"\n")
        ARTIFACTS.mkdir(exist_ok=True)
        for path in temporary.iterdir():
            path.replace(ARTIFACTS / path.name)
    print(json.dumps({"artifacts": str(ARTIFACTS),
                      "baseline_sha256": report["binary_sha256"],
                      "candidate_sha256": candidate_report["binary_sha256"]}))


if __name__ == "__main__":
    main()
