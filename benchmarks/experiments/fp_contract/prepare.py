#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build an isolated strict/FMA pair from one captured source snapshot.

No production file is changed. The existing native build report supplies the
compiler and flags, but its binary is not reused as the strict reference. Both
private binaries compile the same copied C/H files, differing only in whether
-ffp-contract is off or fast.
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
    if ARTIFACTS.is_symlink():
        parser.error("The private artifacts path must not be a symlink.")
    if ARTIFACTS.exists() and not args.replace:
        parser.error("Private artifacts already exist; use --replace only when no experiment is running.")
    files = sorted(source.parent.glob("*.c")) + sorted(source.parent.glob("*.h"))
    # Preserve the layout required by fast_dft.c's relative polyreg.c include.
    files += sorted((ROOT / "lpsd/c_sources").glob("*.c"))
    files += sorted((ROOT / "lpsd/c_sources").glob("*.h"))
    flags = ["-ffp-contract=fast" if flag == "-ffp-contract=off" else flag
             for flag in report["flags"]]
    with tempfile.TemporaryDirectory(prefix="fp-contract-", dir=HERE) as temporary:
        temporary = Path(temporary)
        snapshot = temporary / "source_snapshot"
        sources = {}
        for path in files:
            relative = path.relative_to(ROOT)
            payload = path.read_bytes()
            copied_source = snapshot / relative
            copied_source.parent.mkdir(parents=True, exist_ok=True)
            copied_source.write_bytes(payload)
            sources[str(relative)] = hashlib.sha256(payload).hexdigest()
        manifest_bytes = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
        snapshot_hash = hashlib.sha256(manifest_bytes).hexdigest()
        source_copy = snapshot / source.relative_to(ROOT)
        strict = temporary / "baseline-final.so"
        candidate = temporary / "experimental-fma.so"
        common = dict(report, source_sha256=sources,
            source_snapshot_sha256=snapshot_hash,
            source_snapshot_directory="source_snapshot",
            baseline_provenance="Fresh private strict build from the shared source snapshot; not the installed binary.",
            installed_library_sha256=report["binary_sha256"],
            installed_library_role="Compiler/flags provenance only; not the reference for new measurements.")
        reports = []
        for target_file, target_flags in ((strict, report["flags"]), (candidate, flags)):
            command = compiler + target_flags + [str(source_copy), "-o", str(target_file)] + report["link_flags"]
            subprocess.run(command, check=True)
            reports.append(dict(common, library=target_file.name, flags=target_flags,
                binary_sha256=sha256(target_file), experimental=True,
                compile_command=command))
        baseline_report, candidate_report = reports
        candidate_report["baseline_sha256"] = baseline_report["binary_sha256"]
        for name, content in (("baseline-final", baseline_report),
                              ("experimental-fma", candidate_report)):
            (temporary / (name + ".build.json")).write_text(json.dumps(content, indent=2)+"\n")
        # Install the completed private pair only after both builds succeed.
        # Production paths are never written, and old mapped files are never
        # truncated. Concurrent experiment execution is explicitly unsupported.
        if ARTIFACTS.exists():
            shutil.rmtree(ARTIFACTS)
        ARTIFACTS.mkdir(exist_ok=True)
        for path in temporary.iterdir():
            path.replace(ARTIFACTS / path.name)
    print(json.dumps({"artifacts": str(ARTIFACTS),
                      "baseline_sha256": baseline_report["binary_sha256"],
                      "installed_library_sha256": report["binary_sha256"],
                      "source_snapshot_sha256": snapshot_hash,
                      "candidate_sha256": candidate_report["binary_sha256"]}))


if __name__ == "__main__":
    main()
