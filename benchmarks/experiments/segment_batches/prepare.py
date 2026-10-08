# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a private native baseline/candidate pair for segment experiments.

The client uses the baseline Python API for both libraries, isolating native
segment changes from any later Python dispatch or estimator additions.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from lpsd_fast.build import build_shared

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE_REF = "3f2864391db3bfb22942d99106026358fef5fe96"
CLIENT_FILES = ("__init__.py", "api.py", "planning.py", "_windows.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def baseline_blob(revision, relative):
    return subprocess.check_output(
        ["git", "show", f"{revision}:{relative}"], cwd=ROOT,
    )


def normalized(relative):
    # Resolve quoted includes without allowing a source snapshot to leave
    # the repository's two known native source directories.
    parts = []
    for part in PurePosixPath(relative).parts:
        if part == "..":
            if not parts:
                raise ValueError("Native include leaves the repository")
            parts.pop()
        elif part != ".":
            parts.append(part)
    value = PurePosixPath(*parts).as_posix()
    if not value.startswith(("lpsd_fast/_native/", "lpsd/c_sources/")):
        raise ValueError(f"Unexpected native include: {value}")
    return value


def snapshot_native(destination, read):
    pending = ["lpsd_fast/_native/fast_dft.c"]
    sources = {}
    while pending:
        relative = pending.pop()
        if relative in sources:
            continue
        data = read(relative)
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        sources[relative] = digest(data)
        for include in re.findall(rb'^\s*#\s*include\s+"([^"\n]+)"', data, re.MULTILINE):
            joined = PurePosixPath(relative).parent / include.decode("utf-8")
            pending.append(normalized(joined.as_posix()))
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--native", action="store_true", help="Compile for this CPU; omit to test portable dispatch.")
    args = parser.parse_args(argv)
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.baseline_ref}^{{commit}}"],
        cwd=ROOT, text=True,
    ).strip()
    client = artifacts / "client" / "lpsd_fast"
    client.mkdir(parents=True, exist_ok=True)
    client_sources = {}
    for filename in CLIENT_FILES:
        relative = "lpsd_fast/" + filename
        data = baseline_blob(revision, relative)
        (client / filename).write_bytes(data)
        client_sources[relative] = digest(data)
    report = {
        "schema_version": 1,
        "baseline_revision": revision,
        "candidate_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "candidate_worktree_diff": subprocess.check_output(
            ["git", "diff", "--", "lpsd_fast/_native"], cwd=ROOT, text=True,
        ),
        "client_directory": str(client.relative_to(artifacts)),
        "client_sources": client_sources,
        "builds": {},
    }
    for name, read in (
        ("baseline", lambda relative: baseline_blob(revision, relative)),
        ("candidate", lambda relative: (ROOT / relative).read_bytes()),
    ):
        snapshot = artifacts / (name + "-source")
        sources = snapshot_native(snapshot, read)
        target = (client / "_native" / "liblpsd_fast.so") if name == "baseline" else artifacts / "candidate.so"
        build_shared(snapshot / "lpsd_fast/_native/fast_dft.c", target, simd=True, native=args.native)
        metadata = json.loads(target.with_suffix(".build.json").read_text())
        report["builds"][name] = {
            "binary": str(target.relative_to(artifacts)),
            "build": metadata,
            "source_directory": str(snapshot.relative_to(artifacts)),
            "source_sha256": sources,
            "source_snapshot_sha256": digest(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()),
        }
    (artifacts / "builds.json").write_text(json.dumps(report, indent=2) + "\n")
    print(artifacts / "builds.json")


if __name__ == "__main__":
    main()
