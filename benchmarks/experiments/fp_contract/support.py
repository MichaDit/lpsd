# SPDX-License-Identifier: GPL-3.0-or-later
"""Load isolated experiment binaries without replacing a production library."""
import ctypes as ct
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def load_libraries():
    from lpsd_fast import api

    # Reuse the package's ABI declarations; this resolves but does not execute
    # the ordinary production native library. Calls below use separate builds.
    api._native()
    declarations = api._LIB
    libraries, reports = {}, {}
    for name, prefix in (("baseline", "baseline-final"),
                         ("fma", "experimental-fma")):
        report = json.loads((ARTIFACTS / (prefix + ".build.json")).read_text())
        path = ARTIFACTS / report["library"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["binary_sha256"]:
            raise RuntimeError(f"Binary/report mismatch: {path}")
        library = ct.CDLL(str(path))
        for symbol, declaration in list(vars(declarations).items()):
            if isinstance(declaration, ct._CFuncPtr):
                function = getattr(library, symbol)
                function.argtypes = declaration.argtypes
                function.restype = declaration.restype
        libraries[name], reports[name] = library, report
    if reports["fma"]["baseline_sha256"] != reports["baseline"]["binary_sha256"]:
        raise RuntimeError("Candidate was not built against this recorded baseline.")
    if reports["fma"]["source_sha256"] != reports["baseline"]["source_sha256"]:
        raise RuntimeError("The private pair did not use the same captured sources.")
    sources = reports["baseline"]["source_sha256"]
    manifest = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    snapshot_hash = hashlib.sha256(manifest).hexdigest()
    for report in reports.values():
        if report["source_snapshot_sha256"] != snapshot_hash:
            raise RuntimeError("Source snapshot manifest hash mismatch.")
        snapshot = ARTIFACTS / report["source_snapshot_directory"]
        for relative, expected_hash in sources.items():
            if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != expected_hash:
                raise RuntimeError(f"Captured source changed: {relative}")
    return libraries["baseline"], libraries["fma"], reports
