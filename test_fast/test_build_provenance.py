"""Reject stale native source attribution, while identifying older reports."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmarks.bench_rolling_boxcar import evidence
from lpsd_fast import build as builder


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_manifest_follows_relative_includes_and_breaks_cycles(tmp_path):
    source = tmp_path / "source" / "probe.c"
    header = source.parent / "inc dir" / "value.h"
    common = tmp_path / "common.h"
    header.parent.mkdir(parents=True)
    source.write_text('#include "inc dir/value.h"\n#include <stdint.h>\n')
    header.write_text('#include "../probe.c"\n#include "../../common.h"\n')
    common.write_text("#define VALUE 7.0\n")
    expected = {"probe.c": digest(source), "inc dir/value.h": digest(header),
                "../common.h": digest(common)}
    assert builder.source_manifest(source) == expected
    common.write_text("#define VALUE 8.0\n")
    assert builder.source_manifest(source)["../common.h"] != expected["../common.h"]


def test_actual_build_binds_recursive_project_sources_to_binary(tmp_path):
    source = tmp_path / "probe.c"
    header = tmp_path / "value.h"
    source.write_text('#include "value.h"\ndouble provenance_probe(void) { return VALUE; }\n')
    header.write_text("#define VALUE 7.0\n")
    target = tmp_path / "compiled" / "probe.so"
    builder.build_shared(source, target)
    report = json.loads(target.with_suffix(".build.json").read_text())
    assert report["source_sha256"] == {"probe.c": digest(source), "value.h": digest(header)}
    assert report["binary_sha256"] == digest(target)
    library = ct.CDLL(str(target))
    library.provenance_probe.restype = ct.c_double
    assert library.provenance_probe() == 7.0


def test_source_edit_during_build_preserves_previous_binary_and_report(tmp_path, monkeypatch):
    source = tmp_path / "probe.c"
    header = tmp_path / "value.h"
    source.write_text('#include "value.h"\ndouble provenance_probe(void) { return VALUE; }\n')
    header.write_text("#define VALUE 7.0\n")
    target = tmp_path / "probe.so"
    target.write_bytes(b"previous library")
    report = target.with_suffix(".build.json")
    report.write_bytes(b"previous report")
    original_run = builder.subprocess.run

    def run(command, *args, **kwargs):
        result = original_run(command, *args, **kwargs)
        if str(source) in command:
            header.write_text("#define VALUE 8.0\n")
        return result

    monkeypatch.setattr(builder.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="sources changed during native compilation"):
        builder.build_shared(source, target)
    assert target.read_bytes() == b"previous library"
    assert report.read_bytes() == b"previous report"
    assert not list(tmp_path.glob("lpsd-build-*"))


def evidence_fixture(tmp_path, monkeypatch, *, bound):
    package = tmp_path / "lpsd_fast"
    native = package / "_native"
    native.mkdir(parents=True)
    initialization = package / "__init__.py"
    initialization.write_text("# fixture package\n")
    source = native / "fast_dft.c"
    common = tmp_path / "lpsd" / "c_sources" / "polyreg.c"
    common.parent.mkdir(parents=True)
    source.write_text('#include "../../lpsd/c_sources/polyreg.c"\n')
    common.write_text("/* dependency before editing */\n")
    library = native / "liblpsd_fast.so"
    library.write_bytes(b"fixture binary")
    report = {"source": source.name, "binary_sha256": digest(library)}
    if bound:
        report["source_sha256"] = builder.source_manifest(source)
    report_path = library.with_suffix(".build.json")
    report_path.write_text(json.dumps(report))
    module = SimpleNamespace(__name__="lpsd_evidence_fixture", __file__=str(initialization),
                             __version__="fixture")
    native_api = SimpleNamespace(_LIB=SimpleNamespace(_name=str(library)), _native=lambda: None)
    monkeypatch.setitem(sys.modules, module.__name__ + ".api", native_api)
    return module, common, library, report_path


def test_evidence_accepts_bound_sources_and_records_actual_shared_helpers(tmp_path, monkeypatch):
    module, common, library, _ = evidence_fixture(tmp_path, monkeypatch, bound=True)
    report = evidence(module)
    assert report["native_source_binding"] == "verified"
    assert report["native_source_sha256"]["../../lpsd/c_sources/polyreg.c"] == digest(common)
    for name in ("lpsd._helpers", "lpsd.flattop"):
        actual = Path(sys.modules[name].__file__).resolve()
        assert report["shared_python_sources"][name] == {"path": str(actual), "sha256": digest(actual)}
    library.write_bytes(b"a different binary")
    with pytest.raises(AssertionError, match="build report does not match"):
        evidence(module)


def test_evidence_rejects_includes_edited_before_benchmark(tmp_path, monkeypatch):
    module, common, _, _ = evidence_fixture(tmp_path, monkeypatch, bound=True)
    common.write_text("/* source changed after compilation */\n")
    with pytest.raises(AssertionError, match="sources used to build"):
        evidence(module)


def test_evidence_explicitly_marks_legacy_report_without_changing_it(tmp_path, monkeypatch):
    module, common, _, report_path = evidence_fixture(tmp_path, monkeypatch, bound=False)
    original_report = report_path.read_bytes()
    report = evidence(module)
    assert report["native_source_binding"] == "unavailable_legacy_build_report"
    assert report["native_source_sha256"]["../../lpsd/c_sources/polyreg.c"] == digest(common)
    assert report_path.read_bytes() == original_report
