"""Build policy checks against the actual compiled reference and fast libraries."""
import hashlib
import json
import os
from pathlib import Path
import runpy
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


@pytest.fixture
def packaging_namespace(monkeypatch):
    import setuptools

    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    return runpy.run_path(str(ROOT / "setup.py"))


@pytest.mark.parametrize("architecture", ("arm64", "x86_64"))
@pytest.mark.parametrize("built", (False, True))
def test_macos_wheel_tag_uses_native_architecture(packaging_namespace, monkeypatch,
                                                architecture, built):
    namespace = packaging_namespace
    monkeypatch.setattr(namespace["bdist_wheel"], "get_tag", lambda self:
                        ("cp312", "cp312", "macosx_11_0_universal2"))
    distribution = namespace["NativeDistribution"]()
    if built:
        distribution.lpsd_macos_architecture = architecture

        def unexpected_compiler_query():
            pytest.fail("A verified build should reuse its recorded architecture")

        resolver = unexpected_compiler_query
    else:
        # Editable builds calculate their tag before the libraries exist.
        resolver = lambda: architecture
    monkeypatch.setattr(namespace["platform_build"], "macos_native_architecture",
                        resolver)
    wheel = namespace["PlatformWheel"](distribution)
    assert wheel.get_tag() == ("py3", "none", "macosx_11_0_" + architecture)


@pytest.mark.parametrize("target,expected", (
    ("arm64-apple-darwin24.6.0", "arm64"),
    ("aarch64-apple-darwin24.6.0", "arm64"),
    ("x86_64-apple-darwin24.6.0", "x86_64"),
))
def test_macos_native_target_verifies_both_libraries(packaging_namespace, monkeypatch,
                                                   target, expected):
    builder = packaging_namespace["platform_build"]
    inspected = []

    def output(compiler, arguments):
        if arguments == ["-dumpmachine"]:
            return target
        assert compiler == ["lipo"]
        assert arguments[0] == "-archs"
        inspected.append(arguments[1])
        return expected

    monkeypatch.setattr(builder, "_compiler_output", output)
    assert builder.macos_native_architecture(("reference.so", "fast.so")) == expected
    assert inspected == ["reference.so", "fast.so"]


@pytest.mark.parametrize("actual", ("x86_64", "x86_64 arm64"))
def test_macos_rejects_mismatched_or_fat_library(packaging_namespace, monkeypatch,
                                              actual):
    builder = packaging_namespace["platform_build"]
    monkeypatch.setattr(builder, "_compiler_output", lambda compiler, arguments:
                        "arm64-apple-darwin24.6.0" if arguments == ["-dumpmachine"]
                        else actual)
    with pytest.raises(RuntimeError, match="Expected a native arm64 library"):
        builder.macos_native_architecture(("fast.so",))


def test_macos_rejects_non_apple_compiler(packaging_namespace, monkeypatch):
    builder = packaging_namespace["platform_build"]
    monkeypatch.setattr(builder, "_compiler_output", lambda compiler, arguments:
                        "x86_64-linux-gnu")
    with pytest.raises(RuntimeError, match="Unsupported native macOS compiler target"):
        builder.macos_native_architecture()


def test_linux_wheel_platform_is_unchanged(packaging_namespace, monkeypatch):
    namespace = packaging_namespace
    monkeypatch.setattr(namespace["bdist_wheel"], "get_tag", lambda self:
                        ("cp312", "cp312", "linux_x86_64"))
    wheel = namespace["PlatformWheel"](namespace["NativeDistribution"]())
    assert wheel.get_tag() == ("py3", "none", "linux_x86_64")
