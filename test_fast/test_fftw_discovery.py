"""Isolated FFTW discovery contracts; numerical kernels are not mocked here.

Discovery tests use narrow loader doubles. The optional integration tests
below bind the real installed FFTW C library and execute actual transforms.
Extracting the stdlib-only discovery functions avoids an unrelated native
LPSD build requirement for these loader tests.
"""
from __future__ import annotations

import ast
import ctypes
import ctypes.util
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def discovery_namespace(path=None):
    path = path or ROOT / "lpsd_benchmark/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_FFTWUnavailable", "_initialize_fftw_wheel", "_load_fftw"}
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(selected) == len(names)
    namespace = {"ctypes": ctypes, "metadata": metadata, "Path": Path, "re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def loader():
    return discovery_namespace()


def install_discovery(monkeypatch, libraries, wheel=(), *, path_class=Path):
    queries = []
    def find(name):
        queries.append(name)
        return libraries.get(name)
    monkeypatch.setattr(ctypes.util, "find_library", find)
    monkeypatch.setattr(metadata, "distribution", lambda name: SimpleNamespace(
        files=list(wheel), locate_file=lambda entry: path_class("/wheel") / entry))
    return queries


def test_system_success_does_not_read_wheel_metadata_or_probe_openmp(loader, monkeypatch):
    queries = install_discovery(monkeypatch, {"fftw3": "core", "fftw3_threads": "pthread"})
    monkeypatch.setattr(metadata, "distribution", lambda name: pytest.fail("unneeded wheel scan"))
    calls = []
    def load(core, thread, count):
        calls.append((core, thread, count))
        return SimpleNamespace(threads=count)
    result = loader["_load_fftw"](SimpleNamespace(FFTWLibrary=load), 2)
    assert queries == ["fftw3", "fftw3_threads"]
    assert calls == [("core", None, 1), ("core", "pthread", 2)]
    assert result.threading_diagnostics["main_candidates"] == 1


def test_single_thread_system_does_not_probe_optional_runtimes(loader, monkeypatch):
    queries = install_discovery(monkeypatch, {"fftw3": "core"})
    monkeypatch.setattr(metadata, "distribution", lambda name: pytest.fail("unneeded wheel scan"))
    result = loader["_load_fftw"](SimpleNamespace(FFTWLibrary=lambda *args: SimpleNamespace(threads=1)), 1)
    assert queries == ["fftw3"]
    assert result.threading_diagnostics["status"] == "single_thread_requested"
    assert result.threading_diagnostics["thread_attempts"] == 0


def test_openmp_is_probed_after_pthread_failure(loader, monkeypatch):
    queries = install_discovery(monkeypatch, {"fftw3": "core", "fftw3_threads": "pthreads",
                                             "fftw3_omp": "libfftw3_omp.so"})
    calls = []
    def load(core, thread, count):
        calls.append((core, thread, count))
        if thread == "pthreads":
            raise OSError("private filename")
        return SimpleNamespace(threads=count)
    result = loader["_load_fftw"](SimpleNamespace(FFTWLibrary=load), 3)
    assert queries == ["fftw3", "fftw3_threads", "fftw3_omp"]
    assert calls[-1] == ("core", "libfftw3_omp.so", 3)
    assert result.threading_diagnostics["thread_runtime"] == "openmp"
    assert result.threading_diagnostics["failures"] == {"thread_load_failed": 1}
    assert "private" not in json.dumps(result.threading_diagnostics)


@pytest.mark.parametrize("failure_at", ["core", "thread"])
@pytest.mark.parametrize("path_class", [PurePosixPath, PureWindowsPath], ids=["posix", "windows"])
def test_wheel_initialization_retries_only_the_same_failed_library_pair(loader, monkeypatch, failure_at, path_class):
    wheel = ("libs/libfftw3-core.so", "libs/libfftw3_omp-runtime.so")
    # Exercise both path flavours on every host. The wheel locator returns
    # native path strings, so literal POSIX separators are not a contract.
    loader["Path"] = path_class
    install_discovery(monkeypatch, {}, wheel, path_class=path_class)
    expected_core, expected_thread = (str(path_class("/wheel") / entry) for entry in wheel)
    ready, calls, initialized = [False], [], []
    def initialize():
        initialized.append(True)
        ready[0] = True
        return True
    loader["_initialize_fftw_wheel"] = initialize
    def load(core, thread, count):
        calls.append((core, thread, count))
        if not ready[0] and (failure_at == "core" or thread is not None):
            raise OSError("PRIVATE transitive dependency")
        return SimpleNamespace(threads=count)
    result = loader["_load_fftw"](SimpleNamespace(FFTWLibrary=load), 2)
    assert len(initialized) == 1 and result.threads == 2
    details = result.threading_diagnostics
    assert details["main_source"] == details["thread_source"] == "wheel"
    assert details["thread_runtime"] == "openmp"
    assert details["failures"] == {"main_load_failed" if failure_at == "core" else "thread_load_failed": 1}
    assert all(core == expected_core for core, thread, count in calls)
    assert all(thread in (None, expected_thread) for core, thread, count in calls)
    assert calls[0] == calls[1] if failure_at == "core" else calls[1] == calls[2]
    assert "PRIVATE" not in json.dumps(details)


@pytest.mark.parametrize("initializer_result", [False, True])
def test_broken_wheel_dependency_initialization_is_bounded(loader, monkeypatch, initializer_result):
    install_discovery(monkeypatch, {}, ("libs/libfftw3-a.so", "libs/libfftw3-b.so"))
    initialized, calls = [], []
    loader["_initialize_fftw_wheel"] = lambda: initialized.append(True) or initializer_result
    def broken(*args):
        calls.append(args)
        raise OSError("PRIVATE dependency")
    with pytest.raises(loader["_FFTWUnavailable"]) as caught:
        loader["_load_fftw"](SimpleNamespace(FFTWLibrary=broken), 2)
    assert len(initialized) == 1 and len(calls) == (3 if initializer_result else 2)
    assert caught.value.diagnostics["actual"] == 0
    assert "PRIVATE" not in json.dumps(caught.value.diagnostics)


@pytest.mark.parametrize("error", [AttributeError, RuntimeError, ValueError])
def test_non_load_errors_do_not_import_wheel(loader, monkeypatch, error):
    install_discovery(monkeypatch, {}, ("libs/libfftw3-a.so",))
    loader["_initialize_fftw_wheel"] = lambda: pytest.fail("not a dependency-load failure")
    def broken(*args):
        raise error("PRIVATE details")
    with pytest.raises(loader["_FFTWUnavailable"]):
        loader["_load_fftw"](SimpleNamespace(FFTWLibrary=broken), 2)


def test_explicit_success_does_not_discover_system_or_wheel(loader, monkeypatch):
    install_discovery(monkeypatch, {})
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: pytest.fail("unneeded system search"))
    monkeypatch.setattr(metadata, "distribution", lambda name: pytest.fail("unneeded wheel scan"))
    calls = []
    def load(core, thread, count):
        calls.append((core, thread, count))
        return SimpleNamespace(threads=count)
    result = loader["_load_fftw"](SimpleNamespace(FFTWLibrary=load), 4, "compiled-core")
    assert calls == [("compiled-core", None, 1), ("compiled-core", "compiled-core", 4)]
    assert result.threading_diagnostics["thread_runtime"] == "integrated"


@pytest.mark.parametrize("path_class", [PurePosixPath, PureWindowsPath], ids=["posix", "windows"])
def test_core_and_thread_families_do_not_cross(loader, monkeypatch, path_class):
    loader["Path"] = path_class
    wheel = ("libs/libfftw3-core.so", "libs/libfftw3_omp-companion.so")
    install_discovery(monkeypatch, {"fftw3": "system-core", "fftw3_threads": "system-threads"},
                      wheel, path_class=path_class)
    calls = []
    def load(core, thread, count):
        calls.append((core, thread, count))
        if core == "system-core" and thread:
            raise AttributeError("PRIVATE symbols")
        return SimpleNamespace(threads=count)
    result = loader["_load_fftw"](SimpleNamespace(FFTWLibrary=load), 2)
    assert result.threading_diagnostics["main_source"] == "wheel"
    wheel_paths = {str(path_class("/wheel") / entry) for entry in wheel}
    assert not any(core == "system-core" and thread in wheel_paths
                   or core in wheel_paths and thread == "system-threads" for core, thread, count in calls)


def test_duplicate_wheel_entries_do_not_repeat_attempts(loader, monkeypatch):
    install_discovery(monkeypatch, {}, ("libs/libfftw3-core.so", "libs/libfftw3-core.so",
                                       "libs/libfftw3_threads.so", "libs/libfftw3_threads.so"))
    calls = []
    def load(core, thread, count):
        calls.append((core, thread, count))
        if thread:
            raise AttributeError("no threading")
        return SimpleNamespace(threads=1)
    result = loader["_load_fftw"](SimpleNamespace(FFTWLibrary=load), 2)
    assert len(calls) == 3 and result.threading_diagnostics["main_candidates"] == 1
    assert result.threading_diagnostics["status"] == "fallback_single_thread"


@pytest.mark.parametrize("kind", [ImportError, OSError, RuntimeError, ValueError, AttributeError])
def test_initializer_failure_remains_optional_and_private(loader, monkeypatch, kind):
    import importlib
    def unavailable(name):
        assert name == "pyfftw"
        raise kind("PRIVATE package path")
    monkeypatch.setattr(importlib, "import_module", unavailable)
    assert loader["_initialize_fftw_wheel"]() is False


def test_initializer_uses_only_the_named_installed_package(loader, monkeypatch):
    import importlib
    calls = []
    monkeypatch.setattr(importlib, "import_module", lambda name: calls.append(name))
    assert loader["_initialize_fftw_wheel"]() is True
    assert calls == ["pyfftw"]


class CFFTW:
    """Narrow real-C-API integration adapter, not an LPSD estimator."""
    _thread_runtimes = {}

    def __init__(self, core, thread, count):
        self.lib = ctypes.CDLL(core, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        self.threads = count
        self.threadlib = None
        self.lib.fftw_execute.argtypes = [ctypes.c_void_p]
        self.lib.fftw_execute.restype = None
        self.lib.fftw_destroy_plan.argtypes = [ctypes.c_void_p]
        self.lib.fftw_destroy_plan.restype = None
        self.lib.fftw_plan_dft_r2c_1d.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.lib.fftw_plan_dft_r2c_1d.restype = ctypes.c_void_p
        if thread:
            self.threadlib = ctypes.CDLL(thread, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            self.threadlib.fftw_init_threads.argtypes = []
            self.threadlib.fftw_init_threads.restype = ctypes.c_int
            self.threadlib.fftw_plan_with_nthreads.argtypes = [ctypes.c_int]
            self.threadlib.fftw_plan_with_nthreads.restype = None
            if not self.threadlib.fftw_init_threads():
                raise RuntimeError("thread initialization failed")
            self._thread_runtimes[self.lib._handle] = self.threadlib

    def transform(self, values):
        import numpy as np
        source = np.array(values, dtype=np.float64, copy=True)
        output = np.empty(source.size // 2 + 1, dtype=np.complex128)
        runtime = self.threadlib or self._thread_runtimes.get(self.lib._handle)
        if runtime:
            runtime.fftw_plan_with_nthreads(self.threads)
        plan = self.lib.fftw_plan_dft_r2c_1d(source.size, source.ctypes.data, output.ctypes.data, 64)
        if not plan:
            raise RuntimeError("FFTW planning failed")
        try:
            self.lib.fftw_execute(plan)
        finally:
            self.lib.fftw_destroy_plan(plan)
        assert np.array_equal(source, values)
        return output


@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("n", [64, 4096, 4099])
def test_actual_fftw_transform_with_selected_runtime(loader, threads, n):
    import numpy as np
    if not ctypes.util.find_library("fftw3"):
        pytest.skip("Real FFTW integration requires the optional installed runtime")
    backend = loader["_load_fftw"](SimpleNamespace(FFTWLibrary=CFFTW), threads)
    values = np.random.default_rng(n).standard_normal(n)
    actual = backend.transform(values)
    np.testing.assert_allclose(actual, np.fft.rfft(values), rtol=3e-12, atol=3e-11)
    assert backend.threading_diagnostics["actual"] == backend.threads
    assert backend.threads in (1, threads)
    if threads == 1:
        assert backend.threading_diagnostics["status"] == "single_thread_requested"
    else:
        # An unavailable thread runtime is a documented fallback, not proof
        # of a multi-thread execution. Qualification scripts check it apart.
        assert backend.threading_diagnostics["status"] in ("enabled", "fallback_single_thread")


@pytest.mark.skipif(not sys.platform.startswith("linux") or not shutil.which("gcc"),
                    reason="Linux ELF dependency recovery requires local GCC")
def test_real_elf_missing_companion_dependency_is_retried(loader, monkeypatch, tmp_path):
    dependency = tmp_path / "liblpsd_discovery_test_dependency.so"
    core = tmp_path / "libfftw3-test.so"
    (tmp_path / "dependency.c").write_text("int lpsd_dependency(void) { return 73; }\n")
    (tmp_path / "core.c").write_text("extern int lpsd_dependency(void); int lpsd_probe(void) { return lpsd_dependency(); }\n")
    subprocess.run(["gcc", "-shared", "-fPIC", str(tmp_path / "dependency.c"),
                    "-Wl,-soname," + dependency.name, "-o", str(dependency)], check=True, timeout=30)
    subprocess.run(["gcc", "-shared", "-fPIC", str(tmp_path / "core.c"), "-L" + str(tmp_path),
                    "-llpsd_discovery_test_dependency", "-o", str(core)], check=True, timeout=30)
    # Run dlopen in a fresh interpreter: prior tests must not supply the SONAME.
    probe = '''
import ctypes, ctypes.util, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location("discovery_test", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns = m.discovery_namespace(Path(sys.argv[2]))
ctypes.util.find_library = lambda name: None
m.metadata.distribution = lambda name: SimpleNamespace(files=[Path(sys.argv[3])], locate_file=lambda p:p)
keep = []
def initialize():
    keep.append(ctypes.CDLL(sys.argv[4], mode=ctypes.RTLD_GLOBAL))
    return True
ns["_initialize_fftw_wheel"] = initialize
class Adapter:
    def __init__(self, core, thread, count):
        self.lib = ctypes.CDLL(core); self.threads=count
        assert self.lib.lpsd_probe() == 73
result = ns["_load_fftw"](SimpleNamespace(FFTWLibrary=Adapter), 1)
assert result.threading_diagnostics["failures"] == {"main_load_failed":1}
assert len(keep)==1
print(json.dumps(result.threading_diagnostics))
'''
    completed = subprocess.run([sys.executable, "-c", probe, __file__,
                                str(ROOT / "lpsd_benchmark/worker.py"), str(core), str(dependency)],
                               capture_output=True, text=True, timeout=30, check=True)
    details = json.loads(completed.stdout)
    assert details["main_source"] == "wheel" and details["actual"] == 1
