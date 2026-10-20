"""Thread controls must address the FFTW core that will create each plan.

Two tiny independent C runtimes record the thread setting seen by their
planner. They are ABI recorders, not numerical FFT implementations, and run
in a child interpreter so their symbols never enter the real-FFTW tests.
Separate numerical checks below exercise the actual production adapter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import bench_fftw as adapter
from lpsd_benchmark.worker import _FFTWUnavailable, _load_fftw
from lpsd_fast.build import build_shared


_RECORDER = r"""
#include <stddef.h>
#include <stdlib.h>
const char fftw_version[] = "fftw-3.3.10-thread-recorder";
static int configured_threads = 1, observed_threads = 0;
int recorder_configured(void) { return configured_threads; }
int recorder_observed(void) { return observed_threads; }
int fftw_init_threads(void) { return 1; }
void fftw_plan_with_nthreads(int count) { configured_threads = count; }
void *fftw_malloc(size_t bytes) { return malloc(bytes); }
void fftw_free(void *pointer) { free(pointer); }
static void *record_plan(void) {
    observed_threads = configured_threads;
    return malloc(1);
}
void *fftw_plan_dft_r2c_1d(int n, double *in, void *out, unsigned flags) {
    return record_plan();
}
void *fftw_plan_dft_1d(int n, void *in, void *out, int sign, unsigned flags) {
    return record_plan();
}
void *fftw_plan_many_dft(int rank, const int *n, int howmany,
                       void *in, const int *inembed, int istride, int idist,
                       void *out, const int *onembed, int ostride, int odist,
                       int sign, unsigned flags) { return record_plan(); }
void fftw_execute(void *plan) { }
void fftw_execute_dft(void *plan, void *in, void *out) { }
void fftw_destroy_plan(void *plan) { free(plan); }
void fftw_forget_wisdom(void) { }
void fftw_set_timelimit(double seconds) { }
void fftw_flops(void *plan, double *add, double *mul, double *fma) {
    *add = *mul = *fma = 0.0;
}
char *fftw_sprint_plan(void *plan) { return NULL; }
"""


@pytest.fixture(scope="module")
def recording_libraries(tmp_path_factory):
    compiler = shlex.split(os.environ.get("CC", "gcc" if os.name == "nt" else "cc"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("The two-runtime ABI regression requires a local C compiler")
    directory = tmp_path_factory.mktemp("fftw-runtime-recorders")
    source = directory / "recorder.c"
    source.write_text(_RECORDER, encoding="utf-8")
    suffix = ".dll" if os.name == "nt" else ".so"
    libraries = [directory / ("recorder-" + name + suffix) for name in ("a", "b")]
    for library in libraries:
        build_shared(source, library)
    return libraries


@pytest.mark.parametrize("kind", ["real", "bluestein", "disk"])
def test_single_thread_plan_resets_its_own_core_only(recording_libraries, tmp_path, kind):
    probe = r'''
import itertools
import json
from pathlib import Path
import sys
from benchmarks import bench_fftw as adapter
from benchmarks._fftw_bluestein import BluesteinRealFFT
from benchmarks._fftw_disk import DiskFFT

path_a, path_b, kind, directory = sys.argv[1:]
a = adapter.FFTWLibrary(path_a, path_a, threads=2)
b = adapter.FFTWLibrary(path_b, path_b, threads=4)
assert a.lib._handle != b.lib._handle
workspace_ids = itertools.count()

def plan(backend):
    if kind == "real":
        return adapter.RealFFT(backend, 64)
    if kind == "bluestein":
        return BluesteinRealFFT(backend, 64, native_ops="numpy")
    workspace = Path(directory) / ("workspace-" + str(next(workspace_ids)) + ".c128")
    return DiskFFT(backend, 64, workspace, memory_mb=1)

with plan(a):
    pass
with plan(b):
    pass
assert a.lib.recorder_observed() == 2
assert b.lib.recorder_observed() == 4

# Loading another handle to A is a single-thread request. B was initialized
# last; it must not receive A's reset or keep A's earlier setting in force.
single_a = adapter.FFTWLibrary(path_a)
with plan(single_a):
    pass
assert a.lib.recorder_observed() == 1
assert a.lib.recorder_configured() == 1
assert b.lib.recorder_configured() == 4
assert b.lib.recorder_observed() == 4
print(json.dumps({"core_a_threads": 1, "core_b_threads": 4, "plan": kind}))
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, *(str(path) for path in recording_libraries),
         kind, str(tmp_path)], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(core_a_threads=1, core_b_threads=4, plan=kind)


@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("n", [64, 4099])
def test_discovery_executes_production_fftw_adapter(threads, n):
    try:
        backend = _load_fftw(adapter, threads, os.environ.get("LPSD_TEST_FFTW_LIBRARY"),
                             os.environ.get("LPSD_TEST_FFTW_THREADS_LIBRARY"))
    except _FFTWUnavailable:
        pytest.skip("Real FFTW integration requires an optional system or wheel runtime")
    values = np.random.default_rng(n).standard_normal(n)
    with adapter.RealFFT(backend, n) as transform:
        transform.x[:] = values
        transform.execute()
        np.testing.assert_allclose(transform.y, np.fft.rfft(values), rtol=3e-12, atol=3e-11)
        np.testing.assert_array_equal(transform.x, values)
    assert backend.threading_diagnostics["actual"] == backend.threads
    assert backend.threads in (1, threads)
