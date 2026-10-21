"""Common native calls avoid loading unused reference/SciPy backends."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def isolated(code, blocked):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    guard = f'''
import sys
class RejectUnusedImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.')
               for name in {blocked!r}):
            raise AssertionError('Unused dependency was imported: ' + fullname)
sys.meta_path.insert(0, RejectUnusedImports())
'''
    result = subprocess.run([sys.executable, '-c', guard + code], cwd=ROOT,
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_metadata_and_builder_do_not_import_numerical_dependencies():
    isolated('''
import lpsd_fast
from lpsd_fast import build
assert lpsd_fast.__version__.startswith('1.0.6+fast.')
assert callable(build.build)
''', ('numpy', 'pandas', 'scipy', 'lpsd'))


@pytest.mark.parametrize('window', ('kaiser', 'hann', 'boxcar'))
@pytest.mark.parametrize('statistics', ('corrected', 'legacy'))
def test_common_public_calls_need_no_reference_import(window, statistics):
    isolated(f'''
import inspect
import numpy as np
from lpsd_fast import lpsd, lcsd, lnsd, available_workers
from lpsd_fast import api
assert lpsd is api.lpsd and lnsd is api.lnsd and lcsd is api.lcsd
assert 'sample_rate' in inspect.signature(lpsd).parameters
values = np.random.default_rng(613).normal(size=4097)
result = lpsd(values, sample_rate=50., window_function={window!r},
              statistics={statistics!r}, workers=2, n_frequencies=32,
              n_averages=8, overlap=.5, outputs=('psd', 'nsd'))
assert len(result) > 2 and np.isfinite(result.to_numpy()).all()
assert available_workers() >= 1
''', ('scipy', 'lpsd'))


def test_white_comparison_imports_need_no_scipy_or_reference():
    isolated('''
from benchmarks import bench_fftw, matched_smoothing
from benchmarks.fftw_comparison_signals import signal, theory
from lpsd_fast.prepared import PreparedLPSDSubset
values, tones = signal('white', 4097, 50., 613)
assert values.shape == (4097,) and not tones
''', ('scipy', 'lpsd'))


@pytest.mark.parametrize('algorithm', ('full', 'packed-real'))
def test_first_disk_call_needs_no_numpy_testing(algorithm):
    from benchmarks.bench_fftw import FFTWLibrary

    try:
        FFTWLibrary(os.environ.get('LPSD_TEST_FFTW_LIBRARY'))
    except OSError as exc:
        pytest.skip(f'Optional FFTW shared library unavailable: {exc}')
    # pytest may already have imported numpy.testing in this process. Only a
    # fresh interpreter exposes its avoidable first-call initialization cost.
    isolated(f'''
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
from benchmarks.bench_fftw import FFTWLibrary
from benchmarks.bench_fftw_disk import disk_pipeline
from lpsd_fast import api

api._native()
backend = FFTWLibrary(os.environ.get('LPSD_TEST_FFTW_LIBRARY'))
samples = np.random.default_rng(613).normal(size=1000)
args = SimpleNamespace(sample_rate=50., workers=1, smoothing_workers=1,
                       low_working_mb=8., fft_memory_mb=.1,
                       power_storage='workspace', disk_fft_algorithm={algorithm!r})
with tempfile.TemporaryDirectory() as directory:
    result, details = disk_pipeline('fftw', samples, backend, Path(directory), args)
    assert len(result) > 2 and np.isfinite(result.to_numpy()).all()
    assert (result.psd.to_numpy() >= 0).all()
    assert np.array_equal(np.sqrt(result.psd.to_numpy().astype(np.complex64)).real,
                          result.nsd.to_numpy())
    assert details['parseval']['relative_error'] <= 1e-10
    assert not list(Path(directory).iterdir())
assert 'numpy.testing' not in sys.modules
''', ('numpy.testing',))


def test_lightweight_planning_function_bodies_match_pinned_upstream():
    # Source equality is stronger than a handful of rounded numerical examples.
    # The separate reference-manifest test pins the original upstream bytes.
    names = ('_kaiser_alpha', '_myround', '_kaiser_rov', '_ltf_plan', '_asdrms')
    trees = []
    for path in ('lpsd/_helpers.py', 'lpsd_fast/planning.py'):
        parsed = ast.parse((ROOT / path).read_text(encoding='utf-8'))
        trees.append({node.name: ast.dump(node, include_attributes=False)
                      for node in parsed.body if isinstance(node, ast.FunctionDef)})
    for name in names:
        assert trees[0][name] == trees[1][name], name
