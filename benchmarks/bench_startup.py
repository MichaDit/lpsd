#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Alternate fresh-process startup measurements of two source checkouts.

Use a small standard-library-only parent so a previously warmed numerical
process cannot contaminate a child's inherited maximum-resident measurement.
File caches are not flushed. No builds, installs or remote execution occur.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


CHILD = r'''
import hashlib, json, sys, time
from pathlib import Path
kind = sys.argv[1]
started = time.perf_counter()
if kind == 'metadata':
    import lpsd_fast
    from lpsd_fast import build
elif kind == 'api-import':
    from lpsd_fast import lpsd
elif kind == 'comparison-import':
    from benchmarks import bench_fftw, matched_smoothing
    from benchmarks.fftw_comparison_signals import signal
    signal('white', 1024, 50., 20610)
else:
    import numpy as np
    from lpsd_fast import lpsd
    values = np.random.default_rng(20610).standard_normal(10000)
    result = lpsd(values, sample_rate=50., n_frequencies=1000, n_averages=100,
                  kernel='fast', workers=4, outputs=('psd', 'nsd'), statistics='corrected')
elapsed = time.perf_counter() - started
row = {'inside_wall_s': elapsed, 'current_rss_bytes': None, 'peak_rss_bytes': None,
       'scipy_loaded': 'scipy' in sys.modules, 'reference_loaded': 'lpsd' in sys.modules}
if sys.platform.startswith('linux'):
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            row['current_rss_bytes'] = int(line.split()[1]) * 1024
try:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin' or sys.platform.startswith('linux'):
        row['peak_rss_bytes'] = int(peak if sys.platform == 'darwin' else peak * 1024)
except ImportError:
    pass
if kind == 'first-call':
    row['output_sha256'] = hashlib.sha256(result.index.to_numpy().tobytes()
                                         + result.to_numpy().tobytes()).hexdigest()
print(json.dumps(row, allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 100:
        parser.error('Use between two and 100 paired repetitions')
    roots = {'baseline': args.baseline_root.resolve(), 'candidate': args.candidate_root.resolve()}
    for root in roots.values():
        if not (root / 'lpsd_fast/api.py').is_file():
            parser.error('Both roots must be complete source checkouts')
    report = {'schema': 'lpsd-startup-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'environment': {'system': platform.system(), 'machine': platform.machine(),
                              'python': platform.python_version()},
              'source_sha256': {name: {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted((root / 'lpsd_fast').glob('*.py'))}
                                for name, root in roots.items()},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'methodology': 'Fresh processes, alternating A/B then B/A, all observations retained; warm filesystem caches; small parent; BLAS/OMP threads one; shared host, clocks uncontrolled',
              'cases': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for kind in ('metadata', 'api-import', 'comparison-import', 'first-call'):
        row = {'kind': kind, 'observations': []}
        expected = None
        for pair in range(args.repeats):
            for name in (('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')):
                env = os.environ.copy()
                env['PYTHONPATH'] = str(roots[name])
                for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                            'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                    env[key] = '1'
                started = time.perf_counter()
                process = subprocess.run([sys.executable, '-c', CHILD, kind], cwd=roots[name],
                                         env=env, text=True, capture_output=True, timeout=120)
                elapsed = time.perf_counter() - started
                if process.returncode:
                    raise RuntimeError(process.stderr)
                found = json.loads(process.stdout)
                if 'output_sha256' in found:
                    if expected is None:
                        expected = found['output_sha256']
                    if found['output_sha256'] != expected:
                        raise AssertionError('First-call output changed')
                row['observations'].append({'pair': pair, 'implementation': name,
                                            'process_wall_s': elapsed, **found})
        row['process_medians_s'] = {name: statistics.median(x['process_wall_s'] for x in row['observations']
                                                           if x['implementation'] == name) for name in roots}
        row['speedup'] = row['process_medians_s']['baseline'] / row['process_medians_s']['candidate']
        report['cases'].append(row)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({key: row[key] for key in ('kind', 'process_medians_s', 'speedup')}), flush=True)


if __name__ == '__main__':
    main()
