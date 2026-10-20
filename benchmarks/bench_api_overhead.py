#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired complete-API timings from two independently built source checkouts.

Each case warms both APIs on its complete input, then alternates A/B and B/A.
Every timed call prepares its own plan, windows, coefficients and result. Imports,
input construction, hashes and report writes are outside the timer. Run this
serially without other benchmarks. Startup is measured by bench_startup.py.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
SEED = 20610


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def output_fingerprint(frame):
    return sha256(frame.index.to_numpy().tobytes() + frame.to_numpy().tobytes())


def require_identical(actual, expected, reference):
    if (not actual.columns.equals(expected.columns)
            or not actual.dtypes.equals(expected.dtypes)
            or actual.index.dtype != expected.index.dtype
            or actual.index.name != expected.index.name
            or output_fingerprint(actual) != reference):
        raise AssertionError('Frequency indices, output dtypes or output bytes changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, default=ROOT)
    parser.add_argument('--n', type=int, nargs='+', default=[10_000, 100_000, 1_000_000])
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--extra-cases', action='store_true',
                        help='Also measure auto/all and fast/Boxcar at N=100000, four workers')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 100:
        parser.error('Use between two and 100 paired repetitions')
    if any(n < 5 or n > 2_147_483_647 for n in args.n) or min(args.workers) < 1:
        parser.error('Require 5 <= N <= 2147483647 and positive worker counts')
    roots = {'baseline': args.baseline_root.resolve(),
             'candidate': args.candidate_root.resolve()}
    for root in roots.values():
        if not (root / 'lpsd_fast/api.py').is_file():
            parser.error('Both roots must be complete, independently built source checkouts')
    # Apply before importing NumPy or either API. Frequency workers remain
    # the explicitly requested count; unrelated numerical pools use one thread.
    for key in THREAD_KEYS:
        os.environ[key] = '1'
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    from benchmarks.bench_rolling_boxcar import evidence, load_package

    modules = {name: load_package(root, 'lpsd_api_overhead_' + name)
               for name, root in roots.items()}
    for module in modules.values():
        importlib.import_module(module.__name__ + '.api')._native()
    calls = {name: module.lpsd for name, module in modules.items()}
    implementations = {name: evidence(module) for name, module in modules.items()}
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    helper = Path(load_package.__code__.co_filename).resolve()
    report = {
        'schema': 'lpsd-api-overhead-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'state': 'running',
        'environment': {'system': platform.system(), 'machine': platform.machine(),
                        'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'affinity': affinity,
                        'logical_cpus': os.cpu_count(),
                        'thread_environment': {key: os.environ[key] for key in THREAD_KEYS}},
        'script_sha256': sha256(Path(__file__).read_bytes()),
        'load_package_helper_sha256': sha256(helper.read_bytes()),
        'implementations': implementations,
        'methodology': {
            'pairs': args.repeats, 'seed': SEED, 'statistics': 'corrected',
            'order': 'Alternating A/B then B/A; all observations retained; no concurrent benchmarks',
            'timer': 'Complete warmed public API; fresh plan, windows and coefficients on every call',
            'warmup': 'One complete call per implementation and configuration',
            'exclusions': 'Imports, input construction, warmup, hashes, comparisons and report writes',
            'limits': 'One shared host; CPU clocks and filesystem caches uncontrolled; no cross-platform speedup inferred',
        },
        'cases': [],
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(args.output)

    configs = [(n, workers, 'fast', ('psd', 'nsd'), 'kaiser')
               for n in args.n for workers in args.workers]
    if args.extra_cases:
        configs += [(100_000, 4, 'auto', None, 'kaiser'),
                    (100_000, 4, 'fast', ('psd', 'nsd'), 'boxcar')]
    save()
    try:
        for n, workers, kernel, outputs, window in configs:
            values = np.random.default_rng(SEED).standard_normal(n)
            input_hash = sha256(values.tobytes())
            kwargs = dict(sample_rate=50., n_frequencies=1000, n_averages=100,
                          kernel=kernel, workers=workers, outputs=outputs,
                          statistics='corrected', max_working_mb=512,
                          window_cache_mb=128, window_function=window)
            if window != 'kaiser':
                kwargs['overlap'] = .8
            expected = calls['baseline'](values, **kwargs)
            reference = output_fingerprint(expected)
            require_identical(calls['candidate'](values, **kwargs), expected, reference)
            row = {'n': n, 'parameters': kwargs, 'actual_points': len(expected),
                   'input_sha256': input_hash, 'output_sha256': reference,
                   'columns': list(expected.columns),
                   'dtypes': [str(dtype) for dtype in expected.dtypes],
                   'index_dtype': str(expected.index.dtype), 'observations': []}
            report['cases'].append(row)
            for pair in range(args.repeats):
                order = ('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')
                for name in order:
                    started = time.perf_counter()
                    actual = calls[name](values, **kwargs)
                    elapsed = time.perf_counter() - started
                    fingerprint = output_fingerprint(actual)
                    row['observations'].append({'pair': pair, 'implementation': name,
                                                'wall_s': elapsed, 'output_sha256': fingerprint})
                    require_identical(actual, expected, reference)
                save()
            row['input_unchanged'] = sha256(values.tobytes()) == input_hash
            if not row['input_unchanged']:
                raise AssertionError('Input bytes changed during the API comparison')
            timings = {name: [item['wall_s'] for item in row['observations']
                              if item['implementation'] == name] for name in modules}
            row['medians_s'] = {name: statistics.median(times) for name, times in timings.items()}
            row['ranges_s'] = {name: [min(times), max(times)] for name, times in timings.items()}
            row['speedup'] = row['medians_s']['baseline'] / row['medians_s']['candidate']
            ratios = [before / after for before, after in zip(timings['baseline'], timings['candidate'])]
            row['paired_geometric_speedup'] = statistics.geometric_mean(ratios)
            row['paired_speedup_range'] = [min(ratios), max(ratios)]
            row['faster_pairs'] = sum(ratio > 1 for ratio in ratios)
            row['outputs_bit_identical'] = True
            save()
            print(json.dumps({key: row[key] for key in ('n', 'actual_points', 'medians_s', 'speedup')}
                             | {'workers': workers, 'kernel': kernel, 'window': window}), flush=True)
        final_evidence = {name: evidence(module) for name, module in modules.items()}
        if final_evidence != implementations:
            raise AssertionError('Implementation source or native evidence changed during measurement')
        if sha256(Path(__file__).read_bytes()) != report['script_sha256']:
            raise AssertionError('Measurement script changed during measurement')
        if sha256(helper.read_bytes()) != report['load_package_helper_sha256']:
            raise AssertionError('Package-loading helper changed during measurement')
        report['source_and_native_evidence_unchanged'] = True
        report['state'] = 'completed'
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save()
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
