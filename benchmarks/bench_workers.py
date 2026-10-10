# SPDX-License-Identifier: GPL-3.0-or-later
"""Serialize balanced worker sweeps; retain every complete API call."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from lpsd_fast import lpsd, lcsd
from benchmarks.bench_lpsd import build_evidence, fingerprint, profile_summary
from benchmarks.check_accuracy import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000000, 10000000, 30000000])
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 6, 8, 12, 16, 24, 32],help='Worker counts; 0 requests the API automatic choice')
    parser.add_argument('--max-working-mb',type=float,default=4096,help='Concurrency budget; 0 requests the API default heuristic')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--kernel', default='fast', choices=['auto', 'fast', 'scalar'])
    parser.add_argument('--kind', default='psd', choices=['psd', 'csd'])
    parser.add_argument('--outputs', nargs='+', default=['psd'])
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--cpus',type=int,nargs='+',help='Explicit process affinity; record the chosen CPU IDs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cpus: psutil.Process().cpu_affinity(args.cpus)
    if min(args.sizes) < 35 or min(args.workers) < 0 or args.repeats < 1 or args.max_working_mb<0:
        parser.error('Require samples >=35, workers/budget >=0, repeats >=1')
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  environment=dict(platform=platform.platform(), python=platform.python_version(),
                                   numpy=np.__version__, pandas=pd.__version__,
                                   logical_cpus=psutil.cpu_count(), physical_cores=psutil.cpu_count(logical=False),
                                   affinity=psutil.Process().cpu_affinity(),
                                   total_ram_bytes=psutil.virtual_memory().total),
                  build_reports=build_evidence(), method='Full API; full-problem warmup per worker; fixed shuffled order per round; all observations retained; data generation, comparison and JSON outside timer',
                  kind=args.kind, configurations=[], completed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    for n in args.sizes:
        rng = np.random.default_rng(20261008)
        x = rng.normal(size=n)
        data = pd.Series(x, copy=False) if args.kind == 'psd' else pd.DataFrame({'x': x, 'y': .7*x + .6*rng.normal(size=n)})
        call = lpsd if args.kind == 'psd' else lcsd
        kwargs = dict(sample_rate=1., n_frequencies=1000, n_averages=100,
                      kernel=args.kernel, outputs='all' if args.outputs == ['all'] else args.outputs,
                      max_working_mb=args.max_working_mb or None)
        config = dict(n=n, input_sha256=digest(data.to_numpy()), raw_input_sha256=hashlib.sha256(data.to_numpy().tobytes()).hexdigest(), parameters=kwargs,
                      calls=[], warmups=[], profiles={})
        report['configurations'].append(config)
        reference = None
        for workers in args.workers:
            t = time.perf_counter()
            result = call(data, workers=workers or None, **kwargs)
            config['warmups'].append(dict(workers=workers, wall_s=time.perf_counter()-t))
            if reference is None:
                reference = result
        for repeat in range(args.repeats):
            order = list(args.workers)
            random.Random(5729+repeat).shuffle(order)
            for workers in order:
                cpu = time.process_time()
                t = time.perf_counter()
                result = call(data, workers=workers or None, **kwargs)
                elapsed = time.perf_counter()-t
                cpu = time.process_time()-cpu
                pd.testing.assert_index_equal(result.index, reference.index, exact=True)
                assert result.dtypes.equals(reference.dtypes)
                a, b = reference.to_numpy().astype(np.complex128), result.to_numpy().astype(np.complex128)
                nonzero = a != 0
                rel = float(np.max(np.abs((a-b)[nonzero])/np.abs(a[nonzero]), initial=0))
                item = dict(workers=workers, repeat=repeat, wall_s=elapsed, process_cpu_s=cpu,
                            max_relative_error_nonzero_reference=rel,
                            output_sha256=fingerprint(result), output_bitwise_equal=fingerprint(result)==fingerprint(reference),
                            actual_frequencies=len(result), metadata=result.attrs['lpsd_fast'],
                            process_memory=psutil.Process().memory_info()._asdict())
                config['calls'].append(item)
                print(n, workers, repeat, round(elapsed, 6), flush=True)
                save()
        config['median_wall_s'] = {str(w): statistics.median(row['wall_s'] for row in config['calls'] if row['workers']==w) for w in args.workers}
        config['best_observed_workers'] = min(args.workers, key=lambda w: config['median_wall_s'][str(w)])
        if args.profile:
            for w in sorted(set([1, 8, config['best_observed_workers']])):
                t = time.perf_counter()
                result = call(data, workers=w, profile=True, **kwargs)
                config['profiles'][str(w)] = dict(wall_s=time.perf_counter()-t,
                                                  summary=profile_summary(result.attrs['lpsd_profile']),
                                                  detail=result.attrs['lpsd_profile'])
        print('medians', n, config['median_wall_s'], flush=True)
        save()
    report['completed'] = True
    save()


if __name__ == '__main__':
    main()
