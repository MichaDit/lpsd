"""Complete-call confirmation of the fixed 256<=L<1024 split candidate."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import statistics
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path('/workspace/scratch/5346007753d8/lpsd')
HERE = Path(__file__).resolve().parent / 'four-fma'
sys.path.insert(0, str(ROOT))
from benchmarks.bench_rolling_boxcar import evidence, load_package
from benchmarks.bench_segment_core import compare_spectra
from benchmarks.bench_lpsd import fingerprint, json_scalar, profile_summary

names = ('strict', 'control8', 'split256_middle')
modules = {name: load_package(HERE / name, 'lpsd_four_screen_' + name) for name in names}
identities = {name: evidence(module) for name, module in modules.items()}
report = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'implementations': identities,
    'methodology': {
        'pairs': 4,
        'orders': [list(names), list(reversed(names))] * 2,
        'input': 'resident float64 pandas Series, PCG64 seed 20261008',
        'timing': 'complete API wall time; input creation, warmups, comparison, hashes and file output excluded',
        'warmup': 'one full call per implementation and case',
        'profiles': 'separate diagnostic calls for the one-worker cases only',
        'limits': 'shared host; every observation retained; no repeat extension or outlier removal',
        'selection': 'fixed 256<=L<1024 split only, chosen from separately retained predetermined actual-plan frequency probes; remaining lengths retain eight-way FMA',
    },
    'cases': [],
}


def save():
    (HERE / 'api-screen.json').write_text(json.dumps(report, indent=2, default=json_scalar) + '\n')


for n, workers in ((1_000_000, 1), (10_000_000, 1), (10_000_000, 8)):
    values = pd.Series(np.random.default_rng(20261008).standard_normal(n), copy=False)
    parameters = dict(sample_rate=1., n_frequencies=1000, n_averages=100,
                      window_function='kaiser', psll=200., detrending_order=0,
                      kernel='fast', outputs='psd', workers=workers,
                      max_working_mb=4096, window_cache_mb=128)
    case = {'n': n, 'workers': workers, 'parameters': parameters,
            'input_sha256': hashlib.sha256(values.to_numpy().tobytes()).hexdigest(),
            'calls': [], 'profiles': {}}
    report['cases'].append(case)
    reference = None
    for name, module in modules.items():
        result = module.lpsd(values, **parameters)
        if name == 'strict':
            reference = result
            case['actual_frequencies'] = len(result)
        else:
            compare_spectra(reference, result)
    repetitions = {name: [] for name in names}
    for pair in range(4):
        order = names if pair % 2 == 0 else tuple(reversed(names))
        for name in order:
            start = time.perf_counter()
            result = modules[name].lpsd(values, **parameters)
            wall_s = time.perf_counter() - start
            repetitions[name].append(wall_s)
            row = {'name': name, 'pair': pair, 'wall_s': wall_s,
                   'fingerprint': fingerprint(result), 'comparison': compare_spectra(reference, result)}
            case['calls'].append(row)
            print(n, workers, name, pair, round(wall_s, 6), flush=True)
            save()
    case['median_s'] = {name: statistics.median(times) for name, times in repetitions.items()}
    case['strict_over_candidate'] = case['median_s']['strict'] / case['median_s']['split256_middle']
    case['control8_over_candidate'] = case['median_s']['control8'] / case['median_s']['split256_middle']
    case['candidate_faster_pairs'] = {
        baseline: sum(candidate < control for candidate, control in
                      zip(repetitions['split256_middle'], repetitions[baseline]))
        for baseline in ('strict', 'control8')}
    if workers == 1:
        for name, module in modules.items():
            result = module.lpsd(values, profile=True, **parameters)
            case['profiles'][name] = {'summary': profile_summary(result.attrs['lpsd_profile']),
                                      'full_profile': result.attrs['lpsd_profile'],
                                      'comparison': compare_spectra(reference, result)}
    save()
    print('SUMMARY', n, workers, case['median_s'],
          case['strict_over_candidate'], case['control8_over_candidate'], flush=True)

assert identities == {name: evidence(module) for name, module in modules.items()}
report['completed'] = True
save()
