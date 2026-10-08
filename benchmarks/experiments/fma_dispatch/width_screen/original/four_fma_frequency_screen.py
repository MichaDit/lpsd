"""Fixed actual-plan lengths, one 10M input, balanced native comparisons."""
from pathlib import Path
import ctypes as ct
import hashlib
import importlib.util
import json
import statistics
import sys
import time

import numpy as np

ROOT = Path('/workspace/scratch/5346007753d8/lpsd')
HERE = Path(__file__).resolve().parent / 'four-fma'
sys.path.insert(0, str(ROOT))
from lpsd_fast import api

api._native()
declarations = dict(vars(api._LIB))
libraries = {}
for name in ('strict', 'control8', 'split256_large', 'split256_all', 'split512_large', 'selective8'):
    library = ct.CDLL(str(HERE / name / 'lpsd_fast/_native/liblpsd_fast.so'))
    for symbol, previous in declarations.items():
        if isinstance(previous, ct._CFuncPtr):
            function = getattr(library, symbol)
            function.argtypes, function.restype = previous.argtypes, previous.restype
    libraries[name] = library

n = 10_000_000
x = np.random.default_rng(20261008).standard_normal(n)
peak = float(np.max(np.abs(x)))
beta = api._kaiser_alpha(200.) * np.pi
overlap = api._kaiser_rov(api._kaiser_alpha(200.))
plan = api._frequency_plan(n, 1., overlap, 1, 0, 1000, 100)
targets = (160, 224, 320, 512, 768, 1280, 1792, 3072, 6144,
           12288, 24576, 49152, 98304, 196608, 393216, 786432,
           1572864, 3145728)
indices = [int(np.argmin(abs(np.asarray(plan[3]) - target))) for target in targets]
assert len(set(indices)) == len(indices)
rows = []
report = {
    'n': n, 'input_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
    'targets': targets, 'pairs': 3, 'overlap': overlap,
    'methodology': 'Fixed actual-plan lengths nearest predeclared targets; one native warmup per variant and three balanced forward/reverse orders; preparation included and separately reported; every observation retained.',
    'builds': {name: json.loads((HERE / name / 'lpsd_fast/_native/liblpsd_fast.build.json').read_text())
               for name in libraries}, 'rows': rows,
}
pointer = api._pointer(x)
out = [ct.c_double() for _ in range(4)]
count, batches = ct.c_long(), ct.c_long()
preparation, segments = ct.c_double(), ct.c_double()

for target, index in zip(targets, indices):
    length = int(plan[3][index])
    frequency_bin = float(plan[2][index])
    w = np.kaiser(length + 1, beta)[:-1]
    cr, ci = api._coefficients(w, frequency_bin, length, blocked=True)
    arguments = (*(ct.byref(item) for item in out), ct.byref(count),
                 pointer, None, n, length, api._pointer(cr), api._pointer(ci),
                 overlap * 100., 0, False, 2, False, False)

    def call(name):
        lib = libraries[name]
        start = time.perf_counter()
        if name == 'strict':
            status = lib.fast_dft_selected_profile(
                *arguments, ct.byref(preparation), ct.byref(segments))
            batches.value = 0
        else:
            status = lib.fast_dft_selected_bounded(
                *arguments, peak, ct.byref(preparation), ct.byref(segments), ct.byref(batches))
        wall_s = time.perf_counter() - start
        assert status == 0
        return dict(method=name, target=target, length=length, frequency_bin=frequency_bin,
                    K=count.value, batches=batches.value, power=out[0].value,
                    wall_s=wall_s, preparation_s=preparation.value, segments_s=segments.value)

    reference = None
    for name in libraries:
        current = call(name)
        if name == 'strict':
            reference = current['power']
    for pair in range(3):
        order = list(libraries) if pair % 2 == 0 else list(reversed(libraries))
        for name in order:
            row = call(name)
            row['pair'] = pair
            row['relative_error'] = abs(row['power'] - reference) / abs(reference)
            assert row['relative_error'] < 1e-8
            rows.append(row)
    medians = {name: statistics.median(row['segments_s'] for row in rows
                                      if row['length'] == length and row['method'] == name)
               for name in libraries}
    print(length, {name: round(medians['strict'] / value, 3) for name, value in medians.items()}, flush=True)
    (HERE / 'frequency-screen.json').write_text(json.dumps(report, indent=2) + '\n')
