"""Shared-input, memory-budgeted parallel LPSD with compatible statistics.

Derived from lpsd 1.0.6, with unchanged planning and normalization formulas.
No reduction of the number of frequencies, segments, or input samples.
GPL-3.0-or-later. See LICENSE and docs/provenance.md.
"""
from collections import OrderedDict
from collections.abc import Mapping, Set
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import math
import os
from pathlib import Path
import threading
import time
import warnings

import numpy as np
import pandas as pd

from . import __version__
from .planning import _asdrms, _kaiser_alpha, _kaiser_rov, _ltf_plan

_LIB = None
_LIB_LOCK = threading.Lock()
_CHUNK = 262144
_KERNELS = {'scalar': 0, 'simd': 1, 'projected': 2, 'auto': 2}
_DEFAULT_OUTPUTS = ('ps', 'psd', 'ps_std', 'psd_std', 'enbw', 'asd', 'asdrms')
_BASE_OUTPUTS = ('ps', 'psd', 'ps_std', 'psd_std', 'enbw')
_WINDOW_FUNCTIONS = (np.kaiser, np.hanning, np.hamming, np.blackman, np.bartlett, np.ones)
_WINDOW_NAMES = {
    'kaiser': np.kaiser, 'hann': np.hanning, 'hanning': np.hanning,
    'hamming': np.hamming, 'blackman': np.blackman,
    'bartlett': np.bartlett, 'boxcar': np.ones,
}


def _requested_outputs(outputs, csd):
    if outputs is None or (isinstance(outputs, str) and outputs == 'all'):
        result = _DEFAULT_OUTPUTS
    elif isinstance(outputs, str):
        result = (outputs,)
    else:
        if isinstance(outputs, (Mapping, Set)):
            raise TypeError('outputs must be an ordered sequence of column names.')
        try:
            result = tuple(outputs)
        except TypeError as exc:
            raise TypeError('outputs must be a column name or a sequence of column names.') from exc
    if not result or any(not isinstance(name, str) for name in result):
        raise ValueError('outputs must contain at least one valid column name.')
    if len(set(result)) != len(result):
        raise ValueError('outputs must not contain duplicate column names.')
    allowed = _DEFAULT_OUTPUTS + ('nsd',)
    if any(name not in allowed for name in result):
        raise ValueError(f'outputs must be selected from {allowed}, or be "all".')
    if csd and 'nsd' in result:
        raise ValueError('NSD requires an auto spectrum (csd=False); use asd for the legacy CSD square root.')
    return result


def _resolve_window(window_function):
    if isinstance(window_function, str):
        try:
            return _WINDOW_NAMES[window_function.lower()]
        except KeyError as exc:
            raise ValueError(f'Unknown window; choose from {tuple(_WINDOW_NAMES)} or supply a callable.') from exc
    if not callable(window_function):
        raise TypeError('window_function must be a recognized name or a callable.')
    return window_function


def available_workers():
    """Respect process affinity and a Linux cgroup v2 CPU quota."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        count = os.cpu_count() or 1
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            count = min(count, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count)


def _available_memory():
    candidates = []
    try:
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        if limit != 'max':
            candidates.append(int(limit) - current)
    except (OSError, ValueError):
        pass
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                candidates.append(int(line.split()[1]) * 1024)
    except (OSError, ValueError):
        pass
    return min(candidates) if candidates else 2 * 1024**3


def _native():
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            suffix = '.dll' if os.name == 'nt' else '.so'
            filename = Path(__file__).parent / '_native' / ('liblpsd_fast' + suffix)
            if not filename.is_file():
                raise RuntimeError('Native library is missing. Run: python -m lpsd_fast.build --native')
            _LIB = ct.CDLL(str(filename))
            fn = _LIB.fast_dft
            dp = ct.POINTER(ct.c_double)
            fn.argtypes = [dp, dp, dp, dp, ct.POINTER(ct.c_long),
                           dp, dp, ct.c_long, ct.c_long, dp, dp,
                           ct.c_double, ct.c_int, ct.c_bool, ct.c_int]
            fn.restype = ct.c_int
            _LIB.fast_dft_profile.argtypes = fn.argtypes + [dp, dp]
            _LIB.fast_dft_profile.restype = ct.c_int
            selected = _LIB.fast_dft_selected
            selected.argtypes = fn.argtypes + [ct.c_bool, ct.c_bool]
            selected.restype = ct.c_int
            _LIB.fast_dft_selected_profile.argtypes = selected.argtypes + [dp, dp]
            _LIB.fast_dft_selected_profile.restype = ct.c_int
            _LIB.window_sums.argtypes = [dp, ct.c_long, dp, dp]
            _LIB.window_sums.restype = ct.c_int
            _LIB.generate_kaiser.argtypes = [dp, ct.c_long, ct.c_double]
            _LIB.generate_kaiser.restype = ct.c_int
            _LIB.generate_window.argtypes = [dp, ct.c_long, ct.c_int, ct.c_double]
            _LIB.generate_window.restype = ct.c_int
            _LIB.generate_coefficients.argtypes = [dp, dp, dp, ct.c_long, ct.c_double]
            _LIB.generate_coefficients.restype = ct.c_int
        return _LIB.fast_dft


def _pointer(x):
    return x.ctypes.data_as(ct.POINTER(ct.c_double))


class _MemoryGate:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.used = 0
        self.peak = 0
        self.condition = threading.Condition()

    def acquire(self, amount):
        # An unusually large frequency runs alone. The budget is a concurrency
        # target, not a promise about process RSS or the user's input storage.
        amount = min(int(amount), self.capacity)
        with self.condition:
            self.condition.wait_for(lambda: self.used + amount <= self.capacity)
            self.used += amount
            self.peak = max(self.peak, self.used)
        return amount

    def release(self, amount):
        with self.condition:
            self.used -= amount
            self.condition.notify_all()


def _kaiser_window(length, beta):
    """The same asymmetric Kaiser window, with bounded-size temporaries."""
    if length <= _CHUNK:
        return np.kaiser(length + 1, beta)[:-1]
    result = np.empty(length, dtype=np.float64)
    alpha = length / 2.0
    denominator = np.i0(beta)
    for start in range(0, length, _CHUNK):
        end = min(start + _CHUNK, length)
        z = np.arange(start, end, dtype=np.float64)
        z -= alpha
        z /= alpha
        z *= z
        np.subtract(1.0, z, out=z)
        np.sqrt(z, out=z)
        z *= beta
        result[start:end] = np.i0(z) / denominator
    return result


def _coefficients(window, frequency_bin, length, strict=False):
    cr = np.empty(length, dtype=np.float64)
    ci = np.empty(length, dtype=np.float64)
    if not strict:
        _native()
        status = _LIB.generate_coefficients(_pointer(cr), _pointer(ci), _pointer(window), length, frequency_bin)
        if status:
            raise RuntimeError('Native Fourier coefficient generation failed.')
        return cr, ci
    # Retain upstream's complex arithmetic and libm path in scalar mode.
    # sin/cos on a real NumPy vector can use a different vector math library.
    scale = 1j * 2 * np.pi * frequency_bin / length
    for start in range(0, length, _CHUNK):
        end = min(start + _CHUNK, length)
        coefficients = window[start:end] * np.exp(scale * np.arange(start, end))
        cr[start:end] = coefficients.real
        ci[start:end] = coefficients.imag
    return cr, ci


def _frequency_plan(n, fs, overlap, n_min_bins, min_segment_length, n_frequencies, n_averages):
    if n < 5:
        raise ValueError('At least five input samples are required.')
    if n > 2147483647:
        raise ValueError('The inherited C detrending routines support at most 2,147,483,647 samples.')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('sample_rate must be finite and positive.')
    if not 0 <= overlap < 1:
        raise ValueError('overlap must lie in [0, 1).')
    if not all(np.isfinite(v) for v in (n_frequencies, n_averages, n_min_bins, min_segment_length)):
        raise ValueError('Frequency-planning parameters must be finite.')
    if n_frequencies < 1 or n_averages < 1 or n_min_bins < 1:
        raise ValueError('Frequency, averaging and minimum-bin parameters must be positive.')
    if not 0 <= min_segment_length <= n:
        raise ValueError('min_segment_length must lie between zero and the data length.')
    plan = _ltf_plan(n, fs, overlap, n_min_bins, min_segment_length, n_frequencies, n_averages)
    f, r, m, lengths, segments = plan
    if len(f) < 2:
        raise ValueError('The requested settings produce fewer than two spectral points.')
    # Reject upstream's unsafe overlap/short-segment combination before C reads.
    for length, count in zip(lengths, segments):
        if count > n - length + 1:
            raise ValueError('Overlap is too large for the selected segment lengths.')
    return plan


def _infer_fs(data):
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError('sample_rate is required for NumPy input.')
    if isinstance(data.index, pd.DatetimeIndex):
        diffs = data.index.to_series().diff()
        period = diffs.median().total_seconds()
        std = diffs.std().total_seconds()
    else:
        diffs = np.diff(data.index)
        period = np.median(diffs)
        std = diffs.std()
    if std / period > 1e-6:
        warnings.warn('The input time steps are not uniform.', UserWarning)
    return 1.0 / period


def _array(data):
    values = data.to_numpy() if isinstance(data, (pd.Series, pd.DataFrame)) else np.asarray(data)
    if values.ndim != 1 or np.iscomplexobj(values):
        raise ValueError('Each input channel must be a one-dimensional real array.')
    result = np.require(values, dtype=np.float64, requirements=['C', 'A'])
    if not np.isfinite(result).all():
        raise ValueError('Input samples must be finite; clean missing values before calling LPSD.')
    return result


def _run_channel(x1, x2, fs, plan, window_function, psll, overlap, order,
                 workers, kernel, max_working_mb, window_cache_mb, profile, outputs, csd):
    start_all = time.perf_counter()
    _native()
    fn = _LIB.fast_dft_selected_profile if profile else _LIB.fast_dft_selected
    f, r, m, lengths, counts = plan
    nf = len(f)
    n = len(x1)
    mode = (2 if order == 0 else 1) if kernel == 'auto' else _KERNELS[kernel]
    statistics = 'ps_std' in outputs or 'psd_std' in outputs
    spectrum = any(name != 'enbw' for name in outputs)
    # PSD also controls upstream's real/complex output convention. Keep its
    # complex64 rounding even when only PS or a deviation is requested.
    base_names = tuple(name for name in _BASE_OUTPUTS
                       if name in outputs or (name == 'psd' and spectrum))
    window_kind = next((kind for kind, fn_window in enumerate(_WINDOW_FUNCTIONS)
                        if window_function is fn_window), None)
    beta = _kaiser_alpha(psll) * np.pi
    capacity = int(max_working_mb * 1024**2) if max_working_mb is not None else int(_available_memory() * 0.65)
    capacity = max(64 * 1024**2, capacity)
    gate = _MemoryGate(capacity)
    cache = OrderedDict()
    cache_size = 0
    cache_limit = int(window_cache_mb * 1024**2)
    cache_lock = threading.Lock()
    custom_window_lock = threading.Lock()
    rows = [None] * nf
    x1p, x2p = _pointer(x1), _pointer(x2)

    def window_for(length):
        nonlocal cache_size
        with cache_lock:
            if length in cache:
                value = cache.pop(length)
                cache[length] = value
                return value, True, 0.0, 0.0
        generation_started = time.perf_counter() if profile else 0.0
        native_window = kernel != 'scalar' and window_kind is not None
        if native_window:
            w = np.empty(length, dtype=np.float64)
            if _LIB.generate_window(_pointer(w), length, window_kind, beta):
                raise ValueError('Native window generation failed.')
        elif window_function is np.kaiser:
            w = _kaiser_window(length, beta)
        else:
            # A callback may reuse its own work buffer. Keep invocation and
            # copying together, even when frequencies execute concurrently.
            with custom_window_lock:
                w = np.array(window_function(length), dtype=np.float64, order='C', copy=True)
        if w.shape != (length,) or (not native_window and not np.isfinite(w).all()):
            raise ValueError('window_function must return a finite vector of the requested length.')
        generation_s = time.perf_counter() - generation_started if profile else 0.0
        # Original sequential sums in C remove Python iteration overhead
        # while retaining the upstream normalization's arithmetic order.
        s1_native, s2_native = ct.c_double(), ct.c_double()
        sums_started = time.perf_counter() if profile else 0.0
        status = _LIB.window_sums(_pointer(w), length, ct.byref(s1_native), ct.byref(s2_native))
        if status:
            raise RuntimeError('Native window normalization failed.')
        s1, s2 = s1_native.value, s2_native.value
        if not math.isfinite(s1) or not math.isfinite(s2):
            raise ValueError('The selected window has nonfinite normalization.')
        if s1 == 0 or s2 == 0:
            raise ValueError('The selected window has zero normalization.')
        sums_s = time.perf_counter() - sums_started if profile else 0.0
        value = (w, s1, s2)
        if w.nbytes <= cache_limit:
            with cache_lock:
                if length not in cache:
                    while cache and cache_size + w.nbytes > cache_limit:
                        _, old = cache.popitem(last=False)
                        cache_size -= old[0].nbytes
                    cache[length] = value
                    cache_size += w.nbytes
        return value, False, generation_s, sums_s

    def one(j):
        length = int(lengths[j])
        temporary_bytes = min(length, _CHUNK) * 96 if kernel == 'scalar' else 8192
        # Mode 2 projects our private coefficients in place; there are no
        # extra projected arrays. Detrending in modes 0/1 needs residuals.
        bytes_per_sample = 24 if mode == 2 or order < 0 else (40 if csd else 32)
        reserved = gate.acquire(length * (bytes_per_sample if spectrum else 8) + temporary_bytes)
        t0 = time.perf_counter()
        try:
            (w, s1, s2), cache_hit, generation_s, sums_s = window_for(length)
            t1 = time.perf_counter()
            pr, pi, vr, vi = ct.c_double(), ct.c_double(), ct.c_double(), ct.c_double()
            navg = ct.c_long()
            prep_s, segments_s = ct.c_double(), ct.c_double()
            if spectrum:
                cr, ci = _coefficients(w, m[j], length, strict=(kernel == 'scalar'))
                t2 = time.perf_counter()
                extra = (ct.byref(prep_s), ct.byref(segments_s)) if profile else ()
                status = fn(ct.byref(pr), ct.byref(pi), ct.byref(vr), ct.byref(vi), ct.byref(navg),
                            x1p, x2p, n, length, _pointer(cr), _pointer(ci), overlap * 100,
                            order, csd, mode, statistics, True, *extra)
                t3 = time.perf_counter()
                if status:
                    errors = {1: 'invalid native arguments', 2: 'unsafe segment bounds',
                              3: 'native allocation failed', 4: 'kernel/order combination unsupported'}
                    raise RuntimeError(errors.get(status, f'native error {status}'))
            else:
                t2 = t3 = t1
            a2 = 2.0 * complex(pr.value, pi.value)
            s12 = s1 * s1
            b2 = 4.0 * complex(vr.value, vi.value) / navg.value if statistics else None
            values = []
            for name in base_names:
                if name == 'ps':
                    values.append(a2 / s12)
                elif name == 'psd':
                    values.append(a2 / fs / s2)
                elif name == 'ps_std':
                    values.append(np.sqrt(b2 / s12**2))
                elif name == 'psd_std':
                    values.append(np.sqrt(b2 / fs**2 / s2**2))
                else:
                    values.append(fs * s2 / s12)
            if profile:
                rows[j] = {'j': j, 'frequency': f[j], 'L': length, 'K': navg.value,
                           'sample_iterations': length * navg.value, 'window_cache_hit': cache_hit,
                           'window_and_sums_s': t1 - t0,
                           'window_generation_s': generation_s, 'window_sums_s': sums_s,
                           'coefficients_s': t2 - t1,
                           'c_kernel_s': t3 - t2, 'c_preparation_s': prep_s.value,
                           'c_segments_s': segments_s.value,
                           'worker_elapsed_s': time.perf_counter() - t0}
            return values
        finally:
            gate.release(reserved)

    if workers == 1:
        computed = list(map(one, range(nf)))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='lpsd') as pool:
            computed = list(pool.map(one, range(nf)))
    assembly_started = time.perf_counter()
    # Match upstream's output precision and the point at which ASD is rounded.
    output = np.asarray(computed, dtype=np.complex64)
    columns = {name: output[:, i] for i, name in enumerate(base_names)}
    if any(name in outputs for name in ('asd', 'asdrms', 'nsd')):
        asd = np.sqrt(columns['psd'])
        if 'asd' in outputs:
            columns['asd'] = asd
        if 'nsd' in outputs:
            columns['nsd'] = asd
        if 'asdrms' in outputs:
            columns['asdrms'], _ = _asdrms(asd, f)
    if spectrum and not np.iscomplex(columns['psd']).any():
        columns = {name: values if name == 'enbw' else values.real
                   for name, values in columns.items()}
    result = pd.DataFrame({name: columns[name] for name in outputs}, index=f)
    result.index.name = 'frequency'
    result.attrs['lpsd_fast'] = {'version': __version__, 'kernel': kernel, 'workers': workers,
                                'native_mode': mode, 'estimator': 'lpsd-1.0.6',
                                'legacy_statistics': True, 'outputs': list(outputs),
                                'variance_computed': statistics,
                                'native_window': kernel != 'scalar' and window_kind is not None}
    if profile:
        result.attrs['lpsd_profile'] = {'channel_wall_s': time.perf_counter() - start_all,
                                       'frequencies': rows, 'workers': workers, 'kernel': kernel,
                                       'concurrency_budget_bytes': capacity,
                                       'peak_reserved_bytes': gate.peak,
                                       'output_assembly_s': time.perf_counter() - assembly_started,
                                       'note': 'Per-worker elapsed times overlap and must not be summed as wall clock.'}
    return result


def lpsd(data, sample_rate=None, window_function=np.kaiser, overlap=None,
         detrending_order=0, n_frequencies=1000, n_averages=100, n_min_bins=1,
         min_segment_length=0, psll=200, use_c_core=True, csd=False, *,
         workers=None, kernel='auto', max_working_mb=None, window_cache_mb=128,
         profile=False, outputs=None):
    """Compute the original LPSD estimator faster, with parallel frequencies.

    Pandas input and upstream spectral parameters are supported. A real NumPy
    vector (including a memory map) is also accepted with an explicit sample
    rate. By default, result column names/dtypes retain the upstream API.
    Inputs are never changed. Nonfinite inputs and unsafe overlap
    configurations are rejected.

    ``outputs='psd'``, ``outputs='nsd'`` or a sequence of names computes only
    the requested columns and their dependencies. ``None`` and ``'all'``
    return the original seven columns. NSD is exactly the existing ASD
    (square root of the rounded PSD), with a more explicit name, and is
    available for auto spectra only. Omitting both deviation columns skips
    native variance updates; it does not reduce samples, segments or bins.

    Window names ``kaiser``, ``hann``/``hanning``, ``hamming``, ``blackman``,
    ``bartlett`` and ``boxcar`` are accepted alongside callables. Kaiser
    retains upstream's periodic convention; the others match the symmetric
    NumPy windows (boxcar is all ones). Non-Kaiser windows need an explicit
    overlap. Known NumPy functions also use the native fast window path;
    arbitrary callbacks retain their exact invocation/copying behavior.

    ``scalar`` keeps upstream accumulation order, coefficients and windows.
    ``simd`` reorders dot-product sums after unchanged long-double detrending.
    ``projected`` moves order-0/1 detrending into the coefficients and uses
    per-segment centering for small signals with a large DC offset. ``auto``
    selects projected for order 0 and SIMD for other supported orders.
    Explicit projected/order-1 can lose accuracy for large ramps plus tiny
    residual noise; auto retains the original long-double linear detrending.
    Fast kernels also evaluate the same Kaiser formula in native code.
    They preserve the mathematical estimator, not bitwise rounding results.
    """
    started = time.perf_counter()
    outputs = _requested_outputs(outputs, csd)
    window_function = _resolve_window(window_function)
    if not use_c_core:
        raise ValueError('lpsd_fast requires its C core; use the original package for the Python backend.')
    if kernel not in _KERNELS:
        raise ValueError(f'kernel must be one of {tuple(_KERNELS)}')
    if detrending_order is not None and not isinstance(detrending_order, (int, np.integer)):
        raise TypeError('detrending_order must be an integer or None.')
    order = -1 if detrending_order is None else int(detrending_order)
    if order not in range(-1, 11):
        raise ValueError('Supported detrending orders are None and integers 0 through 10.')
    if kernel == 'projected' and order not in (0, 1):
        raise ValueError('The projected kernel supports detrending orders 0 and 1.')
    if not np.isfinite(psll):
        raise ValueError('psll must be finite.')
    if max_working_mb is not None and (not np.isfinite(max_working_mb) or max_working_mb <= 0):
        raise ValueError('max_working_mb must be finite and positive.')
    if not np.isfinite(window_cache_mb) or window_cache_mb < 0:
        raise ValueError('window_cache_mb must be finite and nonnegative.')
    if sample_rate is None:
        sample_rate = _infer_fs(data)
    if overlap is None:
        if window_function is not np.kaiser:
            raise ValueError('Specify overlap when using a custom window.')
        overlap = _kaiser_rov(_kaiser_alpha(psll))
    if workers is not None and not isinstance(workers, (int, np.integer)):
        raise TypeError('workers must be an integer or None.')
    workers = available_workers() if workers is None else int(workers)
    if workers < 1:
        raise ValueError('workers must be a positive integer.')
    _native()  # Resolve the shared library once before creating worker threads.
    plan_start = time.perf_counter()
    plan = _frequency_plan(len(data), float(sample_rate), float(overlap), n_min_bins,
                           min_segment_length, n_frequencies, n_averages)
    if order >= min(plan[3]):
        raise ValueError('detrending_order must be smaller than every planned segment length.')
    planning_s = time.perf_counter() - plan_start
    worker_args = (float(sample_rate), plan, window_function, psll, float(overlap), order,
                   min(workers, len(plan[0])), kernel, max_working_mb, window_cache_mb, profile, outputs)
    conversion_started = time.perf_counter()
    if csd:
        if not isinstance(data, pd.DataFrame) or data.shape[1] != 2:
            raise ValueError('CSD requires a DataFrame with exactly two channels.')
        x1, x2 = (_array(data.iloc[:, j]) for j in range(2))
        conversion_s = time.perf_counter() - conversion_started
        result = _run_channel(x1, x2, *worker_args, True)
    elif isinstance(data, pd.DataFrame) and data.shape[1] > 1:
        result = {}
        for name in data.columns:
            x = _array(data[name])
            result[name] = _run_channel(x, x, *worker_args, False)
        conversion_s = None
    else:
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        x = _array(data)
        conversion_s = time.perf_counter() - conversion_started
        result = _run_channel(x, x, *worker_args, False)
    if profile:
        results = result.values() if isinstance(result, dict) else [result]
        for item in results:
            item.attrs['lpsd_profile'].update({'api_wall_s': time.perf_counter() - started,
                                              'planning_s': planning_s,
                                              'input_conversion_s': conversion_s})
    return result


def lcsd(data, *args, **kwargs):
    kwargs['csd'] = True
    return lpsd(data, *args, **kwargs)


def lnsd(data, *args, **kwargs):
    """Return only logarithmic noise spectral density, sqrt(PSD), as ``nsd``.

    Accepts the same inputs and spectral parameters as :func:`lpsd`.
    The result remains a frequency-indexed DataFrame (or a channel dict).
    """
    if 'outputs' in kwargs:
        raise TypeError('lnsd fixes outputs="nsd"; use lpsd to select multiple columns.')
    return lpsd(data, *args, outputs='nsd', **kwargs)
