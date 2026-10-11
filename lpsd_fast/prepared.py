"""Reusable Kaiser/order-0 PSD subsets of the existing ``kernel='fast'``.

The original five-array frequency plan is supplied by the caller. Preparation
can retain the projected Fourier coefficients and window normalizations; every
call still evaluates the original input segments, local anchors and legacy
power recurrence. No samples, frequencies or averages are removed.

Only an auto-spectrum PSD is exposed here. The existing public LPSD API and its
statistics are unchanged. GPL-3.0-or-later; see LICENSE and docs/provenance.md.
"""
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
from dataclasses import dataclass
import heapq
import math
import numbers
import threading
import time

import numpy as np

from . import api
from .planning import _kaiser_alpha, _kaiser_rov


_BIND_LOCK = threading.Lock()
_BOUND_LIB = None
_MIB = 1024 ** 2
_INT_MAX = np.iinfo(np.int32).max
_ERRORS = {1: 'invalid native arguments', 2: 'unsafe segment bounds',
           3: 'native allocation failed', 4: 'kernel/order combination unsupported'}


def _native_prepared():
    """Bind the additive ABI without changing the original API bindings."""
    global _BOUND_LIB
    with _BIND_LOCK:
        api._native()
        lib = api._LIB
        if _BOUND_LIB is not lib:
            try:
                version = lib.fast_dft_prepared_version
                prepare = lib.fast_dft_prepare_order0
                compute = lib.fast_dft_prepared_order0
            except AttributeError as exc:
                raise RuntimeError('Prepared LPSD native symbols are missing. Run: '
                                   'python -m lpsd_fast.build --native') from exc
            version.argtypes = []
            version.restype = ct.c_int
            if version() != 1:
                raise RuntimeError('Unsupported prepared LPSD native ABI version.')
            dp = ct.POINTER(ct.c_double)
            lp = ct.POINTER(ct.c_long)
            prepare.argtypes = [dp, dp, ct.c_long, ct.c_int]
            prepare.restype = ct.c_int
            compute.argtypes = [dp, dp, lp, dp, ct.c_long, ct.c_long,
                               dp, dp, ct.c_double, ct.c_int, ct.c_double,
                               dp, dp, lp]
            compute.restype = ct.c_int
            _BOUND_LIB = lib
        return lib


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer.')
    value = int(value)
    if value < minimum or value > _INT_MAX:
        raise ValueError(f'{name} must be between {minimum} and {_INT_MAX}.')
    return value


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number.')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number.') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number.')
    return value


def _copy_plan(plan, n, fs, overlap):
    """Copy, validate and freeze a uniformly selected original frequency plan."""
    try:
        if len(plan) != 5:
            raise ValueError
        arrays = [np.array(part, dtype=np.float64, order='C', copy=True)
                  for part in plan]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('plan must contain the five arrays (f, r, m, lengths, counts).') from exc
    size = arrays[0].size
    if any(a.ndim != 1 or len(a) != size or not np.isfinite(a).all() for a in arrays):
        raise ValueError('The five plan arrays must be finite 1-D arrays of equal length.')
    f, r, m, lengths, counts = arrays
    if (np.any(f <= 0) or np.any(f > fs / 2) or np.any(r <= 0) or np.any(m <= 0)):
        raise ValueError('Plan frequencies must lie in (0, sample_rate/2]; r and m must be positive.')
    if (np.any(lengths != np.floor(lengths)) or np.any(lengths < 1)
            or np.any(lengths > n) or np.any(counts != np.floor(counts))
            or np.any(counts < 1) or np.any(counts > n - lengths + 1)):
        raise ValueError('The plan has invalid segment lengths or counts.')
    # Use exactly the native count arithmetic for the safety check. The
    # planner's algebraically equal expression can round differently at ties;
    # its supplied counts never replace the native calculation.
    ovfact = 1.0 / (1.0 - (overlap * 100.0) / 100.0)
    native_counts = np.floor(((n - lengths) * ovfact) / lengths + 1.0 + 0.5)
    if (not np.isfinite(native_counts).all() or np.any(native_counts >= _INT_MAX)
            or np.any(native_counts > n - lengths + 1)):
        raise ValueError('Overlap is unsafe for the selected segment lengths.')
    arrays[3] = lengths.astype(np.int64)
    arrays[4] = counts.astype(np.int64)
    for a in arrays:
        a.setflags(write=False)
    return tuple(arrays)


@dataclass(frozen=True)
class _ProjectedPoint:
    qr: np.ndarray
    qi: np.ndarray
    qr_pointer: object
    qi_pointer: object
    s1: float
    s2: float


class PreparedLPSDSubset:
    """Prepare reusable order-0 Kaiser PSD points from an original LPSD plan.

    ``plan`` is ``(f, r, m, lengths, counts)`` from the existing planner, with
    the same subset applied to each array. Computations return a new, owning
    float32 PSD vector in that order, including the original rounding point.
    Caller input is never changed or retained after the call.

    ``max_cache_mb`` bounds persistent projected-vector storage. Points are
    admitted in plan order; points that do not fit are prepared during each
    computation. ``max_working_mb`` separately limits concurrent temporary
    reservations, as in the existing API. A single larger point runs alone;
    this is not an RSS limit and excludes input and persistent cache storage.

    Calls on one instance are serialized. The persistent frequency workers
    run independent points, and cached vectors remain read-only across calls.
    Use ``close()`` or a context manager to release the workers and cache.
    """

    def __init__(self, n, sample_rate, plan, psll=200, overlap=None,
                 workers=8, max_cache_mb=512, max_working_mb=2048):
        started = time.perf_counter()
        self._lock = threading.RLock()
        self._closed = False
        self._pool = None
        self._points = []
        self.last_profile = None
        self.n = _positive_integer(n, 'n')
        self.sample_rate = _finite_number(sample_rate, 'sample_rate')
        if self.sample_rate <= 0:
            raise ValueError('sample_rate must be positive.')
        self.psll = _finite_number(psll, 'psll')
        self.beta = float(_kaiser_alpha(self.psll) * np.pi)
        if not math.isfinite(self.beta):
            raise ValueError('psll produces a nonfinite Kaiser parameter.')
        self.overlap = _finite_number(
            _kaiser_rov(_kaiser_alpha(self.psll)) if overlap is None else overlap,
            'overlap')
        if not 0 <= self.overlap < 1:
            raise ValueError('overlap must be in [0, 1).')
        self.workers = _positive_integer(workers, 'workers')
        max_cache_mb = _finite_number(max_cache_mb, 'max_cache_mb')
        max_working_mb = _finite_number(max_working_mb, 'max_working_mb')
        if max_cache_mb < 0 or max_working_mb <= 0:
            raise ValueError('max_cache_mb must be nonnegative and max_working_mb positive.')
        self._cache_limit = int(max_cache_mb * _MIB)
        self._working_capacity = max(1, int(max_working_mb * _MIB))
        self.plan = _copy_plan(plan, self.n, self.sample_rate, self.overlap)
        self.frequencies = self.plan[0]
        nf = len(self.frequencies)
        self.effective_workers = min(self.workers, nf)
        self._lib = _native_prepared()
        self._mantissa_bits = self._lib.native_long_double_mantissa_bits()
        self._mode = 3 if self._mantissa_bits > 64 else 2
        self._use_bounded = (self.effective_workers == 1
                             and bool(self._lib.native_segment_fma_supported()))
        self._points = [None] * nf
        cached_indices = []
        cached_bytes = 0
        for j, length in enumerate(self.plan[3]):
            needed = 16 * int(length)
            if cached_bytes + needed <= self._cache_limit:
                cached_indices.append(j)
                cached_bytes += needed
        gate = api._MemoryGate(self._working_capacity)
        cache_started = time.perf_counter()
        try:
            if self.effective_workers > 1:
                self._pool = ThreadPoolExecutor(max_workers=self.effective_workers,
                                                thread_name_prefix='lpsd-prepared')

            def prepare_cached(j):
                length = int(self.plan[3][j])
                reserved = gate.acquire(24 * length + 8192)
                try:
                    point, _ = self._prepare_point(j, profile=False)
                    self._points[j] = point
                finally:
                    gate.release(reserved)

            if self._pool is None:
                list(map(prepare_cached, cached_indices))
            else:
                list(self._pool.map(prepare_cached, cached_indices))
        except BaseException:
            self.close()
            raise
        cache_wall = time.perf_counter() - cache_started
        self._tasks, task_costs = self._frequency_tasks()
        self._metadata = {
            'native_prepared_abi': 1, 'estimator': 'lpsd-1.0.6',
            'kernel': 'fast', 'window': 'kaiser', 'psll': self.psll,
            'kaiser_beta': self.beta, 'overlap': self.overlap,
            'detrending_order': 0, 'outputs': 'psd', 'dtype': 'float32',
            'n': self.n, 'sample_rate': self.sample_rate,
            'frequencies': nf, 'workers': self.workers,
            'effective_workers': self.effective_workers,
            'frequency_task_count': len(self._tasks),
            'min_task_estimated_work': min(task_costs, default=0),
            'max_task_estimated_work': max(task_costs, default=0),
            'task_cost_note': 'Scheduling proxy only: L*K plus 5*L for an uncached point; '
                              'greedy cost balancing into at most twice the worker count.',
            'native_mode': self._mode,
            'native_long_double_mantissa_bits': self._mantissa_bits,
            'bounded_segment_fma_enabled': self._use_bounded,
            'cached_frequencies': len(cached_indices),
            'uncached_frequencies': nf - len(cached_indices),
            'cached_coefficient_bytes': cached_bytes,
            'all_coefficient_bytes': sum(16 * int(v) for v in self.plan[3]),
            'cache_limit_bytes': self._cache_limit,
            'concurrency_budget_bytes': self._working_capacity,
            'cache_preparation_wall_s': cache_wall,
            'preparation_wall_s': time.perf_counter() - started,
            'preparation_peak_reserved_bytes': gate.peak,
            'cache_policy': 'Greedy admission in supplied plan order; no eviction.',
            'prepared_data': 'Read-only projected q_r/q_i and window S1/S2; no input samples.',
            'compute_scope': 'Input validation, optional input bound, uncached preparation, '
                             'all original segment projections and power updates, PSD rounding.',
            'validated_compute_scope': 'Private _compute_validated requires finite aligned '
                                       'contiguous float64 input checked by its caller; '
                                       'the full input validation pass is omitted there.',
            'working_budget_note': 'Concurrent temporary reservations only; a larger point '
                                   'runs alone. Input and persistent cache are separate.',
            'legacy_first_segment_recurrence': True,
        }

    @property
    def metadata(self):
        """JSON-serializable preparation details, not a complete-call timer."""
        return self._metadata.copy()

    def _frequency_tasks(self):
        """Balance immutable task groups once, without using sample values.

        L*K estimates segment work. Five additional L-sized passes stand in
        for uncached window generation, sums, coefficients and projection.
        These weights only schedule independent, unchanged frequency points;
        they are not measurements or an alternative operation-count claim.
        """
        nf = len(self.frequencies)
        if not nf:
            return (), ()
        costs = [int(length) * (int(count) + (5 if point is None else 0))
                 for length, count, point in zip(self.plan[3], self.plan[4], self._points)]
        if self.effective_workers <= 1:
            return (tuple(range(nf)),), (sum(costs),)
        count = min(nf, 2 * self.effective_workers)
        groups = [[] for _ in range(count)]
        loads = [(0, j) for j in range(count)]
        heapq.heapify(loads)
        for j in sorted(range(nf), key=lambda j: (-costs[j], j)):
            work, group = heapq.heappop(loads)
            groups[group].append(j)
            heapq.heappush(loads, (work + costs[j], group))
        costs_by_group = dict((group, work) for work, group in loads)
        return tuple(tuple(group) for group in groups), tuple(costs_by_group[j] for j in range(count))

    def _prepare_point(self, j, profile=False):
        length = int(self.plan[3][j])
        phases = {} if profile else None
        started = time.perf_counter() if profile else 0.0
        window = np.empty(length, dtype=np.float64)
        status = self._lib.generate_kaiser_series(api._pointer(window), length, self.beta)
        if status:
            raise ValueError('Native Kaiser window generation failed.')
        if profile:
            phases['window_generation_s'] = time.perf_counter() - started
            started = time.perf_counter()
        s1, s2 = ct.c_double(), ct.c_double()
        status = self._lib.window_sums(api._pointer(window), length,
                                       ct.byref(s1), ct.byref(s2))
        if status:
            raise RuntimeError('Native window normalization failed.')
        if not math.isfinite(s1.value) or not math.isfinite(s2.value) or not s1.value or not s2.value:
            raise ValueError('The selected Kaiser window has invalid normalization.')
        if profile:
            phases['window_sums_s'] = time.perf_counter() - started
            started = time.perf_counter()
        qr, qi = api._coefficients(window, self.plan[2][j], length, blocked=True)
        if profile:
            phases['coefficients_s'] = time.perf_counter() - started
            started = time.perf_counter()
        qr_pointer, qi_pointer = api._pointer(qr), api._pointer(qi)
        status = self._lib.fast_dft_prepare_order0(qr_pointer, qi_pointer, length, self._mode)
        if status:
            raise RuntimeError(_ERRORS.get(status, f'native error {status}'))
        if profile:
            phases['coefficient_projection_s'] = time.perf_counter() - started
        qr.setflags(write=False)
        qi.setflags(write=False)
        return _ProjectedPoint(qr, qi, qr_pointer, qi_pointer, s1.value, s2.value), phases

    def compute(self, values, *, profile=False):
        """Evaluate every prepared point and return an owning float32 PSD.

        With ``profile=True``, ``last_profile`` separates per-frequency
        uncached coefficient preparation from the native input-dependent
        segment work. Worker durations overlap and are not wall-clock sums.
        Ordinary calls do not collect those per-frequency timers.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError('PreparedLPSDSubset is closed.')
            self.last_profile = None
            started = time.perf_counter() if profile else 0.0
            x = api._array(values)
            if len(x) != self.n:
                raise ValueError('Input length differs from the prepared length.')
            validation_s = time.perf_counter() - started if profile else 0.0
            result = self._compute_validated(x, profile=profile)
            if profile:
                self.last_profile['input_validation_s'] = validation_s
                self.last_profile['input_validation_performed'] = True
                self.last_profile['call_wall_s'] = time.perf_counter() - started
            return result

    def _compute_validated(self, values, profile=False):
        """Evaluate input already fully validated by a surrounding pipeline.

        The caller must have checked that all samples are finite and must
        keep them fixed until this call completes. Constant-time array/shape
        checks remain here; there is no conversion or second finite scan.
        Public ``compute`` always performs the full validation itself.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError('PreparedLPSDSubset is closed.')
            self.last_profile = None
            started = time.perf_counter() if profile else 0.0
            if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('float64')
                    or values.ndim != 1 or len(values) != self.n
                    or not values.flags.c_contiguous or not values.flags.aligned):
                raise ValueError('Validated input must be an aligned contiguous float64 '
                                 'vector of the prepared length.')
            x = values
            bound_started = time.perf_counter() if profile else 0.0
            input_peak = (max(abs(float(np.min(x))), abs(float(np.max(x))))
                          if self._use_bounded else math.nan)
            bound_s = time.perf_counter() - bound_started if profile else 0.0
            xp = api._pointer(x)
            gate = api._MemoryGate(self._working_capacity)
            rows = [None] * len(self.frequencies) if profile else None

            def one(j):
                wait_started = time.perf_counter() if profile else 0.0
                point = self._points[j]
                length = int(self.plan[3][j])
                reserved = gate.acquire(8192 if point is not None else 24 * length + 8192)
                worker_started = time.perf_counter() if profile else 0.0
                try:
                    cached = point is not None
                    phases = None
                    if not cached:
                        point, phases = self._prepare_point(j, profile=profile)
                    pr, pi, navg = ct.c_double(), ct.c_double(), ct.c_long()
                    if profile:
                        prep_s, segment_s, fused_batches = ct.c_double(), ct.c_double(), ct.c_long()
                    kernel_started = time.perf_counter() if profile else 0.0
                    status = self._lib.fast_dft_prepared_order0(
                        ct.byref(pr), ct.byref(pi), ct.byref(navg), xp,
                        self.n, length, point.qr_pointer, point.qi_pointer,
                        self.overlap * 100.0, self._mode, input_peak,
                        ct.byref(prep_s) if profile else None,
                        ct.byref(segment_s) if profile else None,
                        ct.byref(fused_batches) if profile else None)
                    if status:
                        raise RuntimeError(_ERRORS.get(status, f'native error {status}'))
                    # Retain the original complex arithmetic and division
                    # order before the public float32 rounding point.
                    density = 2.0 * complex(pr.value, pi.value) / self.sample_rate / point.s2
                    if density.imag != 0.0:
                        raise RuntimeError('Native auto-spectrum returned an imaginary PSD.')
                    if profile:
                        rows[j] = {
                            'j': j, 'frequency': float(self.frequencies[j]),
                            'L': length, 'K': navg.value, 'cached_coefficients': cached,
                            'memory_gate_wait_s': worker_started - wait_started,
                            'window_generation_s': 0.0, 'window_sums_s': 0.0,
                            'coefficients_s': 0.0, 'coefficient_projection_s': 0.0,
                            'c_preparation_s': prep_s.value, 'c_segments_s': segment_s.value,
                            'c_kernel_s': time.perf_counter() - kernel_started,
                            'fused_segment_batches': fused_batches.value,
                            'worker_elapsed_s': time.perf_counter() - worker_started,
                        }
                        if phases is not None:
                            rows[j].update(phases)
                    return density
                finally:
                    gate.release(reserved)

            computed = [None] * len(self.frequencies)

            def group(indices):
                for j in indices:
                    computed[j] = one(j)

            if self._pool is None:
                for indices in self._tasks:
                    group(indices)
            else:
                futures = [self._pool.submit(group, indices) for indices in self._tasks]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    # Do not return while another worker still reads this
                    # call's input. Keep the instance reusable after failure.
                    for future in futures:
                        future.cancel()
                    for future in futures:
                        if not future.cancelled():
                            try:
                                future.result()
                            except BaseException:
                                pass
                    raise
            assembly_started = time.perf_counter() if profile else 0.0
            result = np.asarray(computed, dtype=np.complex64).real.copy()
            if profile:
                self.last_profile = {
                    'call_wall_s': time.perf_counter() - started,
                    'input_validation_s': 0.0, 'input_validation_performed': False,
                    'input_bound_s': bound_s,
                    'output_assembly_s': time.perf_counter() - assembly_started,
                    'frequencies': rows, 'workers': self.effective_workers,
                    'frequency_task_count': len(self._tasks),
                    'concurrency_budget_bytes': self._working_capacity,
                    'peak_reserved_bytes': gate.peak,
                    'note': 'Per-worker durations overlap and must not be summed as wall clock. '
                            'Uncached window/coefficient preparation is data-independent but '
                            'is included in this call. Native preparation now contains setup '
                            'and the optional coefficient bound, not coefficient projection.',
                }
            return result

    def close(self):
        """Release persistent coefficient storage and frequency workers."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
            self._points.clear()

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('PreparedLPSDSubset is closed.')
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
