"""Prepared order-0 vectors must preserve the existing fast PSD estimator."""
import ctypes as ct
import json

import numpy as np
import pytest

from lpsd_fast import api
from lpsd_fast.planning import _kaiser_alpha, _kaiser_rov, _ltf_plan
from lpsd_fast.prepared import PreparedLPSDSubset, _native_prepared


FS = 100.0
OVERLAP = _kaiser_rov(_kaiser_alpha(200))


def _plan(n=4097):
    full = _ltf_plan(n, FS, OVERLAP, 1, 0, 64, 16)
    selected = np.unique(np.linspace(0, len(full[0]) - 1, 7).astype(int))
    return tuple(np.asarray(part)[selected] for part in full)


def _signal(n, kind):
    rng = np.random.default_rng(7321)
    result = rng.normal(size=n)
    if kind == 'dc_noise':
        result = 10 + 1e-9 * result
    elif kind == 'tone':
        result = 1.3 * np.sin(2 * np.pi * 4.371 * np.arange(n) / FS) + .01 * result
    elif kind == 'initial_transient':
        # Exercise the cancellation-sensitive, inherited ii=1 power reset.
        result[1] = 1e10
    elif kind == 'constant':
        result.fill(10)
    return result


def _reference(x, plan, workers):
    return api._run_channel(
        x, x, FS, plan, np.kaiser, 200, OVERLAP, 0,
        min(workers, len(plan[0])), 'fast', 64, 0, False, ('psd',), False
    ).psd.to_numpy()


@pytest.mark.parametrize('workers', (1, 4))
@pytest.mark.parametrize('cache_mb', (0, .025, 4))
@pytest.mark.parametrize('kind', ('white', 'dc_noise', 'tone', 'initial_transient', 'constant'))
def test_prepared_psd_preserves_reference_and_rounding(workers, cache_mb, kind):
    n = 4097
    plan = _plan(n)
    x = _signal(n, kind)
    original = x.copy()
    expected = _reference(x, plan, workers)
    with PreparedLPSDSubset(n, FS, plan, workers=workers,
                            max_cache_mb=cache_mb, max_working_mb=1) as prepared:
        actual = prepared.compute(x)
        assert actual.dtype == np.float32 and actual.flags.owndata
        assert actual.shape == (len(plan[0]),)
        assert actual.tobytes() == expected.tobytes()
        assert x.tobytes() == original.tobytes()
        metadata = prepared.metadata
        assert metadata['cached_coefficient_bytes'] <= metadata['cache_limit_bytes']
        assert metadata['cached_frequencies'] + metadata['uncached_frequencies'] == len(plan[0])
        assert metadata['legacy_first_segment_recurrence']
        json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize('mode', (2, 3))
@pytest.mark.parametrize('length,segments', ((128, 1), (128, 17), (259, 19), (1027, 17), (2049, 33)))
@pytest.mark.parametrize('bounded', (False, True))
def test_native_prepared_projection_reuses_identical_segment_arithmetic(mode, length, segments, bounded):
    lib = _native_prepared()
    n = length + round((segments - 1) * (length / 4 + .125))
    x = _signal(n, 'dc_noise')
    phase = 2 * np.pi * 4.371 * np.arange(length) / length
    window = np.kaiser(length + 1, 23.7)[:-1]
    cr, ci = window * np.cos(phase), window * np.sin(phase)
    reference_r, reference_i = ct.c_double(), ct.c_double()
    reference_count = ct.c_long()
    original_cr, original_ci = cr.copy(), ci.copy()
    args = (ct.byref(reference_r), ct.byref(reference_i), None, None,
            ct.byref(reference_count), api._pointer(x), None, n, length,
            api._pointer(original_cr), api._pointer(original_ci),
            75., 0, False, mode, False, True)
    peak = max(abs(float(np.min(x))), abs(float(np.max(x)))) if bounded else np.nan
    if bounded:
        status = lib.fast_dft_selected_bounded(*args, peak, None, None, None)
    else:
        status = lib.fast_dft_selected(*args)
    assert status == 0
    qr, qi = cr.copy(), ci.copy()
    assert lib.fast_dft_prepare_order0(api._pointer(qr), api._pointer(qi), length, mode) == 0
    assert qr.tobytes() == original_cr.tobytes()
    assert qi.tobytes() == original_ci.tobytes()
    saved = (qr.tobytes(), qi.tobytes())
    qr.setflags(write=False)
    qi.setflags(write=False)
    actual_r, actual_i, count = ct.c_double(), ct.c_double(), ct.c_long()
    preparation_s, segment_s = ct.c_double(), ct.c_double()
    assert lib.fast_dft_prepared_order0(
        ct.byref(actual_r), ct.byref(actual_i), ct.byref(count),
        api._pointer(x), n, length, api._pointer(qr), api._pointer(qi),
        75., mode, peak, ct.byref(preparation_s), ct.byref(segment_s), None
    ) == 0
    assert count.value == reference_count.value == segments
    assert actual_r.value == reference_r.value
    assert actual_i.value == reference_i.value == 0
    assert saved == (qr.tobytes(), qi.tobytes())
    assert preparation_s.value >= 0 and segment_s.value >= 0


def test_cached_vectors_and_previous_output_survive_changed_inputs_and_profiles():
    n, plan = 4097, _plan()
    x = _signal(n, 'white')
    with PreparedLPSDSubset(n, FS, plan, workers=4, max_cache_mb=4) as prepared:
        cache_before = [(p.qr.tobytes(), p.qi.tobytes()) for p in prepared._points]
        assert all(not p.qr.flags.writeable and not p.qi.flags.writeable for p in prepared._points)
        first = prepared.compute(x)
        saved = first.copy()
        second = prepared.compute(10 + 1e-9 * x, profile=True)
        assert not np.shares_memory(first, second)
        assert first.tobytes() == saved.tobytes()
        profile = prepared.last_profile
        assert profile['call_wall_s'] >= 0
        assert all(row['cached_coefficients'] for row in profile['frequencies'])
        assert all(row['window_generation_s'] == row['coefficients_s'] ==
                   row['coefficient_projection_s'] == 0 for row in profile['frequencies'])
        assert prepared.compute(x, profile=True).tobytes() == saved.tobytes()
        assert cache_before == [(p.qr.tobytes(), p.qi.tobytes()) for p in prepared._points]
        assert prepared.compute(x).tobytes() == saved.tobytes()
        assert prepared.last_profile is None


def test_uncached_profile_and_budget_do_not_change_the_estimator():
    n, plan = 4097, _plan()
    x = _signal(n, 'tone')
    with PreparedLPSDSubset(n, FS, plan, workers=4, max_cache_mb=0,
                            max_working_mb=.01) as prepared:
        result = prepared.compute(x, profile=True)
        profile = prepared.last_profile
        assert profile['peak_reserved_bytes'] <= profile['concurrency_budget_bytes']
        assert not any(row['cached_coefficients'] for row in profile['frequencies'])
        assert all(row['coefficient_projection_s'] >= 0 for row in profile['frequencies'])
        assert result.tobytes() == _reference(x, plan, 4).tobytes()


def test_plan_is_copied_and_close_is_idempotent():
    plan = tuple(a.copy() for a in _plan())
    x = _signal(4097, 'white')
    prepared = PreparedLPSDSubset(len(x), FS, plan, max_cache_mb=4)
    expected = prepared.compute(x)
    for a in plan:
        a.fill(0)
    assert prepared.compute(x).tobytes() == expected.tobytes()
    prepared.close()
    prepared.close()
    assert prepared._points == [] and prepared._pool is None
    with pytest.raises(RuntimeError, match='closed'):
        prepared.compute(x)


@pytest.mark.parametrize('bad', (np.zeros(4), np.ones((4097, 1)),
                               np.zeros(4097, dtype=complex), np.full(4097, np.nan)))
def test_input_validation_preserves_the_prepared_cache(bad):
    with PreparedLPSDSubset(4097, FS, _plan(), max_cache_mb=4) as prepared:
        with pytest.raises(ValueError):
            prepared.compute(bad)
        assert np.isfinite(prepared.compute(_signal(4097, 'white'))).all()


def test_one_point_and_empty_subsets():
    plan = _plan()
    x = _signal(4097, 'white')
    one = tuple(a[-1:] for a in plan)
    with PreparedLPSDSubset(len(x), FS, one, workers=8) as prepared:
        assert prepared.effective_workers == 1
        assert prepared.compute(x).tobytes() == _reference(x, one, 8).tobytes()
    empty = tuple(a[:0] for a in plan)
    with PreparedLPSDSubset(len(x), FS, empty) as prepared:
        result = prepared.compute(x)
        assert result.shape == (0,) and result.dtype == np.float32 and result.flags.owndata


@pytest.mark.parametrize('change', ('length', 'count', 'nan', 'shape', 'nyquist'))
def test_invalid_plans_rejected_before_native_access(change):
    plan = [a.copy() for a in _plan()]
    if change == 'length':
        plan[3][0] = 4098
    elif change == 'count':
        plan[4][0] = 4097
    elif change == 'nan':
        plan[0][0] = np.nan
    elif change == 'shape':
        plan[1] = plan[1][1:]
    else:
        plan[0][0] = FS
    with pytest.raises(ValueError):
        PreparedLPSDSubset(4097, FS, plan)


@pytest.mark.parametrize('cache_mb', (0, .08, 8))
@pytest.mark.parametrize('kind', ('white', 'dc_noise', 'initial_transient'))
def test_grouped_frequency_tasks_keep_reference_order_and_budget(cache_mb, kind):
    n = 8193
    plan = tuple(np.asarray(part) for part in _ltf_plan(n, FS, OVERLAP, 1, 0, 64, 16))
    x = _signal(n, kind)
    expected = _reference(x, plan, 4)
    with PreparedLPSDSubset(n, FS, plan, workers=4, max_cache_mb=cache_mb,
                            max_working_mb=.02) as prepared:
        assert len(plan[0]) > 8
        assert prepared.metadata['frequency_task_count'] == 8
        actual = prepared.compute(x, profile=True)
        assert actual.tobytes() == expected.tobytes()
        assert prepared._compute_validated(x).tobytes() == expected.tobytes()
        prepared._compute_validated(x, profile=True)
        profile = prepared.last_profile
        assert [row['j'] for row in profile['frequencies']] == list(range(len(plan[0])))
        assert profile['peak_reserved_bytes'] <= profile['concurrency_budget_bytes']
        assert profile['frequency_task_count'] == 8


def test_private_validated_path_avoids_a_second_input_validation(monkeypatch):
    x = _signal(4097, 'white')
    with PreparedLPSDSubset(len(x), FS, _plan(), max_cache_mb=4) as prepared:
        original = api._array
        calls = []

        def track_array(values):
            calls.append(values)
            return original(values)

        monkeypatch.setattr(api, '_array', track_array)
        expected = prepared.compute(x, profile=True)
        assert len(calls) == 1
        assert prepared.last_profile['input_validation_performed']
        assert prepared.last_profile['input_validation_s'] >= 0
        actual = prepared._compute_validated(x, profile=True)
        assert len(calls) == 1
        assert actual.tobytes() == expected.tobytes()
        assert not prepared.last_profile['input_validation_performed']
        assert prepared.last_profile['input_validation_s'] == 0
        with pytest.raises(ValueError, match='finite'):
            prepared.compute(np.full(len(x), np.nan))
        assert len(calls) == 2


@pytest.mark.parametrize('kind', ('list', 'dtype', 'shape', 'length', 'strided', 'unaligned'))
def test_private_validated_path_requires_its_array_contract(kind):
    x = _signal(4097, 'white')
    if kind == 'list':
        bad = x.tolist()
    elif kind == 'dtype':
        bad = x.astype(np.float32)
    elif kind == 'shape':
        bad = x[:, None]
    elif kind == 'length':
        bad = x[:-1]
    elif kind == 'strided':
        bad = np.zeros(2 * len(x))[::2]
    else:
        storage = np.zeros(8 * len(x) + 1, dtype=np.uint8)
        bad = np.ndarray(x.shape, dtype=np.float64, buffer=storage, offset=1)
    with PreparedLPSDSubset(len(x), FS, _plan(), max_cache_mb=4) as prepared:
        with pytest.raises(ValueError, match='aligned contiguous float64'):
            prepared._compute_validated(bad)
        assert np.isfinite(prepared.compute(x)).all()
