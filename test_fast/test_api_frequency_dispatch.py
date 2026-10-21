"""Dynamic frequency dispatch retain public output and failure semantics."""
import threading

import numpy as np
import pandas as pd
import pytest

from lpsd_fast import api


@pytest.mark.parametrize('statistics', ('corrected', 'legacy'))
@pytest.mark.parametrize('kernel', ('scalar', 'auto', 'fast'))
@pytest.mark.parametrize('kind', ('white', 'dc_noise', 'csd', 'enbw'))
def test_frequency_dispatch_keeps_each_output_and_profile_index(monkeypatch, statistics, kernel, kind):
    rng = np.random.default_rng(2067)
    values = rng.standard_normal(4097)
    if kind == 'dc_noise':
        values = 10 + 1e-9 * values
    elif kind == 'csd':
        values = pd.DataFrame({'a': values, 'b': .7 * values + rng.standard_normal(len(values))})
    kwargs = dict(sample_rate=50., n_frequencies=96, n_averages=8,
                  workers=4, kernel=kernel, statistics=statistics,
                  csd=kind == 'csd', outputs='enbw' if kind == 'enbw' else None,
                  max_working_mb=64, window_cache_mb=.01, profile=True)
    original = values.copy()
    actual = api.lpsd(values, **kwargs)
    # The original one-task-per-point scheduler is an independent dispatch
    # control, with the same kernel/worker count (and hence FMA admission).
    def original_dispatch(one, size, workers):
        with api.ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, range(size)))

    monkeypatch.setattr(api, '_parallel_frequency_map', original_dispatch)
    expected = api.lpsd(values, **kwargs)
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    assert tuple(actual.columns) == tuple(expected.columns)
    for name in actual.columns:
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].to_numpy().tobytes() == expected[name].to_numpy().tobytes()
    for result in (actual, expected):
        rows = result.attrs['lpsd_profile']['frequencies']
        assert [row['j'] for row in rows] == list(range(len(result)))
        np.testing.assert_array_equal([row['frequency'] for row in rows], result.index)
        assert result.attrs['lpsd_profile']['peak_reserved_bytes'] <= 64 * 1024**2
    np.testing.assert_array_equal(values, original)


def test_frequency_failure_stops_new_work_and_waits_for_running_input_readers(monkeypatch):
    original = api._coefficients
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    count_lock = threading.Lock()
    calls = 0
    submissions = 0

    class WindowFailure(RuntimeError):
        pass

    class ObservedPool(api.ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            nonlocal submissions
            submission = submissions
            submissions += 1

            def run(*args, **kwargs):
                if submission == 0:
                    assert entered.wait(5), 'Reader did not start before the failing worker'
                try:
                    return fn(*args, **kwargs)
                finally:
                    if submission == 1:
                        finished.set()

            future = super().submit(run, *args, **kwargs)

            def release_after_failure(done):
                if not done.cancelled() and done.exception() is not None:
                    release.set()

            future.add_done_callback(release_after_failure)
            return future

    def coefficients(*args, **kwargs):
        nonlocal calls
        with count_lock:
            call = calls
            calls += 1
        if call == 0:
            entered.set()
            assert release.wait(5), 'Failing peer did not release the running worker'
            return original(*args, **kwargs)
        if call == 1:
            assert entered.wait(5)
            raise WindowFailure('deliberate frequency failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(api, 'ThreadPoolExecutor', ObservedPool)
    monkeypatch.setattr(api, '_coefficients', coefficients)
    try:
        with pytest.raises(WindowFailure, match='deliberate'):
            api.lpsd(np.ones(4097), sample_rate=50., n_frequencies=96,
                     n_averages=8, workers=2, outputs='psd')
        assert finished.is_set(), 'The API returned while a frequency worker was still running'
        assert calls == 2, 'A failed call kept computing unstarted frequencies'
    finally:
        release.set()


def test_failed_late_native_binding_does_not_cache_a_partial_handle(monkeypatch):
    api._native()
    complete = api._LIB
    calls = []

    class MissingLateSymbol:
        def __getattr__(self, name):
            if name == 'generate_cosine_window':
                raise AttributeError(name)
            return getattr(complete, name)

    def load(*args, **kwargs):
        calls.append(args)
        return MissingLateSymbol() if len(calls) == 1 else complete

    monkeypatch.setattr(api, '_LIB', None)
    monkeypatch.setattr(api.ct, 'CDLL', load)
    with pytest.raises(AttributeError, match='generate_cosine_window'):
        api._native()
    assert api._LIB is None
    assert api._native() is complete.fast_dft
    assert api._LIB is complete
    assert len(calls) == 2
