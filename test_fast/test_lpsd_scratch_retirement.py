"""Gates for one owned-pool retirement, with small test-only thresholds.

These tests are prepared for the candidate patch. They never need a billion
sample input or a large scratch allocation. Numerical references keep the
same plan, native kernels, streaming choices and input values.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import weakref

import numpy as np
import pandas as pd
import pytest

from benchmarks import _lpsd_disk as disk


FS = 50.0


@pytest.mark.parametrize("lengths,streamed,workers,initial,expected", [
    ([900, 10, 900, 30, 20], [False] * 5, 2, 1800, (3, 50)),
    ([900, 32, 32], [False] * 3, 2, 932, (1, 64)),
    ([900, 40, 60], [False] * 3, 1, 900, (1, 60)),
    ([900, 12000, 20, 16], [False, True, False, False], 2, 920, (1, 36)),
    ([900, 12000], [False, True], 2, 900, (1, 1)),
    ([10, 20, 900], [False] * 3, 2, 920, None),
    ([12000, 24000], [True, True], 2, 5000, None),
    ([32, 32, 16], [False] * 3, 2, 64, None),
    ([900], [False], 1, 900, None),
    ([], [], 0, 1, None),
])
def test_scalar_split_uses_the_entire_remaining_plan(
        monkeypatch, lengths, streamed, workers, initial, expected):
    monkeypatch.setattr(disk, "_SCRATCH_RETIRE_TARGET_BYTES", 64 * 16)
    before = (list(lengths), list(streamed))
    assert disk._scratch_retirement_split(lengths, streamed, workers, initial) == expected
    assert (lengths, streamed) == before


def test_suffix_planning_reads_each_item_once(monkeypatch):
    class Counted:
        def __init__(self, values):
            self.values = values
            self.reads = 0

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            self.reads += 1
            return self.values[index]

    lengths = Counted(list(range(5000, 0, -1)))
    streamed = Counted([False] * len(lengths))
    monkeypatch.setattr(disk, "_SCRATCH_RETIRE_TARGET_BYTES", 256 * 16)
    assert disk._scratch_retirement_split(lengths, streamed, 7, 35000) is not None
    assert lengths.reads == streamed.reads == len(lengths)


def test_real_billion_plan_split_needs_no_sample_or_scratch_allocation():
    with disk.DiskLPSDSubset(1_000_000_000, FS, workers=8, max_working_mb=2048) as method:
        assert method.recommended_ram_scratch_slots == 89478485
        split, slots = disk._scratch_retirement_split(
            method.plan[3], method._streaming_mask, method.effective_workers,
            method.recommended_ram_scratch_slots)
        assert 0 < split < len(method.frequencies)
        remaining = sorted((int(v) for v, s in zip(
            method.plan[3][split:], method._streaming_mask[split:]) if not s), reverse=True)
        previous = sorted((int(v) for v, s in zip(
            method.plan[3][split - 1:], method._streaming_mask[split - 1:]) if not s), reverse=True)
        assert slots == sum(remaining[:8])
        assert 16 * slots <= 128 * 1024**2
        assert 16 * sum(previous[:8]) > 128 * 1024**2


def _plan(n, lengths):
    lengths = np.asarray(lengths, dtype=np.float64)
    f = np.linspace(0.217, 17.013, len(lengths))
    r = FS / lengths
    overlap = disk.api._kaiser_rov(disk.api._kaiser_alpha(200.0))
    ovfact = 1.0 / (1.0 - (overlap * 100.0) / 100.0)
    counts = np.floor(((n - lengths) * ovfact) / lengths + 1.0 + 0.5)
    return f, r, f / r, lengths, counts


def _values(kind, n=8192):
    noise = np.random.default_rng(20261012).standard_normal(n)
    if kind == "dc_nanovolt":
        return 10.0 + 1e-9 * noise
    if kind == "offbin_tone":
        return (np.sqrt(2.0) * 1e-6 * np.sin(2 * np.pi * 3.123456 * np.arange(n) / FS)
                + 1e-12 * noise)
    return 1e-8 * noise


@contextmanager
def _mapped_input(tmp_path, values):
    path = tmp_path / "input.f64"
    data = np.memmap(path, dtype=np.float64, mode="w+", shape=(len(values),))
    data[:] = values
    data.flags.writeable = False
    try:
        yield data
    finally:
        data._mmap.close()


def _track_resources(monkeypatch):
    gates, pools, executors = [], [], []
    original_gate, original_pool = disk._FairMemoryGate, disk._ScratchPool
    original_executor = disk.ThreadPoolExecutor

    class Gate(original_gate):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            gates.append(self)

    class Pool(original_pool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)

    class Executor(original_executor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            executors.append(self)

    monkeypatch.setattr(disk, "_FairMemoryGate", Gate)
    monkeypatch.setattr(disk, "_ScratchPool", Pool)
    monkeypatch.setattr(disk, "ThreadPoolExecutor", Executor)
    return gates, pools, executors


def _assert_released(gates, pools):
    assert all(gate.used == 0 and not gate._waiters for gate in gates)
    assert all(pool.used_slots == 0 and not pool._waiters for pool in pools)


@pytest.mark.parametrize("kind", ["white", "dc_nanovolt", "offbin_tone"])
@pytest.mark.parametrize("workers,split,final_slots", [(1, 3, 127), (3, 4, 176)])
def test_owned_pool_is_gone_before_replacement_and_results_are_unchanged(
        tmp_path, monkeypatch, kind, workers, split, final_slots):
    values = _values(kind)
    plan = _plan(len(values), [4096, 1537, 513, 127, 113, 63])
    monkeypatch.setattr(disk, "_SCRATCH_RETIRE_TARGET_BYTES", 4096)
    with disk.DiskLPSDSubset(len(values), FS, plan, workers=workers,
                             streaming="off", max_working_mb=64) as method:
        expected = method.compute(values, outputs=("psd", "nsd"))
        expected_rows = copy.deepcopy(method.last_profile["frequencies"])
        gate_capacity, mask = method._working_capacity, method._streaming_mask
        gates, pools, executors = _track_resources(monkeypatch)
        finished, events, references, validation_calls = set(), [], [], []
        original_empty, original_validate = disk.np.empty, method._validate

        def progress(event):
            events.append(copy.deepcopy(event))
            finished.update(point["j"] for point in event["points"])

        def validate(*args, **kwargs):
            validation_calls.append(1)
            return original_validate(*args, **kwargs)

        def allocate(*args, **kwargs):
            dtype = kwargs.get("dtype", args[1] if len(args) > 1 else np.float64)
            is_pool = np.dtype(dtype) == np.dtype(np.complex128)
            if is_pool and references:
                assert len(references) == 1
                # A drained gate alone is insufficient: every old NumPy
                # view and native-call pointer must have stopped owning it.
                assert references[0]() is None
                _assert_released(gates, pools)
                assert finished == set(range(split))
            array = original_empty(*args, **kwargs)
            if is_pool:
                references.append(weakref.ref(array))
            return array

        method._progress, method._progress_interval_s = progress, 0.0
        monkeypatch.setattr(method, "_validate", validate)
        monkeypatch.setattr(disk.np, "empty", allocate)
        with _mapped_input(tmp_path, values) as data:
            actual = method.compute(data, outputs=("psd", "nsd"))
            assert data.tobytes() == values.tobytes()
            assert not data._mmap.closed
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_flags=False)
        assert method._working_capacity == gate_capacity
        assert method._streaming_mask == mask
        assert validation_calls == [1]
        assert len(executors) == (1 if workers > 1 else 0)
        assert len(gates) == 1 and len(pools) == len(references) == 2
        assert all(reference() is None for reference in references)
        _assert_released(gates, pools)
        details = method.last_profile
        lifecycle = details["scratch_pool_lifecycle"]
        initial = 16 * method.recommended_ram_scratch_slots
        assert lifecycle["initial_bytes"] == lifecycle["peak_allocated_bytes"] == initial
        assert lifecycle["final_bytes"] == 16 * final_slots
        assert lifecycle["planned_retirement_index"] == lifecycle["retirement_index"] == split
        assert lifecycle["state"] == "retired"
        assert details["scratch_buffer_bytes"] == initial
        assert details["peak_active_scratch_bytes"] == 16 * max(pool.peak_slots for pool in pools)
        assert actual.attrs["lpsd_disk"]["actual_scratch_bytes"] == initial
        assert actual.attrs["lpsd_disk"]["scratch_pool_lifecycle"] == lifecycle
        assert [event["phase"] for event in events].count("lpsd_points_started") == 1
        assert [event["phase"] for event in events].count("lpsd_points_completed") == 1
        assert details["progress"]["completed_points"] == len(plan[0])
        for row, old in zip(details["frequencies"], expected_rows):
            for key in ("j", "source_j", "frequency", "L", "K", "planned_K", "execution_method",
                        "window_s1", "window_s2", "enbw_hz"):
                assert row[key] == old[key]


@pytest.mark.parametrize("lengths,streaming,workers,split", [
    ([4096, 63, 1537, 127, 113], "off", 3, 3),
    ([1537, 4096, 63], "on", 2, 1),
    ([4096, 3072], "on", 2, None),
    ([4096], "off", 1, None),
    ([], "on", 1, None),
])
def test_nonmonotone_and_streaming_subsets(tmp_path, monkeypatch, lengths, streaming, workers, split):
    monkeypatch.setattr(disk, "_SCRATCH_RETIRE_TARGET_BYTES", 4096)
    values = _values("dc_nanovolt")
    with disk.DiskLPSDSubset(len(values), FS, _plan(len(values), lengths),
                             workers=workers, streaming=streaming) as method:
        expected = method.compute(values, outputs=("psd", "nsd"))
        mask = method._streaming_mask
        with _mapped_input(tmp_path, values) as data:
            first = method.compute(data, outputs=("psd", "nsd"))
            first_lifecycle = copy.deepcopy(method.metadata["scratch_pool_lifecycle"])
            second = method.compute(data, outputs=("psd", "nsd"), profile=False)
            assert data.tobytes() == values.tobytes()
        pd.testing.assert_frame_equal(first, expected, check_exact=True, check_flags=False)
        pd.testing.assert_frame_equal(second, expected, check_exact=True, check_flags=False)
        assert first_lifecycle == method.metadata["scratch_pool_lifecycle"]
        assert first_lifecycle["retirement_index"] == split
        assert method._streaming_mask == mask
        assert method.last_profile["frequencies"] == []


def test_masked_plan_keeps_local_and_original_point_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(disk, "_SCRATCH_RETIRE_TARGET_BYTES", 4096)
    values = _values("white")
    plan = _plan(len(values), [4096, 8192, 1537, 37, 513, 127, 113, 63])
    mask = np.array([True, False, True, False, True, True, True, True])
    with disk.DiskLPSDSubset(len(values), FS, plan, plan_mask=mask,
                             workers=3, streaming="off") as method:
        expected = method.compute(values, outputs=("psd", "nsd"))
        with _mapped_input(tmp_path, values) as data:
            actual = method.compute(data, outputs=("psd", "nsd"))
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_flags=False)
        lifecycle = method.metadata["scratch_pool_lifecycle"]
        assert lifecycle["retirement_index"] == 4
        assert lifecycle["retirement_source_index"] == 6
        assert [row["source_j"] for row in method.last_profile["frequencies"]] == [0, 2, 4, 5, 6, 7]


@pytest.mark.parametrize("external", [False, True])
def test_ram_input_and_external_scratch_keep_the_existing_lifecycle(tmp_path, monkeypatch, external):
    monkeypatch.setattr(disk, "_SCRATCH_RETIRE_TARGET_BYTES", 4096)
    values = _values("white")
    with disk.DiskLPSDSubset(len(values), FS, _plan(len(values), [4096, 513, 63]),
                             workers=2, streaming="off") as method:
        if external:
            scratch = np.memmap(tmp_path / "scratch.c128", dtype=np.complex128,
                                mode="w+", shape=(len(values),))
            try:
                with _mapped_input(tmp_path, values) as data:
                    method.compute(data, scratch)
                    assert not scratch._mmap.closed and not data._mmap.closed
                    assert data.tobytes() == values.tobytes()
                expected_bytes = scratch.nbytes
            finally:
                scratch._mmap.close()
        else:
            method.compute(values)
            expected_bytes = 16 * method.recommended_ram_scratch_slots
        lifecycle = method.metadata["scratch_pool_lifecycle"]
        assert lifecycle["eligible_storage"] is False
        assert lifecycle["state"] == "unchanged"
        assert lifecycle["planned_retirement_index"] is lifecycle["retirement_index"] is None
        assert lifecycle["initial_bytes"] == lifecycle["final_bytes"] == expected_bytes


@pytest.mark.parametrize("failed_length", [513, 113])
def test_native_failure_before_or_after_retirement_releases_all_reservations(
        tmp_path, monkeypatch, failed_length):
    monkeypatch.setattr(disk, "_SCRATCH_RETIRE_TARGET_BYTES", 4096)
    values = _values("white")
    gates, pools, _ = _track_resources(monkeypatch)
    events = []
    with disk.DiskLPSDSubset(len(values), FS,
                             _plan(len(values), [4096, 1537, 513, 127, 113, 63]),
                             workers=1, streaming="off", progress=events.append,
                             progress_interval_s=0) as method:
        real_lib = method._lib

        class Failure:
            def __getattr__(self, name):
                return getattr(real_lib, name)

            def fast_dft_selected_corrected(self, *args):
                return 3 if args[8] == failed_length else real_lib.fast_dft_selected_corrected(*args)

        method._lib = Failure()
        with _mapped_input(tmp_path, values) as data:
            with pytest.raises(RuntimeError, match="native allocation failed"):
                method.compute(data)
            _assert_released(gates, pools)
            assert data.tobytes() == values.tobytes()
            assert not data._mmap.closed
            assert method.last_profile is None
            lifecycle = method.metadata["scratch_pool_lifecycle"]
            assert lifecycle["retirement_index"] == (None if failed_length == 513 else 3)
            assert events[-1]["phase"] == "lpsd_points_failed"
            method._lib = real_lib
            assert np.isfinite(method.compute(data).psd).all()
            _assert_released(gates, pools)


def test_failed_replacement_has_already_released_the_old_owner_and_can_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(disk, "_SCRATCH_RETIRE_TARGET_BYTES", 4096)
    values = _values("white")
    gates, pools, _ = _track_resources(monkeypatch)
    original_empty, references = disk.np.empty, []
    with disk.DiskLPSDSubset(len(values), FS,
                             _plan(len(values), [4096, 1537, 513, 127, 113, 63]),
                             workers=3, streaming="off") as method:
        def allocate(*args, **kwargs):
            dtype = kwargs.get("dtype", args[1] if len(args) > 1 else np.float64)
            is_pool = np.dtype(dtype) == np.dtype(np.complex128)
            if is_pool and references:
                assert references[0]() is None
                _assert_released(gates, pools)
                raise MemoryError("controlled replacement failure")
            array = original_empty(*args, **kwargs)
            if is_pool:
                references.append(weakref.ref(array))
            return array

        with _mapped_input(tmp_path, values) as data:
            with monkeypatch.context() as patch:
                patch.setattr(disk.np, "empty", allocate)
                with pytest.raises(MemoryError, match="controlled replacement failure"):
                    method.compute(data)
            assert method.last_profile is None
            lifecycle = method.metadata["scratch_pool_lifecycle"]
            assert lifecycle["state"] == "replacement_failed"
            assert lifecycle["final_bytes"] == 0
            assert lifecycle["retirement_index"] is None
            _assert_released(gates, pools)
            assert np.isfinite(method.compute(data).psd).all()
            assert method.metadata["scratch_pool_lifecycle"]["state"] == "retired"
            assert data.tobytes() == values.tobytes()
            _assert_released(gates, pools)


def test_callback_failure_does_not_leave_active_scratch_or_start_the_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(disk, "_SCRATCH_RETIRE_TARGET_BYTES", 4096)
    values = _values("white")
    gates, pools, _ = _track_resources(monkeypatch)

    def progress(event):
        if event["phase"] == "lpsd_points":
            raise RuntimeError("controlled callback failure")

    with disk.DiskLPSDSubset(len(values), FS,
                             _plan(len(values), [4096, 1537, 513, 127, 113, 63]),
                             workers=3, streaming="off", progress=progress,
                             progress_interval_s=0) as method:
        with _mapped_input(tmp_path, values) as data:
            with pytest.raises(RuntimeError, match="controlled callback failure"):
                method.compute(data)
            assert method.last_profile is None
            assert len(pools) == 1
            assert method.metadata["scratch_pool_lifecycle"]["retirement_index"] is None
            _assert_released(gates, pools)
            method._progress = None
            assert np.isfinite(method.compute(data).psd).all()
            assert data.tobytes() == values.tobytes()
            _assert_released(gates, pools)
