"""Deterministic admission and scalar memory-plan gates for the disk adapter.

The concurrency tests synchronize with condition/event handshakes, not sleeps
or wall-time speed assertions. Large-N cases construct frequency plans only.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from benchmarks._lpsd_disk import DiskLPSDSubset, _FairMemoryGate, _ScratchPool


def _resource(kind, capacity):
    if kind == "scratch":
        resource = _ScratchPool(capacity)
        return resource, resource.acquire, lambda handle, amount: resource.release(handle, amount)
    resource = _FairMemoryGate(capacity)
    return resource, resource.acquire, lambda handle, amount: resource.release(handle)


@pytest.mark.parametrize("kind", ["scratch", "gate"])
def test_existing_large_waiter_cannot_be_overtaken_by_fitting_short_request(kind):
    resource, acquire, release = _resource(kind, 8)
    attempted = {name: threading.Event() for name in ("older-large", "newer-small")}
    waiting = {name: threading.Event() for name in attempted}
    acquired = {name: threading.Event() for name in attempted}
    release_large = threading.Event()
    order, errors = [], []

    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            name = threading.current_thread().name
            if name in waiting:
                waiting[name].set()
                attempted[name].set()
            return super().wait(timeout)

    condition = ObservedCondition()
    if kind == "scratch":
        resource._condition = condition
    else:
        resource.condition = condition
    held = acquire(4)

    def run(name, amount):
        handle = None
        try:
            handle = acquire(amount)
            order.append(name)
            acquired[name].set()
            attempted[name].set()
            if name == "older-large":
                if not release_large.wait(10):
                    raise AssertionError("Test did not release the large reservation")
        except BaseException as error:
            errors.append(error)
            attempted[name].set()
        finally:
            if handle is not None:
                release(handle, amount)

    large = threading.Thread(target=run, args=("older-large", 8),
                             name="older-large", daemon=True)
    small = threading.Thread(target=run, args=("newer-small", 4),
                             name="newer-small", daemon=True)
    started = []
    try:
        large.start()
        started.append(large)
        assert attempted["older-large"].wait(10)
        assert waiting["older-large"].is_set()
        small.start()
        started.append(small)
        assert attempted["newer-small"].wait(10)
        # Four slots/bytes are currently free. The short request must wait
        # anyway, because the older full-pool request is already queued.
        assert waiting["newer-small"].is_set()
        assert not acquired["newer-small"].is_set()
        release(held, 4)
        held = None
        assert acquired["older-large"].wait(10)
        assert not acquired["newer-small"].is_set()
        release_large.set()
        assert acquired["newer-small"].wait(10)
    finally:
        if held is not None:
            release(held, 4)
        release_large.set()
        for thread in started:
            thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in started)
    assert not errors
    assert order == ["older-large", "newer-small"]
    assert not resource._waiters
    if kind == "scratch":
        assert resource.used_slots == 0
        assert resource._free == [(0, 8)]
    else:
        assert resource.used == 0


def test_scratch_coalesces_disjoint_releases_and_reuses_the_whole_pool():
    pool = _ScratchPool(12)
    first, middle, last = pool.acquire(3), pool.acquire(5), pool.acquire(4)
    assert (first, middle, last) == (0, 3, 8)
    assert pool.used_slots == pool.peak_slots == 12
    pool.release(first, 3)
    pool.release(last, 4)
    assert pool._free == [(0, 3), (8, 4)]
    pool.release(middle, 5)
    assert pool._free == [(0, 12)]
    whole = pool.acquire(12)
    assert whole == 0
    pool.release(whole, 12)
    assert pool.used_slots == 0
    assert not pool._waiters


@pytest.mark.parametrize("kind", ["scratch", "gate"])
def test_cancelled_admission_removes_its_fifo_ticket(kind):
    resource, acquire, release = _resource(kind, 8)

    class CancelledWait(RuntimeError):
        pass

    class CancelCondition(threading.Condition):
        def wait(self, timeout=None):
            raise CancelledWait("controlled cancellation before allocation")

    if kind == "scratch":
        resource._condition = CancelCondition()
    else:
        resource.condition = CancelCondition()
    held = acquire(4)
    with pytest.raises(CancelledWait):
        acquire(8)
    assert not resource._waiters
    release(held, 4)
    whole = acquire(8)
    release(whole, 8)
    assert not resource._waiters


def test_local_gate_retains_oversized_mapped_point_admission():
    gate = _FairMemoryGate(64)
    reservation = gate.acquire(128)
    assert reservation == gate.used == gate.peak == 64
    gate.release(reservation)
    assert gate.used == 0


@pytest.mark.parametrize("kind", ["scratch", "gate"])
def test_failed_admission_notification_has_not_committed_a_reservation(kind):
    resource, acquire, release = _resource(kind, 8)

    class CancelNotification(threading.Condition):
        cancelled = False

        def notify_all(self):
            if not self.cancelled:
                self.cancelled = True
                raise RuntimeError("controlled admission cancellation")
            super().notify_all()

    if kind == "scratch":
        resource._condition = CancelNotification()
    else:
        resource.condition = CancelNotification()
    with pytest.raises(RuntimeError, match="controlled admission cancellation"):
        acquire(8)
    assert not resource._waiters
    if kind == "scratch":
        assert resource.used_slots == 0
        assert resource._free == [(0, 8)]
    else:
        assert resource.used == 0
    whole = acquire(8)
    release(whole, 8)


@pytest.mark.parametrize("requested_j,expected_ram_slots", [(1000, 6573014), (2240, 8000000)])
def test_one_million_ram_plan_can_keep_eight_large_points_but_mapped_pool_stays_n(
        requested_j, expected_ram_slots, tmp_path):
    n = 1_000_000
    # No spectrum is computed here. The native planner needs only O(J)
    # scalars; these small files check the existing explicit-map contract.
    with DiskLPSDSubset(n, 50, n_frequencies=requested_j, n_averages=100,
                        workers=8, max_working_mb=2048) as method:
        assert not any(method._streaming_mask)
        assert method.minimum_scratch_slots == method.recommended_scratch_slots == n
        assert method.recommended_ram_scratch_slots == expected_ram_slots > n
        description = method.metadata
        assert description["recommended_mapped_scratch_bytes"] == 16 * n
        assert description["recommended_scratch_bytes"] == 16 * n
        assert description["recommended_ram_scratch_bytes"] == 16 * expected_ram_slots
        model = description["bounded_memory_model"]
        assert model["scratch_bytes"] == 16 * expected_ram_slots
        assert model["working_bytes_upper_bound"] == 16 * expected_ram_slots + min(n, 262144)
        assert model["fits_working_budget"]
        data = np.memmap(tmp_path / "input.f64", mode="w+", dtype=np.float64, shape=(n,))
        scratch = np.memmap(tmp_path / "scratch.c128", mode="w+", dtype=np.complex128, shape=(n,))
        try:
            input_region, scratch_region = method._validate(data, scratch, mapped=True)
            assert input_region["mapped_bytes"] == 8 * n
            assert scratch_region["mapped_bytes"] == 16 * n
        finally:
            scratch._mmap.close()
            data._mmap.close()


@pytest.mark.parametrize("budget", [32_000_000_000, 32 * 1024**3])
def test_billion_plan_uses_explicit_32gb_headroom_without_allocating_input(budget):
    n, workers = 1_000_000_000, 8
    runtime_reserve = 256 * 1024**2
    thread_reserve = 3 * workers * 1024**2
    working = budget - 8 * n - runtime_reserve - thread_reserve
    with DiskLPSDSubset(n, 50, workers=workers, max_working_mb=working / 1024**2) as method:
        model = method.metadata["bounded_memory_model"]
        assert method.recommended_scratch_slots <= n
        assert method.recommended_ram_scratch_slots >= method.recommended_scratch_slots
        assert model["scratch_bytes"] == 16 * method.recommended_ram_scratch_slots
        assert model["working_bytes_upper_bound"] <= working
        assert model["input_scratch_streaming_validation_bytes_upper_bound"] + runtime_reserve + thread_reserve <= budget
        assert model["fits_working_budget"]
        assert method.metadata["streaming"]["eligible_points"] > 0
        if budget == 32 * 1024**3:
            assert method.recommended_ram_scratch_slots > n
