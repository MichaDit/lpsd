"""Scalar native-memory admission contracts; no numerical estimators needed."""
from __future__ import annotations

import pytest

from lpsd_benchmark.resources import GIB, MIB
from lpsd_benchmark.worker import _native_working_budget


def _case(*, n=100_000_000, ram=32 * GIB, workers=8, storage="ram", low=128.):
    return {"n": n, "ram_budget_bytes": ram, "requested_threads": workers,
            "storage": storage, "low_working_mb": low}


def test_ram_native_allowance_uses_32gib_budget_instead_of_a_2gib_cap():
    large = _case()
    small = {**large, "ram_budget_bytes": 2 * GIB}
    large_mb, large_bytes = _native_working_budget(large)
    small_mb, small_bytes = _native_working_budget(small)
    reserved = 8 * large["n"] + (256 + 3 * large["requested_threads"]) * MIB

    assert large_bytes == 32 * GIB - reserved
    assert small_bytes == 2 * GIB - reserved
    assert large_mb * MIB == large_bytes
    assert small_mb * MIB == small_bytes
    assert large_mb > 2048 > small_mb >= 64
    assert large_bytes - small_bytes == 30 * GIB


def test_billion_sample_input_is_accounted_without_allocating_it():
    working_mb, headroom = _native_working_budget(_case(n=1_000_000_000))
    assert headroom == 32 * GIB - 8_000_000_000 - 280 * MIB
    assert working_mb == headroom / MIB
    assert working_mb > 2048

    working_mb, headroom = _native_working_budget(_case(n=1_000_000_000, ram=2 * GIB))
    assert working_mb is None
    assert headroom < 0


@pytest.mark.parametrize("workers", [1, 8])
@pytest.mark.parametrize("remaining_bytes", [64 * MIB - 1, 0, -1])
def test_less_than_64mib_headroom_is_not_inflated_to_adapter_minimum(workers, remaining_bytes):
    n = 100_000
    ram = 8 * n + (256 + 3 * workers) * MIB + remaining_bytes
    working_mb, headroom = _native_working_budget(_case(n=n, ram=ram, workers=workers))
    assert headroom == remaining_bytes
    assert working_mb is None


def test_exact_64mib_headroom_can_be_checked_by_adapter_preflight():
    n, workers = 100_000, 8
    ram = 8 * n + (256 + 3 * workers + 64) * MIB
    assert _native_working_budget(_case(n=n, ram=ram, workers=workers)) == (64., 64 * MIB)


@pytest.mark.parametrize("working_mb", [1.5, 32., 128., 2048.])
def test_disk_keeps_configured_low_working_budget_and_existing_headroom(working_mb):
    case = _case(n=1_000_000_000, ram=640 * MIB, storage="disk", low=working_mb)
    assert _native_working_budget(case) == (working_mb, 384 * MIB)


def test_disk_also_skips_when_minimum_adapter_capacity_cannot_fit():
    case = _case(storage="disk", ram=(256 + 64) * MIB - 1, low=1.)
    assert _native_working_budget(case) == (None, 64 * MIB - 1)


def test_eight_worker_memory_limited_arm_case_reserves_each_worker():
    # A 1-GiB board can expose a much smaller usable benchmark budget.
    case = _case(n=10_000_000, ram=600 * MIB, workers=8)
    working_mb, headroom = _native_working_budget(case)
    assert working_mb == (600 - 256 - 24) - 80_000_000 / MIB
    assert working_mb >= 64
    assert headroom + 80_000_000 + 280 * MIB == case["ram_budget_bytes"]

    single_mb, single_headroom = _native_working_budget({**case, "requested_threads": 1})
    assert single_mb - working_mb == 21
    assert single_headroom - headroom == 21 * MIB
    assert _native_working_budget({**case, "n": 100_000_000})[0] is None
