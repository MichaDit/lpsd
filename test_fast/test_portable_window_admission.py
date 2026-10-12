"""Scalar RAM admission and public metadata gates; no giant allocations."""
from __future__ import annotations

import json

import pytest

from lpsd_benchmark import SCHEMA, privacy, resources, runner


def mode(n, method, budget, layout="auto", window="auto"):
    chosen = resources.choose_fft_layout(n, method, "white", layout, budget, window)
    storage = resources.choose_window_storage(n, method, "white", window, chosen, budget)
    return chosen, storage


@pytest.mark.parametrize("method", ("fftw", "matched"))
def test_auto_keeps_full_window_when_its_existing_layout_fits(method):
    n = 1_000_000
    full_out = resources.required_ram(n, method)
    full_in = resources.required_ram(n, method, fft_layout="in-place")
    assert mode(n, method, full_out) == ("out-of-place", "full")
    assert mode(n, method, full_out - 1) == ("in-place", "full")
    assert mode(n, method, full_in) == ("in-place", "full")


@pytest.mark.parametrize("method", ("fftw", "matched"))
def test_one_billion_fits_32_decimal_gb_machine_after_os_reserve(method):
    # A scalar model, not a 32-GB measurement. Physical memory less the
    # existing fixed 2-GiB controller/OS reserve, assuming memory is available.
    n = 1_000_000_000
    budget = 32_000_000_000 - 2 * resources.GIB
    layout, window = mode(n, method, budget)
    assert (layout, window) == ("in-place", "fft_input_then_blocked")
    detected = resources.Resources(32_000_000_000, 32_000_000_000, 1_000_000_000,
                                    budget, 512 * resources.MIB, 8, 8.)
    selected = resources.select_storage(n, method, "white", "auto", detected, layout, window)
    assert selected[0] == "ram" and selected[3] is None
    assert selected[1] <= budget
    assert resources.fftw_plan_reserve(n, method) == 8_000_000_000


def test_explicit_modes_do_not_silently_change_and_low_ram_still_skips():
    n = 1_000_000_000
    budget = 26 * resources.GIB
    assert mode(n, "fftw", budget, "out-of-place", "full") == ("out-of-place", "full")
    assert mode(n, "fftw", budget, "out-of-place", "auto") == ("out-of-place", "fft_input_then_blocked")
    assert mode(100_000, "fftw", resources.GIB, window="fft_input_then_blocked") == ("out-of-place", "fft_input_then_blocked")
    assert mode(n, "lpsd", budget) == ("not_applicable", "not_applicable")
    detected = resources.Resources(512 * resources.MIB, 400 * resources.MIB,
                                    2 * resources.GIB, 300 * resources.MIB,
                                    resources.GIB, 2, 2.)
    layout, window = mode(n, "matched", detected.ram_budget)
    selected = resources.select_storage(n, "matched", "white", "ram", detected, layout, window)
    assert selected[3] == "memory_budget"


def test_colored_signal_generation_does_not_shrink_with_window_mode():
    n = 10_000_000
    a = resources.required_ram(n, "matched", "pink", "in-place", "full")
    b = resources.required_ram(n, "matched", "pink", "in-place", "fft_input_then_blocked")
    assert a == b == 72 * n + 256 * resources.MIB


def test_window_metadata_round_trips_through_closed_public_schema():
    report = {"schema": SCHEMA, "settings": {"window_storage": "auto"},
              "cases": [{"n": 100_000, "method": "matched", "signal": "white",
                         "target_points": 1000, "window_storage": "fft_input_then_blocked",
                         "fftw_plan_reserve_bytes": 800_000, "window_buffer_bytes": 0,
                         "window_scratch_bytes": 800_000, "window_block_samples": 100_000,
                         "one_shot_window": True, "window_storage_details": {"path": "/home/private"}}]}
    clean = privacy.sanitize(report)
    row = clean["cases"][0]
    assert row["window_storage"] == "fft_input_then_blocked"
    assert row["window_buffer_bytes"] == 0 and row["one_shot_window"] is True
    assert row["fftw_plan_reserve_bytes"] == 800_000
    assert "private" not in json.dumps(clean)
    assert privacy.sanitize(clean) == clean
    report["settings"]["window_storage"] = "/home/private"
    report["cases"][0]["window_storage"] = "owner@example.com"
    serialized = json.dumps(privacy.sanitize(report))
    assert "private" not in serialized and "owner" not in serialized


def test_parser_exposes_reproducible_window_selection():
    assert runner.parser().parse_args(["run"]).window_storage == "auto"
    parsed = runner.parser().parse_args(["run", "--window-storage", "fft_input_then_blocked"])
    assert parsed.window_storage == "fft_input_then_blocked"
