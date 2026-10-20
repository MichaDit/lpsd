"""Forecast regressions tied to the retained physical fast.10 campaign."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lpsd_benchmark import resources


REPORTS = (Path(__file__).resolve().parents[1] / "benchmarks" / "results"
           / "device-tests-fast10-20261013")


def _calibration(**changes):
    return {"method": "fftw", "signal": "white", "n": 100_000,
            "target_points": 1000, "point_factor": 1., "storage": "ram",
            "status": "completed", "process_wall_s": 20.,
            "compute_wall_s": 1., **changes}


def _target(**changes):
    return {**_calibration(), "n": 1_000_000, **changes}


def _load(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def test_startup_dominated_calibration_no_longer_consumes_half_hour_forecast():
    row = _calibration()
    forecast = resources.predict_seconds(_target(), [row])
    old_forecast = resources.predict_seconds(_target(), [{k: v for k, v in row.items()
                                                        if k != "compute_wall_s"}])
    # Retain substantial allowance for noncompute work: 20 s is NOT treated
    # as a fixed startup cost. The complete process is still forecast >800 s.
    assert 800. < forecast < 900. < 1700. < old_forecast


@pytest.mark.parametrize("compute", [None, -1., 20.01, float("nan"),
                                    float("inf"), True, "1.0", []])
def test_unusable_compute_measurement_keeps_complete_process_fallback(compute):
    missing = _calibration()
    missing.pop("compute_wall_s")
    expected = resources.predict_seconds(_target(), [missing])
    assert resources.predict_seconds(_target(), [_calibration(compute_wall_s=compute)]) == expected


@pytest.mark.parametrize("process", [None, 0., -1., float("nan"),
                                    float("inf"), True, "20.0", []])
def test_unusable_process_measurement_cannot_admit_an_uncalibrated_large_case(process):
    row = _calibration(process_wall_s=process)
    assert resources.predict_seconds(_target(), [row]) is None
    assert resources.predict_seconds(_target(n=100_000), [row]) == 180.


def test_missing_process_measurement_is_not_a_calibration():
    row = _calibration()
    row.pop("process_wall_s")
    assert resources.predict_seconds(_target(), [row]) is None


@pytest.mark.parametrize("n", [4096, 100_000])
@pytest.mark.parametrize("point_factor", [.5, 1., 2.])
def test_equal_and_smaller_sizes_preserve_exact_previous_forecast(n, point_factor):
    row = _calibration(process_wall_s=19.676637017997564,
                       compute_wall_s=.20579506300055073)
    without_compute = {k: v for k, v in row.items() if k != "compute_wall_s"}
    target = _target(n=n, point_factor=point_factor)
    assert resources.predict_seconds(target, [row]) == resources.predict_seconds(target, [without_compute])


def test_compute_only_observation_and_existing_multipliers_remain_conservative():
    row = _calibration(compute_wall_s=20.)
    without_compute = {k: v for k, v in row.items() if k != "compute_wall_s"}
    target = _target()
    assert resources.predict_seconds(target, [row]) == resources.predict_seconds(target, [without_compute])
    prediction = resources.predict_seconds(target, [_calibration()])
    assert resources.predict_seconds({**target, "storage": "disk"}, [_calibration()]) == prediction * 8
    assert resources.predict_seconds({**target, "point_factor": 2.}, [_calibration()]) == pytest.approx(
        prediction * 2 ** 1.2)
    assert resources.predict_seconds(target, [_calibration()], safety_factor=8.) == prediction * 2


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_retained_instrument_skips_receive_forecasts_within_initial_remaining_budget(workers):
    report = _load(f"arm64-instrument-workers-{workers}.json")
    calibrations = [row for row in report["cases"] if row["case_role"] == "calibration"]
    remaining = report["settings"]["max_seconds"] - sum(row["process_wall_s"] for row in calibrations)
    skipped = [row for row in report["cases"] if row["status"] == "skipped"]
    assert len(skipped) == {1: 2, 2: 3, 4: 2}[workers]
    for row in skipped:
        assert row["reason"] == "time_budget"
        prediction = resources.predict_seconds(row, calibrations)
        assert 790. < prediction < 1100. < remaining - 20.
        assert prediction < row["predicted_process_s"]
        # A new forecast does not create a timing for an unexecuted row.
        assert "compute_wall_s" not in row and "process_wall_s" not in row


def test_forecast_replay_retains_margin_for_every_completed_physical_observation():
    manifest = _load("manifest.json")
    compared = 0
    minimum_margin = float("inf")
    for entry in manifest["reports"]:
        report = _load(entry["file"])
        previous = []
        for row in report["cases"]:
            if (row["status"] == "completed"
                    and any(resources.calibration_matches(row, old) for old in previous)):
                prediction = resources.predict_seconds(row, previous)
                observed = row["process_wall_s"]
                assert prediction >= observed, (entry["file"], row["n"], row["method"],
                                                 row["signal"], row["point_factor"])
                minimum_margin = min(minimum_margin, prediction / observed)
                compared += 1
            previous.append(row)
    assert len(manifest["reports"]) == 20 and compared == 394
    assert minimum_margin > 2.4
