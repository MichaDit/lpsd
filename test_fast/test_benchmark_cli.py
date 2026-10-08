"""Small end-to-end checks that benchmark options reach the measured API."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cli():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_lpsd.py"
    spec = importlib.util.spec_from_file_location("lpsd_benchmark_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("options,columns,entry", (
    (["--outputs", "psd"], ["psd"], "lpsd"),
    (["--outputs", "nsd", "psd"], ["nsd", "psd"], "lpsd"),
    (["--entry-point", "lnsd"], ["nsd"], "lnsd"),
))
def test_benchmark_records_selected_output_and_window(cli, tmp_path, options, columns, entry):
    output = tmp_path / "benchmark.json"
    cli.main(["--backend", "fast", "--n", "257", "--n-frequencies", "24",
              "--n-averages", "4", "--repeats", "1", "--warmups", "0",
              "--window", "hamming", "--overlap", "0.5", "--profile",
              "--output", str(output)] + options)
    report = json.loads(output.read_text())
    assert tuple(report["output_dtypes"]) == tuple(columns)
    assert report["parameters"]["window_function"] == "hamming"
    assert report["parameters"]["overlap"] == .5
    assert report["entry_point"] == entry
    assert len(report["repeats"]) == 1
    assert report["additional_profiled_call"]["detail"]["frequencies"]
    summary = report["additional_profiled_call"]["summary"]
    assert summary["sum_worker_stage_seconds"]["window_generation_s"] >= 0
    assert summary["sum_worker_stage_seconds"]["window_sums_s"] >= 0
    assert summary["channel_phase_seconds"]["output_assembly_s"] >= 0
    assert report["median_wall_s"] >= 0


@pytest.mark.parametrize("options", (
    ["--outputs", "psd", "psd"],
    ["--outputs", "all", "psd"],
    ["--window", "hann"],
    ["--overlap", "1"],
    ["--entry-point", "lnsd", "--outputs", "nsd"],
    ["--backend", "original", "--outputs", "psd"],
))
def test_invalid_benchmark_combinations_fail_before_measurement(cli, tmp_path, options):
    with pytest.raises(SystemExit) as error:
        cli.main(["--backend", "fast", "--output", str(tmp_path / "unused.json")] + options)
    assert error.value.code == 2
    assert not (tmp_path / "unused.json").exists()
