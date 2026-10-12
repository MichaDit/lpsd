"""Algorithm admission, source provenance and actual original-N readback."""
import json

import pytest

from lpsd_benchmark import SCHEMA, privacy, resources, runner, worker
from test_fast.test_native_working_controls import _case


def test_cli_algorithm_default_and_explicit_choice():
    assert runner.parser().parse_args(["run"]).disk_fft_algorithm == "full"
    assert runner.parser().parse_args(["run", "--disk-fft-algorithm", "packed-real"]).disk_fft_algorithm == "packed-real"
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["run", "--disk-fft-algorithm", "hostname.invalid"])


@pytest.mark.parametrize("bad", [True, 1, None, [], {}, "filename", "not_applicable"])
def test_invalid_disk_algorithm_is_rejected_before_work(tmp_path, bad):
    case = {**_case(method="fftw"), "disk_fft_algorithm": bad}
    with pytest.raises(ValueError, match="algorithm"):
        worker.run(case, tmp_path)
    assert not list(tmp_path.iterdir())


def test_forecast_routes_are_separate():
    case = {**_case(method="matched"), "disk_fft_algorithm": "packed-real",
            "native_working_policy": "auto", "smoothing_workers_limit": 2}
    completed = {**case, "status": "completed", "process_wall_s": 1.}
    assert resources.calibration_matches(case, completed)
    assert not resources.calibration_matches(case, {**completed, "disk_fft_algorithm": "full"})
    assert not resources.calibration_matches(case, {**completed, "storage": "ram"})
    larger = {**case, "n": 1_000_000}
    assert resources.predict_seconds(larger, [{**completed, "disk_fft_algorithm": "full"}]) is None
    # The established full RAM-to-disk scaling policy remains available.
    assert resources.calibration_matches({**case, "disk_fft_algorithm": "full"},
                                        {**completed, "storage": "ram"})


@pytest.mark.parametrize("n,inner,effective", [(4096, 2048, "packed-real"), (4095, 4095, "full")])
def test_actual_transform_readback(n, inner, effective):
    case = {**_case(method="fftw", n=n), "disk_fft_algorithm": "packed-real"}
    assert worker._disk_fft_result_fields(case, {"fft": {"n": n, "inner_n": inner}}) == {
        "disk_fft_inner_samples": inner, "disk_fft_effective_algorithm": effective}
    for broken in ({}, {"fft": {"n": n, "inner_n": True}}, {"fft": {"n": n + 1, "inner_n": inner}}):
        with pytest.raises(ValueError, match="metadata"):
            worker._disk_fft_result_fields(case, broken)


@pytest.mark.parametrize("method", ["fftw", "matched"])
def test_worker_executes_packed_route_with_input_and_output_provenance(tmp_path, method):
    from benchmarks import bench_fftw as bf
    try:
        worker._load_fftw(bf, 1)
    except ImportError as error:
        pytest.skip(str(error))
    reports = []
    for algorithm in ("full", "packed-real"):
        directory = tmp_path / algorithm
        directory.mkdir()
        case = {**_case(method=method, n=4096, threads=1), "disk_fft_algorithm": algorithm}
        result = worker.run(case, directory)
        assert result["status"] == "completed"
        assert result["disk_fft_algorithm"] == algorithm
        assert result["disk_fft_effective_algorithm"] == algorithm
        assert result["disk_fft_inner_samples"] == (4096 if algorithm == "full" else 2048)
        reports.append(result)
    assert reports[0]["input_sha256"] == reports[1]["input_sha256"]
    assert reports[0]["frequency_sha256"] == reports[1]["frequency_sha256"]
    assert reports[0]["source_sha256"] != reports[1]["source_sha256"]


def test_public_algorithm_allowlist_roundtrip_and_rejection():
    row = {**_case(method="fftw"), "status": "completed", "disk_fft_algorithm": "packed-real",
           "disk_fft_effective_algorithm": "packed-real", "disk_fft_inner_samples": 50_000}
    report = {"schema": SCHEMA, "settings": {"disk_fft_algorithm": "packed-real"}, "cases": [row]}
    clean = privacy.sanitize(report)
    assert clean["settings"]["disk_fft_algorithm"] == "packed-real"
    assert clean["cases"][0]["disk_fft_inner_samples"] == 50_000
    assert privacy.sanitize(clean) == clean
    row.update(disk_fft_algorithm="private-host.invalid", disk_fft_effective_algorithm=["secret"],
               disk_fft_inner_samples=True)
    clean = privacy.sanitize(report)
    assert "private-host" not in json.dumps(clean)
    assert "disk_fft_effective_algorithm" not in clean["cases"][0]
    assert "disk_fft_inner_samples" not in clean["cases"][0]
