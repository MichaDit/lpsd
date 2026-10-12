"""Evidence gates for the complete-call direct-power comparison harness."""
import json

import numpy as np
import pytest

from benchmarks import bench_disk_algorithms as comparison
from benchmarks import bench_fftw_disk as pipeline


def arguments(tmp_path):
    return ["--compare-direct-power", "--sizes", "4096", "--point-factors", "1",
            "--threads", "1", "--repeats", "1", "--max-seconds", "30",
            "--work-dir", str(tmp_path / "work"), "--output", str(tmp_path / "report.json")]


@pytest.mark.parametrize("extra, message", [
    (["--compare-packed-phase"], "not allowed"),
    (["--methods", "matched"], "fftw only"),
    (["--fft-memory-mib", "64"], "128 MiB"),
    (["--sizes", "1000000002"], "N<=1B"),
    # N=1002 is even, but the original full FFT has factors [6, 167].
    (["--sizes", "1002"], "even B"),
])
def test_direct_comparison_rejects_unqualified_request_before_files(tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as error:
        comparison.main(arguments(tmp_path) + extra)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("relative_change, accepted", [(1e-6, True), (1e-5, False)])
def test_direct_comparison_gates_actual_float32_outputs_at_3ppm(tmp_path, monkeypatch,
                                                               relative_change, accepted):
    original = pipeline.disk_pipeline

    def altered(method, values, backend, directory, options):
        frame, details = original(method, values, backend, directory, options)
        if options.disk_power_reduction == "direct-log":
            frame = frame.copy()
            frame["psd"] *= np.float32(1. + relative_change)
            frame["nsd"] = np.sqrt(frame["psd"].to_numpy().astype(np.complex64)).real
        return frame, details

    monkeypatch.setattr(pipeline, "disk_pipeline", altered)
    assert comparison.main(arguments(tmp_path)) == (0 if accepted else 1)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["schema"] == "lpsd-direct-power-pairs-v1"
    assert report["status"] == ("completed" if accepted else "failed")
    assert report["settings"]["methods"] == ["fftw"]
    assert report["settings"]["fft_memory_mib"] == 128.
    assert report["direct_log_power_abi"] == 3
    for source in ("benchmarks/_fftw_direct_power.py", "lpsd_fast/_native/fftw_direct_log_power.c",
                   "lpsd_fast/_native/fast_dft.c"):
        assert len(report["source_sha256"][source]) == 64
    pair = report["cases"][0]["pairs"][0]
    assert pair["identical_frequency_grid"] and pair["passed"] is accepted
    assert all(item["passed"] for item in pair["comparison"].values())  # Both changes are below 1%.
    assert pair["comparison"]["psd"]["strict_3ppm_passed"] is accepted
    assert not pair["comparison"]["psd"]["bit_identical"]
    for variant, sample in pair["samples"].items():
        assert sample["fft_n"] == 4096 and sample["route_verified"]
        assert sample["disk_power_reduction"] == variant
        assert sample["output_dtypes"] == {"psd": "float32", "nsd": "float32"}
        assert sample["nsd_matches_sqrt_psd"]
    assert not list((tmp_path / "work").iterdir())


@pytest.mark.parametrize("corruption", ["abi", "half_n", "power_vector"])
def test_direct_comparison_rejects_wrong_observed_route(tmp_path, monkeypatch, corruption):
    original = pipeline.disk_pipeline

    def altered(method, values, backend, directory, options):
        frame, details = original(method, values, backend, directory, options)
        if options.disk_power_reduction == "direct-log":
            if corruption == "abi":
                details["direct_group_reduction"]["abi_version"] = 2
            elif corruption == "half_n":
                details["fft"]["n"] = len(values) // 2
            else:
                details["power_vector_created"] = True
        return frame, details

    monkeypatch.setattr(pipeline, "disk_pipeline", altered)
    with pytest.raises(AssertionError, match="Direct power comparison did not execute"):
        comparison.main(arguments(tmp_path))
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "AssertionError"
    assert not list((tmp_path / "work").iterdir())
