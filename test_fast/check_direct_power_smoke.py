"""Small portable direct-power correctness gate; no performance threshold."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve(strict=True)))
    import numpy as np
    from benchmarks._fftw_direct_power import DirectLibrary, reduce_matrix
    from benchmarks import bench_fftw as bf, bench_fftw_disk as disk
    from lpsd_fast import api
    from lpsd_benchmark import SCHEMA, privacy, worker

    api._native()
    library = DirectLibrary(api._LIB)
    assert library.abi_version == 3
    matrix = np.array([[1e-145, 3e-145], [2e-145, 0]], dtype=np.complex128)
    original = matrix.copy()
    cuts = np.array([1, 2, 3], dtype=np.int64)
    small = reduce_matrix(matrix, cuts, 2., 1., library)
    np.testing.assert_allclose(small["sums"], [4e-290, 4.5e-290], rtol=1e-12, atol=0.)
    np.testing.assert_array_equal(matrix, original)
    assert small["metadata"]["energy_bins"] == 3

    cases = []
    for signal in ("white", "dc_nanovolt"):
        spectra = []
        for reduction in ("materialized", "direct-log"):
            case = {"n": 4096, "method": "fftw", "signal": signal, "storage": "disk",
                    "requested_threads": 1, "sample_rate_hz": 50., "seed": 20261010,
                    "n_averages": 100, "base_target_points": 1000, "target_points": 1000,
                    "point_factor": 1., "fft_memory_mb": 128., "low_working_mb": 64.,
                    "ram_budget_bytes": 2 * 1024**3, "disk_budget_bytes": 32 * 1024**2,
                    "disk_fft_algorithm": "full", "disk_power_reduction": reduction,
                    "fft_layout": "blocked", "window_storage": "blocked"}
            with tempfile.TemporaryDirectory(prefix="lpsd-direct-smoke-") as directory:
                result = worker.run(case, Path(directory))
                assert result["status"] == "completed", (signal, reduction, result.get("reason"))
                assert not list(Path(directory).iterdir())
            assert result["input_unchanged"] and result["nsd_matches_sqrt_psd"]
            assert result["finite"] and result["psd_nonnegative"]
            assert result["disk_power_reduction"] == reduction
            if reduction == "direct-log":
                assert result["direct_log_power_abi"] == 3 and result["power_vector_created"] is False
                assert result["direct_log_power_state_bytes"] == 48 * result["actual_points"] + 48
            spectra.append(result.pop("_spectrum"))
            cases.append({**case, **result})
        assert cases[-1]["input_sha256"] == cases[-2]["input_sha256"]
        assert cases[-1]["frequency_sha256"] == cases[-2]["frequency_sha256"]
        assert cases[-1]["native_sha256"] == cases[-2]["native_sha256"]
        for name in ("psd", "nsd"):
            np.testing.assert_allclose(spectra[1][name], spectra[0][name], rtol=3e-6, atol=0.)
        np.testing.assert_array_equal(spectra[1]["frequency_hz"], spectra[0]["frequency_hz"])

    backend = worker._load_fftw(bf, 1)
    options = SimpleNamespace(sample_rate=50., n_frequencies=1000, n_averages=100,
                              fft_memory_mb=128., power_storage="workspace", disk_fft_algorithm="full",
                              disk_power_reduction="direct-log", workers=1, smoothing_workers=1,
                              low_working_mb=64.)
    with tempfile.TemporaryDirectory(prefix="lpsd-direct-zero-") as directory:
        zero, details = disk.disk_pipeline("fftw", np.full(4096, 10.), backend, Path(directory), options)
        np.testing.assert_array_equal(zero.to_numpy(), np.zeros_like(zero.to_numpy()))
        assert details["parseval"]["relative_error"] == 0.
        assert not list(Path(directory).iterdir())
    report = privacy.sanitize({"schema": SCHEMA, "state": "completed", "cases": cases,
                               "settings": {"disk_power_reduction": "direct-log"}})
    payload = {"scope": "portable_direct_power_correctness_only", "abi_version": 3,
               "tiny_nonzero_powers_passed": True, "zero_parseval_guard_passed": True,
               "all_point_rtol": 3e-6, "all_point_atol": 0., "report": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print("Direct ABI3, tiny powers, current full-disk PSD/NSD pairs, ownership and zero-energy guard passed")


if __name__ == "__main__":
    main()
