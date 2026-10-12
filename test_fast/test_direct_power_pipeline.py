"""Bounded current-owner full-disk integration gates; no performance claims."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bench_fftw as bf
from benchmarks import bench_fftw_disk as disk
from benchmarks import _fftw_direct_power as direct
from benchmarks._fftw_disk import DiskFFT
from benchmarks.fftw_comparison_signals import signal


@pytest.fixture(scope="module")
def fftw():
    try:
        return bf.FFTWLibrary(os.environ.get("LPSD_TEST_FFTW_LIBRARY"))
    except OSError as error:
        pytest.skip(f"Optional FFTW shared library unavailable: {error}")


def _options(reduction="direct-log", **changes):
    values = dict(sample_rate=50., n_frequencies=1000, n_averages=100,
                  workers=1, smoothing_workers=1, low_working_mb=8.,
                  fft_memory_mb=128., power_storage="workspace",
                  disk_fft_algorithm="full", disk_power_reduction=reduction)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("kind", ["white", "dc_nanovolt", "offbin_tone", "zero", "constant", "tiny"])
def test_direct_matches_current_materialized_pipeline(fftw, tmp_path, monkeypatch, kind):
    n = 4096
    if kind in ("zero", "constant"):
        values = np.full(n, 0. if kind == "zero" else 10., dtype=np.float64)
    elif kind == "tiny":
        values = np.random.default_rng(813).standard_normal(n) * 1e-145
    else:
        values, _ = signal(kind, n, 50., 20261010)
    before = values.copy()
    expected, original = disk.disk_pipeline("fftw", values, fftw, tmp_path, _options("materialized"))

    def forbidden(*args, **kwargs):
        pytest.fail("Direct groups must never materialize an N-sized power vector")

    monkeypatch.setattr(DiskFFT, "power_into", forbidden)
    monkeypatch.setattr(DiskFFT, "power_in_workspace", forbidden)
    actual, details = disk.disk_pipeline("fftw", values, fftw, tmp_path, _options())
    np.testing.assert_array_equal(actual.index.to_numpy(), expected.index.to_numpy())
    for name in ("psd", "nsd"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=3e-6, atol=0.)
    np.testing.assert_array_equal(actual.nsd, np.sqrt(actual.psd.to_numpy().astype(np.complex64)).real)
    np.testing.assert_array_equal(values, before)
    assert original["disk_power_reduction"] == "materialized" and original["power_vector_created"]
    assert details["disk_power_reduction"] == "direct-log" and details["power_vector_created"] is False
    source = details["direct_group_reduction"]
    assert source["groups"] == len(actual) == details["point_plan"]["output_points"]
    assert source["abi_version"] == 3 and source["energy_bins"] == n // 2 + 1
    assert source["dc_bins"] == source["nyquist_bins"] == 1
    assert source["matrix_written"] is False and source["additional_workers"] == 0
    assert source["array_state_bytes"] == 48 * len(actual) + 48
    assert details["power_storage"]["maximum_large_file_bytes"] == 24 * n
    assert details["power_storage"]["extra_power_file_bytes"] == 0
    assert details["power_storage"]["workspace_power_offset_bytes"] is None
    assert "direct_log_groups_and_energy_s" in details["phases"]
    assert "ordered_one_sided_power_s" not in details["phases"]
    assert details["parseval"]["relative_error"] <= 1e-10
    if kind in ("zero", "constant"):
        assert details["parseval"]["relative_error"] == 0.
        assert details["parseval"]["time_power_v2"] == details["parseval"]["fft_power_v2"] == 0.
        np.testing.assert_array_equal(actual.to_numpy(), np.zeros_like(actual.to_numpy()))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changes", [
    {"sample_rate": 51.}, {"n_averages": 99}, {"psll": 199.},
    {"fft_memory_mb": 127.}, {"power_storage": "external"},
    {"disk_fft_algorithm": "packed-real"},
])
def test_unqualified_settings_rejected_before_workspace(tmp_path, monkeypatch, changes):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported direct profile must not allocate a workspace")

    monkeypatch.setattr(disk.tempfile, "TemporaryDirectory", forbidden)
    with pytest.raises(ValueError, match="profile"):
        disk.disk_pipeline("fftw", np.zeros(16), None, tmp_path, _options(**changes))
    assert not list(tmp_path.iterdir())


def test_direct_retains_original_even_b_workspace_admission(tmp_path):
    with pytest.raises(ValueError, match="even B"):
        disk.disk_pipeline("fftw", np.zeros(1009), None, tmp_path, _options())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["abi", "planning", "reduction", "cleanup"])
def test_direct_keeps_caller_files_and_nested_cleanup(fftw, tmp_path, monkeypatch, failure):
    sentinels = {tmp_path / "workspace.c128": b"caller workspace",
                 tmp_path / "powers.f64": b"caller power"}
    for path, content in sentinels.items():
        path.write_bytes(content)

    def fail(*args, **kwargs):
        raise RuntimeError("controlled integration failure")

    if failure == "abi":
        monkeypatch.setattr(direct, "DirectLibrary", fail)
    elif failure == "planning":
        monkeypatch.setattr(fftw.lib, "fftw_plan_many_dft", lambda *args: None)
    elif failure == "reduction":
        monkeypatch.setattr(direct, "reduce_matrix", fail)
    else:
        close = DiskFFT.close

        def close_then_fail(self):
            close(self)
            raise RuntimeError("controlled integration failure")

        monkeypatch.setattr(DiskFFT, "close", close_then_fail)
    with pytest.raises(MemoryError if failure == "planning" else RuntimeError):
        disk.disk_pipeline("fftw", np.arange(1000, dtype=np.float64), fftw, tmp_path, _options())
    assert {path: path.read_bytes() for path in sentinels} == sentinels
    assert not list(tmp_path.glob("fft-work-*"))


@pytest.mark.parametrize("method,algorithm", [("fftw", "packed-real"), ("matched", "full"),
                                               ("matched", "packed-real"), ("lpsd", "full")])
def test_other_routes_never_bind_direct_reducer(fftw, tmp_path, monkeypatch, method, algorithm):
    def forbidden(*args, **kwargs):
        pytest.fail("This estimator route must not bind or invoke the direct reducer")

    monkeypatch.setattr(direct, "DirectLibrary", forbidden)
    monkeypatch.setattr(DiskFFT, "group_power", forbidden)
    # The option is not applicable to other methods. Packed simple FFTW
    # explicitly retains its materialized route.
    selection = "materialized" if method == "fftw" else "direct-log"
    frame, details = disk.disk_pipeline(method, np.zeros(1000), fftw, tmp_path,
                                        _options(selection, disk_fft_algorithm=algorithm, fft_memory_mb=.1))
    assert len(frame) > 0
    assert details["disk_power_reduction"] == ("materialized" if method == "fftw" else "not_applicable")
    assert "direct_group_reduction" not in details
    assert not list(tmp_path.iterdir())
