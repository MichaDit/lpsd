"""Planning/inventory checks; none of these tests asserts GPU execution works."""

import json
from pathlib import Path
import subprocess

import pytest

from benchmarks import probe_accelerators as probe


def test_payload_model_agrees_with_saved_native_profile():
    evidence = json.loads((Path(__file__).resolve().parents[1] /
                           "benchmarks/results_fftw.json").read_text())["headroom"]
    workload = probe.plan_workload(10000000)
    storage = workload["storage"]
    assert storage["frequencies"] == evidence["frequencies"]
    assert storage["sum_lengths"] == evidence["sum_segment_lengths"]
    assert storage["sum_segments"] == evidence["segments"]
    assert storage["sample_iterations"] == evidence["sample_iterations"]
    assert storage["all_frequency_coefficients_fp64_bytes"] == evidence[
        "all_projected_coefficient_bytes_at_16_bytes_per_length"]
    assert workload["native_vs_planner_segment_count_differences"] == 0


def test_streamed_payload_combines_workspaces_of_the_same_frequency():
    # L=100,K=1 is the largest workspace. Adding the separate maximum K=90
    # to maximum L=100 would instead report a configuration that does not exist.
    storage = probe.workload_storage(100, [100, 2], [1, 90])
    assert storage["one_frequency_streamed_payload_peak_bytes"] == 2412
    assert storage["host_prepared_h2d_payload_bytes"] == 2796
    assert storage["host_legacy_reduction_d2h_segment_powers_bytes"] == 728
    assert storage["materialized_segment_samples_fp64_bytes_avoid"] == 2240


def test_int32_start_budget_rejects_lengths_outside_the_native_api():
    with pytest.raises(ValueError):
        probe.workload_storage(2**31, [1], [1])


def test_visible_packages_and_nodes_do_not_imply_a_working_fp64_runtime(monkeypatch):
    monkeypatch.setattr(probe.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(probe.shutil, "which", lambda name: "/mock/" + name)
    monkeypatch.setattr(probe.glob, "glob", lambda pattern: ["/dev/mock-accelerator"])
    monkeypatch.setattr(probe, "pci_accelerators", lambda: [])
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("Inventory must not initialize a runtime."))
    report = probe.inventory()
    assert all(report["python_modules_present_not_imported"].values())
    assert report["runtime_computation"] == report["fp64_computation"] == "not_tested"
    assert "command_probes" not in report


def test_command_timeout_is_reported_without_claiming_no_hardware(monkeypatch):
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(probe.subprocess, "run", timed_out)
    result = probe.command_probe("nvidia-smi", [], .1)
    assert result["status"] == "timed_out"
    assert result["timeout_s"] == .1
    assert "devices" not in result


def test_macos_inventory_records_metal_information_without_inferring_fp64(monkeypatch):
    monkeypatch.setattr(probe.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(probe.shutil, "which", lambda name:
                        "/usr/sbin/system_profiler" if name == "system_profiler" else None)
    monkeypatch.setattr(probe.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(probe, "pci_accelerators", lambda: [])
    output = '{"SPDisplaysDataType": [{"sppci_model": "Apple GPU", "metal": "supported"}]}'
    calls = []

    def completed(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(probe.subprocess, "run", completed)
    result = probe.inventory(probe_commands=True)
    assert calls == [["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"]]
    assert result["command_probes"]["system_profiler"]["stdout"] == output
    assert result["platform"] == "Darwin"
    assert result["fp64_computation"] == "not_tested"


def test_command_failure_and_long_output_are_retained(monkeypatch):
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 1, "x" * 20000, "driver error"))
    result = probe.command_probe("rocminfo", [], 1)
    assert result["status"] == "failed"
    assert result["output_truncated"]
    assert len(result["stdout"]) == 16384
    assert result["stderr"] == "driver error"


def test_inventory_only_never_loads_lpsd_planning(monkeypatch, capsys):
    monkeypatch.setattr(probe, "inventory", lambda *args: {"test": "inventory only"})
    monkeypatch.setattr(probe, "plan_workload", lambda *args:
                        pytest.fail("Inventory-only must not plan or import LPSD."))
    probe.main(["--inventory-only"])
    result = json.loads(capsys.readouterr().out)
    assert result["workloads"] == []
    assert result["inventory"] == {"test": "inventory only"}
