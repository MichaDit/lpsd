"""CI gate for the real private bootstrap; no timing threshold or upload."""
import json
from pathlib import Path
import sys


def check(path, expected_window=None, expected_smoothing_request=None,
          expected_statistics="corrected"):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [case for case in report["cases"] if case["n"] == 100000
            and case["signal"] == "white" and case["case_role"] == "comparison"]
    expected = {(method, points, storage) for method in ("fftw", "matched", "lpsd")
                for points in (469, 938) for storage in ("ram", "disk")}
    actual = {(case["method"], case["actual_points"], case["storage"])
              for case in rows if case["status"] == "completed"}
    assert actual == expected, (actual, [(case["method"], case["status"],
                                         case.get("reason")) for case in rows])
    assert report["state"] == "completed", report["state"]
    assert len(rows) == len(expected)
    assert report["settings"]["requested_statistics"] == expected_statistics
    assert all(case["input_unchanged"] and case["nsd_matches_sqrt_psd"]
               and case["finite"] and case["psd_nonnegative"]
               and case["frequency_matches_plan"] for case in rows)
    assert report["settings"].get("requested_smoothing_workers") == expected_smoothing_request
    for case in rows:
        assert case["requested_statistics"] == expected_statistics
        assert case["statistics"] == ("not_applicable" if case["method"] == "fftw"
                                      else expected_statistics)
        validation = case["validation"]
        assert validation["expected_frequency_points"] == case["actual_points"]
        assert validation["nsd_sqrt_mismatch_points"] == validation["frequency_mismatch_points"] == 0
        guard = case["memory_guard"]
        assert guard["scope"] == "process_tree" and guard["measurement"] == "sampled"
        assert guard["termination"] == "none" and guard["limit_bytes"] > 0
        assert case["live_memory_observed"] and guard["peak_bytes"] > 0
        if case["method"] != "lpsd":
            threading = case["fftw_threading"]
            assert threading["requested"] == case["requested_threads"]
            assert threading["actual"] == case["actual_threads"]
            assert 1 <= threading["actual"] <= threading["requested"]
            assert threading["status"] == ("enabled" if threading["actual"] > 1 else
                                            "single_thread_requested" if threading["requested"] == 1 else
                                            "fallback_single_thread")
            assert case["comparison"]["reference_statistics"] == expected_statistics
        if case["method"] != "matched":
            assert case["smoothing_worker_policy"] == "not_applicable"
            assert case["smoothing_workers_limit"] == case["smoothing_runtime_reserve_bytes"] == 0
            continue
        admitted = case["requested_threads"]
        baseline = min(4, admitted)
        limit = min(baseline if expected_smoothing_request is None else expected_smoothing_request, admitted)
        assert case.get("requested_smoothing_workers") == expected_smoothing_request
        assert case["smoothing_worker_policy"] == ("default_cap4" if expected_smoothing_request is None else "explicit")
        assert case["smoothing_workers_limit"] == limit
        assert case["smoothing_runtime_reserve_bytes"] == 9 * 1024**2 * max(0, limit - baseline)
        assert case["smoothing_runtime_reserve_model"] == "incremental_9mib_above_cap4_v1"
        assert case["smoothing_backend"] == "native"
        assert 0 <= case["smoothing_pool_capacity"] <= limit
        assert 0 < case["smoothing_task_ranges"] <= min(case["smoothed_frequency_points"], 2 * limit)
    if expected_window is not None:
        fft_rows = [case for case in rows if case["method"] != "lpsd"
                    and case["storage"] == "ram"]
        disk_rows = [case for case in rows if case["method"] != "lpsd"
                     and case["storage"] == "disk"]
        assert all(case["window_storage"] == expected_window for case in fft_rows)
        assert all(case["window_storage"] == "blocked" for case in disk_rows)
        if expected_window == "fft_input_then_blocked":
            assert all(case["one_shot_window"] and case["window_buffer_bytes"] == 0
                       and case["window_scratch_bytes"] > 0 for case in fft_rows)
    print("All 12 method/point/storage cases passed with verified statistics, outputs, memory, threading and smoothing controls")


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
          int(sys.argv[3]) if len(sys.argv) > 3 else None,
          sys.argv[4] if len(sys.argv) > 4 else "corrected")
