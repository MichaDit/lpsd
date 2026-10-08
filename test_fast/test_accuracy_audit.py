"""Verify audit accounting, especially zeros and tiny reference values."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def audit():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "check_accuracy.py"
    spec = importlib.util.spec_from_file_location("lpsd_accuracy_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_references_are_not_hidden_by_a_floor(audit):
    metrics = audit.error_metrics([0., 1., 100., 1e-30],
                                  [1e-20, 1.02, 100.5, 2e-30], [1., 2., 3., 4.])
    assert metrics["max_absolute_error"] == .5
    assert metrics["max_relative_error_nonzero_reference"] == 1.
    assert metrics["max_relative_error_at"]["frequency"] == 4.
    assert metrics["count_above_relative_limit"] == 2
    assert metrics["reference_zero_candidate_nonzero_count"] == 1
    assert metrics["worst_changed_zero_points"][0]["relative_error"] is None
    assert not metrics["strict_limit_satisfied"]
    assert not metrics["combined_limit_satisfied"]
    assert metrics["absolute_limit"] == 0
    assert metrics["count_accepted_by_absolute_limit_only"] == 0
    json.dumps(metrics, allow_nan=False)


def test_relative_limit_is_strict(audit):
    metrics = audit.error_metrics([2.], [3.], [1.], relative_limit=.5)
    assert metrics["count_above_relative_limit"] == 0
    assert metrics["count_at_or_above_relative_limit"] == 1
    assert not metrics["strict_limit_satisfied"]
    assert not metrics["combined_limit_satisfied"]
    allowed = audit.error_metrics([2.], [3.], [1.], relative_limit=.5, absolute_limit=1.)
    assert not allowed["strict_limit_satisfied"]
    assert allowed["combined_limit_satisfied"]  # The absolute boundary is inclusive.
    assert allowed["count_accepted_by_absolute_limit_only"] == 1


def test_explicit_absolute_allowance_keeps_all_raw_outliers_visible(audit):
    args = ([0., 1e-30, 100., 0.], [1e-24, 2e-30, 101., 0.], [1., 2., 3., 4.])
    strict = audit.error_metrics(*args)
    allowed = audit.error_metrics(*args, absolute_limit=1e-24)
    for key in ("max_absolute_error", "max_relative_error_nonzero_reference",
                "count_at_or_above_relative_limit", "reference_zero_candidate_nonzero_count",
                "worst_relative_points", "worst_changed_zero_points", "strict_limit_satisfied"):
        assert allowed[key] == strict[key]
    assert allowed["count_accepted_by_absolute_limit_only"] == 2
    assert allowed["count_not_meeting_combined_limit"] == 1
    assert not allowed["combined_limit_satisfied"]
    assert [point["index"] for point in allowed["worst_absolute_only_points"]] == [0, 1]
    assert allowed["worst_combined_failure_points"][0]["index"] == 2
    assert allowed["max_relative_error_nonzero_reference"] == 1.
    json.dumps(allowed, allow_nan=False)


def test_combined_criterion_uses_or_instead_of_adding_tolerances(audit):
    metrics = audit.error_metrics([100.], [101.5], [1.], absolute_limit=1.)
    assert not metrics["combined_limit_satisfied"]
    assert metrics["count_not_meeting_combined_limit"] == 1


def test_identical_zero_and_nonfinite_outputs_are_reported_separately(audit):
    zeros = audit.error_metrics([0., 0.], [0., 0.], [1., 2.])
    assert zeros["max_relative_error_nonzero_reference"] is None
    assert zeros["strict_limit_satisfied"]
    assert zeros["combined_limit_satisfied"]
    assert zeros["count_accepted_by_absolute_limit_only"] == 0
    invalid = audit.error_metrics([0., 1., np.nan], [np.inf, 1., 2.], [1., 2., 3.])
    assert invalid["reference_nonfinite_count"] == 1
    assert invalid["candidate_nonfinite_count"] == 1
    assert invalid["reference_zero_candidate_nonfinite_count"] == 1
    assert not invalid["strict_limit_satisfied"]
    assert not invalid["combined_limit_satisfied"]
    json.dumps(invalid, allow_nan=False)


def test_absolute_allowance_never_accepts_nonfinite_values(audit):
    metrics = audit.error_metrics([0., np.nan, np.inf], [1e-30, np.nan, np.inf],
                                  [1., 2., 3.], absolute_limit=1e-24)
    assert metrics["count_accepted_by_absolute_limit_only"] == 1
    assert metrics["count_not_meeting_combined_limit"] == 2
    assert not metrics["combined_limit_satisfied"]
    json.dumps(metrics, allow_nan=False)


def test_frequency_mismatch_prevents_misaligned_comparison(audit):
    reference = pd.DataFrame({"psd": [1.], "asd": [1.]}, index=[1.])
    candidate = pd.DataFrame({"psd": [1.], "nsd": [1.]}, index=[1.0000000001])
    result = audit.compare_outputs(reference, candidate)
    assert result["status"] == "frequency_mismatch"
    assert "metrics" not in result


def test_psd_and_nsd_absolute_allowances_are_independent(audit):
    reference = pd.DataFrame({"psd": [0.], "asd": [0.]}, index=[1.])
    candidate = pd.DataFrame({"psd": [1e-26], "nsd": [1e-13]}, index=[1.])
    psd_only = audit.compare_outputs(reference, candidate, psd_absolute_limit=1e-24)
    assert psd_only["metrics"]["psd"]["combined_limit_satisfied"]
    assert not psd_only["metrics"]["nsd"]["combined_limit_satisfied"]
    both = audit.compare_outputs(reference, candidate, psd_absolute_limit=1e-24,
                                  nsd_absolute_limit=1e-12)
    assert all(metric["combined_limit_satisfied"] for metric in both["metrics"].values())
    assert not any(metric["strict_limit_satisfied"] for metric in both["metrics"].values())


@pytest.mark.parametrize("option", ("--psd-absolute-limit", "--nsd-absolute-limit"))
@pytest.mark.parametrize("value", ("-1", "nan", "inf"))
def test_cli_rejects_invalid_absolute_allowances_before_native_calls(audit, tmp_path, option, value):
    with pytest.raises(SystemExit) as error:
        audit.main([option, value, "--output", str(tmp_path / "unused.json")])
    assert error.value.code == 2


@pytest.mark.parametrize("allowances, expected_status", (
    ([], 1),
    (["--psd-absolute-limit", "1e-24"], 1),
    (["--psd-absolute-limit", "1e-24", "--nsd-absolute-limit", "1e-12"], 0),
))
def test_cli_exit_gate_reports_relative_failures_when_absolute_limits_pass(
        audit, tmp_path, monkeypatch, allowances, expected_status):
    def known_outputs(data, *, kernel, **kwargs):
        columns = {"psd": [0.], "asd": [0.]} if kernel == "scalar" else {"psd": [1e-26], "nsd": [1e-13]}
        result = pd.DataFrame(columns, index=[1.], dtype=np.float32)
        result.attrs["lpsd_fast"] = {"native_mode": 0 if kernel == "scalar" else 2}
        return result

    monkeypatch.setattr(audit, "source_evidence", lambda: {})
    monkeypatch.setattr(audit, "scalar_anchor", lambda: {})
    monkeypatch.setattr(audit.lpsd_fast, "lpsd", known_outputs)
    output = tmp_path / "explicit-allowances.json"
    status = audit.main(["--n", "35", "--cases", "white_kaiser200", "--fail-on-limit",
                         "--output", str(output), *allowances])
    report = json.loads(output.read_text())
    assert status == expected_status
    assert report["summary"]["cases_not_meeting_strict_limit"] == ["white_kaiser200"]
    assert not report["summary"]["all_cases_meet_strict_limit"]
    assert report["summary"]["all_cases_meet_combined_limit"] == (expected_status == 0)
    assert report["absolute_limits"]["psd"] == (1e-24 if allowances else 0.)


def test_small_audit_records_both_densities_and_source_evidence(audit, tmp_path):
    output = tmp_path / "accuracy.json"
    status = audit.main(["--n", "257", "--n-frequencies", "24", "--n-averages", "4",
                         "--cases", "white_kaiser200", "offbin_tone_kaiser200",
                         "--worst-points", "2", "--output", str(output)])
    report = json.loads(output.read_text())
    assert status == 0  # Measurement mode records outliers instead of hiding them.
    assert report["summary"]["cases"] == 2
    assert report["absolute_limits"] == {"psd": 0., "nsd": 0.}
    assert report["summary"]["cases_not_meeting_combined_limit"] == report["summary"]["cases_not_meeting_strict_limit"]
    assert report["source_evidence"]["upstream_sources_match_pinned"]
    assert report["scalar_anchor"]["all_seven_columns_and_frequencies_bitwise_equal"]
    assert report["reference"] == {"kernel": "scalar", "workers": 1,
                                    "outputs": "all", "nsd_reference_column": "asd"}
    for case in report["cases"]:
        assert case["frequency_index_exact"]
        assert tuple(case["metrics"]) == ("psd", "nsd")
        assert len(case["metrics"]["psd"]["worst_relative_points"]) <= 2
        assert case["metrics"]["nsd"]["reference_dtype"] == "float32"


def test_case_catalog_includes_high_dynamic_range_flat_top(audit):
    assert len(audit.CASES) == 19
    assert audit.CASES["offbin_tone_hft248d"] == ("offbin_tone", "hft248d")
    assert audit.CASES["onbin_tone_hft248d"] == ("onbin_tone", "hft248d")
    assert audit.WINDOWS["hft248d"][2] == .841
