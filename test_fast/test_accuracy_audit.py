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
    json.dumps(metrics, allow_nan=False)


def test_relative_limit_is_strict(audit):
    metrics = audit.error_metrics([2.], [3.], [1.], relative_limit=.5)
    assert metrics["count_above_relative_limit"] == 0
    assert metrics["count_at_or_above_relative_limit"] == 1
    assert not metrics["strict_limit_satisfied"]


def test_identical_zero_and_nonfinite_outputs_are_reported_separately(audit):
    zeros = audit.error_metrics([0., 0.], [0., 0.], [1., 2.])
    assert zeros["max_relative_error_nonzero_reference"] is None
    assert zeros["strict_limit_satisfied"]
    invalid = audit.error_metrics([0., 1., np.nan], [np.inf, 1., 2.], [1., 2., 3.])
    assert invalid["reference_nonfinite_count"] == 1
    assert invalid["candidate_nonfinite_count"] == 1
    assert invalid["reference_zero_candidate_nonfinite_count"] == 1
    assert not invalid["strict_limit_satisfied"]
    json.dumps(invalid, allow_nan=False)


def test_frequency_mismatch_prevents_misaligned_comparison(audit):
    reference = pd.DataFrame({"psd": [1.], "asd": [1.]}, index=[1.])
    candidate = pd.DataFrame({"psd": [1.], "nsd": [1.]}, index=[1.0000000001])
    result = audit.compare_outputs(reference, candidate)
    assert result["status"] == "frequency_mismatch"
    assert "metrics" not in result


def test_small_audit_records_both_densities_and_source_evidence(audit, tmp_path):
    output = tmp_path / "accuracy.json"
    status = audit.main(["--n", "257", "--n-frequencies", "24", "--n-averages", "4",
                         "--cases", "white_kaiser200", "offbin_tone_kaiser200",
                         "--worst-points", "2", "--output", str(output)])
    report = json.loads(output.read_text())
    assert status == 0  # Measurement mode records outliers instead of hiding them.
    assert report["summary"]["cases"] == 2
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
    assert len(audit.CASES) == 18
    assert audit.CASES["offbin_tone_hft248d"] == ("offbin_tone", "hft248d")
    assert audit.WINDOWS["hft248d"][2] == .841
