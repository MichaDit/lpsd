"""Revision-report checks; CI itself supplies the actual native timing pairs."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ci_architecture", ROOT / "benchmarks/ci_architecture.py")
architecture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(architecture)


def test_paired_report_uses_ratio_of_medians_and_retains_scatter():
    # These ratios are 2, 1/10 and 5/2. The ratio of medians is instead 1/2.
    pairs = [
        {"baseline": {"wall_s": a}, "current": {"wall_s": b}, "outputs_identical": True}
        for a, b in ((2., 1.), (3., 30.), (15., 6.))
    ]
    result = architecture._paired_summary(pairs)
    assert result["ratio_of_medians"] == 0.5
    assert result["wall_time_change_percent"] == 100.
    assert result["baseline_range_wall_s"] == [2., 15.]
    assert result["current_range_wall_s"] == [1., 30.]
    assert result["outputs_identical"]
    pairs[1]["outputs_identical"] = False
    result = architecture._paired_summary(pairs)
    assert result["comparison_status"] == "numerical_comparison_required"
    assert not result["outputs_identical"]


@pytest.mark.parametrize("arguments", [
    ["--comparison-only"],
    ["--comparison-sizes", "1000000000"],
    ["--comparison-sizes", "0"],
])
def test_revision_comparison_rejects_missing_baseline_or_unbounded_sizes(arguments):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks/ci_architecture.py"), *arguments],
        capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 2
    assert "error:" in completed.stderr
