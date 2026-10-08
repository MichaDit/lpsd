"""The comparison harness must measure and retain complex spectral errors."""
import numpy as np
import pandas as pd
import pytest

from benchmarks.bench_segment_core import compare_spectra, main


def test_csd_comparison_keeps_imaginary_error():
    reference = pd.DataFrame({"psd": np.asarray([1 + 2j, 0j], dtype=np.complex64)})
    candidate = reference.copy()
    candidate.iloc[0, 0] += np.complex64(.001j)
    report = compare_spectra(reference, candidate)["psd"]
    assert not report["bitwise_equal"]
    assert .00099 < report["max_absolute_error"] < .00101
    candidate.iloc[0, 0] += np.complex64(.1j)
    with pytest.raises(AssertionError, match="numerical criterion"):
        compare_spectra(reference, candidate)


def test_comparison_rejects_changed_zeros_and_dtypes():
    reference = pd.DataFrame({"psd": np.asarray([0.], dtype=np.float32)})
    candidate = reference.copy()
    candidate.iloc[0, 0] = np.float32(1e-30)
    with pytest.raises(AssertionError, match="numerical criterion"):
        compare_spectra(reference, candidate)
    with pytest.raises(AssertionError, match="dtypes"):
        compare_spectra(reference, reference.astype(np.complex64))


@pytest.mark.parametrize("kind", ("psd", "csd"))
def test_small_comparison_records_all_calls_and_separate_profiles(kind, tmp_path):
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "comparison.json"
    main(["--baseline-root", str(root), "--candidate-root", str(root),
          "--kind", kind, "--n", "1025", "--frequencies", "16", "--averages", "4",
          "--repeats", "2", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["completed"]
    assert report["kind"] == kind
    assert [call["implementation"] for call in report["calls"]] == [
        "baseline", "candidate", "candidate", "baseline"]
    assert all(call["comparison"]["psd"]["bitwise_equal"] for call in report["calls"])
    assert len(report["profiles"]) == 2
    assert all(profile["summary"]["sample_iterations"] > 0
               for profile in report["profiles"].values())
