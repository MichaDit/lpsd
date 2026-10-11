"""CI gate for the real private bootstrap; no timing threshold or upload."""
import json
from pathlib import Path
import sys


def check(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [case for case in report["cases"] if case["n"] == 100000
            and case["signal"] == "white" and case["case_role"] == "comparison"]
    expected = {(method, points) for method in ("fftw", "matched", "lpsd")
                for points in (469, 938)}
    actual = {(case["method"], case["actual_points"])
              for case in rows if case["status"] == "completed"}
    assert actual == expected, (actual, [(case["method"], case["status"],
                                         case.get("reason")) for case in rows])
    assert report["state"] == "completed", report["state"]
    assert all(case["input_unchanged"] and case["nsd_matches_sqrt_psd"]
               and case["finite"] and case["psd_nonnegative"] for case in rows)
    print("All three methods completed with both actual point counts and valid outputs")


if __name__ == "__main__":
    check(sys.argv[1])
