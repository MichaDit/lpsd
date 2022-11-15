import sys
import subprocess
from warnings import warn


def build(setup_kwargs):
    if sys.platform in ["linux", "darwin"]:
        subprocess.call(["make", "compile"])
    else:
        warn(
            "C code is only automatically compiled when using Linux or Mac.\nThis will be a Python-only version."
        )


if __name__ == "__main__":
    build({})
