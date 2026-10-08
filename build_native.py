"""Build the original ctypes library from source, failing on compiler errors."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def build_legacy(output_directory=None):
    root = Path(__file__).resolve().parent / "lpsd"
    destination = Path(output_directory or root)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if os.name == "nt" else ".so"
    target = destination / ("ltpda_dft" + suffix)
    compiler = shlex.split(os.environ.get("CC", "gcc"))
    flags = ["-O3"]
    if os.name != "nt":
        flags.append("-fPIC")
    descriptor, temporary = tempfile.mkstemp(
        prefix="lpsd-build-", suffix=suffix, dir=destination
    )
    os.close(descriptor)
    try:
        subprocess.run(
            compiler + flags + ["-shared", str(root / "ltpda_dft.c"),
                                "-o", temporary, "-lm"],
            check=True,
        )
        os.chmod(temporary, 0o755)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


if __name__ == "__main__":
    print("Built", build_legacy())
