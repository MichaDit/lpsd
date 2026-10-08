"""Build the unchanged original ctypes library with the shared FP policy."""
import importlib.util
import os
from pathlib import Path


def build_legacy(output_directory=None):
    repository = Path(__file__).resolve().parent
    # Do not import lpsd_fast.__init__: isolated builds need no NumPy/Pandas.
    spec = importlib.util.spec_from_file_location(
        "lpsd_native_builder", repository / "lpsd_fast" / "build.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    root = repository / "lpsd"
    suffix = ".dll" if os.name == "nt" else ".so"
    target = Path(output_directory or root) / ("ltpda_dft" + suffix)
    return builder.build_shared(root / "ltpda_dft.c", target)


if __name__ == "__main__":
    build_legacy()
