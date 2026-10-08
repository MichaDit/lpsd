"""Build platform wheels for the plain C ABI used through ctypes."""
import importlib.util
import os
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


def load_build_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildNative(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        destination = root if self.editable_mode else Path(self.build_lib)
        legacy = load_build_module(root / "build_native.py", "lpsd_legacy_build")
        legacy.build_legacy(destination / "lpsd")
        fast = load_build_module(root / "lpsd_fast" / "build.py", "lpsd_fast_build")
        fast.build(native=os.environ.get("LPSD_NATIVE") == "1",
                   output_directory=destination / "lpsd_fast" / "_native")


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        # Neither library imports the CPython or NumPy C API.
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(
    distclass=NativeDistribution,
    cmdclass={"build_py": BuildNative, "bdist_wheel": PlatformWheel},
)
