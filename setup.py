"""Build platform wheels for the plain C ABI used through ctypes."""
import importlib.util
import os
from pathlib import Path
import sys

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
        native = os.environ.get("LPSD_NATIVE") == "1"
        if native and not self.editable_mode:
            raise RuntimeError(
                "LPSD_NATIVE=1 is only for editable/local builds. "
                "Build redistributable wheels with LPSD_NATIVE=0."
            )
        super().run()
        root = Path(__file__).resolve().parent
        destination = root if self.editable_mode else Path(self.build_lib)
        legacy = load_build_module(root / "build_native.py", "lpsd_legacy_build")
        legacy_library = legacy.build_legacy(destination / "lpsd")
        fast = load_build_module(root / "lpsd_fast" / "build.py", "lpsd_fast_build")
        fast_library = fast.build(
            native=native, output_directory=destination / "lpsd_fast" / "_native"
        )
        if sys.platform == "darwin":
            self.distribution.lpsd_macos_architecture = fast.macos_native_architecture(
                (legacy_library, fast_library)
            )


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        # Neither library imports the CPython or NumPy C API.
        _, _, platform_tag = super().get_tag()
        if platform_tag.startswith("macosx_"):
            architecture = getattr(self.distribution, "lpsd_macos_architecture", None)
            if architecture is None:
                # Editable wheels request their tag before BuildNative.run.
                architecture = platform_build.macos_native_architecture()
            platform_tag = "_".join(platform_tag.split("_")[:3] + [architecture])
        return "py3", "none", platform_tag


# This precedes wheel tag calculation as well as the actual compiler call.
platform_build = load_build_module(
    Path(__file__).resolve().parent / "lpsd_fast" / "build.py", "lpsd_platform_config"
)
platform_build.configure_macos_target()

setup(
    distclass=NativeDistribution,
    cmdclass={"build_py": BuildNative, "bdist_wheel": PlatformWheel},
)
