"""Build the plain C libraries with a GCC- or Clang-compatible compiler.

This module uses only the standard library so setuptools can load it before the
package's runtime dependencies are installed. Default builds keep the compiler's
baseline ISA; optional x86 ELF function clones retain that baseline at runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import warnings


def configure_macos_target():
    """Use the same deployment floor for the linker and the wheel platform tag."""
    if sys.platform != "darwin":
        return None
    target = os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")
    try:
        version = tuple(int(part) for part in target.split("."))
    except ValueError as error:
        raise ValueError("MACOSX_DEPLOYMENT_TARGET must be a version number") from error
    if version < (11, 0):
        raise ValueError("The supported macOS deployment target is 11.0 or newer")
    return target


def _compiler_output(compiler, arguments, input_text=None):
    return subprocess.run(
        compiler + arguments, input=input_text, text=True, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def macos_native_architecture(libraries=()):
    """Resolve our single native target, then verify any finished Mach-O files.

    A universal2 Python can run on either architecture, but our compiler invocation
    emits one architecture. Editable wheel tags are requested before compilation;
    the compiler target is available then, and lipo checks the finished libraries.
    """
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler:
        raise ValueError("CC must name a compiler")
    target = _compiler_output(compiler, ["-dumpmachine"])
    architecture = target.split("-", 1)[0]
    architecture = {"aarch64": "arm64"}.get(architecture, architecture)
    if architecture not in ("arm64", "x86_64") or "apple" not in target:
        raise RuntimeError("Unsupported native macOS compiler target: " + target)
    for library in libraries:
        architectures = _compiler_output(["lipo"], ["-archs", str(library)]).split()
        if architectures != [architecture]:
            raise RuntimeError(
                f"Expected a native {architecture} library, got {architectures}: {library}"
            )
    return architecture


def _supports(compiler, flags, source, link_flag):
    """Compile and link, without running target code (also safe for cross CC)."""
    with tempfile.TemporaryDirectory(prefix="lpsd-probe-") as temporary:
        result = subprocess.run(
            compiler + flags + ["-Werror", link_flag, "-x", "c", "-", "-o",
                                str(Path(temporary) / "probe.so"), "-lm"],
            input=source, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    return result.returncode == 0


_SIMD_PROBE = """
double lpsd_probe(const double *x, int n) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < n; ++i) sum += x[i];
    return sum;
}
"""

_CLONES_PROBE = """
#if !defined(__ELF__) || !(defined(__x86_64__) || defined(__i386__))
#error Function dispatch is enabled only for x86 ELF targets
#else
#include <features.h>
#if !defined(__GLIBC__) || !__GLIBC_PREREQ(2, 23)
#error The supported x86 IFUNC loader requires glibc 2.23 or newer
#endif
#endif
#define LPSD_PROBE_CLONES __attribute__((target_clones("default", "avx2", "avx512f")))
/* Match the actual header/definition structure and calls between clones.
 * Clang requires matching attributes on both declaration and definition. */
LPSD_PROBE_CLONES double lpsd_probe(const double *x, int n);
LPSD_PROBE_CLONES double lpsd_probe(const double *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += x[i];
    return sum;
}
LPSD_PROBE_CLONES double lpsd_probe_caller(const double *x, int n);
LPSD_PROBE_CLONES double lpsd_probe_caller(const double *x, int n) {
    return lpsd_probe(x, n);
}
"""


def build_shared(source, target, *, simd=False, native=False):
    """Compile a source file atomically, with a JSON report beside the binary."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get("CC", "gcc" if os.name == "nt" else "cc"))
    if not compiler:
        raise ValueError("CC must name a compiler")
    flags = ["-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off"]
    if os.name != "nt":
        flags.append("-fPIC")
    deployment_target = configure_macos_target()
    if deployment_target:
        flags.append("-mmacosx-version-min=" + deployment_target)
    link_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    macro_text = _compiler_output(compiler, flags + ["-dM", "-E", "-x", "c", "-"], "")
    macros = {}
    for line in macro_text.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[0] == "#define":
            macros[parts[1]] = parts[2]
    if native:
        if "__aarch64__" in macros or "__arm64__" in macros:
            flags.append("-mcpu=native")
        elif "__x86_64__" in macros or "__i386__" in macros:
            flags.append("-march=native")
        else:
            raise ValueError("--native is currently supported on x86 and AArch64 only")

    simd_flags = []
    clones_requested, clones_enabled = "disabled", False
    if simd:
        # Some Apple Clang drivers expose SIMD only through their cc1 frontend.
        # Neither spelling links libomp or creates native worker threads.
        for candidate in (["-fopenmp-simd"], ["-Xclang", "-fopenmp-simd"]):
            if _supports(compiler, flags + candidate + ["-Wunknown-pragmas"],
                         _SIMD_PROBE, link_flag):
                simd_flags = candidate
                flags += candidate
                break
        if not simd_flags:
            warnings.warn("Compiler does not accept OpenMP SIMD; using plain C loops.",
                          RuntimeWarning, stacklevel=2)
        clones_requested = os.environ.get("LPSD_TARGET_CLONES", "auto")
        if clones_requested not in ("auto", "0", "1"):
            raise ValueError("LPSD_TARGET_CLONES must be auto, 0 or 1")
        if native and clones_requested == "1":
            raise ValueError("--native and required portable target clones are alternatives")
        if not native and clones_requested != "0":
            is_x86_elf = "__ELF__" in macros and any(
                name in macros for name in ("__x86_64__", "__i386__")
            )
            clones_enabled = is_x86_elf and _supports(
                compiler, flags + ["-Wl,-z,defs"], _CLONES_PROBE, link_flag
            )
            if clones_enabled:
                flags.append("-DLPSD_HAVE_TARGET_CLONES=1")
            elif clones_requested == "1":
                raise RuntimeError("Compiler/target failed the x86 ELF target_clones probe")

    report = {
        "schema_version": 1,
        "source": source.name,
        "library": target.name,
        "compiler": compiler,
        "compiler_version": _compiler_output(compiler, ["--version"]).splitlines()[0],
        "compiler_target": _compiler_output(compiler, ["-dumpmachine"]),
        "flags": flags,
        "link_flags": [link_flag, "-lm"],
        "native": bool(native),
        "openmp_simd_flags": simd_flags,
        "target_clones_requested": clones_requested,
        "target_clones": bool(clones_enabled),
        "macos_deployment_target": deployment_target,
        "floating_point": {
            "double_mant_dig": int(macros["__DBL_MANT_DIG__"]),
            "long_double_mant_dig": int(macros["__LDBL_MANT_DIG__"]),
            "long_double_bytes": int(macros["__SIZEOF_LONG_DOUBLE__"]),
        },
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix="lpsd-build-", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    metadata_temporary = temporary + ".json"
    command = compiler + flags + [link_flag, str(source), "-o", temporary, "-lm"]
    try:
        subprocess.run(command, check=True)
        report["binary_sha256"] = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
        Path(metadata_temporary).write_text(json.dumps(report, indent=2) + "\n")
        os.chmod(temporary, 0o755)
        # Never truncate a library that an active Python process has mapped.
        os.replace(temporary, target)
        os.replace(metadata_temporary, target.with_suffix(".build.json"))
    finally:
        for path in (temporary, metadata_temporary):
            if os.path.exists(path):
                os.unlink(path)
    print("Built", target)
    print("Compiler command:", shlex.join(command))
    return target


def build(native=False, output_directory=None):
    root = Path(__file__).resolve().parent / "_native"
    extension = ".dll" if os.name == "nt" else ".so"
    target = Path(output_directory or root) / ("liblpsd_fast" + extension)
    return build_shared(root / "fast_dft.c", target, simd=True, native=native)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true",
                        help="Optimize for this CPU; rebuild on another CPU.")
    arguments = parser.parse_args()
    build(native=arguments.native)
