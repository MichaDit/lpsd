"""Build the plain C shared library; no Python or NumPy headers are needed."""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import tempfile


def build(native=False, output_directory=None):
    root = Path(__file__).resolve().parent / '_native'
    extension = '.dll' if os.name == 'nt' else '.so'
    target = Path(output_directory or root) / ('liblpsd_fast' + extension)
    target.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CC', 'gcc'))
    flags = ['-O3', '-std=c11', '-fopenmp-simd', '-ffp-contract=off']
    if os.name != 'nt':
        flags += ['-fPIC']
    if native:
        flags += ['-march=native']
    # Never truncate a library that an active Python process has mapped.
    fd, temporary = tempfile.mkstemp(prefix='lpsd-build-', suffix=extension, dir=target.parent)
    os.close(fd)
    command = compiler + flags + ['-shared', str(root / 'fast_dft.c'), '-o', temporary, '-lm']
    try:
        subprocess.run(command, check=True)
        os.chmod(temporary, 0o755)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print('Built', target)
    print('Compiler command:', shlex.join(command))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true', help='Optimize for this CPU; rebuild on another CPU.')
    arguments = parser.parse_args()
    build(native=arguments.native)
