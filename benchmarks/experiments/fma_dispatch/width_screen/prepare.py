#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reconstruct the measured FMA dispatch candidates in a separate directory.

This extracts pinned Git objects, applies the recorded snapshot and candidate
patches, and verifies every source hash. Compilation happens only with --build.
The repository containing the pinned objects is never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ARCHIVE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    return json.loads((ARCHIVE / 'manifest.json').read_text())


def verify_files(root, expected):
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f'Source hash mismatch: {path}')


def verify_prepared(output):
    """Source-only check; never imports the package or builds native code."""
    description = manifest()
    for name, variant in description['variants'].items():
        expected = dict(description['snapshot_files_sha256'])
        expected.update(variant['changed_files_sha256'])
        verify_files(Path(output) / name, expected)


def apply_patch(root, path):
    # An independent empty repository makes git apply unambiguous even when
    # output lies inside another worktree. No files are staged or committed.
    subprocess.run(['git', 'init', '--quiet', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'apply', '--whitespace=nowarn',
                    str(path)], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True,
                        help='Git repository containing the pinned base commit')
    parser.add_argument('--output', type=Path, required=True,
                        help='New scratch directory; existing paths are refused')
    parser.add_argument('--build', action='store_true',
                        help='Compile the portable candidates after source verification')
    args = parser.parse_args(argv)
    repo, output = args.repo.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f'Output already exists: {output}')
    description = manifest()
    commit = description['base_git_commit']
    subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', commit + '^{commit}'],
                   check=True)
    control = output / 'control8'
    control.mkdir(parents=True)
    for name, expected in description['base_files_sha256'].items():
        data = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{name}'])
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f'Pinned base hash mismatch: {name}')
        target = control / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    apply_patch(control, ARCHIVE / description['snapshot_patch'])
    verify_files(control, description['snapshot_files_sha256'])
    for name, variant in description['variants'].items():
        if name == 'control8':
            continue
        directory = output / name
        shutil.copytree(control, directory,
                        ignore=shutil.ignore_patterns('.git', '__pycache__'))
        if variant['patch']:
            apply_patch(directory, ARCHIVE / variant['patch'])
    verify_prepared(output)
    if args.build:
        for name in description['variants']:
            if name == 'strict':
                # The measured strict control used the same native binary.
                for suffix in ('.so', '.build.json'):
                    relative = 'lpsd_fast/_native/liblpsd_fast' + suffix
                    shutil.copyfile(control / relative, output / name / relative)
            else:
                subprocess.run([sys.executable,
                                str(output / name / 'lpsd_fast/build.py')], check=True)
        verify_prepared(output)
    print(json.dumps({'prepared': str(output), 'built': bool(args.build),
                      'base_commit': commit, 'variants': list(description['variants'])}, indent=2))


if __name__ == '__main__':
    main()
