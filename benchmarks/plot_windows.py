# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the retained local worker and CUDA observations; no estimator runs.

python -m benchmarks.plot_windows --results benchmarks/results_windows_20261010_raw.json.gz
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.plot_comparisons import configure, save
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, default=Path('docs/figures'))
    args = parser.parse_args()
    raw = args.results.read_bytes()
    reports = json.loads(gzip.decompress(raw))['reports']
    sources = {str(args.results): hashlib.sha256(raw).hexdigest(),
               'benchmarks/plot_windows.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    configure()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5))
    for ax, n in zip(axes, (1000000, 10000000, 30000000)):
        for report_name, label, color, marker in (
            ('worker-screen.json', 'Screen: 3 calls / count', '#777777', 'o'),
            ('worker-confirm.json', 'Confirmation: 7 calls / count', '#0072B2', 's'),
        ):
            config = next(c for c in reports[report_name]['configurations'] if c['n'] == n)
            counts = sorted({r['workers'] for r in config['calls']})
            values = [[r['wall_s'] for r in config['calls'] if r['workers'] == w] for w in counts]
            medians = np.array([statistics.median(v) for v in values])
            errors = np.array([[m-min(v) for m, v in zip(medians, values)],
                               [max(v)-m for m, v in zip(medians, values)]])
            ax.errorbar(counts, medians, yerr=errors, color=color, marker=marker,
                        markersize=4, linewidth=1.3, capsize=2, label=label)
        ax.set_title(f'{n/1e6:g} million samples')
        ax.set_xlabel('Frequency workers')
        ax.set_xticks([1, 8, 16, 24, 32])
        ax.set_xlim(0, 33)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Complete PSD API wall time (s)')
    axes[1].legend(loc='upper right', fontsize=8)
    fig.suptitle('Ryzen 9 7945HX, Windows, Silent power plan — native fast PSD', y=1.01)
    fig.text(.5, -.04, 'Points: medians; bars: all retained minima/maxima. Separate series are not pooled.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, args.output_directory, 'windows-workers', 'Local frequency worker measurements', sources)

    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    x = np.arange(3)
    for offset, name, label, color in ((-.19, 'cpu', 'CPU fast, complete host call', '#0072B2'),
                                     (.19, 'gpu', 'RTX 4070 FP64 prototype, complete host call', '#E69F00')):
        values = [[r['wall_s'] for r in reports[f'cuda-{size}m.json']['calls'] if r['name'] == name]
                  for size in (1, 10, 30)]
        medians = np.array([statistics.median(v) for v in values])
        errors = np.array([[m-min(v) for m, v in zip(medians, values)],
                           [max(v)-m for m, v in zip(medians, values)]])
        bars = ax.bar(x+offset, medians, .36, yerr=errors, capsize=3, color=color, label=label)
        ax.bar_label(bars, labels=[f'{m:.3f} s' for m in medians], padding=5, fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(.1, 25)
    ax.set_xticks(x, ['1 million\n16 CPU workers', '10 million\n16 CPU workers', '30 million\n24 CPU workers'])
    ax.set_ylabel('Complete PSD + NSD API wall time (s, log scale)')
    ax.set_title('CUDA prototype: seven alternating CPU/GPU pairs per size')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=.2)
    fig.text(.5, -.02, 'Fresh host preparation and transfers on every call; medians and full ranges. White-noise input.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, args.output_directory, 'windows-cuda', 'Local complete CPU and CUDA host calls', sources)


if __name__ == '__main__':
    main()
