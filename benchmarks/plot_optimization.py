# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the sanitized local optimization evidence without running estimators."""
import argparse,gzip,hashlib,json,statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from benchmarks.plot_comparisons import configure,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output-directory',type=Path,default=Path('docs/figures'))
    args=p.parse_args()
    raw=args.results.read_bytes();reports=json.loads(gzip.decompress(raw))['reports']
    sources={str(args.results):hashlib.sha256(raw).hexdigest(),
             'benchmarks/plot_optimization.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    configure();fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    panels=[
        ('LPSD, full host call',[('packed-full-final-10m.json',n) for n in ('public_cpu','packed_gpu','legacy_gpu')],
         ['CPU 16','Packed GPU','Earlier GPU']),
        ('LPSD, reused plan',[(f'packed-resident-confirm-10000000.json',n) for n in ('public_cpu','prepared_cpu','resident_gpu')],
         ['Public CPU 16','Prepared CPU 16','Resident GPU']),
        ('FFT periodogram, reused plan',[(f'fft-tune-final-10000000.json',n) for n in ('fftw_16','cufft_graph','vkfft_tuned_graph')],
         ['FFTW 16','cuFFT graph','VkFFT tuned graph'])]
    for ax,(title,series,labels) in zip(axes,panels):
        values=[[r['wall_s'] for r in reports[f]['calls'] if r['name']==name and r.get('scope','prepared_complete')=='prepared_complete'] for f,name in series]
        medians=np.array([statistics.median(v) for v in values])
        errors=np.array([[m-min(v) for m,v in zip(medians,values)],[max(v)-m for m,v in zip(medians,values)]])
        bars=ax.bar(np.arange(3),medians,color=['#0072B2','#E69F00','#009E73'],yerr=errors,capsize=3,width=.65)
        ax.bar_label(bars,labels=[f'{v:.4g} s' for v in medians],padding=5,fontsize=9)
        ax.set_xticks(np.arange(3),labels,rotation=15,ha='right',fontsize=9)
        ax.set_title(title,fontsize=11);ax.set_ylabel('Host wall time (s)');ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,max(max(v) for v in values)*1.2)
    fig.suptitle('10 million samples — Ryzen 9 7945HX / RTX 4070 Laptop, Windows, Silent',fontsize=12)
    fig.text(.5,.01,'Seven calls per series; medians and full ranges. Reused-plan construction excluded. FFT is a different estimator.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));args.output_directory.mkdir(parents=True,exist_ok=True)
    save(fig,args.output_directory,'windows-optimization','Local LPSD and FFT optimization measurements',sources)


if __name__=='__main__':main()
