# SPDX-License-Identifier: GPL-3.0-or-later
"""Matched portable/native complete API pairs on one CPU, with both binaries."""
import argparse
import ctypes as ct
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time
import numpy as np
from lpsd_fast import api,lpsd
from benchmarks.bench_lpsd import fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portable',type=Path,required=True)
    parser.add_argument('--sizes',type=int,nargs='+',default=[1000000,10000000])
    parser.add_argument('--workers',type=int,nargs='+',default=[1,16])
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    api._native()
    original=api._LIB
    portable=ct.CDLL(str(args.portable.resolve()))
    for name,declaration in vars(original).items():
        if isinstance(declaration,ct._CFuncPtr):
            function=getattr(portable,name)
            function.argtypes,function.restype=declaration.argtypes,declaration.restype
    libraries={'portable':portable,'native':original}
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                builds={name:json.loads(Path(lib._name).with_suffix('.build.json').read_text()) for name,lib in libraries.items()},
                method='Identical input/parameters; full warmup per library; balanced A/B B/A complete API pairs; no outliers removed',
                configurations=[],completed=False)
    try:
        for n in args.sizes:
            x=np.random.default_rng(20261008).normal(size=n)
            for workers in args.workers:
                kwargs=dict(sample_rate=1.,kernel='fast',workers=workers,outputs='psd',max_working_mb=4096)
                config=dict(n=n,workers=workers,calls=[])
                report['configurations'].append(config)
                reference=None
                for lib in libraries.values():
                    api._LIB=lib
                    result=lpsd(x,**kwargs)
                    if reference is None: reference=result
                for repeat in range(args.repeats):
                    for name in (['portable','native'] if repeat%2==0 else ['native','portable']):
                        api._LIB=libraries[name]
                        t=time.perf_counter()
                        result=lpsd(x,**kwargs)
                        wall=time.perf_counter()-t
                        assert result.index.equals(reference.index) and result.dtypes.equals(reference.dtypes)
                        rel=float(np.max(np.abs(result.psd.to_numpy().astype(float)/reference.psd.to_numpy()-1)))
                        assert rel<.01
                        config['calls'].append(dict(name=name,repeat=repeat,wall_s=wall,output_sha256=fingerprint(result),
                                                    max_relative_error=rel,metadata=result.attrs['lpsd_fast']))
                        print(n,workers,name,repeat,round(wall,6),flush=True)
                        args.output.write_text(json.dumps(report,indent=2)+'\n')
                config['median_wall_s']={name:statistics.median(r['wall_s'] for r in config['calls'] if r['name']==name) for name in libraries}
        report['completed']=True
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        api._LIB=original


if __name__=='__main__': main()
