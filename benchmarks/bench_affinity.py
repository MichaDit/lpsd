# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows SMT control using actual processor-core masks, never guessed IDs."""
import argparse
import ctypes as ct
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import statistics
import struct
import time
import numpy as np
import psutil
from lpsd_fast import lpsd
from benchmarks.bench_lpsd import build_evidence,fingerprint


def physical_cpu_ids():
    kernel=ct.WinDLL('kernel32',use_last_error=True)
    fn=kernel.GetLogicalProcessorInformationEx
    fn.argtypes=[ct.c_int,ct.c_void_p,ct.POINTER(ct.c_ulong)]
    size=ct.c_ulong()
    fn(0,None,ct.byref(size))
    if not size.value: raise OSError(ct.get_last_error())
    buffer=ct.create_string_buffer(size.value)
    if not fn(0,buffer,ct.byref(size)): raise OSError(ct.get_last_error())
    pos=0; rows=[]
    while pos<size.value:
        relation,length=struct.unpack_from('II',buffer.raw,pos)
        groups=struct.unpack_from('H',buffer.raw,pos+30)[0]
        if relation!=0 or groups!=1: raise RuntimeError('Only one processor group is supported by this experiment')
        mask,group=struct.unpack_from('QH',buffer.raw,pos+32)
        if group!=0: raise RuntimeError('Unexpected processor group')
        rows.append([i for i in range(64) if mask&(1<<i)])
        pos+=length
    return [row[0] for row in rows],rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',type=int,nargs='+',default=[10000000,30000000])
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    physical,topology=physical_cpu_ids()
    process=psutil.Process()
    original=process.cpu_affinity()
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),build_reports=build_evidence(),
                topology=topology,original_affinity=original,physical_affinity=physical,
                scope='Only this process affinity changes; balanced shuffled complete API calls; all observations retained',
                configurations=[],completed=False)
    names={'all-16':(original,16),'all-24':(original,24),'physical-16':(physical,16)}
    try:
        for n in args.sizes:
            x=np.random.default_rng(20261008).normal(size=n)
            kwargs=dict(sample_rate=1.,kernel='fast',outputs='psd',max_working_mb=4096)
            config=dict(n=n,calls=[])
            report['configurations'].append(config)
            for affinity,workers in names.values():
                process.cpu_affinity(affinity)
                lpsd(x,workers=workers,**kwargs)
            for repeat in range(args.repeats):
                order=list(names)
                random.Random(137+repeat).shuffle(order)
                for name in order:
                    affinity,workers=names[name]
                    process.cpu_affinity(affinity)
                    t=time.perf_counter()
                    result=lpsd(x,workers=workers,**kwargs)
                    wall=time.perf_counter()-t
                    config['calls'].append(dict(name=name,repeat=repeat,wall_s=wall,output_sha256=fingerprint(result)))
                    print(n,name,repeat,round(wall,6),flush=True)
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
            config['median_wall_s']={name:statistics.median(row['wall_s'] for row in config['calls'] if row['name']==name) for name in names}
            assert len({row['output_sha256'] for row in config['calls']})==1
        report['completed']=True
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        process.cpu_affinity(original)


if __name__=='__main__': main()
