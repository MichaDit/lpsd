# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual verified pageable and pinned transfers; diagnostic, not LPSD speed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np


def main():
    import cupy as cp
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),cupy=cp.__version__,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),rows=[],completed=False,
                scope='Host wall time through final stream synchronization; allocations and data validation excluded; one full-size warmup per direction/memory type')
    for size in (8*1024**2,80000000,240000000):
        for memory in ('pageable','pinned'):
            if memory=='pinned':
                backing=cp.cuda.alloc_pinned_memory(size)
                host=np.frombuffer(backing,dtype=np.float64)
            else: host=np.empty(size//8)
            host[:]=np.arange(len(host),dtype=np.float64)
            if memory=='pinned':
                output_backing=cp.cuda.alloc_pinned_memory(size)
                output=np.frombuffer(output_backing,dtype=np.float64)
            else: output=np.empty_like(host)
            device=cp.empty_like(host)
            for rep in range(-1,args.repeats):
                for direction in ('h2d','d2h'):
                    cp.cuda.get_current_stream().synchronize()
                    t=time.perf_counter()
                    if direction=='h2d': device.set(host)
                    else: device.get(out=output)
                    cp.cuda.get_current_stream().synchronize()
                    wall=time.perf_counter()-t
                    if direction=='d2h': assert output.tobytes()==host.tobytes()
                    report['rows'].append(dict(bytes=size,memory=memory,repeat=rep,direction=direction,
                                               wall_s=wall,decimal_gb_per_s=size/wall/1e9,roundtrip_verified=True))
            del device,host,output
            cp.get_default_memory_pool().free_all_blocks()
    report['completed']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
