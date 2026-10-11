# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional NVML clocks/power/temperature only; never read device identities."""
import ctypes as ct
import os

_library=None
_device=None
_attempted=False


def snapshot():
    global _library,_device,_attempted
    if not _attempted:
        _attempted=True
        try:
            lib=ct.CDLL('nvml.dll' if os.name=='nt' else 'libnvidia-ml.so.1')
            if lib.nvmlInit_v2(): return {'available':False}
            device=ct.c_void_p()
            lib.nvmlDeviceGetHandleByIndex_v2.argtypes=[ct.c_uint,ct.POINTER(ct.c_void_p)]
            if lib.nvmlDeviceGetHandleByIndex_v2(0,ct.byref(device)): return {'available':False}
            _library,_device=lib,device
        except OSError: return {'available':False}
    if _library is None: return {'available':False}
    values={'available':True}
    for key,name,extra in (
        ('sm_mhz','nvmlDeviceGetClockInfo',[1]),('memory_mhz','nvmlDeviceGetClockInfo',[2]),
        ('temperature_c','nvmlDeviceGetTemperature',[0]),('power_mw','nvmlDeviceGetPowerUsage',[]),
        ('pstate','nvmlDeviceGetPerformanceState',[])):
        result=ct.c_uint()
        fn=getattr(_library,name)
        fn.argtypes=[ct.c_void_p]+[ct.c_uint]*len(extra)+[ct.POINTER(ct.c_uint)]
        status=fn(_device,*extra,ct.byref(result))
        values[key]=result.value if status==0 else None
    return values
