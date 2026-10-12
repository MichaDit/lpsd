"""Small platform queries used only to size concurrent workspaces."""
import ctypes as ct
import os
import sys


def _windows_available_memory():
    # MEMORYSTATUSEX contains fixed-width Windows types, including on Win64.
    # ullAvailPhys includes immediately reclaimable standby pages.
    class MemoryStatus(ct.Structure):
        _fields_ = [("length", ct.c_uint32), ("load", ct.c_uint32)] + [
            (name, ct.c_uint64) for name in (
                "total_physical", "available_physical", "total_pagefile",
                "available_pagefile", "total_virtual", "available_virtual", "extended")]

    query = ct.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    query.argtypes = [ct.POINTER(MemoryStatus)]
    query.restype = ct.c_int
    status = MemoryStatus()
    status.length = ct.sizeof(status)
    if query(ct.byref(status)) and status.available_physical <= status.total_physical:
        return int(status.available_physical)
    return None


def _macos_available_memory():
    # The original HOST_VM_INFO ABI is a fixed array of fifteen natural_t
    # fields; Apple explicitly freezes this structure. We only need its
    # page gauges, which fit uint32 on the supported machines. This avoids
    # depending on a particular revision/size of vm_statistics64.
    libc = ct.CDLL(None)
    host_self = libc.mach_host_self
    host_self.argtypes = []
    host_self.restype = ct.c_uint32
    query = libc.host_statistics
    query.argtypes = [ct.c_uint32, ct.c_int, ct.POINTER(ct.c_int32),
                     ct.POINTER(ct.c_uint32)]
    query.restype = ct.c_int
    release = libc.mach_port_deallocate
    release.argtypes = [ct.c_uint32, ct.c_uint32]
    release.restype = ct.c_int
    task = ct.c_uint32.in_dll(libc, "mach_task_self_").value
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size <= 0:
        return None
    stats = (ct.c_uint32 * 15)()
    count = ct.c_uint32(len(stats))
    host = host_self()
    if not host:
        return None
    try:
        result = query(host, 2, ct.cast(stats, ct.POINTER(ct.c_int32)), ct.byref(count))
        if result == 0 and count.value >= 4:
            # free_count already includes speculative_count. Count inactive
            # pages once; never add purgeable, speculative or compressed
            # gauges a second time. This is a concurrency estimate, not an
            # allocation guarantee or a prediction of compression pressure.
            return (int(stats[0]) + int(stats[2])) * page_size
    finally:
        release(task, host)
    return None


def host_available_memory():
    """Return available bytes on Windows/macOS, or None if unavailable."""
    try:
        if sys.platform == "win32":
            return _windows_available_memory()
        if sys.platform == "darwin":
            return _macos_available_memory()
    except (AttributeError, OSError, ValueError):
        pass
    return None
