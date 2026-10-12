"""Exercise platform ctypes layouts and failure handling with a C ABI stub."""
import ctypes as ct
import os
import sys

import pytest

from lpsd_fast import _resources
from lpsd_fast.build import build_shared


@pytest.fixture(scope='module')
def platform_stub(tmp_path_factory):
    temporary = tmp_path_factory.mktemp('platform-memory-abi')
    source = temporary / 'platform_memory.c'
    source.write_text(r'''
#include <stdint.h>
uint32_t mach_task_self_ = 101;
int query_state = 0;
uint32_t last_release_task = 0, last_release_host = 0, host_releases = 0;
uint32_t mach_host_self(void) { return query_state == 3 ? 0 : 203; }
int host_statistics(uint32_t host, int flavor, int32_t *info, uint32_t *count) {
    if (host != 203 || flavor != 2 || *count != 15) return -1;
    info[0] = 4; info[1] = 128; info[2] = 7; info[3] = 64;
    info[12] = 999; info[14] = 2;
    if (query_state == 2) *count = 2;
    return query_state == 1 ? -1 : 0;
}
int mach_port_deallocate(uint32_t task, uint32_t host) {
    last_release_task = task; last_release_host = host; ++host_releases;
    return 0;
}
struct memory_status {
    uint32_t length, load;
    uint64_t total_physical, available_physical, total_pagefile;
    uint64_t available_pagefile, total_virtual, available_virtual, extended;
};
int GlobalMemoryStatusEx(struct memory_status *status) {
    if (status->length != sizeof(*status)) return 0;
    status->total_physical = UINT64_C(32) << 30;
    status->available_physical = query_state == 2 ? UINT64_C(64) << 30 :
                                 query_state == 3 ? 0 : UINT64_C(5) << 30;
    return query_state == 1 ? 0 : 1;
}
''')
    target = temporary / ('platform_memory.dll' if os.name == 'nt' else 'platform_memory.so')
    build_shared(source, target)
    return ct.CDLL(str(target))


@pytest.mark.parametrize('state,expected', ((0, 11 * 16384), (1, None), (2, None), (3, None)))
def test_macos_counts_available_pages_once_and_releases_host_right(platform_stub, monkeypatch,
                                                                 state, expected):
    ct.c_int.in_dll(platform_stub, 'query_state').value = state
    ct.c_uint32.in_dll(platform_stub, 'host_releases').value = 0
    monkeypatch.setattr(_resources.ct, 'CDLL', lambda filename: platform_stub)
    monkeypatch.setattr(_resources.os, 'sysconf', lambda name: 16384, raising=False)
    assert _resources._macos_available_memory() == expected
    releases = ct.c_uint32.in_dll(platform_stub, 'host_releases').value
    assert releases == (0 if state == 3 else 1)
    if releases:
        assert ct.c_uint32.in_dll(platform_stub, 'last_release_task').value == 101
        assert ct.c_uint32.in_dll(platform_stub, 'last_release_host').value == 203


@pytest.mark.parametrize('state,expected', ((0, 5 * 1024**3), (1, None), (2, None), (3, 0)))
def test_windows_reads_available_physical_bytes_without_32_bit_truncation(
        platform_stub, monkeypatch, state, expected):
    ct.c_int.in_dll(platform_stub, 'query_state').value = state
    monkeypatch.setattr(_resources.ct, 'WinDLL', lambda *args, **kwargs: platform_stub,
                        raising=False)
    assert _resources._windows_available_memory() == expected


@pytest.mark.parametrize('platform,function', (
    ('darwin', '_macos_available_memory'), ('win32', '_windows_available_memory'),
))
@pytest.mark.parametrize('failure', (AttributeError, OSError, ValueError))
def test_platform_queries_fail_back_to_unknown(monkeypatch, platform, function, failure):
    def unavailable():
        raise failure('Unavailable host interface')

    monkeypatch.setattr(_resources.sys, 'platform', platform)
    monkeypatch.setattr(_resources, function, unavailable)
    assert _resources.host_available_memory() is None


@pytest.mark.skipif(sys.platform not in ('darwin', 'win32'), reason='Native platform query')
def test_native_platform_available_memory_query_succeeds():
    # A silently failing real ABI must not hide behind the 2 GiB fallback.
    available = _resources.host_available_memory()
    assert available is not None and 0 <= available < 2**60
