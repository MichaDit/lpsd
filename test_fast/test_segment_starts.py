"""Exercise the actual native start helper without allocating large signals."""
import ctypes as ct
import math
import os
from pathlib import Path

import pytest

from lpsd_fast.build import build_shared


HEADER = Path(__file__).resolve().parents[1] / "lpsd_fast" / "_native" / "segment_starts.h"


@pytest.fixture(scope="module")
def starts_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("segment-starts")
    source = directory / "probe.c"
    source.write_text(f'#include "{HEADER.as_posix()}"\n' + r'''
long int resolve_start(double start, long int index, long int count,
                       long int endpoint)
{
    return lpsd_segment_start(start, index, count, endpoint);
}

/* This is an O(1)-memory index trace, not a DFT or a signal allocation.
 * Match the production count/shift arithmetic and call its actual inline
 * helper at each position, using absolute indices for batch members. */
int walk_starts(long int n, long int length, double overlap, int batch_width,
                long int *counts, double *last)
{
    const double ovfact = 1.0 / (1.0 - overlap / 100.0);
    const double davg = ((double)(n - length) * ovfact) / length + 1.0;
    const long int count = (long int)floor(davg + 0.5);
    const long int endpoint = n - length;
    double shift = count == 1 ? 1.0 : (double)endpoint / (double)(count - 1);
    if (shift < 1.0) shift = 1.0;
    if (count < 1 || count > endpoint + 1 || batch_width < 1) return 1;
    for (int j = 0; j < 8; ++j) counts[j] = 0;
    for (int j = 0; j < 4; ++j) last[j] = 0.0;
    counts[0] = count;
    counts[1] = -1;  /* first formerly out-of-bounds index */
    last[0] = shift;
    last[3] = (double)(count - 1) * shift;
    double start = 0.0;
    for (long int ii = 0; ii < count;) {
        const int batch = count - ii < batch_width ? (int)(count - ii) : batch_width;
        for (int b = 0; b < batch; ++b) {
            const long int index = ii + b;
            const long int original = (long int)floor(start + 0.5);
            const long int corrected = lpsd_segment_start(start, index, count, endpoint);
            const int safe = original >= 0 && original <= endpoint;
            if (!safe && counts[1] < 0) counts[1] = index;
            if (corrected != original) ++counts[2];
            if (corrected < 0 || corrected > endpoint) ++counts[3];
            if (safe && corrected != original) ++counts[4];
            if (index == count - 2) {
                last[2] = start;
                counts[7] = original;
            }
            if (index == count - 1) {
                last[1] = start;
                counts[5] = corrected;
                counts[6] = original;
            }
            start += shift;
        }
        ii += batch;
    }
    return 0;
}
''', encoding="utf-8")
    library_path = directory / ("probe.dll" if os.name == "nt" else "probe.so")
    build_shared(source, library_path)
    library = ct.CDLL(str(library_path))
    library.resolve_start.argtypes = [ct.c_double, ct.c_long, ct.c_long, ct.c_long]
    library.resolve_start.restype = ct.c_long
    library.walk_starts.argtypes = [ct.c_long, ct.c_long, ct.c_double, ct.c_int,
                                   ct.POINTER(ct.c_long), ct.POINTER(ct.c_double)]
    library.walk_starts.restype = ct.c_int
    return library


@pytest.mark.parametrize("start,index,count,endpoint", (
    (0.0, 0, 1, 0), (-0.49, 0, 2, 10),
    (0.5, 1, 4, 10), (math.nextafter(0.5, 0.0), 1, 4, 10),
    (8.6, 2, 4, 10), (9.0, 3, 4, 10),
    (9.49, 3, 4, 10), (10.49, 3, 4, 10),
    (2_147_483_647.0, 8, 9, 2_147_483_647),
))
def test_every_previously_safe_start_is_unchanged(starts_probe, start, index, count, endpoint):
    expected = math.floor(start + 0.5)
    assert 0 <= expected <= endpoint
    assert starts_probe.resolve_start(start, index, count, endpoint) == expected


@pytest.mark.parametrize("index", range(8))
def test_only_last_absolute_batch_index_may_repair_an_overshoot(starts_probe, index):
    # Using the batch base instead of ii+b must not grant another segment
    # the final-endpoint exception. This also covers all tail positions.
    assert starts_probe.resolve_start(100.51, index, 8, 100) == (100 if index == 7 else -1)


@pytest.mark.parametrize("start", (-1.0, -0.500001, -1e300, math.inf, -math.inf, math.nan))
@pytest.mark.parametrize("index", (0, 7))
def test_negative_rounded_and_nonfinite_starts_are_rejected_before_integer_conversion(starts_probe, start, index):
    assert starts_probe.resolve_start(start, index, 8, 100) == -1


def test_final_positive_overflow_is_bounded_before_a_32_bit_long_cast(starts_probe):
    endpoint = 2_147_483_647
    assert starts_probe.resolve_start(float(endpoint) + 0.5, 6, 8, endpoint) == -1
    assert starts_probe.resolve_start(float(endpoint) + 0.5, 7, 8, endpoint) == endpoint
    assert starts_probe.resolve_start(1e300, 6, 8, endpoint) == -1
    assert starts_probe.resolve_start(1e300, 7, 8, endpoint) == endpoint


def trace(starts_probe, n, length, overlap, batch):
    counts, values = (ct.c_long * 8)(), (ct.c_double * 4)()
    assert starts_probe.walk_starts(n, length, overlap, batch, counts, values) == 0
    return list(counts), list(values)


@pytest.mark.parametrize("batch", (1, 8))
def test_actual_billion_sample_plan_repairs_one_last_start_in_constant_memory(starts_probe, batch):
    counts, values = trace(starts_probe, 1_000_000_000, 146,
                           0.7658465180619339 * 100.0, batch)
    assert counts == [29_251_388, 29_251_387, 1, 0, 0,
                      999_999_854, 999_999_855, 999_999_820]
    assert values[0] == 34.18640811801505
    assert values[1] == 999_999_854.5296893
    assert values[2] == 999_999_820.3432811
    assert values[3] == 999_999_854.0


@pytest.mark.parametrize("n,length,overlap,batch", (
    (146, 146, 76.58465180619339, 1),
    (4099, 146, 76.58465180619339, 8),
    (1_000_003, 127, 76.58465180619339, 4),
    (1_000_003, 2049, 75.0, 8),
))
def test_safe_recurrences_keep_every_start_and_count(starts_probe, n, length, overlap, batch):
    counts, _ = trace(starts_probe, n, length, overlap, batch)
    assert counts[1] == -1
    assert counts[2:5] == [0, 0, 0]
    assert 0 <= counts[5] <= n - length
