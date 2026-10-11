/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef LPSD_SEGMENT_STARTS_H
#define LPSD_SEGMENT_STARTS_H

#include "fast_dft.h"
#include <math.h>

/* The caller has checked 0 <= last_start <= INT_MAX and iterates
 * 0 <= segment_index < segment_count. Keep every previously valid rounded
 * start, including a final start below last_start. Repeated binary64
 * addition can overshoot the intended final endpoint. Repair that formerly
 * rejected position only at the last segment, without changing the recurrence.
 *
 * Check the rounded value before converting to long, which is only 32 bits
 * on Windows. A negative return retains the caller's invalid-start guard.
 */
static LPSD_ALWAYS_INLINE long int lpsd_segment_start(
    double start, long int segment_index, long int segment_count,
    long int last_start)
{
    const double rounded = floor(start + 0.5);
    if (rounded >= 0.0 && rounded <= (double)last_start) {
        return (long int)rounded;
    }
    if (rounded > (double)last_start && segment_index == segment_count - 1 &&
        isfinite(rounded)) {
        return last_start;
    }
    return -1;
}

#endif
