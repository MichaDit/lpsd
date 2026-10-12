/* Recovery revision R3: independently readable post-FFT group reduction.
 *
 * This is NEW source, not a reconstruction of the lost R2 C file.  It reads
 * the unchanged exact-N E FFT matrix Y[r,c], k=r+A*c, and returns bounded
 * log-group sums plus an independent complete one-sided energy sum.
 * No allocation, file operation, worker, FFT, window, or matrix write.
 *
 * Included in the native library with strict floating-point flags. No
 * fast-math, reassociation, or multiply/add contraction is permitted.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

_Static_assert(sizeof(double) == 8, "R3 requires eight-byte double");
_Static_assert(sizeof(int64_t) == 8, "R3 requires eight-byte int64_t");
_Static_assert(sizeof(uint64_t) == 8, "R3 requires eight-byte uint64_t");

#define R3_MAX_N UINT64_C(1000000000)
#define R3_MAX_GROUPS UINT64_C(65536)
#define R3_PARTIAL_BINS UINT64_C(256)

enum { R3_OK = 0, R3_ARGUMENT = 1, R3_ALIAS = 2,
       R3_NONFINITE = 3, R3_COUNT = 4, R3_ROUNDING = 5 };

static int span(const void *pointer, uint64_t bytes,
                uintptr_t *first, uintptr_t *stop)
{
    uintptr_t address = (uintptr_t)pointer;
    if (pointer == NULL || bytes > (uint64_t)UINTPTR_MAX
            || address > UINTPTR_MAX - (uintptr_t)bytes)
        return 0;
    *first = address;
    *stop = address + (uintptr_t)bytes;
    return 1;
}

static int disjoint(uintptr_t a, uintptr_t b, uintptr_t c, uintptr_t d)
{
    return b <= c || d <= a;
}

/* Neumaier compensated merge; the two doubles have sum+correction meaning.
 * A sequential double partial contains at most 256 original bin powers.
 * This order is explicitly different from a frequency-ordered NumPy sum.
 */
static int merge(double *sum, double *correction, double partial)
{
    double previous = *sum;
    double joined = previous + partial;
    double extra;
    if (fabs(previous) >= fabs(partial))
        extra = (previous - joined) + partial;
    else
        extra = (partial - joined) + previous;
    *correction = *correction + extra;
    *sum = joined;
    if (!isfinite(*sum) || !isfinite(*correction))
        return R3_NONFINITE;
    return R3_OK;
}

/* The elementary operations deliberately match E's original power path.
 * Endpoints are halved AFTER the common one-sided scale multiplication.
 */
static int power(const double *matrix, uint64_t offset, uint64_t k,
                 uint64_t n, double scale, double *result)
{
    double real = matrix[2 * offset];
    double imag = matrix[2 * offset + 1];
    double real_square = real * real;
    double imag_square = imag * imag;
    double value = (real_square + imag_square) * scale;
    if (k == 0 || ((n % 2) == 0 && k == n / 2))
        value = value * 0.5;
    if (!isfinite(value) || value < 0.0)
        return R3_NONFINITE;
    *result = value;
    return R3_OK;
}

/* All scalars have already been bounded by R3_MAX_N, so addition below
 * cannot overflow uint64_t. The result is clipped to the matrix row.
 */
static uint64_t column_start(uint64_t cut, uint64_t row, uint64_t a, uint64_t b)
{
    uint64_t column = cut > row ? (cut - row + a - 1) / a : 0;
    return column < b ? column : b;
}

static int accumulate(const double *matrix, uint64_t n, uint64_t a, uint64_t b,
                      uint64_t row, uint64_t first, uint64_t stop, double scale,
                      double *sum, double *correction)
{
    uint64_t base = row * b;
    double partial = 0.0;
    uint64_t partial_count = 0;
    for (uint64_t column = first; column < stop; ++column) {
        double value;
        int status = power(matrix, base + column, row + a * column,
                           n, scale, &value);
        if (status != R3_OK)
            return status;
        partial = partial + value;
        ++partial_count;
        if (partial_count == R3_PARTIAL_BINS) {
            status = merge(sum, correction, partial);
            if (status != R3_OK)
                return status;
            partial = 0.0;
            partial_count = 0;
        }
    }
    if (partial_count != 0)
        return merge(sum, correction, partial);
    return R3_OK;
}

uint64_t dlp_r3_abi_version(void)
{
    return UINT64_C(3);
}

/* state: 2*(groups+1) doubles, groups first, complete energy last.
 * counts: groups+3 uint64_t values: group bins, energy bins, DC, Nyquist.
 * The caller initializes both arrays to zero and visits disjoint row blocks.
 * On any nonzero return, partial state is invalid and must be discarded.
 */
int dlp_r3_reduce_rows(const double *matrix, uint64_t n, uint64_t a, uint64_t b,
                       uint64_t row_stride_complex, uint64_t first_row, uint64_t rows,
                       const int64_t *cuts, uint64_t groups, double scale,
                       double *state, uint64_t *counts)
{
    if (n < 2 || n > R3_MAX_N || a == 0 || b == 0
            || a > R3_MAX_N / b || a * b != n || row_stride_complex != b
            || first_row >= a || rows == 0 || rows > a - first_row
            || groups == 0 || groups > R3_MAX_GROUPS || groups > n / 2
            || !isfinite(scale) || scale <= 0.0
            || matrix == NULL || cuts == NULL || state == NULL || counts == NULL)
        return R3_ARGUMENT;
    if (fegetround() != FE_TONEAREST)
        return R3_ROUNDING;
    if ((uintptr_t)matrix % _Alignof(double) != 0
            || (uintptr_t)cuts % _Alignof(int64_t) != 0
            || (uintptr_t)state % _Alignof(double) != 0
            || (uintptr_t)counts % _Alignof(uint64_t) != 0)
        return R3_ARGUMENT;
    uintptr_t lo[4], hi[4];
    if (!span(matrix, 16 * n, &lo[0], &hi[0])
            || !span(cuts, 8 * (groups + 1), &lo[1], &hi[1])
            || !span(state, 16 * (groups + 1), &lo[2], &hi[2])
            || !span(counts, 8 * (groups + 3), &lo[3], &hi[3]))
        return R3_ARGUMENT;
    for (unsigned i = 0; i < 4; ++i)
        for (unsigned j = i + 1; j < 4; ++j)
            if (!disjoint(lo[i], hi[i], lo[j], hi[j]))
                return R3_ALIAS;
    if (cuts[0] != 1 || cuts[groups] != (int64_t)(n / 2 + 1))
        return R3_ARGUMENT;
    for (uint64_t group = 0; group < groups; ++group) {
        if (cuts[group] < 1 || cuts[group + 1] <= cuts[group]
                || cuts[group + 1] > (int64_t)(n / 2 + 1))
            return R3_ARGUMENT;
        if (counts[group] > (uint64_t)(cuts[group + 1] - cuts[group]))
            return R3_COUNT;
    }
    if (counts[groups] > n / 2 + 1 || counts[groups + 1] > 1
            || counts[groups + 2] > (uint64_t)((n % 2) == 0))
        return R3_COUNT;
    for (uint64_t i = 0; i < 2 * (groups + 1); ++i)
        if (!isfinite(state[i]))
            return R3_NONFINITE;

    for (uint64_t row = first_row; row < first_row + rows; ++row) {
        /* Independent energy traversal includes DC and every original
         * nonnegative bin; it is not a sum of rounded/grouped PSD outputs.
         */
        uint64_t energy_stop = column_start(n / 2 + 1, row, a, b);
        if (energy_stop > n / 2 + 1 - counts[groups])
            return R3_COUNT;
        int status = accumulate(matrix, n, a, b, row, 0, energy_stop,
                                scale, &state[2 * groups], &state[2 * groups + 1]);
        if (status != R3_OK)
            return status;
        counts[groups] += energy_stop;
        if (row == 0 && energy_stop != 0) {
            if (counts[groups + 1] != 0)
                return R3_COUNT;
            counts[groups + 1] = 1;
        }
        if ((n % 2) == 0 && row == (n / 2) % a && energy_stop != 0) {
            if (counts[groups + 2] != 0)
                return R3_COUNT;
            counts[groups + 2] = 1;
        }
        for (uint64_t group = 0; group < groups; ++group) {
            uint64_t first = column_start((uint64_t)cuts[group], row, a, b);
            uint64_t stop = column_start((uint64_t)cuts[group + 1], row, a, b);
            uint64_t amount = stop - first;
            uint64_t maximum = (uint64_t)(cuts[group + 1] - cuts[group]);
            if (amount > maximum - counts[group])
                return R3_COUNT;
            status = accumulate(matrix, n, a, b, row, first, stop,
                                scale, &state[2 * group], &state[2 * group + 1]);
            if (status != R3_OK)
                return status;
            counts[group] += amount;
        }
    }
    return R3_OK;
}
