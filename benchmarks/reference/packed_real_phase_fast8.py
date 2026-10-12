# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen fast.8 packed reconstruction method for paired benchmark evidence.

Only _tile differs from the current implementation. Planning, transform,
buffers, input, normalization, outputs and cleanup use the current pipeline.
The method below is copied verbatim from the named source commit.
"""
import numpy as np

from benchmarks._fftw_packed_real import PackedRealDiskFFT

REFERENCE_COMMIT = "b8c500217f7c0ad996f55b039bbf09d685800614"
REFERENCE_FILE = "benchmarks/_fftw_packed_real.py"
REFERENCE_METHOD_SHA256 = "8bb729171b0bf4c6a2e300fc6e1882172b8f58b039e4f29a40424c099578f938"


class LegacyPhasePackedRealDiskFFT(PackedRealDiskFFT):
    def _tile(self, output, first_row, row_count, first_column, column_count, scale):
        """Read both source rectangles before modifying any arithmetic view."""
        core = self._core
        capacity = self.memory["reconstruction_elements"]
        count = row_count * column_count
        if not 1 <= count <= capacity:
            raise AssertionError("Partner tile exceeds the admitted reused scratch")
        left = core._work[:count].reshape(row_count, column_count)
        right = core._work[capacity:capacity + count].reshape(row_count, column_count)
        difference = core._twiddle.reshape(-1)[:count].reshape(row_count, column_count)
        matrix = core.matrix
        np.copyto(left, matrix[first_row:first_row + row_count,
                               first_column:first_column + column_count])
        if first_row == 0:
            if row_count != 1 or first_column < 1:
                raise AssertionError("DC is handled separately; row-zero partner wraps at DC")
            partner = matrix[0:1, self.b - first_column - column_count + 1:
                                     self.b - first_column + 1]
            np.copyto(right, partner[:, ::-1])
        else:
            partner = matrix[self.a - first_row - row_count + 1:self.a - first_row + 1,
                             self.b - first_column - column_count:self.b - first_column]
            np.copyto(right, partner[::-1, ::-1])
        np.conjugate(right, out=right)
        np.subtract(left, right, out=difference)
        np.add(left, right, out=left)
        # Reuse the now-dead partner buffer for W_N^k. Integers <= M are
        # represented exactly; no M-based frequency or power normalization.
        for column in range(column_count):
            np.add(core._k2[first_row:first_row + row_count],
                   float(self.a * (first_column + column)), out=right.imag[:, column])
        right.imag *= -2.0 * np.pi / self.n
        right.real.fill(0.0)
        np.exp(right, out=right)
        np.multiply(difference, right, out=difference)
        difference *= -1j
        np.add(left, difference, out=left)
        left *= 0.5
        # W is dead; its two real halves hold powers and imag-square scratch.
        doubles = right.reshape(-1).view(np.float64)
        power = doubles[:count].reshape(row_count, column_count)
        scratch = doubles[count:2 * count].reshape(row_count, column_count)
        np.square(left.real, out=power)
        np.square(left.imag, out=scratch)
        np.add(power, scratch, out=power)
        np.multiply(power, scale, out=power)
        np.copyto(output[first_column:first_column + column_count,
                         first_row:first_row + row_count], power.T)

