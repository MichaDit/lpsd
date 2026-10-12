"""Preserve the original residual rounding on the native FP64 long-double ABI."""
import ctypes as ct
import os
from pathlib import Path

import numpy as np
import pytest

from lpsd_fast import api
from lpsd_fast.build import build_shared


def test_inline_mean_reports_actual_use_and_adjusts_only_residual_workspace():
    values = 10 + 1e-9 * np.random.default_rng(835).normal(size=4097)
    api._native()
    supported = bool(api._LIB.native_inline_mean_supported())
    results = {}
    for kernel, order in (('simd', 0), ('scalar', 0), ('simd', 1)):
        result = api.lpsd(values, sample_rate=50, n_frequencies=24, n_averages=8,
                          kernel=kernel, detrending_order=order, workers=1,
                          outputs='psd', window_cache_mb=0, profile=True)
        results[kernel, order] = result
        used = supported and kernel == 'simd' and order == 0
        assert result.attrs['lpsd_fast']['inline_mean_residual'] is used
        rows = result.attrs['lpsd_profile']['frequencies']
        if kernel != 'scalar':
            bytes_per_sample = 24 if used else 32
            expected_peak = max(bytes_per_sample * row['L'] + 8192 for row in rows)
            assert result.attrs['lpsd_profile']['peak_reserved_bytes'] == expected_peak
    np.testing.assert_allclose(results['simd', 0], results['scalar', 0],
                               rtol=2e-6, atol=0)


def test_native_inline_dots_match_original_residual_vectors_bitwise(tmp_path):
    api._native()
    if not api._LIB.native_inline_mean_supported():
        pytest.skip('Native long double differs from FP64; original residual path remains active')
    # Include the unchanged polyreg0 and SIMD reference helpers in the same
    # compiler/ISA context as the candidate. Dot equality also protects the
    # cancellation-sensitive first two updates of the inherited statistics.
    native = Path(api.__file__).parent / '_native' / 'fast_dft.c'
    source = tmp_path / 'residual_rounding.c'
    source.write_text('#include "' + native.as_posix() + '"\n' + r'''
LPSD_TARGET_CLONES int probe_residual_dots(const double *x1, const double *x2,
        const double *cr, const double *ci, long int length, bool csd, double *out) {
    double *residual1 = malloc((size_t)length * sizeof(double));
    double *residual2 = malloc((size_t)length * sizeof(double));
    if (!residual1 || !residual2) { free(residual1); free(residual2); return 3; }
    double coefficients[11];
    polyreg0((double *)x1, (int)length, residual1, coefficients);
    if (csd) {
        polyreg0((double *)x2, (int)length, residual2, coefficients);
        dot_csd_simd(residual1, residual2, cr, ci, length,
                     out, out + 1, out + 2, out + 3);
    } else {
        dot_psd_simd(residual1, cr, ci, length, out, out + 1);
        out[2] = out[0]; out[3] = out[1];
    }
    free(residual1); free(residual2);
#if LPSD_INLINE_MEAN
    if (csd) {
        dot_csd_mean_simd(x1, x2, cr, ci, length,
                          out + 4, out + 5, out + 6, out + 7);
    } else {
        dot_psd_mean_simd(x1, cr, ci, length, out + 4, out + 5);
        out[6] = out[4]; out[7] = out[5];
    }
    return 0;
#else
    return 4;
#endif
}
''')
    target = tmp_path / ('residual_rounding.dll' if os.name == 'nt' else 'residual_rounding.so')
    build_shared(source, target, simd=True)
    probe = ct.CDLL(str(target)).probe_residual_dots
    dp = ct.POINTER(ct.c_double)
    probe.argtypes = [dp, dp, dp, dp, ct.c_long, ct.c_bool, dp]
    probe.restype = ct.c_int
    for length in (2, 3, 4, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                   255, 256, 257, 1023, 1024, 1025, 4097, 65537):
        rng = np.random.default_rng(918 + length)
        cr, ci = rng.normal(size=(2, length))
        for signal in ('white', 'dc_noise', 'transient', 'constant'):
            x, y = rng.normal(size=(2, length))
            if signal == 'dc_noise':
                x = 10 + 1e-9 * x
                y = 3 + 1e-10 * y
            elif signal == 'transient':
                x[0] = 1e12
                y[0] = 1e10
            elif signal == 'constant':
                x.fill(.1)
                y.fill(10)
            for csd in (False, True):
                output = np.empty(8)
                status = probe(api._pointer(x), api._pointer(y), api._pointer(cr),
                               api._pointer(ci), length, csd, api._pointer(output))
                assert status == 0
                assert output[:4].tobytes() == output[4:].tobytes(), (length, signal, csd)
