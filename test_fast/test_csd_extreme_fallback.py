"""Observe the actual CSD route while preserving the production arithmetic."""
import ctypes as ct
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pytest

from lpsd_fast import api
from lpsd_fast.build import build_shared, source_manifest


@pytest.fixture(scope="module")
def csd_route_probe(tmp_path_factory):
    # Insert counters into an isolated copy, without changing any dot,
    # statistics, guard or branch. No public ABI or test hook is needed in
    # the production library. Counts establish that an extreme late input
    # really replays earlier batches and that ordinary inputs still batch.
    native = Path(api.__file__).resolve().parent / "_native" / "fast_dft.c"
    directory = tmp_path_factory.mktemp("csd-route")
    copy_root = directory / "source"
    project_root = native.parents[2]
    original_manifest = source_manifest(native)
    # Keep every included file's bytes and relative project layout. In
    # particular, Windows source and pytest temp may be on different drives;
    # absolute includes back into the checkout would break relative provenance.
    for relative, digest in original_manifest.items():
        original = (native.parent / relative).resolve()
        copied = copy_root / original.relative_to(project_root)
        copied.parent.mkdir(parents=True, exist_ok=True)
        data = original.read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        copied.write_bytes(data)
    source = copy_root / native.relative_to(project_root)
    assert source_manifest(source) == original_manifest
    content = source.read_text(encoding="utf-8")
    insertions = (
        ("if (csd_batch != 0) {", "if (csd_batch != 0) {\n"
         "            ++probe_batch_calls;", 1),
        ("dot_csd_anchored_simd(segment1, segment2,",
         "++probe_single_calls;\n"
         "                dot_csd_anchored_simd(segment1, segment2,", 1),
        ("goto restart_csd_segments;", "++probe_restarts;\n"
         "            goto restart_csd_segments;", 2),
    )
    for original, instrumented, count in insertions:
        assert content.count(original) == count
        content = content.replace(original, instrumented)
    source.write_text(
        "static long int probe_batch_calls, probe_single_calls, probe_restarts;\n" +
        content + r'''
void csd_probe_reset(void)
{
    probe_batch_calls = probe_single_calls = probe_restarts = 0;
}
long int csd_probe_count(int kind)
{
    return kind == 0 ? probe_batch_calls :
           kind == 1 ? probe_single_calls : probe_restarts;
}
int csd_probe_range(double r1, double i1, double r2, double i2)
{
    return csd_projection_has_variance_range(r1, i1, r2, i2);
}
int csd_probe_max_exp(void) { return DBL_MAX_EXP; }
''', encoding="utf-8")
    target = directory / ("probe.dll" if os.name == "nt" else "probe.so")
    build_shared(source, target, simd=True)
    library = ct.CDLL(str(target))
    api._native()
    library.fast_dft.argtypes = api._LIB.fast_dft.argtypes
    library.fast_dft.restype = ct.c_int
    library.fast_dft_selected.argtypes = api._LIB.fast_dft_selected.argtypes
    library.fast_dft_selected.restype = ct.c_int
    library.csd_probe_reset.argtypes = []
    library.csd_probe_reset.restype = None
    library.csd_probe_count.argtypes = [ct.c_int]
    library.csd_probe_count.restype = ct.c_long
    library.csd_probe_range.argtypes = [ct.c_double] * 4
    library.csd_probe_range.restype = ct.c_int
    library.csd_probe_max_exp.argtypes = []
    library.csd_probe_max_exp.restype = ct.c_int
    return library


def _coefficients(length):
    phase = 2 * np.pi * 4.371 * np.arange(length) / length
    window = np.kaiser(length + 1, 23.7)[:-1]
    return window * np.cos(phase), window * np.sin(phase)


def _run(library, first, second, real, imag, *, selected, statistics=True):
    outputs = [ct.c_double() for _ in range(4)]
    count = ct.c_long()
    real, imag = real.copy(), imag.copy()
    arguments = (
        *(ct.byref(value) for value in outputs), ct.byref(count),
        api._pointer(first), api._pointer(second), len(first), len(real),
        api._pointer(real), api._pointer(imag), 75.0, 0, True, 2,
    )
    library.csd_probe_reset()
    if selected:
        status = library.fast_dft_selected(*arguments, statistics, True)
    else:
        status = library.fast_dft(*arguments)
    assert status == 0
    route = tuple(library.csd_probe_count(kind) for kind in range(3))
    return np.asarray([value.value for value in outputs]), count.value, route


@pytest.mark.parametrize("component", range(4))
def test_csd_variance_range_has_an_exponent_boundary(csd_route_probe, component):
    limit = math.ldexp(1.0, csd_route_probe.csd_probe_max_exp() // 4 - 4)
    for value in (0.0, limit, -limit):
        projections = [0.0] * 4
        projections[component] = value
        assert csd_route_probe.csd_probe_range(*projections) == 1
    for value in (math.nextafter(limit, math.inf),
                  math.nextafter(-limit, -math.inf), math.inf, -math.inf, math.nan):
        projections = [0.0] * 4
        projections[component] = value
        assert csd_route_probe.csd_probe_range(*projections) == 0


def test_normal_csd_keeps_batches_without_replaying(csd_route_probe):
    length = 2049
    first, second = np.random.default_rng(79171).normal(size=(2, length * 5))
    real, imag = _coefficients(length)
    expected, count, reference_route = _run(
        csd_route_probe, first, second, real, imag, selected=False)
    actual, actual_count, (batches, singles, replays) = _run(
        csd_route_probe, first, second, real, imag, selected=True)
    assert actual_count == count == 17
    assert reference_route == (0, count, 0)
    assert batches > 0 and 2 <= singles < count and replays == 0
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("position", ("repeated", "late_batch", "last_single"))
@pytest.mark.parametrize("value", (1e80, 1e155, -1e155, 1e308, np.nan, np.inf, -np.inf))
def test_extreme_csd_replays_the_complete_single_sequence(
        csd_route_probe, position, value):
    length = 2049
    first, second = np.random.default_rng(79171).normal(size=(2, length * 5))
    if position == "repeated":
        # The reported physical ARM failure, including its exact geometry.
        first[::11] = value
        second[::13] = value
    else:
        index = 3 * length if position == "late_batch" else len(first) - 1
        first[index] = second[index] = value
    real, imag = _coefficients(length)
    expected, count, _ = _run(
        csd_route_probe, first, second, real, imag, selected=False)
    actual, actual_count, (batches, singles, replays) = _run(
        csd_route_probe, first, second, real, imag, selected=True)
    assert actual_count == count == 17
    assert replays == 1
    # All original segments run again; a late event must also discard the
    # earlier ordinary batches, including their already updated statistics.
    assert singles > count
    if position != "repeated":
        assert batches > 0
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    assert actual[finite].tobytes() == expected[finite].tobytes()


@pytest.mark.parametrize("second_scale", (0.0, 1e-155))
def test_csd_checks_each_projection_even_when_cross_power_is_finite(
        csd_route_probe, second_scale):
    length = 2049
    first, second = np.random.default_rng(67239).normal(size=(2, length * 5))
    first *= 1e155
    second *= second_scale
    real, imag = _coefficients(length)
    expected, count, _ = _run(
        csd_route_probe, first, second, real, imag, selected=False)
    actual, actual_count, (_, singles, replays) = _run(
        csd_route_probe, first, second, real, imag, selected=True)
    spectrum, spectrum_count, (_, _, spectrum_replays) = _run(
        csd_route_probe, first, second, real, imag, selected=True, statistics=False)
    assert actual_count == spectrum_count == count
    assert replays == spectrum_replays == 1 and singles > count
    assert np.isfinite(expected).all()
    assert actual.tobytes() == expected.tobytes()
    assert spectrum[:2].tobytes() == actual[:2].tobytes()
    assert np.isnan(spectrum[2:]).all()
