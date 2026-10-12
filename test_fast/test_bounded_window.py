"""Bounded-window arithmetic and ownership gates; no large FFT is constructed.

Reference windows and preprocessing come from the existing full-window paths.
The billion-length admission case constructs only a 256-sample helper block.
"""
from __future__ import annotations

import gc
import math
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from benchmarks import _bounded_window as bounded_module
from benchmarks._bounded_window import BoundedWindow
from benchmarks._fftw_native import FFTWOperations
from benchmarks.bench_fftw import KaiserWindow
from benchmarks.matched_smoothing import _periodic_tukey
from lpsd_fast import api


BETA = 25.402566024390598


@pytest.fixture(scope="module")
def native():
    api._native()
    return KaiserWindow("native", api._LIB._name), FFTWOperations("native", api._LIB)


def _fake_native_provider():
    """Scalar preflight/metadata fixture; its numerical functions are unused."""
    lib = SimpleNamespace(window_sums=lambda *args: 0,
                          generate_kaiser_series=lambda *args: 0,
                          generate_kaiser_coefficients_range=lambda *args: 0)
    return SimpleNamespace(lib=lib, metadata={"backend": "lpsd_fast_native_series"},
                           sums=lambda values: (0., 0.))


def _check_window_and_prepare(native, n, kind, beta=BETA, fraction=.05, block=128):
    provider, operations = native
    expected_window = (provider.generate(n, beta) if kind == "kaiser" else
                       _periodic_tukey(n, fraction) if kind == "tukey" else
                       np.ones(n, dtype=np.float64))
    helper = BoundedWindow(n, beta, provider, kind=kind,
                           taper_fraction=fraction, block_bins=block)
    out = np.empty(n, dtype=np.float64)
    try:
        assert helper.fill_into(out) is out
        np.testing.assert_array_equal(out, expected_window)
        assert provider.sums(out) == provider.sums(expected_window)
        # Strided source plus large DC verifies unchanged caller ownership and
        # the anchored mean that protects nanovolt-scale variation.
        backing = np.random.default_rng(4711 + n).normal(size=2 * n) * 1e-9 + 7.
        source = backing[::2]
        before = source.copy()
        expected = source - source[0]
        expected_mean = float(np.mean(expected))
        operations.center_window(expected, expected_window, expected_mean)
        seen = []

        class ObservedOperations:
            def center_window(self, data, window, mean):
                seen.append((len(data), len(window), mean))
                return operations.center_window(data, window, mean)

        actual_mean = helper.prepare_into(source, out, ObservedOperations())
        assert actual_mean == expected_mean
        np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(source, before)
        assert seen and all(a == b and 0 < a <= helper.block_samples and mean == expected_mean
                            for a, b, mean in seen)
        # Every sample must receive exactly one elementwise center/window pass.
        assert sum(size for size, _, _ in seen) == n
    finally:
        helper.close()


@pytest.mark.parametrize("n,beta,block", [
    (2048, BETA, 128), (2051, BETA, 256), (8193, 32., 384), (65539, BETA, 4096),
])
def test_kaiser_matches_original_window_sums_and_full_anchored_prepare(native, n, beta, block):
    _check_window_and_prepare(native, n, "kaiser", beta=beta, block=block)


@pytest.mark.parametrize("n,fraction,block", [
    (n, fraction, 128) for n in (2, 3, 257, 4099) for fraction in (0., .05, 1.)
] + [(65539, .05, 256), (4099, .001, 256)])
def test_tukey_matches_original_periodic_mirrors_and_prepare(native, n, fraction, block):
    _check_window_and_prepare(native, n, "tukey", fraction=fraction, block=block)


@pytest.mark.parametrize("n,fraction", [
    (100, np.float32(.1)), (100, np.float64(.1)), (8197, np.float16(1.)),
])
def test_tukey_retains_original_scalar_precision_and_overlap_precedence(native, n, fraction):
    _check_window_and_prepare(native, n, "tukey", fraction=fraction, block=128)


@pytest.mark.parametrize("n", [2, 257, 4099])
def test_rectangular_matches_original_prepare(native, n):
    _check_window_and_prepare(native, n, "rectangular")


def test_mean_is_called_once_on_the_complete_anchored_output(native, monkeypatch):
    provider, operations = native
    n = 4099
    source = np.random.default_rng(91).normal(size=n)
    out = np.empty(n)
    helper = BoundedWindow(n, BETA, provider, kind="tukey", block_bins=256)
    original_mean, observed = np.mean, []

    def mean(values, *args, **kwargs):
        observed.append((values.shape, values.ctypes.data))
        np.testing.assert_array_equal(values, source - source[0])
        return original_mean(values, *args, **kwargs)

    monkeypatch.setattr(bounded_module.np, "mean", mean)
    try:
        helper.prepare_into(source, out, operations)
        assert observed == [((n,), out.ctypes.data)]
    finally:
        helper.close()


@pytest.mark.parametrize("kind,multiplier", [("kaiser", 8), ("tukey", 32), ("rectangular", 8)])
def test_billion_length_budget_only_allocates_a_small_helper_block(monkeypatch, kind, multiplier):
    original_empty, allocations = np.empty, []

    def empty(shape, *args, **kwargs):
        allocations.append(shape)
        assert shape == 256
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(bounded_module.np, "empty", empty)
    helper = BoundedWindow(1_000_000_000, BETA, _fake_native_provider(),
                           kind=kind, block_bins=256)
    try:
        assert allocations == [256]
        assert helper.metadata["block_samples"] == 256
        assert helper.metadata["persistent_scratch_bytes"] == 8 * 256
        assert helper.metadata["scratch_bytes"] == multiplier * 256
        assert helper.metadata["window_buffer_bytes"] == 0
        assert helper.metadata["window_storage"] == "fft_input_then_blocked"
        assert [value.nbytes for value in vars(helper).values() if isinstance(value, np.ndarray)] == [8 * 256]
    finally:
        helper.close()


@pytest.mark.parametrize("kind", ["kaiser", "tukey", "rectangular"])
@pytest.mark.parametrize("missing", ["lib", "window_sums", "native_metadata", "sums"])
def test_every_window_kind_rejects_non_native_sums_before_scratch(monkeypatch, kind, missing):
    provider = _fake_native_provider()
    if missing == "lib":
        provider.lib = None
    elif missing == "window_sums":
        del provider.lib.window_sums
    elif missing == "native_metadata":
        provider.metadata["backend"] = "numpy"
    else:
        provider.sums = None

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid provider allocated scratch")

    monkeypatch.setattr(bounded_module.np, "empty", forbidden)
    with pytest.raises(ValueError, match="native"):
        BoundedWindow(4096, BETA, provider, kind=kind)


@pytest.mark.parametrize("settings", [
    {"n": 1}, {"n": 2.5}, {"n": True}, {"n": 2_147_483_648},
    {"n": 1024}, {"beta": 0.}, {"beta": -1.}, {"beta": 33.},
    {"beta": float("nan")}, {"taper_fraction": -1.},
    {"taper_fraction": float("inf")}, {"block_bins": 0},
    {"block_bins": 257}, {"block_bins": 1_048_577}, {"kind": "unknown"},
])
def test_unsupported_parameters_fail_before_scratch(monkeypatch, settings):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid request allocated scratch")

    monkeypatch.setattr(bounded_module.np, "empty", forbidden)
    options = dict(n=4096, beta=BETA, provider=_fake_native_provider())
    options.update(settings)
    with pytest.raises(ValueError):
        BoundedWindow(**options)


@pytest.mark.parametrize("symbol", ["generate_kaiser_series", "generate_kaiser_coefficients_range"])
def test_missing_kaiser_symbols_fail_before_scratch(monkeypatch, symbol):
    provider = _fake_native_provider()
    delattr(provider.lib, symbol)

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing ABI allocated scratch")

    monkeypatch.setattr(bounded_module.np, "empty", forbidden)
    with pytest.raises(ValueError, match="generation"):
        BoundedWindow(4096, BETA, provider)


def test_nonfinite_low_precision_tukey_ramp_fails_before_scratch(monkeypatch):
    n, fraction = 1_000_000, np.float16(.1)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            half_ramp = float(n * fraction / 2)
    except (OverflowError, FloatingPointError):
        half_ramp = float("inf")
    if math.isfinite(half_ramp):
        pytest.skip("This NumPy version promotes the scalar and avoids float16 overflow")

    def forbidden(*args, **kwargs):
        raise AssertionError("Nonfinite scalar ramp allocated scratch")

    monkeypatch.setattr(bounded_module.np, "empty", forbidden)
    with pytest.raises(ValueError, match="half_ramp"):
        BoundedWindow(n, BETA, _fake_native_provider(), kind="tukey", taper_fraction=fraction)


def test_no_caller_or_output_retention_and_idempotent_close(native):
    provider, operations = native
    helper = BoundedWindow(2051, BETA, provider, block_bins=128)
    source, out = np.arange(2051, dtype=np.float64), np.empty(2051)
    source_ref, out_ref, scratch_ref = weakref.ref(source), weakref.ref(out), weakref.ref(helper._scratch)
    helper.fill_into(out)
    helper.prepare_into(source, out, operations)
    del source, out
    gc.collect()
    assert source_ref() is None and out_ref() is None
    capacity = helper.metadata["scratch_bytes"]
    helper.close()
    helper.close()
    gc.collect()
    assert scratch_ref() is None
    assert helper.metadata["closed"] is True and helper.metadata["scratch_bytes"] == capacity
    with pytest.raises(RuntimeError, match="closed"):
        helper.fill_into(np.empty(2051))


def test_invalid_or_aliasing_arrays_are_rejected_before_overwrite(native):
    provider, operations = native
    helper = BoundedWindow(257, BETA, provider, kind="rectangular", block_bins=128)
    source, out = np.arange(257, dtype=np.float64), np.full(257, 777.)
    try:
        with pytest.raises(ValueError, match="overlap"):
            helper.prepare_into(source, source, operations)
        np.testing.assert_array_equal(source, np.arange(257, dtype=np.float64))
        with pytest.raises(ValueError, match="Samples"):
            helper.prepare_into(source.astype(np.float32), out, operations)
        np.testing.assert_array_equal(out, 777.)
        for invalid in (np.empty(514)[::2], np.empty(257, dtype=np.float32)):
            with pytest.raises(ValueError, match="Output"):
                helper.fill_into(invalid)
    finally:
        helper.close()
