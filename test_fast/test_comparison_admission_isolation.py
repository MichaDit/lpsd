"""Guard the tiny-test admission fixture itself, not FFTW numerical correctness."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def fixture_namespace(monkeypatch):
    path = ROOT / "test_fast/test_disk_algorithm_comparison.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == "small_comparison_resources")
    node.decorator_list = []  # Invoke the unchanged fixture body as an ordinary function.
    package = ModuleType("lpsd_benchmark")
    resources = ModuleType("lpsd_benchmark.resources")
    resources.detect_resources = lambda *args, **kwargs: None
    package.resources = resources
    monkeypatch.setitem(sys.modules, "lpsd_benchmark", package)
    monkeypatch.setitem(sys.modules, "lpsd_benchmark.resources", resources)
    calls = []
    pipeline = SimpleNamespace(disk_pipeline=lambda *args: calls.append(args) or "pipeline-result")
    ns = dict(SimpleNamespace=SimpleNamespace, pipeline=pipeline)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    ns["small_comparison_resources"](monkeypatch)
    return pipeline, resources, calls


def test_tiny_fixture_retains_real_pipeline_call_and_deterministic_observation(monkeypatch):
    pipeline, resources, calls = fixture_namespace(monkeypatch)
    assert pipeline.disk_pipeline("fftw", [0] * 4096, None, None,
        SimpleNamespace(workers=1, fft_memory_mb=128.)) == "pipeline-result"
    assert len(calls) == 1
    admitted = resources.detect_resources(None, threads=1)
    assert admitted.ram_budget == admitted.disk_budget == 512 * 1024**2
    assert admitted.threads == 1


@pytest.mark.parametrize("method,n,workers,memory", [
    ("fftw", 8192, 1, 128.), ("matched", 4096, 1, 128.),
    ("fftw", 4096, 2, 128.), ("fftw", 4096, 1, 256.)])
def test_fixture_cannot_accidentally_admit_a_larger_or_different_computation(
        monkeypatch, method, n, workers, memory):
    pipeline, _, calls = fixture_namespace(monkeypatch)
    with pytest.raises(AssertionError):
        pipeline.disk_pipeline(method, [0] * n, None, None,
                               SimpleNamespace(workers=workers, fft_memory_mb=memory))
    assert not calls


def test_numerical_contracts_opt_in_but_resource_refusal_does_not():
    tree = ast.parse((ROOT / "test_fast/test_disk_algorithm_comparison.py").read_text())
    arguments = {n.name: [a.arg for a in n.args.args] for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("test_direct_comparison_gates_actual_float32_outputs_at_3ppm",
                 "test_direct_comparison_rejects_wrong_observed_route"):
        assert "small_comparison_resources" in arguments[name]
    assert "small_comparison_resources" not in arguments["test_direct_comparison_resource_skip_is_not_a_numerical_pass"]
