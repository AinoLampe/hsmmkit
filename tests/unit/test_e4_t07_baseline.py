"""E4-T07: the committed benchmark baseline covers every benchmark and works with the comparison script."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _load_compare_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compare_benchmarks", ROOT / "scripts" / "compare_benchmarks.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _benchmark_test_names() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "tests" / "benchmark").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name.startswith("test_")
                and any(arg.arg == "benchmark" for arg in node.args.args)
            ):
                names.add(node.name)
    return names


def test_e4_t07_baseline_has_an_entry_for_every_benchmark() -> None:
    """AC1: `benchmarks/baseline.json` is populated (no benchmark is silently unguarded)."""
    compare = _load_compare_module()
    baseline = compare.load_baseline(ROOT / "benchmarks" / "baseline.json")
    names = _benchmark_test_names()
    assert names, "no benchmark tests found"
    assert names <= set(baseline), sorted(names - set(baseline))
    assert all(mean > 0 for mean in baseline.values())


def test_e4_t07_comparison_passes_on_baseline_and_fails_on_a_31_percent_regression() -> None:
    """AC1: the comparison script accepts the baseline itself and rejects a > 30 % slowdown."""
    compare = _load_compare_module()
    baseline = compare.load_baseline(ROOT / "benchmarks" / "baseline.json")
    ok, _ = compare.compare(baseline, dict(baseline))
    assert ok
    slower = {name: mean * 1.31 for name, mean in baseline.items()}
    ok, lines = compare.compare(baseline, slower)
    assert not ok
    assert all(line.endswith("FAIL") for line in lines)
