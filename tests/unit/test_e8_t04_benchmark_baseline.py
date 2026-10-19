"""E8-T04 follow-up: the CI benchmark baseline is the mean over several runner results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module() -> object:
    spec = importlib.util.spec_from_file_location(
        "make_benchmark_baseline", ROOT / "scripts" / "make_benchmark_baseline.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bench(path: Path, means: dict[str, float]) -> Path:
    path.write_text(json.dumps({"benchmarks": [{"name": n, "stats": {"mean": m}} for n, m in means.items()]}))
    return path


def test_e8_t04_baseline_averages_runs_and_keeps_only_common_benchmarks(tmp_path: Path) -> None:
    """The entry is the mean of the per-run means; a benchmark missing from one run is left out."""
    module = _module()
    one = _bench(tmp_path / "a.json", {"x": 1.0, "y": 4.0, "only_a": 9.0})
    two = _bench(tmp_path / "b.json", {"x": 3.0, "y": 4.0})
    built = module.build([one, two])  # type: ignore[attr-defined]
    assert built == [{"name": "x", "mean": 2.0}, {"name": "y", "mean": 4.0}]


def test_e8_t04_committed_baseline_has_the_simplified_format() -> None:
    """`compare_benchmarks.py` reads `[{name, mean}]` entries with positive means."""
    entries = json.loads((ROOT / "benchmarks" / "baseline.json").read_text())
    assert entries
    assert all(set(e) == {"name", "mean"} and e["mean"] > 0 for e in entries)
