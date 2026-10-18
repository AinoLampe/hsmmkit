"""E0-T03: `scripts/compare_benchmarks.py` regression-threshold logic (AC3)."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_benchmarks.py"


def _write_inputs(tmp_path: Path, *, baseline_mean: float, current_mean: float) -> tuple[Path, Path]:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([{"name": "test_forward", "mean": baseline_mean}]))
    current_path = tmp_path / "bench.json"
    current_path.write_text(
        json.dumps({"benchmarks": [{"name": "test_forward", "stats": {"mean": current_mean}}]})
    )
    return current_path, baseline_path


def _run(current_path: Path, baseline_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(current_path), str(baseline_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_e0_t03_compare_benchmarks_fails_on_31_percent_regression(tmp_path: Path) -> None:
    """A 31% slowdown vs. baseline exits non-zero."""
    current_path, baseline_path = _write_inputs(tmp_path, baseline_mean=1.0, current_mean=1.31)
    result = _run(current_path, baseline_path)
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_e0_t03_compare_benchmarks_passes_on_29_percent_regression(tmp_path: Path) -> None:
    """A 29% slowdown vs. baseline exits zero."""
    current_path, baseline_path = _write_inputs(tmp_path, baseline_mean=1.0, current_mean=1.29)
    result = _run(current_path, baseline_path)
    assert result.returncode == 0
    assert "FAIL" not in result.stdout


def test_e0_t03_compare_benchmarks_empty_baseline_never_fails(tmp_path: Path) -> None:
    """An empty baseline (the E0-T03 state, before E4-T07 fills it in) never fails the comparison."""
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("[]")
    current_path = tmp_path / "bench.json"
    current_path.write_text(json.dumps({"benchmarks": [{"name": "test_forward", "stats": {"mean": 5.0}}]}))
    result = _run(current_path, baseline_path)
    assert result.returncode == 0
    assert "no baseline, skipped" in result.stdout
