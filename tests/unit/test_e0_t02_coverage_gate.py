"""E0-T02 AC4: `scripts/check_package_coverage.py` fails when a package is below 85%."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_package_coverage.py"


def _write_coverage_json(path: Path, *, model_percent: float) -> None:
    covered = round(model_percent)
    files = {
        "src/regimekit/model/spec.py": {"summary": {"covered_lines": covered, "num_statements": 100}},
        "src/regimekit/engine/base.py": {"summary": {"covered_lines": 100, "num_statements": 100}},
    }
    (path / "coverage.json").write_text(json.dumps({"files": files}))


def test_e0_t02_coverage_gate_fails_below_threshold(tmp_path: Path) -> None:
    """A package below 85% coverage makes the script exit non-zero."""
    _write_coverage_json(tmp_path, model_percent=50.0)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert result.returncode == 1
    assert "model" in result.stdout
    assert "FAIL" in result.stdout


def test_e0_t02_coverage_gate_passes_above_threshold(tmp_path: Path) -> None:
    """All tracked packages at or above 85% coverage exits zero."""
    _write_coverage_json(tmp_path, model_percent=90.0)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert result.returncode == 0
    assert "FAIL" not in result.stdout
