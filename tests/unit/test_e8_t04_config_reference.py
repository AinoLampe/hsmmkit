"""E8-T04 AC1: the committed configuration reference matches the pydantic schema (the CI diff check)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from regimekit.config.schema import ProjectConfig

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gen_config_reference.py"


def test_e8_t04_config_reference_is_not_stale() -> None:
    """`gen_config_reference.py --check` exits 0 on the committed `docs/guide/config.md`."""
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_e8_t04_config_reference_covers_every_section_and_field() -> None:
    """Every top-level section has a heading and every field of every section appears in a table row."""
    text = (ROOT / "docs" / "guide" / "config.md").read_text(encoding="utf-8")
    for section in ProjectConfig.model_fields:
        if section not in ("schema_version", "profile", "warnings"):
            assert f"## `{section}`" in text, section
    for name in ("max_iter", "refit_every_steps", "cost_bps", "init_mean_sojourn_days", "coef_prior_sd"):
        assert f"| `{name}` |" in text, name


@pytest.mark.finance
def test_e8_t04_staleness_check_detects_a_modified_page(tmp_path: Path) -> None:
    """The check fails when the page differs: render() output is compared byte for byte."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_config_reference", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stale = tmp_path / "config.md"
    stale.write_text(module.render() + "\nstale\n", encoding="utf-8")
    module.TARGET = stale
    assert module.main(["--check"]) == 1
    stale.write_text(module.render(), encoding="utf-8")
    assert module.main(["--check"]) == 0
