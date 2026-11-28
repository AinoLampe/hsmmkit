"""E14-T05: the `estimation.diagnostics` / `ppc_replicates` config fields."""

from __future__ import annotations

from pathlib import Path

import pytest

from hsmmkit.config.schema import parse_project_config
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw


def test_e14_t05_diagnostics_default_and_validation(tmp_path: Path) -> None:
    """Default `[pseudo_residuals]`, 200 replicates; unknown names and `ppc_replicates < 2` are rejected."""
    cfg = parse_project_config(core_raw(tmp_path))
    assert cfg.estimation.diagnostics == ["pseudo_residuals"]
    assert cfg.estimation.ppc_replicates == 200
    assert "diagnostics" in cfg.model_dump()["estimation"]
    raw = core_raw(tmp_path)
    raw["estimation"]["diagnostics"] = ["pseudo_residuals", "ppc"]
    assert parse_project_config(raw).estimation.diagnostics == ["pseudo_residuals", "ppc"]
    raw["estimation"]["diagnostics"] = ["qq"]
    with pytest.raises(ConfigError):
        parse_project_config(raw)
    raw["estimation"]["diagnostics"] = []
    raw["estimation"]["ppc_replicates"] = 1
    with pytest.raises(ConfigError):
        parse_project_config(raw)
