"""Test helpers: load Part I (schema version 1) configuration dicts and files as `FinanceConfig`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from regimekit.config.migrate import migrate
from regimekit.config.schema import parse_project_config
from regimekit.config.yaml_io import load_config
from regimekit.finance import FinanceConfig


def parse_v1(data: dict[str, Any]) -> FinanceConfig:
    """Migrate a Part I configuration dict to version 2 and validate it as a `FinanceConfig`."""
    config = parse_project_config(migrate(data, FinanceConfig), FinanceConfig)
    assert isinstance(config, FinanceConfig)
    return config


def load_finance(path: str | Path) -> FinanceConfig:
    """`load_config(path, FinanceConfig)` with the precise return type."""
    config = load_config(path, FinanceConfig)
    assert isinstance(config, FinanceConfig)
    return config
