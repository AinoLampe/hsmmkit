"""Load/save `ProjectConfig` from/to YAML; JSON-Schema export (Part II editor-support hook)."""

from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML

from regimekit.config.migrate import migrate
from regimekit.config.schema import ProjectConfig, parse_project_config

_SAFE_YAML = YAML(typ="safe")
_ROUND_TRIP_YAML = YAML()
_ROUND_TRIP_YAML.default_flow_style = False


def load_config(path: str | Path) -> ProjectConfig:
    """Load a `ProjectConfig` from a YAML file: safe-load -> `migrate()` -> validate.

    Environment variables are **not** expanded (provider API keys come from the environment at
    the provider level, never from config).
    """
    raw = _SAFE_YAML.load(Path(path))
    migrated = migrate(raw)
    return parse_project_config(migrated)


def dump_resolved(config: ProjectConfig, path: str | Path) -> None:
    """Write the fully resolved config (every field, including defaults) as YAML to `path`."""
    data = config.model_dump(mode="json")
    with Path(path).open("w", encoding="utf-8") as f:
        _ROUND_TRIP_YAML.dump(data, f)


def export_json_schema(path: str | Path) -> None:
    """Write `ProjectConfig.model_json_schema()` as JSON to `path` (Part II editor-support hook)."""
    schema = ProjectConfig.model_json_schema()
    Path(path).write_text(json.dumps(schema, indent=2), encoding="utf-8")
