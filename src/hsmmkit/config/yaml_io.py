"""Load/save `ProjectConfig` (or an application's subclass) from/to YAML; JSON-Schema export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from hsmmkit.config.migrate import migrate
from hsmmkit.config.schema import ProjectConfig, _registered_names, parse_project_config
from hsmmkit.plugins import REGISTRY

_SAFE_YAML = YAML(typ="safe")
_ROUND_TRIP_YAML = YAML()
_ROUND_TRIP_YAML.default_flow_style = False


def load_config(path: str | Path, config_cls: type[ProjectConfig] = ProjectConfig) -> ProjectConfig:
    """Load a `config_cls` from a YAML file: safe-load -> `migrate()` -> validate.

    Environment variables are **not** expanded (API keys come from the environment at the point of use,
    never from config).
    """
    raw = _SAFE_YAML.load(Path(path))
    migrated = migrate(raw, config_cls)
    return parse_project_config(migrated, config_cls)


def dump_resolved(config: ProjectConfig, path: str | Path) -> None:
    """Write the fully resolved config (every field, including defaults) as YAML to `path`."""
    data = config.model_dump(mode="json")
    with Path(path).open("w", encoding="utf-8") as f:
        _ROUND_TRIP_YAML.dump(data, f)


def _enum_of(schema: dict[str, Any], defs: str, field: str, names: list[str], *, items: bool = False) -> None:
    """Set the enum of `field` in every schema definition named `defs` (or an extension's `<X>defs`)."""
    for name, definition in schema.get("$defs", {}).items():
        prop = definition.get("properties", {}).get(field)
        if prop is None or not name.endswith(defs):
            continue
        target = prop["items"] if items and "items" in prop else prop
        nullable = any(option.get("type") == "null" for option in target.get("anyOf", []))
        target.pop("anyOf", None)
        target["type"] = ["string", "null"] if nullable else "string"
        target["enum"] = [*names, None] if nullable else names


def json_schema(config_cls: type[ProjectConfig] = ProjectConfig) -> dict[str, Any]:
    """`config_cls.model_json_schema()` with the registered names of registry-validated fields as enums."""
    schema = config_cls.model_json_schema()
    _enum_of(schema, "EmissionCfg", "family", _registered_names("emission", needs="from_config"))
    _enum_of(schema, "DurationCfg", "family", _registered_names("duration", needs="from_config"))
    _enum_of(schema, "EstimationSection", "init", _registered_names("init"), items=True)
    _enum_of(schema, "CovariateForecasterCfg", "type", _registered_names("covariate_forecaster"))
    _enum_of(schema, "EvaluationSection", "metrics", _registered_names("metric"), items=True)
    sources = list(REGISTRY.names("source"))
    if sources:
        _enum_of(schema, "SourceCfg", "kind", sources)
    return schema


def export_json_schema(path: str | Path, config_cls: type[ProjectConfig] = ProjectConfig) -> None:
    """Write the JSON-Schema of `config_cls` as JSON to `path` (editor-support hook)."""
    Path(path).write_text(json.dumps(json_schema(config_cls), indent=2), encoding="utf-8")
