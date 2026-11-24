"""E1-T02b: YAML I/O round trip, resolved-config defaults, JSON-Schema export, migration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from regimekit.config.migrate import _MIGRATIONS, CURRENT_SCHEMA_VERSION, migrate, register_migration
from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config
from regimekit.finance import FinanceConfig
from regimekit.utils.errors import ConfigError

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "configs"
_YAML = YAML(typ="safe")


# --- AC2: load_config(dump_resolved(load_config(x))) == load_config(x) -----------------------


def test_e1_t02b_round_trip_through_resolved_dump(tmp_path: Path) -> None:
    """Dumping the resolved config and reloading it yields an equal `ProjectConfig`."""
    config = load_config(FIXTURES / "appendix_a.yaml", FinanceConfig)
    resolved_path = tmp_path / "config.resolved.yaml"
    dump_resolved(config, resolved_path)
    reloaded = FinanceConfig.model_validate(_YAML.load(resolved_path))
    assert reloaded == config


# --- AC3: resolved YAML contains every defaulted field ----------------------------------------


def test_e1_t02b_resolved_yaml_contains_defaulted_fields(tmp_path: Path) -> None:
    """A field omitted from the input (`estimation.tol_f`) is present in the resolved dump."""
    config = load_config(FIXTURES / "appendix_a.yaml", FinanceConfig)
    resolved_path = tmp_path / "config.resolved.yaml"
    dump_resolved(config, resolved_path)
    data = _YAML.load(resolved_path)
    assert data["estimation"]["tol_f"] == pytest.approx(1e-9)
    assert "profile" not in data
    assert data["schema_version"] == 2
    assert "warnings" in data


# --- AC6: resolved dump of Appendix A contains profile: finance; JSON-Schema export -----------


def test_e1_t02b_resolved_dump_of_appendix_a_has_schema_version_2_and_no_profile(tmp_path: Path) -> None:
    """The resolved dump of Appendix A contains `schema_version: 2` and no `profile` (E10-T02b AC4)."""
    config = load_config(FIXTURES / "appendix_a.yaml", FinanceConfig)
    resolved_path = tmp_path / "config.resolved.yaml"
    dump_resolved(config, resolved_path)
    data = _YAML.load(resolved_path)
    assert data["schema_version"] == 2
    assert "profile" not in data


def test_e1_t02b_export_json_schema_has_registered_family_enum_and_no_profile(tmp_path: Path) -> None:
    """`config-schema` writes a JSON object without `profile` and with the registered families as an enum."""
    out_path = tmp_path / "schema.json"
    export_json_schema(out_path, FinanceConfig)
    schema = json.loads(out_path.read_text())
    assert "profile" not in schema["properties"]
    assert schema["$defs"]["FinanceEmissionCfg"]["properties"]["family"]["enum"] == ["gbm", "student_t"]
    assert "provider" in schema["$defs"]["SourceCfg"]["properties"]["kind"]["enum"]

    # DataSection is referenced via $ref/$defs (standard pydantic v2 JSON-Schema shape); resolve
    # it to reach `data.clock`'s enum, since a literally inlined `properties.data.properties.clock`
    # path does not exist with pydantic's default (and idiomatic) $defs-based schema generation.
    data_ref = schema["properties"]["data"]["$ref"]
    def_name = data_ref.removeprefix("#/$defs/")
    clock_schema = schema["$defs"][def_name]["properties"]["clock"]
    assert clock_schema["default"] == "trading"


# --- AC4: schema_version migration -------------------------------------------------------------


def test_e1_t02b_missing_schema_version_raises_config_error() -> None:
    """A config with no `schema_version` raises `ConfigError` on path `schema_version`."""
    with pytest.raises(ConfigError) as exc_info:
        migrate({})
    assert exc_info.value.path == "schema_version"


def test_e1_t02b_newer_schema_version_raises_config_error() -> None:
    """`schema_version: 99` (newer than the library) raises `ConfigError`."""
    with pytest.raises(ConfigError) as exc_info:
        migrate({"schema_version": 99})
    assert exc_info.value.path == "schema_version"


def test_e1_t02b_registered_migration_step_runs() -> None:
    """A registered `0 -> 1` migration step runs and the result reaches `CURRENT_SCHEMA_VERSION`."""
    assert ("core", 0) not in _MIGRATIONS, "test relies on no real v0 migration being registered elsewhere"

    @register_migration("core", 0)
    def _v0_to_v1(data: dict[str, object]) -> dict[str, object]:
        data = dict(data)
        data["schema_version"] = 1
        data["migrated_from_v0"] = True
        return data

    try:
        result = migrate({"schema_version": 0})
    finally:
        del _MIGRATIONS[("core", 0)]

    assert result["schema_version"] == CURRENT_SCHEMA_VERSION
    assert result["migrated_from_v0"] is True


def test_e1_t02b_missing_migration_step_raises_config_error() -> None:
    """A `schema_version` below current with no registered step raises `ConfigError`."""
    assert ("core", 0) not in _MIGRATIONS
    with pytest.raises(ConfigError) as exc_info:
        migrate({"schema_version": 0})
    assert exc_info.value.path == "schema_version"


# --- load_config: safe-load -> migrate -> validate --------------------------------------------


def test_e1_t02b_load_config_appendix_a() -> None:
    """`load_config` loads Appendix A end to end."""
    config = load_config(FIXTURES / "appendix_a.yaml", FinanceConfig)
    assert config.model.n_states == 3
    assert config.warnings == []
