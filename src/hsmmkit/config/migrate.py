"""`schema_version` migration hook: applies registered `v -> v+1` steps until the config class is current.

Migrations are registered per schema id (`ProjectConfig.SCHEMA_ID`: `"core"` for the core, an application's
own id for its subclass), so the core never sees an application's older layouts (D24).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from hsmmkit.utils.errors import ConfigError

if TYPE_CHECKING:
    from hsmmkit.config.schema import ProjectConfig

CURRENT_SCHEMA_VERSION = 1  # of the core schema (`ProjectConfig.SCHEMA_VERSION`)

MigrationStep = Callable[[dict[str, Any]], dict[str, Any]]

_MIGRATIONS: dict[tuple[str, int], MigrationStep] = {}


def register_migration(schema_id: str, from_version: int) -> Callable[[MigrationStep], MigrationStep]:
    """Register `fn` as the migration of schema `schema_id` from `from_version` to `from_version + 1`."""

    def decorator(fn: MigrationStep) -> MigrationStep:
        _MIGRATIONS[(schema_id, from_version)] = fn
        return fn

    return decorator


def migrate(raw: dict[str, Any], config_cls: type[ProjectConfig] | None = None) -> dict[str, Any]:
    """Apply registered migration steps until `schema_version == config_cls.SCHEMA_VERSION`."""
    schema_id = "core" if config_cls is None else config_cls.SCHEMA_ID
    current = CURRENT_SCHEMA_VERSION if config_cls is None else config_cls.SCHEMA_VERSION
    if "schema_version" not in raw:
        raise ConfigError("config is missing schema_version", path="schema_version")
    version = raw["schema_version"]
    if version > current:
        raise ConfigError(
            f"config newer than library (schema_version={version} > {current})", path="schema_version"
        )
    data = dict(raw)
    while data["schema_version"] < current:
        step = _MIGRATIONS.get((schema_id, data["schema_version"]))
        if step is None:
            raise ConfigError(
                f"no migration registered for schema_version={data['schema_version']}", path="schema_version"
            )
        data = step(data)
    return data
