"""`schema_version` migration hook: applies registered `v -> v+1` steps until current."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from regimekit.utils.errors import ConfigError

CURRENT_SCHEMA_VERSION = 1

MigrationStep = Callable[[dict[str, Any]], dict[str, Any]]

_MIGRATIONS: dict[int, MigrationStep] = {}


def register_migration(from_version: int) -> Callable[[MigrationStep], MigrationStep]:
    """Register `fn` as the migration from `from_version` to `from_version + 1`."""

    def decorator(fn: MigrationStep) -> MigrationStep:
        _MIGRATIONS[from_version] = fn
        return fn

    return decorator


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply registered migration steps until `schema_version == CURRENT_SCHEMA_VERSION`."""
    if "schema_version" not in raw:
        raise ConfigError("config is missing schema_version", path="schema_version")
    version = raw["schema_version"]
    if version > CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            f"config newer than library (schema_version={version} > {CURRENT_SCHEMA_VERSION})",
            path="schema_version",
        )
    data = dict(raw)
    while data["schema_version"] < CURRENT_SCHEMA_VERSION:
        step = _MIGRATIONS.get(data["schema_version"])
        if step is None:
            raise ConfigError(
                f"no migration registered for schema_version={data['schema_version']}",
                path="schema_version",
            )
        data = step(data)
    return data
