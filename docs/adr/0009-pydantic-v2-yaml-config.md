# ADR-0009: pydantic v2 + YAML with `extra="forbid"` for configuration

## Context

regimekit configuration spans data preparation, model specification, estimation, forecasting, evaluation and
CLI concerns. It needs validation with clear error messages (config path + reason), a stable on-disk format
that is diffable and human-editable, and protection against silent typos in config keys. Hydra was considered
for its composition features but adds a heavier dependency surface and a different mental model (config
groups, defaults lists) than the project needs for a single, mostly-flat configuration tree.

## Decision

Configuration is a pydantic v2 model tree (`config/schema.py`) with `model_config = ConfigDict(extra="forbid")`
throughout, serialized to/from YAML via `ruamel.yaml` (`config/yaml_io.py`), not Hydra.

## Consequences

- Unknown or misspelled config keys are rejected at load time with a field path, rather than silently ignored.
- A JSON-Schema can be exported directly from the pydantic models for editor tooling and the
  `validate-config` CLI command.
- Config composition (overrides, includes) is implemented directly in `config/yaml_io.py` rather than
  inherited from a framework, keeping the dependency footprint (overview §4.1) minimal.

## Status

accepted
