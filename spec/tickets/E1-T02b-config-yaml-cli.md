# E1-T02b — YAML I/O, resolved-config dump, `schema_version` migration hook, `validate-config` CLI

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-C01, FR-C02, NFR-10 | M0 | §5.7, §5.9 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendment 6: `profile` in the resolved dump, JSON-Schema export hook). No behavioural change for the finance profile.

## Goal
Load/save the configuration from YAML, write the fully resolved config, provide a migration hook keyed by `schema_version`, and ship the first CLI command.

## Context (self-contained)
- `config/yaml_io.py`: `load_config(path) -> ProjectConfig` (ruamel.yaml safe load → `migrate()` → `ProjectConfig.model_validate`); `dump_resolved(config, path)` writes all fields including defaults as `config.resolved.yaml` (ruamel round-trip dumper, stable key order). Environment variables are **not** expanded (API keys come from env at provider level, not from config).
- `config/migrate.py`: `CURRENT_SCHEMA_VERSION = 1`; `migrate(raw: dict) -> dict` applies registered step functions `(v → v+1)` until current; missing `schema_version` → `ConfigError`; newer than current → `ConfigError("config newer than library")`.
- `cli/app.py`: typer app with global options `--out DIR`, `--offline`, `--device cpu|gpu|auto`, `--seed N`, `--log-level L`; command `validate-config CONFIG` prints a summary (n_states, engine, families, covariate entry points, warnings) and exits 0, or prints the `ConfigError` and exits 2. `main()` maps `RegimekitError` subclasses to `EXIT_CODES` (E0-T05).
- Overrides: `--seed` and `--device` override config values and are reflected in the resolved dump.
- `profile` is always written to the resolved dump (explicitly `finance` when defaulted), so downstream artefacts (bundle manifest, E8-T01) can copy it without re-validating.
- `config/yaml_io.py::export_json_schema(path: Path) -> None` writes `ProjectConfig.model_json_schema()` as JSON (Part II hook for editor support); CLI command `regimekit config-schema OUT.json`.

## Creates
- `src/regimekit/config/{yaml_io,migrate}.py`, `src/regimekit/cli/app.py` (skeleton with `validate-config`)
- `tests/unit/test_e1_t02b_yaml.py`, `tests/integration/test_e1_t02b_cli.py`

## Acceptance criteria
1. `regimekit validate-config tests/fixtures/configs/appendix_a.yaml` exits 0; an invalid fixture exits 2 with the path in stderr.
2. `load_config(dump_resolved(load_config(x)))` equals `load_config(x)` (model equality).
3. Resolved YAML contains every defaulted field (assert e.g. `estimation.tol_f` present when omitted in input).
4. A fixture with `schema_version: 0` and a registered `0→1` test migration loads; `schema_version: 99` raises `ConfigError`.
5. `--seed 7` appears as `project.seed: 7` in the resolved dump.
6. The resolved dump of Appendix A contains `profile: finance`; `regimekit config-schema out.json` writes a JSON object with `properties.profile` and `properties.data.properties.clock.enum == ["trading", "calendar", "steps"]`.

## Tests
Round-trip, defaults, migration, CLI exit codes (typer `CliRunner`).

## Out of scope
Other CLI commands (added by their own tickets); `Project` facade (E8-T02).

## Connections
- Depends on: E1-T02a
- Blocks: E2-T08, E6-T02, E6-T05, E7-T04, E8-T02
- Relates to: E0-T05 (`EXIT_CODES`)
