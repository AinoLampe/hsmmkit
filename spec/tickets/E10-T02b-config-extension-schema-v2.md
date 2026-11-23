# E10-T02b — Config extension by subclassing, `FinanceConfig`, regimekit `schema_version: 2` with v1 → v2 migration, ADR-0016

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | M | FR-G05, FR-G07, D24 | M7a | Part II §15.4, §15.8 (move map), D24; Part I §5.7, Appendix A |

## Goal
Split the configuration into a domain-neutral core schema and a finance subclass, remove the `profile` key from the core, and migrate every existing finance config automatically.

## Context (self-contained)
**Core models** (`src/regimekit/config/schema.py`, future `hsmmkit.config`): `ProjectConfig(schema_version: int, project, data, model, estimation, forecast, evaluation, output)` — **no `profile`**. Name fields (`emission.family`, `duration.family`, `estimation.init[*]`, `evaluation.metrics[*]`, `forecast.covariate_forecaster.type`, `data.source.kind`, `data.clock`) are `str` validated against `REGISTRY` (E10-T02a) by `field_validator`s; error path = field path, message lists registered names. Core sections keep only neutral fields:
- `DataSection`: `source: SourceCfg`, `clock: str = "steps"`, `clock_options: dict = {}` (validated in E10-T03), `covariates: list[CovariateCfg]`, `preparation: PrepCfg`.
- `SourceCfg`: `kind: str` + `model_config = ConfigDict(extra="allow")`; the extra keys are validated by the options model the source registers (E10-T04a). In this ticket the only kind is `provider` (registered by finance, options model = the Part I `PriceSource` fields).
- `CovariateCfg`: `name, source: SourceCfg | None = None, column: str | None, transform, lag ≥ 1, missing, max_staleness, use_in`.
- `PrepCfg`: `observation_transform: str | None = None`, `standardize_observations: bool = False`, `standardize_covariates: bool = True`, `missing_observations: Literal["mask","drop","error"] = "mask"`.
- `ParamCfg`: `init_mean_sojourn_steps` stays; `init_mean_sojourn_days` leaves the core.
- `BacktestCfg`: `initial_train: float`, `rolling_window: float | None` (clock units), plus `initial_train_steps`/`rolling_window_steps` alternatives; exactly one of each pair.
- `OutputSection.write`: `posterior, forecast, backtest, cv, report` (+ extension values via a registered set).
**Finance subclass** (`src/regimekit/finance/config.py`): `FinanceConfig(ProjectConfig)` with `data: FinanceDataSection` (adds `cache: CacheCfg`; default `clock="trading"`, `preparation.observation_transform="log_returns"`, `missing_observations="drop"`), `ParamCfg` subclass adding `init_mean_sojourn_days`, `evaluation: FinanceEvaluationSection` (adds `strategy: StrategyCfg | None`), `output.write` += `strategy`. Validation rules of Part I that are finance-specific move with their fields (e.g. `_days` vs `steps` clock rule, `bars_per_day`).
**Loader.** `load_config(path, config_cls: type[ProjectConfig] = ProjectConfig)`; finance call sites pass `FinanceConfig`. `migrate()` keyed per config class (`config_cls.SCHEMA_ID`: core `"core"` starts at version 1; finance `"regimekit"` current version 2).
**Finance migration 1 → 2** (`regimekit/finance/config.py::migrate_1_to_2(raw)`): drop `profile`; `data.primary` → `data.source = {kind: provider, **primary}`; each covariate `{provider, series}` → `source: {kind: provider, provider, series}`; `data.bars_per_day` → `data.clock_options.bars_per_day`; `data.preparation.returns: log` → `observation_transform: log_returns`; `missing_returns` → `missing_observations`; `evaluation.backtest.initial_train_years` → `initial_train`, `rolling_years` → `rolling_window` (years under `trading`/`calendar` are clock units); emission `family: student_t` without `scale_by_dt` → `scale_by_dt: true`. (E10-T03, E10-T07a extend this function for their keys.)
**JSON Schema export** (E1-T02b hook) now uses `config_cls`; enums for registry-validated fields are injected via `json_schema_extra` from `REGISTRY.names(kind)` at export time.
**ADR-0016** (`docs/adr/0016-config-extension.md`, accepted): registries + subclassing instead of `profile`; extensions never change core key semantics.

## Creates
- `src/regimekit/config/{schema,migrate,yaml_io}.py` (edited), `src/regimekit/finance/config.py`
- `tests/unit/test_e10_t02b_schema_core.py`, `tests/unit/test_e10_t02b_finance_migration.py`, fixtures `tests/fixtures/configs/v2/appendix_a_v2.yaml`
- `docs/adr/0016-config-extension.md`; neutral-name allow-list entries tagged `E10-T02b` removed

## Acceptance criteria
1. Appendix A (`schema_version: 1`, with `profile: finance`) loads through `FinanceConfig`, is migrated to version 2, equals `appendix_a_v2.yaml` after `model_dump`, and builds a `ModelSpec`/`ParamManager` identical to v0.2.0 (same param specs, same `u` layout).
2. Core `ProjectConfig` rejects `profile`, `data.primary`, `bars_per_day`, `strategy` with `ConfigError` paths; `FinanceConfig` accepts them only through v1 migration or their v2 locations.
3. Unknown `emission.family: foo` → `ConfigError(path="model.emission.family")` listing registered names with provenance.
4. Resolved-config dump of a finance run contains `schema_version: 2` and no `profile`; `config-schema` export contains the registered family names as an enum.
5. v0.2.0 golden values bitwise; allow-list entries tagged `E10-T02b` gone.

## Tests
Parametrized migration tests (each rule), core-vs-finance validation, JSON-schema export, golden test.

## Out of scope
Clock options validation (E10-T03); generic source kinds (E10-T04a); hsmmkit-only keys of Part II §15.4 (E10-T10).

## Connections
- Depends on: E10-T02a
- Blocks: E10-T03, E10-T04a, E10-T06, E10-T07a
- Relates to: E1-T02a/b, E10-T09 (bundle configs), E10-T10
