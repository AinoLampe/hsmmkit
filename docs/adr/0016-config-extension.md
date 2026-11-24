# ADR-0016: Registries and config subclassing instead of the `profile` key

## Context

Part I carried a `profile` key (`core` or `finance`) in the configuration: only `finance` could run, and every
finance field (`data.primary`, `bars_per_day`, `strategy`, ...) lived in the one schema. The core is to become a
general library (ADR-0014), so its configuration must not know finance, and applications must be able to add
their own components and keys.

## Decision

- The core `ProjectConfig` has no `profile` and only neutral fields. Name fields (`emission.family`,
  `duration.family`, `estimation.init[*]`, `evaluation.metrics[*]`, `forecast.covariate_forecaster.type`,
  `data.source.kind`, `data.clock`) are strings validated against the component registries of ADR-0016's
  companion ticket (E10-T02a); the error carries the field path and lists the registered names.
- An application extends the configuration by subclassing the sections it needs (`FinanceConfig`,
  `FinanceDataSection`, ...) and sets `SCHEMA_ID` and `SCHEMA_VERSION`. Migrations are registered per schema id.
  `load_config(path, config_cls)` takes the class; the finance call sites pass `FinanceConfig`.
- The finance schema is version 2. Version-1 configurations (with `profile`, `data.primary`, ...) migrate
  automatically (`migrate_1_to_2`).
- Extensions never change the meaning of a core key (D24).

## Consequences

- The JSON Schema export is generated from the class and gets the registered names as enums at export time.
- Bundles and tools that load a configuration must be told the class (`ModelStore.load(path, config_cls)`),
  until the bundle format records the application (E10-T07a).
- `profile` disappears from resolved dumps; the bundle manifest keeps a `profile` entry until bundle v2.

## Status

accepted
