# Presets and feature gates

## Presets

A *preset* is a validated YAML configuration plus a generator for the synthetic data it refers to. Presets are
a registry kind (`preset`); applications register their own with `hsmmkit.presets.register_preset`.

```bash
hsmmkit presets list
hsmmkit presets show core/segmentation_gaussian_ltr
hsmmkit presets write core/segmentation_gaussian_ltr work/   # YAML + data/sensor_windows.parquet
hsmmkit validate-config work/segmentation_gaussian_ltr.yaml
```

| Preset | What it is |
|---|---|
| `core/segmentation_gaussian_ltr` | The reference configuration of the specification (Appendix H): 50 labelled sessions of 3-dimensional Gaussian windows, left-to-right 4-regime structure, explicit durations per state, GEM, online alerts, sequence cross-validation. |
| `core/two_regime_gaussian` | The two-regime sanity example (calm / turbulent) in Gaussian form on a daily `calendar` grid. |

`core/counts_poisson_hazard` and `core/msar2_hamilton` follow with E16-T05.

## Feature gates

The configuration schema (`schema_version: 1`) already contains every key of the 0.2.0–0.5.0 releases, so
presets and the JSON Schema validate today. A key whose implementation has not landed is rejected when the
model is **built** (`hsmmkit.config.gates.check`), with `ConfigError(path, "not available before <ticket>")`.
`gates.used_gates(cfg)` lists the gated features a configuration uses, in the order of `gates.GATES`.

Each ticket that implements a feature removes its own row from `GATES` and adds a test that the key builds.
Names of families, metrics, inits and sources need no gate: an unregistered name fails validation with the list
of registered names; metrics that are planned but not implemented (`gates.PLANNED_METRICS`) validate and are
rejected at build.

## JSON Schema

`hsmmkit config-schema OUT.json [--config-class dotted.path.Class]` writes the JSON Schema of the
configuration; registry-validated fields (emission and duration families, inits, covariate forecasters,
metrics, sources) get an `enum` of the names registered at export time, including those of installed
extensions.
