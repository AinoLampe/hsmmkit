# E11-T09 — Correlated multi-primary finance configuration (`gbm` with `covariance: full`)

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast | S | FR-M12, FR-EM06 | M8 (`regimekit` 0.4.0) | Part II §14.2 (GBM = Gaussian), Appendix F; Part I E9-T01 (multi-primary data layer) |

## Goal
Let `regimekit` fit one shared regime process to several instruments with correlated returns, using `hsmmkit`'s full-covariance Gaussian through the `gbm` alias.

## Context (self-contained)
- `gbm` alias (E10-T06) gains option `covariance: diag | full`: `full` maps to `hsmmkit` `gaussian(scale_by_dt=True, ito_correction=True, covariance="full")` with finance spec names `emission.mu (N, M)` and `emission.chol` (no `sigma` in full mode; config `params.sigma` with `covariance: full` → `ConfigError` explaining `chol` + optional `lkj` prior). Itô term uses `diag(Σ_i)`.
- Data: the E9-T01 multi-primary layer (`data.source.symbols` with `M > 1`, master grid from the first symbol, others aligned by timestamp, missing → `mask=False`) is unchanged; with a joint full block a row with any missing series is masked as a whole (E11-T08 rule) — the quality report counts such rows and warns if > 5 %.
- Init: `rolling_kmeans` on the first series' rolling scale; `Σ_i` initialized from the empirical covariance of each cluster (documented).
- Preset/example: `examples/two_index_full_cov.yaml` (`^GDAXI`, `^GSPC`, daily, 2 states, `trading` clock) running offline on fixture data.
- Bundle: `extensions/regimekit.json` records the per-series provider metadata (already supported).

## Creates
- `src/regimekit/emission/gbm.py` (full option), `examples/two_index_full_cov.yaml`, `tests/integration/test_e11_t09_two_index.py`, `tests/recovery/test_e11_t09_gbm_full.py` (`-m slow`)

## Acceptance criteria
1. Two-index synthetic recovery (shared 2-regime process, correlations 0.3/0.8 per regime, `T = 5000`, 3 seeds): correlation and variances within 15 % relative.
2. `M = 1` and diagonal multi-primary configurations reproduce the v0.2.0 / E9-T01 golden values bitwise.
3. The example runs `fit → decode → forecast` through the `regimekit` CLI offline; `forecast.parquet` contains both series.

## Tests
Integration, slow recovery, golden regression.

## Out of scope
Different regime processes per instrument (factorial, E9-T02).

## Connections
- Depends on: E11-T08, E10-T09
- Blocks: E17-T05
- Relates to: E9-T01, E10-T06
