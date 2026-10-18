# E2-T08 — `fetch` command and `DataQualityReport`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-D06, FR-C02 | M1 | §5.6 Quality report, §5.9 |

## Goal
Wire providers, cache and preparer into `regimekit fetch CONFIG` and emit a data-quality report next to the cache.

## Context (self-contained)
- Flow: `load_config` → build `CachedProvider` for primary and each covariate provider (respect `--offline`, `cache.snapshot`, `max_age_days`) → fetch → `Preparer.fit(train_end = end of the initial training window from `evaluation.backtest.initial_train_years`, or full span if no backtest)` → `transform` → write `PreparedDataset` as `<out>/prepared.parquet` (+ `prep_stats.json`, `snapshots.json`) → write quality report.
- `DataQualityReport` (`data/quality.py`) fields: gaps vs. a weekday calendar (list of missing dates > 1 business day), duplicated timestamps, zero/negative prices, `|return| > 8σ` (flag only, list), adjustment discontinuities (compare overlapping range of the two latest cache generations: max relative price difference > 1e-6), covariate staleness histogram per covariate, counts of dropped rows per policy. Written as `quality.json` and `quality.md`.
- Exit codes: 0; `DataError` (e.g. `missing: error` policy) → 3; `ProviderError` → 5.
- Logs provider ToS notes (from `tos_note`) at INFO.

## Creates
- `src/regimekit/data/quality.py`, `fetch` command in `cli/app.py`, `src/regimekit/persistence/results.py` (initial: `write_prepared`, `read_prepared`)
- `tests/integration/test_e2_t08_fetch.py`, `tests/fixtures/csv/defects/*.csv`

## Acceptance criteria
1. On a fixture with injected gap, duplicate, zero price and 10σ outlier, the report lists each defect with timestamps.
2. `fetch` on a csv-provider config writes `prepared.parquet`, `prep_stats.json`, `snapshots.json`, `quality.json/.md` and exits 0.
3. With `missing: error` and a stale covariate, exit code 3.
4. `read_prepared(write_prepared(ds)) == ds` (arrays bitwise).

## Out of scope
Model fitting; `Project` facade (E8-T02).

## Connections
- Depends on: E2-T05, E2-T07, E1-T02b
- Blocks: E8-T02, E8-T03
- Relates to: E2-T04 (Appendix A covariates)
