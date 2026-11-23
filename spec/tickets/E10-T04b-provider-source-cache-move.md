# E10-T04b — xarray source; providers as source kind `provider`; parquet cache and snapshots moved to finance

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | M | FR-G04, FR-PL02, FR-D03 | M7a | Part II §15.3 (`DataProvider` as `Source`), §15.8 move map; Part I §5.6 (cache, snapshots), E2-T01, E2-T05 |

## Goal
Finish the source layer: add the xarray source, express all finance providers as one registered source kind, and move the provider cache/snapshot machinery out of the core.

## Context (self-contained)
**xarray source** (`data/sources/xarray.py`, kind `xarray`): `obj: xr.Dataset` (API) or `path` (`.nc` via `xr.open_dataset`, `.zarr` via `xr.open_zarr`); options `TabularOpts` with `observation_columns` = data variables, `time_column` = a coordinate name or `None`, `sequence_column` = coordinate or variable. Import via the lazy helper (raises `ImportError("pip install hsmmkit[xarray]")`; until E10-T11 the extra is spelled in the message only). Converted through the pandas logic.
**Providers as a source** (`finance/providers/source.py`): one registered source kind `provider` (provenance `regimekit.finance`) whose `options_model` is the Part I `PriceSource` (`provider, symbols, interval, start, end, adjusted, field`) for `data.source`, and `CovariateProviderOpts(provider, series)` for covariate sources. `load()` resolves the provider adapter by name from the finance provider registry (E2-T01, unchanged), goes through the cache, and returns `ObservationSet` (+ covariates). Covariates with `source: {kind: provider, …}` are loaded one by one and merged into the covariate list.
**Cache move**: `data/cache.py` → `finance/cache.py` (parquet cache, `.meta.json`, `snapshot_id`, `max_age_days`, `--offline` replay). Core keeps only the generic `snapshot_id` computation helper used by E10-T04a (`data/provenance.py::hash_bytes/hash_arrays`). `CacheCfg` already lives in `FinanceDataSection` (E10-T02b).
**Contract suites**: the E2-T01 provider contract suite stays in finance for provider-specific checks (`tos_note`, `interval`, retries); in addition the `provider` source kind is parametrized into `testing.source_contract` with the CSV provider adapter as fixture (offline).
**Fetch path**: `Project.fetch` / `regimekit fetch` now build data via `REGISTRY.get("source", cfg.data.source.kind).load(...)`; the quality report (E2-T08) still runs on fetched provider data (split in E10-T05).

## Creates
- `src/regimekit/data/sources/xarray.py`, `src/regimekit/data/provenance.py`
- `src/regimekit/finance/providers/source.py`, `src/regimekit/finance/cache.py` (moved from `data/cache.py`)
- `tests/unit/test_e10_t04b_xarray_source.py`, `tests/finance/test_e10_t04b_provider_source.py`; E2-T05 tests moved to `tests/finance/`

## Acceptance criteria
1. xarray and `provider` (CSV adapter, offline) sources pass `source_contract`.
2. `regimekit fetch` on the Appendix A v2 config (offline fixtures) produces a `PreparedDataset` equal bitwise to the v0.2.0 golden `prepared.npz`; second fetch hits the cache (no network, asserted via mock).
3. `src/regimekit/data/` contains no cache, snapshot-replay or provider code; `HSMM_CORE_ONLY=1` import of `regimekit.data` succeeds.
4. Scanner/import allow-list entries tagged `E10-T04b` gone; v0.2.0 golden values bitwise.

## Tests
Contract suite over all sources; cache tests moved and green; golden test.

## Out of scope
Preparation/transform split (E10-T05); entry points (E16-T04).

## Connections
- Depends on: E10-T04a
- Blocks: E10-T07b
- Relates to: E2-T01, E2-T05, E2-T08
