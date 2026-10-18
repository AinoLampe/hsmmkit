# E2-T05 — Parquet cache, snapshot IDs, `--offline` replay, `max_age`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-D03, NFR-03, NFR-08 | M1 | §5.6 Cache and snapshots |

## Goal
Make every provider fetch cacheable, hashable and replayable offline so a run is reproducible from its snapshot IDs.

## Context (self-contained)
- Layout: `<cache_dir>/<provider>/<symbol>_<interval>_<adjusted>.parquet` (+ `.meta.json`: `fetched_at`, provider version, request params, `snapshot_id`); covariates: `<cache_dir>/<provider>/series_<series_id>.parquet`. `cache_dir` default `~/.cache/regimekit` (expand `~`).
- `snapshot_id = sha256(provider | symbols | interval | adjusted | raw parquet bytes)` — parquet written with fixed pyarrow options (no timestamps in metadata, fixed compression) so bytes are stable.
- `CachedProvider(inner: DataProvider, cache: Cache, policy)` wraps any adapter: `snapshot="latest"` → serve from cache if `fetched_at` younger than `max_age_days`, else refetch and write a new generation (old generation kept under `<name>.<snapshot_id>.parquet` for adjustment-discontinuity checks in E2-T08); `snapshot="<id>"` → serve exactly that generation or raise `DataError`.
- `offline=True` → never call `inner`; missing cache → `DataError("run `regimekit fetch` first")`.
- Provider adapters return `snapshot_id=""`; `CachedProvider` fills it. Snapshot IDs are recorded in `PreparedDataset.snapshot_ids` (E2-T07) and in every result bundle.

## Creates
- `src/regimekit/data/cache.py` (`Cache`, `CachedProvider`, `SnapshotPolicy`)
- `tests/unit/test_e2_t05_cache.py`

## Acceptance criteria
1. Second identical fetch hits the cache: `inner.fetch_prices` called once (mock counter).
2. `snapshot_id` is identical across two fresh caches for identical input data.
3. `offline=True` with an existing snapshot returns a bitwise-identical `ObservationSet`; without cache raises `DataError`.
4. `max_age_days=0` forces refetch; pinned `snapshot="<id>"` returns the old generation even after a refetch.

## Out of scope
Data-quality report (E2-T08); network adapters.

## Connections
- Depends on: E2-T01
- Blocks: E2-T08, E7-T04 (`compare` refuses differing snapshot IDs), E8-T01
- Relates to: E2-T02…T04, T09 (wrapped adapters)
