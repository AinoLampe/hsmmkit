# E2-T03 — `stooq` adapter (EOD fallback, EU coverage)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-D01 | M1 | §5.6 Providers table |

## Goal
Provide a daily-EOD fallback provider with API-key handling and quota-aware throttling.

## Context (self-contained)
- Via `pandas-datareader` (`data.DataReader(symbol, "stooq")`) or direct CSV endpoint; key from env `STOOQ_API_KEY` (absent → `ProviderError` explaining how to set it, *before* any network call).
- `supports_intraday=False`; requesting `1h` raises `ProviderError`. `1wk`/`1mo` are resampled from daily (last close of period) with a note in `meta`.
- Daily quota: a client-side token bucket (`max_calls_per_day`, default 100, configurable) persisted in the cache dir; exceeding it raises `ProviderError("stooq quota")`.
- Symbol mapping helper: `^GDAXI → ^dax`, `^GSPC → ^spx` (document, keep a small mapping table + passthrough).

## Creates
- `src/regimekit/finance/providers/stooq.py` (Part II §15.7 amendment 3: providers live in the finance profile)
- `tests/unit/test_e2_t03_stooq.py` (recorded)
- docs section in `data_providers.md`

## Acceptance criteria
1. Passes the E2-T01 contract suite with recorded responses.
2. Missing key → `ProviderError` with setup instructions, no network attempted (assert via mock).
3. Quota exhaustion → `ProviderError`; token bucket persists across two adapter instances.

## Out of scope
Intraday; covariate series.

## Connections
- Depends on: E2-T01
- Blocks: —
- Relates to: E2-T02 (primary), risk table "Yahoo throttling"
