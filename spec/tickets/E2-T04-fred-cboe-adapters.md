# E2-T04 — `fred` and `cboe` covariate adapters

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-D01 | M1 | §5.6 Providers table, Appendix A covariates |

## Goal
Fetch macro/volatility covariates (`VIXCLS`, `T10Y2Y`, CBOE VIX history) as `CovariateSeries`.

## Context (self-contained)
- `fred`: `fredapi` with key from env `FRED_API_KEY` (missing → `ProviderError` with instructions, no network). Values `"."` → NaN. Rate limiter: ≤ 120 requests/min (token bucket, in-process). `fetch_series(series_id, start, end)` → `CovariateSeries(frequency=<from FRED metadata: "d","w","m","q">)`. `fetch_prices` raises `ProviderError` (not a price provider).
- `cboe`: no key; downloads daily CSV files whose URL patterns are configurable (`urls: dict[str, str]` in adapter kwargs; defaults for `VIX_History.csv`, `VVIX_History.csv`). Columns `DATE, OPEN, HIGH, LOW, CLOSE` → `value = CLOSE`. `frequency="d"`.
- Both: UTC dates at 00:00 (observation date; alignment/lag handled in E2-T06), float64, retries.
- Recorded fixtures: short synthetic CSV/JSON snippets shaped like the real responses (no real data committed beyond a few rows).

## Creates
- `src/regimekit/finance/providers/{fred,cboe}.py` (Part II §15.7 amendment 3: providers live in the finance profile)
- `tests/unit/test_e2_t04_fred_cboe.py`, `tests/fixtures/cassettes/{fred,cboe}/*`

## Acceptance criteria
1. Both adapters pass the E2-T01 contract suite (`fetch_series` part) with recorded responses for `VIXCLS`, `T10Y2Y`, `VIX_History.csv`.
2. `"."` values become NaN; NaN count asserted on a fixture.
3. FRED limiter blocks the 121st call within a minute (test with a fake clock).
4. Missing `FRED_API_KEY` → `ProviderError` without network.

## Out of scope
Transforms, lagging, standardization (E2-T06/T07).

## Connections
- Depends on: E2-T01
- Blocks: E2-T08 (Appendix A covariates), E8-T03
- Relates to: E2-T06
