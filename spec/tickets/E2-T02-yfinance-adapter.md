# E2-T02 — `yfinance` adapter (default price provider)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-D01, FR-D02, NFR-08 | M1 | §5.6 Providers table |

## Goal
Fetch adjusted prices for indices/ETFs/FX via yfinance with retries, throttling, intraday window capping and a ToS log line.

## Context (self-contained)
- Uses `yfinance ≥ 1.7` with a `curl_cffi` session (`impersonate="chrome"`), `auto_adjust=True`; `adjusted=False` requests `Close` without adjustment (`auto_adjust=False`, take `Close`).
- Intervals map 1:1 (`1h, 1d, 1wk, 1mo`). Intraday window caps: `1h` → at most 730 days back (`1m` ≈ 7 d is not exposed but the cap table is data-driven); requests exceeding the cap are clipped with a `warnings.warn` and a log line.
- Batch download for multiple symbols → `ObservationSet` with `prices (T0, M)` aligned on the union of timestamps (NaN where missing; alignment to the master grid happens later in E2-T06).
- Every fetch logs `provider="yfinance", tos=<tos_note>` at INFO (personal-use ToS note).
- Retries: `with_retries` (E2-T01), throttle ≥ 0.5 s between calls.
- Recorded responses: store the *processed* DataFrame for `^GDAXI` daily/hourly (short window, synthetic-looking values allowed; never commit long raw histories) under `tests/fixtures/cassettes/yfinance/*.parquet`; the adapter takes an injectable `downloader` callable so tests replace the network.

## Creates
- `src/regimekit/finance/providers/yfinance.py` (Part II §15.7 amendment 3: providers live in the finance profile)
- `tests/unit/test_e2_t02_yfinance.py` (recorded), `tests/live/test_e2_t02_yfinance_live.py` (`-m live`)
- `docs/guide/data_providers.md` (yfinance section incl. ToS)

## Acceptance criteria
1. Passes the E2-T01 contract suite with recorded responses for `^GDAXI` `1d` and `1h`.
2. Requesting 3 years of `1h` data warns and clips to the cap.
3. Live test (`-m live`) fetches the last 30 days of `^GDAXI` daily and validates.
4. ToS note appears in captured logs on each fetch.

## Tests
Recorded-response tests; live test opt-in.

## Out of scope
Caching (E2-T05); other providers.

## Connections
- Depends on: E2-T01
- Blocks: E2-T05 (default provider), E8-T03
- Relates to: E2-T03/T09 (fallbacks)
