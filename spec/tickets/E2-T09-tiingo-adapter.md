# E2-T09 — `tiingo` adapter (optional extra)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-D01 | M1 | §5.6 Providers table |

## Goal
Provide clean US EOD prices via Tiingo behind the same adapter interface, with free-tier limits enforced client-side.

## Context (self-contained)
- REST `https://api.tiingo.com/tiingo/daily/<ticker>/prices?startDate=&endDate=&token=`; key from env `TIINGO_API_KEY` (missing → `ProviderError` before network). Import guarded: the adapter registers only if its optional dependency (`requests`, already transitively available, or `tiingo` package if chosen — document in ADR-0010 addendum if a new dependency is needed) is present.
- Fields: `adjClose` when `adjusted=True`, else `close`. `supports_intraday=False`. Weekly/monthly resampled from daily as in E2-T03.
- Free-tier limits (configurable defaults): 500 unique symbols/month, 50 requests/hour, 1000/day — token buckets persisted in the cache dir; exceeding → `ProviderError`.

## Creates
- `src/regimekit/finance/providers/tiingo.py` (Part II §15.7 amendment 3: providers live in the finance profile)
- `tests/unit/test_e2_t09_tiingo.py` (recorded)

## Acceptance criteria
1. Passes the E2-T01 contract suite with recorded responses.
2. Hourly limit enforced with a fake clock; missing key → `ProviderError` without network.

## Out of scope
Intraday; fundamentals.

## Connections
- Depends on: E2-T01
- Blocks: —
- Relates to: E2-T02/T03 (fallback chain)
