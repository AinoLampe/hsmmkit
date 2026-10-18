# E2-T06 — Calendar and alignment: master grid, clocks, `Δ_t`, covariate `merge_asof`, lag ≥ 1, staleness

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-D04, FR-D05, D1; Part II FR-G02 | M1 | §4.1, §4.4 Alignment, §5.6 Calendar; Part II §14.1, §15.7 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendment 2: third clock `steps`). No behavioural change for the finance profile.

## Goal
Build the master observation grid from the primary instrument, compute `Δ_t` per clock, and align/lag every covariate onto it without look-ahead.

## Context (self-contained)
- Master grid = timestamps of the primary instrument after dropping rows with a missing price (`missing_returns: drop`) or raising (`error`). Grid end times `τ_0 < … < τ_T`; observation `t` refers to `(τ_{t−1}, τ_t]`, so `T = len(prices) − 1`.
- `Δ_t` (clock units): `clock="trading"` → `1 / (252 · bars_per_day)` years, constant; `clock="calendar"` → `(τ_t − τ_{t−1}) / 365.25` years (days as float, from ns); `clock="steps"` (ADR-0013) → `1.0` for every `t`, `nominal_dt = 1.0`, `irregular = False`; the grid still comes from the primary instrument's timestamps in 0.1.0 (timeless grids are Part II). `nominal_dt = median(Δ_t)`; if `max Δ / min Δ > 1.5` store `irregular=True` (the explicit engine warns on it later). Nothing downstream of `calendar.py` branches on the clock name — only on `dt`/`nominal_dt`.
- Covariate alignment: for each `CovariateSeries`, `pd.merge_asof(grid, series, direction="backward")` (value known at or before the grid end time), then **shift by `lag` grid steps** (`lag ≥ 1` enforced; `lag = 0` → `ConfigError(path="data.covariates[i].lag")`), then forward-fill up to `max_staleness` grid steps beyond the last observed value; beyond that → NaN. Result is a step function for monthly series. Output: `AlignedCovariates(names, values (T, K), staleness (T, K) int)`.
- Functions: `build_grid(obs, missing_returns) -> Grid(times, prices, dt, nominal_dt, irregular)`, `align_covariate(grid, series, lag, max_staleness) -> (values, staleness)`.

## Creates
- `src/regimekit/data/calendar.py`
- `tests/unit/test_e2_t06_calendar.py`

## Acceptance criteria
1. Trading vs calendar vs steps `Δ` on a fixture spanning a known 4-day holiday gap: trading gives `1/252` everywhere, calendar gives `4/365.25` at the gap, steps gives `1.0` everywhere with `irregular=False`.
2. A monthly covariate aligned onto a daily grid becomes a step function whose value at grid step `t` is the last publication at or before `τ_{t−1−(lag−1)}` (i.e. shifted by `lag` steps) — assert exact indices on a hand-built fixture.
3. `lag = 0` → `ConfigError`; `max_staleness` exceeded → NaN with the staleness counter reported.
4. No value at step `t` depends on data timestamped after `τ_{t−1}` (property test with a random spike injected at `τ_t` and asserting no change at `t`).

## Out of scope
Transforms/standardization (E2-T07); missing-value policy decisions (E2-T07 consumes staleness/NaN).

## Connections
- Depends on: E1-T01
- Blocks: E2-T07, E9-T01
- Relates to: ADR-0001, ADR-0007
