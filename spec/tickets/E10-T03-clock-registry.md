# E10-T03 — Clock registry: `steps` and `calendar` in the core, `trading` (and its future grid) in finance, ADR-0015

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| RegimeDetectionAndForecast* | M | FR-G02, D13 | M7a | Part II §14.1, §15.3 (`Clock`), D13; Part I §4.1, §5.6 Calendar, E6-T04a (`future_grid`) |

## Goal
Make clocks pluggable so that the core knows only `steps` and `calendar`, while the trading-day clock and its trading-day future grid live in the finance package.

## Context (self-contained)
```python
class Clock(Protocol):
    name: str
    unit: Literal["year", "step"]                 # unit of Δ and of mean_sojourn reporting
    options_model: type[BaseModel]               # validated from data.clock_options
    def dt(self, times: np.ndarray | None, n: int, opts: BaseModel) -> np.ndarray: ...   # (n,) Δ_t > 0
    def future_grid(self, last_time: Any, h: int, opts: BaseModel,
                    times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]: ...       # (horizon_times (h,), dt (h,))
```
Core clocks (`src/regimekit/data/clocks.py`, registered kind `clock`):
- `steps` (default): `dt = ones(n)`; `unit="step"`; `future_grid`: if `times` is `None` → integer steps `last+1..last+h`; else continue with the median timestamp spacing (Part I E6-T04a rule); `dt = ones(h)`. No options.
- `calendar`: `dt_t = (τ_t − τ_{t−1}) / 365.25` days→years (ns arithmetic as in E2-T06); requires `times`; `unit="year"`; `future_grid`: median spacing continuation, `dt` from the same rule. No options (unit configurability is backlog, Part II §19 #20).
Finance clock (`src/regimekit/finance/clocks.py`): `trading` with `options_model = TradingOpts(bars_per_day: int = 1)`, `dt = 1 / (252 · bars_per_day)`, `unit="year"`, `future_grid` = next weekdays (Part I rule: weekend skipping, no holiday calendar). Source of the moved code: `src/regimekit/utils/clock.py` (CLAUDE.md §3.2), which today holds `TRADING_DAYS_PER_YEAR = 252`, `CALENDAR_DAYS_PER_YEAR = 365.25`, `future_grid(...)`, `days_to_steps(...)` and `is_irregular(dt)`. Move `TRADING_DAYS_PER_YEAR`, the trading branch of `future_grid` and of `days_to_steps` to the finance clock; `utils/clock.py` keeps `CALENDAR_DAYS_PER_YEAR`, `is_irregular` and dispatches the rest through the clock registry. `estimation/recovery.py`'s `NOMINAL_DT = 1/252` is derived from the `trading` clock (recovery scenarios that need a daily grid get it from the clock, not a literal). The literal `252` exists only in the finance clock.
`data/calendar.py::build_grid` becomes clock-agnostic: `clock = REGISTRY.get("clock", cfg.clock)`; `opts = clock.options_model.model_validate(cfg.clock_options)` (validation error → `ConfigError(path="data.clock_options.<field>")`); `dt = clock.dt(times, n, opts)`; `nominal_dt = median(dt)`; `irregular` as before. `PreparedDataset.clock` becomes `str` (registered name) instead of a `Literal`; `validate()` checks the name is registered and, for `unit == "step"`, that `dt == 1`.
`inference/forecast.py` (via `utils/clock.py`) calls `clock.future_grid(...)` instead of the moved function. Duration families already report `mean_sojourn` in clock units (Part I amendment) — no change.
Config: core `DataSection.clock` default `"steps"`; `FinanceDataSection.clock` default `"trading"`; extend `migrate_1_to_2` (E10-T02b) — `bars_per_day` → `clock_options.bars_per_day` already there; add: `clock: steps` in a v1 finance config stays `steps`.
Clock contract suite (`src/regimekit/testing/clock_contract.py`, becomes `hsmmkit.testing`): for each registered clock with fixture times: `dt > 0`, shape `(n,)`, `future_grid` returns strictly increasing times after `last_time` and `dt > 0`, `unit` consistency (`steps` → all ones), options model round-trips.
**ADR-0015** (`docs/adr/0015-clock-registry.md`, accepted; amends ADR-0013).

## Creates
- `src/regimekit/data/clocks.py`, `src/regimekit/finance/clocks.py`, edits in `utils/clock.py`, `data/calendar.py`, `inference/forecast.py`, `data/contracts.py`, `estimation/recovery.py`
- `src/regimekit/testing/{__init__,clock_contract}.py`
- `tests/unit/test_e10_t03_clocks.py`, `tests/finance/test_e10_t03_trading_clock.py`
- `docs/adr/0015-clock-registry.md`, `docs/math/time.md` (clock section updated)

## Acceptance criteria
1. Appendix E numbers under `trading` unchanged (`a_calm = 0.0079`, `a_turb = 0.0235`; daily return mean/sd); E2-T06 holiday-gap test passes for all three clocks, `trading` now from the finance package.
2. `grep -rn "252" src/regimekit/{data,model,statespace,engine,estimation,inference,evaluation,config,utils}` returns nothing.
3. `data.clock: trading` in a core `ProjectConfig` with `HSMM_CORE_ONLY=1` → `ConfigError(path="data.clock")` listing `calendar, steps`.
4. `clock_options: {bars_per_day: 0}` → `ConfigError(path="data.clock_options.bars_per_day")`.
5. All three clocks pass the clock contract suite; v0.2.0 golden values bitwise; allow-list entries tagged `E10-T03` gone.

## Tests
Contract suite, per-clock unit tests, forecast `horizon_times` per clock, golden test.

## Out of scope
Timeless grids from generic sources (E10-T04a provides `times=None` data); entry-point discovery (E16-T04).

## Connections
- Depends on: E10-T02b
- Blocks: E10-T05, E10-T07a
- Relates to: E2-T06, E6-T04a, E10-T04a
