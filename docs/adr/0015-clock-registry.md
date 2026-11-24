# ADR-0015: Clock registry; the `trading` clock lives in the finance application

## Context

ADR-0013 introduced three clocks (`trading`, `calendar`, `steps`) and allowed code to branch on the clock name
in `utils/clock.py` and `data/calendar.py`. The trading-day clock (252 days per year, weekday grids) is
finance vocabulary; the core must not know it (ADR-0014).

## Decision

- Clocks are components of the registry kind `clock` (ADR-0016's mechanism, E10-T02a). A clock has a `name`,
  a `unit` (`"year"` or `"step"`), an options model (validated from `data.clock_options`), `dt(times, n, opts)`
  and `future_grid(last_time, h, opts, times)`; it also states `days_per_year` and `steps_per_day(opts)` for the
  config-time days-to-steps heuristic.
- The core registers `steps` (`Δ ≡ 1`, the default) and `calendar` (`Δ` = elapsed days / 365.25).
  The finance package registers `trading` (`Δ = 1 / (252 · bars_per_day)`, weekday grids); the literal `252`
  exists only there.
- `utils/clock.py` keeps `CALENDAR_DAYS_PER_YEAR` and `is_irregular` and dispatches the rest
  (`get_clock`, `days_to_steps`, `future_grid`) through the registry. `PreparedDataset.clock` is the registered
  name (a `str`); `validate()` checks it and, for a step clock, that `dt == 1`.
- The `future_grid` of the weekday clocks keeps the Part I rule (weekend skipping, no holiday calendar); the
  future spacing is the median spacing of the observed timestamps.

## Consequences

- ADR-0013 is amended: the name branching it allowed is replaced by registry dispatch.
- A new clock needs no change in the core: it registers and passes the clock contract suite
  (`regimekit.testing.clock_contract`).
- Until the finance import leaves `regimekit/__init__` (E10-T07b), the `HSMM_CORE_ONLY` job cannot show a core
  without `trading`; the tests unregister it explicitly.

## Status

accepted
