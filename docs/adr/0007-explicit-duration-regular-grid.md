# ADR-0007: Explicit-duration engine requires a regular grid

## Context

The explicit-duration engine (`engine/explicit_duration.py`) sums duration-family log-pmf mass over sojourn
lengths measured in **steps**, not in elapsed continuous time. If the grid is highly irregular (large gaps
between consecutive observations relative to the typical spacing), a fixed number of steps no longer
corresponds to a consistent elapsed duration, and the explicit-duration semantics silently drift from the
continuous-time duration the user configured.

## Decision

The explicit-duration engine requires an approximately regular grid, defined as `max(Δ_t) / min(Δ_t) ≤ 1.5`
(`utils.clock.is_irregular`); it warns (does not error) when this bound is exceeded, since the expanded engine
remains correct on irregular grids and is the recommended fallback.

## Consequences

- `utils/clock.py` owns the one `is_irregular(dt)` check used by both `data/calendar.py` and the explicit
  engine, so the threshold is defined exactly once (overview §4.3's "no literal outside `clock.py`" rule).
- Users fitting on data with large gaps (e.g. multi-day halts) get a warning rather than silently-wrong
  duration semantics, and are pointed at the expanded engine.
- The expanded engine has no such restriction and remains the general-purpose default.

## Status

accepted
