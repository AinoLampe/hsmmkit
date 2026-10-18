"""Clock-unit conversions (overview §4.3, CLAUDE.md §3.2).

`utils/clock.py` is the only module allowed a literal `252`/`365.25` outside `data/calendar.py`
(which imports the constants from here); no other module may branch on the clock's *name* either
(both restrictions live only here and in `data/calendar.py`).

Created by E1-T02a, which is the first ticket needing a days->steps conversion (to size
`model.d_max`'s default from `init_mean_sojourn_days`), ahead of E2-T06/E3-T04 in build order.
E2-T06 added `is_irregular`; E6-T04a adds `future_grid` here rather than recreating this module.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from regimekit.utils.errors import ConfigError, DataError

Clock = Literal["trading", "calendar", "steps"]

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25
IRREGULAR_RATIO = 1.5


def days_to_steps(days: float, clock: Clock, nominal_dt: float, bars_per_day: int) -> float:
    """Approximate the number of `nominal_dt`-sized (years) steps that `days` clock-days span.

    A config-time heuristic used before any real dataset exists (`nominal_dt` is not yet a fitted
    value): `nominal_dt` is treated as one calendar/trading day's worth of years, so `bars_per_day`
    intraday bars each span `nominal_dt / bars_per_day` years. Under `clock="steps"`, `days` is
    already a step count and is returned unchanged.
    """
    if clock == "steps":
        return days
    days_per_year = TRADING_DAYS_PER_YEAR if clock == "trading" else CALENDAR_DAYS_PER_YEAR
    return (days / days_per_year) * bars_per_day / nominal_dt


def is_irregular(dt: ArrayLike) -> bool:
    """Whether a `Δ_t` grid counts as irregular: `max Δ_t / min Δ_t > 1.5` (concept §4.1).

    The explicit-duration engine assumes a regular grid and warns when this holds (ADR-0007).
    Always `False` under `clock="steps"` (`Δ_t ≡ 1`).

    Raises:
        DataError: `dt` is empty or not strictly positive.
    """
    arr = np.asarray(dt, dtype=np.float64)
    if arr.size == 0 or not bool(np.all(arr > 0)):
        raise DataError("dt must be non-empty and > 0 everywhere", path="dt")
    return bool(arr.max() / arr.min() > IRREGULAR_RATIO)


_NS_PER_DAY = 86_400 * 10**9


def _is_weekend(t: np.datetime64) -> bool:
    days = int(t.astype("datetime64[D]").astype(np.int64))
    return (days + 3) % 7 >= 5  # 1970-01-01 was a Thursday; Monday == 0


def future_grid(
    last_time: np.datetime64,
    h: int,
    clock: Clock,
    bars_per_day: int = 1,
    step: np.timedelta64 | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The next `h` forecast timestamps after `last_time` and their `Δ` per clock: `(times (h,), dt (h,))`.

    Weekday calendar only (no holiday calendar in 0.1.0): candidates are spaced `step` apart
    (default one day / `bars_per_day`) and Saturday/Sunday candidates are skipped. `trading` gives
    `Δ = 1/(252·bars_per_day)`, `calendar` the gaps (from `last_time`) in days / 365.25. Under `steps`
    `Δ ≡ 1` and the timestamps continue with `step` (callers pass the median spacing), weekends included.

    Raises:
        ConfigError: `h < 1`, `bars_per_day < 1`, a non-positive `step` or an unknown clock.
    """
    if h < 1:
        raise ConfigError(f"horizon must be >= 1, got {h}", path="forecast.horizon")
    if bars_per_day < 1:
        raise ConfigError(f"bars_per_day must be >= 1, got {bars_per_day}", path="data.bars_per_day")
    if clock not in ("trading", "calendar", "steps"):
        raise ConfigError(f"unknown clock {clock!r}", path="data.clock")
    step_ns = np.timedelta64(_NS_PER_DAY // bars_per_day, "ns") if step is None else step.astype("m8[ns]")
    if step_ns <= np.timedelta64(0, "ns"):
        raise ConfigError("step must be positive", path="forecast.step")

    last = np.datetime64(last_time, "ns")
    times = np.empty(h, dtype="datetime64[ns]")
    cur = last
    for k in range(h):
        cur = cur + step_ns
        if clock != "steps":
            while _is_weekend(cur):
                cur = cur + step_ns
        times[k] = cur

    if clock == "trading":
        dt = np.full(h, 1.0 / (TRADING_DAYS_PER_YEAR * bars_per_day))
    elif clock == "calendar":
        gaps = np.diff(np.concatenate([[last], times])).astype(np.int64).astype(np.float64)
        dt = gaps / _NS_PER_DAY / CALENDAR_DAYS_PER_YEAR
    else:
        dt = np.ones(h)
    return times, dt
