"""Clocks (E10-T03, FR-G02, D13): how the grid spacing `Δ_t` is defined, registered as kind `clock`.

The core knows two clocks: `steps` (`Δ_t ≡ 1`, the default) and `calendar` (`Δ_t` in years from the
timestamps). Applications register more. A clock gives the
spacing of an observed grid (`dt`) and of the future grid of a forecast (`future_grid`).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from regimekit.plugins import REGISTRY, provenance_of
from regimekit.utils.clock import CALENDAR_DAYS_PER_YEAR
from regimekit.utils.errors import ConfigError

LastTime = np.datetime64 | int | np.integer

NS_PER_DAY = 86_400 * 10**9


class Clock(Protocol):
    """A grid-spacing definition (see the module docstring)."""

    name: str
    unit: Literal["year", "step"]
    options_model: type[BaseModel]
    days_per_year: float | None

    def dt(self, times: np.ndarray | None, n: int, opts: BaseModel) -> np.ndarray:
        """`(n,)` spacing `Δ_t > 0` of an observed grid (`times` are the `n + 1` timestamps, if any)."""
        ...

    def future_grid(
        self,
        last_time: LastTime,
        h: int,
        opts: BaseModel,
        times: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`(horizon_times (h,), dt (h,))` of the `h` steps after `last_time`."""
        ...

    def steps_per_day(self, opts: BaseModel) -> int:
        """Grid steps per calendar day, for the config-time days-to-steps heuristic."""
        ...


class NoOptions(BaseModel):
    """Options model of clocks without options."""

    model_config = ConfigDict(extra="forbid")


def is_weekend(t: np.datetime64) -> bool:
    """Saturday or Sunday (1970-01-01 was a Thursday)."""
    days = int(t.astype("datetime64[D]").astype(np.int64))
    return (days + 3) % 7 >= 5


def median_step(times: np.ndarray | None, default: np.timedelta64) -> np.timedelta64:
    """Median spacing of `times` (nanoseconds), or `default` when fewer than two timestamps are given."""
    if times is None or np.asarray(times).shape[0] < 2:
        return default
    stamps = np.asarray(times).astype("datetime64[ns]")
    return np.median(np.diff(stamps)).astype("m8[ns]")  # type: ignore[no-any-return]


def check_horizon(h: int) -> None:
    """Reject a non-positive horizon."""
    if h < 1:
        raise ConfigError(f"horizon must be >= 1, got {h}", path="forecast.horizon")


def check_step(step_ns: np.timedelta64) -> None:
    """Reject a non-positive step."""
    if step_ns <= np.timedelta64(0, "ns"):
        raise ConfigError("step must be positive", path="forecast.step")


def continue_times(last: LastTime, h: int, step_ns: np.timedelta64, *, skip_weekends: bool) -> np.ndarray:
    """`h` timestamps after `last`, `step_ns` apart, optionally skipping Saturdays and Sundays."""
    times = np.empty(h, dtype="datetime64[ns]")
    cur = np.datetime64(cast(Any, last), "ns")
    for k in range(h):
        cur = cur + step_ns
        if skip_weekends:
            while is_weekend(cur):
                cur = cur + step_ns
        times[k] = cur
    return times


class StepsClock:
    """`Δ_t ≡ 1` (parameters per step, no conversion to years)."""

    name = "steps"
    unit: Literal["year", "step"] = "step"
    options_model = NoOptions
    days_per_year: float | None = None

    def dt(self, times: np.ndarray | None, n: int, opts: BaseModel) -> np.ndarray:
        """Ones."""
        return np.ones(n, dtype=np.float64)

    def future_grid(
        self,
        last_time: LastTime,
        h: int,
        opts: BaseModel,
        times: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integer steps for integer `last_time`, else timestamps continued with the median spacing."""
        check_horizon(h)
        if isinstance(last_time, int | np.integer):
            return np.arange(int(last_time) + 1, int(last_time) + 1 + h), np.ones(h)
        step_ns = median_step(times, np.timedelta64(NS_PER_DAY, "ns"))
        check_step(step_ns)
        return continue_times(last_time, h, step_ns, skip_weekends=False), np.ones(h)

    def steps_per_day(self, opts: BaseModel) -> int:
        """One."""
        return 1


class CalendarClock:
    """`Δ_t` = elapsed calendar days / 365.25 (years)."""

    name = "calendar"
    unit: Literal["year", "step"] = "year"
    options_model = NoOptions
    days_per_year: float | None = CALENDAR_DAYS_PER_YEAR

    def dt(self, times: np.ndarray | None, n: int, opts: BaseModel) -> np.ndarray:
        """Gaps between the timestamps in years (needs `times`)."""
        if times is None:
            raise ConfigError("the calendar clock needs timestamps", path="data.clock")
        days = np.diff(np.asarray(times).astype("datetime64[ns]")).astype(np.int64).astype(np.float64)
        return days / NS_PER_DAY / CALENDAR_DAYS_PER_YEAR

    def future_grid(
        self,
        last_time: LastTime,
        h: int,
        opts: BaseModel,
        times: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weekday timestamps continuing with the median spacing; `Δ` from the calendar gaps."""
        check_horizon(h)
        step_ns = median_step(times, np.timedelta64(NS_PER_DAY, "ns"))
        check_step(step_ns)
        last = np.datetime64(cast(Any, last_time), "ns")
        out = continue_times(last, h, step_ns, skip_weekends=True)
        gaps = np.diff(np.concatenate([[last], out])).astype(np.int64).astype(np.float64)
        return out, gaps / NS_PER_DAY / CALENDAR_DAYS_PER_YEAR

    def steps_per_day(self, opts: BaseModel) -> int:
        """One."""
        return 1


for _clock in (StepsClock(), CalendarClock()):
    REGISTRY.register("clock", _clock.name, _clock, provenance=provenance_of(__name__))
