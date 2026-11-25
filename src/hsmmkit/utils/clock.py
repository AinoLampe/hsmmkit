"""Clock-unit conversions: the registry-dispatching helpers (E10-T03, CLAUDE.md §3.2).

The clocks themselves live in `hsmmkit.data.clocks` (core: `steps`, `calendar`) and in the
application; this module looks them up in the registry, so that nothing else branches on a clock's name.
`CALENDAR_DAYS_PER_YEAR` is the only year constant of the core.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pydantic
from numpy.typing import ArrayLike

from hsmmkit.plugins import REGISTRY
from hsmmkit.utils.errors import ConfigError, DataError

if TYPE_CHECKING:
    from pydantic import BaseModel

    from hsmmkit.data.clocks import Clock

CALENDAR_DAYS_PER_YEAR = 365.25
IRREGULAR_RATIO = 1.5


def get_clock(name: str) -> Clock:
    """The registered clock `name`.

    Raises:
        ConfigError: unknown clock (`path="data.clock"`, the message lists the registered names).
    """
    importlib.import_module("hsmmkit.data.clocks")  # registers the core clocks
    names = REGISTRY.names("clock")
    if name not in names:
        raise ConfigError(f"unknown clock '{name}'; registered: {', '.join(names)}", path="data.clock")
    clock: Clock = REGISTRY.get("clock", name)
    return clock


def parse_clock_options(clock: str, options: Mapping[str, Any] | None) -> BaseModel:
    """`options` validated by the options model of `clock`.

    Raises:
        ConfigError: unknown clock or an invalid option (`path="data.clock_options.<field>"`).
    """
    model = get_clock(clock).options_model
    try:
        return model.model_validate(dict(options or {}))
    except pydantic.ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(part) for part in first["loc"])
        raise ConfigError(
            first["msg"], path=f"data.clock_options.{loc}" if loc else "data.clock_options"
        ) from exc


def days_to_steps(
    days: float, clock: str, nominal_dt: float, options: Mapping[str, Any] | None = None
) -> float:
    """Approximate the number of `nominal_dt`-sized steps that `days` clock-days span.

    A config-time heuristic used before any real dataset exists (`nominal_dt` is not yet a fitted value):
    `nominal_dt` is treated as one day's worth of the clock's unit, so several steps per day each span
    `nominal_dt / steps_per_day`. A step clock (`unit == "step"`) has no days: `days` is already a step
    count and is returned unchanged.
    """
    clk = get_clock(clock)
    if clk.unit == "step" or clk.days_per_year is None:
        return days
    return (days / clk.days_per_year) * clk.steps_per_day(parse_clock_options(clock, options)) / nominal_dt


def future_grid(
    last_time: Any,  # noqa: ANN401
    h: int,
    clock: str,
    options: Mapping[str, Any] | None = None,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The next `h` forecast timestamps after `last_time` and their `Δ`: `(times (h,), dt (h,))`.

    The clock decides the calendar (the weekday clocks skip Saturdays and Sundays; there is no holiday
    calendar) and the spacing: the median spacing of `times` (the observed grid), or one day when `times` is
    not given.

    Raises:
        ConfigError: `h < 1`, a non-positive step, unknown clock or invalid options.
    """
    clk = get_clock(clock)
    return clk.future_grid(last_time, h, parse_clock_options(clock, options), times)


def is_irregular(dt: ArrayLike) -> bool:
    """Whether a `Δ_t` grid counts as irregular: `max Δ_t / min Δ_t > 1.5` (concept §4.1).

    The explicit-duration engine assumes a regular grid and warns when this holds (ADR-0007).
    Always `False` under a step clock (`Δ_t ≡ 1`).

    Raises:
        DataError: `dt` is empty or not strictly positive.
    """
    arr = np.asarray(dt, dtype=np.float64)
    if arr.size == 0 or not bool(np.all(arr > 0)):
        raise DataError("dt must be non-empty and > 0 everywhere", path="dt")
    return bool(arr.max() / arr.min() > IRREGULAR_RATIO)
