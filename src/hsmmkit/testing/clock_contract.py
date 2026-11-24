"""Clock contract suite (E10-T03): the checks every registered clock must pass.

`check_clock(name, times)` raises `AssertionError` with a message naming the clock when a check fails; the
test modules parametrize it over `REGISTRY.names("clock")`.
"""

from __future__ import annotations

import numpy as np

from regimekit.utils.clock import future_grid, get_clock, parse_clock_options

HORIZON = 5


def fixture_times(n: int = 12) -> np.ndarray:
    """`n` consecutive weekday timestamps (a Monday start), the fixture grid of the contract."""
    days = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")
    weekdays = days[(days.astype(np.int64) + 3) % 7 < 5]
    return weekdays[:n].astype("datetime64[ns]")


def check_clock(name: str, times: np.ndarray | None = None) -> None:
    """Run the contract on the registered clock `name` (default fixture times).

    Raises:
        AssertionError: a check failed.
    """
    times = fixture_times() if times is None else times
    clock = get_clock(name)
    opts = parse_clock_options(name, None)
    assert clock.name == name, f"{name}: the clock names itself {clock.name!r}"
    assert clock.unit in ("year", "step"), f"{name}: unit {clock.unit!r}"
    assert type(opts) is clock.options_model, f"{name}: default options are not of the options model"
    assert clock.options_model.model_validate(opts.model_dump()) == opts, f"{name}: options do not round-trip"

    n = times.shape[0] - 1
    dt = clock.dt(times, n, opts)
    assert dt.shape == (n,), f"{name}: dt shape {dt.shape}, expected {(n,)}"
    assert bool(np.all(dt > 0)), f"{name}: dt must be positive"
    if clock.unit == "step":
        assert bool(np.all(dt == 1)), f"{name}: a step clock has dt == 1"

    out, dt_future = future_grid(times[-1], HORIZON, name, None, times)
    assert out.shape == (HORIZON,), f"{name}: future times shape"
    assert dt_future.shape == (HORIZON,), f"{name}: future dt shape"
    assert bool(np.all(dt_future > 0)), f"{name}: future dt must be positive"
    assert bool(out[0] > times[-1]), f"{name}: future times must come after the last time"
    assert bool(np.all(np.diff(out) > np.timedelta64(0, "ns"))), (
        f"{name}: future times must increase strictly"
    )
    if clock.unit == "step":
        assert bool(np.all(dt_future == 1)), f"{name}: a step clock has future dt == 1"
