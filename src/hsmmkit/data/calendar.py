"""Master calendar: observation grid, `Δ_t` per clock, covariate alignment and lagging (FR-D04, FR-D05).

Grid convention (concept §4.1, §5.6). The master grid is the primary series' timestamps after
missing-value handling, `τ_0 < … < τ_T`. Observation `t` (math, 1-based) covers `(τ_{t-1}, τ_t]`,
so there are `T = len(times) - 1` observations and `Δ_t` has shape `(T,)`. In code, row `t`
(0-based) is math step `t + 1`, i.e. the interval `(τ_t, τ_{t+1}]` (CLAUDE.md §3.7).

Covariate alignment (no look-ahead). Each series is aligned "as of" every grid time (the last
non-NaN print dated at or before `τ_j`, as `pd.merge_asof(direction="backward")` would), then
shifted by `lag ≥ 1` grid steps: code row `t` holds the value as of `τ_{t+1-lag}`, which is never
later than `τ_t`, the start of observation `t`. Values are carried forward at most `max_staleness`
grid steps after the print first became visible; beyond that they are NaN (the missing policy of
E2-T07 decides what happens next).

This module and `utils/clock.py` are the only places that branch on the clock *name*
(CLAUDE.md §3.2); everything downstream uses `dt`/`nominal_dt`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from hsmmkit.data.contracts import Clock, CovariateSeries, ObservationSet
from hsmmkit.utils.clock import future_grid as future_grid  # re-export (E6-T04a)
from hsmmkit.utils.clock import get_clock, is_irregular, parse_clock_options
from hsmmkit.utils.errors import ConfigError, DataError

MissingObservations = Literal["mask", "drop", "error"]

_NS_PER_DAY = 86_400 * 10**9
_NEVER = -1  # staleness of a row with no usable print (before the first one, or lost to the lag)


@dataclass(frozen=True, eq=False)
class Grid:
    """The master observation grid.

    Attributes:
        times: `(T+1,)` `datetime64[ns]` grid end times `τ_0 … τ_T` (`int64` step index if time-less).
        values: `(T+1, M)` float64 values on the grid; column 0 (the primary) has no NaN, the other
            series may (a missing value masks that series' observation, E9-T01).
        dt: `(T,)` float64 `Δ_t` in clock units (years for a year clock such as `calendar`, 1 for `steps`).
        nominal_dt: `median(Δ_t)`.
        irregular: `max Δ_t / min Δ_t > 1.5` (always `False` under `steps`).
        clock: the clock `dt` was computed under.
    """

    times: np.ndarray
    values: np.ndarray
    dt: np.ndarray
    nominal_dt: float
    irregular: bool
    clock: Clock

    @property
    def n_steps(self) -> int:
        """Number of observations `T` (one fewer than grid times)."""
        return int(self.dt.shape[0])

    @property
    def end_times(self) -> np.ndarray:
        """`(T,)` observation end times `τ_1 … τ_T` (what `PreparedDataset.times` holds)."""
        return self.times[1:]


@dataclass(frozen=True, eq=False)
class AlignedCovariates:
    """Covariates aligned and lagged onto a grid, in declared order.

    Attributes:
        names: `(K,)` covariate names.
        values: `(T, K)` float64; NaN where no print is usable or the last one is too stale.
        staleness: `(T, K)` int64 grid steps since the print in use became visible (0 = fresh);
            `-1` where no print is usable at all. Reported even where `values` is NaN. A print
            dated before `τ_0` is aged in median grid spacings (rounded up).
    """

    names: tuple[str, ...]
    values: np.ndarray
    staleness: np.ndarray


def _as_ns(times: np.ndarray) -> np.ndarray:
    return np.asarray(times).astype("datetime64[ns]")


def build_grid(
    obs: ObservationSet,
    missing_observations: MissingObservations,
    *,
    clock: str,
    clock_options: Mapping[str, Any] | None = None,
) -> Grid:
    """Build the master grid from the primary instrument's observations.

    Args:
        obs: values `(T0, M)` on strictly increasing `times`; series 0 is the primary and defines the grid.
        missing_observations: `"mask"` keeps every row (non-finite values are masked later); `"drop"`
            removes every row where the primary has a missing (NaN) value (the other
            series keep their NaN and are masked by `Preparer`); `"error"` raises on the first row where
            any series is missing.
        clock: name of a registered clock (`steps`, `calendar`, or an application's, e.g. the finance
            application's own clock): it defines `Δ_t` (ADR-0013, ADR-0015).
        clock_options: the clock's options (`data.clock_options`), validated by its options model.

    Returns:
        The grid with `T + 1` times and `T` values of `Δ_t`.

    Raises:
        ConfigError: unknown `missing_observations` policy or clock, invalid clock options.
        DataError: non-increasing times, a missing value under `"error"`, fewer than two rows left.
    """
    if missing_observations not in ("mask", "drop", "error"):
        raise ConfigError(
            f"missing_observations must be 'mask', 'drop' or 'error', got {missing_observations!r}",
            path="data.preparation.missing_observations",
        )
    clk = get_clock(clock)
    opts = parse_clock_options(clock, clock_options)

    values = np.asarray(obs.values, dtype=np.float64)
    times: np.ndarray
    if obs.times is None:
        if clk.unit != "step":
            raise ConfigError(
                f"observations without a time column need a step clock (data.clock: steps), not {clock!r}",
                path="data.clock",
            )
        times = np.arange(values.shape[0], dtype=np.int64)  # the step index is the time axis
    else:
        times = _as_ns(obs.times)
        if times.shape[0] > 1 and not bool(np.all(np.diff(times) > np.timedelta64(0, "ns"))):
            raise DataError("primary times must be strictly increasing", path="times")

    if missing_observations == "error":
        gaps = np.argwhere(np.isnan(values))
        if gaps.size:
            row, col = gaps[np.argmin(gaps[:, 0])]
            first = str(times[row]) if obs.times is None else np.datetime_as_string(times[row], unit="s")
            name = obs.series_names[col] if col < len(obs.series_names) else str(col)
            what = "primary value" if col == 0 and len(obs.series_names) == 1 else f"value of {name!r}"
            raise DataError(
                f"missing {what} at {first} (missing_observations='error')",
                path="data.preparation.missing_observations",
            )
    missing = (
        np.isnan(values[:, 0]) if missing_observations != "mask" else np.zeros(values.shape[0], dtype=bool)
    )
    if missing.any():
        times, values = times[~missing], values[~missing]
    if times.shape[0] < 2:
        raise DataError("the master grid needs at least two rows with a value", path="values")

    dt = clk.dt(times, times.shape[0] - 1, opts)
    return Grid(
        times=times,
        values=values,
        dt=dt,
        nominal_dt=float(np.median(dt)),
        irregular=False if clk.unit == "step" else is_irregular(dt),
        clock=clock,
    )


def _visible_at(grid_times: np.ndarray, pub_times: np.ndarray) -> np.ndarray:
    """Grid index at which each print first becomes visible (first `τ_j >= print time`).

    Prints dated before `τ_0` get a negative index: their age is extrapolated backwards in units of
    the median grid spacing (rounded up), so an old print is not mistaken for a fresh one at `τ_0`.
    """
    visible = np.searchsorted(grid_times, pub_times, side="left").astype(np.int64)
    before = pub_times < grid_times[0]
    if before.any():
        spacing = int(np.median(np.diff(grid_times).astype(np.int64)))
        age_ns = (grid_times[0] - pub_times[before]).astype(np.int64)
        visible[before] = (-age_ns) // spacing  # = -ceil(age / spacing)
    return visible


def _align_by_row(
    grid: Grid, series: CovariateSeries, raw_values: np.ndarray, lag: int, max_staleness: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row-index alignment for time-less grids: print `j` is dated at grid row `j`.

    Same semantics as the timestamp path with `τ_j = j`: row `t` takes the last non-NaN print at index
    `<= t + 1 - lag`, carried at most `max_staleness` steps.
    """
    n_grid = grid.times.shape[0]
    if raw_values.shape[0] == n_grid - 1:  # one print per observation; the grid has a virtual start time
        raw_values = np.concatenate([[np.nan], raw_values])
    if raw_values.shape[0] != n_grid:
        raise DataError(
            f"covariate {series.name!r}: a series without timestamps needs {n_grid} values "
            f"(the grid length, or one fewer with a virtual start time), got {raw_values.shape[0]}",
            path="data.covariates",
        )
    j = np.arange(n_grid)
    last_pub = np.maximum.accumulate(np.where(np.isnan(raw_values), -1, j))  # last non-NaN print <= j
    has_pub = last_pub >= 0
    pick = np.maximum(last_pub, 0)
    asof_values = np.where(has_pub, raw_values[pick], np.nan)
    asof_stale = np.where(has_pub, j - last_pub, _NEVER)
    source = np.arange(grid.n_steps) + 1 - lag
    in_range = source >= 0
    safe = np.maximum(source, 0)
    values = np.where(in_range, asof_values[safe], np.nan)
    staleness = np.where(in_range, asof_stale[safe], _NEVER).astype(np.int64)
    values = np.where(staleness > max_staleness, np.nan, values)
    return values.astype(np.float64), staleness


def align_covariate(
    grid: Grid,
    series: CovariateSeries,
    lag: int,
    max_staleness: int,
    *,
    index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Align one covariate onto the grid: as-of backward, shift by `lag`, stale-limit.

    Code row `t` (observation `(τ_t, τ_{t+1}]`) receives the last non-NaN print dated at or before
    `τ_{t+1-lag}`; rows with `t + 1 - lag < 0` have none. NaN values in `series` count as "no print";
    of several prints with the same timestamp the last one in input order wins (as `merge_asof`).

    Args:
        grid: the master grid.
        series: the raw covariate (any frequency; monthly series become step functions).
        lag: grid steps of lag, `≥ 1` (`lag = 1`: the value known at the end of the previous step).
        max_staleness: grid steps a print may be carried forward after it became visible.
        index: position in `data.covariates`, used in error paths.

    Returns:
        `values (T,)` float64 and `staleness (T,)` int64 (see `AlignedCovariates`).

    Raises:
        ConfigError: `lag < 1` (`data.covariates[i].lag`) or `max_staleness < 0`.
    """
    if lag < 1:
        raise ConfigError(
            f"covariate {series.name!r}: lag must be >= 1 (no look-ahead), got {lag}",
            path=f"data.covariates[{index}].lag",
        )
    if max_staleness < 0:
        raise ConfigError(
            f"covariate {series.name!r}: max_staleness must be >= 0, got {max_staleness}",
            path=f"data.covariates[{index}].max_staleness",
        )

    raw_values = np.asarray(series.values, dtype=np.float64)
    grid_by_row = np.issubdtype(np.asarray(grid.times).dtype, np.integer)
    if (
        grid_by_row
        and series.times is not None
        and not np.issubdtype(np.asarray(series.times).dtype, np.integer)
    ):
        raise DataError(
            f"covariate {series.name!r} has timestamps but the grid is a step index", path="data.covariates"
        )
    if series.times is None or grid_by_row:
        return _align_by_row(grid, series, raw_values, lag, max_staleness)
    if np.issubdtype(np.asarray(series.times).dtype, np.integer):
        raise DataError(
            f"covariate {series.name!r} has a step index but the grid has timestamps", path="data.covariates"
        )
    keep = ~np.isnan(raw_values)
    pub_times = _as_ns(series.times)[keep]
    pub_values = raw_values[keep]
    order = np.argsort(pub_times, kind="stable")
    pub_times, pub_values = pub_times[order], pub_values[order]

    # As-of at every grid time τ_j: index of the last print with time <= τ_j (-1: none yet).
    last_pub = np.searchsorted(pub_times, grid.times, side="right") - 1
    has_pub = last_pub >= 0
    pick = np.maximum(last_pub, 0)
    visible_at = _visible_at(grid.times, pub_times)
    asof_values = np.where(has_pub, pub_values[pick] if pub_values.size else np.nan, np.nan)
    asof_stale = np.where(
        has_pub, np.arange(grid.times.shape[0]) - (visible_at[pick] if visible_at.size else 0), _NEVER
    )

    # Row t takes grid time j = t + 1 - lag.
    source = np.arange(grid.n_steps) + 1 - lag
    in_range = source >= 0
    safe = np.maximum(source, 0)
    values = np.where(in_range, asof_values[safe], np.nan)
    staleness = np.where(in_range, asof_stale[safe], _NEVER).astype(np.int64)
    values = np.where(staleness > max_staleness, np.nan, values)
    return values.astype(np.float64), staleness


def align_covariates(
    grid: Grid,
    series: Sequence[CovariateSeries],
    lags: Sequence[int],
    max_staleness: Sequence[int],
) -> AlignedCovariates:
    """Align several covariates in declared order (`data.covariates[i]` ↔ position `i`).

    Returns:
        `AlignedCovariates` with `values`/`staleness` of shape `(T, K)`; `K = 0` gives `(T, 0)`.
    """
    if not (len(series) == len(lags) == len(max_staleness)):
        raise ValueError("series, lags and max_staleness must have the same length")
    n_steps, n_cov = grid.n_steps, len(series)
    values = np.empty((n_steps, n_cov), dtype=np.float64)
    staleness = np.empty((n_steps, n_cov), dtype=np.int64)
    for i, (s, lag, stale) in enumerate(zip(series, lags, max_staleness, strict=True)):
        values[:, i], staleness[:, i] = align_covariate(grid, s, lag, stale, index=i)
    return AlignedCovariates(names=tuple(s.name for s in series), values=values, staleness=staleness)
