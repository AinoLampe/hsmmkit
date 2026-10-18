"""Master calendar: observation grid, `Δ_t` per clock, covariate alignment and lagging (FR-D04, FR-D05).

Grid convention (concept §4.1, §5.6). The master grid is the primary instrument's timestamps after
missing-price handling, `τ_0 < … < τ_T`. Observation `t` (math, 1-based) covers `(τ_{t-1}, τ_t]`,
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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from regimekit.data.contracts import Clock, CovariateSeries, ObservationSet
from regimekit.utils.clock import CALENDAR_DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR, is_irregular
from regimekit.utils.clock import future_grid as future_grid  # re-export (E6-T04a)
from regimekit.utils.errors import ConfigError, DataError

MissingReturns = Literal["drop", "error"]

_NS_PER_DAY = 86_400 * 10**9
_NEVER = -1  # staleness of a row with no usable print (before the first one, or lost to the lag)


@dataclass(frozen=True, eq=False)
class Grid:
    """The master observation grid.

    Attributes:
        times: `(T+1,)` `datetime64[ns]` grid end times `τ_0 … τ_T`.
        prices: `(T+1, M)` float64 primary prices on the grid (no NaN).
        dt: `(T,)` float64 `Δ_t` in clock units (years for `trading`/`calendar`, 1 for `steps`).
        nominal_dt: `median(Δ_t)`.
        irregular: `max Δ_t / min Δ_t > 1.5` (always `False` under `steps`).
        clock: the clock `dt` was computed under.
    """

    times: np.ndarray
    prices: np.ndarray
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


def _grid_dt(times: np.ndarray, clock: Clock, bars_per_day: int) -> np.ndarray:
    n_steps = times.shape[0] - 1
    if clock == "trading":
        return np.full(n_steps, 1.0 / (TRADING_DAYS_PER_YEAR * bars_per_day), dtype=np.float64)
    if clock == "calendar":
        days = np.diff(times).astype(np.int64).astype(np.float64) / _NS_PER_DAY
        return days / CALENDAR_DAYS_PER_YEAR
    if clock == "steps":
        return np.ones(n_steps, dtype=np.float64)
    raise ConfigError(f"unknown clock {clock!r}", path="data.clock")


def build_grid(
    obs: ObservationSet,
    missing_returns: MissingReturns,
    *,
    clock: Clock,
    bars_per_day: int = 1,
) -> Grid:
    """Build the master grid from the primary instrument's observations.

    Args:
        obs: primary prices `(T0, M)` on strictly increasing `times`.
        missing_returns: `"drop"` removes every row where any series has a missing (NaN) price;
            `"error"` raises on the first such row.
        clock: `trading` (`Δ = 1/(252·bars_per_day)` years), `calendar` (`Δ_t = (τ_t - τ_{t-1})`
            in days / 365.25) or `steps` (`Δ_t ≡ 1`, ADR-0013).
        bars_per_day: intraday bars per trading day `H` (trading clock only; 1 = daily).

    Returns:
        The grid with `T + 1` times and `T` values of `Δ_t`.

    Raises:
        ConfigError: unknown `missing_returns` policy or clock, `bars_per_day < 1`.
        DataError: non-increasing times, a missing price under `"error"`, fewer than two rows left.
    """
    if missing_returns not in ("drop", "error"):
        raise ConfigError(
            f"missing_returns must be 'drop' or 'error', got {missing_returns!r}",
            path="data.preparation.missing_returns",
        )
    if bars_per_day < 1:
        raise ConfigError(f"bars_per_day must be >= 1, got {bars_per_day}", path="data.bars_per_day")

    times = _as_ns(obs.times)
    prices = np.asarray(obs.prices, dtype=np.float64)
    if times.shape[0] > 1 and not bool(np.all(np.diff(times) > np.timedelta64(0, "ns"))):
        raise DataError("primary times must be strictly increasing", path="times")

    missing = np.isnan(prices).any(axis=1)
    if missing.any():
        if missing_returns == "error":
            first = np.datetime_as_string(times[np.flatnonzero(missing)[0]], unit="s")
            raise DataError(
                f"missing primary price at {first} (missing_returns='error')",
                path="data.preparation.missing_returns",
            )
        times, prices = times[~missing], prices[~missing]
    if times.shape[0] < 2:
        raise DataError("the master grid needs at least two rows with a price", path="prices")

    dt = _grid_dt(times, clock, bars_per_day)
    return Grid(
        times=times,
        prices=prices,
        dt=dt,
        nominal_dt=float(np.median(dt)),
        irregular=False if clock == "steps" else is_irregular(dt),
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
