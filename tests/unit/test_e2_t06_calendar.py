"""E2-T06: master grid, `Δ_t` per clock, covariate `merge_asof` alignment, lag ≥ 1, staleness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimekit.data.calendar import (
    AlignedCovariates,
    Grid,
    align_covariate,
    align_covariates,
    build_grid,
)
from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.utils.clock import is_irregular
from regimekit.utils.errors import ConfigError, DataError

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)


def _ns(dates: list[str] | pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(dates).as_unit("ns").values, dtype="datetime64[ns]")


def _obs(times: np.ndarray, prices: np.ndarray) -> ObservationSet:
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim == 1:
        prices = prices[:, None]
    return ObservationSet(
        series_names=tuple(f"S{i}" for i in range(prices.shape[1])),
        times=times,
        values=prices,
        source="test",
        snapshot_id="snap",
        meta={"interval": "1d", "adjusted": True},
    )


def _series(times: np.ndarray, values: list[float] | np.ndarray, name: str = "cov") -> CovariateSeries:
    return CovariateSeries(
        name=name,
        times=times,
        values=np.asarray(values, dtype=np.float64),
        source="test",
        series_id=name,
        snapshot_id="snap",
        frequency="m",
    )


# Good Friday 2024 (US): Thu 2024-03-28 -> Mon 2024-04-01 is a 4-calendar-day gap.
HOLIDAY_DATES = ["2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02"]
GAP_STEP = 3  # dt index of the interval (Thu 03-28, Mon 04-01]


def _holiday_grid(clock: str, bars_per_day: int = 1) -> Grid:
    times = _ns(HOLIDAY_DATES)
    return build_grid(
        _obs(times, np.linspace(100, 105, len(times))),
        "drop",
        clock=clock,
        clock_options={"bars_per_day": bars_per_day} if clock == "trading" else None,
    )  # type: ignore[arg-type]


# --- AC1: Δ per clock on a known holiday span -------------------------------------------------


def test_e2_t06_trading_clock_dt_is_constant_across_holiday() -> None:
    """AC1: the trading clock gives 1/252 on every step, the holiday gap included."""
    grid = _holiday_grid("trading")
    assert grid.dt.shape == (len(HOLIDAY_DATES) - 1,)
    np.testing.assert_array_equal(grid.dt, np.full(5, 1.0 / 252.0))
    assert grid.nominal_dt == 1.0 / 252.0
    assert grid.irregular is False
    assert grid.clock == "trading"


def test_e2_t06_trading_clock_intraday_bars() -> None:
    """Trading clock with H bars per day gives 1/(252 H)."""
    grid = _holiday_grid("trading", bars_per_day=8)
    np.testing.assert_array_equal(grid.dt, np.full(5, 1.0 / (252.0 * 8)))


def test_e2_t06_calendar_clock_dt_shows_the_holiday_gap() -> None:
    """AC1: the calendar clock gives 4/365.25 across the Good-Friday gap."""
    grid = _holiday_grid("calendar")
    expected = np.array([1.0, 1.0, 1.0, 4.0, 1.0]) / 365.25
    np.testing.assert_allclose(grid.dt, expected, rtol=0, atol=1e-15)
    assert grid.dt[GAP_STEP] == pytest.approx(4.0 / 365.25, abs=1e-15)
    assert grid.nominal_dt == pytest.approx(1.0 / 365.25, abs=1e-15)
    assert grid.irregular is True  # 4 / 1 > 1.5


def test_e2_t06_steps_clock_dt_is_one_and_regular() -> None:
    """AC1: the steps clock gives 1.0 everywhere and is never irregular."""
    grid = _holiday_grid("steps")
    np.testing.assert_array_equal(grid.dt, np.ones(5))
    assert grid.nominal_dt == 1.0
    assert grid.irregular is False
    assert grid.dt.dtype == np.float64


def test_e2_t06_calendar_clock_uses_fractional_days_intraday() -> None:
    """Calendar Δ uses fractional days from nanosecond differences."""
    times = _ns(["2024-01-02 09:00", "2024-01-02 15:00", "2024-01-03 09:00"])
    grid = build_grid(_obs(times, [1.0, 2.0, 3.0]), "drop", clock="calendar")
    np.testing.assert_allclose(grid.dt, np.array([0.25, 0.75]) / 365.25, rtol=0, atol=1e-15)


def test_e2_t06_is_irregular_threshold() -> None:
    """`is_irregular` is strict `max/min > 1.5`."""
    assert is_irregular(np.array([1.0, 1.5, 1.0])) is False
    assert is_irregular(np.array([1.0, 1.51, 1.0])) is True
    assert is_irregular(np.ones(3)) is False
    for bad in (np.array([]), np.array([1.0, 0.0])):
        with pytest.raises(DataError):
            is_irregular(bad)


# --- master grid: missing prices ---------------------------------------------------------------


def test_e2_t06_grid_drops_rows_with_missing_price() -> None:
    """`drop` removes rows with a missing primary price."""
    times = _ns(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    grid = build_grid(_obs(times, [1.0, np.nan, 3.0, 4.0]), "drop", clock="trading")
    np.testing.assert_array_equal(grid.times, times[[0, 2, 3]])
    np.testing.assert_array_equal(grid.values[:, 0], [1.0, 3.0, 4.0])
    assert grid.n_steps == 2
    assert grid.dt.shape == (2,)


def test_e2_t06_grid_drops_row_if_primary_series_missing() -> None:
    """With M > 1 the primary (first) series defines the grid (E9-T01 supersedes "any series")."""
    times = _ns(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, 30.0]])
    grid = build_grid(_obs(times, prices), "drop", clock="trading")
    np.testing.assert_array_equal(grid.times, times[[0, 2]])
    assert grid.values.shape == (2, 2)


def test_e2_t06_grid_missing_error_policy_names_first_timestamp() -> None:
    """`error` raises DataError naming the first offending timestamp."""
    times = _ns(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    with pytest.raises(DataError, match="2024-01-03") as info:
        build_grid(_obs(times, [1.0, np.nan, 3.0, np.nan]), "error", clock="trading")
    assert info.value.path == "data.preparation.missing_observations"


def test_e2_t06_grid_needs_two_rows() -> None:
    """A grid needs at least two priced rows (T >= 1)."""
    times = _ns(["2024-01-02", "2024-01-03"])
    with pytest.raises(DataError, match="at least two"):
        build_grid(_obs(times, [1.0, np.nan]), "drop", clock="trading")


def test_e2_t06_grid_rejects_non_increasing_times() -> None:
    """Non-increasing primary times are a DataError."""
    times = _ns(["2024-01-03", "2024-01-02", "2024-01-04"])
    with pytest.raises(DataError, match="strictly increasing"):
        build_grid(_obs(times, [1.0, 2.0, 3.0]), "drop", clock="trading")


def test_e2_t06_grid_rejects_bad_policy_and_bars() -> None:
    """Unknown missing policy and bars_per_day < 1 are ConfigErrors with paths."""
    times = _ns(["2024-01-02", "2024-01-03"])
    with pytest.raises(ConfigError) as info:
        build_grid(_obs(times, [1.0, 2.0]), "ffill", clock="trading")  # type: ignore[arg-type]
    assert info.value.path == "data.preparation.missing_observations"
    with pytest.raises(ConfigError) as info:
        build_grid(_obs(times, [1.0, 2.0]), "drop", clock="trading", clock_options={"bars_per_day": 0})
    assert info.value.path == "data.clock_options.bars_per_day"
    with pytest.raises(ConfigError) as info:
        build_grid(_obs(times, [1.0, 2.0]), "drop", clock="hourly")  # type: ignore[arg-type]
    assert info.value.path == "data.clock"


# --- AC2: monthly covariate becomes a lagged step function -------------------------------------

DAILY = pd.bdate_range("2024-01-02", "2024-03-29")  # 64 business days, no holiday calendar
J_JAN, J_FEB, J_MAR = 21, 42, 63  # grid indices of 2024-01-31, 2024-02-29, 2024-03-29
MONTHLY = _ns(["2023-12-29", "2024-01-31", "2024-02-29", "2024-03-29"])
V_DEC, V_JAN, V_FEB, V_MAR = 10.0, 20.0, 30.0, 40.0
DEC_AGE = 4  # Dec 29 -> τ_0 = Jan 2 is 4 days = 4 median grid spacings


def _daily_grid() -> Grid:
    times = _ns(DAILY)
    return build_grid(_obs(times, np.linspace(100, 110, len(times))), "drop", clock="trading")


def test_e2_t06_fixture_indices() -> None:
    """Sanity check of the hand-counted month-end grid indices."""
    grid = _daily_grid()
    assert grid.times[J_JAN] == np.datetime64("2024-01-31", "ns")
    assert grid.times[J_FEB] == np.datetime64("2024-02-29", "ns")
    assert grid.times[J_MAR] == np.datetime64("2024-03-29", "ns")
    assert grid.n_steps == 63


@pytest.mark.parametrize("lag", [1, 2, 5])
def test_e2_t06_monthly_covariate_is_lagged_step_function(lag: int) -> None:
    """AC2: a monthly series becomes a step function shifted by exactly `lag` steps."""
    grid = _daily_grid()
    values, staleness = align_covariate(grid, _series(MONTHLY, [V_DEC, V_JAN, V_FEB, V_MAR]), lag, 30)
    assert values.shape == (63,)
    assert staleness.shape == (63,)
    assert staleness.dtype == np.int64
    # Row t (code index, observation (τ_t, τ_{t+1}]) holds the last publication at or before τ_{t+1-lag}.
    first_valid = lag - 1
    assert np.all(np.isnan(values[:first_valid]))
    assert np.all(staleness[:first_valid] == -1)
    first_jan = J_JAN + lag - 1
    first_feb = J_FEB + lag - 1
    np.testing.assert_array_equal(values[first_valid:first_jan], V_DEC)
    np.testing.assert_array_equal(values[first_jan:first_feb], V_JAN)
    np.testing.assert_array_equal(values[first_feb:], V_FEB)  # March print at τ_63 is never usable
    # Staleness restarts at 0 on the step a new print becomes usable.
    np.testing.assert_array_equal(
        staleness[first_valid:first_jan], DEC_AGE + np.arange(first_jan - first_valid)
    )
    assert staleness[first_jan] == 0
    assert staleness[first_feb] == 0
    assert staleness[-1] == 63 - first_feb - 1


def test_e2_t06_lag_one_hand_indices() -> None:
    """AC2: exact rows around the January print with lag 1."""
    grid = _daily_grid()
    values, _ = align_covariate(grid, _series(MONTHLY, [V_DEC, V_JAN, V_FEB, V_MAR]), 1, 30)
    # Observation (τ_20, τ_21] = (Jan 30, Jan 31]: the Jan print (dated Jan 31) is not yet usable.
    assert values[20] == V_DEC
    # Observation (τ_21, τ_22] = (Jan 31, Feb 1]: the Jan print is usable.
    assert values[21] == V_JAN
    assert values[41] == V_JAN
    assert values[42] == V_FEB


def test_e2_t06_nan_values_are_not_publications() -> None:
    """A NaN print does not replace the previous value."""
    grid = _daily_grid()
    values, staleness = align_covariate(grid, _series(MONTHLY, [V_DEC, np.nan, V_FEB, V_MAR]), 1, 60)
    np.testing.assert_array_equal(values[:J_FEB], V_DEC)
    assert staleness[J_FEB - 1] == DEC_AGE + J_FEB - 1


def test_e2_t06_publications_between_grid_points_use_latest() -> None:
    """Several prints between grid times: the latest one at or before τ_j wins."""
    grid = _daily_grid()
    times = _ns(["2024-01-05 03:00", "2024-01-05 20:00", "2024-01-06"])  # Fri 03:00, Fri 20:00, Sat
    values, staleness = align_covariate(grid, _series(times, [1.0, 2.0, 3.0]), 1, 30)
    # τ_3 = Fri 00:00 precedes all prints; τ_4 = Mon 00:00 -> Saturday's print is the latest.
    assert np.all(np.isnan(values[:4]))
    assert values[4] == 3.0
    assert staleness[4] == 0


def test_e2_t06_align_covariates_stacks_in_order() -> None:
    """`align_covariates` stacks series in declared order with their own lags."""
    grid = _daily_grid()
    a = _series(MONTHLY, [V_DEC, V_JAN, V_FEB, V_MAR], name="a")
    b = _series(_ns(DAILY), np.arange(len(DAILY), dtype=float), name="b")
    aligned = align_covariates(grid, [a, b], lags=[1, 2], max_staleness=[30, 5])
    assert isinstance(aligned, AlignedCovariates)
    assert aligned.names == ("a", "b")
    assert aligned.values.shape == (63, 2)
    assert aligned.staleness.shape == (63, 2)
    # Daily series with lag 2: row t holds the value dated τ_{t-1}.
    assert np.isnan(aligned.values[0, 1])
    np.testing.assert_array_equal(aligned.values[1:, 1], np.arange(62, dtype=float))
    np.testing.assert_array_equal(aligned.staleness[1:, 1], 0)


def test_e2_t06_align_covariates_empty() -> None:
    """No covariates gives (T, 0) blocks."""
    aligned = align_covariates(_daily_grid(), [], lags=[], max_staleness=[])
    assert aligned.names == ()
    assert aligned.values.shape == (63, 0)
    assert aligned.staleness.shape == (63, 0)


def test_e2_t06_align_covariates_length_mismatch() -> None:
    """Mismatched argument lengths are rejected."""
    with pytest.raises(ValueError, match="same length"):
        align_covariates(_daily_grid(), [], lags=[1], max_staleness=[])


# --- AC3: lag 0 and max_staleness --------------------------------------------------------------


def test_e2_t06_lag_zero_raises_config_error() -> None:
    """AC3: lag 0 raises ConfigError at data.covariates[i].lag."""
    with pytest.raises(ConfigError) as info:
        align_covariate(_daily_grid(), _series(MONTHLY, [1.0, 2.0, 3.0, 4.0]), 0, 5)
    assert info.value.path == "data.covariates[0].lag"


def test_e2_t06_lag_zero_path_names_covariate_index() -> None:
    """AC3: the error path names the offending covariate index."""
    grid = _daily_grid()
    a = _series(MONTHLY, [1.0, 2.0, 3.0, 4.0], name="a")
    with pytest.raises(ConfigError) as info:
        align_covariates(grid, [a, a], lags=[1, 0], max_staleness=[5, 5])
    assert info.value.path == "data.covariates[1].lag"


def test_e2_t06_negative_max_staleness_raises() -> None:
    """Negative max_staleness is a ConfigError."""
    with pytest.raises(ConfigError) as info:
        align_covariate(_daily_grid(), _series(MONTHLY, [1.0, 2.0, 3.0, 4.0]), 1, -1, index=2)
    assert info.value.path == "data.covariates[2].max_staleness"


def test_e2_t06_staleness_exceeded_gives_nan_and_reports_counter() -> None:
    """AC3: values older than max_staleness steps are NaN; the counter is still reported."""
    grid = _daily_grid()
    values, staleness = align_covariate(grid, _series(MONTHLY, [V_DEC, V_JAN, V_FEB, V_MAR]), 1, 5)
    # Staleness counts grid steps since the print became usable, NaN once it exceeds 5.
    np.testing.assert_array_equal(values[:2], V_DEC)  # staleness 4, 5
    assert np.all(np.isnan(values[2:J_JAN]))
    np.testing.assert_array_equal(staleness[:J_JAN], DEC_AGE + np.arange(J_JAN))
    np.testing.assert_array_equal(values[J_JAN : J_JAN + 6], V_JAN)
    assert np.all(np.isnan(values[J_JAN + 6 : J_FEB]))
    assert staleness[J_FEB - 1] == J_FEB - 1 - J_JAN


def test_e2_t06_max_staleness_zero_keeps_only_fresh_prints() -> None:
    """max_staleness 0 keeps only the steps where a print is fresh."""
    grid = _daily_grid()
    values, staleness = align_covariate(grid, _series(MONTHLY, [V_DEC, V_JAN, V_FEB, V_MAR]), 1, 0)
    fresh = staleness == 0
    assert np.flatnonzero(fresh).tolist() == [J_JAN, J_FEB]
    assert np.all(np.isnan(values[~fresh]))


def test_e2_t06_print_before_grid_start_is_aged_not_fresh() -> None:
    """A print long before τ_0 is aged in median grid spacings, so it is not used as fresh."""
    grid = _daily_grid()
    values, staleness = align_covariate(grid, _series(_ns(["2023-12-02"]), [5.0]), 1, 40)
    # Dec 2 -> Jan 2 is 31 days = 31 median (1-day) spacings.
    np.testing.assert_array_equal(staleness, 31 + np.arange(63))
    np.testing.assert_array_equal(values[:10], 5.0)
    assert np.all(np.isnan(values[10:]))


def test_e2_t06_matches_pandas_merge_asof_then_shift() -> None:
    """On NaN-free data the as-of step equals `pd.merge_asof(direction="backward")` shifted by lag."""
    rng = np.random.default_rng(0)
    grid = _daily_grid()
    offsets_h = np.sort(rng.choice(np.arange(-72, 24 * 90), size=40, replace=False))
    pub = (np.datetime64("2024-01-02T00:00", "ns") + offsets_h * np.timedelta64(1, "h")).astype(
        "datetime64[ns]"
    )
    vals = rng.normal(size=40)
    left = pd.DataFrame({"t": grid.times})
    right = pd.DataFrame({"t": pub, "v": vals})
    asof = pd.merge_asof(left, right, on="t", direction="backward")["v"].to_numpy()
    for lag in (1, 3):
        values, _ = align_covariate(grid, _series(pub, vals), lag, 10**6)
        expected = np.concatenate([np.full(lag - 1, np.nan), asof[: grid.n_steps + 1 - lag]])
        np.testing.assert_array_equal(values, expected)


def test_e2_t06_grid_end_times() -> None:
    """`end_times` drops τ_0 and has one entry per observation."""
    grid = _daily_grid()
    np.testing.assert_array_equal(grid.end_times, grid.times[1:])
    assert grid.end_times.shape == (grid.n_steps,)


def test_e2_t06_covariate_series_entirely_after_grid() -> None:
    """A series starting after the grid yields NaN and staleness -1 everywhere."""
    grid = _daily_grid()
    values, staleness = align_covariate(grid, _series(_ns(["2025-01-01"]), [1.0]), 1, 5)
    assert np.all(np.isnan(values))
    assert np.all(staleness == -1)
