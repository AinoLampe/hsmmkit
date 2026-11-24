"""E7-T02a: walk-forward fold scheduler."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from regimekit.config.schema import BacktestCfg
from regimekit.evaluation.backtest import Fold, fold_table, schedule_folds
from regimekit.finance.clocks import TRADING_DAYS_PER_YEAR
from regimekit.utils.clock import CALENDAR_DAYS_PER_YEAR
from regimekit.utils.errors import ConfigError, DataError

T = 1000
DT = 1.0 / TRADING_DAYS_PER_YEAR


def cfg(**kw: object) -> BacktestCfg:
    """Backtest config for the ticket's example (T=1000, 2 years, refit 63, embargo 5, horizons 1/5/20)."""
    base: dict[str, object] = {
        "scheme": "expanding",
        "initial_train": 2.0,
        "refit_every_steps": 63,
        "horizons": [1, 5, 20],
        "embargo_steps": 5,
    }
    return BacktestCfg(**{**base, **kw})  # type: ignore[arg-type]


def grid(n: int = T) -> tuple[np.ndarray, np.ndarray]:
    """Daily weekday `times` and trading-clock `dt`."""
    return pd.bdate_range("2015-01-01", periods=n).to_numpy().astype("datetime64[ns]"), np.full(n, DT)


def test_e7_t02a_hand_checked_boundaries() -> None:
    """AC1: T=1000, 504 initial steps, refit 63, embargo 5, max horizon 20."""
    times, dt = grid()
    folds = schedule_folds(times, dt, cfg(), 20)
    assert folds[0] == Fold(0, 0, 504, 509, 572)
    assert folds[1] == Fold(1, 0, 567, 572, 635)
    assert folds[-1].test_end == 980
    assert folds[-1] == Fold(7, 0, 945, 950, 980)  # clipped at T - max_horizon
    assert [f.fold for f in folds] == list(range(8))


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
@pytest.mark.parametrize("embargo", [0, 5, 30])
def test_e7_t02a_no_overlap_and_contiguous_disjoint_tests(scheme: str, embargo: int) -> None:
    """AC2: test rows start after the training rows; consecutive tests touch and never overlap."""
    times, dt = grid()
    folds = schedule_folds(times, dt, cfg(scheme=scheme, rolling_window=1.0, embargo_steps=embargo), 20)
    for f in folds:
        assert f.train_end <= f.test_start
        assert f.test_start - f.train_end == embargo
        assert 0 <= f.train_start < f.train_end
        assert f.test_end <= T - 20
    for prev, nxt in pairwise(folds):
        assert prev.test_end == nxt.test_start  # contiguous (the last fold may be shorter, never overlapping)
        assert nxt.train_end - prev.train_end == 63


def test_e7_t02a_every_origin_has_max_horizon_realized_steps() -> None:
    """The last origin leaves `max_horizon` future rows: `origin + max_h <= T - 1`."""
    times, dt = grid()
    folds = schedule_folds(times, dt, cfg(), 20)
    assert max(o for f in folds for o in f.origins()) + 20 <= T - 1


def test_e7_t02a_rolling_window_has_constant_length() -> None:
    """AC3: rolling keeps the training length constant (+- 1 step) and equal to rolling_window of steps."""
    times, dt = grid()
    folds = schedule_folds(times, dt, cfg(scheme="rolling", rolling_window=1.0), 20)
    lengths = [f.train_end - f.train_start for f in folds]
    assert max(lengths) - min(lengths) <= 1
    assert all(abs(n - TRADING_DAYS_PER_YEAR) <= 1 for n in lengths)
    assert folds[0].train_start == 504 - TRADING_DAYS_PER_YEAR
    assert folds[1].train_start == folds[0].train_start + 63


def test_e7_t02a_rolling_window_over_irregular_time_spans_the_requested_years() -> None:
    """With an irregular `dt` the window spans `rolling_window` of elapsed time, not a row count."""
    rng = np.random.default_rng(0)
    dt = rng.choice([1.0, 1.0, 3.0], size=T) / CALENDAR_DAYS_PER_YEAR
    times = pd.date_range("2015-01-01", periods=T, freq="D").to_numpy().astype("datetime64[ns]")
    folds = schedule_folds(times, dt, cfg(scheme="rolling", rolling_window=1.0, initial_train=2.0), 20)
    cum = np.concatenate([[0.0], np.cumsum(dt)])
    for f in folds:
        span = cum[f.train_end] - cum[f.train_start]
        assert span >= 1.0 - 1e-9
        assert span - dt[f.train_start] < 1.0  # dropping the oldest row would fall below one year


def test_e7_t02a_trailing_fold_without_origins_is_dropped_and_empty_schedules_raise() -> None:
    """AC4: a fold whose test range is empty is dropped; no fold at all is a `ConfigError`."""
    times, dt = grid(600)
    folds = schedule_folds(times, dt, cfg(refit_every_steps=30, embargo_steps=0), 20)
    assert folds[-1].test_end == 580
    assert all(f.test_end > f.test_start for f in folds)
    # 504 + 5 embargo already lies beyond T - max_horizon = 500: nothing to test
    short_times, short_dt = grid(520)
    with pytest.raises(ConfigError):
        schedule_folds(short_times, short_dt, cfg(), 20)


def test_e7_t02a_initial_window_longer_than_the_data_is_a_config_error() -> None:
    """AC4: `initial_train` beyond the span raises with the config path."""
    times, dt = grid(300)
    with pytest.raises(ConfigError) as err:
        schedule_folds(times, dt, cfg(initial_train=2.0), 20)
    assert err.value.path == "evaluation.backtest.initial_train"


def test_e7_t02a_invalid_settings_are_rejected() -> None:
    """Rolling without / with an oversized window, non-positive cadence, bad horizons, bad grids."""
    times, dt = grid()
    with pytest.raises(ConfigError):
        schedule_folds(times, dt, cfg(scheme="rolling"), 20)
    with pytest.raises(ConfigError):
        schedule_folds(times, dt, cfg(scheme="rolling", rolling_window=3.0), 20)
    with pytest.raises(ConfigError):
        schedule_folds(times, dt, cfg(refit_every_steps=0), 20)
    with pytest.raises(ConfigError):
        schedule_folds(times, dt, cfg(embargo_steps=-1), 20)
    with pytest.raises(ConfigError):
        schedule_folds(times, dt, cfg(), 5)  # max_horizon below the configured horizons
    with pytest.raises(DataError):
        schedule_folds(times, dt[:-1], cfg(), 20)


def test_e7_t02a_steps_clock_counts_steps_as_years() -> None:
    """Under `clock: steps` (`dt == 1`) the initial window is `initial_train` rows."""
    times = np.arange(300).astype("datetime64[D]").astype("datetime64[ns]")
    folds = schedule_folds(times, np.ones(300), cfg(initial_train=100.0, embargo_steps=0, horizons=[1, 5]), 5)
    assert folds[0].train_end == 100


def test_e7_t02a_fold_table_carries_the_dates() -> None:
    """`fold_table` has the indices plus the timestamps of the first/last training row and test origin."""
    times, dt = grid()
    folds = schedule_folds(times, dt, cfg(), 20)
    table = fold_table(folds, times)
    assert list(table.columns[:5]) == ["fold", "train_start", "train_end", "test_start", "test_end"]
    assert len(table) == len(folds)
    first = table.iloc[0]
    assert first["train_start_time"] == pd.Timestamp(times[0])
    assert first["train_end_time"] == pd.Timestamp(times[503])
    assert first["test_start_time"] == pd.Timestamp(times[509])
    assert first["test_end_time"] == pd.Timestamp(times[571])
