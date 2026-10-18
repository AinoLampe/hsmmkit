"""Walk-forward fold scheduler (E7-T02a): pure index arithmetic, no model code.

Concept refs: §3.1 FR-E02, §5.7 `BacktestCfg`. All indices are rows of the prepared grid
(`PreparedDataset.times` / `dt`, length `T`), with **exclusive** ends (CLAUDE.md §3.7), i.e. the same
`[start, end)` convention as `PreparedDataset.window`. A forecast origin is a row `o` of
`[test_start, test_end)`: the model is fitted on rows `[train_start, train_end)` and forecasts from the
data `[0, o + 1)`; the realized `Y_h` uses rows `o + 1 .. o + h`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regimekit.utils.errors import ConfigError, DataError

if TYPE_CHECKING:
    from regimekit.config.schema import BacktestCfg

_YEARS_TOL = 1e-9  # elapsed-time comparisons: `504 * (1/252)` must count as 2 years


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold; all indices are grid rows, ends exclusive."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def origins(self) -> range:
        """Forecast origins of the fold: every row of `[test_start, test_end)`."""
        return range(self.test_start, self.test_end)


def schedule_folds(times: np.ndarray, dt: np.ndarray, cfg: BacktestCfg, max_horizon: int) -> list[Fold]:
    """The fold table for a grid of `T` rows.

    The initial training window is the shortest prefix whose elapsed time `cumsum(dt)` reaches
    `initial_train_years`. Fold `k` has `train_end = initial_end + k * refit_every_steps`,
    `test_start = train_end + embargo_steps` and
    `test_end = min(test_start + refit_every_steps, T - max_horizon)`, so every origin has
    `max_horizon` realized future rows. `expanding` folds start at row 0; `rolling` folds start at the
    latest row whose window still spans `rolling_years` (constant length up to one step). Folds with an
    empty test range are dropped.

    Args:
        times: `(T,)` grid end times (only their length is used here).
        dt: `(T,)` `Δ` per row in clock units (years under the trading/calendar clocks).
        cfg: the `evaluation.backtest` section.
        max_horizon: largest forecast horizon (`max(cfg.horizons)`).

    Raises:
        ConfigError: invalid backtest settings, `initial_train_years` longer than the data span, a
            rolling window longer than the initial window, or no fold left.
        DataError: `times` and `dt` disagree in length or `dt` is not positive.
    """
    dt = np.asarray(dt, dtype=np.float64)
    n_rows = int(np.asarray(times).shape[0])
    if dt.shape != (n_rows,) or n_rows == 0 or not bool(np.all(dt > 0)):
        raise DataError("times and dt must be equal-length non-empty (T,) arrays with dt > 0", path="dt")
    _check_cfg(cfg, max_horizon)

    elapsed = np.cumsum(dt)
    if elapsed[-1] < cfg.initial_train_years - _YEARS_TOL:
        raise ConfigError(
            f"initial_train_years={cfg.initial_train_years} exceeds the data span of {elapsed[-1]:.4g}",
            path="evaluation.backtest.initial_train_years",
        )
    initial_end = int(np.searchsorted(elapsed, cfg.initial_train_years - _YEARS_TOL, side="left")) + 1

    if cfg.scheme == "rolling":
        if cfg.rolling_years is None or cfg.rolling_years <= 0:
            raise ConfigError(
                "scheme 'rolling' requires rolling_years > 0", path="evaluation.backtest.rolling_years"
            )
        if cfg.rolling_years > cfg.initial_train_years + _YEARS_TOL:
            raise ConfigError(
                "rolling_years may not exceed initial_train_years (the first window could not be filled)",
                path="evaluation.backtest.rolling_years",
            )
    cum0 = np.concatenate([[0.0], elapsed])  # cum0[e] - cum0[s] = elapsed time of rows [s, e)

    last_origin_end = n_rows - max_horizon
    folds: list[Fold] = []
    k = 0
    while True:
        train_end = initial_end + k * cfg.refit_every_steps
        test_start = train_end + cfg.embargo_steps
        if test_start >= last_origin_end:
            break
        test_end = min(test_start + cfg.refit_every_steps, last_origin_end)
        if cfg.scheme == "expanding":
            train_start = 0
        else:
            assert cfg.rolling_years is not None
            # the latest start whose window [s, train_end) still spans rolling_years
            target = cum0[train_end] - cfg.rolling_years + _YEARS_TOL
            train_start = max(int(np.searchsorted(cum0, target, side="right")) - 1, 0)
        folds.append(Fold(len(folds), train_start, train_end, test_start, test_end))
        k += 1

    if not folds:
        raise ConfigError(
            f"no walk-forward fold fits: training ends at row {initial_end}, embargo {cfg.embargo_steps}, "
            f"max_horizon {max_horizon}, T={n_rows}",
            path="evaluation.backtest",
        )
    return folds


def _check_cfg(cfg: BacktestCfg, max_horizon: int) -> None:
    if cfg.refit_every_steps < 1:
        raise ConfigError("refit_every_steps must be >= 1", path="evaluation.backtest.refit_every_steps")
    if cfg.embargo_steps < 0:
        raise ConfigError("embargo_steps must be >= 0", path="evaluation.backtest.embargo_steps")
    if cfg.initial_train_years <= 0:
        raise ConfigError("initial_train_years must be > 0", path="evaluation.backtest.initial_train_years")
    if max_horizon < 1 or (cfg.horizons and max_horizon < max(cfg.horizons)):
        raise ConfigError(
            f"max_horizon={max_horizon} must be >= 1 and cover horizons {cfg.horizons}",
            path="evaluation.backtest.horizons",
        )


def fold_table(folds: list[Fold], times: np.ndarray) -> pd.DataFrame:
    """The folds as a table: row indices plus the timestamps they refer to.

    Columns: `fold`, `train_start`, `train_end`, `test_start`, `test_end` (grid rows, ends exclusive) and
    `train_start_time`, `train_end_time` (the last training row), `test_start_time`, `test_end_time` (the
    last test origin). `BacktestResult.folds` extends this with `fit_time`.
    """
    stamps = np.asarray(times).astype("datetime64[ns]")
    return pd.DataFrame(
        {
            "fold": [f.fold for f in folds],
            "train_start": [f.train_start for f in folds],
            "train_end": [f.train_end for f in folds],
            "test_start": [f.test_start for f in folds],
            "test_end": [f.test_end for f in folds],
            "train_start_time": stamps[[f.train_start for f in folds]],
            "train_end_time": stamps[[f.train_end - 1 for f in folds]],
            "test_start_time": stamps[[f.test_start for f in folds]],
            "test_end_time": stamps[[f.test_end - 1 for f in folds]],
        }
    )
