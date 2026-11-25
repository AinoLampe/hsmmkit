"""E7-T03: a strategy inside the walk-forward backtest (trace, costs, per-fold strategy metrics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regimekit.data.prepare import Preparer
from regimekit.evaluation.backtest import run_backtest, schedule_folds
from regimekit.evaluation.metrics.registry import resolve
from regimekit.finance.config import StrategyCfg
from regimekit.finance.strategies import StrategyHook, make_strategy
from tests.helpers import backtest_fixtures as bf

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

SHORT = {"initial_train": 100, "refit_every": 30, "horizons": (1,), "restarts": 1, "n_sim": 50}
STRATEGY_METRICS = ("sharpe", "max_drawdown", "turnover", "hit_rate")


def _run(tmp_path: Path, strategy_cfg: StrategyCfg):  # type: ignore[no-untyped-def]  # noqa: ANN202
    obs = bf.observations(bf.simulate_returns(199, seed=5))
    cfg = bf.make_config(tmp_path, metrics=("log_score", *STRATEGY_METRICS), **SHORT)  # type: ignore[arg-type]
    cfg.evaluation.strategy = strategy_cfg
    grid = Preparer(cfg.data).fit(obs, [], train_end=obs.times[-1]).transform(obs, [])
    strategy = make_strategy(strategy_cfg, grid.nominal_dt, n_series=1)
    result = run_backtest(
        cfg, lambda: (obs, []), seed=5, hooks=[StrategyHook(strategy, strategy_cfg.cost_bps)]
    )
    folds = schedule_folds(np.asarray(grid.times), np.asarray(grid.dt), cfg.evaluation.backtest, 1)
    return result, grid, folds


def _check_trace(result, grid, folds, cost_bps: float) -> pd.DataFrame:  # type: ignore[no-untyped-def]  # noqa: ANN001
    trace = result.extras.get("strategy")
    assert trace is not None
    origins = np.concatenate([np.arange(f.test_start, f.test_end) for f in folds])
    # the position decided at origin o is held over row o + 1
    assert len(trace) == len(origins)
    np.testing.assert_array_equal(trace["time"].to_numpy(), np.asarray(grid.times)[origins + 1])
    np.testing.assert_array_equal(
        trace["fold"].to_numpy(), np.concatenate([np.full(f.test_end - f.test_start, f.fold) for f in folds])
    )
    held = trace["position"].to_numpy()
    y = np.asarray(grid.observations)[origins + 1, 0]
    steps = np.abs(np.diff(np.concatenate([[0.0], held])))
    np.testing.assert_allclose(trace["gross"], held * y, rtol=0, atol=1e-15)
    np.testing.assert_allclose(trace["turnover"], steps, rtol=0, atol=1e-15)
    np.testing.assert_allclose(trace["return"], held * y - cost_bps * 1e-4 * steps, rtol=0, atol=1e-15)
    np.testing.assert_allclose(trace["equity"], np.exp(np.cumsum(trace["return"])), rtol=0, atol=1e-12)

    # strategy metrics: one row per fold at horizon 0, equal to the metric on that fold's slice
    metrics = result.metrics
    for fold in folds:
        part = trace[trace["fold"] == fold.fold].reset_index(drop=True)
        rows = metrics[(metrics["fold"] == fold.fold) & (metrics["horizon"] == 0)].set_index("metric")[
            "value"
        ]
        for metric in resolve(list(STRATEGY_METRICS)):
            assert rows[metric.name] == metric(extras={"strategy": part}), (fold.fold, metric.name)
    summary = result.summary.set_index(["horizon", "metric"])
    for name in STRATEGY_METRICS:
        assert summary.loc[(0, name), "n"] == len(folds)
    return trace


def test_e7_t03_backtest_collects_long_flat_positions_costs_and_fold_metrics(tmp_path: Path) -> None:
    """`StrategyHook` gives the trace of the positions decided at each origin, with costs."""
    cfg = StrategyCfg(
        type="regime_long_flat", flat_if={"regime": 0, "prob_above": 0.6, "use": "smoothed"}, cost_bps=5.0
    )
    with pytest.warns(UserWarning, match="smoothed"):  # look-ahead request: warns, runs on filtered probs
        result, grid, folds = _run(tmp_path, cfg)
    trace = _check_trace(result, grid, folds, 5.0)
    assert set(np.unique(trace["position"])) <= {0.0, 1.0}
    with pytest.warns(UserWarning, match="smoothed"):
        again, _, _ = _run(tmp_path, cfg)
    pd.testing.assert_frame_equal(
        result.extras["strategy"], again.extras["strategy"], check_exact=True
    )  # bitwise (AC5)
    pd.testing.assert_frame_equal(result.metrics, again.metrics, check_exact=True)


def test_e7_t03_backtest_target_vol_positions_follow_the_forecast_and_stay_capped(tmp_path: Path) -> None:
    """Vol targeting from the origin's one-step forecast: positions in [0, cap], not all equal."""
    cfg = StrategyCfg(type="regime_target_vol", target_vol=0.01, cap=0.8, cost_bps=2.0)
    result, grid, folds = _run(tmp_path, cfg)
    trace = _check_trace(result, grid, folds, 2.0)
    pos = trace["position"].to_numpy()
    assert np.all((pos >= 0.0) & (pos <= 0.8 + 1e-12))
    assert pos.max() > pos.min()  # the size follows the predictive volatility
