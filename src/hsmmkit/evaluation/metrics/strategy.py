"""Strategy metrics (E7-T03): Sharpe, maximum drawdown, turnover and hit rate of a strategy trace.

They read `ctx['strategy']`, the table of `evaluation.strategies.strategy_trace` (columns `return`,
`gross`, `dt` and `position` / `position_<m>`), or a slice of it (one fold). Drawdown and Sharpe are
recomputed from the `return` column, so a slice is scored on its own, starting from equity 1.

`sharpe` returns `+inf` / `-inf` for a zero-variance non-zero mean return and `0.0` for an all-zero return
(the ticket's explicit guard, an exception to "no Inf leaves a function").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimekit.evaluation.metrics.registry import metric
from regimekit.utils.errors import ConfigError


def _trace(ctx: dict[str, object]) -> pd.DataFrame:
    trace = ctx.get("strategy")
    if not isinstance(trace, pd.DataFrame) or trace.empty:
        raise ConfigError("this metric needs a non-empty ctx['strategy'] trace", path="evaluation.metrics")
    return trace


def _positions(trace: pd.DataFrame) -> np.ndarray:
    cols = [c for c in trace.columns if c == "position" or str(c).startswith("position_")]
    return trace[cols].to_numpy(dtype=np.float64)


@metric("sharpe", {"strategy"})
def sharpe(**ctx: object) -> float:
    """Annualized Sharpe `mean(r) / std(r) * sqrt(1 / mean(dt))` (sample std; constant return: `+-inf`)."""
    trace = _trace(ctx)
    r = trace["return"].to_numpy(dtype=np.float64)
    periods_per_unit = 1.0 / float(trace["dt"].mean())
    sd = float(np.std(r, ddof=1)) if r.shape[0] > 1 else 0.0
    mean = float(np.mean(r))
    if sd <= 1e-12 * abs(mean) or sd == 0.0:  # constant return: float noise leaves sd ~1e-19
        return 0.0 if mean == 0.0 else float(np.sign(mean) * np.inf)
    return mean / sd * float(np.sqrt(periods_per_unit))


@metric("max_drawdown", {"strategy"})
def max_drawdown(**ctx: object) -> float:
    """Largest peak-to-trough fall of the equity curve, as a fraction of the peak (`>= 0`; starts at 1)."""
    equity = np.exp(np.cumsum(_trace(ctx)["return"].to_numpy(dtype=np.float64)))
    equity = np.concatenate([[1.0], equity])
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


@metric("turnover", {"strategy"})
def turnover(**ctx: object) -> float:
    """Mean `sum_m |change in position|` per unit of clock time (per year on the trading/calendar clocks)."""
    trace = _trace(ctx)
    return float(trace["turnover"].sum() / trace["dt"].sum())


@metric("hit_rate", {"strategy"})
def hit_rate(**ctx: object) -> float:
    """Share of the periods with a non-zero position whose gross return `position . y` is positive.

    `0.0` when the strategy was never invested.
    """
    trace = _trace(ctx)
    invested = np.abs(_positions(trace)).sum(axis=1) > 0.0
    if not invested.any():
        return 0.0
    return float(np.mean(trace["gross"].to_numpy(dtype=np.float64)[invested] > 0.0))
