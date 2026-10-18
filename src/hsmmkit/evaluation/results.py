"""`BacktestResult` (E1-T05): walk-forward evaluation output."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Walk-forward folds, per-fold metrics, an optional strategy trace, and aggregated summary."""

    folds: pd.DataFrame  # fold, train_start, train_end, test_start, test_end, fit_time
    metrics: pd.DataFrame  # long format: fold, horizon, metric, value
    strategy: pd.DataFrame | None  # time, position, return, equity, turnover
    summary: pd.DataFrame  # aggregated per horizon/metric (mean, sd across folds)
