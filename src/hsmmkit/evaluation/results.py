"""`BacktestResult` (E1-T05): walk-forward evaluation output."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Walk-forward folds, per-fold metrics, the aggregated summary and the frames of the origin hooks."""

    folds: pd.DataFrame  # fold, train_start, train_end, test_start, test_end, fit_time
    metrics: pd.DataFrame  # long format: fold, horizon, metric, value
    summary: pd.DataFrame  # aggregated per horizon/metric (mean, sd across folds)
    extras: dict[str, pd.DataFrame] = field(default_factory=dict)  # one frame per origin hook, by hook name
