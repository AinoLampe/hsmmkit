"""`FitResult` (E1-T05): the outcome of a MAP fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from regimekit.data.contracts import PrepStats
from regimekit.model.params import Params
from regimekit.model.spec import ModelSpec
from regimekit.utils.typing import Array


@dataclass(frozen=True)
class FitResult:
    """The result of an estimation run (`estimator` is always `"map"` in 0.1.0)."""

    estimator: Literal["map"]
    params: Params
    uparams: Array
    objective: float
    loglik: float
    restarts: pd.DataFrame  # restart, init_strategy, objective, loglik, n_iter, converged, wall_s
    converged: bool
    n_iter: int
    std_errors: dict[str, Array] | None
    diagnostics: dict[str, Any]
    spec: ModelSpec
    prep_stats: PrepStats
    config_resolved: dict[str, Any]
