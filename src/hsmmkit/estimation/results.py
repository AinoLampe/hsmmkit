"""`FitResult` (E1-T05): the outcome of a MAP fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from hsmmkit.data.contracts import PrepStats
from hsmmkit.model.params import Params
from hsmmkit.model.spec import ModelSpec
from hsmmkit.utils.typing import Array


@dataclass(frozen=True)
class FitResult:
    """The result of an estimation run (`estimator`: `"map"`, `"em"` or `"gem"`)."""

    estimator: Literal["map", "em", "gem", "vb"]
    params: Params
    uparams: Array
    objective: float
    loglik: float
    restarts: pd.DataFrame  # restart, init_method, objective, loglik, n_iter, converged, wall_s
    converged: bool
    n_iter: int
    std_errors: dict[str, Array] | None
    diagnostics: dict[str, Any]
    spec: ModelSpec
    prep_stats: PrepStats
    config_resolved: dict[str, Any]
