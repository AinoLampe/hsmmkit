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
    """The result of an estimation run (`estimator`: `"map"`, `"em"`, `"gem"`, `"vb"` or `"mcmc"`)."""

    estimator: Literal["map", "em", "gem", "vb", "mcmc"]
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
    posterior_samples: Any = None
    """ArviZ posterior draws (`InferenceData` / `DataTree`) of an MCMC fit; not persisted."""
    pruned_states: tuple[int, ...] = ()
    """Regimes with expected occupancy below `estimation.vb.prune_below` (VB; the model keeps them)."""

    @property
    def elbo(self) -> float | None:
        """The final ELBO of a variational fit (`-objective`), `None` for every other estimator."""
        return -self.objective if self.estimator == "vb" else None
