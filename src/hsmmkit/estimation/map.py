"""MAP estimation: multi-start orchestration -> `FitResult` (E5-T03).

`fit(spec, data, cfg, seed) -> FitResult` (CLAUDE.md §3.1) builds the engine/state-space from `spec`
itself (`spec.engine` picks `expanded`/`explicit`), runs `cfg.restarts` independent `minimize` (E5-T02)
calls from starting points cycled over `cfg.init`'s strategies, and keeps the best finite run.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.engine.explicit_duration import ExplicitDurationEngine
from regimekit.estimation.init import lattice_jitter, random, rollvol_kmeans
from regimekit.estimation.objective import make_objective
from regimekit.estimation.optimizers import minimize
from regimekit.estimation.results import FitResult
from regimekit.model.lattice import has_identifiability_guard, post_sort_states
from regimekit.model.params.manager import ParamManager
from regimekit.statespace.builders import build_expanded, build_explicit
from regimekit.utils.errors import NumericalError
from regimekit.utils.jax_setup import setup as jax_setup
from regimekit.utils.logging import get_logger
from regimekit.utils.rng import key_for

if TYPE_CHECKING:
    from regimekit.config.schema import EstimationSection
    from regimekit.data.contracts import PreparedDataset
    from regimekit.engine.base import InferenceEngine
    from regimekit.model.params.manager import Params
    from regimekit.model.spec import ModelSpec
    from regimekit.statespace.index import StateSpace
    from regimekit.utils.typing import Array

_LOG = get_logger("regimekit.estimation.map")

_DATA_STRATEGIES = {"rollvol_kmeans": rollvol_kmeans, "random": random}
_SPEC_ONLY_STRATEGIES = {"lattice_jitter": lattice_jitter}


def _restart_plan(strategies: list[str], restarts: int) -> list[str]:
    """`strategies` cycled over `restarts` starts; the first strategy absorbs the remainder."""
    if not strategies:
        strategies = ["random"]
    reps, remainder = divmod(restarts, len(strategies))
    counts = [reps] * len(strategies)
    counts[0] += remainder
    return [name for name, count in zip(strategies, counts, strict=True) for _ in range(count)]


def _init_theta(strategy: str, data: PreparedDataset, spec: ModelSpec, key: Array) -> Params:
    if strategy in _DATA_STRATEGIES:
        return _DATA_STRATEGIES[strategy](data, spec, key)
    return _SPEC_ONLY_STRATEGIES[strategy](spec, key)


def fit(spec: ModelSpec, data: PreparedDataset, cfg: EstimationSection, seed: int) -> FitResult:
    """Run `cfg.restarts` MAP minimizations and return the best (finite-objective) `FitResult`.

    Restarts run sequentially, in a deterministic order, each with its own `key_for(seed, restart_idx)`.
    If no per-state emission location/scale spec identifies the labelling (`has_identifiability_guard`),
    the winner is relabelled by ascending `emission.sigma` (`post_sort_states`) with a warning.
    """
    jax_setup(cfg.device)
    manager = ParamManager(spec.param_specs)
    space: StateSpace
    engine: InferenceEngine
    if spec.engine == "explicit":
        space = build_explicit(spec)
        engine = ExplicitDurationEngine(chunk_size=cfg.chunk_size)
    else:
        space = build_expanded(spec)
        engine = ExpandedHMMEngine(parallel_scan=cfg.parallel_scan)
    objective = make_objective(spec, space, engine, manager, data, use_priors=cfg.use_priors)

    plan = _restart_plan(list(cfg.init), cfg.restarts)
    rows: list[dict[str, object]] = []
    records = []
    for i, strategy in enumerate(plan):
        theta0 = _init_theta(strategy, data, spec, key_for(seed, i))
        u0 = manager.flatten(theta0)
        record = minimize(objective, u0, cfg, _LOG)
        records.append(record)
        rows.append(
            {
                "restart": i,
                "init_strategy": strategy,
                "objective": record.objective,
                "loglik": record.loglik,
                "n_iter": record.n_iter,
                "converged": record.converged,
                "wall_s": record.wall_s,
            }
        )

    finite = [i for i, r in enumerate(records) if np.isfinite(r.objective)]
    if not finite:
        raise NumericalError(
            f"every one of {len(records)} restarts produced a non-finite objective",
            path="estimation.restarts",
        )
    best_idx = min(finite, key=lambda i: records[i].objective)
    best = records[best_idx]

    theta = manager.unflatten(best.u)
    u = best.u
    if not has_identifiability_guard(spec):
        scale_name = next(
            (name for name in theta if name in ("emission.sigma", "emission.s")), "emission.sigma"
        )
        warnings.warn(
            f"no identifiability guard (no fixed or ordered emission location/scale spec): "
            f"relabelling states by ascending {scale_name} to fix label switching",
            UserWarning,
            stacklevel=2,
        )
        theta, _ = post_sort_states(theta, spec, by=scale_name)
        u = manager.flatten(theta)

    loglik = float(engine.loglik(theta, spec, space, data))

    return FitResult(
        estimator="map",
        params=theta,
        uparams=u,
        objective=best.objective,
        loglik=loglik,
        restarts=pd.DataFrame(rows),
        converged=best.converged,
        n_iter=best.n_iter,
        std_errors=None,
        diagnostics={},
        spec=spec,
        prep_stats=data.prep_stats,
        config_resolved={},
    )
