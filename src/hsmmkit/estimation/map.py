"""MAP estimation: multi-start orchestration -> `FitResult` (E5-T03).

`fit(spec, data, cfg, seed) -> FitResult` (CLAUDE.md §3.1) builds the engine/state-space from `spec`
itself (`spec.engine` picks `expanded`/`explicit`), runs `cfg.restarts` independent `minimize` (E5-T02)
calls from starting points cycled over `cfg.init`'s methods, and keeps the best finite run.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.estimation.init import observed_rows
from hsmmkit.estimation.objective import canonical_data, make_objective
from hsmmkit.estimation.optimizers import minimize
from hsmmkit.estimation.results import FitResult
from hsmmkit.model.lattice import has_identifiability_guard, post_sort_states
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.plugins import REGISTRY
from hsmmkit.statespace.builders import build_expanded, build_explicit
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.jax_setup import setup as jax_setup
from hsmmkit.utils.logging import get_logger
from hsmmkit.utils.rng import key_for

if TYPE_CHECKING:
    from hsmmkit.config.schema import EstimationSection
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.engine.base import InferenceEngine
    from hsmmkit.model.params.manager import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace
    from hsmmkit.utils.typing import Array

_LOG = get_logger("hsmmkit.estimation.map")

_WARM_START = "warm_start"


def _restart_plan(methods: list[str], restarts: int) -> list[str]:
    """`methods` cycled over `restarts` starts; the first method absorbs the remainder."""
    if not methods:
        methods = ["random"]
    reps, remainder = divmod(restarts, len(methods))
    counts = [reps] * len(methods)
    counts[0] += remainder
    return [name for name, count in zip(methods, counts, strict=True) for _ in range(count)]


def _init_theta(method: str, data: PreparedDataset, spec: ModelSpec, key: Array) -> Params:
    result: Params = REGISTRY.get("init", method)(data, spec, key)
    return result


def fit(
    spec: ModelSpec,
    data: PreparedDataset,
    cfg: EstimationSection,
    seed: int,
    warm_start: Array | None = None,
) -> FitResult:
    """Run `cfg.restarts` MAP minimizations and return the best (finite-objective) `FitResult`.

    Restarts run sequentially, in a deterministic order, each with its own `key_for(seed, restart_idx)`.
    `warm_start` (unconstrained parameters `(n_free,)`, e.g. the previous walk-forward fold's optimum)
    adds one *extra* restart, last, with `init_method == "warm_start"`; the `cfg.restarts` regular
    restarts and their keys are unchanged.
    If no per-state emission location/scale spec identifies the labelling (`has_identifiability_guard`),
    the winner is relabelled by ascending `emission.scale` (`post_sort_states`) with a warning.
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

    init_data = observed_rows(data)  # masked rows (walk-forward padding, gaps) must not bias the inits
    plan = _restart_plan(list(cfg.init), cfg.restarts)
    if warm_start is not None:
        if tuple(warm_start.shape) != (manager.n_free,):
            raise ConfigError(
                f"warm_start has shape {tuple(warm_start.shape)}, expected ({manager.n_free},)",
                path="estimation.warm_start",
            )
        plan.append(_WARM_START)
    rows: list[dict[str, object]] = []
    records = []
    for i, method in enumerate(plan):
        if method == _WARM_START:
            assert warm_start is not None
            u0 = warm_start
        else:
            u0 = manager.flatten(_init_theta(method, init_data, spec, key_for(seed, i)))
        record = minimize(objective, u0, cfg, _LOG)
        records.append(record)
        rows.append(
            {
                "restart": i,
                "init_method": method,
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
            (
                name
                for name in theta
                if name in ("emission.scale", "emission.sigma", "emission.s", "emission.chol", "emission.eta")
            ),
            "emission.scale",
        )
        warnings.warn(
            f"no identifiability guard (no fixed or ordered emission location/scale spec): "
            f"relabelling states by ascending {scale_name} to fix label switching",
            UserWarning,
            stacklevel=2,
        )
        theta, _ = post_sort_states(theta, spec, by=scale_name)
        u = manager.flatten(theta)

    # canonical data: the engine's own kernel, without a retrace per walk-forward fold
    loglik = float(engine.loglik(theta, spec, space, canonical_data(data)))

    diagnostics: dict[str, Any] = {}
    spectral_radius = getattr(spec.emission, "spectral_radius", None)
    if spectral_radius is not None:  # autoregressive emission: stationarity is a diagnostic, not a constraint
        radius = np.asarray(spectral_radius(theta))
        diagnostics["ar_spectral_radius"] = radius.tolist()
        if bool(np.any(radius >= 1.0)):
            warnings.warn(
                f"non-stationary autoregression: spectral radius {radius.round(4).tolist()} "
                "(>= 1 in regimes " + str(np.flatnonzero(radius >= 1.0).tolist()) + ")",
                UserWarning,
                stacklevel=2,
            )

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
        diagnostics=diagnostics,
        spec=spec,
        prep_stats=data.prep_stats,
        config_resolved={},
    )
