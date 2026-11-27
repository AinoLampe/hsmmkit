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

from hsmmkit.data.contracts import PreparedDataset
from hsmmkit.engine.batch import BatchedEngine, batched_loglik
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.estimation.init import observed_rows, pool_sequences
from hsmmkit.estimation.objective import canonical_data, make_objective
from hsmmkit.estimation.optimizers import minimize
from hsmmkit.estimation.results import FitResult
from hsmmkit.model.lattice import has_identifiability_guard, post_sort_states
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.statespace.builders import build_expanded, build_explicit
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.jax_setup import setup as jax_setup
from hsmmkit.utils.logging import get_logger
from hsmmkit.utils.rng import key_for

if TYPE_CHECKING:
    from hsmmkit.config.schema import EstimationSection
    from hsmmkit.data.batch import SequenceBatch
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


def _init_theta(
    method: str, data: PreparedDataset, spec: ModelSpec, key: Array, segments: np.ndarray | None = None
) -> Params:
    fn = REGISTRY.get("init", method)
    if segments is not None and method in (
        "rolling_kmeans",
        "labels",
    ):  # pooled batch: windows stay inside a sequence
        result: Params = fn(data, spec, key, segments=segments)
    else:
        result = fn(data, spec, key)
    return result


def _per_sequence_table(
    engine: InferenceEngine, theta: Params, spec: ModelSpec, space: StateSpace, batch: SequenceBatch
) -> list[dict[str, Any]]:
    """`[{id, group, T, loglik}]`: the log-likelihood of every sequence under `theta`."""
    ll = np.asarray(batched_loglik(engine, theta, spec, space, batch.padded(), batch.ids))
    return [
        {"id": sid, "group": seq.group, "T": int(seq.observations.shape[0]), "loglik": float(ll[b])}
        for b, (sid, seq) in enumerate(zip(batch.ids, batch.sequences, strict=True))
    ]


def _check_groups(spec: ModelSpec, batch: SequenceBatch) -> None:
    """The batch's initial-distribution groups must match the model's (`spec.n_groups`)."""
    names = batch.group_names
    if spec.n_groups > 1 and len(names) != spec.n_groups:
        raise ConfigError(
            f"the model has {spec.n_groups} initial-distribution groups but the batch has {len(names)}",
            path="model.initial_groups",
        )


def relabel_states(theta: Params, spec: ModelSpec, first: PreparedDataset) -> tuple[Params, bool]:
    """Fix label switching by ascending scale when the model has no identifiability guard (shared with EM).

    Returns `(theta, changed)`; supervision (`first.labels`) fixes the numbering, so nothing is relabelled.
    """
    sorted_theta, perm = relabel_states_perm(theta, spec, first)
    return sorted_theta, perm is not None


def relabel_states_perm(
    theta: Params, spec: ModelSpec, first: PreparedDataset
) -> tuple[Params, np.ndarray | None]:
    """`relabel_states` that also returns the permutation (`perm[new] = old`, `None` when unchanged)."""
    if not has_identifiability_guard(spec) and first.labels is not None:
        # regime ids of `data.labels` / boundaries refer to the current numbering: do not relabel (E12-T06)
        warnings.warn(
            "no identifiability guard, but data.labels fix the regime numbering: the fitted regimes are not "
            "relabelled by scale",
            UserWarning,
            stacklevel=4,
        )
        return theta, None
    if has_identifiability_guard(spec):
        return theta, None
    scale_name = next(
        (
            name
            for name in theta
            if name in ("emission.scale", "emission.sigma", "emission.s", "emission.chol", "emission.eta")
        ),
        "emission.scale",
    )
    sort_hook = getattr(spec.emission, "default_sort_name", None)
    if sort_hook is not None and sort_hook(theta) is not None:
        scale_name = sort_hook(theta)
    warnings.warn(
        f"no identifiability guard (no fixed or ordered emission location/scale spec): "
        f"relabelling states by ascending {scale_name} to fix label switching",
        UserWarning,
        stacklevel=4,
    )
    sorted_theta, perm = post_sort_states(theta, spec, by=scale_name)
    return sorted_theta, perm


def fit(
    spec: ModelSpec,
    data: PreparedDataset | SequenceBatch,
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

    `data` may be a `SequenceBatch` (E12-T02): the objective is the sum of the per-sequence log-likelihoods
    (`batched_loglik`), the data-driven inits pool the sequences (`rolling_kmeans` windows never cross a
    boundary), and `diagnostics` gains `initial_groups` and the per-sequence table `per_sequence`.
    A `B = 1` batch without initial groups is fitted as the single sequence it holds (bitwise identical).
    """
    jax_setup(cfg.device)
    batch: SequenceBatch | None = None
    if not isinstance(data, PreparedDataset):
        if len(data.sequences) == 1 and spec.n_groups == 1:
            data = data.sequences[0]
        else:
            batch = data
            _check_groups(spec, batch)
    manager = ParamManager(spec.param_specs)
    space: StateSpace
    engine: InferenceEngine
    if spec.engine == "explicit":
        space = build_explicit(spec)
        engine = ExplicitDurationEngine(chunk_size=cfg.chunk_size)
    else:
        space = build_expanded(spec)
        engine = ExpandedHMMEngine(parallel_scan=cfg.parallel_scan)
    segments: np.ndarray | None = None
    if batch is not None:
        objective = make_objective(
            spec, space, BatchedEngine(engine), manager, batch.padded(), use_priors=cfg.use_priors
        )
        init_data, segments = pool_sequences(batch)
        data = batch.sequences[0]  # the preparation statistics and metadata of the batch
    else:
        assert isinstance(data, PreparedDataset)
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
            u0 = manager.flatten(_init_theta(method, init_data, spec, key_for(seed, i), segments))
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
    first = data if isinstance(data, PreparedDataset) else data.sequences[0]
    theta, relabelled = relabel_states(theta, spec, first)
    if relabelled:
        u = manager.flatten(theta)

    # canonical data: the engine's own kernel, without a retrace per walk-forward fold
    diagnostics: dict[str, Any] = {}
    if batch is not None:
        table = _per_sequence_table(engine, theta, spec, space, batch)
        loglik = float(sum(row["loglik"] for row in table))
        diagnostics["per_sequence"] = table
        diagnostics["initial_groups"] = [g for g in batch.group_names] if spec.n_groups > 1 else None
    else:
        loglik = float(engine.loglik(theta, spec, space, canonical_data(data)))
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


class MapEstimator:
    """Registered estimator `"map"`: the multi-start L-BFGS / Adam MAP fit (`fit`)."""

    name = "map"

    def fit(
        self,
        spec: ModelSpec,
        data: PreparedDataset | SequenceBatch,
        cfg: EstimationSection,
        seed: int,
        warm_start: Array | None = None,
    ) -> FitResult:
        """`fit(spec, data, cfg, seed, warm_start)`."""
        return fit(spec, data, cfg, seed, warm_start)


REGISTRY.register("estimator", MapEstimator.name, MapEstimator(), provenance=provenance_of(__name__))
