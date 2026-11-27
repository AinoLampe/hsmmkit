"""`EMEstimator`: Baum-Welch EM on the expanded engine, a peer of MAP (E13-T02a, ADR-0018; supersedes E5-T06).

Concept refs: Part II §14.6, §15.3. One iteration is one `expanded_estep` (E13-T01a: expected counts without a
`(T, S, S)` array, padded batches, supervision) followed by the closed-form M-steps of `estimation.mstep`; the
whole iteration is one `jit`-compiled function. Components without a closed form (covariate links, families
whose `m_step` answers `None`, non-constant steps) raise `ConfigError` naming `E13-T02b` (ECM).

Convergence: stop when the log-likelihood gain is `<= tol_f * max(1, |loglik|)` or after `max_iter`
iterations.
Monotonicity is asserted when `HSMMKIT_DEBUG=1` (or `debug=True`): a decrease beyond `1e-10` raises
`NumericalError`. Multi-start follows MAP: `cfg.restarts` starts cycled over `cfg.init` with the same keys;
the best final log-likelihood wins. `use_priors` adds the Dirichlet pseudo-counts of the jump rows
(MAP-EM); the other priors are not used (emission M-steps are maximum likelihood).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.data.contracts import PreparedDataset
from hsmmkit.engine.estep import _estep_batch
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.estimation.init import observed_rows, pool_sequences, rolling_kmeans
from hsmmkit.estimation.map import (
    _check_groups,
    _init_theta,
    _per_sequence_table,
    _restart_plan,
    relabel_states,
)
from hsmmkit.estimation.mstep import em_update
from hsmmkit.estimation.objective import canonical_data
from hsmmkit.estimation.results import FitResult
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.statespace.builders import build_expanded
from hsmmkit.utils.clock import is_irregular
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.jax_setup import setup as jax_setup
from hsmmkit.utils.logging import get_logger
from hsmmkit.utils.rng import PURPOSE_EM_INIT, key_for
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EstimationSection
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

_LOG = get_logger("hsmmkit.estimation.em")
MONOTONE_SLACK = 1e-10  # absolute decrease tolerated before debug mode raises (plus 1e-13 relative)


@dataclass(frozen=True)
class EMResult:
    """Outcome of a single EM run (`em_fit`).

    Attributes:
        params: constrained parameters, shaped like `spec.param_specs`.
        uparams: unconstrained vector `(n_free,)` (`ParamManager.flatten`).
        loglik: final log-likelihood (Python float).
        logliks: log-likelihood at the start of each iteration plus the final value.
        n_iter: EM iterations run.
        converged: whether the relative change fell below `tol_f`.
    """

    params: Params
    uparams: Array
    loglik: float
    logliks: tuple[float, ...]
    n_iter: int
    converged: bool


def check_em_supported(spec: ModelSpec, data: PreparedDataset | SequenceBatch | None = None) -> None:
    """Raise `ConfigError` unless every component of `spec` has a closed-form EM update.

    The component checks that need the statistics (`m_step` answering `None`) happen at the first iteration.
    """
    if spec.engine != "expanded":
        raise ConfigError(
            "EM supports the expanded engine only (explicit engine: E13-T03a/b)", path="model.engine"
        )
    if spec.memory_mode != "segments":
        raise ConfigError("EM needs segment memory (not memory_mode: steps)", path="model")
    if spec.duration_conditioning != "current":
        raise ConfigError("EM supports duration_conditioning 'current' only", path="model.transition")
    if spec.initial == "stationary":
        raise ConfigError(
            "initial_state: stationary depends on the jump and duration parameters and has no closed-form EM "
            "update (use uniform or free; ECM: E13-T02b)",
            path="model.initial_state",
        )
    if spec.duration_dependent_jumps or spec.left_censored:
        raise ConfigError("EM does not support duration-dependent jumps or left-censoring", path="model")
    if not hasattr(spec.duration, "from_advance_prob"):
        raise ConfigError(
            f"component {type(spec.duration).__name__} has no closed-form M-step; "
            "available from E13-T02b (ECM)",
            path="model.duration",
        )
    if getattr(spec.duration, "cov_idx", ()):
        raise ConfigError(
            "duration covariates have no closed-form M-step; available from E13-T02b", path="model.duration"
        )
    if getattr(spec.transition, "cov_idx", ()) or any(s.name == "transition.b" for s in spec.param_specs):
        raise ConfigError(
            "transition covariates have no closed-form M-step; available from E13-T02b",
            path="model.transition",
        )
    for s in spec.param_specs:
        if s.constraint == "increasing":
            raise ConfigError(
                f"EM does not keep an 'increasing' ordering ({s.name}); use a free parameter",
                path=s.name,
            )
        if s.tie_group is not None or (not isinstance(s.mode, str) and "tied" in s.mode) or s.mode == "tied":
            raise ConfigError(
                f"tied parameters ({s.name}) have no EM update yet; available from E13-T02b", path=s.name
            )
    if data is not None:
        seqs = data.sequences if isinstance(data, SequenceBatch) else (data,)
        for seq in seqs:
            dt = np.asarray(seq.dt)
            if is_irregular(dt) or not np.allclose(dt, dt[0], rtol=1e-9, atol=0.0):
                raise ConfigError(
                    "EM needs a constant time step (closed-form advance-probability update); "
                    "irregular steps: E13-T02b",
                    path="data",
                )


@partial(jax.jit, static_argnames=("spec", "use_priors"))
def _em_step(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    use_priors: bool,
) -> tuple[Params, Array]:
    """`(theta', loglik(theta))`: one E-step and the closed-form M-step."""
    stats = _estep_batch(theta, spec, space, pb, jnp.asarray(1.0), False)
    return em_update(theta, spec, space, pb, stats, use_priors=use_priors), jnp.sum(stats.loglik)


def _debug_on(debug: bool | None) -> bool:
    return bool(debug) if debug is not None else os.environ.get("HSMMKIT_DEBUG") == "1"


def run_em(
    theta0: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    cfg: EstimationSection,
    *,
    debug: bool | None = None,
) -> tuple[Params, list[float], bool, int]:
    """EM from `theta0` on the padded batch `pb`: `(theta, loglik trace, converged, n_iter)`.

    The trace holds the log-likelihood at the start of every iteration, then the final one.
    """
    check = _debug_on(debug)
    theta = dict(theta0)
    trace: list[float] = []
    converged = False
    n_iter = 0
    for n_iter in range(1, cfg.max_iter + 1):
        new, ll_array = _em_step(theta, spec, space, pb, cfg.use_priors)
        ll = float(ll_array)
        if not np.isfinite(ll):
            raise NumericalError("non-finite log-likelihood during EM", path="loglik")
        if check and trace and ll < trace[-1] - (MONOTONE_SLACK + 1e-13 * abs(trace[-1])):
            raise NumericalError(
                f"EM log-likelihood decreased: {trace[-1]!r} -> {ll!r} at iteration {n_iter}", path="loglik"
            )
        trace.append(ll)
        if len(trace) > 1 and abs(ll - trace[-2]) <= cfg.tol_f * max(abs(ll), 1.0):
            converged = True
            break
        theta = dict(new)
    return theta, trace, converged, n_iter


def _final_loglik(theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch) -> float:
    stats = _estep_batch(theta, spec, space, pb, jnp.asarray(1.0), False)
    return float(jnp.sum(stats.loglik))


class EMEstimator:
    """Registered estimator `"em"`: closed-form Baum-Welch with MAP's multi-start and `FitResult` contract."""

    name = "em"

    def fit(
        self,
        spec: ModelSpec,
        data: PreparedDataset | SequenceBatch,
        cfg: EstimationSection,
        seed: int,
        warm_start: Array | None = None,
    ) -> FitResult:
        """Run `cfg.restarts` EM runs and return the best as a `FitResult(estimator="em")`.

        `data` may be a `SequenceBatch` (the counts are pooled over the sequences). `FitResult.objective` is
        `-loglik` (EM's objective is the likelihood; no prior term); `restarts` has one row per start,
        with its `loglik_trace`.

        Raises:
            ConfigError: a component without a closed-form M-step (see `check_em_supported`).
            NumericalError: a non-finite log-likelihood in every restart, or a decrease in debug mode
                (`HSMMKIT_DEBUG=1`).
        """
        jax_setup(cfg.device)
        batch: SequenceBatch | None = None
        if not isinstance(data, PreparedDataset):
            if len(data.sequences) == 1 and spec.n_groups == 1:
                data = data.sequences[0]
            else:
                batch = data
                _check_groups(spec, batch)
        check_em_supported(spec, batch if batch is not None else data)
        manager = ParamManager(spec.param_specs)
        space = build_expanded(spec)
        segments: np.ndarray | None = None
        if batch is not None:
            pb = batch.padded()
            init_data, segments = pool_sequences(batch)
            first = batch.sequences[0]
        else:
            assert isinstance(data, PreparedDataset)
            pb = SequenceBatch.of(data).padded()
            init_data = observed_rows(data)
            first = data
        plan = _restart_plan(list(cfg.init), cfg.restarts)
        if warm_start is not None:
            if tuple(warm_start.shape) != (manager.n_free,):
                raise ConfigError(
                    f"warm_start has shape {tuple(warm_start.shape)}, expected ({manager.n_free},)",
                    path="estimation.warm_start",
                )
            plan.append("warm_start")
        rows: list[dict[str, object]] = []
        runs: list[tuple[Params, float]] = []
        for i, method in enumerate(plan):
            started = time.perf_counter()
            if method == "warm_start":
                assert warm_start is not None
                theta0 = dict(manager.unflatten(warm_start))
            else:
                theta0 = dict(_init_theta(method, init_data, spec, key_for(seed, i), segments))
            theta, trace, converged, n_iter = run_em(theta0, spec, space, pb, cfg)
            ll = _final_loglik(theta, spec, space, pb)
            runs.append((theta, ll))
            rows.append(
                {
                    "restart": i,
                    "init_method": method,
                    "objective": -ll,
                    "loglik": ll,
                    "n_iter": n_iter,
                    "converged": converged,
                    "wall_s": time.perf_counter() - started,
                    "loglik_trace": [*trace, ll],
                }
            )
        finite = [i for i, (_, ll) in enumerate(runs) if np.isfinite(ll)]
        if not finite:
            raise NumericalError(
                f"every one of {len(runs)} restarts gave a non-finite log-likelihood", path="restarts"
            )
        best = max(finite, key=lambda i: runs[i][1])
        theta, ll = runs[best]
        theta, relabelled = relabel_states(theta, spec, first)
        u = manager.flatten(theta)
        diagnostics: dict[str, Any] = {}
        if batch is not None:
            table = _per_sequence_table(ExpandedHMMEngine(), theta, spec, space, batch)
            ll = float(sum(r["loglik"] for r in table))
            diagnostics["per_sequence"] = table
            diagnostics["initial_groups"] = [g for g in batch.group_names] if spec.n_groups > 1 else None
        elif relabelled:
            ll = float(ExpandedHMMEngine().loglik(theta, spec, space, canonical_data(data)))  # type: ignore[arg-type]
        return FitResult(
            estimator="em",
            params=theta,
            uparams=u,
            objective=-ll,
            loglik=ll,
            restarts=pd.DataFrame(rows),
            converged=bool(rows[best]["converged"]),
            n_iter=int(rows[best]["n_iter"]),  # type: ignore[call-overload]
            std_errors=None,
            diagnostics=diagnostics,
            spec=spec,
            prep_stats=first.prep_stats,
            config_resolved={},
        )


def em_fit(
    spec: ModelSpec,
    data: PreparedDataset,
    cfg: EstimationSection,
    seed: int,
    *,
    theta0: Params | None = None,
    debug: bool = False,
) -> EMResult:
    """One EM run from `theta0` (default: the `rolling_kmeans` start for `seed`); the Part I entry point.

    Raises:
        ConfigError: unsupported model (see `check_em_supported`).
        NumericalError: a non-finite log-likelihood, or a decrease under `debug`.
    """
    check_em_supported(spec, data)
    manager = ParamManager(spec.param_specs)
    space = build_expanded(spec)
    pb = SequenceBatch.of(data).padded()
    start = (
        dict(theta0)
        if theta0 is not None
        else dict(rolling_kmeans(observed_rows(data), spec, key_for(seed, PURPOSE_EM_INIT)))
    )
    theta, trace, converged, n_iter = run_em(start, spec, space, pb, cfg, debug=debug)
    final = _final_loglik(theta, spec, space, pb)
    return EMResult(
        params=theta,
        uparams=manager.flatten(theta),
        loglik=final,
        logliks=(*trace, final),
        n_iter=n_iter,
        converged=converged,
    )


REGISTRY.register("estimator", EMEstimator.name, EMEstimator(), provenance=provenance_of(__name__))
