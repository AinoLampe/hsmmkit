"""`EMEstimator`: Baum-Welch EM / ECM on the expanded engine, a peer of MAP (E13-T02a/b, ADR-0018).

Concept refs: Part II §14.6, §15.3. One iteration is one `expanded_estep` (E13-T01a: expected counts without a
`(T, S, S)` array, padded batches, supervision) followed by the M-steps of `estimation.mstep`; the whole
iteration is one `jit`-compiled function. Blocks (emission, duration, transition) with a closed
form are updated exactly; blocks without one (a family whose `m_step` is `None`, covariate links,
irregular steps, tied or
ordered parameters) by ECM (`estimation.ecm`: a few BFGS steps on the block's expected complete-data
log-likelihood, accepted only if it improves). `FitResult.diagnostics["em"]` records which was used.

Tempering (`estimation.em.tempering`): the E-step runs on `beta * ell` with `beta_k = beta0^(1 - k / steps)`
for the first `steps` iterations, then `beta = 1`; convergence and the monotonicity check apply from the
first `beta = 1` iteration on (the `beta < 1` entries stay in `loglik_trace` and need not be monotone).
Otherwise stop when the log-likelihood gain is `<= tol_f * max(1, |loglik|)` or after `max_iter`
iterations. Monotonicity is asserted when `HSMMKIT_DEBUG=1` (or `debug=True`): a decrease beyond
`1e-10` raises
`NumericalError`. Multi-start follows MAP (`cfg.restarts` starts cycled over `cfg.init`, same keys, best final
log-likelihood wins); `estimation.em.warm_start_from: map` runs EM from the MAP optimum instead. `use_priors`
adds the Dirichlet pseudo-counts of the jump rows (closed form) or `log p + log|J|` (ECM blocks).
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

from hsmmkit.config.schema import EmCfg
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
    relabel_states_perm,
)
from hsmmkit.estimation.map import fit as map_fit
from hsmmkit.estimation.mstep import block_plan, em_update, needs_per_t
from hsmmkit.estimation.objective import canonical_data
from hsmmkit.estimation.results import FitResult
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.statespace.builders import build_expanded
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

Plan = tuple[tuple[str, str], ...]


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
    """Raise `ConfigError` for structures EM / ECM do not handle (the explicit engine: GEM, E13-T03)."""
    del data
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
            "initial_state: stationary depends on the jump and duration parameters and has no EM update "
            "(use uniform or free)",
            path="model.initial_state",
        )
    if spec.duration_dependent_jumps or spec.left_censored:
        raise ConfigError("EM does not support duration-dependent jumps or left-censoring", path="model")
    groups: dict[str, set[str]] = {}
    for s in spec.param_specs:
        if s.tie_group is not None:
            groups.setdefault(s.tie_group, set()).add(s.name.split(".")[0])
    crossing = sorted(g for g, parts in groups.items() if len(parts) > 1)
    if crossing:
        raise ConfigError(
            f"tie groups {crossing} link parameters of different blocks (emission / duration / transition): "
            "ECM updates one block at a time",
            path="model",
        )


def tempering_schedule(cfg: EmCfg | None, max_iter: int) -> list[float]:
    """`beta_k = beta0^(1 - k / steps)` for `k = 0 .. steps - 1` (empty when tempering is off)."""
    if cfg is None or not cfg.tempering.enabled:
        return []
    t = cfg.tempering
    return [float(t.beta0 ** (1.0 - k / t.steps)) for k in range(min(t.steps, max_iter))]


@partial(jax.jit, static_argnames=("spec", "use_priors", "plan", "inner_iter", "tempered"))
def _em_step(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    use_priors: bool,
    plan: Plan | None = None,
    inner_iter: int = 5,
    tempered: bool = False,
    beta: Array | None = None,
) -> tuple[Params, Array]:
    """`(theta', loglik(theta))`: one (tempered) E-step and the closed-form / ECM M-step."""
    keep = plan is not None and needs_per_t(plan)
    b = jnp.asarray(1.0) if beta is None else beta
    stats = _estep_batch(theta, spec, space, pb, b, tempered, keep)
    new = em_update(theta, spec, space, pb, stats, use_priors=use_priors, plan=plan, inner_iter=inner_iter)
    return new, jnp.sum(stats.loglik)


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
    plan: Plan | None = None,
) -> tuple[Params, list[float], bool, int]:
    """EM from `theta0` on the padded batch `pb`: `(theta, loglik trace, converged, n_iter)`.

    The trace holds the log-likelihood at the start of every iteration, then the final one. During the
    tempering schedule neither convergence nor monotonicity is checked.
    """
    check = _debug_on(debug)
    em_cfg = cfg.em
    schedule = tempering_schedule(em_cfg, cfg.max_iter)
    inner = em_cfg.ecm.inner_iter if em_cfg is not None else 5
    plan = plan if plan is not None else block_plan(spec, pb)
    theta = dict(theta0)
    trace: list[float] = []
    converged = False
    n_iter = 0
    for n_iter in range(1, cfg.max_iter + 1):
        k = n_iter - 1
        warm = k < len(schedule)
        new, ll_array = _em_step(
            theta,
            spec,
            space,
            pb,
            cfg.use_priors,
            plan,
            inner,
            warm,
            jnp.asarray(schedule[k]) if warm else None,
        )
        ll = float(ll_array)
        if not np.isfinite(ll):
            raise NumericalError("non-finite log-likelihood during EM", path="loglik")
        if warm:
            trace.append(ll)
            theta = dict(new)
            continue
        if (
            check
            and len(trace) > len(schedule)
            and ll < trace[-1] - (MONOTONE_SLACK + 1e-13 * abs(trace[-1]))
        ):
            raise NumericalError(
                f"EM log-likelihood decreased: {trace[-1]!r} -> {ll!r} at iteration {n_iter}", path="loglik"
            )
        trace.append(ll)
        if len(trace) > len(schedule) + 1 and abs(ll - trace[-2]) <= cfg.tol_f * max(abs(ll), 1.0):
            converged = True
            break
        theta = dict(new)
    return theta, trace, converged, n_iter


def _final_loglik(theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch) -> float:
    stats = _estep_batch(theta, spec, space, pb, jnp.asarray(1.0), False)
    return float(jnp.sum(stats.loglik))


class EMEstimator:
    """Registered estimator `"em"`: EM / ECM with MAP's multi-start and `FitResult` contract."""

    name = "em"
    diagnostics_key = "em"
    score_from_trace = False  # VB: the final value of the trace (the ELBO) ranks the runs
    trace_key = "loglik_trace"

    def _section(self, cfg: EstimationSection) -> EmCfg:
        """The option block of this estimator (`estimation.em`)."""
        return cfg.em if cfg.em is not None else EmCfg()

    def _check(self, spec: ModelSpec, data: PreparedDataset | SequenceBatch) -> None:
        check_em_supported(spec, data)

    def _space(self, spec: ModelSpec) -> StateSpace:
        return build_expanded(spec)

    def _engine(self) -> Any:  # noqa: ANN401
        return ExpandedHMMEngine()

    def _plan(self, spec: ModelSpec, pb: PaddedBatch) -> Plan:
        return block_plan(spec, pb)

    def _run(
        self,
        theta0: Params,
        spec: ModelSpec,
        space: StateSpace,
        pb: PaddedBatch,
        cfg: EstimationSection,
        plan: Plan,
    ) -> tuple[Params, list[float], bool, int]:
        return run_em(theta0, spec, space, pb, cfg, plan=plan)

    def _final(self, theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch) -> float:
        return _final_loglik(theta, spec, space, pb)

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
        with its `loglik_trace`; `diagnostics["em"]` names the method of every block and the tempering used.

        Raises:
            ConfigError: an unsupported structure (see `check_em_supported`).
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
        self._check(spec, batch if batch is not None else data)
        manager = ParamManager(spec.param_specs)
        space = self._space(spec)
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
        em_cfg = self._section(cfg)
        plan = self._plan(spec, pb)
        starts: list[tuple[str, Params]] = []
        if em_cfg.warm_start_from == "map":
            optimum = map_fit(spec, data if batch is None else batch, cfg, seed)
            starts.append(("map", dict(optimum.params)))
        else:
            for i, method in enumerate(_restart_plan(list(cfg.init), cfg.restarts)):
                starts.append(
                    (method, dict(_init_theta(method, init_data, spec, key_for(seed, i), segments)))
                )
        if warm_start is not None:
            if tuple(warm_start.shape) != (manager.n_free,):
                raise ConfigError(
                    f"warm_start has shape {tuple(warm_start.shape)}, expected ({manager.n_free},)",
                    path="estimation.warm_start",
                )
            starts.append(("warm_start", dict(manager.unflatten(warm_start))))
        rows: list[dict[str, object]] = []
        runs: list[tuple[Params, float]] = []
        lls: list[float] = []
        extras: list[Any] = []
        for i, (method, theta0) in enumerate(starts):
            started = time.perf_counter()
            out = self._run(theta0, spec, space, pb, cfg, plan)
            theta, trace, converged, n_iter = out[:4]
            extras.append(out[4] if len(out) > 4 else None)
            ll = self._final(theta, spec, space, pb)
            score = trace[-1] if self.score_from_trace else ll
            runs.append((theta, score))
            lls.append(ll)
            rows.append(
                {
                    "restart": i,
                    "init_method": method,
                    "objective": -score,
                    "loglik": ll,
                    "n_iter": n_iter,
                    "converged": converged,
                    "wall_s": time.perf_counter() - started,
                    self.trace_key: trace if self.score_from_trace else [*trace, ll],
                }
            )
        finite = [i for i, (_, ll) in enumerate(runs) if np.isfinite(ll)]
        if not finite:
            raise NumericalError(
                f"every one of {len(runs)} restarts gave a non-finite log-likelihood", path="restarts"
            )
        best = max(finite, key=lambda i: runs[i][1])
        theta, score = runs[best]
        ll = lls[best]
        theta, perm = relabel_states_perm(theta, spec, first)
        relabelled = perm is not None
        u = manager.flatten(theta)
        best_extras = _permute_extras(dict(extras[best] or {}), perm, spec)
        std_errors = best_extras.pop("std_errors", None) or None
        pruned_states = tuple(best_extras.get("pruned_states", ()))
        diagnostics: dict[str, Any] = {
            self.diagnostics_key: {
                **dict(plan),
                "tempering": em_cfg.tempering.model_dump(),
                "ecm_inner_iter": em_cfg.ecm.inner_iter,
                "warm_start_from": em_cfg.warm_start_from,
                **best_extras,
                **({"relabelled": relabelled} if extras[best] else {}),
            }
        }
        if batch is not None:
            table = _per_sequence_table(self._engine(), theta, spec, space, batch)
            ll = float(sum(r["loglik"] for r in table))
            diagnostics["per_sequence"] = table
            diagnostics["initial_groups"] = [g for g in batch.group_names] if spec.n_groups > 1 else None
        elif relabelled:
            ll = float(self._engine().loglik(theta, spec, space, canonical_data(data)))  # type: ignore[arg-type]
        return FitResult(
            estimator=self.name,  # type: ignore[arg-type]
            params=theta,
            uparams=u,
            objective=-score if self.score_from_trace else -ll,
            loglik=ll,
            restarts=pd.DataFrame(rows),
            converged=bool(rows[best]["converged"]),
            n_iter=int(rows[best]["n_iter"]),  # type: ignore[call-overload]
            std_errors=std_errors,
            diagnostics=diagnostics,
            spec=spec,
            prep_stats=first.prep_stats,
            config_resolved={},
            pruned_states=pruned_states,
        )


def _permute_extras(extras: dict[str, Any], perm: np.ndarray | None, spec: ModelSpec) -> dict[str, Any]:
    """Renumber the per-regime entries of an estimator's extras like `params` (`perm[new] = old`)."""
    if perm is None or not extras:
        return extras
    n = spec.n_states
    out = dict(extras)
    if "occupancy" in out:
        out["occupancy"] = np.asarray(out["occupancy"])[perm]
    if "pruned_states" in out:
        old = set(out["pruned_states"])
        out["pruned_states"] = tuple(int(i) for i, o in enumerate(perm) if int(o) in old)
    if out.get("std_errors"):
        out["std_errors"] = {
            k: (np.asarray(v)[perm] if np.ndim(v) >= 1 and np.shape(v)[0] == n else v)
            for k, v in out["std_errors"].items()
        }
    q = out.get("q_params")
    if isinstance(q, dict):
        new_q: dict[str, Any] = {}
        for block, value in q.items():
            if block == "jump":
                if spec.order == 1:  # rows are the regimes, columns the targets
                    new_q[block] = np.asarray(value)[perm][:, perm]
                continue  # memory-tuple rows of higher orders are not renumbered: dropped
            new_q[block] = {k: np.asarray(v)[perm] for k, v in value.items()}
        out["q_params"] = new_q
    return out


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
        ConfigError: unsupported structure (see `check_em_supported`).
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
