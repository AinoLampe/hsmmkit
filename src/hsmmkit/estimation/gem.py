"""`GEMEstimator`: generalized EM on the explicit-duration engine (E13-T03a; Guedon 2003).

Concept refs: Part II §14.6 (explicit engine), §14.4. One iteration is one `explicit_estep` (E13-T01b: segment
posteriors `eta`, final-segment survival counts `eta_censored`, residual first segments `eta_first`,
jump counts
`xi`) followed by:

* **emissions**: the family's closed form on the regime marginals, else an ECM block (`estimation.ecm`);
* **jumps**: `pi_ij ~ xi[i, j]` (+ Dirichlet pseudo-counts under `use_priors`), also under a topology;
* **durations**: nonparametric pmfs by the censored maximizer (below); geometric durations in closed form
  `a = n / (n + sum (d - 1) (eta + eta_first + eta_censored))`, `rate = -log(1 - a) / dt`; parametric
  families are refused until E13-T03b;
* **initial distribution**: `free` -> `rho ~ rho_post`.

**Censored nonparametric pmf.** With `S(d) = sum_{d' >= d} p(d')` the expected complete-data objective is
`Q(p) = sum_d c_d log p_d + sum_d w_d log S(d) - n_first log mu(p)`, where `c = eta + prior`, `w =
eta_censored +
eta_first` (the final segment covering `T` and the residual-life first segment both contribute a survival
term) and `mu = sum_d d p_d` (the residual-life normalizer, only with left-censoring). Jensen on `log S`
and the
tangent of the convex `-log mu` give the minorizer
`p_d = (c_d + p0_d sum_{d' <= d} w_{d'} / S0(d')) / (nu + lambda d)`, `lambda = n_first / mu0`, with
`nu` fixed by
`sum p = 1` (bisection); iterating to `||dp||_1 < 1e-12` (at most 200 times) never decreases `Q`. Without
censoring this is `p ~ eta`. A segment that is both first and last is counted as a plain survival term
(its exact likelihood is a tail sum of the residual pmf), a negligible approximation for `T >> D_max`.

Not supported (`ConfigError`): the expanded engine, parametric durations, per-state duration families,
duration or transition covariates, duration-dependent jumps, tempering, tied parameters across blocks.
"""

from __future__ import annotations

import os
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.config.schema import EmCfg
from hsmmkit.engine.estep_explicit import _estep_batch as _explicit_estep_batch
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.estimation import ecm
from hsmmkit.estimation.em import MONOTONE_SLACK, EMEstimator, Plan
from hsmmkit.estimation.mstep import (
    NoClosedFormError,
    _plain,
    apply_modes,
    block_plan,
    emission_mstep,
    transition_mstep,
)
from hsmmkit.model.duration.nonparametric import NonparametricDuration
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.statespace.builders import build_explicit
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EstimationSection
    from hsmmkit.data.batch import PaddedBatch, SequenceBatch
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.engine.estep_explicit import ExplicitEStats
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

__all__ = ["GEMEstimator", "censored_pmf", "check_gem_supported", "pmf_objective"]

_TINY = 1e-300
_INNER_MAX = 200
_INNER_TOL = 1e-12


def pmf_objective(p: Array, c: Array, w: Array, n_first: Array) -> Array:
    """`Q(p) = sum c log p + sum w log S + (-n_first log mu)` of one state (`p` positive pmf over `1..D`)."""
    d = jnp.arange(1, p.shape[-1] + 1)
    surv = jnp.cumsum(p[::-1])[::-1]
    mu = jnp.sum(d * p)
    return jnp.sum(c * jnp.log(p)) + jnp.sum(w * jnp.log(surv)) - n_first * jnp.log(mu)


def _mm_step(p: Array, c: Array, w: Array, n_first: Array) -> Array:
    """One minorize-maximize step of the censored pmf objective (never decreases `Q`)."""
    d = jnp.arange(1, p.shape[-1] + 1, dtype=p.dtype)
    surv = jnp.maximum(jnp.cumsum(p[::-1])[::-1], _TINY)
    num = c + p * jnp.cumsum(w / surv)
    lam = n_first / jnp.sum(d * p)
    total = jnp.sum(num)
    d_min = jnp.min(jnp.where(num > 0, d, jnp.inf))
    lo = jnp.where(jnp.isfinite(d_min), -lam * d_min, 0.0) + 1e-12 * jnp.maximum(total, 1.0)
    hi = jnp.maximum(total, lo + 1.0)

    def bisect(_: Array, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        a, b = bounds
        m = 0.5 * (a + b)
        f = jnp.sum(num / (m + lam * d)) - 1.0
        return jnp.where(f > 0, m, a), jnp.where(f > 0, b, m)

    lo, hi = jax.lax.fori_loop(0, 100, bisect, (lo, hi))
    nu = 0.5 * (lo + hi)
    new: Array = num / (nu + lam * d)
    return new / jnp.sum(new)


def censored_pmf(p0: Array, c: Array, w: Array, n_first: Array) -> Array:
    """Maximizer of `Q` for one state by iterating `_mm_step` from `p0` (a positive pmf)."""
    p0 = jnp.maximum(p0, 1e-12)
    p0 = p0 / jnp.sum(p0)

    def cond(state: tuple[Array, Array, Array]) -> Array:
        return (state[2] < _INNER_MAX) & (state[1] >= _INNER_TOL)

    def body(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        p, _, it = state
        new = jnp.maximum(_mm_step(p, c, w, n_first), 1e-300)
        new = new / jnp.sum(new)
        return new, jnp.sum(jnp.abs(new - p)), it + 1

    init = (p0, jnp.asarray(jnp.inf, dtype=p0.dtype), jnp.zeros((), jnp.int32))
    p, _, _ = jax.lax.while_loop(cond, body, init)
    out: Array = p
    return out


def _dirichlet_extra(spec: ModelSpec, name: str, use_priors: bool) -> float:
    for s in spec.param_specs:
        if s.name == name and s.mode == "prior" and getattr(s.prior, "alpha", None) is not None:
            return float(s.prior.alpha) - 1.0 if use_priors else 0.0  # type: ignore[union-attr]
    return 0.0


def duration_mstep(
    spec: ModelSpec, theta: Params, stats: ExplicitEStats, use_priors: bool, dt: float
) -> Params:
    """New duration parameters (nonparametric pmf or geometric rate) from the segment counts."""
    family = spec.duration
    out = dict(theta)
    if isinstance(family, NonparametricDuration):
        extra = _dirichlet_extra(spec, "duration.logits", use_priors)
        c = jnp.maximum(stats.eta + extra, 0.0)
        w = stats.eta_censored + stats.eta_first
        n_first = jnp.sum(stats.eta_first, axis=-1)
        total = jnp.sum(c, axis=-1) + jnp.sum(w, axis=-1)
        p0 = theta["duration.logits"]
        new = jax.vmap(censored_pmf)(p0, c, w, n_first)
        out["duration.logits"] = jnp.where((total > 0)[:, None], new, p0)
        return out
    d = jnp.arange(1, stats.eta.shape[-1] + 1)
    n_done = jnp.sum(stats.eta + stats.eta_first, axis=-1)
    waiting = jnp.sum((d - 1) * (stats.eta + stats.eta_first + stats.eta_censored), axis=-1)
    denom = n_done + waiting
    a = jnp.where(denom > 0, n_done / jnp.where(denom > 0, denom, 1.0), 0.5)
    a = jnp.clip(a, 1e-9, 1.0 - 1e-9)
    new_rate = family.from_advance_prob(a, dt)["duration.rate"]  # type: ignore[attr-defined]
    out["duration.rate"] = jnp.where(denom > 0, new_rate, theta["duration.rate"])
    return out


def gem_update(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    stats: ExplicitEStats,
    *,
    use_priors: bool,
    emission_kind: str,
    inner_iter: int = 5,
) -> Params:
    """One full generalized M-step (emission, duration, jumps, initial law, then parameter modes)."""
    view = SimpleNamespace(gamma_regime=stats.gamma_regime, jump=stats.xi)
    if emission_kind == "ecm":
        q = partial(ecm.q_emission, spec=spec, pb=pb, stats=view)  # type: ignore[arg-type]
        new = ecm.ecm_block(
            dict(theta), spec, "emission.", lambda th: q(th), use_priors=use_priors, inner_iter=inner_iter
        )
    else:
        new = emission_mstep(spec, dict(theta), pb, view)  # type: ignore[arg-type]
    new = duration_mstep(spec, new, stats, use_priors, pb.nominal_dt)
    new = transition_mstep(spec, space, new, view, use_priors)  # type: ignore[arg-type]
    if spec.initial == "free":
        rho = stats.rho_post
        total = jnp.sum(rho, axis=-1, keepdims=True)
        probs = jnp.where(total > 0, rho / jnp.where(total > 0, total, 1.0), 1.0 / rho.shape[-1])
        new["initial.logits"] = jnp.reshape(probs, jnp.shape(theta["initial.logits"]))
    return apply_modes(new, theta, spec.param_specs)


@partial(jax.jit, static_argnames=("spec", "use_priors", "emission_kind", "inner_iter"))
def _gem_step(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    use_priors: bool,
    emission_kind: str,
    inner_iter: int,
) -> tuple[Params, Array]:
    stats = _explicit_estep_batch(theta, spec, space, pb, False)
    new = gem_update(
        theta,
        spec,
        space,
        pb,
        stats,
        use_priors=use_priors,
        emission_kind=emission_kind,
        inner_iter=inner_iter,
    )
    return new, jnp.sum(stats.loglik)


def check_gem_supported(spec: ModelSpec) -> None:
    """Raise `ConfigError` for the structures GEM does not handle in this ticket."""
    if spec.engine != "explicit":
        raise ConfigError("GEM needs engine: explicit (the expanded engine: method em)", path="model.engine")
    family = spec.duration
    if not isinstance(family, NonparametricDuration) and not (
        hasattr(family, "from_advance_prob") and type(family).__name__ == "GeometricDuration"
    ):
        raise ConfigError(
            "parametric durations under GEM available from E13-T03b "
            f"(got {type(family).__name__}; supported now: nonparametric, geometric)",
            path="model.duration.family",
        )
    if getattr(family, "cov_idx", ()) or getattr(spec.transition, "cov_idx", ()):
        raise ConfigError("GEM does not support duration or transition covariates yet", path="model")
    if spec.duration_dependent_jumps:
        raise ConfigError("GEM does not support duration-dependent jumps", path="model.duration")
    if spec.initial == "stationary":
        raise ConfigError("initial_state: stationary has no GEM update (use uniform or free)", path="model")
    blocks: dict[str, set[str]] = {}
    for s in spec.param_specs:
        if s.tie_group is not None:
            blocks.setdefault(s.tie_group, set()).add(s.name.split(".")[0])
    crossing = sorted(g for g, parts in blocks.items() if len(parts) > 1)
    if crossing:
        raise ConfigError(f"tie groups {crossing} link different blocks", path="model")
    if not _plain(tuple(s for s in spec.param_specs if s.name.startswith(("duration.", "transition.")))):
        raise ConfigError("GEM does not support tied or ordered duration/transition parameters", path="model")


class GEMEstimator(EMEstimator):
    """Registered estimator `"gem"`: generalized EM on the explicit-duration engine."""

    name = "gem"
    diagnostics_key = "gem"

    def _section(self, cfg: EstimationSection) -> EmCfg:
        if cfg.gem is not None:
            return cfg.gem
        return cfg.em if cfg.em is not None else EmCfg()

    def _check(self, spec: ModelSpec, data: PreparedDataset | SequenceBatch) -> None:
        del data
        check_gem_supported(spec)

    def _space(self, spec: ModelSpec) -> StateSpace:
        return build_explicit(spec)

    def _engine(self) -> Any:  # noqa: ANN401
        return ExplicitDurationEngine()

    def _plan(self, spec: ModelSpec, pb: PaddedBatch) -> Plan:
        emission = dict(block_plan(spec, pb))["emission"]
        kind = "geometric" if not isinstance(spec.duration, NonparametricDuration) else "censored-pmf"
        return (
            ("emission", emission),
            ("duration", f"closed-form ({kind})"),
            ("transition", "closed-form"),
            ("initial", "closed-form" if spec.initial == "free" else "fixed"),
        )

    def _run(
        self,
        theta0: Params,
        spec: ModelSpec,
        space: StateSpace,
        pb: PaddedBatch,
        cfg: EstimationSection,
        plan: Plan,
    ) -> tuple[Params, list[float], bool, int]:
        section = self._section(cfg)
        if section.tempering.enabled:
            raise ConfigError("tempering is not available for GEM", path="estimation.gem.tempering")
        check = os.environ.get("HSMMKIT_DEBUG") == "1"
        emission_kind = "ecm" if dict(plan)["emission"] == "ecm" else "closed-form"
        theta = dict(theta0)
        trace: list[float] = []
        converged = False
        n_iter = 0
        for n_iter in range(1, cfg.max_iter + 1):
            new, ll_array = _gem_step(
                theta, spec, space, pb, cfg.use_priors, emission_kind, section.ecm.inner_iter
            )
            ll = float(ll_array)
            if not np.isfinite(ll):
                raise NumericalError("non-finite log-likelihood during GEM", path="loglik")
            if check and trace and ll < trace[-1] - (MONOTONE_SLACK + 1e-13 * abs(trace[-1])):
                raise NumericalError(
                    f"GEM log-likelihood decreased: {trace[-1]!r} -> {ll!r} at iteration {n_iter}",
                    path="loglik",
                )
            trace.append(ll)
            if len(trace) > 1 and abs(ll - trace[-2]) <= cfg.tol_f * max(abs(ll), 1.0):
                converged = True
                break
            theta = dict(new)
        return theta, trace, converged, n_iter

    def _final(self, theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch) -> float:
        stats = _explicit_estep_batch(theta, spec, space, pb, False)
        return float(jnp.sum(stats.loglik))


_ = NoClosedFormError  # re-exported for callers that catch it
REGISTRY.register("estimator", GEMEstimator.name, GEMEstimator(), provenance=provenance_of(__name__))
