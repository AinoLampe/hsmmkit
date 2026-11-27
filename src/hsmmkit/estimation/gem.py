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

Left-censoring needs `initial_state: free` (else the engine ties the first-segment law to the jump matrix).
Geometric durations ignore the truncation at `d_max` (a warning when more than 1e-6 of the mass is beyond it).
Other families (gamma, Weibull, lognormal, Poisson, negative binomial, every member of a per-state mix) are
updated by weighted maximum likelihood on the same objective with damped Newton steps (`estimation.newton`,
BFGS fallback), also with entry-time duration covariates (`eta_entry`).
Not supported (`ConfigError`): the expanded engine, transition covariates, left-censoring with duration
covariates, duration-dependent jumps, tempering, tied parameters across blocks.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
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
    _plain,
    apply_modes,
    block_plan,
    emission_mstep,
    transition_mstep,
)
from hsmmkit.estimation.newton import newton_minimize
from hsmmkit.model.duration.nonparametric import NonparametricDuration
from hsmmkit.model.duration.per_state import PerStateDuration
from hsmmkit.model.params.manager import ParamManager
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


def _members(spec: ModelSpec) -> list[tuple[Any, str, slice]]:
    """`(family, parameter prefix, rows of the (N, .) statistics)` of each duration member."""
    family = spec.duration
    if isinstance(family, PerStateDuration):
        return [(f, f"duration.s{i}.", slice(i, i + 1)) for i, f in enumerate(family.families)]
    return [(family, "duration.", slice(None))]


def _view(theta: Params, prefix: str) -> Params:
    """The member's `duration.<name>` view of the (possibly prefixed) parameters."""
    if prefix == "duration.":
        return theta
    return {f"duration.{k[len(prefix) :]}": v for k, v in theta.items() if k.startswith(prefix)}


def _member_specs(spec: ModelSpec, prefix: str) -> tuple[Any, ...]:
    """The member's `ParamSpec`s renamed to the `duration.<name>` view."""
    own = [s for s in spec.param_specs if s.name.startswith(prefix)]
    if prefix == "duration.":
        return tuple(own)
    return tuple(dataclasses.replace(s, name=f"duration.{s.name[len(prefix) :]}") for s in own)


def _q_member(
    family: Any,  # noqa: ANN401
    th: Params,
    stats: ExplicitEStats,
    rows: slice,
    mask: Array,
    pb: PaddedBatch,
    entry: bool,
) -> Array:
    """Expected complete-data objective of one duration member (complete + censored + residual segments)."""
    d_max = int(stats.eta.shape[-1])
    if not entry:
        lp = family.log_pmf(th, d_max, pb.nominal_dt, None)
        surv = jax.lax.cumlogsumexp(lp, axis=lp.ndim - 1, reverse=True)  # log S(d) = log P(D >= d)
        eta = jnp.where(mask[rows, None], stats.eta[rows], 0.0)
        w = jnp.where(mask[rows, None], stats.eta_censored[rows] + stats.eta_first[rows], 0.0)
        n_first = jnp.where(mask[rows], jnp.sum(stats.eta_first[rows], axis=-1), 0.0)
        log_mu = jax.scipy.special.logsumexp(surv, axis=-1)
        q = jnp.sum(jnp.where(eta > 0, eta * lp, 0.0)) + jnp.sum(jnp.where(w > 0, w * surv, 0.0))
        return q - jnp.sum(jnp.where(n_first > 0, n_first * log_mu, 0.0))
    assert stats.eta_entry is not None
    assert stats.eta_entry_censored is not None
    b, t = stats.eta_entry.shape[:2]
    z = pb.cov_duration.reshape(b * t, -1)
    lp = family.log_pmf(th, d_max, pb.nominal_dt, z).reshape(b, t, -1, d_max)[:, :, rows]
    surv = jax.lax.cumlogsumexp(lp, axis=lp.ndim - 1, reverse=True)
    m = mask[rows]
    eta = jnp.where(m[None, None, :, None], stats.eta_entry[:, :, rows], 0.0)
    q = jnp.sum(jnp.where(eta > 0, eta * lp, 0.0))
    elapsed = jnp.clip(pb.lengths[:, None] - jnp.arange(t)[None, :] - 1, 0, d_max - 1)  # (B, T): d - 1
    s_at = jnp.take_along_axis(surv, elapsed[:, :, None, None], axis=-1)[..., 0]  # (B, T, n)
    ec = jnp.where(m[None, None, :], stats.eta_entry_censored[:, :, rows], 0.0)
    return q + jnp.sum(jnp.where(ec > 0, ec * s_at, 0.0))


def _neg_member(
    family: Any,  # noqa: ANN401
    th: Params,
    manager: Any,  # noqa: ANN401
    rows: slice,
    stats: ExplicitEStats,
    keep_rows: Array,
    pb: PaddedBatch,
    entry: bool,
    use_priors: bool,
    u: Array,
) -> Array:
    """`-(Q + log p + log|J|)` of one duration member at the free coordinates `u`."""
    cand = {**th, **manager.unflatten(u)}
    value = _q_member(family, cand, stats, rows, keep_rows, pb, entry)
    if use_priors:
        value = value + manager.log_prior(cand) + manager.log_det_jac(u)
    return -value


def duration_mstep(
    spec: ModelSpec,
    theta: Params,
    stats: ExplicitEStats,
    use_priors: bool,
    pb: PaddedBatch,
    entry: bool = False,
) -> Params:
    """New duration parameters: nonparametric pmf (censored maximizer), geometric closed form, else Newton.

    Every member (all regimes share one family, or one family per regime) is updated on its own rows of the
    segment counts; the weighted-MLE objective of the Newton path is `_q_member` (+ prior and log-Jacobian
    under `use_priors`).
    """
    out = dict(theta)
    topology = getattr(spec.transition, "topology", None)
    absorbing = (
        np.isin(np.arange(spec.n_states), topology.absorbing)
        if topology is not None
        else np.zeros(spec.n_states, bool)
    )
    keep_rows = jnp.asarray(~absorbing)  # the engine replaces an absorbing pmf by a point mass at D_max
    dt = pb.nominal_dt
    for family, prefix, rows in _members(spec):
        th = _view(out, prefix)
        n_rows = len(range(*rows.indices(spec.n_states)))
        sub = stats if rows == slice(None) else None
        del sub, n_rows
        if isinstance(family, NonparametricDuration) and not entry:
            key = f"{prefix}logits"
            extra = _dirichlet_extra(spec, key, use_priors)
            c = jnp.maximum(stats.eta[rows] + extra, 0.0)
            w = stats.eta_censored[rows] + stats.eta_first[rows]
            n_first = jnp.sum(stats.eta_first[rows], axis=-1)
            total = jnp.sum(c, axis=-1) + jnp.sum(w, axis=-1)
            p0 = out[key]
            new = jax.vmap(censored_pmf)(p0, c, w, n_first)
            out[key] = jnp.where(((total > 0) & keep_rows[rows])[:, None], new, p0)
        elif type(family).__name__ == "GeometricDuration" and not entry:
            eta, cens, first = stats.eta[rows], stats.eta_censored[rows], stats.eta_first[rows]
            d = jnp.arange(1, eta.shape[-1] + 1)
            n_done = jnp.sum(eta + first, axis=-1)
            denom = n_done + jnp.sum((d - 1) * (eta + first + cens), axis=-1)
            a = jnp.clip(jnp.where(denom > 0, n_done / jnp.where(denom > 0, denom, 1.0), 0.5), 1e-9, 1 - 1e-9)
            new_rate = family.from_advance_prob(a, dt)["duration.rate"]
            key = f"{prefix}rate"
            out[key] = jnp.where((denom > 0) & keep_rows[rows], new_rate, out[key])
        else:
            specs = _member_specs(spec, prefix)
            manager = ParamManager(specs)
            if manager.n_free == 0:
                continue
            u0 = manager.flatten(th)

            neg = partial(_neg_member, family, th, manager, rows, stats, keep_rows, pb, entry, use_priors)
            u, _, _ = newton_minimize(neg, u0)
            for k, v in manager.unflatten(u).items():
                out[k if prefix == "duration." else f"{prefix}{k.removeprefix('duration.')}"] = v
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
    entry: bool = False,
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
    new = duration_mstep(spec, new, stats, use_priors, pb, entry)
    new = transition_mstep(spec, space, new, view, use_priors)  # type: ignore[arg-type]
    if spec.initial == "free":
        rho = stats.rho_post
        total = jnp.sum(rho, axis=-1, keepdims=True)
        probs = jnp.where(total > 0, rho / jnp.where(total > 0, total, 1.0), 1.0 / rho.shape[-1])
        new["initial.logits"] = jnp.reshape(probs, jnp.shape(theta["initial.logits"]))
    return apply_modes(new, theta, spec.param_specs)


@partial(jax.jit, static_argnames=("spec", "use_priors", "emission_kind", "inner_iter", "entry"))
def _gem_step(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    use_priors: bool,
    emission_kind: str,
    inner_iter: int,
    entry: bool,
) -> tuple[Params, Array]:
    stats = _explicit_estep_batch(theta, spec, space, pb, entry)
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


def _entry(spec: ModelSpec, pb: PaddedBatch) -> bool:
    """Whether the E-step keeps the per-start segment counts (entry-time duration covariates)."""
    return bool(pb.cov_duration.shape[-1] > 0 and getattr(spec.duration, "cov_idx", ()))


def check_gem_supported(spec: ModelSpec) -> None:
    """Raise `ConfigError` for the structures GEM does not handle in this ticket."""
    if spec.engine != "explicit":
        raise ConfigError("GEM needs engine: explicit (the expanded engine: method em)", path="model.engine")
    family = spec.duration
    if spec.left_censored and spec.initial != "free":
        raise ConfigError(
            "left-censoring under GEM needs initial_state: free (otherwise the engine ties the first-segment "
            "law to the jump matrix, which the M-steps do not model)",
            path="model.initial_state",
        )
    dur_cov = bool(getattr(family, "cov_idx", ()))
    if getattr(spec.transition, "cov_idx", ()):
        raise ConfigError("GEM does not support transition covariates yet", path="model.transition")
    if dur_cov and spec.left_censored:
        raise ConfigError(
            "entry-time duration covariates with left-censoring are not supported by GEM", path="model"
        )
    if dur_cov and isinstance(family, NonparametricDuration):
        raise ConfigError("nonparametric durations have no covariate link", path="model.duration")
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
    if not _plain(tuple(s for s in spec.param_specs if s.name.startswith("transition."))):
        raise ConfigError("GEM does not support tied or ordered transition parameters", path="model")


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
        name = type(spec.duration).__name__
        kind = {"NonparametricDuration": "censored-pmf", "GeometricDuration": "geometric"}.get(name, "newton")
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
        entry = _entry(spec, pb)
        theta = dict(theta0)
        trace: list[float] = []
        converged = False
        n_iter = 0
        for n_iter in range(1, cfg.max_iter + 1):
            new, ll_array = _gem_step(
                theta, spec, space, pb, cfg.use_priors, emission_kind, section.ecm.inner_iter, entry
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
        if "duration.rate" in theta:
            lost = float(jnp.max((1.0 - -jnp.expm1(-theta["duration.rate"] * pb.nominal_dt)) ** spec.d_max))
            if lost > 1e-6:
                warnings.warn(
                    f"geometric durations under GEM ignore the truncation at d_max={spec.d_max}: "
                    f"up to {lost:.2g} of the pmf lies beyond it (the closed form is then approximate)",
                    stacklevel=2,
                )
        return theta, trace, converged, n_iter

    def _final(self, theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch) -> float:
        stats = _explicit_estep_batch(theta, spec, space, pb, _entry(spec, pb))
        return float(jnp.sum(stats.loglik))


REGISTRY.register("estimator", GEMEstimator.name, GEMEstimator(), provenance=provenance_of(__name__))
