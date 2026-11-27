"""Closed-form M-steps of EM on the expanded engine (E13-T02a).

Concept refs: Part II §14.6 (Baum-Welch with pooling). Each function maps the expected counts of one E-step
(`ExpandedEStats`) to new constrained parameters:

* **emission**: `family.sufficient_stats(y, dt, gamma_regime, z)` over the pooled (padded) batch, then
  `family.m_step(stats)`; a family without a closed form (`None`) is refused with a `ConfigError`;
* **phase advance** (constant step, no duration covariates): `a_k = advance_k / occupancy_k` per rate row and
  `rate = -log(1 - a_k) / nominal_dt` (`family.from_advance_prob`);
* **jump matrix** (no transition covariates): `pi_rj ~ jump[r, j] (+ alpha - 1)` over the allowed targets,
  in the `transition.pi_rows` or the flat topology parameterization (`transition.a`);
* **initial distribution** (`initial_state: free`): `rho_g ~ gamma1[g]` over the first-phase states.

`apply_modes` keeps `fixed` entries and clips `bounded` ones. Everything is `jit`-compatible except the
`ConfigError`s, which are raised while tracing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from hsmmkit.model.transition import _flat_structure, _target_order
from hsmmkit.utils.errors import ConfigError

if TYPE_CHECKING:
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.engine.estep import ExpandedEStats
    from hsmmkit.model.params import Params
    from hsmmkit.model.params.spec import ParamSpec
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

__all__ = [
    "NoClosedFormError",
    "apply_modes",
    "block_plan",
    "em_update",
    "emission_mstep",
    "initial_mstep",
    "needs_per_t",
    "phase_mstep",
    "transition_mstep",
]


class NoClosedFormError(ConfigError):
    """A component has no closed-form M-step (the caller falls back to ECM, or refuses)."""


_TINY = 1e-300
_A_MIN, _A_MAX = 1e-12, 1.0 - 1e-12


def _shapes(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    return {s.name: tuple(s.shape) for s in spec.param_specs}


def emission_mstep(spec: ModelSpec, theta: Params, pb: PaddedBatch, stats: ExpandedEStats) -> Params:
    """New emission parameters from the pooled weighted sufficient statistics (padding has weight 0)."""
    m = pb.observations.shape[-1]
    observed = pb.valid & jnp.all(pb.mask, axis=-1)
    gamma = (stats.gamma_regime * observed[..., None]).reshape(-1, stats.gamma_regime.shape[-1])
    y = pb.observations.reshape(-1, m)
    dt = pb.dt.reshape(-1)
    k_e = pb.cov_emission.shape[-1]
    z = pb.cov_emission.reshape(-1, k_e) if k_e > 0 else None
    suff = spec.emission.sufficient_stats(y, dt, gamma, z)
    update = None if suff is None else spec.emission.m_step(suff)
    if update is None:
        raise NoClosedFormError(
            f"component {type(spec.emission).__name__} has no closed-form M-step; "
            "available from E13-T02b (ECM)",
            path="model.emission",
        )
    shapes = _shapes(spec)
    out = dict(theta)
    for name, value in update.items():
        out[name] = jnp.reshape(value, shapes.get(name, jnp.shape(theta[name]))) if name in theta else value
    return out


def phase_mstep(spec: ModelSpec, theta: Params, stats: ExpandedEStats, nominal_dt: float) -> Params:
    """`a_k = advance_k / occupancy_k`, then `from_advance_prob` (rows with no mass keep their rate)."""
    family = spec.duration
    inverse = getattr(family, "from_advance_prob", None)
    has_cov = bool(getattr(family, "cov_idx", None))
    if inverse is None or has_cov:
        raise NoClosedFormError(
            f"component {type(family).__name__} has no closed-form M-step"
            + (" with duration covariates" if has_cov else "")
            + "; available from E13-T02b (ECM)",
            path="model.duration",
        )
    occ = stats.occupancy
    a_old = spec.duration.advance_prob_t(theta, nominal_dt, None)
    a = jnp.where(occ > _TINY, stats.advance / jnp.where(occ > _TINY, occ, 1.0), a_old)
    update = inverse(jnp.clip(a, _A_MIN, _A_MAX), nominal_dt)
    shapes = _shapes(spec)
    out = dict(theta)
    for name, value in update.items():
        out[name] = jnp.reshape(value, shapes.get(name, jnp.shape(theta[name])))
    return out


def transition_mstep(
    spec: ModelSpec, space: StateSpace, theta: Params, stats: ExpandedEStats, use_priors: bool
) -> Params:
    """New jump probabilities from `jump[r, j]` (+ the Dirichlet pseudo-counts `alpha - 1` under MAP-EM)."""
    transition = spec.transition
    if getattr(transition, "cov_idx", ()) or "transition.b" in theta:
        raise NoClosedFormError(
            "transition covariates have no closed-form M-step; available from E13-T02b (ECM)",
            path="model.transition.covariates",
        )
    layout = space.layout
    n = spec.n_states
    alpha = getattr(transition, "dirichlet_alpha", None)
    if alpha is None:  # the default `pi_rows` carries its Dirichlet prior on the ParamSpec
        for s in spec.param_specs:
            if (
                s.name == "transition.pi_rows"
                and s.mode == "prior"
                and getattr(s.prior, "alpha", None) is not None
            ):
                alpha = float(s.prior.alpha)  # type: ignore[union-attr]
    extra = (alpha - 1.0) if (use_priors and alpha is not None) else 0.0
    counts = stats.jump
    out = dict(theta)
    if "transition.pi_rows" in theta:
        target_idx = jnp.asarray(_target_order(layout.current_of, n))
        picked = jnp.maximum(jnp.take_along_axis(counts, target_idx, axis=1) + extra, 0.0)  # (R, N-1)
        total = jnp.sum(picked, axis=1, keepdims=True)
        probs = jnp.where(
            total > _TINY, picked / jnp.where(total > _TINY, total, 1.0), theta["transition.pi_rows"]
        )
        out["transition.pi_rows"] = probs
    elif "transition.a" in theta:
        free_row, free_col = _flat_structure(layout)
        ref = np.array([np.flatnonzero(layout.target_mask[r])[0] for r in range(layout.target_mask.shape[0])])
        c = jnp.maximum(counts + jnp.where(jnp.asarray(layout.target_mask), extra, 0.0), 0.0)
        num = c[free_row, free_col]
        den = c[free_row, ref[free_row]]
        ok = (num > _TINY) & (den > _TINY)
        safe = jnp.where(ok, num / jnp.where(ok, den, 1.0), 1.0)
        out["transition.a"] = jnp.where(
            ok, jnp.log(safe), jnp.where(den > _TINY, -30.0, theta["transition.a"])
        )
    return out


def initial_mstep(spec: ModelSpec, space: StateSpace, theta: Params, stats: ExpandedEStats) -> Params:
    """`initial_state: free`: `rho_g ~ gamma1[g]` over the first-phase states (others: unchanged)."""
    if spec.initial != "free":
        return theta
    first = stats.gamma1[:, space.phase1_idx]  # (G, S1)
    total = jnp.sum(first, axis=1, keepdims=True)
    probs = jnp.where(total > _TINY, first / jnp.where(total > _TINY, total, 1.0), 1.0 / first.shape[1])
    target = theta["initial.logits"]
    out = dict(theta)
    out["initial.logits"] = jnp.reshape(probs, target.shape)
    return out


def _entry_mask(spec: ParamSpec, mode: str) -> np.ndarray | None:
    """`(leading,)` bool of the entries in `mode`, or `None` when no entry is."""
    leading = spec.shape[0] if spec.shape else 1
    modes = (spec.mode,) * leading if isinstance(spec.mode, str) else tuple(spec.mode)
    mask = np.array([m == mode for m in modes])
    return mask if mask.any() else None


def apply_modes(new: Params, old: Params, specs: tuple[ParamSpec, ...]) -> Params:
    """Keep `fixed` entries of `old`; clip `bounded` entries of `new` to their bounds."""
    out = dict(new)
    for spec in specs:
        if spec.name not in out:
            continue
        value = out[spec.name]
        fixed = _entry_mask(spec, "fixed")
        if fixed is not None:
            m = (
                jnp.asarray(fixed).reshape((-1,) + (1,) * (value.ndim - 1))
                if value.ndim
                else jnp.asarray(True)
            )
            value = jnp.where(m, old[spec.name], value)
        bounded = _entry_mask(spec, "bounded")
        if bounded is not None and spec.bounds is not None:
            m = (
                jnp.asarray(bounded).reshape((-1,) + (1,) * (value.ndim - 1))
                if value.ndim
                else jnp.asarray(True)
            )
            value = jnp.where(m, jnp.clip(value, spec.bounds[0], spec.bounds[1]), value)
        out[spec.name] = value
    return out


def _block_specs(spec: ModelSpec, prefix: str) -> tuple[ParamSpec, ...]:
    return tuple(p for p in spec.param_specs if p.name.startswith(prefix))


def _plain(specs: tuple[ParamSpec, ...]) -> bool:
    """No tie group and no state-axis ordering: the closed form can update every entry independently."""
    return all(p.tie_group is None and p.constraint == "none" and p.mode != "tied" for p in specs)


def block_plan(spec: ModelSpec, pb: PaddedBatch) -> tuple[tuple[str, str], ...]:
    """`(component, "closed-form" | "ecm" | "fixed")` per block (emission, duration, transition, initial).

    A block is closed-form when its family offers the update (`m_step` answers, `from_advance_prob`
    exists), no covariate link or irregular step is involved and its parameters are untied and
    unordered; otherwise ECM.
    """
    import numpy as np

    m = int(pb.observations.shape[-1])
    k_e = int(pb.cov_emission.shape[-1])
    probe_gamma = jnp.full((4, spec.n_states), 1.0 / spec.n_states)
    probe_z = jnp.zeros((4, k_e)) if k_e > 0 else None
    suff = spec.emission.sufficient_stats(jnp.ones((4, m)), jnp.ones(4), probe_gamma, probe_z)
    emission = (
        "closed-form"
        if suff is not None
        and spec.emission.m_step(suff) is not None
        and _plain(_block_specs(spec, "emission."))
        else "ecm"
    )
    dt = np.asarray(pb.dt)[np.asarray(pb.valid)]
    regular = bool(dt.size == 0 or np.allclose(dt, dt[0], rtol=1e-9, atol=0.0))
    duration = (
        "closed-form"
        if hasattr(spec.duration, "from_advance_prob")
        and not getattr(spec.duration, "cov_idx", ())
        and regular
        and _plain(_block_specs(spec, "duration."))
        else "ecm"
    )
    transition = (
        "closed-form"
        if not getattr(spec.transition, "cov_idx", ())
        and not any(p.name == "transition.b" for p in spec.param_specs)
        and _plain(_block_specs(spec, "transition."))
        else "ecm"
    )
    initial = "closed-form" if spec.initial == "free" else "fixed"
    return (("emission", emission), ("duration", duration), ("transition", transition), ("initial", initial))


def needs_per_t(plan: tuple[tuple[str, str], ...]) -> bool:
    """Whether an ECM block of `plan` needs the per-step counts of the E-step."""
    return any(kind == "ecm" and name in ("duration", "transition") for name, kind in plan)


def em_update(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
    stats: ExpandedEStats,
    *,
    use_priors: bool,
    plan: tuple[tuple[str, str], ...] | None = None,
    inner_iter: int = 5,
) -> Params:
    """One full M-step: emission, duration, transition, initial (closed form or ECM per `plan`)."""
    from functools import partial

    from hsmmkit.estimation import ecm

    kinds = (
        dict(plan)
        if plan is not None
        else dict.fromkeys(("emission", "duration", "transition"), "closed-form")
    )
    new = dict(theta)
    if kinds["emission"] == "ecm":
        q = partial(ecm.q_emission, spec=spec, pb=pb, stats=stats)
        new = ecm.ecm_block(
            new, spec, "emission.", lambda th: q(th), use_priors=use_priors, inner_iter=inner_iter
        )
    else:
        new = emission_mstep(spec, new, pb, stats)
    if kinds["duration"] == "ecm":
        qd = partial(ecm.q_duration, spec=spec, pb=pb, stats=stats)
        new = ecm.ecm_block(
            new, spec, "duration.", lambda th: qd(th), use_priors=use_priors, inner_iter=inner_iter
        )
    else:
        new = phase_mstep(spec, new, stats, pb.nominal_dt)
    if kinds["transition"] == "ecm":
        qt = partial(ecm.q_transition, spec=spec, space=space, pb=pb, stats=stats)
        new = ecm.ecm_block(
            new, spec, "transition.", lambda th: qt(th), use_priors=use_priors, inner_iter=inner_iter
        )
    else:
        new = transition_mstep(spec, space, new, stats, use_priors)
    new = initial_mstep(spec, space, new, stats)
    return apply_modes(new, theta, spec.param_specs)
