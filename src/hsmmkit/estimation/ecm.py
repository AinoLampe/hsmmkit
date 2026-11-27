"""ECM blocks: conditional maximization of the expected complete-data log-likelihood (E13-T02b).

Concept refs: Part II §14.6 (ECM). Components without a closed-form M-step (a family whose `m_step` is `None`,
covariate links, irregular steps, tied or ordered parameters) are updated one block at a time with the other
blocks fixed. A block is a name prefix (`emission.`, `duration.`, `transition.`); its unconstrained
coordinates `u_b` (`ParamManager` over the block's specs, so ties and orderings are handled by the layout)
are moved by a few BFGS iterations on `-Q_b(theta(u_b))` (plus `log p + log|J|` when `use_priors`, the
MAP convention of
`estimation.objective`), and the step is accepted only if the objective decreased: ECM never decreases the
log-likelihood.

`Q_e = sum_t sum_i gamma_ti log f_i(y_t | theta_e, z_t)`; `Q_d = sum_t sum_k [stay_tk log(1 - a_k(t)) +
advance_tk log a_k(t)]`; `Q_pi = sum_t sum_r sum_j jump_trj log pi_rj(t)` (per-step counts, `keep_per_t`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.scipy.optimize as jopt

from hsmmkit.engine.expanded_hmm import regime_log_emis
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.engine.estep import ExpandedEStats
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

__all__ = ["ecm_block", "q_duration", "q_emission", "q_transition"]

_A_MIN, _A_MAX = 1e-12, 1.0 - 1e-12


def ecm_block(
    theta: Params,
    spec: ModelSpec,
    prefix: str,
    q_fn: Callable[[Params], Array],
    *,
    use_priors: bool,
    inner_iter: int,
) -> Params:
    """Maximize `q_fn` over the block `prefix`, others fixed (BFGS; accepted only if it improves)."""
    specs = tuple(s for s in spec.param_specs if s.name.startswith(prefix))
    if not specs:
        return theta
    sub = ParamManager(specs)
    if sub.n_free == 0:
        return theta
    u0 = sub.flatten(theta)

    def neg(u: Array) -> Array:
        th = {**theta, **sub.unflatten(u)}
        value = q_fn(th)
        if use_priors:
            value = value + sub.log_prior(th) + sub.log_det_jac(u)
        return -value

    start = neg(u0)
    result = jopt.minimize(neg, u0, method="BFGS", options={"maxiter": inner_iter})
    better = jnp.isfinite(result.fun) & (result.fun <= start)
    u = jnp.where(better, result.x, u0)
    return {**theta, **sub.unflatten(u)}


def q_emission(theta: Params, spec: ModelSpec, pb: PaddedBatch, stats: ExpandedEStats) -> Array:
    """`sum gamma_ti log f_i(y_t)` over the pooled batch (padding and unobserved series carry no weight)."""
    m = pb.observations.shape[-1]
    k_e = pb.cov_emission.shape[-1]
    z = pb.cov_emission.reshape(-1, k_e) if k_e > 0 else jnp.zeros((pb.dt.size, 0))
    mask = (pb.mask & pb.valid[..., None]).reshape(-1, m)
    ell = regime_log_emis(theta, spec, pb.observations.reshape(-1, m), mask, pb.dt.reshape(-1), z)
    gamma = stats.gamma_regime.reshape(-1, stats.gamma_regime.shape[-1])
    return jnp.sum(jnp.where(gamma > 0, gamma * ell, 0.0))


def q_duration(theta: Params, spec: ModelSpec, pb: PaddedBatch, stats: ExpandedEStats) -> Array:
    """`sum [stay log(1 - a) + advance log a]` with the per-step advance probability `a_k(t)`."""
    assert stats.stay_t is not None
    assert stats.advance_t is not None
    k_d = pb.cov_duration.shape[-1]
    z = pb.cov_duration.reshape(-1, k_d) if k_d > 0 else None
    a = jnp.clip(spec.duration.advance_prob(theta, pb.dt.reshape(-1), z), _A_MIN, _A_MAX)
    n_rate = stats.stay_t.shape[-1]
    stay, adv = stats.stay_t.reshape(-1, n_rate), stats.advance_t.reshape(-1, n_rate)
    return jnp.sum(stay * jnp.log1p(-a) + adv * jnp.log(a))


def q_transition(
    theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch, stats: ExpandedEStats
) -> Array:
    """`sum jump_trj log pi_rj(t)` with the (covariate-dependent) jump probabilities."""
    assert stats.jump_t is not None
    k_pi = pb.cov_transition.shape[-1]
    z = pb.cov_transition.reshape(-1, k_pi) if k_pi > 0 else None
    pi = spec.transition.jump_probs(theta, z, space.layout)  # (BT or 1, R, N)
    n_row, n_reg = stats.jump_t.shape[-2:]
    jump = stats.jump_t.reshape(-1, n_row, n_reg)
    log_pi = jnp.log(jnp.where(pi > 0, pi, 1.0))
    return jnp.sum(jnp.where(pi > 0, jump * log_pi, 0.0))
