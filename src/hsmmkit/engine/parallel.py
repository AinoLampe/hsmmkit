"""Parallel-scan smoothing for the expanded engine (E4-T07, `estimation.parallel_scan`).

Forward-backward as two `jax.lax.associative_scan`s over `(S, S)` log-space matrix products
`M_t = log A_t + ell_t[None, :]` (`t = 1..T-1`): prefix products give the forward messages
`alpha_t = alpha_0 (x) M_1 (x) ... (x) M_t`, suffix products the backward messages
`beta_t = M_{t+1} (x) ... (x) M_{T-1} (x) 1`. Depth is `O(log T)` instead of `O(T)`, at the cost of
materializing `(T, S, S)` arrays (only enabled when `T * S^2 <= 5e7`, off by default; useful on GPUs).

Elements are kept as `(E, c)` = `exp(M - c)` with a scalar log-scale `c` per matrix, so products are ordinary
matrix multiplications; entries more than ~700 nats below the maximum of their (prefix/suffix)
product underflow to 0: loglik, smoothed and filtered probabilities are unaffected, while
`log_alpha`/`log_beta` may be `-inf` there.
Not meant to be differentiated (estimation uses the sequential loglik).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from regimekit.engine.expanded_hmm import _emission_inputs
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.data.contracts import PreparedDataset
    from regimekit.model.params import Params
    from regimekit.model.spec import ModelSpec
    from regimekit.statespace.index import StateSpace

MAX_CELLS = 5e7
_NEG_INF = -jnp.inf


def _scaled(log_m: Array) -> tuple[Array, Array]:
    """`(E, c)` with `exp(log_m) = exp(c) * E`, `max(E) == 1` (`c = -inf`, `E = 0` if all `-inf`)."""
    c = jnp.max(log_m, axis=(-2, -1))
    c_safe = jnp.where(jnp.isfinite(c), c, 0.0)
    return jnp.exp(log_m - c_safe[..., None, None]), c


def _combine(a: tuple[Array, Array], b: tuple[Array, Array]) -> tuple[Array, Array]:
    """Product of two scaled matrices (batched over leading axes), renormalized to `max == 1`."""
    (ea, ca), (eb, cb) = a, b
    prod = ea @ eb
    m = jnp.max(prod, axis=(-2, -1))
    m_safe = jnp.where(m > 0, m, 1.0)
    log_scale = jnp.where(m > 0, ca + cb + jnp.log(m_safe), _NEG_INF)
    return prod / m_safe[..., None, None], log_scale


def _log_vec_mat(log_v: Array, e: Array, c: Array) -> Array:
    """`log(exp(log_v) @ (exp(c) E))` for `log_v (S,)` and batched `E (T, S, S)`, `c (T,)`; `-inf` where 0."""
    shift = jnp.max(log_v)
    shift_safe = jnp.where(jnp.isfinite(shift), shift, 0.0)
    w = jnp.exp(log_v - shift_safe) @ e  # (T, S)
    positive = w > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, w, 1.0)) + shift_safe + c[:, None], _NEG_INF)


@partial(jax.jit, static_argnames=("spec",))
def parallel_forward_backward(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array, Array, Array]:
    """`(log_alpha, log_beta, smoothed, loglik)` as `ExpandedHMMEngine.forward_backward`, in parallel."""
    log_rho, ell_0, _, xs = _emission_inputs(theta, spec, space, data)
    ell_rest, z_pi, z_d, dt = xs
    s = log_rho.shape[0]
    alpha_0 = log_rho + ell_0

    def log_a(z_pi_t: Array, z_d_t: Array, dt_t: Array) -> Array:
        return space.transition_fn(theta, z_pi_t, z_d_t, dt_t)[1]

    log_m = jax.vmap(log_a)(z_pi, z_d, dt) + ell_rest[:, None, :]  # (T-1, S, S): step t moves t-1 -> t
    elems = _scaled(log_m)

    prefix_e, prefix_c = jax.lax.associative_scan(_combine, elems, axis=0)
    log_alpha_rest = _log_vec_mat(alpha_0, prefix_e, prefix_c)  # (T-1, S)
    log_alpha = jnp.concatenate([alpha_0[None, :], log_alpha_rest], axis=0)
    loglik = jax.nn.logsumexp(log_alpha[-1])

    # suffix[i] = M_{i+1} (x) ... (x) M_{T-1}; beta_t = suffix[t] (x) 1 for t = 0..T-2, beta_{T-1} = 0
    suffix_e, suffix_c = jax.lax.associative_scan(lambda x, y: _combine(y, x), elems, axis=0, reverse=True)
    row_sum = jnp.sum(suffix_e, axis=-1)  # (T-1, S)
    positive = row_sum > 0
    log_beta_rest = jnp.where(
        positive, jnp.log(jnp.where(positive, row_sum, 1.0)) + suffix_c[:, None], _NEG_INF
    )
    log_beta = jnp.concatenate([log_beta_rest, jnp.zeros((1, s))], axis=0)

    log_gamma = log_alpha + log_beta
    finite = jnp.isfinite(log_gamma)
    norm = jax.nn.logsumexp(jnp.where(finite, log_gamma, _NEG_INF), axis=1)[:, None]
    smoothed = jnp.where(finite, jnp.exp(jnp.where(finite, log_gamma - norm, 0.0)), 0.0)
    return log_alpha, log_beta, smoothed, loglik
