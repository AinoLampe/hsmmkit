"""Explicit-duration E-step: segment posteriors `eta` and jump posteriors `xi` (E13-T01b).

Concept refs: Part II §14.6 (Guedon), §14.4; Part I §4.7.3. With the segmental recursion of
`engine.explicit_duration` (`g[s, j]` = log-prob of a regime-`j` segment *starting* at boundary `s`,
`H[tau, j]` = emissions up to `tau` plus the future given a segment of `j` ends at `tau`, `end[tau, j]` =
log-prob that a segment of `j` ends at `tau`, `cont[s, j]` = the future after a segment of `j` starts at `s`):

    eta[j, d]    = sum_s exp(g[s, j] + lp_s[j, d] + H[s + d, j] - loglik)      complete segments, s + d < T
    cens[j, d]   = sum_s exp(inner[s, j] + term[s, j] - loglik),  d = T - s          the segment covering `T`
    xi[i, j]     = sum_tau exp(end[tau, i] + log pi_tau[i, j] + cont[tau, j] - loglik)

Each is accumulated inside one scan over `s` (or `tau`) into `(N, D_max)` / `(N, N)` carries: no
`(T, N, D_max)` array exists unless `eta_entry` is requested (entry-time duration covariates). The
segment that covers `T` is right-censored (survival term), so a segment ending exactly at `T` counts as
censored with `d = T - s <= D_max`. Left-censoring (E12-T04): the closed first segment (`s = 0`,
residual-life pmf) is `eta_first`, not `eta`; a first segment that is also the last one is `eta_censored`.

Mass balance: `sum_{j, d} d (eta + eta_censored + eta_first)[j, d] = T_b` per sequence, and
`sum_j xi[i, j] = sum_d (eta + eta_first)[i, d]`.

Not supported (`ConfigError`): the expanded engine, duration-dependent jumps (their `(regime, elapsed)`
chain is handled by `expanded_estep`; `xi_by_d` stays `None`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.engine.batch import _dataset_view
from hsmmkit.engine.explicit_duration import (
    MAX_CELLS,
    _backward_cont,
    _build_inputs,
    _forward_end,
    _h_ext,
    _loglik,
    _regime_tables,
    _terminal,
)
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

__all__ = ["ExplicitEStats", "explicit_estep"]

_NEG_INF = -jnp.inf
_CHUNK = 128


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "loglik",
        "eta",
        "eta_censored",
        "eta_first",
        "eta_entry",
        "eta_entry_censored",
        "xi",
        "xi_by_d",
        "gamma_regime",
        "rho_post",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class ExplicitEStats:
    """Expected segment and jump counts of one explicit-engine E-step (a pytree).

    Attributes:
        loglik: `(B,)` right-censored log-likelihood per sequence (0 for empty padding sequences).
        eta: `(N, D_max)` expected complete segments of regime `j` with duration `d` (sum over sequences).
        eta_censored: `(N, D_max)` expected final segments covering `T`, by elapsed `d = T - s`
            (an absorbing regime's final segment longer than `D_max` is counted in the last bin).
        eta_first: `(N, D_max)` expected closed left-censored first segments (zeros without left-censoring).
        eta_entry: `(B, T_max, N, D_max)` expected complete segments per start `s`, only with entry-time
            duration covariates (else `None`).
        eta_entry_censored: `(B, T_max, N)` expected final segment per start (same condition).
        xi: `(N, N)` expected jumps `i -> j`.
        xi_by_d: `(N, N, D_max)` by the duration of the segment left; always `None` here.
        gamma_regime: `(B, T_max, N)` regime marginals, 0 on padding.
        rho_post: `(G, N)` posterior of the first segment's regime, summed per initial group.
    """

    loglik: Array
    eta: Array
    eta_censored: Array
    eta_first: Array
    eta_entry: Array | None
    eta_entry_censored: Array | None
    xi: Array
    xi_by_d: Array | None
    gamma_regime: Array
    rho_post: Array


def _estep_one(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    data: Any,  # noqa: ANN401
    length: Array,
    code: Array | None,
    *,
    entry: bool,
) -> tuple[Array, ...]:
    """One sequence of true length `length` (>= 1): the accumulators of `ExplicitEStats` (no batch sums)."""
    inp = _build_inputs(theta, spec, space, data, length, code)
    n, d_max = spec.n_states, spec.d_max
    t_len = inp.cum.shape[0] - 1
    inner, g, end = _forward_end(inp)
    term = _terminal(inp)
    ll = _loglik(inner, term)
    beta, cont = _backward_cont(inp, term)
    h_ext = _h_ext(inp, beta)  # (T + D, N)
    lp, log_pi = inp.log_pmf, inp.log_pi
    first_censored = spec.left_censored

    def norm(x: Array) -> Array:
        ok = jnp.isfinite(x)
        return jnp.where(ok, jnp.exp(jnp.where(ok, x - ll, 0.0)), 0.0)

    def seg(carry: tuple[Array, Array, Array], s: Array) -> tuple[tuple[Array, Array, Array], Any]:
        eta, eta_c, eta_f = carry
        lp_s = lp[s] if lp.ndim == 3 else lp  # (N, D)
        h = jax.lax.dynamic_slice(h_ext, (s + 1, 0), (d_max, n))  # H[s + 1 .. s + D], (D, N)
        closed = norm(g[s][:, None] + lp_s + h.T)  # (N, D)
        elapsed = jnp.minimum(length - s, d_max) - 1  # bin of the final segment starting at s
        final = norm(inner[s] + term[s])  # (N,) mass of the final segment from s
        cens = jnp.zeros((n, d_max)).at[:, jnp.maximum(elapsed, 0)].set(final)
        first = (s == 0) & first_censored
        eta = eta + jnp.where(first, 0.0, closed)
        eta_f = eta_f + jnp.where(first, closed, 0.0)
        eta_c = eta_c + cens
        return (eta, eta_c, eta_f), ((closed, final) if entry else None)

    zeros = jnp.zeros((n, d_max))
    (eta, eta_c, eta_f), per_start = jax.lax.scan(seg, (zeros, zeros, zeros), jnp.arange(t_len))

    def jump(xi: Array, tau: Array) -> tuple[Array, None]:
        pi_t = log_pi[tau] if log_pi.ndim == 3 else log_pi
        return xi + norm(end[tau][:, None] + pi_t + cont[tau][None, :]), None

    xi, _ = jax.lax.scan(jump, jnp.zeros((n, n)), jnp.arange(1, t_len))
    rho_post = norm(inp.log_rho + cont[0])
    smoothed = _regime_tables(inp, inner, g, term, beta, ll, min(max(_CHUNK, d_max), max(t_len, 1)))[2]
    out: tuple[Array, ...] = (ll, eta, eta_c, eta_f, xi, smoothed, rho_post)
    if entry and per_start is not None:
        out = (*out, per_start[0], per_start[1])
    return out


@partial(jax.jit, static_argnames=("spec", "entry"))
def _estep_batch(
    theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch, entry: bool
) -> ExplicitEStats:
    data = _dataset_view(pb)
    grouped = "initial.logits" in theta and theta["initial.logits"].ndim == 2
    active = pb.boundary_code is not None
    codes = jnp.zeros(pb.valid.shape, dtype=jnp.int8) if pb.boundary_code is None else pb.boundary_code

    def one(d: Any, n_b: Array, g: Array, c: Array) -> tuple[Array, ...]:  # noqa: ANN401
        th = {**theta, "initial.logits": theta["initial.logits"][g]} if grouped else theta
        return _estep_one(th, spec, space, d, jnp.maximum(n_b, 1), c if active else None, entry=entry)

    out = jax.vmap(one)(data, pb.lengths, pb.group_index, codes)
    ll, eta, eta_c, eta_f, xi, smoothed, rho_post = out[:7]
    keep = (pb.lengths > 0).astype(ll.dtype)

    def total(x: Array) -> Array:
        return jnp.sum(x * keep.reshape((-1,) + (1,) * (x.ndim - 1)), axis=0)

    gamma = jnp.where(pb.valid[:, :, None] & (keep > 0)[:, None, None], smoothed, 0.0)
    rho = jax.ops.segment_sum(rho_post * keep[:, None], pb.group_index, num_segments=spec.n_groups)
    entry_k = out[7] * keep[:, None, None, None] if entry else None
    entry_c = out[8] * keep[:, None, None] if entry else None
    return ExplicitEStats(
        loglik=jnp.where(pb.lengths > 0, ll, 0.0),
        eta=total(eta),
        eta_censored=total(eta_c),
        eta_first=total(eta_f),
        eta_entry=entry_k,
        eta_entry_censored=entry_c,
        xi=total(xi),
        xi_by_d=None,
        gamma_regime=gamma,
        rho_post=rho,
    )


def explicit_estep(
    engine: Any,  # noqa: ANN401
    params: Params,
    spec: ModelSpec,
    space: StateSpace,
    pb: PaddedBatch,
) -> ExplicitEStats:
    """The expected segment / jump counts of `pb` under `params` (`engine: explicit`).

    Args:
        engine: the `ExplicitDurationEngine` (kept for symmetry with `expanded_estep`).
        params: constrained parameters.
        spec: model structure with `engine: explicit` (left-censoring, absorbing regimes, entry covariates,
            per-state durations and supervision are honoured as in the forward-backward pass).
        space: `build_explicit(spec)`.
        pb: the padded batch (`SequenceBatch.padded()`).

    Raises:
        ConfigError: expanded engine, or duration-dependent jumps.
        NumericalError: `T N D_max > 5e7` with entry-time covariates (`eta_entry` needs the full table), or a
            non-finite log-likelihood of a non-empty sequence.
    """
    del engine
    if spec.engine != "explicit":
        raise ConfigError(
            "explicit_estep needs engine: explicit (the expanded engine: expanded_estep)", path="model.engine"
        )
    if spec.duration_dependent_jumps:
        raise ConfigError(
            "explicit_estep does not support duration-dependent jumps "
            "(use the expanded (regime, elapsed) chain)",
            path="model.duration.duration_dependent_jumps",
        )
    t_max = int(pb.observations.shape[1])
    entry = bool(pb.cov_duration.shape[-1] > 0 and getattr(spec.duration, "cov_idx", ()))
    if entry and t_max * spec.n_states * spec.d_max > MAX_CELLS:
        raise NumericalError(
            f"T*N*D_max = {t_max * spec.n_states * spec.d_max:.3g} > {MAX_CELLS:.0e}: "
            "eta_entry needs the full table",
            path="model.d_max",
        )
    stats: ExplicitEStats = _estep_batch(params, spec, space, pb, entry)
    live = np.asarray(pb.lengths) > 0
    ll = np.asarray(stats.loglik)[live]
    if not np.all(np.isfinite(ll)):
        raise NumericalError(
            f"non-finite log-likelihood in the E-step (sequence {int(np.flatnonzero(~np.isfinite(ll))[0])})",
            path="loglik",
        )
    return stats
