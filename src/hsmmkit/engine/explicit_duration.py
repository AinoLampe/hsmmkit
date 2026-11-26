"""`ExplicitDurationEngine`: Guedon/Yu explicit-duration forward-backward with right-censoring (E4-T06a).

Concept refs: §4.7.3, Appendix C, ADR-0007. Segment-based recursion over boundaries `tau = 0..T`
(`tau` steps consumed; a segment starting at boundary `s` covers steps `s..s+d-1`), with cumulative
emission sums `C[tau, j] = sum_{u < tau} ell[u, j]` so a segment's emission is `C[s+d, j] - C[s, j]`:

    E[t, j]     = C[t, j] + logsumexp_d (G[t-d, j] + lp_{t-d}[j, d])        segment ends exactly at t
    inner[s, j] = log rho_j (s = 0)  |  logsumexp_i (E[s, i] + log pi_s[i, j])   segment starts at s
    G[s, j]     = inner[s, j] - C[s, j]

`lp_s[j, d]` is the log duration pmf and `log pi_s` the log jump matrix of a segment *starting* at `s`
(entry-time covariates, CLAUDE.md §3.7: covariate row `s` is usable at step `s`). Forward is a scan over
`s` that scatters each segment start into a rolling `(D_max, N)` buffer of pending segment ends, so a
step costs `O(N^2 + N D_max)`. The last segment may still be in progress at `T`: the log-likelihood uses
the survival `log P(d >= T - s)` for it. The augmented state is `(regime, elapsed)` with
`s = j D_max + (e - 1)`; `log_alpha[t, (j, e)] = log P(y_{1:t}, regime j, e steps elapsed at t)`,
`smoothed` the elapsed posterior `P(z_t = j, e | y)` and `log_beta = log(smoothed) + loglik - log_alpha`
(exact: the `(regime, elapsed)` chain is Markov, so `alpha * beta` is the joint).

The grid is assumed regular: `nominal_dt` replaces `dt` everywhere and an irregular grid warns once
per call. Full `(T, N, D_max)` tables are returned, so `T * N * D_max > 5e7` raises `NumericalError`
(chunked scan: E4-T06b).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.engine.base import Posterior
from hsmmkit.engine.expanded_hmm import _check_finite, _check_posterior, _regime_log_emis, _safe_logsumexp
from hsmmkit.model.duration.base import residual_log_pmf
from hsmmkit.model.transition import build_memory_layout
from hsmmkit.statespace.index import StateIndex
from hsmmkit.utils.clock import is_irregular
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

MAX_CELLS = 5e7
_AUTO_CHUNK = 1024
_NEG_INF = -jnp.inf


class _Inputs(NamedTuple):
    """Recursion inputs: `(N,)` initial, `(T+1, N)` cumulative emissions, jump and pmf tables."""

    log_rho: Array  # (N,)
    cum: Array  # (T+1, N)
    log_pi: Array  # (N, N) or (T, N, N), indexed by segment start
    log_pmf: Array  # (N, D) or (T, N, D), indexed by segment start
    absorbing: Array | None = None  # (N,) bool: regimes whose final segment never ends (E12-T03)


def _logaddexp(a: Array, b: Array) -> Array:
    """`log(exp(a) + exp(b))` that is `-inf` (with zero gradient) when both are `-inf`."""
    a, b = jnp.broadcast_arrays(a, b)
    return _safe_logsumexp(jnp.stack([a, b]), axis=0)


def _log_survival(log_pmf: Array) -> Array:
    """`log P(d >= e)` at `[..., j, e-1]`: reverse cumulative `logsumexp` over the duration axis."""
    return jax.lax.cumlogsumexp(log_pmf, axis=log_pmf.ndim - 1, reverse=True)


def _log_jump(spec: ModelSpec, space: StateSpace, theta: Params, data: PreparedDataset) -> Array:
    """`log pi`: `(N, N)` when constant, `(T, N, N)` with transition covariates (row = segment start)."""
    z_pi = data.cov_transition
    if z_pi is None or z_pi.shape[-1] == 0 or "transition.b" not in theta:
        pi = spec.transition.jump_probs_t(theta, None, space.layout)
    else:
        pi = spec.transition.jump_probs(theta, z_pi, space.layout)
    with_zero_safe = jnp.where(pi > 0, pi, 1.0)
    return jnp.where(pi > 0, jnp.log(with_zero_safe), _NEG_INF)


def _log_pmf(spec: ModelSpec, theta: Params, data: PreparedDataset) -> Array:
    """`(N, D_max)`, or `(T, N, D_max)` with duration covariates (entry-time link at the segment start)."""
    family, d_max, nominal_dt = spec.duration, spec.d_max, data.nominal_dt
    z_d = data.cov_duration
    if z_d is None or z_d.shape[-1] == 0 or not getattr(family, "cov_idx", ()):
        lp = jnp.asarray(family.log_pmf(theta, d_max, nominal_dt, None))
    else:
        lp = jax.vmap(lambda z: family.log_pmf(theta, d_max, nominal_dt, z))(z_d)
    topology = getattr(spec.transition, "topology", None)
    if topology is not None and topology.absorbing:
        # an absorbing regime never leaves: `p(d) = delta_{d, D_max}` and the final segment is right-censored
        stays = np.full(d_max, -np.inf)
        stays[-1] = 0.0
        mask = np.isin(np.arange(spec.n_states), topology.absorbing)[:, None]
        lp = jnp.where(jnp.asarray(mask), jnp.asarray(stays), lp)
    return lp


def _stationary_jump(pi: Array, n_iter: int = 2000) -> Array:
    """Stationary distribution of the jump matrix (lazy power iteration: also converges if periodic)."""
    n = pi.shape[0]

    def body(v: Array, _: None) -> tuple[Array, None]:
        v = 0.5 * (v + v @ pi)
        return v / jnp.sum(v), None

    v, _ = jax.lax.scan(body, jnp.full((n,), 1.0 / n), None, length=n_iter)
    return v


def _log_rho(spec: ModelSpec, theta: Params, log_pmf0: Array | None = None) -> Array:
    """Initial regime log-distribution of the first segment.

    Left-censored (`log_pmf0` = the first-start pmf `(N, D)`): `rho_j ~ nu_j E[d_j]`, the stationary occupancy
    (`nu` the stationary distribution of the jump matrix at `z = 0`), unless `initial_state: free`.
    """
    if log_pmf0 is not None and spec.initial != "free":
        pi = spec.transition.jump_probs_t(
            theta, None, build_memory_layout(spec.n_states, 1, getattr(spec.transition, "topology", None))
        )
        nu = _stationary_jump(pi)
        d = jnp.arange(1, log_pmf0.shape[-1] + 1)
        mean_d = jnp.sum(jnp.exp(log_pmf0) * d, axis=-1)
        w = nu * mean_d
        return jnp.log(w / jnp.sum(w))
    if spec.initial == "uniform":
        return jnp.full((spec.n_states,), -jnp.log(spec.n_states))
    if spec.initial == "free":
        return jnp.log(theta["initial.logits"])
    raise ConfigError(
        "the explicit-duration engine supports initial in {'uniform', 'free'}", path="model.initial_state"
    )


def _cells(spec: ModelSpec, data: PreparedDataset) -> int:
    """`T * N * D_max`, the size of a full `(T, N, D_max)` table."""
    return int(data.observations.shape[0] * spec.n_states * spec.d_max)


def _has_entry_covariates(spec: ModelSpec, data: PreparedDataset) -> bool:
    z_d = data.cov_duration
    return bool(z_d is not None and z_d.shape[-1] > 0 and getattr(spec.duration, "cov_idx", ()))


def _prepare(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset, *, full_tables: bool
) -> _Inputs:
    """Validate the call (once per public method) and build the recursion inputs.

    `full_tables` is set by calls that return `(T, N, D_max)` tables; the others only need `(T, N)`
    arrays unless entry-time duration covariates make the pmf itself `(T, N, D_max)`.
    """
    cells = _cells(spec, data)
    if cells > MAX_CELLS and (full_tables or spec.left_censored or _has_entry_covariates(spec, data)):
        raise NumericalError(
            f"T*N*D_max = {cells:.3g} > {MAX_CELLS:.0e}: use chunked scan (estimation.chunk_size)",
            path="model.d_max",
        )
    if space.index.n_expanded != spec.n_states * spec.d_max:
        raise ConfigError(
            "state space does not match the explicit engine; build it with build_explicit(spec)",
            path="model.engine",
        )
    if not isinstance(data.dt, jax.core.Tracer) and is_irregular(data.dt):
        warnings.warn(
            "explicit-duration engine assumes a regular grid: using nominal_dt for an irregular dt",
            UserWarning,
            stacklevel=3,
        )
    return _build_inputs(theta, spec, space, data)


def _build_inputs(theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> _Inputs:
    dt = jnp.full_like(data.dt, data.nominal_dt)
    ell = _regime_log_emis(theta, spec, data, dt)  # (T, N)
    cum = jnp.concatenate([jnp.zeros((1, ell.shape[1])), jnp.cumsum(ell, axis=0)], axis=0)
    topology = getattr(spec.transition, "topology", None)
    log_pmf = _log_pmf(spec, theta, data)
    log_pmf0 = None
    if spec.left_censored:
        # first segment: residual-life pmf (needs a per-start table), stationary occupancy as the initial law
        table = (
            jnp.broadcast_to(log_pmf, (ell.shape[0], *log_pmf.shape[-2:])) if log_pmf.ndim == 2 else log_pmf
        )
        log_pmf0 = table[0]
        log_pmf = table.at[0].set(residual_log_pmf(log_pmf0))
    absorbing = None
    if topology is not None and topology.absorbing:
        absorbing = jnp.asarray(np.isin(np.arange(spec.n_states), topology.absorbing))
    return _Inputs(
        _log_rho(spec, theta, log_pmf0),
        cum,
        _log_jump(spec, space, theta, data),
        log_pmf,
        absorbing,
    )


def _at(x: Array, s: int) -> Array:
    """`x[s]` for a per-start table, `x` itself for a constant one (`ndim` is static)."""
    return x[s] if x.ndim == 3 else x


def _forward(inp: _Inputs) -> tuple[Array, Array]:
    """`(inner (T, N), G (T, N))`: log-prob of a segment starting at each boundary `0..T-1`."""
    cum, lp = inp.cum, inp.log_pmf
    t_len, (d_max, n) = cum.shape[0] - 1, (lp.shape[-1], lp.shape[-2])
    pad = jnp.full((1, n), _NEG_INF)

    def scatter(pend: Array, g: Array, lp_s: Array) -> Array:
        # contributions of a start at s to the ends at s+1..s+D; `pend[k]` is the end at boundary s+1+k
        return _logaddexp(jnp.concatenate([pend[1:], pad], axis=0), g[None, :] + lp_s.T)

    g0 = inp.log_rho  # cum[0] == 0
    pend1 = scatter(jnp.full((d_max, n), _NEG_INF), g0, _at(lp, 0))

    dummy = jnp.zeros((t_len - 1,))
    xs = (cum[1:t_len], inp.log_pi[1:] if inp.log_pi.ndim == 3 else dummy, lp[1:] if lp.ndim == 3 else dummy)

    def body(pend: Array, s_in: tuple[Array, Array, Array]) -> tuple[Array, tuple[Array, Array]]:
        c_s, pi_x, lp_x = s_in
        log_pi_s = pi_x if inp.log_pi.ndim == 3 else inp.log_pi
        end = c_s + pend[0]  # log E[s, :]
        inner = _safe_logsumexp(end[:, None] + log_pi_s, axis=0)
        g = inner - c_s
        return scatter(pend, g, lp_x if lp.ndim == 3 else lp), (inner, g)

    _, (inner_rest, g_rest) = jax.lax.scan(body, pend1, xs)
    inner = jnp.concatenate([inp.log_rho[None, :], inner_rest], axis=0)
    g = jnp.concatenate([g0[None, :], g_rest], axis=0)
    return inner, g


def _terminal(inp: _Inputs) -> Array:
    """`(T, N)`: `log S_s[j, T-s] + C[T, j] - C[s, j]`, a final segment from `s` (`-inf` if `T-s > D`)."""
    cum, lp = inp.cum, inp.log_pmf
    t_len, n, d_max = cum.shape[0] - 1, lp.shape[-2], lp.shape[-1]
    surv = _log_survival(lp)
    length = t_len - jnp.arange(t_len)  # observed final-segment length d = T - s
    idx = jnp.clip(length - 1, 0, d_max - 1)
    if surv.ndim == 2:
        picked = surv[:, idx].T  # (T, N)
    else:
        picked = surv[jnp.arange(t_len)[:, None], jnp.arange(n)[None, :], idx[:, None]]
    value = picked + cum[t_len][None, :] - cum[:t_len]
    fits = (length <= d_max)[:, None]
    if inp.absorbing is not None:
        # an absorbing regime never leaves: its final segment has survival 1 at every length (no D_max cap)
        never_ends = cum[t_len][None, :] - cum[:t_len]
        value = jnp.where(inp.absorbing[None, :], never_ends, value)
        fits = fits | inp.absorbing[None, :]
    return jnp.where(fits, value, _NEG_INF)


def _loglik(inner: Array, term: Array) -> Array:
    return _safe_logsumexp((inner + term).reshape(-1), axis=0)


def _backward(inp: _Inputs, term: Array) -> tuple[Array, Array]:
    """`(B (T, N), cont0 (N,))`: `B[tau, i]` = future log-prob given a regime-`i` segment ends at `tau`."""
    cum, lp, log_pi = inp.cum, inp.log_pmf, inp.log_pi
    t_len, n, d_max = cum.shape[0] - 1, lp.shape[-2], lp.shape[-1]

    def body(hbuf: Array, s_in: tuple[Array, Array, Array, Array]) -> tuple[Array, tuple[Array, Array]]:
        c_s, term_s, pi_x, lp_x = s_in
        pi_s = pi_x if log_pi.ndim == 3 else log_pi
        lp_s = lp_x if lp.ndim == 3 else lp
        closed = _safe_logsumexp(lp_s.T + hbuf, axis=0) - c_s  # (N,), `hbuf[k]` = H[tau+1+k]
        cont = _logaddexp(closed, term_s)
        beta = _safe_logsumexp(pi_s + cont[None, :], axis=1)
        h = c_s + beta
        return jnp.concatenate([h[None, :], hbuf[:-1]], axis=0), (beta, cont)

    dummy = jnp.zeros((t_len,))
    xs = (cum[:t_len], term, log_pi if log_pi.ndim == 3 else dummy, lp if lp.ndim == 3 else dummy)
    _, (beta, cont) = jax.lax.scan(body, jnp.full((d_max, n), _NEG_INF), xs, reverse=True)
    return beta, cont[0]


def _h_ext(inp: _Inputs, beta: Array) -> Array:
    """`(T + D, N)`: `H = C + B` per boundary, `-inf` past the last boundary."""
    t_len, n, d_max = inp.cum.shape[0] - 1, inp.log_pmf.shape[-2], inp.log_pmf.shape[-1]
    return jnp.concatenate([inp.cum[:t_len] + beta, jnp.full((d_max, n), _NEG_INF)], axis=0)


def _chunk(
    inp: _Inputs, inner: Array, g: Array, term: Array, h_ext: Array, loglik: Array, a: Array | int, c: int
) -> tuple[Array, Array]:
    """Filtered and smoothed `(regime, elapsed)` tables for steps `t = a .. a+c-1`: two `(c, N, D)` arrays.

    Only the `c + D - 1` segment starts that can cover those steps are touched, so a chunk costs
    `O((c + D) N D)` memory whatever `T` is. `log_alpha` is `-inf` and `smoothed` 0 where the
    elapsed count is impossible (`e > t + 1`) or `t >= T` (last, padded chunk).
    """
    cum, lp = inp.cum, inp.log_pmf
    t_len, n, d_max = cum.shape[0] - 1, lp.shape[-2], lp.shape[-1]
    span = c + d_max - 1
    s = a - d_max + 1 + jnp.arange(span)  # segment starts touched by this chunk
    in_range = ((s >= 0) & (s <= t_len - 1))[:, None, None]
    sc = jnp.clip(s, 0, t_len - 1)
    lp_s = lp[sc] if lp.ndim == 3 else lp[None]
    surv_s = _log_survival(lp_s)

    h_gather = h_ext[sc[:, None] + jnp.arange(1, d_max + 1)[None, :]]  # (span, D, N)
    closed = inner[sc][:, :, None] - cum[sc][:, :, None] + lp_s + jnp.swapaxes(h_gather, 1, 2)
    final = (inner[sc] + term[sc])[:, :, None]
    weight = _logaddexp(jax.lax.cumlogsumexp(closed, axis=2, reverse=True), final)  # (span, N, D)
    weight = jnp.where(in_range, weight, _NEG_INF)
    alpha_part = jnp.where(in_range, g[sc][:, :, None] + surv_s, _NEG_INF)

    u = jnp.arange(c)[:, None]
    e = jnp.arange(1, d_max + 1)[None, :]
    local = (u + d_max - e)[:, None, :]  # index into the touched starts of s = t - e + 1
    t = a + u  # (c, 1)
    valid = ((t < t_len) & (t - e + 1 >= 0))[:, None, :]
    jn = jnp.arange(n)[None, :, None]
    ek = (e - 1)[None]
    log_elapsed = weight[local, jn, ek]
    log_alpha = alpha_part[local, jn, ek] + cum[jnp.minimum(t[:, 0] + 1, t_len)][:, :, None]
    finite = valid & jnp.isfinite(log_elapsed)
    smoothed = jnp.where(finite, jnp.exp(jnp.where(finite, log_elapsed - loglik, 0.0)), 0.0)
    return jnp.where(valid, log_alpha, _NEG_INF), smoothed


def _beta_from(log_alpha: Array, smoothed: Array, loglik: Array) -> Array:
    """`log_beta = log(smoothed) + loglik - log_alpha` where both are finite (else `-inf`)."""
    log_gamma = jnp.where(smoothed > 0, jnp.log(jnp.where(smoothed > 0, smoothed, 1.0)) + loglik, _NEG_INF)
    ok = jnp.isfinite(log_gamma) & jnp.isfinite(log_alpha)
    return jnp.where(ok, jnp.where(ok, log_gamma - log_alpha, 0.0), _NEG_INF)


def _full_tables(
    inp: _Inputs, inner: Array, g: Array, term: Array, beta: Array, loglik: Array
) -> tuple[Array, Array, Array]:
    """`(log_alpha, log_beta, smoothed)`, all `(T, N * D_max)` (regime-major state order)."""
    t_len, n, d_max = inp.cum.shape[0] - 1, inp.log_pmf.shape[-2], inp.log_pmf.shape[-1]
    log_alpha, smoothed = _chunk(inp, inner, g, term, _h_ext(inp, beta), loglik, 0, t_len)
    log_beta = _beta_from(log_alpha, smoothed, loglik)
    flat = (t_len, n * d_max)
    return log_alpha.reshape(flat), log_beta.reshape(flat), smoothed.reshape(flat)


def _regime_tables(
    inp: _Inputs, inner: Array, g: Array, term: Array, beta: Array, loglik: Array, chunk: int
) -> tuple[Array, Array, Array]:
    """`(log_alpha, log_beta, smoothed)` collapsed over elapsed, `(T, N)`, in chunks of `chunk` steps."""
    t_len = inp.cum.shape[0] - 1
    h_ext = _h_ext(inp, beta)
    n_chunks = -(-t_len // chunk)

    def one(a: Array) -> tuple[Array, Array]:
        log_alpha, smoothed = _chunk(inp, inner, g, term, h_ext, loglik, a, chunk)
        return _safe_logsumexp(log_alpha, axis=2), jnp.sum(smoothed, axis=2)

    log_alpha, smoothed = jax.lax.map(one, jnp.arange(n_chunks) * chunk)
    n = smoothed.shape[-1]
    log_alpha, smoothed = log_alpha.reshape(-1, n)[:t_len], smoothed.reshape(-1, n)[:t_len]
    return log_alpha, _beta_from(log_alpha, smoothed, loglik), smoothed


@partial(jax.jit, static_argnames=("spec",))
def _loglik_core(theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
    inp = _build_inputs(theta, spec, space, data)
    inner, _ = _forward(inp)
    return _loglik(inner, _terminal(inp))


@partial(jax.jit, static_argnames=("spec",))
def _forward_core(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array]:
    inp = _build_inputs(theta, spec, space, data)
    inner, g = _forward(inp)
    term = _terminal(inp)
    t_len, n, d_max = inp.cum.shape[0] - 1, inp.log_pmf.shape[-2], inp.log_pmf.shape[-1]
    h_ext = jnp.full((t_len + d_max, n), _NEG_INF)  # the filter does not need the backward pass
    log_alpha, _ = _chunk(inp, inner, g, term, h_ext, jnp.zeros(()), 0, t_len)
    return log_alpha.reshape(t_len, n * d_max), _loglik(inner, term)


@partial(jax.jit, static_argnames=("spec", "chunk"))
def _forward_backward_core(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset, chunk: int | None = None
) -> tuple[Array, Array, Array, Array, Array]:
    """`(log_alpha, log_beta, smoothed, loglik, loglik_backward)`; `(T, S)` tables, `(T, N)` if `chunk`."""
    inp = _build_inputs(theta, spec, space, data)
    inner, g = _forward(inp)
    term = _terminal(inp)
    loglik = _loglik(inner, term)
    beta, cont0 = _backward(inp, term)
    loglik_b = _safe_logsumexp(inp.log_rho + cont0, axis=0)
    if chunk is None:
        log_alpha, log_beta, smoothed = _full_tables(inp, inner, g, term, beta, loglik)
    else:
        log_alpha, log_beta, smoothed = _regime_tables(inp, inner, g, term, beta, loglik, chunk)
    return log_alpha, log_beta, smoothed, loglik, loglik_b


class _ViterbiTables(NamedTuple):
    inner: Array  # (T, N) best log-score of a segment starting at each boundary
    pred: Array  # (T, N) regime before a segment starting at s in regime j (-1 at s = 0)
    start_of: Array  # (T, N) start boundary of the best segment ending at t in regime j


def _viterbi_forward(inp: _Inputs) -> _ViterbiTables:
    """Max-product twin of `_forward`, keeping the argmax start of every pending segment end.

    Ties break to the shortest duration (later start wins on `>=`) and the lowest previous regime,
    as `ref_hsmm.segmental_viterbi` does.
    """
    cum, lp = inp.cum, inp.log_pmf
    t_len, n, d_max = cum.shape[0] - 1, lp.shape[-2], lp.shape[-1]

    def scatter(pend: Array, arg: Array, g: Array, lp_s: Array, s: Array | int) -> tuple[Array, Array]:
        shifted = jnp.concatenate([pend[1:], jnp.full((1, n), _NEG_INF)], axis=0)
        shifted_arg = jnp.concatenate([arg[1:], jnp.zeros((1, n), arg.dtype)], axis=0)
        new = g[None, :] + lp_s.T  # (D, N): the start at s ends at s + d
        take = new >= shifted
        return jnp.where(take, new, shifted), jnp.where(take, jnp.asarray(s, arg.dtype), shifted_arg)

    pend0 = jnp.full((d_max, n), _NEG_INF)
    arg0 = jnp.zeros((d_max, n), jnp.int32)
    pend1, arg1 = scatter(pend0, arg0, inp.log_rho, _at(lp, 0), 0)

    dummy = jnp.zeros((t_len - 1,))
    xs = (
        jnp.arange(1, t_len),
        cum[1:t_len],
        inp.log_pi[1:] if inp.log_pi.ndim == 3 else dummy,
        lp[1:] if lp.ndim == 3 else dummy,
    )

    def body(
        carry: tuple[Array, Array], s_in: tuple[Array, Array, Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        pend, arg = carry
        s, c_s, pi_x, lp_x = s_in
        log_pi_s = pi_x if inp.log_pi.ndim == 3 else inp.log_pi
        scores = (c_s + pend[0])[:, None] + log_pi_s  # best segment ending at s in i, then jump to j
        pred = jnp.argmax(scores, axis=0).astype(jnp.int32)
        inner = jnp.max(scores, axis=0)
        carry = scatter(pend, arg, inner - c_s, lp_x if lp.ndim == 3 else lp, s)
        return carry, (inner, pred, arg[0])

    _, (inner_rest, pred_rest, start_rest) = jax.lax.scan(body, (pend1, arg1), xs)
    return _ViterbiTables(
        jnp.concatenate([inp.log_rho[None, :], inner_rest], axis=0),
        jnp.concatenate([jnp.full((1, n), -1, jnp.int32), pred_rest], axis=0),
        jnp.concatenate([jnp.zeros((1, n), jnp.int32), start_rest], axis=0),
    )


def _viterbi_impl(inp: _Inputs) -> tuple[Array, Array]:
    """`(state path (T,), best log-probability)`; state `j * D_max + (e - 1)`."""
    t_len, n, d_max = inp.cum.shape[0] - 1, inp.log_pmf.shape[-2], inp.log_pmf.shape[-1]
    tables = _viterbi_forward(inp)
    final = tables.inner + _terminal(inp)  # (T, N): the last segment starts at s in regime j
    # first maximum over (regime, then shortest duration = latest start), as the reference
    flat = final.T[:, ::-1].reshape(-1)
    k = jnp.argmax(flat)
    j_last, s_last = k // t_len, t_len - 1 - k % t_len

    def back(carry: tuple[Array, Array], t: Array) -> tuple[tuple[Array, Array], Array]:
        j, s0 = carry
        state = (j * d_max + jnp.minimum(t - s0, d_max - 1)).astype(
            jnp.int32
        )  # elapsed - 1 = t - s0 (capped)
        prev = jnp.maximum(tables.pred[s0, j], 0)
        switch = t == s0
        s_prev = tables.start_of[s0, prev]
        return (jnp.where(switch, prev, j), jnp.where(switch, s_prev, s0)), state

    _, path = jax.lax.scan(
        back, (j_last.astype(jnp.int32), s_last.astype(jnp.int32)), jnp.arange(t_len - 1, -1, -1)
    )
    del n
    return path[::-1], jnp.max(flat)


@partial(jax.jit, static_argnames=("spec",))
def _viterbi_core(
    theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
) -> tuple[Array, Array]:
    return _viterbi_impl(_build_inputs(theta, spec, space, data))


def _regime_index(n_states: int) -> StateIndex:
    """`StateIndex` of the elapsed-collapsed posterior (`S = N`, `phase == 1`), used in chunked mode."""
    eye = jnp.eye(n_states)
    return StateIndex(
        regime_of=jnp.arange(n_states),
        phase_of=jnp.ones((n_states,), dtype=jnp.int64),
        memory_of=jnp.zeros((n_states, 0), dtype=jnp.int64),
        collapse=eye,
        row_of_state=jnp.arange(n_states),
        n_expanded=n_states,
        n_states=n_states,
        order=1,
    )


@dataclass(frozen=True)
class ExplicitDurationEngine:
    """HSMM inference with explicit sojourn pmfs (`engine == "explicit"`); state `(regime, elapsed)`.

    `chunk_size` (`estimation.chunk_size`) bounds memory: with `None`, `forward_backward` returns the
    full `(T, N * D_max)` posterior while `T * N * D_max <= 5e7` and otherwise switches to chunks of
    `max(D_max, 1024)` steps; an explicit value always chunks. A chunked `Posterior` is collapsed over
    the elapsed count: `(T, N)` arrays, `index.n_expanded == N`, `phase_of == 1`, so `regime_*` work and
    `elapsed_posterior()` is trivial. `loglik` and `viterbi` never allocate `(T, N, D_max)` tables.
    """

    chunk_size: int | None = None

    def forward(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> tuple[Array, Array]:
        """`(log_alpha (T, N * D_max), loglik)`; `loglik` is the right-censored 0-d log-likelihood."""
        _prepare(theta, spec, space, data, full_tables=True)
        log_alpha, loglik = _forward_core(theta, spec, space, data)
        _check_finite(loglik)
        return log_alpha, loglik

    def loglik(self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
        """`log p(y_1:T)`, differentiable w.r.t. `theta`."""
        _prepare(theta, spec, space, data, full_tables=False)
        loglik: Array = _loglik_core(theta, spec, space, data)
        _check_finite(loglik)
        return loglik

    def forward_backward(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> Posterior:
        """Posterior over `(regime, elapsed)`, or over regimes when chunked; `engine == "explicit"`."""
        chunk = self.chunk_size
        if chunk is None and _cells(spec, data) > MAX_CELLS:
            chunk = max(spec.d_max, _AUTO_CHUNK)
        if chunk is not None and chunk < 1:
            raise ConfigError("chunk_size must be >= 1", path="estimation.chunk_size")
        topology = getattr(spec.transition, "topology", None)
        if topology is not None and topology.absorbing and data.observations.shape[0] > spec.d_max:
            raise ConfigError(
                "the explicit-engine posterior of an absorbing regime needs d_max >= T "
                f"(d_max={spec.d_max}, T={data.observations.shape[0]}); loglik and viterbi do not",
                path="model.d_max",
            )
        _prepare(theta, spec, space, data, full_tables=chunk is None)
        log_alpha, log_beta, smoothed, loglik, _ = _forward_backward_core(theta, spec, space, data, chunk)
        _check_posterior(log_alpha, log_beta, smoothed, loglik)
        return Posterior(
            log_alpha=log_alpha,
            log_beta=log_beta,
            smoothed=smoothed,
            loglik=loglik,
            index=space.index if chunk is None else _regime_index(spec.n_states),
            engine="explicit",
        )

    def viterbi(self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
        """Most likely `(regime, elapsed)` path `(T,)`: state `j * D_max + e - 1`, censored last segment.

        The regime path is `space.index.regime_of[path]` and the elapsed count `space.index.phase_of[path]`.

        Raises:
            NumericalError: if the best segmentation has non-finite log-probability.
        """
        _prepare(theta, spec, space, data, full_tables=False)
        path: Array
        best: Array
        path, best = _viterbi_core(theta, spec, space, data)
        if not isinstance(best, jax.core.Tracer) and not bool(jnp.isfinite(best)):
            raise NumericalError("non-finite best-path log-probability in Viterbi", path="viterbi")
        return path

    def approx(self) -> None:
        """No approximate-inference plug-in (FR-X01 hook)."""
        return None
