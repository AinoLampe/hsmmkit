"""`RingBuffer` and the fixed-lag smoother (E15-T02).

Concept refs: Part II §14.11 (fixed-lag), §16.3 item 12. The buffer keeps the last `L + 1` steps `t-L … t`
as fixed-size arrays with a write pointer (`jit`-friendly, never reallocated): the emission log-terms
`ell`, the transition/duration covariate rows and `dt` (the transition `log A_u` is rebuilt on demand,
`O(L (S + K))` memory instead of `O(L S^2)`), and the forward vectors `log_alpha`.

`smoothed_at` runs the backward recursion over the buffer from `beta_t = 0` for `L` steps and returns the
regime marginals of step `t - L`. These equal the offline smoother run on the *prefix* `y_{1:t}`
(`data.window(0, t + 1)`), not the full-data smoother `gamma_{t-L | T}` (which they approach for `T >> t`);
at the end of a stream (`t = T`) the two coincide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from hsmmkit.engine.expanded_hmm import _safe_logsumexp, backward_step

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.estimation.results import FitResult
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace
    from hsmmkit.utils.typing import Array


@dataclass(frozen=True)
class RingBuffer:
    """The last `L + 1` steps for fixed-lag smoothing (a pytree of arrays).

    Attributes:
        ell: `(L+1, S)` emission log-terms `ell_u`.
        z_pi: `(L+1, K_pi)` transition covariate rows.
        z_d: `(L+1, K_d)` duration covariate rows.
        dt: `(L+1,)` step lengths.
        log_alpha: `(L+1, S)` unnormalized forward log-vectors.
        head: `()` int32, the slot the next step is written to.
        count: `()` int32, filled slots (`<= L + 1`).
    """

    ell: Array
    z_pi: Array
    z_d: Array
    dt: Array
    log_alpha: Array
    head: Array
    count: Array

    @property
    def lag(self) -> int:
        """`L`: the capacity is `L + 1` steps."""
        return int(self.ell.shape[0]) - 1


def _flatten(b: RingBuffer) -> tuple[tuple[Array, ...], None]:
    return (b.ell, b.z_pi, b.z_d, b.dt, b.log_alpha, b.head, b.count), None


def _unflatten(_: None, children: tuple[Array, ...]) -> RingBuffer:
    return RingBuffer(*children)


jax.tree_util.register_pytree_node(RingBuffer, _flatten, _unflatten)


def empty_buffer(lag: int, n_expanded: int, k_pi: int, k_d: int) -> RingBuffer:
    """A buffer with no step written yet."""
    n = lag + 1
    return RingBuffer(
        ell=jnp.zeros((n, n_expanded)),
        z_pi=jnp.zeros((n, k_pi)),
        z_d=jnp.zeros((n, k_d)),
        dt=jnp.ones((n,)),
        log_alpha=jnp.zeros((n, n_expanded)),
        head=jnp.zeros((), dtype=jnp.int32),
        count=jnp.zeros((), dtype=jnp.int32),
    )


def push(
    buf: RingBuffer, ell_t: Array, z_pi_t: Array, z_d_t: Array, dt_t: Array, alpha_t: Array
) -> RingBuffer:
    """Write one step at `head` and advance the pointer."""
    n = buf.ell.shape[0]
    h = buf.head
    return RingBuffer(
        ell=buf.ell.at[h].set(ell_t),
        z_pi=buf.z_pi.at[h].set(z_pi_t),
        z_d=buf.z_d.at[h].set(z_d_t),
        dt=buf.dt.at[h].set(dt_t),
        log_alpha=buf.log_alpha.at[h].set(alpha_t),
        head=(h + 1) % n,
        count=jnp.minimum(buf.count + 1, n),
    )


def smooth_window(theta: Params, space: StateSpace, buf: RingBuffer, lag: int) -> Array:
    """Regime marginals `(N,)` of step `t - lag` given data up to `t`; `NaN` until `lag + 1` steps exist.

    `lag <= buf.lag` (static). Cost `O(lag S^2)`.
    """
    n = buf.ell.shape[0]
    k = jnp.arange(lag)
    idx = (buf.head - 1 - k) % n  # entries t, t-1, ..., t-lag+1 (newest first)
    xs = (
        buf.ell[idx],
        buf.z_pi[idx],
        buf.z_d[idx],
        buf.dt[idx],
        jnp.zeros((lag,), dtype=jnp.int8),
    )

    def body(beta_next: Array, x: tuple[Array, ...]) -> tuple[Array, None]:
        return backward_step(space, theta, False, beta_next, *x), None

    beta, _ = jax.lax.scan(body, jnp.zeros((buf.ell.shape[1],)), xs)
    alpha = buf.log_alpha[(buf.head - 1 - lag) % n]
    log_gamma = alpha + beta
    norm = _safe_logsumexp(log_gamma, axis=0)
    finite = jnp.isfinite(log_gamma)
    gamma = jnp.where(finite, jnp.exp(jnp.where(finite, log_gamma - norm, 0.0)), 0.0)
    out = gamma @ space.index.collapse
    return jnp.where(buf.count >= lag + 1, out, jnp.nan)


def buffer_from_prefix(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    data: PreparedDataset,
    log_alpha: Array,
    lag: int,
) -> RingBuffer:
    """The buffer after the steps of `data` (warm start); `log_alpha (T, S)` is its offline forward pass."""
    from hsmmkit.engine.expanded_hmm import _block, _expanded_log_emis

    t_len = int(data.observations.shape[0])
    k_pi = int(theta["transition.b"].shape[-1]) if "transition.b" in theta else 0
    k_d = int(theta["duration.c"].shape[-1]) if "duration.c" in theta else 0
    buf = empty_buffer(lag, space.n_expanded, k_pi, k_d)
    take = min(t_len, lag + 1)
    if take == 0:
        return buf
    lo = t_len - take
    ell = _expanded_log_emis(theta, spec, space, data)
    z_pi, z_d = _block(data.cov_transition, t_len), _block(data.cov_duration, t_len)
    n = lag + 1
    slots = jnp.arange(take)  # the first `take` steps of the window go to slots 0..take-1
    return RingBuffer(
        ell=buf.ell.at[slots].set(ell[lo:]),
        z_pi=buf.z_pi.at[slots].set(z_pi[lo:]),
        z_d=buf.z_d.at[slots].set(z_d[lo:]),
        dt=buf.dt.at[slots].set(data.dt[lo:]),
        log_alpha=buf.log_alpha.at[slots].set(log_alpha[lo:]),
        head=jnp.asarray(take % n, dtype=jnp.int32),
        count=jnp.asarray(take, dtype=jnp.int32),
    )


def smoothed_series(fit: FitResult, data: PreparedDataset, lag: int) -> Array:
    """`(T, N)` fixed-lag smoothed marginals of a replay: row `t` is `gamma_{t-lag | t}` (`NaN` if `t < lag`).

    One offline forward pass, then a backward window of `lag` steps per `t` (vectorized over `t`); equals
    what the streaming filter returns at each step.
    """
    from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine, _block, _expanded_log_emis
    from hsmmkit.statespace.builders import build_expanded

    spec = fit.spec
    space = build_expanded(spec)
    theta: Params = dict(fit.params)
    log_alpha, _ = ExpandedHMMEngine().forward(theta, spec, space, data)
    t_len = int(data.observations.shape[0])
    ell = _expanded_log_emis(theta, spec, space, data)
    z_pi, z_d = _block(data.cov_transition, t_len), _block(data.cov_duration, t_len)
    s = space.n_expanded

    def one(t: Array) -> Array:
        steps = t - jnp.arange(lag)  # t, t-1, ..., t-lag+1
        safe = jnp.maximum(steps, 0)
        xs = (ell[safe], z_pi[safe], z_d[safe], data.dt[safe], jnp.zeros((lag,), dtype=jnp.int8))

        def body(beta_next: Array, x: tuple[Array, ...]) -> tuple[Array, None]:
            return backward_step(space, theta, False, beta_next, *x), None

        beta, _ = jax.lax.scan(body, jnp.zeros((s,)), xs)
        log_gamma = log_alpha[jnp.maximum(t - lag, 0)] + beta
        norm = _safe_logsumexp(log_gamma, axis=0)
        finite = jnp.isfinite(log_gamma)
        gamma = jnp.where(finite, jnp.exp(jnp.where(finite, log_gamma - norm, 0.0)), 0.0)
        return jnp.where(t >= lag, gamma @ space.index.collapse, jnp.nan)

    out: Array = jax.jit(jax.vmap(one))(jnp.arange(t_len))
    return out


__all__ = ["RingBuffer", "buffer_from_prefix", "empty_buffer", "push", "smooth_window", "smoothed_series"]
