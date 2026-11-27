"""Forward-filtering backward-sampling: exact posterior regime paths for both engines (E13-T04).

Concept refs: Part II §14.7; Part I §4.7.2-4.7.3.

* **Expanded engine** (`ffbs_expanded`): draw `s_T ~ softmax(alpha_T)`, then for `t = T-1 .. 1`
  `P(s_{t-1} = s | s_t) ~ exp(alpha_{t-1}(s)) A_t(s, s_t)`. The `(S, S)` matrix `A_t` is built once per
  step and shared by all samples (only the columns `A_t[:, s_t]` are gathered per sample): memory
  `O(S^2 + n S)` per step. Padding steps have identity transitions, so the chain stays put there; padded
  states are reported as `-1`.
* **Explicit engine** (`ffbs_explicit`): the last segment `(s, j)` is drawn from `inner + term` (the segment
  covering `T` is right-censored); then, for a segment of `j` starting at `tau`, the previous segment
  `(i, d)` is drawn with probability
  `~ exp(g[tau - d, i] + log p_{tau-d}[i, d] + cum[tau, i] + log pi_tau[i, j])` until `tau = 0` (a
  `lax.while_loop` of at most `T` iterations per sample). Left-censoring (residual first-segment pmf),
  labels, boundaries and topology enter through the same inputs as the forward pass.

Keys: sample `k` of sequence `b` uses `fold_in(fold_in(key, b), k)`; the draw is deterministic per `key`.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from hsmmkit.engine.batch import (
    _dataset_view,
    _expanded_forward,
    _expanded_inputs,
    _step_log_a,
)
from hsmmkit.engine.explicit_duration import _build_inputs, _forward_end, _terminal
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

__all__ = ["ffbs_expanded", "ffbs_explicit"]

_NEG_INF = -jnp.inf


def _sample_keys(key: Array, n_seq: int, n_samples: int) -> Array:
    """`(B, n)` keys: `fold_in(fold_in(key, b), k)`."""

    def per_seq(b: Array) -> Array:
        kb = jax.random.fold_in(key, b)
        return jax.vmap(lambda k: jax.random.fold_in(kb, k))(jnp.arange(n_samples))

    return jax.vmap(per_seq)(jnp.arange(n_seq))


def _expanded_one(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    data: Any,  # noqa: ANN401
    valid: Array,
    code: Array,
    keys: Array,
    active: bool,
) -> Array:
    """`(n, T)` expanded-state paths of one padded sequence (padding not yet masked)."""
    alpha_0, log_eye, xs, _ = _expanded_inputs(theta, spec, space, data, valid, code)
    log_alpha = _expanded_forward(theta, space, alpha_0, log_eye, xs, active)  # (T, S)
    t_len = log_alpha.shape[0]
    last = jax.vmap(lambda k: jax.random.categorical(jax.random.fold_in(k, t_len), log_alpha[-1]))(keys)

    def step(s_next: Array, x: tuple[Any, Array, Array]) -> tuple[Array, Array]:
        elems, alpha_prev, t = x
        log_a = _step_log_a(theta, space, elems, log_eye, active)  # A_t, built once for all samples
        logits = alpha_prev[:, None] + log_a[:, s_next]  # (S, n)
        s_prev = jax.vmap(lambda k, lg: jax.random.categorical(jax.random.fold_in(k, t), lg))(keys, logits.T)
        return s_prev.astype(s_next.dtype), s_prev

    steps = jnp.arange(1, t_len)
    _, earlier = jax.lax.scan(step, last, (xs, log_alpha[:-1], steps), reverse=True)  # (T-1, n): s_0..s_{T-2}
    path = jnp.concatenate([earlier, last[None, :]], axis=0)  # (T, n)
    return path.T.astype(jnp.int32)


@partial(jax.jit, static_argnames=("spec", "n_samples"))
def _ffbs_expanded(
    key: Array, theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch, n_samples: int
) -> Array:
    data = _dataset_view(pb)
    grouped = "initial.logits" in theta and theta["initial.logits"].ndim == 2
    active = pb.boundary_code is not None
    codes = jnp.zeros(pb.valid.shape, dtype=jnp.int8) if pb.boundary_code is None else pb.boundary_code
    keys = _sample_keys(key, pb.valid.shape[0], n_samples)

    def one(d: Any, v: Array, g: Array, c: Array, k: Array) -> Array:  # noqa: ANN401
        th = {**theta, "initial.logits": theta["initial.logits"][g]} if grouped else theta
        return _expanded_one(th, spec, space, d, v, c, k, active)

    paths = jax.vmap(one)(data, pb.valid, pb.group_index, codes, keys)  # (B, n, T)
    paths = jnp.where(pb.valid[:, None, :], paths, -1)
    return jnp.transpose(paths, (1, 0, 2))


def ffbs_expanded(
    key: Array, params: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch, n_samples: int
) -> Array:
    """`(n_samples, B, T_max)` exact posterior paths over the expanded states (`-1` on padding).

    Raises:
        ConfigError: the explicit engine (use `ffbs_explicit`) or `n_samples < 1`.
    """
    if spec.engine != "expanded":
        raise ConfigError(
            "ffbs_expanded needs engine: expanded (the explicit engine: ffbs_explicit)", path="model.engine"
        )
    if n_samples < 1:
        raise ConfigError(f"n_samples must be >= 1, got {n_samples}", path="samples")
    out: Array = _ffbs_expanded(key, params, spec, space, pb, n_samples)
    return out


def _explicit_one(
    theta: Params,
    spec: ModelSpec,
    space: StateSpace,
    data: Any,  # noqa: ANN401
    length: Array,
    code: Array | None,
    keys: Array,
) -> Array:
    """`(n, T)` regime paths of one padded sequence of true length `length >= 1` (padding not yet masked)."""
    inp = _build_inputs(theta, spec, space, data, length, code)
    inner, g, _ = _forward_end(inp)
    term = _terminal(inp)
    n_reg, d_max = spec.n_states, spec.d_max
    t_len = inp.cum.shape[0] - 1
    lp, log_pi, cum = inp.log_pmf, inp.log_pi, inp.cum
    final = (inner + term).reshape(-1)  # flat (s, j)
    durations = jnp.arange(1, d_max + 1)
    regimes = jnp.arange(n_reg)

    def one(key: Array) -> Array:
        idx = jax.random.categorical(jax.random.fold_in(key, t_len + 1), final)
        s0, j0 = idx // n_reg, idx % n_reg
        is_start = jnp.zeros((t_len,), dtype=bool).at[s0].set(True)
        reg = jnp.zeros((t_len,), dtype=jnp.int32).at[s0].set(j0.astype(jnp.int32))

        def cond(c: tuple[Array, ...]) -> Array:
            return c[0] > 0

        def body(c: tuple[Array, ...]) -> tuple[Array, ...]:
            tau, j, starts, rg, it = c
            sp = tau - durations  # (D,) candidate starts of the previous segment
            ok = sp >= 0
            spc = jnp.maximum(sp, 0)
            if lp.ndim == 3:
                lp_d = lp[spc[:, None], regimes[None, :], (durations - 1)[:, None]]  # (D, N)
            else:
                lp_d = lp[:, durations - 1].T
            pi_tau = log_pi[tau] if log_pi.ndim == 3 else log_pi
            logits = g[spc] + lp_d + cum[tau][None, :] + pi_tau[:, j][None, :]
            logits = jnp.where(ok[:, None], logits, _NEG_INF).reshape(-1)
            pick = jax.random.categorical(jax.random.fold_in(key, it), logits)
            d_pick, i_pick = pick // n_reg + 1, pick % n_reg
            start = tau - d_pick
            return (
                start,
                i_pick,
                starts.at[start].set(True),
                rg.at[start].set(i_pick.astype(jnp.int32)),
                it + 1,
            )

        init: Any = (s0, j0, is_start, reg, jnp.zeros((), jnp.int32))
        _, _, is_start, reg, _ = jax.lax.while_loop(cond, body, init)
        steps = jnp.arange(t_len)
        last_start = jax.lax.cummax(jnp.where(is_start, steps, -1), axis=0)
        path: Array = reg[jnp.maximum(last_start, 0)]
        return path

    return jax.vmap(one)(keys)


@partial(jax.jit, static_argnames=("spec", "n_samples"))
def _ffbs_explicit(
    key: Array, theta: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch, n_samples: int
) -> Array:
    data = _dataset_view(pb)
    grouped = "initial.logits" in theta and theta["initial.logits"].ndim == 2
    active = pb.boundary_code is not None
    codes = jnp.zeros(pb.valid.shape, dtype=jnp.int8) if pb.boundary_code is None else pb.boundary_code
    keys = _sample_keys(key, pb.valid.shape[0], n_samples)

    def one(d: Any, n_b: Array, g: Array, c: Array, k: Array) -> Array:  # noqa: ANN401
        th = {**theta, "initial.logits": theta["initial.logits"][g]} if grouped else theta
        return _explicit_one(th, spec, space, d, jnp.maximum(n_b, 1), c if active else None, k)

    paths = jax.vmap(one)(data, pb.lengths, pb.group_index, codes, keys)  # (B, n, T)
    paths = jnp.where(pb.valid[:, None, :], paths, -1)
    return jnp.transpose(paths, (1, 0, 2))


def ffbs_explicit(
    key: Array, params: Params, spec: ModelSpec, space: StateSpace, pb: PaddedBatch, n_samples: int
) -> Array:
    """`(n_samples, B, T_max)` exact posterior regime paths of the explicit engine (`-1` on padding).

    Segment draws are expanded to steps. Duration-dependent jumps run on the expanded `(regime, elapsed)`
    chain: use `ffbs_expanded` with `dd_space` there.

    Raises:
        ConfigError: the expanded engine, duration-dependent jumps, or `n_samples < 1`.
    """
    if spec.engine != "explicit":
        raise ConfigError(
            "ffbs_explicit needs engine: explicit (the expanded engine: ffbs_expanded)", path="model.engine"
        )
    if spec.duration_dependent_jumps:
        raise ConfigError(
            "ffbs_explicit does not support duration-dependent jumps",
            path="model.duration.duration_dependent_jumps",
        )
    if n_samples < 1:
        raise ConfigError(f"n_samples must be >= 1, got {n_samples}", path="samples")
    out: Array = _ffbs_explicit(key, params, spec, space, pb, n_samples)
    return out
