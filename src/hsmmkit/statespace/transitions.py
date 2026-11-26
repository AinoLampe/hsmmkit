"""Per-step expanded transition matrix `A_t` (E4-T02b): `StateSpace.transition_fn`.

`make_transition_fn(spec, space)` builds a closure over `spec`'s family instances and `space`'s
precomputed static index arrays (`next_phase_state`, `is_last_phase`, `exit_targets`,
`rate_index_of`), so the returned function's only traced arguments are `theta`, `z_pi_t`, `z_d_t`,
`dt_t` -- safe to call from inside a `jax.lax.scan` body without retracing per step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.model.params import Params
from hsmmkit.model.transition import transition_matrix_generator
from hsmmkit.utils.errors import DataError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

TransitionFn = Callable[[Params, "Array | None", "Array | None", Array], tuple[Array, Array]]


def _log_with_structural_zeros(a: Array) -> Array:
    """`log(A)`, `-inf` exactly at `A == 0` (double-`where`, CLAUDE.md §5 gradient safety)."""
    safe = jnp.where(a > 0, a, 1.0)
    return jnp.where(a > 0, jnp.log(safe), -jnp.inf)


def make_transition_fn(spec: ModelSpec, space: StateSpace) -> TransitionFn:
    """Build the per-step `(A_t, log A_t)` function for `spec`'s transition mode."""
    if spec.memory_mode == "steps":
        return _make_steps_transition_fn(spec, space)
    if spec.transition.mode == "generator":
        return _make_generator_transition_fn(spec)
    return _make_jump_transition_fn(spec, space)


def _make_generator_transition_fn(spec: ModelSpec) -> TransitionFn:
    def transition_fn(
        theta: Params, z_pi_t: Array | None, z_d_t: Array | None, dt_t: Array
    ) -> tuple[Array, Array]:
        """`A_t = expm(Q dt_t)` (HMM-only generator mode: geometric, `r=1`, order 1)."""
        del z_pi_t, z_d_t
        lam = theta["duration.rate"]
        a = transition_matrix_generator(theta, lam, jnp.reshape(dt_t, (1,)))[0]
        return a, _log_with_structural_zeros(a)

    del spec
    return transition_fn


def _absorbing_states(spec: ModelSpec, space: StateSpace) -> Array | None:
    """`(S,)` bool mask of the expanded states of absorbing regimes (E12-T03), or `None` without any."""
    topology = getattr(spec.transition, "topology", None)
    if topology is None or not topology.absorbing:
        return None
    return jnp.asarray(np.isin(np.asarray(space.index.regime_of), topology.absorbing))


def _make_jump_transition_fn(spec: ModelSpec, space: StateSpace) -> TransitionFn:
    s = space.n_expanded
    row_of_state = space.index.row_of_state
    next_phase_state = space.next_phase_state
    is_last_phase = space.is_last_phase
    exit_targets = space.exit_targets
    rate_index_of = space.rate_index_of
    n_rates = space.n_rates
    diag_idx = jnp.arange(s)
    absorb = _absorbing_states(spec, space)

    def transition_fn(
        theta: Params, z_pi_t: Array | None, z_d_t: Array | None, dt_t: Array
    ) -> tuple[Array, Array]:
        """`A_t` from `stay`/`advance`/`exit` scatter-adds over precomputed index arrays."""
        a_rate = spec.duration.advance_prob_t(theta, dt_t, z_d_t)  # (n_rates,)
        if a_rate.shape[0] != n_rates:
            raise DataError(
                f"theta['duration.rate'] has {a_rate.shape[0]} rows, expected n_rates={n_rates} "
                f"(duration_conditioning={spec.duration_conditioning!r})",
                path="duration.rate",
            )
        a_state = a_rate[rate_index_of]  # (S,)
        if absorb is not None:
            a_state = jnp.where(absorb, 0.0, a_state)  # absorbing regimes never advance or leave

        pi_rows = spec.transition.jump_probs_t(theta, z_pi_t, space.layout)  # (R, N)
        pi_state = pi_rows[row_of_state]  # (S, N)

        a = jnp.zeros((s, s))
        a = a.at[diag_idx, diag_idx].add(1.0 - a_state)  # stay

        valid_next = next_phase_state >= 0
        safe_next = jnp.where(valid_next, next_phase_state, 0)
        a = a.at[diag_idx, safe_next].add(jnp.where(valid_next, a_state, 0.0))  # advance

        valid_exit = (exit_targets >= 0) & is_last_phase[:, None]  # (S, N)
        safe_exit = jnp.where(valid_exit, exit_targets, 0)
        exit_contrib = jnp.where(valid_exit, a_state[:, None] * pi_state, 0.0)
        a = a.at[diag_idx[:, None], safe_exit].add(exit_contrib)  # exit

        return a, _log_with_structural_zeros(a)

    return transition_fn


def _make_steps_transition_fn(spec: ModelSpec, space: StateSpace) -> TransitionFn:
    """`A_t` of the time-step memory space: `(s_{t-p..t}) -> (s_{t-p+1..t}, j)`.

    Geometric durations: the chain stays in `s_t` with probability `1 - a_{s_t}` and jumps to `j != s_t`
    with probability `a_{s_t} pi_{s_t j}`; every row has exactly `N` possible successors.
    """
    s = space.n_expanded
    regime_of = space.index.regime_of
    exit_targets = space.exit_targets  # (S, N): the successor tuple for each next regime
    rows = jnp.arange(s)[:, None]
    absorb = _absorbing_states(spec, space)

    def transition_fn(
        theta: Params, z_pi_t: Array | None, z_d_t: Array | None, dt_t: Array
    ) -> tuple[Array, Array]:
        a_rate = spec.duration.advance_prob_t(theta, dt_t, z_d_t)  # (N,)
        if a_rate.shape[0] != space.n_rates:
            raise DataError(
                f"theta['duration.rate'] has {a_rate.shape[0]} rows, expected n_rates={space.n_rates}",
                path="duration.rate",
            )
        a_state = a_rate[regime_of]  # (S,)
        if absorb is not None:
            a_state = jnp.where(absorb, 0.0, a_state)
        pi_state = spec.transition.jump_probs_t(theta, z_pi_t, space.layout)[regime_of]  # (S, N), 0 on stay
        stay = jax.nn.one_hot(regime_of, space.index.n_states, dtype=pi_state.dtype)  # (S, N)
        probs = stay * (1.0 - a_state)[:, None] + a_state[:, None] * pi_state
        a = jnp.zeros((s, s)).at[rows, exit_targets].add(probs)
        return a, _log_with_structural_zeros(a)

    return transition_fn
