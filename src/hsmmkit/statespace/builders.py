"""`build_expanded(spec) -> StateSpace`: enumerate `(memory, regime, phase)` (E4-T02a).

Concept refs: §4.5, §4.7.1, Appendix B (`S` formula).
"""

from __future__ import annotations

import dataclasses
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.model.params import Params
from hsmmkit.model.transition import MemoryLayout, build_memory_layout
from hsmmkit.statespace.index import StateIndex, StateSpace
from hsmmkit.statespace.transitions import make_transition_fn
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.model.spec import ModelSpec


def _shift(memory: tuple[int, ...], current: int, order: int) -> tuple[int, ...]:
    """The new memory tuple (length `order - 1`) once `current` has just been entered."""
    if order == 1:
        return ()
    return (*memory[1:], current)


def build_expanded(spec: ModelSpec) -> StateSpace:
    """Enumerate the expanded state space `(memory, regime, phase)` from a `ModelSpec`.

    Memoized per `spec`: `StateSpace` carries its `transition_fn` closure in the pytree metadata, which jit
    compares by identity, so two builds of the same spec would otherwise compile every jitted function twice
    (a walk-forward run refits once per fold).
    """
    space = _build_expanded_cached(spec)
    if spec.order > 2:  # outside the cache: every build of such a model warns
        warnings.warn(
            f"transition.order={spec.order} > 2 expands the state space to S={space.n_expanded} states",
            stacklevel=2,
        )
    return space


@lru_cache(maxsize=16)
def _build_expanded_cached(spec: ModelSpec) -> StateSpace:
    """`build_expanded` without the order warning (memoized per `spec`)."""
    if spec.memory_mode == "steps":
        return _build_steps(spec)
    n_states = spec.n_states
    order = spec.order
    layout = build_memory_layout(n_states, order)
    r_rows = layout.tuples.shape[0]
    memories = layout.tuples[:, :-1]  # (R, order-1)

    regime_of: list[int] = []
    phase_of: list[int] = []
    memory_of: list[np.ndarray] = []
    row_of_state: list[int] = []
    is_last_phase: list[bool] = []
    next_phase_state: list[int] = []
    row_start = np.empty(r_rows, dtype=np.int64)

    s = 0
    for row_idx in range(r_rows):
        current = int(layout.current_of[row_idx])
        row_start[row_idx] = s
        n_phases = spec.duration.n_phases(current)
        for phase in range(1, n_phases + 1):
            regime_of.append(current)
            phase_of.append(phase)
            memory_of.append(memories[row_idx])
            row_of_state.append(row_idx)
            last = phase == n_phases
            is_last_phase.append(last)
            next_phase_state.append(-1 if last else s + 1)
            s += 1

    n_expanded = s

    regime_of_arr = np.asarray(regime_of, dtype=np.int64)
    phase_of_arr = np.asarray(phase_of, dtype=np.int64)
    memory_of_arr = np.stack(memory_of, axis=0) if order > 1 else np.zeros((n_expanded, 0), dtype=np.int64)
    row_of_state_arr = np.asarray(row_of_state, dtype=np.int64)
    collapse = np.eye(n_states, dtype=np.float64)[regime_of_arr]  # (S, N)

    index = StateIndex(
        regime_of=jnp.asarray(regime_of_arr),
        phase_of=jnp.asarray(phase_of_arr),
        memory_of=jnp.asarray(memory_of_arr),
        collapse=jnp.asarray(collapse),
        row_of_state=jnp.asarray(row_of_state_arr),
        n_expanded=n_expanded,
        n_states=n_states,
        order=order,
    )

    entry_state_of = _build_entry_state_of(layout, memories, row_start, n_states, order)
    n_rates, rate_index_of = _build_rate_index(
        spec.duration_conditioning, n_states, order, regime_of_arr, memory_of_arr
    )
    phase1_idx = np.flatnonzero(phase_of_arr == 1)
    exit_targets = entry_state_of[row_of_state_arr, :]  # (S, N), gathered once

    space = StateSpace(
        index=index,
        layout=layout,
        n_expanded=n_expanded,
        n_rates=n_rates,
        rate_index_of=jnp.asarray(rate_index_of),
        entry_state_of=jnp.asarray(entry_state_of),
        phase1_idx=jnp.asarray(phase1_idx),
        next_phase_state=jnp.asarray(next_phase_state),
        is_last_phase=jnp.asarray(is_last_phase),
        exit_targets=jnp.asarray(exit_targets),
    )
    return dataclasses.replace(space, transition_fn=make_transition_fn(spec, space))


def _build_steps(spec: ModelSpec) -> StateSpace:
    """Time-step memory state space (`memory_mode: steps`, E11-T05): the tuples `(s_{t-p}, ..., s_t)`.

    All `N^(p+1)` tuples with repeats allowed, the oldest regime most significant; `p` is
    `spec.emission.memory_steps`. Durations are geometric (`r = 1`, checked at model build), so a state is
    exactly a tuple (`phase == 1`). `exit_targets[s, j]` is the tuple `(s_{t-p+1}, ..., s_t, j)` that follows
    `s` when the next regime is `j` (including `j == s_t`, the stay), which is what the transition function
    scatters into. Hamilton's mean-adjusted MS-AR is the model that needs this memory.
    """
    n_states = spec.n_states
    p = int(spec.emission.memory_steps)  # type: ignore[attr-defined]
    if p < 1:
        raise ConfigError(
            "memory_mode='steps' needs an emission with memory_steps >= 1", path="model.emission"
        )
    if spec.order != 1 or spec.duration_conditioning != "current":
        raise ConfigError(
            "memory_mode='steps' requires transition.order == 1 and duration_conditioning='current'",
            path="model.transition.order",
        )
    if any(spec.duration.n_phases(i) != 1 for i in range(n_states)):
        raise ConfigError(
            "memory_mode='steps' requires geometric durations (r = 1)", path="model.duration.family"
        )
    n_expanded = n_states ** (p + 1)
    ids = np.arange(n_expanded, dtype=np.int64)
    tuples = np.stack([(ids // n_states ** (p - k)) % n_states for k in range(p + 1)], axis=1)  # (S, p+1)
    regime_of = tuples[:, p]
    layout = build_memory_layout(n_states, 1)
    index = StateIndex(
        regime_of=jnp.asarray(regime_of),
        phase_of=jnp.ones(n_expanded, dtype=jnp.int64),
        memory_of=jnp.asarray(tuples[:, :p]),
        collapse=jnp.asarray(np.eye(n_states, dtype=np.float64)[regime_of]),
        row_of_state=jnp.asarray(regime_of),
        n_expanded=n_expanded,
        n_states=n_states,
        order=p + 1,
    )
    shifted = (ids % n_states**p) * n_states  # drop the oldest regime, make room for the new one
    exit_targets = shifted[:, None] + np.arange(n_states)[None, :]  # (S, N)
    space = StateSpace(
        index=index,
        layout=layout,
        n_expanded=n_expanded,
        n_rates=n_states,
        rate_index_of=jnp.asarray(regime_of),
        entry_state_of=jnp.full((n_states, n_states), -1),
        phase1_idx=jnp.asarray(ids),
        next_phase_state=jnp.full(n_expanded, -1),
        is_last_phase=jnp.ones(n_expanded, dtype=bool),
        exit_targets=jnp.asarray(exit_targets),
    )
    return dataclasses.replace(space, transition_fn=make_transition_fn(spec, space))


def _build_entry_state_of(
    layout: MemoryLayout, memories: np.ndarray, row_start: np.ndarray, n_states: int, order: int
) -> np.ndarray:
    """`(R, N)`: the `phase == 1` state entered by `target` from each row (`-1` if `target == current`)."""
    row_lookup = {tuple(row): idx for idx, row in enumerate(layout.tuples)}
    r_rows = layout.tuples.shape[0]
    entry_state_of = np.full((r_rows, n_states), -1, dtype=np.int64)
    for row_idx in range(r_rows):
        current = int(layout.current_of[row_idx])
        memory_r = tuple(int(m) for m in memories[row_idx])
        shifted = _shift(memory_r, current, order)
        for target in range(n_states):
            if target == current:
                continue
            entry_row = row_lookup[(*shifted, target)]
            entry_state_of[row_idx, target] = row_start[entry_row]
    return entry_state_of


def _build_rate_index(
    duration_conditioning: str, n_states: int, order: int, regime_of: np.ndarray, memory_of: np.ndarray
) -> tuple[int, np.ndarray]:
    """`(n_rates, rate_index_of (S,))` for `duration_conditioning in {"current", "pair"}`."""
    if duration_conditioning == "current":
        return n_states, regime_of.copy()
    if order < 2:
        raise ConfigError(
            "duration_conditioning='pair' requires transition.order >= 2",
            path="model.transition.duration_conditioning",
        )
    pairs = [(i, j) for i in range(n_states) for j in range(n_states) if i != j]
    pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
    rate_index_of = np.asarray(
        [pair_to_idx[(int(memory_of[s, -1]), int(regime_of[s]))] for s in range(regime_of.shape[0])],
        dtype=np.int64,
    )
    return len(pairs), rate_index_of


def _stationary_distribution(a_bar: Array, n_iter: int = 500) -> Array:
    """Power iteration for `A_bar`'s stationary row vector (`pi @ A_bar == pi`).

    Preferred over `jnp.linalg.eig` here: the eigenvector for a general (non-symmetric) matrix is
    only defined up to an arbitrary sign, so naively `clip(..., min=0)`-ing it zeroes the whole
    vector (and gives `0/0 = NaN` on normalization) whenever the sign happens to come back
    negative -- roughly half the time in practice. `jax.grad` through non-symmetric eigenvectors
    is also unsupported by JAX. Power iteration has neither problem and is fully differentiable.
    """
    s = a_bar.shape[0]

    def body(vec: Array, _: None) -> tuple[Array, None]:
        vec = vec @ a_bar
        return vec / jnp.sum(vec), None

    vec, _ = jax.lax.scan(body, jnp.full((s,), 1.0 / s), xs=None, length=n_iter)
    return vec


def initial_log_rho(spec: ModelSpec, space: StateSpace, theta: Params, nominal_dt: float = 1.0) -> Array:
    """`(S,)` initial log-probabilities.

    `uniform`/`free` are `-inf` off the `phase == 1` support. `stationary` is the stationary
    distribution of the mean transition matrix `A_bar = transition_fn(theta, z=0, nominal_dt)`
    (restricted to `z = 0`/no-covariate semantics, per the ticket) and spans every state, not just
    `phase == 1` (the chain starts "already running", not freshly entering a regime).
    """
    s = space.n_expanded
    phase1_idx = space.phase1_idx
    n_phase1 = phase1_idx.shape[0]
    if spec.initial == "uniform":
        log_p = jnp.full((s,), -jnp.inf)
        return log_p.at[phase1_idx].set(-jnp.log(n_phase1))
    if spec.initial == "free":
        probs = theta["initial.logits"]  # (S_phase1,), already the constrained simplex values
        log_p = jnp.full((s,), -jnp.inf)
        return log_p.at[phase1_idx].set(jnp.log(probs))
    if spec.initial == "stationary":
        # "z = 0" semantics: a zero-valued covariate vector of whatever width theta implies (not
        # `None`, which the covariate-logit path treats as "no covariates configured" and errors
        # on a shape mismatch if `transition.b`/`duration.c` are actually present).
        z_pi0 = jnp.zeros((theta["transition.b"].shape[-1],)) if "transition.b" in theta else None
        z_d0 = jnp.zeros((theta["duration.c"].shape[-1],)) if "duration.c" in theta else None
        a_bar, _ = space.transition_fn(theta, z_pi0, z_d0, jnp.asarray(nominal_dt))
        return jnp.log(_stationary_distribution(a_bar))
    raise ConfigError(f"unknown initial distribution {spec.initial!r}", path="model.initial_state")


@lru_cache(maxsize=16)
def build_explicit(spec: ModelSpec) -> StateSpace:
    """Augmented `(regime, elapsed)` state space of the explicit-duration engine (E4-T06a).

    State `s = j * D_max + (e - 1)` is regime `j` with `e` steps elapsed in the current segment
    (`phase_of` holds `e`, 1-based; `elapsed_posterior` therefore buckets to `(T, N, D_max)`).
    Only order-1 memory with `duration_conditioning="current"` exists for this engine. `next_phase_state`,
    `is_last_phase` and `entry_state_of` describe the elapsed-time chain (`elapsed + 1`; entering a new
    regime resets `e = 1`), but no `transition_fn` is attached: the engine works on segments, never on `A_t`.
    """
    if spec.order != 1:
        raise ConfigError(
            "the explicit-duration engine requires transition.order == 1", path="model.transition.order"
        )
    if spec.duration_conditioning != "current":
        raise ConfigError(
            "the explicit-duration engine requires duration_conditioning='current'",
            path="model.transition.duration_conditioning",
        )
    n_states, d_max = spec.n_states, spec.d_max
    layout = build_memory_layout(n_states, 1)
    n_expanded = n_states * d_max
    regime_of = np.repeat(np.arange(n_states, dtype=np.int64), d_max)
    phase_of = np.tile(np.arange(1, d_max + 1, dtype=np.int64), n_states)
    index = StateIndex(
        regime_of=jnp.asarray(regime_of),
        phase_of=jnp.asarray(phase_of),
        memory_of=jnp.zeros((n_expanded, 0), dtype=jnp.int64),
        collapse=jnp.asarray(np.eye(n_states, dtype=np.float64)[regime_of]),
        row_of_state=jnp.asarray(regime_of),
        n_expanded=n_expanded,
        n_states=n_states,
        order=1,
    )
    entry_state_of = np.arange(n_states, dtype=np.int64)[None, :] * d_max + np.zeros((n_states, 1), np.int64)
    entry_state_of[np.arange(n_states), np.arange(n_states)] = -1
    is_last = phase_of == d_max
    return StateSpace(
        index=index,
        layout=layout,
        n_expanded=n_expanded,
        n_rates=n_states,
        rate_index_of=jnp.asarray(regime_of),
        entry_state_of=jnp.asarray(entry_state_of),
        phase1_idx=jnp.asarray(np.flatnonzero(phase_of == 1)),
        next_phase_state=jnp.asarray(np.where(is_last, -1, np.arange(n_expanded) + 1)),
        is_last_phase=jnp.asarray(is_last),
        exit_targets=jnp.asarray(entry_state_of[regime_of, :]),
    )
