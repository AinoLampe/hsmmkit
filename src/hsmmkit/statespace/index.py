"""`StateIndex`/`StateSpace`: the expanded state space's index arrays.

Stub signatures from E1-T05; built by `statespace/builders.py` in E4-T02a.

Index arrays are ordinary (non-static) `eqx.Module` fields, so they are JAX pytree **leaves** --
not `eqx.field(static=True)` metadata. `regime_of`/`phase_of`/`memory_of`/`collapse`/
`row_of_state` are plain integer/bool/float arrays (valid JAX dtypes), so treating them as leaves
avoids the "numpy arrays are unhashable" gotcha (CLAUDE.md §5) entirely, while still supporting
`jax.jit`'s shape-based caching (verified: no recompilation across distinct index-array values of
the same shape). Only `n_expanded`/`n_states`/`order` (plain ints) are `static=True`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.model.transition import MemoryLayout


def _as_int_array(x: object) -> Array:
    return jnp.asarray(x, dtype=jnp.int32)


def _as_float_array(x: object) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


def _as_bool_array(x: object) -> Array:
    return jnp.asarray(x, dtype=jnp.bool_)


class StateIndex(eqx.Module):
    """Static-content (but dynamically-registered) index arrays for the expanded state space."""

    regime_of: Array = eqx.field(converter=_as_int_array)  # (S,) int
    phase_of: Array = eqx.field(converter=_as_int_array)  # (S,) int, 1-based phase within regime
    memory_of: Array = eqx.field(converter=_as_int_array)  # (S, k-1) int; (S, 0) for order 1
    collapse: Array = eqx.field(converter=_as_float_array)  # C (S, N) 0/1, C[s, regime_of[s]] = 1
    row_of_state: Array = eqx.field(converter=_as_int_array)  # (S,) int: jump-matrix row per state
    n_expanded: int = eqx.field(static=True)
    n_states: int = eqx.field(static=True)
    order: int = eqx.field(static=True)


def _not_implemented_transition_fn(*_args: object, **_kwargs: object) -> tuple[Array, Array]:
    """Placeholder `A_t` builder; filled in by E4-T02b."""
    raise NotImplementedError("StateSpace.transition_fn is built by E4-T02b")


class StateSpace(eqx.Module):
    """`StateIndex` plus the `MemoryLayout` it was built from and duration-rate/entry bookkeeping.

    `layout` is closed over as static metadata (`MemoryLayout` has a content-based `__hash__`/
    `__eq__`, CLAUDE.md §5). `rate_index_of (S,)` maps each state to its row in the duration
    parameter array (`n_rates` rows: `N` for `duration_conditioning="current"`, `N(N-1)` for
    `"pair"`). `entry_state_of (R, N)` is the expanded state (`phase == 1`) entered when regime
    `target` starts from jump-matrix row `r` (`-1` where `target` equals that row's current
    regime). `phase1_idx (S_phase1,)` lists the `phase == 1` states, ascending.

    `next_phase_state (S,)` / `is_last_phase (S,)` / `exit_targets (S, N)` (E4-T02b) are the
    per-`A_t`-scatter arrays: `next_phase_state[s]` is `s`'s same-regime, phase-advanced state
    (`-1` at the last phase), `is_last_phase[s]` is `phase_of[s] == n_phases(regime_of[s])`, and
    `exit_targets[s, :] = entry_state_of[row_of_state[s], :]` (gathered once, not per step).
    """

    index: StateIndex
    layout: MemoryLayout = eqx.field(static=True)
    n_expanded: int = eqx.field(static=True)
    n_rates: int = eqx.field(static=True)
    rate_index_of: Array = eqx.field(converter=_as_int_array)
    entry_state_of: Array = eqx.field(converter=_as_int_array)
    phase1_idx: Array = eqx.field(converter=_as_int_array)
    next_phase_state: Array = eqx.field(converter=_as_int_array)
    is_last_phase: Array = eqx.field(converter=_as_bool_array)
    exit_targets: Array = eqx.field(converter=_as_int_array)
    transition_fn: Callable[..., tuple[Array, Array]] = eqx.field(
        static=True, default=_not_implemented_transition_fn
    )
