"""Brute-force expansion of an order-`k`, NB-phase HSMM into a first-order HMM.

No `hsmmkit` import (CLAUDE.md §6). Enumerates expanded states `(memory, regime, phase)`
independently of `hsmmkit.statespace.builders`, so it can test that module's `build_expanded`/
`transition_fn` (E4-T02a/b) without circularity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def enumerate_memory_tuples(n_states: int, order: int) -> list[tuple[int, ...]]:
    """Every length-`order` sequence of regimes with no two adjacent entries equal."""
    tuples: list[tuple[int, ...]] = [(i,) for i in range(n_states)]
    for _ in range(order - 1):
        tuples = [(*t, j) for t in tuples for j in range(n_states) if j != t[-1]]
    return tuples


def expand_states(n_states: int, order: int, r: Sequence[int]) -> list[tuple[tuple[int, ...], int, int]]:
    """`(memory, regime, phase)` for every expanded state, lexicographic (memory-tuple, then phase)."""
    states = []
    for tup in enumerate_memory_tuples(n_states, order):
        current = tup[-1]
        memory = tup[:-1]
        for phase in range(1, r[current] + 1):
            states.append((memory, current, phase))
    return states


def _shift(memory: tuple[int, ...], current: int, order: int) -> tuple[int, ...]:
    if order == 1:
        return ()
    return (*memory[1:], current)


def build_expanded_a(
    n_states: int,
    order: int,
    r: Sequence[int],
    rate: np.ndarray,
    pi_rows: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, list[tuple[tuple[int, ...], int, int]]]:
    """`(A (S, S), states)`: exact stay/advance/exit construction (concept §4.7.2), no covariates.

    `rate (N,)` is the per-regime exit rate; `a_i(t) = 1 - exp(-rate_i * dt)`. `pi_rows` is
    `(R, N-1)`, one row per admissible memory tuple (order `enumerate_memory_tuples` order),
    giving the jump probabilities to every target `j != current`, ascending `j`.
    """
    tuples = enumerate_memory_tuples(n_states, order)
    row_of_tuple = {t: idx for idx, t in enumerate(tuples)}
    states = expand_states(n_states, order, r)
    s_len = len(states)
    index_of_state = {state: idx for idx, state in enumerate(states)}
    a_of_regime = 1.0 - np.exp(-rate * dt)

    def targets_of(current: int) -> list[int]:
        return [j for j in range(n_states) if j != current]

    a_mat = np.zeros((s_len, s_len))
    for s_idx, (memory, current, phase) in enumerate(states):
        a_i = a_of_regime[current]
        if phase < r[current]:
            a_mat[s_idx, s_idx] = 1.0 - a_i
            a_mat[s_idx, index_of_state[(memory, current, phase + 1)]] = a_i
            continue
        a_mat[s_idx, s_idx] = 1.0 - a_i
        row = row_of_tuple[(*memory, current)]
        shifted = _shift(memory, current, order)
        for k, target in enumerate(targets_of(current)):
            entry_state = (shifted, target, 1)
            a_mat[s_idx, index_of_state[entry_state]] += a_i * pi_rows[row, k]
    return a_mat, states


def collapse_matrix(states: list[tuple[tuple[int, ...], int, int]], n_states: int) -> np.ndarray:
    """`(S, N)` one-hot: `C[s, regime_of[s]] = 1`."""
    c = np.zeros((len(states), n_states))
    for s_idx, (_memory, current, _phase) in enumerate(states):
        c[s_idx, current] = 1.0
    return c
