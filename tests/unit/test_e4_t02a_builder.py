"""E4-T02a: `build_expanded` -- S formula, memory-tuple validity, collapse, entry_state_of, pair rates."""

from __future__ import annotations

import numpy as np
import pytest

from regimekit.model.duration.geometric import GeometricDuration
from regimekit.model.duration.negbin import NegBinDuration
from regimekit.model.emission.gbm import GBMFamily
from regimekit.model.spec import ModelSpec
from regimekit.model.transition import JumpTransition
from regimekit.statespace.builders import build_expanded


def _spec(
    n_states: int,
    order: int,
    duration: object,
    *,
    duration_conditioning: str = "current",
    initial: str = "uniform",
) -> ModelSpec:
    return ModelSpec(
        n_states=n_states,
        n_series=1,
        emission=GBMFamily(),
        duration=duration,  # type: ignore[arg-type]
        transition=JumpTransition(),
        order=order,
        duration_conditioning=duration_conditioning,  # type: ignore[arg-type]
        engine="expanded",
        d_max=10,
        initial=initial,  # type: ignore[arg-type]
        param_specs=(),
    )


# --- independent NumPy reference enumeration (AC3), not the E4-T01-owned tests/reference module ---


def _ref_expand(n_states: int, order: int, r: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent enumeration: (memory tuple, regime, phase), lexicographic order."""

    def tuples_of(k: int) -> list[tuple[int, ...]]:
        out: list[tuple[int, ...]] = [(i,) for i in range(n_states)]
        for _ in range(k - 1):
            out = [(*t, j) for t in out for j in range(n_states) if j != t[-1]]
        return sorted(out)

    regime_of, phase_of, memory_of = [], [], []
    for t in tuples_of(order):
        current = t[-1]
        memory = t[:-1]
        for phase in range(1, r[current] + 1):
            regime_of.append(current)
            phase_of.append(phase)
            memory_of.append(memory)
    return np.asarray(regime_of), np.asarray(phase_of), np.asarray(memory_of)


# --- AC1: S formula ------------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("n_states", [2, 3, 4])
def test_e4_t02a_s_formula(n_states: int, order: int) -> None:
    """`S = sum(r_i)` (order 1) / `(N-1) * sum(r_i)` (order 2), for random `r_i in [1,3]`."""
    rng = np.random.default_rng(hash((n_states, order)) % (2**31))
    r = rng.integers(1, 4, size=n_states).tolist()
    duration = NegBinDuration(r=tuple(r))
    space = build_expanded(_spec(n_states, order, duration))

    expected = sum(r) if order == 1 else (n_states - 1) * sum(r)
    assert space.n_expanded == expected


# --- AC2: memory-tuple validity -------------------------------------------------------------------


def test_e4_t02a_order2_memory_never_equals_regime() -> None:
    """Order-2 memory tuples never have `memory[-1] == regime`."""
    duration = NegBinDuration(r=(2, 2, 3))
    space = build_expanded(_spec(3, 2, duration))
    memory_of = np.asarray(space.index.memory_of)
    regime_of = np.asarray(space.index.regime_of)
    assert not np.any(memory_of[:, -1] == regime_of)


def test_e4_t02a_order3_no_adjacent_repeats() -> None:
    """Order-3 tuples (memory + regime) have no adjacent repeats."""
    duration = GeometricDuration()
    space = build_expanded(_spec(3, 3, duration))
    memory_of = np.asarray(space.index.memory_of)
    regime_of = np.asarray(space.index.regime_of)
    full = np.concatenate([memory_of, regime_of[:, None]], axis=1)
    assert not np.any(full[:, :-1] == full[:, 1:])


# --- AC3: collapse one-hot; agreement with an independent reference enumeration -------------------


def test_e4_t02a_collapse_is_one_hot() -> None:
    """`C` rows are one-hot, `C[s, regime_of[s]] == 1`."""
    duration = NegBinDuration(r=(2, 3))
    space = build_expanded(_spec(2, 1, duration))
    collapse = np.asarray(space.index.collapse)
    regime_of = np.asarray(space.index.regime_of)
    assert np.all(collapse.sum(axis=1) == 1.0)
    assert np.all(collapse[np.arange(collapse.shape[0]), regime_of] == 1.0)


@pytest.mark.parametrize("order", [1, 2])
def test_e4_t02a_matches_reference_enumeration(order: int) -> None:
    """`regime_of`/`phase_of`/`memory_of` agree with an independent reference enumeration."""
    n_states, r = 3, [2, 1, 3]
    duration = NegBinDuration(r=tuple(r))
    space = build_expanded(_spec(n_states, order, duration))

    ref_regime, ref_phase, ref_memory = _ref_expand(n_states, order, r)
    assert space.n_expanded == ref_regime.shape[0]
    np.testing.assert_array_equal(np.asarray(space.index.regime_of), ref_regime)
    np.testing.assert_array_equal(np.asarray(space.index.phase_of), ref_phase)
    np.testing.assert_array_equal(np.asarray(space.index.memory_of), ref_memory)


# --- AC4: entry_state_of ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_e4_t02a_entry_state_of_points_to_phase1_correct_regime_and_memory(order: int) -> None:
    """`entry_state_of[r, j]` points to a `phase==1`, `regime==j` state with memory `shift(m_r, i_r)`."""
    n_states = 3
    duration = NegBinDuration(r=(2, 3, 1))
    space = build_expanded(_spec(n_states, order, duration))
    layout = space.layout
    memories = layout.tuples[:, :-1]
    entry = np.asarray(space.entry_state_of)
    regime_of = np.asarray(space.index.regime_of)
    phase_of = np.asarray(space.index.phase_of)
    memory_of = np.asarray(space.index.memory_of)

    for row_idx in range(layout.tuples.shape[0]):
        current = int(layout.current_of[row_idx])
        memory_r = tuple(int(m) for m in memories[row_idx])
        for target in range(n_states):
            s = entry[row_idx, target]
            if target == current:
                assert s == -1
                continue
            assert phase_of[s] == 1
            assert regime_of[s] == target
            expected_memory = (*memory_r[1:], current) if order > 1 else ()
            np.testing.assert_array_equal(memory_of[s], np.asarray(expected_memory))


# --- AC5: pair conditioning -------------------------------------------------------------------------


def test_e4_t02a_pair_conditioning_n_states_times_n_states_minus_1_distinct_values() -> None:
    """`pair` conditioning at order 2 yields `N(N-1)` distinct `rate_index_of` values."""
    n_states = 3
    duration = NegBinDuration(r=(2, 2, 2))
    space = build_expanded(_spec(n_states, 2, duration, duration_conditioning="pair"))
    assert space.n_rates == n_states * (n_states - 1)
    assert len(np.unique(np.asarray(space.rate_index_of))) == n_states * (n_states - 1)


def test_e4_t02a_pair_conditioning_requires_order_2() -> None:
    """`pair` conditioning at order 1 raises `ConfigError`."""
    from regimekit.utils.errors import ConfigError

    duration = GeometricDuration()
    with pytest.raises(ConfigError):
        build_expanded(_spec(3, 1, duration, duration_conditioning="pair"))


def test_e4_t02a_current_conditioning_rate_index_equals_regime() -> None:
    """`current` conditioning: `n_rates == N`, `rate_index_of == regime_of`."""
    duration = NegBinDuration(r=(2, 2, 2))
    space = build_expanded(_spec(3, 1, duration))
    assert space.n_rates == 3
    np.testing.assert_array_equal(np.asarray(space.rate_index_of), np.asarray(space.index.regime_of))
