"""E12-T03 property tests: structural zeros of `A_t` and jump probabilities for random topologies."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hsmmkit.model.params.manager import ParamManager
from hsmmkit.model.topology import Topology
from hsmmkit.model.transition import JumpTransition, _flat_specs, build_memory_layout


@st.composite
def topologies(draw: st.DrawFn) -> Topology:
    """A random `N x N` boolean jump matrix (`N` in 2..5, zero diagonal) with at least one allowed jump."""
    n = draw(st.integers(min_value=2, max_value=5))
    bits = draw(st.lists(st.booleans(), min_size=n * n, max_size=n * n))
    allowed = np.asarray(bits, dtype=bool).reshape(n, n) & ~np.eye(n, dtype=bool)
    if not allowed.any():
        allowed[0, 1] = True
    return Topology(allowed=allowed, absorbing=tuple(int(i) for i in np.flatnonzero(~allowed.any(axis=1))))


@given(topologies(), st.integers(min_value=1, max_value=2), st.integers(min_value=0, max_value=10**6))
def test_e12_t03_jump_probs_are_zero_off_topology_and_rows_sum_to_one(
    topo: Topology, order: int, seed: int
) -> None:
    """`pi` is exactly 0 outside `A_i`; rows with targets sum to 1 (<= 1e-12); absorbing rows are all 0."""
    layout = build_memory_layout(topo.n_states, order, topo)
    specs = _flat_specs(layout, 0, None)
    theta = {}
    if specs:
        mgr = ParamManager(tuple(specs))
        theta = dict(mgr.unflatten(jnp.asarray(np.random.default_rng(seed).normal(size=mgr.n_free) * 2)))
    pi = np.asarray(JumpTransition(topology=topo).jump_probs_t(theta, None, layout))
    allowed = np.asarray(layout.target_mask)
    assert np.all(pi[~allowed] == 0.0)
    has = allowed.any(axis=1)
    np.testing.assert_allclose(pi[has].sum(axis=1), 1.0, atol=1e-12)
    assert np.all(pi[~has] == 0.0)
    assert np.all(pi[allowed] > 0.0)
