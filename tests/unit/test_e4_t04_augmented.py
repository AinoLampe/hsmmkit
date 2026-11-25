"""E4-T04: phase/memory marginals, expected remaining/elapsed steps, jittability."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.engine.augmented import (
    collapse_to_regimes,
    expected_elapsed_steps,
    expected_remaining_steps,
    memory_marginals,
    phase_marginals,
)
from regimekit.engine.base import Posterior
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from tests.helpers.engine_fixtures import make_dataset, make_scenario

ENGINE = ExpandedHMMEngine()


@pytest.mark.parametrize(("order", "r"), [(1, [2, 3, 1]), (2, [2, 3, 1]), (2, [3, 3, 3])])
def test_e4_t04_phase_marginals_sum_to_regime_marginals_and_memory_rows_sum_to_one(
    order: int, r: list[int]
) -> None:
    """AC1: phases sum to the regime marginals (1e-10), zero-padded beyond `r_i`; memory rows sum to 1."""
    sc = make_scenario(3, r, order, 40, seed=order)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)

    phases = np.asarray(phase_marginals(post))
    assert phases.shape == (40, 3, max(r))
    np.testing.assert_allclose(phases.sum(-1), np.asarray(post.regime_smoothed()), atol=1e-10)
    for i, ri in enumerate(r):
        assert np.all(phases[:, i, ri:] == 0.0)

    mem = np.asarray(memory_marginals(post))
    expected_rows = 1 if order == 1 else sc.space.layout.tuples.shape[0]
    assert mem.shape == (40, expected_rows)
    np.testing.assert_allclose(mem.sum(-1), 1.0, atol=1e-10)


def test_e4_t04_memory_marginals_match_a_direct_row_sum() -> None:
    """Order 2: `memory_marginals[t, k]` is the smoothed mass of the states whose row is `k`."""
    sc = make_scenario(3, [2, 1, 2], 2, 25, seed=3)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    rows = np.asarray(sc.space.index.row_of_state)
    sm = np.asarray(post.smoothed)
    expected = np.stack([sm[:, rows == k].sum(-1) for k in range(sc.space.layout.tuples.shape[0])], axis=1)
    np.testing.assert_allclose(np.asarray(memory_marginals(post)), expected, atol=1e-12)


def test_e4_t04_elapsed_posterior_method_delegates_and_carries_engine_tag() -> None:
    """`Posterior.engine` defaults to "expanded"; `elapsed_posterior()` equals `phase_marginals`."""
    sc = make_scenario(2, [2, 3], 1, 20, seed=1)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    assert post.engine == "expanded"
    np.testing.assert_array_equal(np.asarray(post.elapsed_posterior()), np.asarray(phase_marginals(post)))
    flat, tree = jax.tree_util.tree_flatten(post)
    assert jax.tree_util.tree_unflatten(tree, flat).engine == "expanded"


def _simulate_phase_chain(
    rng: np.random.Generator, r: list[int], a: np.ndarray, n_segments: int
) -> tuple[np.ndarray, np.ndarray]:
    """Test-side NumPy phase chain over 2 regimes that alternate; returns (regime path, entry steps)."""
    regimes: list[int] = []
    entries: list[int] = []
    regime = 0
    for _ in range(n_segments):
        entries.append(len(regimes))
        for _phase in range(r[regime]):
            length = int(rng.geometric(a[regime]))
            regimes.extend([regime] * length)
        regime = 1 - regime
    return np.asarray(regimes), np.asarray(entries)


def test_e4_t04_expected_remaining_steps_at_entry_matches_analytic_mean() -> None:
    """AC2: mean of `expected_remaining_steps` at regime-entry steps within 15 % of `r_i / a_i`.

    Data come from a NumPy phase-chain simulation with well-separated emissions, so the smoothed
    posterior identifies each entry step; the engine runs with the true parameters.
    """
    rng = np.random.default_rng(0)
    r = [2, 3]
    rate = np.array([0.10, 0.15])  # per step (dt = 1 in this test, so a = 1 - exp(-rate))
    a = 1.0 - np.exp(-rate)
    regimes, entries = _simulate_phase_chain(rng, r, a, 2000)

    sc = make_scenario(2, r, 1, 10, seed=0)
    mu, sigma = np.array([0.1, -0.1]), np.array([0.01, 0.01])
    dt = np.ones(len(regimes))
    y = (mu[regimes] - 0.5 * sigma[regimes] ** 2) + sigma[regimes] * rng.normal(size=len(regimes))
    data = make_dataset(y[:, None], dt)
    theta = dict(sc.theta)
    theta["emission.mean"] = jnp.asarray(mu)
    theta["emission.scale"] = jnp.asarray(sigma)
    theta["duration.rate"] = jnp.asarray(rate)
    theta["transition.pi_rows"] = jnp.ones((2, 1))

    post = ENGINE.forward_backward(theta, sc.spec, sc.space, data)
    remaining = np.asarray(expected_remaining_steps(post, sc.spec, theta, data.dt))
    assert remaining.shape == (len(regimes), 2)

    entries = entries[1:]  # the first segment starts before any transition is observed
    for regime in (0, 1):
        at = entries[regimes[entries] == regime]
        assert len(at) >= 900
        analytic = r[regime] / a[regime]
        got = remaining[at, regime].mean()
        assert abs(got - analytic) <= 0.15 * analytic, (regime, got, analytic)


def test_e4_t04_expected_elapsed_steps_is_the_phase_based_lower_bound() -> None:
    """Hand-built posterior: phase c contributes `(c - 1) / a_i`, conditional on the regime."""
    sc = make_scenario(2, [3, 2], 1, 4, seed=2)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    a = np.asarray(sc.spec.duration.advance_prob(sc.theta, sc.data.dt, None))  # (T, N)
    phases = np.asarray(phase_marginals(post))
    reg = phases.sum(-1)
    manual_elapsed = np.zeros_like(reg)
    manual_remaining = np.zeros_like(reg)
    for i, ri in enumerate([3, 2]):
        for c in range(1, ri + 1):
            manual_elapsed[:, i] += phases[:, i, c - 1] * (c - 1) / a[:, i]
            manual_remaining[:, i] += phases[:, i, c - 1] * (ri - c + 1) / a[:, i]
    np.testing.assert_allclose(
        np.asarray(expected_elapsed_steps(post, sc.spec, sc.theta, sc.data.dt)),
        manual_elapsed / reg,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(expected_remaining_steps(post, sc.spec, sc.theta, sc.data.dt)),
        manual_remaining / reg,
        rtol=1e-10,
    )


def test_e4_t04_zero_regime_mass_gives_zero_not_nan() -> None:
    """A regime with no posterior mass yields 0 (not NaN) from the conditional expectations."""
    sc = make_scenario(2, [2, 2], 1, 6, seed=0)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    smoothed = post.smoothed * jnp.asarray(sc.space.index.regime_of == 0)[None, :]
    post0 = type(post)(
        log_alpha=post.log_alpha,
        log_beta=post.log_beta,
        smoothed=smoothed / smoothed.sum(-1, keepdims=True),
        loglik=post.loglik,
        index=post.index,
    )
    out = np.asarray(expected_remaining_steps(post0, sc.spec, sc.theta, sc.data.dt))
    assert np.all(np.isfinite(out))
    assert np.all(out[:, 1] == 0.0)


def test_e4_t04_functions_are_jittable_and_collapse_helper_matches_matmul() -> None:
    """AC3: all utilities trace under `jax.jit`; `collapse_to_regimes(x, C) == x @ C`."""
    sc = make_scenario(3, [2, 1, 3], 2, 15, seed=5)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)

    @jax.jit
    def everything(p: Posterior, theta: dict[str, jax.Array], dt: jax.Array) -> tuple[jax.Array, ...]:
        return (
            phase_marginals(p),
            memory_marginals(p),
            expected_remaining_steps(p, sc.spec, theta, dt),
            expected_elapsed_steps(p, sc.spec, theta, dt),
            collapse_to_regimes(p.smoothed, p.index.collapse),
        )

    outs = everything(post, sc.theta, sc.data.dt)
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in outs)
    eager = (
        phase_marginals(post),
        memory_marginals(post),
        expected_remaining_steps(post, sc.spec, sc.theta, sc.data.dt),
        expected_elapsed_steps(post, sc.spec, sc.theta, sc.data.dt),
        collapse_to_regimes(post.smoothed, post.index.collapse),
    )
    for got, want in zip(outs, eager, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(outs[4]), np.asarray(post.regime_smoothed()), atol=1e-14)


def test_e4_t04_explicit_engine_tag_survives_pytree_round_trip() -> None:
    """The static `engine` tag is part of the pytree structure and round-trips."""
    sc = make_scenario(2, [2, 1], 1, 8, seed=1)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    explicit = dataclasses.replace(post, engine="explicit")
    flat, tree = jax.tree_util.tree_flatten(explicit)
    assert jax.tree_util.tree_unflatten(tree, flat).engine == "explicit"
    assert tree != jax.tree_util.tree_structure(post)
