"""E4-T03b: posterior normalization, alpha+beta invariant, state-order invariance, masks, guards."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import logsumexp

from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.statespace.index import StateSpace
from hsmmkit.utils.errors import NumericalError
from tests.golden.test_e4_t03a_forward import reference_full
from tests.golden.test_e4_t03b_forward_backward import to_probs
from tests.helpers.engine_fixtures import Scenario, make_dataset, make_scenario

ENGINE = ExpandedHMMEngine()


@given(
    n_states=st.integers(2, 4),
    order=st.integers(1, 2),
    t_len=st.integers(5, 120),
    seed=st.integers(0, 10_000),
    data=st.data(),
)
def test_e4_t03b_rows_normalized_and_alpha_beta_invariant(
    n_states: int, order: int, t_len: int, seed: int, data: st.DataObject
) -> None:
    """AC2: filtered/smoothed rows sum to 1 (1e-10); `logsumexp(alpha_t + beta_t)` == loglik for all t."""
    r = data.draw(st.lists(st.integers(1, 3), min_size=n_states, max_size=n_states))
    sc = make_scenario(n_states, r, order, t_len, n_cov_d=1, seed=seed)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)

    np.testing.assert_allclose(np.asarray(post.smoothed).sum(-1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(post.regime_smoothed()).sum(-1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(post.regime_filtered()).sum(-1), 1.0, atol=1e-10)

    per_t = logsumexp(np.asarray(post.log_alpha) + np.asarray(post.log_beta), axis=1)
    np.testing.assert_allclose(per_t, float(post.loglik), rtol=1e-8, atol=0)


def _permuted(sc: Scenario, perm: np.ndarray) -> StateSpace:
    """A `StateSpace` whose state `i` is the original state `perm[i]` (same model, other order)."""
    sp, idx = sc.space, sc.space.index
    p = jnp.asarray(perm)
    new_index = dataclasses.replace(
        idx,
        regime_of=idx.regime_of[p],
        phase_of=idx.phase_of[p],
        memory_of=idx.memory_of[p],
        collapse=idx.collapse[p],
        row_of_state=idx.row_of_state[p],
    )
    phase1 = jnp.flatnonzero(new_index.phase_of == 1)
    base_fn = sp.transition_fn

    def transition_fn(
        theta: dict[str, jax.Array], z_pi: jax.Array, z_d: jax.Array, dt: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        a, log_a = base_fn(theta, z_pi, z_d, dt)
        return a[p][:, p], log_a[p][:, p]

    return dataclasses.replace(sp, index=new_index, phase1_idx=phase1, transition_fn=transition_fn)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_e4_t03b_loglik_invariant_under_state_permutation(seed: int) -> None:
    """AC3: permuting the expanded state order leaves loglik (1e-10 rel) and permutes the posteriors."""
    sc = make_scenario(3, [2, 3, 1], 2 if seed % 2 else 1, 50, n_cov_d=1, seed=seed)
    perm = np.random.default_rng(seed).permutation(sc.space.n_expanded)
    permuted = _permuted(sc, perm)

    base = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    other = ENGINE.forward_backward(sc.theta, sc.spec, permuted, sc.data)
    assert abs(float(other.loglik) - float(base.loglik)) <= 1e-10 * abs(float(base.loglik))
    np.testing.assert_allclose(np.asarray(other.smoothed), np.asarray(base.smoothed)[:, perm], atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(other.regime_smoothed()), np.asarray(base.regime_smoothed()), atol=1e-10
    )


def test_e4_t03b_masked_steps_match_reference_with_zero_emission() -> None:
    """AC4: fully masked steps: smoothed regime marginals equal the reference with zero emission."""
    sc0 = make_scenario(3, [2, 1, 3], 1, 60, n_cov_d=1, seed=8)
    d = sc0.data
    mask = np.ones(d.mask.shape, dtype=bool)
    mask[[5, 6, 30]] = False
    sc = dataclasses.replace(
        sc0,
        data=make_dataset(
            np.asarray(d.observations), np.asarray(d.dt), mask, None, np.asarray(d.cov_duration)
        ),
    )
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    _, _, ref_gamma, ref_loglik = reference_full(sc, mask[:, 0])
    ref_regime = to_probs(ref_gamma) @ np.asarray(sc.space.index.collapse)
    np.testing.assert_allclose(np.asarray(post.regime_smoothed()), ref_regime, atol=1e-8)
    assert abs(float(post.loglik) - ref_loglik) <= 1e-8 * abs(ref_loglik)
    # the mask actually matters: the unmasked model gives a different smoothed row at a masked step
    unmasked = ENGINE.forward_backward(sc0.theta, sc0.spec, sc0.space, sc0.data)
    assert np.abs(np.asarray(post.smoothed)[5] - np.asarray(unmasked.smoothed)[5]).max() > 1e-6


def test_e4_t03b_non_finite_output_raises_numerical_error() -> None:
    """AC6: a NaN parameter raises `NumericalError` at the public boundary."""
    sc = make_scenario(2, [2, 1], 1, 12, seed=0)
    bad = dict(sc.theta)
    bad["emission.mean"] = jnp.array([jnp.nan, 0.0])
    with pytest.raises(NumericalError):
        ENGINE.forward_backward(bad, sc.spec, sc.space, sc.data)


def test_e4_t03b_posterior_is_a_pytree_and_grad_of_loglik_is_finite() -> None:
    """`Posterior` round-trips through jit; gradients through forward-backward stay finite."""
    sc = make_scenario(3, [3, 2, 1], 1, 40, seed=6)
    post = jax.jit(lambda th: ENGINE.forward_backward(th, sc.spec, sc.space, sc.data))(sc.theta)
    assert post.smoothed.shape == (40, sc.space.n_expanded)
    g = jax.grad(lambda th: ENGINE.forward_backward(th, sc.spec, sc.space, sc.data).loglik)(sc.theta)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in g.values())


@pytest.mark.parametrize("bad", ["log_alpha", "log_beta", "smoothed", "loglik"])
def test_e4_t03b_each_boundary_guard_raises(bad: str) -> None:
    """AC6: every guard in `_check_posterior` fires on its own array."""
    from hsmmkit.engine.expanded_hmm import _check_posterior

    ok = {"log_alpha": jnp.zeros((3, 2)), "log_beta": jnp.zeros((3, 2)), "smoothed": jnp.ones((3, 2)) / 2}
    ok["loglik"] = jnp.asarray(-1.0)
    ok[bad] = (
        ok[bad].at[...].set(jnp.inf if bad == "smoothed" else jnp.nan)
        if bad != "loglik"
        else jnp.asarray(jnp.nan)
    )
    with pytest.raises(NumericalError):
        _check_posterior(ok["log_alpha"], ok["log_beta"], ok["smoothed"], ok["loglik"])


def test_e4_t03b_filtered_softmax_rows_sum_to_one() -> None:
    """AC2: `softmax(log_alpha)` rows (full expanded filter) sum to 1 within 1e-10."""
    sc = make_scenario(3, [2, 3, 1], 1, 40, seed=9)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    np.testing.assert_allclose(np.asarray(jax.nn.softmax(post.log_alpha, axis=-1)).sum(-1), 1.0, atol=1e-10)
