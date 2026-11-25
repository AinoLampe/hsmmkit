"""E9-T02: factorial state-space prototype -- exact-inference equality for independent chains (ADR-0008)."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.engine.approx import ApproxInference, ApproxPosterior
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.statespace.factorial import (
    CouplingSpec,
    FactorialSpace,
    build_factorial,
    chain_theta,
    joint_model_spec,
)
from hsmmkit.utils.errors import ConfigError
from tests.helpers.engine_fixtures import Scenario, make_scenario

T_LEN = 60
TOL = 1e-10


def _chains(sizes: tuple[int, ...]) -> list[Scenario]:
    """One `r = 1` scenario per chain; every chain is later run on chain 0's data."""
    return [make_scenario(n, [1] * n, 1, T_LEN, seed=10 + c) for c, n in enumerate(sizes)]


def _joint(chains: list[Scenario]) -> tuple[FactorialSpace, dict[str, jnp.ndarray]]:
    fs = build_factorial([c.spec for c in chains], experimental=True)
    theta = {f"c{c}.{k}": v for c, sc in enumerate(chains) for k, v in sc.theta.items()}
    return fs, theta


@pytest.mark.parametrize("sizes", [(2, 2), (2, 3), (3, 3), (2, 2, 2)])
def test_e9_t02_independent_chains_equal_the_product_of_single_chain_results(
    sizes: tuple[int, ...],
) -> None:
    """AC2: the joint space gives the loglik sum and outer-product marginals (1e-10)."""
    chains = _chains(sizes)
    data = chains[0].data
    fs, theta = _joint(chains)
    eng = ExpandedHMMEngine()
    joint = eng.forward_backward(theta, joint_model_spec(fs), fs.space, data)

    singles = [eng.forward_backward(sc.theta, sc.spec, sc.space, data) for sc in chains]
    assert abs(float(joint.loglik) - sum(float(p.loglik) for p in singles)) <= TOL

    # chain marginals of the joint posterior equal each chain's own smoothed regimes
    for marg, single in zip(fs.chain_marginals(joint.smoothed), singles, strict=True):
        np.testing.assert_allclose(np.asarray(marg), np.asarray(single.regime_smoothed()), rtol=0, atol=TOL)

    # the joint posterior is the outer product of the single-chain posteriors
    expected = np.asarray(singles[0].smoothed)
    for p in singles[1:]:
        expected = np.einsum("ta,tb->tab", expected, np.asarray(p.smoothed)).reshape(T_LEN, -1)
    np.testing.assert_allclose(np.asarray(joint.smoothed), expected, rtol=0, atol=TOL)
    np.testing.assert_allclose(np.asarray(joint.smoothed).sum(-1), 1.0, rtol=0, atol=TOL)


def test_e9_t02_joint_transition_is_the_kronecker_product() -> None:
    """`A_t` of the joint space equals `np.kron` of the per-chain matrices and its rows sum to one."""
    chains = _chains((2, 3))
    fs, theta = _joint(chains)
    dt = jnp.asarray(1.0 / 252.0)
    a_joint, log_a = fs.space.transition_fn(theta, None, None, dt)
    parts = [np.asarray(sc.space.transition_fn(sc.theta, None, None, dt)[0]) for sc in chains]
    np.testing.assert_allclose(np.asarray(a_joint), np.kron(parts[0], parts[1]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(a_joint).sum(1), 1.0, rtol=0, atol=1e-12)
    assert not bool(jnp.any(jnp.isnan(log_a)))
    assert fs.regime_of_chain.shape == (fs.space.n_expanded, 2)


def test_e9_t02_coupling_hook_renormalises_and_zero_factor_is_independent() -> None:
    """A zero interaction factor reproduces independence; a non-zero one changes `A_t` but keeps rows at 1."""
    chains = _chains((2, 2))
    base, theta = _joint(chains)
    dt = jnp.asarray(1.0 / 252.0)
    a0 = np.asarray(base.space.transition_fn(theta, None, None, dt)[0])

    zero = build_factorial(
        [c.spec for c in chains], CouplingSpec(lambda th, d: jnp.zeros((4, 4))), experimental=True
    )
    np.testing.assert_allclose(np.asarray(zero.space.transition_fn(theta, None, None, dt)[0]), a0, atol=1e-14)

    bump = jnp.asarray(np.where(np.eye(4) == 0, 0.7, 0.0))
    coupled = build_factorial([c.spec for c in chains], CouplingSpec(lambda th, d: bump), experimental=True)
    a1 = np.asarray(coupled.space.transition_fn(theta, None, None, dt)[0])
    np.testing.assert_allclose(a1.sum(1), 1.0, rtol=0, atol=1e-12)
    assert np.abs(a1 - a0).max() > 1e-3


def test_e9_t02_guards() -> None:
    """The prototype refuses to run without the experimental flag, with one chain, or an unsupported chain."""
    chains = _chains((2, 2))
    specs = [c.spec for c in chains]
    with pytest.raises(ConfigError, match="experimental"):
        build_factorial(specs)
    with pytest.raises(ConfigError, match="two chains"):
        build_factorial(specs[:1], experimental=True)
    order2 = dataclasses.replace(specs[0], order=2)
    with pytest.raises(ConfigError, match="order-1"):
        build_factorial([order2, specs[1]], experimental=True)


def test_e9_t02_existing_engines_still_report_no_approximation() -> None:
    """AC3: `approx()` of the exact engines is `None`; the protocol types are importable and structural."""
    assert ExpandedHMMEngine().approx() is None
    assert ExplicitDurationEngine().approx() is None
    post = ApproxPosterior(marginals=(jnp.ones((3, 2)) / 2,), bound=jnp.asarray(0.0), n_sweeps=1)
    assert post.n_sweeps == 1
    assert ApproxInference.__name__ == "ApproxInference"


def test_e9_t02_chain_theta_strips_the_prefix() -> None:
    """`chain_theta` selects one chain's parameters."""
    assert set(chain_theta({"c0.a": 1, "c1.a": 2, "c1.b": 3}, 1)) == {"a", "b"}  # type: ignore[dict-item]


def test_e9_t02_multi_phase_chains_are_rejected() -> None:
    """Chains with `r > 1` are refused (the joint phase bookkeeping is a placeholder in the spike)."""
    multi = make_scenario(2, [2, 1], 1, T_LEN, seed=3)
    single = _chains((2,))[0]
    with pytest.raises(ConfigError, match="single-phase"):
        build_factorial([multi.spec, single.spec], experimental=True)
