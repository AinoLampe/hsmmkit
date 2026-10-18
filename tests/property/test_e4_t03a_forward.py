"""E4-T03a: masking, initial distribution, gradients, jit caching, error guard."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from regimekit.engine import expanded_hmm
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.model.params import ParamManager, ParamSpec
from regimekit.model.spec import ModelSpec
from regimekit.utils.errors import NumericalError
from tests.golden.test_e4_t03a_forward import reference
from tests.helpers.engine_fixtures import Scenario, make_dataset, make_scenario

ENGINE = ExpandedHMMEngine()


def _masked(sc: Scenario, steps: list[int], nan: bool = False) -> tuple[Scenario, np.ndarray]:
    d = sc.data
    mask = np.ones(d.mask.shape, dtype=bool)
    mask[steps] = False
    y = np.asarray(d.observations).copy()
    if nan:
        y[steps] = np.nan
    data = make_dataset(
        y,
        np.asarray(d.dt),
        mask,
        None if d.cov_transition is None else np.asarray(d.cov_transition),
        None if d.cov_duration is None else np.asarray(d.cov_duration),
    )
    return dataclasses.replace(sc, data=data), mask[:, 0]


def test_e4_t03a_masked_step_propagates_alpha_without_emission() -> None:
    """AC2: a fully masked step has `alpha_t = logsumexp(alpha_{t-1} + log A_t)` (no emission term)."""
    sc0 = make_scenario(3, [2, 1, 3], 1, 40, n_cov_d=1, seed=5)
    sc, mask = _masked(sc0, [10, 11])
    log_alpha, loglik = ENGINE.forward(sc.theta, sc.spec, sc.space, sc.data)
    la = np.asarray(log_alpha)
    for t in (10, 11):
        _, log_a = sc.space.transition_fn(sc.theta, jnp.zeros((0,)), sc.data.cov_duration[t], sc.data.dt[t])
        expected = logsumexp(la[t - 1][:, None] + np.asarray(log_a), axis=0)
        finite = np.isfinite(expected)
        assert np.array_equal(finite, np.isfinite(la[t]))
        np.testing.assert_allclose(la[t][finite], expected[finite], atol=1e-10)
    _, ref_loglik = reference(sc, mask)
    assert abs(float(loglik) - ref_loglik) <= 1e-8 * abs(ref_loglik)
    assert float(loglik) != float(ENGINE.loglik(sc0.theta, sc0.spec, sc0.space, sc0.data))


def test_e4_t03a_nan_in_masked_observation_keeps_loglik_and_grad_finite() -> None:
    """A NaN print under a False mask contributes 0 and leaks no NaN into value or gradient."""
    sc0 = make_scenario(2, [2, 2], 1, 30, seed=2)
    sc, _ = _masked(sc0, [7], nan=True)
    ll = ENGINE.loglik(sc.theta, sc.spec, sc.space, sc.data)
    assert np.isfinite(float(ll))
    grads = jax.grad(lambda th: ENGINE.loglik(th, sc.spec, sc.space, sc.data))(sc.theta)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grads.values())


def test_e4_t03a_uniform_initial_puts_zero_mass_beyond_phase_one() -> None:
    """AC3 (uniform): `alpha_1` is finite exactly on `phase == 1` states."""
    sc = make_scenario(3, [3, 2, 1], 1, 20, seed=1)
    log_alpha, _ = ENGINE.forward(sc.theta, sc.spec, sc.space, sc.data)
    phase = np.asarray(sc.space.index.phase_of)
    first = np.asarray(log_alpha[0])
    assert np.all(np.isfinite(first[phase == 1]))
    assert np.all(np.isneginf(first[phase > 1]))


def _grad_case() -> tuple[Scenario, ModelSpec, ParamManager]:
    sc = make_scenario(2, [2, 2], 1, 30, seed=11)
    specs = (
        ParamSpec("emission.mu", (2,), "real", "free", init=np.array([0.05, -0.02])),
        ParamSpec("emission.sigma", (2,), "positive", "free", init=np.array([0.15, 0.4])),
        ParamSpec("duration.rate", (2,), "positive", "free", init=np.array([5.0, 12.0])),
        ParamSpec("transition.pi_rows", (2, 1), "simplex_ref", "fixed", value=np.ones((2, 1))),
    )
    manager = ParamManager(specs)
    spec = dataclasses.replace(sc.spec, param_specs=specs)
    return sc, spec, manager


def test_e4_t03a_grad_matches_central_finite_differences() -> None:
    """AC4: `jax.grad(loglik)(u)` vs central differences (step 1e-6), rel <= 1e-5 (T=30, N=2, r=2)."""
    sc, spec, manager = _grad_case()

    def f(u: jax.Array) -> jax.Array:
        return ENGINE.loglik(manager.unflatten(u), spec, sc.space, sc.data)

    u0 = manager.init_u() - 0.3
    g = np.asarray(jax.grad(f)(u0))
    h = 1e-6
    fd = np.array(
        [(float(f(u0.at[i].add(h))) - float(f(u0.at[i].add(-h)))) / (2 * h) for i in range(u0.shape[0])]
    )
    assert u0.shape[0] == 6
    # Relative error is only meaningful away from zero (FD noise floor ~1e-8 absolute at |loglik| ~ 1e2).
    assert np.abs(g).min() > 1e-3
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=0)


def test_e4_t03a_jit_compiles_once_across_parameter_values() -> None:
    """Same shapes, different `theta`/data values: no recompilation."""
    sc = make_scenario(3, [2, 1, 2], 1, 25, seed=4)
    ENGINE.forward(sc.theta, sc.spec, sc.space, sc.data)
    before = expanded_hmm._forward_core._cache_size()
    th2 = dict(sc.theta)
    th2["emission.mu"] = sc.theta["emission.mu"] + 0.01
    ENGINE.forward(th2, sc.spec, sc.space, sc.data)
    assert expanded_hmm._forward_core._cache_size() == before


def test_e4_t03a_non_finite_loglik_raises_numerical_error() -> None:
    """Guard outside jit: a NaN parameter yields `NumericalError`, not a NaN return."""
    sc = make_scenario(2, [1, 1], 1, 10, seed=0)
    bad = dict(sc.theta)
    bad["emission.mu"] = jnp.array([jnp.nan, 0.0])
    with pytest.raises(NumericalError):
        ENGINE.loglik(bad, sc.spec, sc.space, sc.data)


def test_e4_t03a_approx_is_none() -> None:
    """The FR-X01 plug-in point returns `None` for the exact engine."""
    assert ENGINE.approx() is None


def test_e4_t03a_nan_parameter_propagates_as_nan_under_jit() -> None:
    """A NaN inside the scan is not silently turned into -inf (it reaches `loglik` as NaN)."""
    sc = make_scenario(2, [2, 1], 1, 10, seed=0)
    bad = dict(sc.theta)
    bad["emission.mu"] = jnp.array([jnp.nan, 0.0])
    _, loglik = expanded_hmm._forward_core(bad, sc.spec, sc.space, sc.data)
    assert bool(jnp.isnan(loglik))
