"""E4-T06a: entry-time covariates, gradients, grid/memory guards and the augmented index."""

from __future__ import annotations

import warnings
from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import norm

from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.model.spec import ModelSpec
from hsmmkit.statespace.builders import build_explicit
from hsmmkit.utils.errors import ConfigError, NumericalError
from tests.helpers.engine_fixtures import DT, Scenario, make_dataset, make_explicit_scenario
from tests.reference import ref_hsmm_by_start

ENGINE = ExplicitDurationEngine()


def respec(spec: ModelSpec, **changes: object) -> ModelSpec:
    """A copy of `spec` with some fields replaced (`ModelSpec` is an `eqx.Module`, not a dataclass)."""
    return ModelSpec(**{**{f.name: getattr(spec, f.name) for f in fields(spec)}, **changes})  # type: ignore[arg-type]


def per_start_reference(sc: Scenario) -> tuple[float, np.ndarray, np.ndarray]:
    """Enumeration with the covariate-dependent jump matrix and pmf of every segment start."""
    d, th, spec = sc.data, sc.theta, sc.spec
    y = np.asarray(d.observations)[:, 0]
    mu, sig = np.asarray(th["emission.mean"]), np.asarray(th["emission.scale"])
    log_emis = norm.logpdf(y[:, None], (mu - 0.5 * sig**2) * d.nominal_dt, sig * np.sqrt(d.nominal_dt))
    z_pi, z_d = d.cov_transition, d.cov_duration
    pis = [
        np.asarray(spec.transition.jump_probs_t(th, None if z_pi is None else z_pi[s], sc.space.layout))
        for s in range(y.shape[0])
    ]
    pmfs = [
        np.asarray(spec.duration.log_pmf(th, spec.d_max, d.nominal_dt, None if z_d is None else z_d[s]))
        for s in range(y.shape[0])
    ]
    with np.errstate(divide="ignore"):
        log_pi_t = np.log(np.stack(pis))
    log_rho = np.full(spec.n_states, -np.log(spec.n_states))
    return ref_hsmm_by_start.brute_force_by_start(log_rho, log_pi_t, np.stack(pmfs), log_emis)


@given(
    seed=st.integers(0, 10_000),
    family=st.sampled_from(["gamma", "weibull", "negbin"]),
    with_pi=st.booleans(),
)
def test_e4_t06a_entry_time_covariates_match_per_start_reference(
    seed: int, family: str, with_pi: bool
) -> None:
    """AC4: `(T, N, D_max)` pmfs (and `(T, N, N)` jumps) equal the enumeration with per-start rows."""
    sc = make_explicit_scenario(3, 3, 6, family=family, n_cov_d=1, n_cov_pi=1 if with_pi else 0, seed=seed)
    ref_loglik, ref_elapsed, _ = per_start_reference(sc)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    assert abs(float(post.loglik) - ref_loglik) <= 1e-10 * max(1.0, abs(ref_loglik))
    np.testing.assert_allclose(np.asarray(post.elapsed_posterior()), ref_elapsed, rtol=0, atol=1e-10)


def test_e4_t06a_covariates_change_the_answer() -> None:
    """Guard for AC4: dropping the covariates changes the log-likelihood (the test is not vacuous)."""
    sc = make_explicit_scenario(3, 3, 6, family="gamma", n_cov_d=1, n_cov_pi=1, seed=5)
    with_cov = float(ENGINE.loglik(sc.theta, sc.spec, sc.space, sc.data))
    zeros = np.zeros((6, 1))
    plain = make_dataset(np.asarray(sc.data.observations), np.asarray(sc.data.dt), z_pi=zeros, z_d=zeros)
    assert abs(float(ENGINE.loglik(sc.theta, sc.spec, sc.space, plain)) - with_cov) > 1e-6


@pytest.mark.parametrize("family", ["gamma", "negbin"])
def test_e4_t06a_gradient_is_finite_and_matches_finite_differences(family: str) -> None:
    """AC5: `grad(loglik)` finite and equal to central differences (step 1e-6, rel <= 1e-5)."""
    sc = make_explicit_scenario(2, 5, 12, family=family, seed=11, mean_steps=2.0)
    keys = ("emission.mean", "emission.scale") + (
        ("duration.shape", "duration.scale") if family == "gamma" else ("duration.rate",)
    )
    shapes = [(k, sc.theta[k].shape) for k in keys]
    x0 = jnp.concatenate([sc.theta[k].ravel() for k in keys])

    def unflatten(x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        theta, offset = dict(sc.theta), 0
        for k, shape in shapes:
            size = int(np.prod(shape))
            theta[k] = x[offset : offset + size].reshape(shape)
            offset += size
        return theta

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return ENGINE.loglik(unflatten(x), sc.spec, sc.space, sc.data)

    grad = np.asarray(jax.grad(f)(x0))
    assert np.all(np.isfinite(grad))
    for i in range(x0.shape[0]):
        step = 1e-6 * max(1.0, abs(float(x0[i])))
        fd = (float(f(x0.at[i].add(step))) - float(f(x0.at[i].add(-step)))) / (2 * step)
        assert abs(grad[i] - fd) <= 1e-5 * max(abs(fd), 1e-3), (keys, i, grad[i], fd)


def test_e4_t06a_gradient_finite_with_structural_zeros_and_short_sequence() -> None:
    """`d_max > T` (unreachable durations) leaves every gradient entry finite."""
    sc = make_explicit_scenario(3, 15, 6, family="weibull", seed=2)

    def f(mu: jnp.ndarray) -> jnp.ndarray:
        return ENGINE.loglik({**sc.theta, "emission.mean": mu}, sc.spec, sc.space, sc.data)

    assert np.all(np.isfinite(np.asarray(jax.grad(f)(sc.theta["emission.mean"]))))


@pytest.mark.parametrize("family", ["weibull", "gamma", "negbin", "geometric"])
def test_e4_t06a_gradient_finite_for_every_parameter_block(family: str) -> None:
    """Every `theta` leaf (emission, duration, `pi_rows`, `initial.logits`) has a finite gradient."""
    sc = make_explicit_scenario(3, 9, 25, family=family, initial="free", seed=6)
    grads = jax.grad(lambda th: ENGINE.loglik(th, sc.spec, sc.space, sc.data))(sc.theta)
    assert set(grads) == set(sc.theta)
    for name, g in grads.items():
        assert np.all(np.isfinite(np.asarray(g))), name
        assert float(jnp.max(jnp.abs(g))) > 0.0, name


def test_e4_t06a_irregular_grid_warns_once_and_uses_nominal_dt() -> None:
    """Irregular `dt` warns exactly once per call; the result equals the regular grid at `nominal_dt`."""
    sc = make_explicit_scenario(3, 6, 40, family="gamma", seed=3, irregular=True)
    for call in (ENGINE.loglik, ENGINE.forward, ENGINE.forward_backward):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(sc.theta, sc.spec, sc.space, sc.data)
        assert sum("regular grid" in str(w.message) for w in caught) == 1
    regular = make_dataset(np.asarray(sc.data.observations), np.full(40, sc.data.nominal_dt))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected = ENGINE.loglik(sc.theta, sc.spec, sc.space, regular)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert float(ENGINE.loglik(sc.theta, sc.spec, sc.space, sc.data)) == float(expected)


def test_e4_t06a_regular_grid_does_not_warn() -> None:
    """A constant `dt` grid emits no warning."""
    sc = make_explicit_scenario(2, 4, 20, family="geometric", seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)


def test_e4_t06a_memory_guard_raises_before_allocating() -> None:
    """Full `(T, N, D_max)` tables above `5e7` cells raise `NumericalError("use chunked scan")` up front."""
    sc = make_explicit_scenario(4, 250, 20, family="geometric", seed=1)
    big = make_dataset(np.zeros((50_001, 1)), np.full(50_001, DT))  # 50_001 * 4 * 250 > 5e7
    with pytest.raises(NumericalError, match="chunked scan"):
        ENGINE.forward(sc.theta, sc.spec, sc.space, big)


def test_e4_t06a_mismatched_state_space_is_rejected() -> None:
    """A state space built for another `D_max` cannot drive the engine."""
    sc = make_explicit_scenario(2, 4, 20, family="geometric", seed=1)
    other = make_explicit_scenario(2, 7, 20, family="geometric", seed=1)
    with pytest.raises(ConfigError, match="build_explicit"):
        ENGINE.loglik(sc.theta, sc.spec, other.space, sc.data)


def test_e4_t06a_unsupported_initial_is_rejected() -> None:
    """`initial="stationary"` has no explicit-engine meaning yet and raises `ConfigError`."""
    sc = make_explicit_scenario(2, 4, 20, family="geometric", seed=1)
    with pytest.raises(ConfigError, match="uniform"):
        ENGINE.loglik(sc.theta, respec(sc.spec, initial="stationary"), sc.space, sc.data)


def test_e4_t06a_build_explicit_layout() -> None:
    """`s = j * D_max + (e - 1)`; `phase_of` holds the elapsed count; order > 1 is rejected."""
    sc = make_explicit_scenario(3, 4, 10, family="geometric", seed=1)
    idx = sc.space.index
    assert idx.n_expanded == 12
    np.testing.assert_array_equal(np.asarray(idx.regime_of), np.repeat(np.arange(3), 4))
    np.testing.assert_array_equal(np.asarray(idx.phase_of), np.tile(np.arange(1, 5), 3))
    np.testing.assert_array_equal(np.asarray(sc.space.phase1_idx), [0, 4, 8])
    np.testing.assert_array_equal(np.asarray(sc.space.entry_state_of), [[-1, 4, 8], [0, -1, 8], [0, 4, -1]])
    np.testing.assert_array_equal(np.asarray(sc.space.next_phase_state)[:5], [1, 2, 3, -1, 5])
    np.testing.assert_allclose(np.asarray(idx.collapse).sum(axis=0), 4.0)
    with pytest.raises(ConfigError, match="order"):
        build_explicit(respec(sc.spec, order=2))
