"""E4-T02b: `A_t` row normalization, structural zeros, generator-mode Appendix E golden."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.model.duration.geometric import GeometricDuration
from regimekit.model.duration.negbin import NegBinDuration
from regimekit.model.emission.gaussian import GaussianFamily
from regimekit.model.spec import ModelSpec
from regimekit.model.transition import JumpTransition
from regimekit.statespace.builders import build_expanded, initial_log_rho


def _spec(
    n_states: int,
    order: int,
    duration: object,
    *,
    mode: str = "jump",
    duration_conditioning: str = "current",
    initial: str = "uniform",
) -> ModelSpec:
    return ModelSpec(
        n_states=n_states,
        n_series=1,
        emission=GaussianFamily(scale_by_dt=True, ito_correction=True),
        duration=duration,  # type: ignore[arg-type]
        transition=JumpTransition(mode=mode),  # type: ignore[arg-type]
        order=order,
        duration_conditioning=duration_conditioning,  # type: ignore[arg-type]
        engine="expanded",
        d_max=10,
        initial=initial,  # type: ignore[arg-type]
        param_specs=(),
    )


# --- AC1: rows sum to 1; structural zeros exact; log A_t == -inf there ---------------------------


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("n_states", [2, 3, 4])
def test_e4_t02b_rows_sum_to_one_and_structural_zeros(n_states: int, order: int) -> None:
    """`A_t` rows sum to 1 within 1e-12; structural zeros are exactly 0; `log A_t` is `-inf` there."""
    rng = np.random.default_rng(hash((n_states, order)) % (2**31))
    r = rng.integers(1, 4, size=n_states).tolist()
    duration = NegBinDuration(r=tuple(r))
    spec = _spec(n_states, order, duration)
    space = build_expanded(spec)

    rate = jnp.asarray(rng.uniform(0.02, 0.3, size=n_states))
    r_rows = space.layout.tuples.shape[0]
    pi_rows = jnp.asarray(rng.dirichlet(np.ones(n_states - 1), size=r_rows))
    theta = {"duration.rate": rate, "transition.pi_rows": pi_rows}

    a, log_a = space.transition_fn(theta, None, None, jnp.array(1.0 / 252.0))
    a_np, log_a_np = np.asarray(a), np.asarray(log_a)

    np.testing.assert_allclose(a_np.sum(axis=1), 1.0, atol=1e-12)
    zero_mask = a_np == 0.0
    assert np.all(np.isneginf(log_a_np[zero_mask]))
    assert np.all(np.isfinite(log_a_np[~zero_mask]))


# --- AC4: generator mode Appendix E; stationary vector ---------------------------------------------


def test_e4_t02b_generator_mode_appendix_e() -> None:
    """Generator mode reproduces the (verified-correct) Appendix E weekly numbers."""
    duration = GeometricDuration()
    spec = _spec(2, 1, duration, mode="generator")
    space = build_expanded(spec)
    theta = {"duration.rate": jnp.array([2.0, 6.0]), "transition.pi_rows": jnp.array([[1.0], [1.0]])}
    a, _ = space.transition_fn(theta, None, None, jnp.array(1.0 / 52.0))
    a_np = np.asarray(a)
    assert round(float(a_np[0, 1]), 4) == pytest.approx(0.0356)
    assert round(float(a_np[1, 0]), 4) == pytest.approx(0.1069)


def test_e4_t02b_stationary_satisfies_pi_a_equals_pi() -> None:
    """`stationary` initial distribution: `pi @ A_bar == pi` within 1e-10."""
    duration = GeometricDuration()
    spec = _spec(2, 1, duration, mode="generator", initial="stationary")
    space = build_expanded(spec)
    theta = {"duration.rate": jnp.array([2.0, 6.0]), "transition.pi_rows": jnp.array([[1.0], [1.0]])}
    nominal_dt = 1.0 / 52.0
    log_pi = initial_log_rho(spec, space, theta, nominal_dt=nominal_dt)
    pi = np.exp(np.asarray(log_pi))

    a_bar, _ = space.transition_fn(theta, None, None, jnp.array(nominal_dt))
    lhs = pi @ np.asarray(a_bar)
    np.testing.assert_allclose(lhs, pi, atol=1e-10)
    assert pi.sum() == pytest.approx(1.0, abs=1e-10)


def test_e4_t02b_stationary_matches_known_2state_formula() -> None:
    """The stationary vector matches the closed-form 2-state formula `lambda_1/(lambda_0+lambda_1)`."""
    duration = GeometricDuration()
    spec = _spec(2, 1, duration, mode="generator", initial="stationary")
    space = build_expanded(spec)
    theta = {"duration.rate": jnp.array([2.0, 6.0]), "transition.pi_rows": jnp.array([[1.0], [1.0]])}
    log_pi = initial_log_rho(spec, space, theta, nominal_dt=1.0 / 52.0)
    pi = np.exp(np.asarray(log_pi))
    np.testing.assert_allclose(pi, [0.75, 0.25], atol=1e-9)
