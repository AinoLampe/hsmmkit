"""E1-T03a: bijector round-trip, domain, log-det-Jacobian, and jit/grad property tests."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hsmmkit.model.params.bijectors import (
    Bijector,
    IdentityBijector,
    IncreasingBijector,
    IntervalBijector,
    NuGt2Bijector,
    PmfBijector,
    PositiveBijector,
    SimplexRefBijector,
)

_finite_floats = st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False, width=64)


def _vec(dim_min: int = 1, dim_max: int = 6) -> st.SearchStrategy[np.ndarray]:
    return st.integers(min_value=dim_min, max_value=dim_max).flatmap(
        lambda n: arrays(dtype=np.float64, shape=(n,), elements=_finite_floats)
    )


def _autodiff_log_det_jac(forward: Callable[[jax.Array], jax.Array], u: jax.Array) -> jax.Array:
    jac = jax.jacfwd(forward)(u)
    return jnp.linalg.slogdet(jac)[1]


ELEMENTWISE_BIJECTOR_FACTORIES = [
    lambda: IdentityBijector(),
    lambda: PositiveBijector(),
    lambda: PositiveBijector(variant="exp"),
    lambda: NuGt2Bijector(),
    lambda: IntervalBijector(lo=-2.0, hi=5.0),
]


@pytest.mark.parametrize("make_bijector", ELEMENTWISE_BIJECTOR_FACTORIES)
@given(u=_vec())
def test_e1_t03a_elementwise_bijector_round_trip(
    make_bijector: Callable[[], Bijector], u: np.ndarray
) -> None:
    """`inverse(forward(u)) == u` for the elementwise bijectors."""
    bijector = make_bijector()
    u_jax = jnp.asarray(u)
    theta = bijector.forward(u_jax)
    u_back = bijector.inverse(theta)
    np.testing.assert_allclose(np.asarray(u_back), np.asarray(u_jax), atol=1e-9, rtol=1e-9)


@given(u=_vec())
def test_e1_t03a_positive_softplus_lands_in_domain(u: np.ndarray) -> None:
    """`PositiveBijector()` output is strictly positive."""
    theta = PositiveBijector().forward(jnp.asarray(u))
    assert bool(jnp.all(theta > 0))


@given(u=_vec())
def test_e1_t03a_positive_exp_lands_in_domain(u: np.ndarray) -> None:
    """`PositiveBijector(variant='exp')` output is strictly positive."""
    theta = PositiveBijector(variant="exp").forward(jnp.asarray(u))
    assert bool(jnp.all(theta > 0))


@given(u=_vec())
def test_e1_t03a_nu_gt_2_lands_in_domain(u: np.ndarray) -> None:
    """`NuGt2Bijector()` output is strictly greater than 2."""
    theta = NuGt2Bijector().forward(jnp.asarray(u))
    assert bool(jnp.all(theta > 2.0))


@given(u=_vec())
def test_e1_t03a_interval_lands_in_bounds(u: np.ndarray) -> None:
    """`IntervalBijector(lo, hi)` output stays strictly inside `(lo, hi)`."""
    bijector = IntervalBijector(lo=-2.0, hi=5.0)
    theta = bijector.forward(jnp.asarray(u))
    assert bool(jnp.all(theta > bijector.lo))
    assert bool(jnp.all(theta < bijector.hi))


@pytest.mark.parametrize("make_bijector", ELEMENTWISE_BIJECTOR_FACTORIES)
@given(u=_vec())
def test_e1_t03a_elementwise_log_det_jac_matches_autodiff(
    make_bijector: Callable[[], Bijector], u: np.ndarray
) -> None:
    """`log_det_jac` matches an autodiff `slogdet` reference, within 1e-8."""
    bijector = make_bijector()
    u_jax = jnp.asarray(u)
    analytic = bijector.log_det_jac(u_jax)
    reference = _autodiff_log_det_jac(bijector.forward, u_jax)
    assert abs(float(analytic) - float(reference)) <= 1e-8


@given(u=_vec(dim_min=2))
def test_e1_t03a_simplex_ref_round_trip_and_domain(u: np.ndarray) -> None:
    """`SimplexRefBijector`: round trip holds and `theta` is a valid simplex."""
    bijector = SimplexRefBijector()
    u_jax = jnp.asarray(u)
    theta = bijector.forward(u_jax)
    assert theta.shape[0] == u_jax.shape[0] + 1
    assert bool(jnp.all(theta > 0))
    assert abs(float(jnp.sum(theta)) - 1.0) <= 1e-8
    u_back = bijector.inverse(theta)
    np.testing.assert_allclose(np.asarray(u_back), np.asarray(u_jax), atol=1e-9, rtol=1e-9)


@given(u=_vec(dim_min=2))
def test_e1_t03a_pmf_round_trip_and_domain(u: np.ndarray) -> None:
    """`PmfBijector`: round trip holds and `theta` is a valid pmf over `1..D_max`."""
    bijector = PmfBijector()
    u_jax = jnp.asarray(u)
    theta = bijector.forward(u_jax)
    assert bool(jnp.all(theta > 0))
    assert abs(float(jnp.sum(theta)) - 1.0) <= 1e-8
    u_back = bijector.inverse(theta)
    np.testing.assert_allclose(np.asarray(u_back), np.asarray(u_jax), atol=1e-9, rtol=1e-9)


@given(u=_vec(dim_min=2))
def test_e1_t03a_simplex_ref_log_det_jac_matches_autodiff(u: np.ndarray) -> None:
    """`SimplexRefBijector.log_det_jac` matches autodiff `slogdet` on the free coordinates."""
    bijector = SimplexRefBijector()
    u_jax = jnp.asarray(u)

    def forward_free(u_: jax.Array) -> jax.Array:
        return bijector.forward(u_)[1:]

    analytic = bijector.log_det_jac(u_jax)
    reference = _autodiff_log_det_jac(forward_free, u_jax)
    assert abs(float(analytic) - float(reference)) <= 1e-8


@given(u=_vec(dim_min=2))
def test_e1_t03a_increasing_round_trip_and_strictly_increasing(u: np.ndarray) -> None:
    """`IncreasingBijector` output is strictly increasing and round-trips."""
    bijector = IncreasingBijector(base=PositiveBijector())
    u_jax = jnp.asarray(u)
    theta = bijector.forward(u_jax)
    assert bool(jnp.all(jnp.diff(theta) > 0))
    u_back = bijector.inverse(theta)
    np.testing.assert_allclose(np.asarray(u_back), np.asarray(u_jax), atol=1e-9, rtol=1e-9)


@given(u=_vec(dim_min=2))
def test_e1_t03a_increasing_log_det_jac_matches_autodiff(u: np.ndarray) -> None:
    """`IncreasingBijector.log_det_jac` matches an autodiff `slogdet` reference."""
    bijector = IncreasingBijector(base=PositiveBijector())
    u_jax = jnp.asarray(u)
    analytic = bijector.log_det_jac(u_jax)
    reference = _autodiff_log_det_jac(bijector.forward, u_jax)
    assert abs(float(analytic) - float(reference)) <= 1e-8


# --- AC4: jittable and differentiable -----------------------------------------------------


@pytest.mark.parametrize(
    "make_bijector",
    [
        *ELEMENTWISE_BIJECTOR_FACTORIES,
        lambda: SimplexRefBijector(),
        lambda: PmfBijector(),
        lambda: IncreasingBijector(base=PositiveBijector()),
    ],
)
def test_e1_t03a_forward_and_log_det_jac_are_jittable_and_differentiable(
    make_bijector: Callable[[], Bijector],
) -> None:
    """`forward` and `log_det_jac` are jittable, and `jax.grad` through them is finite."""
    bijector = make_bijector()
    u = jnp.array([0.3, -0.5, 1.1])

    forward_jit = jax.jit(bijector.forward)
    theta = forward_jit(u)
    assert bool(jnp.all(jnp.isfinite(theta)))

    def summed_log_det(u_: jax.Array) -> jax.Array:
        return bijector.log_det_jac(u_)

    grad = jax.grad(jax.jit(summed_log_det))(u)
    assert bool(jnp.all(jnp.isfinite(grad)))

    def summed_forward(u_: jax.Array) -> jax.Array:
        return jnp.sum(bijector.forward(u_))

    grad_forward = jax.grad(jax.jit(summed_forward))(u)
    assert bool(jnp.all(jnp.isfinite(grad_forward)))
