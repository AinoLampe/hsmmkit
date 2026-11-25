"""E11-T02: the `cholesky` bijector (log-det-Jacobian against autodiff, round trip)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hsmmkit.model.params import CholeskyLowerBijector

BIJ = CholeskyLowerBijector()


@given(m=st.integers(1, 5), seed=st.integers(0, 10_000))
def test_e11_t02_log_det_jacobian_matches_jacfwd(m: int, seed: int) -> None:
    """AC2: `log_det_jac` equals `log |det jacfwd(forward)|` within 1e-8."""
    k = m * (m + 1) // 2
    u = jnp.asarray(np.random.default_rng(seed).normal(scale=1.5, size=k))
    jac = jax.jacfwd(BIJ.forward)(u)
    _, logabsdet = jnp.linalg.slogdet(jac)
    assert abs(float(BIJ.log_det_jac(u)) - float(logabsdet)) <= 1e-8


@given(m=st.integers(1, 5), seed=st.integers(0, 10_000))
def test_e11_t02_round_trip_and_valid_factor(m: int, seed: int) -> None:
    """AC2: `inverse(forward(u)) = u` within 1e-12, and `forward(u)` has a positive diagonal."""
    k = m * (m + 1) // 2
    u = jnp.asarray(np.random.default_rng(seed).normal(scale=2.0, size=(3, k)))
    theta = BIJ.forward(u)
    np.testing.assert_allclose(np.asarray(BIJ.inverse(theta)), np.asarray(u), rtol=0, atol=1e-12)
    rows, cols = np.tril_indices(m)
    assert bool(jnp.all(theta[..., rows == cols] > 0))
