"""E1-T03a: log-prior densities match `scipy.stats` references, and are jittable/differentiable."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from hsmmkit.model.params.priors import DEFAULT_COEF_PRIOR, DirichletPrior, LogNormalPrior, NormalPrior


@pytest.mark.parametrize("seed", range(5))
def test_e1_t03a_normal_prior_matches_scipy(seed: int) -> None:
    """`NormalPrior.log_prob` matches `scipy.stats.norm.logpdf`, summed, within 1e-10."""
    rng = np.random.default_rng(seed)
    mean, sd = rng.uniform(-3, 3), rng.uniform(0.1, 4.0)
    theta = rng.normal(size=6)
    mine = float(NormalPrior(mean=mean, sd=sd).log_prob(jnp.asarray(theta)))
    reference = float(np.sum(stats.norm(mean, sd).logpdf(theta)))
    assert abs(mine - reference) <= 1e-10


@pytest.mark.parametrize("seed", range(5))
def test_e1_t03a_lognormal_prior_matches_scipy(seed: int) -> None:
    """`LogNormalPrior.log_prob` matches `scipy.stats.lognorm.logpdf`, summed, within 1e-10."""
    rng = np.random.default_rng(seed)
    mean_log, sd = rng.uniform(-1, 1), rng.uniform(0.1, 2.0)
    theta = rng.uniform(0.01, 5.0, size=6)
    mine = float(LogNormalPrior(mean_log=mean_log, sd=sd).log_prob(jnp.asarray(theta)))
    reference = float(np.sum(stats.lognorm(s=sd, scale=np.exp(mean_log)).logpdf(theta)))
    assert abs(mine - reference) <= 1e-10


@pytest.mark.parametrize("seed", range(5))
def test_e1_t03a_dirichlet_prior_matches_scipy(seed: int) -> None:
    """`DirichletPrior.log_prob` matches `scipy.stats.dirichlet.logpdf` within 1e-10."""
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 6)
    alpha = rng.uniform(0.5, 5.0)
    raw = rng.uniform(0.1, 1.0, size=n)
    p = raw / raw.sum()
    mine = float(DirichletPrior(alpha=alpha).log_prob(jnp.asarray(p)))
    reference = float(stats.dirichlet(alpha * np.ones(n)).logpdf(p))
    assert abs(mine - reference) <= 1e-10


def test_e1_t03a_dirichlet_prior_sums_over_matrix_rows() -> None:
    """A `(rows, n)` input sums the per-row Dirichlet log-density over all rows."""
    p = jnp.array([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]])
    alpha = 2.0
    mine = float(DirichletPrior(alpha=alpha).log_prob(p))
    reference = sum(
        float(stats.dirichlet(alpha * np.ones(3)).logpdf(np.asarray(row))) for row in np.asarray(p)
    )
    assert abs(mine - reference) <= 1e-10


def test_e1_t03a_default_coef_prior_is_standard_normal() -> None:
    """`DEFAULT_COEF_PRIOR` is `Normal(0, 1)` (the default `coef_prior_sd`)."""
    assert DEFAULT_COEF_PRIOR.mean == 0.0
    assert DEFAULT_COEF_PRIOR.sd == 1.0


@pytest.mark.parametrize(
    "prior_and_theta",
    [
        (NormalPrior(mean=0.0, sd=1.0), jnp.array([0.1, -0.2, 0.3])),
        (LogNormalPrior(mean_log=0.0, sd=1.0), jnp.array([0.5, 1.2, 3.4])),
        (DirichletPrior(alpha=2.0), jnp.array([0.2, 0.3, 0.5])),
    ],
)
def test_e1_t03a_priors_are_jittable_and_differentiable(
    prior_and_theta: tuple[NormalPrior | LogNormalPrior | DirichletPrior, jax.Array],
) -> None:
    """`log_prob` is jittable, and `jax.grad` through it is finite."""
    prior, theta = prior_and_theta
    value = jax.jit(prior.log_prob)(theta)
    assert jnp.isfinite(value)
    grad = jax.grad(jax.jit(prior.log_prob))(theta)
    assert bool(jnp.all(jnp.isfinite(grad)))
