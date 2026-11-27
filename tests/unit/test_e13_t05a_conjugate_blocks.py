"""E13-T05a AC2: KL terms of the conjugate blocks against independent NumPy / SciPy references."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, special, stats

from hsmmkit.model.conjugate import (
    BetaConjugate,
    DirichletConjugate,
    GammaConjugate,
    NormalGamma,
    beta_kl,
    dirichlet_kl,
    gamma_kl,
)
from hsmmkit.model.emission.gaussian import NormalWishart


def _quad(f, lo: float, hi: float) -> float:  # noqa: ANN001
    return float(integrate.quad(f, lo, hi, epsabs=1e-13, epsrel=1e-13, limit=400)[0])


@pytest.mark.parametrize(
    ("a", "b", "a0", "b0"), [(3.2, 1.7, 1.0, 1.0), (10.5, 4.0, 2.0, 0.5), (0.9, 2.2, 1.5, 1.5)]
)
def test_e13_t05a_gamma_kl_matches_numerical_integration(a: float, b: float, a0: float, b0: float) -> None:
    """Gamma KL against quadrature of the defining integral."""
    q, p = stats.gamma(a, scale=1 / b), stats.gamma(a0, scale=1 / b0)
    ref = _quad(lambda x: q.pdf(x) * (q.logpdf(x) - p.logpdf(x)), 0.0, np.inf)
    assert abs(float(gamma_kl(jnp.asarray(a), jnp.asarray(b), a0, b0)) - ref) <= 1e-10


@pytest.mark.parametrize(
    ("a", "b", "a0", "b0"), [(3.2, 1.7, 1.0, 1.0), (10.5, 4.0, 2.0, 0.5), (0.9, 2.2, 1.5, 1.5)]
)
def test_e13_t05a_beta_kl_matches_numerical_integration(a: float, b: float, a0: float, b0: float) -> None:
    """Gamma KL against quadrature of the defining integral."""
    q, p = stats.beta(a, b), stats.beta(a0, b0)
    ref = _quad(lambda x: q.pdf(x) * (q.logpdf(x) - p.logpdf(x)), 0.0, 1.0)
    assert abs(float(beta_kl(jnp.asarray(a), jnp.asarray(b), a0, b0)) - ref) <= 1e-10


def test_e13_t05a_dirichlet_kl_matches_scipy_entropy_and_respects_the_mask() -> None:
    """Beta KL against quadrature of the defining integral."""
    alpha, alpha0 = np.array([4.0, 2.5, 7.1, 1.3]), np.array([1.0, 2.0, 1.0, 0.5])
    psi = special.digamma(alpha) - special.digamma(alpha.sum())
    cross = special.gammaln(alpha0.sum()) - special.gammaln(alpha0).sum() + ((alpha0 - 1) * psi).sum()
    ref = -stats.dirichlet(alpha).entropy() - cross
    assert abs(float(dirichlet_kl(jnp.asarray(alpha), jnp.asarray(alpha0))) - ref) <= 1e-10
    mask = np.array([True, True, True, False])
    sub = alpha[:3]
    psi3 = special.digamma(sub) - special.digamma(sub.sum())
    cross3 = (
        special.gammaln(alpha0[:3].sum())
        - special.gammaln(alpha0[:3]).sum()
        + ((alpha0[:3] - 1) * psi3).sum()
    )
    ref3 = -stats.dirichlet(sub).entropy() - cross3
    got = float(dirichlet_kl(jnp.asarray(alpha), jnp.asarray(alpha0), jnp.asarray(mask)))
    assert abs(got - ref3) <= 1e-10


def test_e13_t05a_normal_gamma_kl_matches_numerical_integration() -> None:
    """Dirichlet KL against SciPy's entropy, also on a masked support."""
    prior = NormalGamma(m0=0.5, kappa0=0.3, a0=2.0, b0=1.5)
    q = {k: jnp.asarray([[v]]) for k, v in {"m": 1.2, "kappa": 4.3, "a": 5.5, "b": 3.1}.items()}
    gq, gp = stats.gamma(5.5, scale=1 / 3.1), stats.gamma(2.0, scale=1 / 1.5)

    def inner(tau: float) -> float:  # KL of the conditional normals at precision tau, then weighted by q(tau)
        sd, sd0 = 1.0 / np.sqrt(4.3 * tau), 1.0 / np.sqrt(0.3 * tau)
        kl = np.log(sd0 / sd) + (sd**2 + (1.2 - 0.5) ** 2) / (2 * sd0**2) - 0.5
        return gq.pdf(tau) * (gq.logpdf(tau) - gp.logpdf(tau) + kl)

    ref = _quad(inner, 0.0, np.inf)
    assert abs(float(prior.kl(q)) - ref) <= 1e-10


def test_e13_t05a_normal_wishart_kl_matches_scipy_wishart_entropy() -> None:
    """Normal-Gamma KL against quadrature over the precision."""
    m_dim = 2
    prior = NormalWishart(
        kappa0=0.4, nu0=5.0, m0=jnp.asarray([0.3, -0.2]), w0=jnp.asarray([[0.8, 0.1], [0.1, 0.5]])
    )
    w = np.array([[0.4, -0.05], [-0.05, 0.3]])
    q = {
        "m": jnp.asarray([[1.0, 0.4]]),
        "kappa": jnp.asarray([7.5]),
        "W": jnp.asarray(w[None]),
        "nu": jnp.asarray([9.0]),
    }
    nu, nu0 = 9.0, 5.0
    w0 = np.array([[0.8, 0.1], [0.1, 0.5]])
    e_logdet = (
        sum(special.digamma(0.5 * (nu + 1 - i)) for i in range(1, m_dim + 1))
        + m_dim * np.log(2)
        + np.linalg.slogdet(w)[1]
    )
    ln_p0 = (
        -0.5 * nu0 * m_dim * np.log(2)
        - 0.5 * nu0 * np.linalg.slogdet(w0)[1]
        - special.multigammaln(0.5 * nu0, m_dim)
        + 0.5 * (nu0 - m_dim - 1) * e_logdet
        - 0.5 * nu * np.trace(np.linalg.inv(w0) @ w)
    )
    kl_wishart = -stats.wishart(df=nu, scale=w).entropy() - ln_p0
    diff = np.array([1.0, 0.4]) - np.array([0.3, -0.2])
    kl_normal = 0.5 * (m_dim * 0.4 / 7.5 - m_dim + 0.4 * nu * diff @ w @ diff + m_dim * np.log(7.5 / 0.4))
    assert abs(float(prior.kl(q)) - (kl_wishart + kl_normal)) <= 1e-10


@pytest.mark.parametrize(
    "prior",
    [
        GammaConjugate(),
        BetaConjugate(),
        DirichletConjugate(),
        NormalGamma(m0=0.2, kappa0=0.1, a0=2.0, b0=3.0),
    ],
)
def test_e13_t05a_zero_data_posterior_is_the_prior(prior) -> None:  # noqa: ANN001
    """With zero data the posterior equals the prior (KL = 0)."""
    n = 3
    if isinstance(prior, DirichletConjugate):
        zero = prior.zero_stats(n, 4)
    elif isinstance(prior, NormalGamma):
        zero = prior.zero_stats(n, 2)
    else:
        zero = prior.zero_stats(n, 2)
    q = prior.update(zero)
    assert abs(float(prior.kl(q))) <= 1e-12


def test_e13_t05a_normal_wishart_zero_data_posterior_is_the_prior() -> None:
    """Normal-Wishart KL against SciPy's Wishart entropy and the analytic normal part."""
    prior = NormalWishart(
        kappa0=0.4, nu0=5.0, m0=jnp.asarray([0.3, -0.2]), w0=jnp.asarray([[0.8, 0.1], [0.1, 0.5]])
    )
    q = prior.update(prior.zero_stats(2, 2))
    assert abs(float(prior.kl(q))) <= 1e-10
    np.testing.assert_allclose(np.asarray(q["nu"]), 5.0)
    np.testing.assert_allclose(np.asarray(q["W"][0]), np.asarray(prior.w0), atol=1e-12)


def test_e13_t05a_expected_log_prob_matches_monte_carlo() -> None:
    """`expected_log_prob` of the families equals the average of `log f` over draws from `q` (3 SE)."""
    from hsmmkit.model.emission.categorical import CategoricalFamily
    from hsmmkit.model.emission.gaussian import GaussianFamily
    from hsmmkit.model.emission.poisson import PoissonFamily

    rng = np.random.default_rng(0)
    n_draw = 400_000
    y = np.array([[0.3], [1.4], [-0.8]])
    dt = jnp.ones(3)
    # Normal-Gamma (diagonal Gaussian), one regime
    q = {
        "m": jnp.asarray([[0.5]]),
        "kappa": jnp.asarray([[3.0]]),
        "a": jnp.asarray([[4.0]]),
        "b": jnp.asarray([[2.5]]),
    }
    got = np.asarray(GaussianFamily().expected_log_prob(jnp.asarray(y), dt, q, None))[:, 0]
    tau = rng.gamma(4.0, 1 / 2.5, n_draw)
    mu = rng.normal(0.5, 1 / np.sqrt(3.0 * tau))
    draws = stats.norm.logpdf(y[:, 0][:, None], mu[None], 1 / np.sqrt(tau)[None])
    assert np.all(np.abs(got - draws.mean(axis=1)) <= 3 * draws.std(axis=1) / np.sqrt(n_draw))
    # Poisson
    qp = {"a": jnp.asarray([[3.0]]), "b": jnp.asarray([[1.5]])}
    yp = np.array([[0.0], [2.0], [5.0]])
    gotp = np.asarray(PoissonFamily(scale_by_dt=False).expected_log_prob(jnp.asarray(yp), dt, qp, None))[:, 0]
    lam = rng.gamma(3.0, 1 / 1.5, n_draw)
    dp = stats.poisson.logpmf(yp[:, 0][:, None], lam[None])
    assert np.all(np.abs(gotp - dp.mean(axis=1)) <= 3 * dp.std(axis=1) / np.sqrt(n_draw))
    # Categorical
    alpha = np.array([[2.0, 5.0, 1.5]])
    gotc = np.asarray(
        CategoricalFamily(n_categories=3).expected_log_prob(
            jnp.asarray([[0.0], [1.0], [2.0]]), dt, {"alpha": jnp.asarray(alpha)}, None
        )
    )[:, 0]
    p = rng.dirichlet(alpha[0], n_draw)
    dc = np.log(p)
    assert np.all(np.abs(gotc - dc.mean(axis=0)) <= 3 * dc.std(axis=0) / np.sqrt(n_draw))


def test_e13_t05a_normal_wishart_expected_log_prob_matches_monte_carlo() -> None:
    """Zero statistics give back the Normal-Wishart prior."""
    from hsmmkit.model.emission.gaussian import GaussianFamily

    rng = np.random.default_rng(1)
    w = np.array([[0.4, -0.05], [-0.05, 0.3]])
    q = {
        "m": jnp.asarray([[1.0, 0.4]]),
        "kappa": jnp.asarray([7.5]),
        "W": jnp.asarray(w[None]),
        "nu": jnp.asarray([9.0]),
    }
    y = np.array([[0.2, 0.1], [1.5, -0.3]])
    got = np.asarray(
        GaussianFamily(covariance="full").expected_log_prob(jnp.asarray(y), jnp.ones(2), q, None)
    )[:, 0]
    n_draw = 200_000
    lam = stats.wishart(df=9.0, scale=w).rvs(n_draw, random_state=rng)
    sigma = np.linalg.inv(7.5 * lam)
    chol = np.linalg.cholesky(sigma)
    mu = np.array([1.0, 0.4]) + np.einsum("nij,nj->ni", chol, rng.normal(size=(n_draw, 2)))
    draws = np.stack([_mvn(y[t], mu, lam) for t in range(2)])
    assert np.all(np.abs(got - draws.mean(axis=1)) <= 3 * draws.std(axis=1) / np.sqrt(n_draw))


def _mvn(y: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """`log N(y | mu, lam^-1)` for each draw."""
    d = y[None] - mu
    quad = np.einsum("ni,nij,nj->n", d, lam, d)
    return 0.5 * np.linalg.slogdet(lam)[1] - np.log(2 * np.pi) - 0.5 * quad
