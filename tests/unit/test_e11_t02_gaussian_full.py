"""E11-T02: `gaussian` with a `full` covariance (Cholesky parameterization)."""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, optimize, stats

from hsmmkit.config import gates
from hsmmkit.config.schema import parse_project_config
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.dists import lkj_logpdf, mvn_logpdf, pack_lower, packed_size, unpack_lower
from hsmmkit.model.emission.gaussian import GaussianFamily, NormalWishart
from hsmmkit.model.params import CholeskyDiagPrior, LkjPrior, LogNormalPrior, ParamManager
from tests.helpers.core_config import core_raw

N = 2


def _random_theta(m: int, seed: int = 0) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    chol = np.zeros((N, m, m))
    for i in range(N):
        a = rng.normal(size=(m, m))
        chol[i] = np.linalg.cholesky(a @ a.T + m * np.eye(m))
    return {
        "emission.mean": jnp.asarray(rng.normal(size=(N, m))),
        "emission.chol": pack_lower(jnp.asarray(chol)),
    }


@pytest.mark.parametrize("m", [2, 3, 5])
@pytest.mark.parametrize("scaled", [False, True])
def test_e11_t02_log_prob_matches_scipy_multivariate_normal(m: int, scaled: bool) -> None:
    """AC1: the joint log-density equals `multivariate_normal.logpdf` (1e-10), with and without scaling."""
    theta = _random_theta(m)
    rng = np.random.default_rng(1)
    t = 30
    y = jnp.asarray(rng.normal(size=(t, m)))
    dt = jnp.asarray(rng.uniform(0.5, 2.0, size=t))
    fam = GaussianFamily(covariance="full", scale_by_dt=scaled)
    got = np.asarray(fam.log_prob(y, dt, theta, None))
    chol = np.asarray(unpack_lower(theta["emission.chol"], m))
    want = np.empty((t, N))
    for i in range(N):
        cov = chol[i] @ chol[i].T
        for k in range(t):
            s = float(dt[k]) if scaled else 1.0
            want[k, i] = stats.multivariate_normal(np.asarray(theta["emission.mean"][i]) * s, cov * s).logpdf(
                y[k]
            )
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


def test_e11_t02_mvn_logpdf_broadcasts() -> None:
    """`mvn_logpdf` broadcasts a batch of means against a batch of factors."""
    chol = jnp.asarray(np.linalg.cholesky(np.array([[2.0, 0.3], [0.3, 1.0]])))
    y = jnp.asarray([[0.1, 0.2], [0.0, -1.0]])
    got = mvn_logpdf(y[:, None, :], jnp.zeros((1, 3, 2)), chol)
    assert got.shape == (2, 3)
    want = stats.multivariate_normal(np.zeros(2), np.asarray(chol @ chol.T)).logpdf(np.asarray(y))
    np.testing.assert_allclose(np.asarray(got[:, 0]), want, atol=1e-12)


def test_e11_t02_ito_correction_and_series_marginals() -> None:
    """Ito: mean `mu - diag(Sigma)/2`; the marginal variances equal `diag(L L^T)` (times `s`)."""
    m = 3
    theta = _random_theta(m)
    fam = GaussianFamily(covariance="full", scale_by_dt=True, ito_correction=True)
    dt = jnp.asarray([0.5, 1.0])
    mean, var = fam.moments(jnp.asarray([0, 1]), dt, theta, None)
    chol = np.asarray(unpack_lower(theta["emission.chol"], m))
    for k, (i, s) in enumerate(((0, 0.5), (1, 1.0))):
        diag = np.diag(chol[i] @ chol[i].T)
        np.testing.assert_allclose(
            np.asarray(mean[k]), (np.asarray(theta["emission.mean"][i]) - 0.5 * diag) * s
        )
        np.testing.assert_allclose(np.asarray(var[k]), diag * s)


def test_e11_t02_samples_have_the_declared_covariance() -> None:
    """The sample covariance of one regime matches `L L^T` within sampling error."""
    m = 3
    theta = _random_theta(m)
    fam = GaussianFamily(covariance="full")
    n = 40_000
    s = np.asarray(fam.sample(jax.random.key(0), jnp.zeros(n, dtype=jnp.int32), jnp.ones(n), theta, None))
    chol = np.asarray(unpack_lower(theta["emission.chol"], m))[0]
    np.testing.assert_allclose(
        np.cov(s.T), chol @ chol.T, atol=6 * np.abs(chol @ chol.T).max() / math.sqrt(n) * 3
    )


def test_e11_t02_m_step_equals_the_numerical_optimum() -> None:
    """AC3: `m_step` is the maximizer of `sum_t gamma_ti log f_i(y_t)` (scipy BFGS agrees within 1e-8)."""
    m, t = 3, 400
    rng = np.random.default_rng(3)
    y = jnp.asarray(rng.normal(size=(t, m)) @ np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, -0.3, 0.8]]))
    gamma = jnp.asarray(rng.dirichlet(np.ones(N), size=t))
    dt = jnp.ones(t)
    fam = GaussianFamily(covariance="full")
    theta_hat = fam.m_step(fam.sufficient_stats(y, dt, gamma))
    assert theta_hat is not None
    for i in range(N):

        def nll(x: np.ndarray, i: int = i) -> float:
            th = {"emission.mean": jnp.asarray(x[None, :m]), "emission.chol": jnp.asarray(x[None, m:])}
            return float(-jnp.sum(gamma[:, i] * fam.log_prob(y, dt, th, None)[:, 0]))

        x0 = np.concatenate(
            [np.asarray(theta_hat["emission.mean"][i]), np.asarray(theta_hat["emission.chol"][i])]
        )
        res = optimize.minimize(nll, x0 + 0.05, method="BFGS", options={"gtol": 1e-10})
        rel = np.abs(res.x - x0) / np.maximum(np.abs(x0), 1.0)
        assert rel.max() <= 1e-4  # parameters of the numerical optimum
        assert abs(nll(x0) - res.fun) / abs(res.fun) <= 1e-8  # objective value


def test_e11_t02_m_step_is_absent_with_covariates_or_ito() -> None:
    """Covariate links and the Ito correction have no closed form: `m_step` answers `None`."""
    y, gamma, dt = jnp.ones((5, 2)), jnp.full((5, 2), 0.5), jnp.ones(5)
    assert (
        GaussianFamily(covariance="full", ito_correction=True).m_step(
            GaussianFamily(covariance="full", ito_correction=True).sufficient_stats(y, dt, gamma)
        )
        is None
    )
    fam = GaussianFamily(covariance="full", mean_idx=(0,))
    assert fam.m_step(fam.sufficient_stats(y, dt, gamma)) is None
    assert GaussianFamily().m_step(None) is None


def test_e11_t02_normal_wishart_zero_data_returns_the_prior_and_updates() -> None:
    """The posterior with no data is the prior; with data `kappa` and `nu` grow by `n`."""
    prior = GaussianFamily(covariance="full").conjugate_prior()
    assert isinstance(prior, NormalWishart)
    zero = prior.update(prior.zero_stats(2, 3))
    np.testing.assert_allclose(np.asarray(zero["m"]), 0.0)
    np.testing.assert_allclose(np.asarray(zero["kappa"]), prior.kappa0)
    np.testing.assert_allclose(np.asarray(zero["W"]), np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(np.asarray(zero["nu"]), 5.0)
    y = jnp.asarray(np.random.default_rng(0).normal(size=(50, 3)))
    gamma = jnp.ones((50, 1))
    fam = GaussianFamily(covariance="full")
    post = prior.update(fam.sufficient_stats(y, jnp.ones(50), gamma))
    assert float(post["kappa"][0]) == pytest.approx(prior.kappa0 + 50)
    assert float(post["nu"][0]) == pytest.approx(5.0 + 50)
    assert GaussianFamily().conjugate_prior() is None


def test_e11_t02_lkj_is_uniform_at_eta_one_and_matches_numerical_normalization() -> None:
    """AC (dists): `log c_3(1) = log(pi^2 / 2)` (volume of 3x3 correlation matrices); M=2 integrates to 1."""
    c3 = jnp.eye(3)
    assert float(lkj_logpdf(c3, 1.0)) == pytest.approx(-math.log(math.pi**2 / 2), abs=1e-12)
    for eta in (0.7, 1.0, 3.0):

        def dens(r: float, eta: float = eta) -> float:
            c = np.array([[1.0, 0.0], [r, math.sqrt(1 - r * r)]])
            return float(np.exp(lkj_logpdf(jnp.asarray(c), eta)))

        # density over R = [[1, r], [r, 1]] integrates over r in (-1, 1) to 1
        area, _ = integrate.quad(dens, -1.0, 1.0)
        assert area == pytest.approx(1.0, rel=1e-8)


def test_e11_t02_priors_on_the_cholesky_parameter() -> None:
    """`CholeskyDiagPrior` evaluates only the diagonal; `LkjPrior` depends on the correlation only."""
    packed = jnp.asarray([[2.0, 0.3, 1.5], [1.0, -0.2, 0.5]])
    base = LogNormalPrior(mean_log=0.0, sd=1.0)
    got = CholeskyDiagPrior(base).log_prob(packed)
    want = base.log_prob(jnp.asarray([2.0, 1.5, 1.0, 0.5]))
    assert float(got) == pytest.approx(float(want))
    scaled = packed * jnp.asarray([3.0, 3.0, 3.0])  # scaling L leaves the correlation unchanged
    assert float(LkjPrior(eta=2.0).log_prob(scaled)) == pytest.approx(
        float(LkjPrior(eta=2.0).log_prob(packed))
    )


def _full_cfg(tmp_path: Path):  # noqa: ANN202
    raw = core_raw(tmp_path)
    raw["data"]["source"]["observation_columns"] = ["a", "b"]
    raw["model"]["emission"] = {
        "family": "gaussian",
        "covariance": "full",
        "params": {
            "mean": {"mode": "free", "init": [0.0, 0.1]},
            "chol": {"mode": "free", "init": [0.5, 0.1, 0.5], "prior": {"type": "lkj", "eta": 2.0}},
        },
    }
    return parse_project_config(raw)


def test_e11_t02_config_builds_and_the_gate_is_gone(tmp_path: Path) -> None:
    """AC5: `covariance: full` is no longer gated and builds a joint (non-product) emission."""
    cfg = _full_cfg(tmp_path)
    assert "model.emission.covariance" not in gates.GATES
    assert gates.used_gates(cfg) == []
    spec = build_model_spec_from(cfg, BuildInputs(n_series=2, cov_names={}, clock="steps", nominal_dt=1.0))
    assert isinstance(spec.emission, GaussianFamily)
    assert spec.emission.joint
    names = {p.name: p for p in spec.param_specs if p.name.startswith("emission.")}
    assert names["emission.chol"].shape == (2, packed_size(2))
    assert names["emission.chol"].domain == "cholesky"
    np.testing.assert_allclose(names["emission.chol"].init[0], [0.5, 0.1, 0.5])


def test_e11_t02_full_covariance_params_must_be_named_chol(tmp_path: Path) -> None:
    """The family asks for `params.chol` (not `scale`) under `covariance: full`."""
    raw = core_raw(tmp_path)
    raw["model"]["emission"]["covariance"] = "full"
    from hsmmkit.utils.errors import ConfigError

    with pytest.raises(ConfigError, match=r"params\.chol"):
        build_model_spec_from(
            parse_project_config(raw), BuildInputs(n_series=1, cov_names={}, clock="steps", nominal_dt=1.0)
        )


def test_e11_t02_param_manager_round_trips_the_cholesky_factor() -> None:
    """The `cholesky` domain is accepted by the `ParamManager` and keeps `L` valid for every `u`."""
    fam = GaussianFamily(covariance="full")
    mgr = ParamManager(fam.param_specs(2, 3, ()))
    u = jnp.asarray(np.random.default_rng(0).normal(size=mgr.n_free))
    theta = mgr.unflatten(u)
    chol = unpack_lower(theta["emission.chol"], 3)
    assert bool(jnp.all(jnp.diagonal(chol, axis1=-2, axis2=-1) > 0))
    np.testing.assert_allclose(np.asarray(mgr.flatten(theta)), np.asarray(u), atol=1e-12)
