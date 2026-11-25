"""E11-T03: the multivariate Student-t emission (diagonal and full scale matrix)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from hsmmkit.config.schema import parse_project_config
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.dists import mvt_logpdf, pack_lower, unpack_lower
from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.emission.student_t import MultivariateStudentT, StudentTFamily
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw

N = 2


def _theta(m: int, full: bool, seed: int = 0) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    theta: dict[str, jax.Array] = {
        "emission.mean": jnp.asarray(rng.normal(size=(N, m))),
        "emission.nu": jnp.asarray([4.5, 9.0]),
    }
    if full:
        chol = np.zeros((N, m, m))
        for i in range(N):
            a = rng.normal(size=(m, m))
            chol[i] = np.linalg.cholesky(a @ a.T + m * np.eye(m))
        theta["emission.chol"] = pack_lower(jnp.asarray(chol))
    else:
        theta["emission.scale"] = jnp.asarray(rng.uniform(0.5, 2.0, size=(N, m)))
    return theta


def _cov(theta: dict[str, jax.Array], i: int, m: int) -> np.ndarray:
    if "emission.chol" in theta:
        chol = np.asarray(unpack_lower(theta["emission.chol"], m))[i]
        return chol @ chol.T
    return np.diag(np.asarray(theta["emission.scale"])[i] ** 2)


@pytest.mark.parametrize("m", [2, 3])
@pytest.mark.parametrize("full", [False, True])
def test_e11_t03_log_prob_matches_scipy_multivariate_t(m: int, full: bool) -> None:
    """AC1: equal to `multivariate_t.logpdf` within 1e-10, diag and full scale matrices."""
    theta = _theta(m, full)
    y = jnp.asarray(np.random.default_rng(1).normal(size=(25, m)))
    fam = MultivariateStudentT(covariance="full" if full else "diag")
    got = np.asarray(fam.log_prob(y, jnp.ones(25), theta, None))
    for i in range(N):
        want = stats.multivariate_t(
            np.asarray(theta["emission.mean"])[i], _cov(theta, i, m), df=float(theta["emission.nu"][i])
        ).logpdf(np.asarray(y))
        np.testing.assert_allclose(got[:, i], want, rtol=0, atol=1e-10)


def test_e11_t03_large_nu_approaches_the_gaussian() -> None:
    """AC2: `nu = 1e8` is within 1e-6 of the Gaussian log-density with the same covariance."""
    m = 3
    theta = _theta(m, True)
    theta["emission.nu"] = jnp.full((N,), 1e8)
    y = jnp.asarray(np.random.default_rng(2).normal(size=(20, m)))
    dt = jnp.ones(20)
    t_lp = MultivariateStudentT(covariance="full").log_prob(y, dt, theta, None)
    g_lp = GaussianFamily(covariance="full").log_prob(
        y, dt, {k: v for k, v in theta.items() if k != "emission.nu"}, None
    )
    np.testing.assert_allclose(np.asarray(t_lp), np.asarray(g_lp), rtol=0, atol=1e-6)


def test_e11_t03_mvt_logpdf_with_m1_matches_the_univariate_density() -> None:
    """`M = 1` reduces to the univariate Student-t (scipy `t`)."""
    y = jnp.asarray([[0.3], [-1.2]])
    got = mvt_logpdf(y, jnp.asarray([[0.1]]), jnp.asarray([[[0.7]]]), jnp.asarray([5.0]))
    want = stats.t.logpdf(np.asarray(y)[:, 0], 5.0, 0.1, 0.7)
    np.testing.assert_allclose(np.asarray(got), want, atol=1e-12)


def test_e11_t03_dt_scaling_covariates_and_moments() -> None:
    """`scale_by_dt` scales mean and covariance; variance is `nu / (nu - 2) diag(Sigma)`."""
    m = 2
    theta = _theta(m, True)
    fam = MultivariateStudentT(covariance="full", scale_by_dt=True)
    dt = jnp.asarray([0.5, 2.0])
    mean, var = fam.moments(jnp.asarray([0, 1]), dt, theta, None)
    for k, (i, s) in enumerate(((0, 0.5), (1, 2.0))):
        nu = float(theta["emission.nu"][i])
        np.testing.assert_allclose(np.asarray(mean[k]), np.asarray(theta["emission.mean"][i]) * s)
        np.testing.assert_allclose(np.asarray(var[k]), np.diag(_cov(theta, i, m)) * s * nu / (nu - 2))


def test_e11_t03_samples_match_the_declared_scale() -> None:
    """The sample mean and the marginal variance agree with `moments` (large nu keeps the SE small)."""
    m = 2
    theta = _theta(m, True)
    theta["emission.nu"] = jnp.asarray([12.0, 12.0])
    fam = MultivariateStudentT(covariance="full")
    n = 60_000
    s = np.asarray(fam.sample(jax.random.key(0), jnp.zeros(n, dtype=jnp.int32), jnp.ones(n), theta, None))
    mean, var = fam.moments(jnp.zeros(1, dtype=jnp.int32), jnp.ones(1), theta, None)
    np.testing.assert_allclose(
        s.mean(axis=0), np.asarray(mean[0]), atol=5 * np.sqrt(np.asarray(var[0]) / n).max()
    )
    np.testing.assert_allclose(s.var(axis=0), np.asarray(var[0]), rtol=0.06)
    # off-diagonal structure (L L^T, not L^T L): Cov = nu / (nu - 2) Sigma
    np.testing.assert_allclose(
        np.cov(s.T), 12.0 / 10.0 * _cov(theta, 0, m), atol=0.1 * np.abs(_cov(theta, 0, m)).max()
    )


def _cfg(tmp_path: Path, **emission: object):  # noqa: ANN202
    raw = core_raw(tmp_path)
    raw["data"]["source"]["observation_columns"] = ["a", "b"]
    raw["model"]["emission"] = {"family": "student_t", **emission}
    return parse_project_config(raw)


def _build(cfg):  # noqa: ANN001, ANN202
    return build_model_spec_from(cfg, BuildInputs(n_series=2, cov_names={}, clock="steps", nominal_dt=1.0))


def test_e11_t03_config_selects_the_multivariate_path(tmp_path: Path) -> None:
    """`mean`/`scale`/`nu` or `covariance: full` select the joint family; `m`/`s` keep the Part I path."""
    diag = _build(
        _cfg(
            tmp_path,
            params={"mean": {"mode": "free"}, "scale": {"mode": "free"}, "nu": {"mode": "tied", "tie": "nu"}},
        )
    )
    assert isinstance(diag.emission, MultivariateStudentT)
    names = {p.name: p for p in diag.param_specs if p.name.startswith("emission.")}
    assert set(names) == {"emission.mean", "emission.scale", "emission.nu"}
    assert names["emission.nu"].tie_group == "nu"
    full = _build(
        _cfg(
            tmp_path,
            covariance="full",
            params={"mean": {"mode": "free"}, "chol": {"mode": "free"}, "nu": {"mode": "free"}},
        )
    )
    assert isinstance(full.emission, MultivariateStudentT)
    assert "emission.chol" in {p.name for p in full.param_specs}
    part1 = _build(
        _cfg(tmp_path, params={"m": {"mode": "free"}, "s": {"mode": "free"}, "nu": {"mode": "free"}})
    )
    assert not isinstance(part1.emission, MultivariateStudentT)


def test_e11_t03_ito_correction_is_rejected(tmp_path: Path) -> None:
    """The Student-t has no Ito term."""
    with pytest.raises(ConfigError, match="Ito"):
        _build(
            _cfg(
                tmp_path,
                ito_correction=True,
                params={"mean": {"mode": "free"}, "scale": {"mode": "free"}, "nu": {"mode": "free"}},
            )
        )


def test_e11_t03_univariate_family_is_unchanged() -> None:
    """The Part I `StudentTFamily` keeps its parameter names and has no `joint` factorization flag."""
    fam = StudentTFamily()
    assert [s.name for s in fam.param_specs(2, 1, ())] == ["emission.m", "emission.s", "emission.nu"]
    assert not getattr(fam, "joint", False)
