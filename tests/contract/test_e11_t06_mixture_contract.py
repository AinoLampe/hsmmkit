"""E11-T06: `mixture-gaussian-2` and `mixture-poisson-2` pass the family contract suite.

Check 7 uses the EM step at fixed responsibilities (`FamilyCase.m_step_check`): the M-step maximizes the
expected complete-data log-likelihood (stationary base parameters, closed-form weights) and does not decrease
the observed log-likelihood; the generic stationarity test of `sum gamma log f` does not apply to a mixture.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special, stats

from hsmmkit.model.dists import pack_lower, unpack_lower
from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.emission.mixture import MixtureFamily
from hsmmkit.model.emission.poisson import PoissonFamily
from hsmmkit.testing import family_contract
from hsmmkit.testing.family_contract import (  # noqa: F401
    FamilyCase,
    pytest_generate_tests,
    test_family_contract,
)

K = 2


def _weights(key: jax.Array, n: int) -> jax.Array:
    return jax.nn.softmax(jax.random.normal(key, (n, K)), axis=-1)


def _gauss_theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    del k
    k1, k2, k3, k4 = jax.random.split(key, 4)
    a = 0.1 * jnp.tril(jax.random.normal(k2, (n * K, m, m)), -1)
    chol = a + jnp.eye(m) * (0.3 + jax.random.uniform(k3, (n * K, 1, 1)) * 0.3)
    return {
        "emission.mean": 0.5 * jax.random.normal(k1, (n * K, m)),
        "emission.chol": pack_lower(chol),
        "emission.weights": _weights(k4, n),
    }


def _gauss_data(key: jax.Array, t: int, m: int) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    return jax.random.normal(key, (t, m)), jnp.ones(t), None


def _gauss_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del dt, z
    mean = np.asarray(theta["emission.mean"])
    chol = np.asarray(unpack_lower(theta["emission.chol"], mean.shape[1]))
    w = np.asarray(theta["emission.weights"])
    out = np.empty((y.shape[0], w.shape[0]))
    for i in range(w.shape[0]):
        comp = [
            np.log(w[i, c])
            + stats.multivariate_normal(mean[i * K + c], chol[i * K + c] @ chol[i * K + c].T).logpdf(
                np.asarray(y)
            )
            for c in range(K)
        ]
        out[:, i] = special.logsumexp(np.stack(comp), axis=0)
    return out


def _pois_theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    del k
    k1, k2 = jax.random.split(key)
    return {
        "emission.eta": 0.5 * jax.random.normal(k1, (n * K, m)) + 1.0,
        "emission.weights": _weights(k2, n),
    }


def _pois_data(key: jax.Array, t: int, m: int) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    return jax.random.poisson(key, 3.0, (t, m)).astype(jnp.float64), jnp.ones(t), None


def _pois_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del dt, z
    lam = np.exp(np.asarray(theta["emission.eta"]))
    w = np.asarray(theta["emission.weights"])
    out = np.empty((y.shape[0], w.shape[0]))
    for i in range(w.shape[0]):
        comp = [
            np.log(w[i, c]) + stats.poisson.logpmf(np.asarray(y), lam[i * K + c][None]).sum(axis=1)
            for c in range(K)
        ]
        out[:, i] = special.logsumexp(np.stack(comp), axis=0)
    return out


def _em_check(case: FamilyCase) -> None:
    """An EM step: stationary for the expected complete-data log-likelihood, monotone for the observed one."""
    fam = case.factory()
    n = family_contract.N_STATES
    theta = case.theta_factory(jax.random.key(2), n, case.n_series, 0)
    y, dt, z = case.data_factory(jax.random.key(1), family_contract.T_DATA, case.n_series)
    gamma = jnp.asarray(np.random.default_rng(0).dirichlet(np.ones(n), size=y.shape[0]))
    assert fam.sufficient_stats(y, dt, gamma, z) is None  # the responsibilities need theta
    st = fam.sufficient_stats(y, dt, gamma, z, theta=theta)
    hat = fam.m_step(st)
    assert hat is not None
    full = {**theta, **hat}
    resp = st["resp"]
    np.testing.assert_allclose(
        np.asarray(hat["emission.weights"]), np.asarray(st["r_sum"] / st["r_sum"].sum(1, keepdims=True))
    )

    def q(th: dict[str, jax.Array]) -> jax.Array:
        ll = fam.base.log_prob(y, dt, fam._base_theta(th), z).reshape(y.shape[0], n, K)
        return jnp.sum(resp * ll)

    grad = jax.grad(q)(full)
    worst = max(float(jnp.max(jnp.abs(grad[k]))) for k in hat if k != "emission.weights")
    assert worst / max(1.0, abs(float(q(full)))) <= 1e-8, (
        f"{case.name}: EM step is not stationary ({worst:.2e})"
    )

    def observed(th: dict[str, jax.Array]) -> float:
        return float(jnp.sum(gamma * fam.log_prob(y, dt, th, z)))

    assert observed(full) >= observed(theta) - 1e-10, f"{case.name}: the EM step decreased the log-likelihood"


family_contract.register_case(
    FamilyCase(
        name="mixture-gaussian-2",
        factory=lambda: MixtureFamily.of(GaussianFamily(covariance="full"), K),
        theta_factory=_gauss_theta,
        data_factory=_gauss_data,
        scipy_logpdf=_gauss_ref,
        m_step_check=_em_check,
    )
)
family_contract.register_case(
    FamilyCase(
        name="mixture-poisson-2",
        factory=lambda: MixtureFamily.of(PoissonFamily(scale_by_dt=False), K),
        theta_factory=_pois_theta,
        data_factory=_pois_data,
        scipy_logpdf=_pois_ref,
        n_series=1,  # `ProductEmission` has no M-step yet (E11-T08); product base: unit tests
        m_step_check=_em_check,
    )
)
