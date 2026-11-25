"""E11-T04: the discrete families pass the family contract suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from hsmmkit.model.emission.bernoulli import BernoulliFamily
from hsmmkit.model.emission.categorical import CategoricalFamily
from hsmmkit.model.emission.poisson import PoissonFamily
from hsmmkit.testing import family_contract
from hsmmkit.testing.family_contract import (  # noqa: F401
    FamilyCase,
    pytest_generate_tests,
    test_family_contract,
)

K = 4


def _eta(key: jax.Array, n: int, width: int, k: int) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(key)
    theta = {"emission.eta": 0.5 * jax.random.normal(k1, (n, width)) + 0.5}
    if k:
        theta["emission.gamma"] = 0.4 * jax.random.normal(k2, (n, width, k))
    return theta


def _cov(key: jax.Array, t: int) -> jax.Array:
    return jax.random.normal(key, (t, 1))


def _poisson_data(key: jax.Array, t: int, m: int) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    k1, k2, k3 = jax.random.split(key, 3)
    dt = jax.random.uniform(k2, (t,), minval=0.5, maxval=2.0)
    return jax.random.poisson(k1, 3.0, (t, m)).astype(jnp.float64), dt, _cov(k3, t)


def _bernoulli_data(key: jax.Array, t: int, m: int) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    k1, k2 = jax.random.split(key)
    return jax.random.bernoulli(k1, 0.4, (t, m)).astype(jnp.float64), jnp.ones((t,)), _cov(k2, t)


def _cat_data(key: jax.Array, t: int, m: int) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    k1, k2 = jax.random.split(key)
    del m
    return jax.random.randint(k1, (t, 1), 0, K).astype(jnp.float64), jnp.ones((t,)), _cov(k2, t)


def _poisson_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del z
    lam = np.exp(np.asarray(theta["emission.eta"]))[None] * np.asarray(dt)[:, None, None]
    return stats.poisson.logpmf(np.asarray(y)[:, None, :], lam).sum(-1)


def _bernoulli_ref(
    y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None
) -> np.ndarray:
    del dt, z
    p = 1.0 / (1.0 + np.exp(-np.asarray(theta["emission.eta"])))[None]
    return stats.bernoulli.logpmf(np.asarray(y)[:, None, :], p).sum(-1)


def _cat_theta(key: jax.Array, n: int, width: int, k: int) -> dict[str, jax.Array]:
    return _eta(key, n, K - 1, k)


def _cat_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del dt, z
    eta = np.asarray(theta["emission.eta"])
    logits = np.concatenate([np.zeros((eta.shape[0], 1)), eta], axis=1)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    yy = np.asarray(y)[:, 0].astype(int)
    return np.stack([stats.multinomial(1, p[i]).logpmf(np.eye(K)[yy]) for i in range(p.shape[0])], axis=1)


family_contract.register_case(
    FamilyCase(
        name="poisson",
        factory=PoissonFamily,
        theta_factory=_eta,
        data_factory=_poisson_data,
        scipy_logpdf=_poisson_ref,
        covariate_factory=lambda: PoissonFamily(mean_idx=(0,)),
    )
)
family_contract.register_case(
    FamilyCase(
        name="bernoulli",
        factory=BernoulliFamily,
        theta_factory=_eta,
        data_factory=_bernoulli_data,
        scipy_logpdf=_bernoulli_ref,
        covariate_factory=lambda: BernoulliFamily(mean_idx=(0,)),
    )
)
family_contract.register_case(
    FamilyCase(
        name="categorical",
        factory=lambda: CategoricalFamily(n_categories=K),
        theta_factory=_cat_theta,
        data_factory=_cat_data,
        scipy_logpdf=_cat_ref,
        covariate_factory=lambda: CategoricalFamily(n_categories=K, mean_idx=(0,)),
        n_series=1,
    )
)
