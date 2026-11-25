"""E11-T08: `product-gauss-full-block` and `product-mixed` pass the family contract suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from hsmmkit.model.dists import pack_lower, unpack_lower
from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.emission.poisson import PoissonFamily
from hsmmkit.model.emission.product import BlockProductEmission
from hsmmkit.testing import family_contract
from hsmmkit.testing.family_contract import (  # noqa: F401
    FamilyCase,
    pytest_generate_tests,
    test_family_contract,
)


def _chol(key: jax.Array, n: int, m: int) -> jax.Array:
    k1, k2 = jax.random.split(key)
    a = 0.1 * jnp.tril(jax.random.normal(k1, (n, m, m)), -1)
    return pack_lower(a + jnp.eye(m) * (0.3 + 0.3 * jax.random.uniform(k2, (n, 1, 1))))


# --- a full-covariance block on columns (0, 1) and a one-series full block on column 2 ----------------


def _gauss_factory() -> BlockProductEmission:
    full = GaussianFamily(covariance="full")
    return BlockProductEmission.of((full, full), ((0, 1), (2,)))


def _gauss_theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    del k, m
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "emission.b0.mean": 0.3 * jax.random.normal(k1, (n, 2)),
        "emission.b0.chol": _chol(k2, n, 2),
        "emission.b1.mean": 0.3 * jax.random.normal(k3, (n, 1)),
        "emission.b1.chol": _chol(k4, n, 1),
    }


def _gauss_data(key: jax.Array, t: int, m: int) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    return jax.random.normal(key, (t, m)), jnp.ones(t), None


def _gauss_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del dt, z
    ya = np.asarray(y)
    out = np.zeros((ya.shape[0], np.asarray(theta["emission.b0.mean"]).shape[0]))
    for b, cols in enumerate(((0, 1), (2,))):
        mean = np.asarray(theta[f"emission.b{b}.mean"])
        chol = np.asarray(unpack_lower(theta[f"emission.b{b}.chol"], len(cols)))
        for i in range(mean.shape[0]):
            out[:, i] += stats.multivariate_normal(mean[i], chol[i] @ chol[i].T).logpdf(ya[:, list(cols)])
    return out


# --- a Gaussian block on column 0 and a Poisson block on column 1 ----------------------------------------


def _mixed_factory() -> BlockProductEmission:
    return BlockProductEmission.of(
        (GaussianFamily(covariance="full"), PoissonFamily(scale_by_dt=False)), ((0,), (1,))
    )


def _mixed_theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    del k, m
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "emission.b0.mean": 0.3 * jax.random.normal(k1, (n, 1)),
        "emission.b0.chol": _chol(k2, n, 1),
        "emission.b1.eta": 0.4 * jax.random.normal(k3, (n, 1)) + 1.0,
    }


def _mixed_data(key: jax.Array, t: int, m: int) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    k1, k2 = jax.random.split(key)
    y = jnp.stack(
        [jax.random.normal(k1, (t,)), jax.random.poisson(k2, 3.0, (t,)).astype(jnp.float64)], axis=1
    )
    del m
    return y, jnp.ones(t), None


def _mixed_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del dt, z
    ya = np.asarray(y)
    mean, chol = np.asarray(theta["emission.b0.mean"]), np.asarray(unpack_lower(theta["emission.b0.chol"], 1))
    lam = np.exp(np.asarray(theta["emission.b1.eta"]))
    return np.stack(
        [
            stats.norm.logpdf(ya[:, 0], mean[i, 0], chol[i, 0, 0]) + stats.poisson.logpmf(ya[:, 1], lam[i, 0])
            for i in range(mean.shape[0])
        ],
        axis=1,
    )


family_contract.register_case(
    FamilyCase(
        name="product-gauss-full-block",
        factory=_gauss_factory,
        theta_factory=_gauss_theta,
        data_factory=_gauss_data,
        scipy_logpdf=_gauss_ref,
        n_series=3,
    )
)
family_contract.register_case(
    FamilyCase(
        name="product-mixed",
        factory=_mixed_factory,
        theta_factory=_mixed_theta,
        data_factory=_mixed_data,
        scipy_logpdf=_mixed_ref,
        n_series=2,
    )
)
_ = family_contract
