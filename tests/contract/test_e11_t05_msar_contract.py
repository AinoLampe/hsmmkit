"""E11-T05: `msar-p2` passes the family contract suite (check 7 through the weighted-LS M-step)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from hsmmkit.data.contracts import own_lag_columns
from hsmmkit.model.emission.msar import MSARFamily
from hsmmkit.testing import family_contract
from hsmmkit.testing.family_contract import (  # noqa: F401
    FamilyCase,
    pytest_generate_tests,
    test_family_contract,
)

P = 2


def _theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    del k
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "emission.intercept": 0.2 * jax.random.normal(k1, (n, m)),
        "emission.ar": 0.25 * jax.random.normal(k2, (n, P, m, m)),
        "emission.scale": 0.5 + jax.random.uniform(k3, (n, m)),
    }


def _data(key: jax.Array, t: int, m: int) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """A series with its own-lag block `z (T, p M)` built the way the preparer builds it."""
    k1, k2 = jax.random.split(key)
    y = jax.random.normal(k1, (t, m))
    lags, _ = own_lag_columns(np.asarray(y), np.ones((t, m), bool), P)
    del k2
    return y, jnp.ones(t), jnp.asarray(lags)


def _ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del dt
    assert z is not None
    ya, za = np.asarray(y), np.asarray(z)
    mu, ar, sd = (np.asarray(theta[k]) for k in ("emission.intercept", "emission.ar", "emission.scale"))
    t, m = ya.shape
    out = np.empty((t, mu.shape[0]))
    for i in range(mu.shape[0]):
        mean = mu[i] + np.einsum("plk,tpk->tl", ar[i], za.reshape(t, P, m))
        out[:, i] = stats.norm.logpdf(ya, mean, sd[i]).sum(axis=1)
    return out


family_contract.register_case(
    FamilyCase(
        name="msar-p2",
        factory=lambda: MSARFamily(p=P, n_series=2),
        theta_factory=_theta,
        data_factory=_data,
        scipy_logpdf=_ref,
        n_series=2,
    )
)
