"""E11-T01: the built-in emission families pass the family contract suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.emission.product import ProductEmission
from hsmmkit.model.emission.student_t import StudentTFamily
from hsmmkit.testing import family_contract
from hsmmkit.testing.family_contract import (  # noqa: F401
    FamilyCase,
    pytest_generate_tests,
    test_family_contract,
)

M2 = 2


def _gauss_theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    theta = {
        "emission.mean": 0.05 * jax.random.normal(k1, (n, m)),
        "emission.scale": 0.1 + jax.random.uniform(k2, (n, m)) * 0.2,
    }
    if k:
        theta["emission.gamma"] = 0.5 * jax.random.normal(k3, (n, k))
        theta["emission.delta"] = 0.3 * jax.random.normal(k4, (n, k))
    return theta


def _real_data(key: jax.Array, t: int, m: int) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    k1, k2 = jax.random.split(key)
    y = 0.1 * jax.random.normal(k1, (t, m))
    dt = jnp.full((t,), 1.0 / 252.0)
    z = jax.random.normal(k2, (t, 1))
    return y, dt, z


def _gauss_ref(fam: GaussianFamily):  # noqa: ANN202
    def ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
        mean, sd = fam._mean_sd(dt, theta, z)  # the family's own location / scale
        return stats.norm.logpdf(np.asarray(y)[:, None, :], np.asarray(mean), np.asarray(sd)).sum(-1)

    return ref


def _student_theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    theta = _gauss_theta(key, n, m, k)
    return {
        "emission.m": theta["emission.mean"],
        "emission.s": theta["emission.scale"],
        "emission.nu": jnp.array([4.5, 9.0][:n]),
        **{k_: v for k_, v in theta.items() if k_ in ("emission.gamma", "emission.delta")},
    }


def _student_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del dt, z
    return stats.t.logpdf(
        np.asarray(y)[:, None, :],
        np.asarray(theta["emission.nu"])[None, :, None],
        np.asarray(theta["emission.m"])[None],
        np.asarray(theta["emission.s"])[None],
    ).sum(-1)


family_contract.register_case(
    FamilyCase(
        name="gaussian-diag",
        factory=GaussianFamily,
        theta_factory=_gauss_theta,
        data_factory=_real_data,
        scipy_logpdf=_gauss_ref(GaussianFamily()),
        covariate_factory=lambda: GaussianFamily(mean_idx=(0,), scale_idx=(0,)),
    )
)
_GBM_OPTIONS = GaussianFamily(scale_by_dt=True, ito_correction=True)
family_contract.register_case(
    FamilyCase(
        name="gaussian-gbm-options",
        factory=lambda: GaussianFamily(scale_by_dt=True, ito_correction=True),
        theta_factory=_gauss_theta,
        data_factory=_real_data,
        scipy_logpdf=_gauss_ref(_GBM_OPTIONS),
    )
)
family_contract.register_case(
    FamilyCase(
        name="student_t-uv",
        factory=StudentTFamily,
        theta_factory=_student_theta,
        data_factory=_real_data,
        scipy_logpdf=_student_ref,
        n_series=1,
    )
)
family_contract.register_case(
    FamilyCase(
        name="product-gaussian",
        factory=lambda: ProductEmission.of(GaussianFamily(), M2),
        theta_factory=_gauss_theta,
        data_factory=_real_data,
        scipy_logpdf=_gauss_ref(GaussianFamily()),
    )
)
