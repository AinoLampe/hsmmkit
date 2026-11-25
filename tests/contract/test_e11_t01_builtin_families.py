"""E11-T01: the built-in emission families pass the family contract suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.emission.product import ProductEmission
from hsmmkit.model.emission.student_t import MultivariateStudentT, StudentTFamily
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


def _gbm_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    """Closed form of the GBM log-return density, N((mu - s^2/2) dt, s^2 dt), independent of the family."""
    del z
    mu, s, d = np.asarray(theta["emission.mean"]), np.asarray(theta["emission.scale"]), np.asarray(dt)
    mean = (mu[None] - 0.5 * s[None] ** 2) * d[:, None, None]
    sd = s[None] * np.sqrt(d)[:, None, None]
    return stats.norm.logpdf(np.asarray(y)[:, None, :], mean, sd).sum(-1)


family_contract.register_case(
    FamilyCase(
        name="gaussian-gbm-options",
        factory=lambda: GaussianFamily(scale_by_dt=True, ito_correction=True),
        theta_factory=_gauss_theta,
        data_factory=_real_data,
        scipy_logpdf=_gbm_ref,
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


def test_e11_t01_student_t_moments_edge_cases() -> None:
    """Variance is `inf` for `nu <= 2` and the mean is `nan` for `nu <= 1`; otherwise the closed forms."""
    fam = StudentTFamily()
    theta = {
        "emission.m": jnp.array([[0.1], [0.2], [0.3]]),
        "emission.s": jnp.full((3, 1), 0.5),
        "emission.nu": jnp.array([0.9, 1.5, 5.0]),
    }
    regime, dt = jnp.array([0, 1, 2]), jnp.ones(3)
    mean, var = fam.moments(regime, dt, theta, None)
    assert np.isnan(mean[0, 0])
    assert np.isinf(var[0, 0])
    assert np.isclose(mean[1, 0], 0.2)
    assert np.isinf(var[1, 0])
    assert np.isclose(mean[2, 0], 0.3)
    assert np.isclose(var[2, 0], 0.25 * 5.0 / 3.0)


def _full_theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    from hsmmkit.model.dists import pack_lower

    k1, k2, k3 = jax.random.split(key, 3)
    a = jax.random.normal(k2, (n, m, m)) * 0.1
    chol = jnp.tril(a) + jnp.eye(m) * (0.2 + jax.random.uniform(k3, (n, 1, 1)) * 0.2)
    theta = {"emission.mean": 0.05 * jax.random.normal(k1, (n, m)), "emission.chol": pack_lower(chol)}
    if k:
        theta["emission.gamma"] = 0.5 * jax.random.normal(k1, (n, k))
        theta["emission.delta"] = 0.3 * jax.random.normal(k2, (n, k))
    return theta


def _full_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del dt, z
    from hsmmkit.model.dists import unpack_lower

    chol = np.asarray(unpack_lower(theta["emission.chol"], y.shape[1]))
    return np.stack(
        [
            stats.multivariate_normal(np.asarray(theta["emission.mean"][i]), chol[i] @ chol[i].T).logpdf(
                np.asarray(y)
            )
            for i in range(chol.shape[0])
        ],
        axis=1,
    )


family_contract.register_case(
    FamilyCase(
        name="gaussian-full",
        factory=lambda: GaussianFamily(covariance="full"),
        theta_factory=_full_theta,
        data_factory=_real_data,
        scipy_logpdf=_full_ref,
        covariate_factory=lambda: GaussianFamily(covariance="full", mean_idx=(0,), scale_idx=(0,)),
    )
)


def _mvt_theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    theta = _full_theta(key, n, m, k)
    theta["emission.nu"] = jnp.array([6.0, 12.0][:n])
    return theta


def _mvt_diag_theta(key: jax.Array, n: int, m: int, k: int) -> dict[str, jax.Array]:
    theta = _gauss_theta(key, n, m, k)
    return {
        "emission.mean": theta["emission.mean"],
        "emission.scale": theta["emission.scale"],
        "emission.nu": jnp.array([6.0, 12.0][:n]),
        **{k_: v for k_, v in theta.items() if k_ in ("emission.gamma", "emission.delta")},
    }


def _mvt_ref(y: jax.Array, dt: jax.Array, theta: dict[str, jax.Array], z: jax.Array | None) -> np.ndarray:
    del dt, z
    from hsmmkit.model.dists import unpack_lower

    m = y.shape[1]
    out = []
    for i in range(theta["emission.mean"].shape[0]):
        if "emission.chol" in theta:
            chol = np.asarray(unpack_lower(theta["emission.chol"], m))[i]
            cov = chol @ chol.T
        else:
            cov = np.diag(np.asarray(theta["emission.scale"])[i] ** 2)
        mvt = stats.multivariate_t(
            np.asarray(theta["emission.mean"][i]), cov, df=float(theta["emission.nu"][i])
        )
        out.append(mvt.logpdf(np.asarray(y)))
    return np.stack(out, axis=1)


family_contract.register_case(
    FamilyCase(
        name="student_t-mv-diag",
        factory=lambda: MultivariateStudentT(covariance="diag"),
        theta_factory=_mvt_diag_theta,
        data_factory=_real_data,
        scipy_logpdf=_mvt_ref,
        covariate_factory=lambda: MultivariateStudentT(covariance="diag", mean_idx=(0,), scale_idx=(0,)),
    )
)
family_contract.register_case(
    FamilyCase(
        name="student_t-mv-full",
        factory=lambda: MultivariateStudentT(covariance="full"),
        theta_factory=_mvt_theta,
        data_factory=_real_data,
        scipy_logpdf=_mvt_ref,
        covariate_factory=lambda: MultivariateStudentT(covariance="full", mean_idx=(0,), scale_idx=(0,)),
    )
)
