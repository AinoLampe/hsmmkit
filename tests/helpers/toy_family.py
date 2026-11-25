# ruff: noqa: D102, D103
"""A minimal count emission (`poisson_toy`) with every optional hook, for the family-contract meta tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from hsmmkit.model.emission.base import BaseEmission
from hsmmkit.model.params.spec import ParamSpec
from hsmmkit.testing.family_contract import FamilyCase


@dataclass(frozen=True)
class PoissonToy(BaseEmission):
    """Independent Poisson counts per series; `offset` corrupts the density (a wrong normalizing constant)."""

    name: str = "poisson_toy"
    native_time: Literal["continuous", "discrete"] = "discrete"
    support: Literal["real", "count", "binary", "categorical"] = "count"
    scale_by_dt: bool = False
    offset: float = 0.0
    bad_moments: bool = False

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        del cov_names
        return (
            ParamSpec(
                name="emission.rate",
                shape=(n_states, n_series),
                domain="positive",
                mode="free",
                init=np.full((n_states, n_series), 3.0),
            ),
        )

    def log_prob(self, y: Any, dt: Any, theta: dict[str, Any], z: Any) -> Any:  # noqa: ANN401
        del dt, z
        lam = theta["emission.rate"][None]  # (1, N, M)
        return jnp.sum(jax.scipy.stats.poisson.logpmf(y[:, None, :], lam), axis=-1) + self.offset

    def sample(self, key: Any, regime: Any, dt: Any, theta: dict[str, Any], z: Any) -> Any:  # noqa: ANN401
        del dt, z
        return jax.random.poisson(key, theta["emission.rate"][regime]).astype(jnp.float64)

    def init_carry(self) -> object:
        return None

    def step_carry(self, carry: object, y_t: Any, theta: dict[str, Any]) -> object:  # noqa: ANN401
        del y_t, theta
        return carry

    def sufficient_stats(self, y: Any, dt: Any, gamma: Any, z: Any = None) -> dict[str, Any]:  # noqa: ANN401
        del dt, z
        return {"w": jnp.sum(gamma, axis=0), "wy": gamma.T @ y}

    def m_step(self, stats_: Any) -> dict[str, Any]:  # noqa: ANN401
        return {"emission.rate": stats_["wy"] / stats_["w"][:, None]}

    def moments(self, regime: Any, dt: Any, theta: dict[str, Any], z: Any) -> tuple[Any, Any]:  # noqa: ANN401
        del dt, z
        lam = theta["emission.rate"][regime]
        return lam, (2.0 * lam if self.bad_moments else lam)


def toy_theta(key: Any, n: int, m: int, k: int) -> dict[str, Any]:  # noqa: ANN401
    del k
    return {"emission.rate": 1.0 + 4.0 * jax.random.uniform(key, (n, m))}


def toy_data(key: Any, t: int, m: int) -> tuple[Any, Any, None]:  # noqa: ANN401
    y = jax.random.poisson(key, 3.0, (t, m)).astype(jnp.float64)
    return y, jnp.ones((t,)), None


def toy_ref(y: Any, dt: Any, theta: dict[str, Any], z: Any) -> np.ndarray:  # noqa: ANN401
    del dt, z
    return stats.poisson.logpmf(np.asarray(y)[:, None, :], np.asarray(theta["emission.rate"])[None]).sum(-1)


def toy_case(name: str = "poisson-toy", **family_kwargs: Any) -> FamilyCase:  # noqa: ANN401
    """A case for `PoissonToy(**family_kwargs)`."""
    return FamilyCase(
        name=name,
        factory=lambda: PoissonToy(**family_kwargs),
        theta_factory=toy_theta,
        data_factory=toy_data,
        scipy_logpdf=toy_ref,
    )
