"""Shifted-Poisson sojourn on a regular grid (E3-T04): `kappa` in clock units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import special as jsp_special

from regimekit.model.dists import poisson_shifted_logpmf
from regimekit.model.duration.base import Params
from regimekit.model.duration.explicit import ExplicitDurationBase, ParamDef
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array


@dataclass(frozen=True)
class PoissonDuration(ExplicitDurationBase):
    """`d = 1 + Poisson(kappa_i / dt)`, truncated at `D_max` and renormalized; `E[x] = dt + kappa`.

    Entry-time link: `log kappa_i = log kappa_i^0 + c_i . z_entry`.
    """

    name: str = "poisson"
    param_defs: ClassVar[tuple[ParamDef, ...]] = (("kappa", "positive", 0.1),)
    primary: ClassVar[str] = "kappa"

    def _lam(self, theta: Params, adj: Array | None, dt: float) -> Array:
        kappa = theta["duration.kappa"]
        return (kappa if adj is None else kappa * jnp.exp(adj)) / dt

    def _log_pmf_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        d = jnp.arange(1, d_max + 1, dtype=jnp.float64)
        return jax.vmap(lambda lam: poisson_shifted_logpmf(d, lam, d_max))(self._lam(theta, adj, dt))

    def _tail_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        # P(1 + Poisson(lam) > D_max) = P(Poisson(lam) >= D_max) = regularized lower gamma P(D_max, lam)
        return jsp_special.gammainc(float(d_max), self._lam(theta, adj, dt))

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`(1 + kappa / dt) * dt` (clock units)."""
        return (1.0 + theta["duration.kappa"] / nominal_dt) * nominal_dt

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """`kappa = (mean_steps - 1) * nominal_dt`; the mean must exceed one step."""
        if bool(np.any(np.asarray(mean_steps) <= 1.0)):
            raise ConfigError(
                "poisson duration needs a mean sojourn > 1 step (d = 1 + Poisson)",
                path="model.duration.params.kappa.init_mean_sojourn_steps",
            )
        return {"duration.kappa": (mean_steps - 1.0) * nominal_dt}
