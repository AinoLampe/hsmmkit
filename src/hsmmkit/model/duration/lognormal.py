"""Log-normal sojourn on a regular grid (E3-T04): `mu_log` (log of clock units), `sd_log`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp

from regimekit.model.dists import _lognormal_cdf
from regimekit.model.duration.base import Params
from regimekit.model.duration.explicit import ExplicitDurationBase, ParamDef, grid_log_pmf
from regimekit.utils.typing import Array

_INIT_SD_LOG = 0.5


@dataclass(frozen=True)
class LogNormalDuration(ExplicitDurationBase):
    """`log x ~ N(mu_log_i, sd_log_i^2)`, `d = ceil(x / dt)`; `E[x] = exp(mu_log + sd_log^2 / 2)`.

    Entry-time link: `mu_log_i = mu_log_i^0 + c_i . z_entry`.
    """

    name: str = "lognormal"
    param_defs: ClassVar[tuple[ParamDef, ...]] = (
        ("mu_log", "real", 0.0),
        ("sd_log", "positive", _INIT_SD_LOG),
    )
    primary: ClassVar[str] = "mu_log"

    def _mu(self, theta: Params, adj: Array | None) -> Array:
        mu = theta["duration.mu_log"]
        return mu if adj is None else mu + adj

    def _log_pmf_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        mu, sd = self._mu(theta, adj)[:, None], theta["duration.sd_log"][:, None]
        return grid_log_pmf(lambda x: _lognormal_cdf(x[None, :], mu, sd), dt, d_max)

    def _tail_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        return 1.0 - _lognormal_cdf(d_max * dt, self._mu(theta, adj), theta["duration.sd_log"])

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`exp(mu_log + sd_log^2 / 2)` (clock units)."""
        del nominal_dt
        return jnp.exp(theta["duration.mu_log"] + 0.5 * theta["duration.sd_log"] ** 2)

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """`sd_log = 0.5`, `mu_log = log(mean) - sd_log^2 / 2` with `mean = mean_steps * nominal_dt`."""
        mean = mean_steps * nominal_dt
        return {
            "duration.mu_log": jnp.log(mean) - 0.5 * _INIT_SD_LOG**2,
            "duration.sd_log": jnp.full_like(mean, _INIT_SD_LOG),
        }
