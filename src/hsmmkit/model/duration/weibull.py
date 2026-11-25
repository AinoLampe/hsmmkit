"""Weibull-distributed sojourn on a regular grid (E3-T04): `shape`, `scale` in clock units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
from jax.scipy import special as jsp_special

from hsmmkit.model.dists import _weibull_cdf
from hsmmkit.model.duration.base import Params
from hsmmkit.model.duration.explicit import ExplicitDurationBase, ParamDef, grid_log_pmf
from hsmmkit.plugins import register
from hsmmkit.utils.typing import Array

_INIT_SHAPE = 1.5


def _gamma_fn(x: Array) -> Array:
    return jnp.exp(jsp_special.gammaln(x))


@register("duration", "weibull")
@dataclass(frozen=True)
class WeibullDuration(ExplicitDurationBase):
    """`x ~ Weibull(shape_i, scale_i)`, `d = ceil(x / dt)`; `E[x] = scale * Gamma(1 + 1/shape)`.

    Entry-time link: `log scale_i = log scale_i^0 + c_i . z_entry`.
    """

    name: str = "weibull"
    param_defs: ClassVar[tuple[ParamDef, ...]] = (
        ("shape", "positive", _INIT_SHAPE),
        ("scale", "positive", 0.1),
    )
    primary: ClassVar[str] = "scale"

    def _scale(self, theta: Params, adj: Array | None) -> Array:
        scale = theta["duration.scale"]
        return scale if adj is None else scale * jnp.exp(adj)

    def _log_pmf_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        shape, scale = theta["duration.shape"][:, None], self._scale(theta, adj)[:, None]
        return grid_log_pmf(lambda x: _weibull_cdf(x[None, :], shape, scale), dt, d_max)

    def _tail_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        return 1.0 - _weibull_cdf(d_max * dt, theta["duration.shape"], self._scale(theta, adj))

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`scale * Gamma(1 + 1/shape)` (clock units)."""
        del nominal_dt
        return theta["duration.scale"] * _gamma_fn(1.0 + 1.0 / theta["duration.shape"])

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """`shape = 1.5`, `scale = mean / Gamma(1 + 1/1.5)` with `mean = mean_steps * nominal_dt`."""
        mean = mean_steps * nominal_dt
        return {
            "duration.shape": jnp.full_like(mean, _INIT_SHAPE),
            "duration.scale": mean / _gamma_fn(jnp.asarray(1.0 + 1.0 / _INIT_SHAPE)),
        }
