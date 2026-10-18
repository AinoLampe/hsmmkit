"""Gamma-distributed sojourn on a regular grid (E3-T04): `shape`, `scale` in clock units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp

from regimekit.model.dists import _gamma_cdf
from regimekit.model.duration.base import Params
from regimekit.model.duration.explicit import ExplicitDurationBase, ParamDef, grid_log_pmf
from regimekit.utils.typing import Array

_INIT_SHAPE = 2.0


@dataclass(frozen=True)
class GammaDuration(ExplicitDurationBase):
    """`x ~ Gamma(shape_i, scale_i)`, `d = ceil(x / dt)`; `E[x] = shape * scale`.

    Entry-time link: `log scale_i = log scale_i^0 + c_i . z_entry`.
    """

    name: str = "gamma"
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
        return grid_log_pmf(lambda x: _gamma_cdf(x[None, :], shape, scale), dt, d_max)

    def _tail_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        return 1.0 - _gamma_cdf(d_max * dt, theta["duration.shape"], self._scale(theta, adj))

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`shape * scale` (clock units)."""
        del nominal_dt
        return theta["duration.shape"] * theta["duration.scale"]

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """`shape = 2`, `scale = mean / 2` with `mean = mean_steps * nominal_dt`."""
        mean = mean_steps * nominal_dt
        return {"duration.shape": jnp.full_like(mean, _INIT_SHAPE), "duration.scale": mean / _INIT_SHAPE}
