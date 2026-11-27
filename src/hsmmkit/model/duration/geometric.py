"""Geometric duration (`r = 1`, plain HMM sojourn): E3-T03."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.model.dists import geometric_logpmf
from hsmmkit.model.duration.base import (
    Params,
    advance_prob_core,
    advance_prob_entry,
    coef_spec,
    cov_idx_from_names,
    rate_spec_from_cfg,
)
from hsmmkit.model.params.spec import ParamSpec
from hsmmkit.plugins import register
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import DurationCfg


@register("duration", "geometric")
@dataclass(frozen=True)
class GeometricDuration:
    """`r_i = 1` for every state: `d - 1 ~ Geometric(a_i)`, `E[d] = 1 / a_i`."""

    name: str = "geometric"
    representation: Literal["phase", "explicit"] = "phase"
    supports_explicit: bool = True
    cov_idx: tuple[int, ...] = ()

    def param_specs(self, n_states: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """Default (config-free) spec: `duration.rate`, free/positive."""
        del cov_names
        return (
            ParamSpec(
                name="duration.rate",
                shape=(n_states,),
                domain="positive",
                mode="free",
                init=np.full((n_states,), 0.1),
            ),
        )

    @classmethod
    def from_config(
        cls, cfg: DurationCfg, n_states: int, cov_names: tuple[str, ...], nominal_dt: float
    ) -> tuple[GeometricDuration, tuple[ParamSpec, ...]]:
        """Map `model.duration` config to a configured `GeometricDuration` and its `ParamSpec`s."""
        cov_idx = cov_idx_from_names(cfg.covariates, cov_names, "model.duration.covariates")
        family = cls(cov_idx=cov_idx)
        specs = [rate_spec_from_cfg(cfg.params["rate"], n_states, nominal_dt, family.init_from_mean_sojourn)]
        if cov_idx:
            specs.append(coef_spec("duration.c", n_states, len(cov_idx)))
        return family, tuple(specs)

    def n_phases(self, i: int) -> int:
        """Always 1."""
        del i
        return 1

    def advance_prob(self, theta: Params, dt: Array, z_d: Array | None) -> Array:
        """`(T, N)` advance probability `a_i(t) = 1 - exp(-lambda_i(t) dt_t)`."""
        if z_d is not None and self.cov_idx:
            z, c = z_d[:, jnp.asarray(self.cov_idx)], theta.get("duration.c")
        else:
            z, c = None, None
        return advance_prob_core(theta["duration.rate"], dt, c, z)

    def advance_prob_t(self, theta: Params, dt: Array | float, z_d_t: Array | None = None) -> Array:
        """`(N,)` advance probability at a single step (E4-T02b's scan-body wrapper of `advance_prob`)."""
        if z_d_t is not None and self.cov_idx:
            z, c = z_d_t[jnp.asarray(self.cov_idx)], theta.get("duration.c")
        else:
            z, c = None, None
        return advance_prob_entry(theta["duration.rate"], dt, c, z)

    def log_pmf(self, theta: Params, d_max: int, nominal_dt: float, z_entry: Array | None) -> Array:
        """`(N, D_max)` geometric log-pmf at a constant (entry-time) covariate value."""
        if z_entry is not None and self.cov_idx:
            z, c = z_entry[jnp.asarray(self.cov_idx)], theta.get("duration.c")
        else:
            z, c = None, None
        a = advance_prob_entry(theta["duration.rate"], nominal_dt, c, z)
        support = jnp.arange(1, d_max + 1, dtype=jnp.float64)
        return jax.vmap(lambda a_i: geometric_logpmf(support, a_i, d_max))(a)

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`(N,)` mean sojourn `(1 / a_i(nominal_dt)) * nominal_dt`, in `nominal_dt`'s clock units."""
        a = advance_prob_entry(theta["duration.rate"], nominal_dt, None, None)
        return nominal_dt / a

    def from_advance_prob(self, a: Array, nominal_dt: float) -> Params:
        """Invert `a = 1 - exp(-rate dt)` (constant step, no covariates): `rate = -log(1 - a) / dt`."""
        return {"duration.rate": -jnp.log1p(-a) / nominal_dt}

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """`rate_i = 1 / (mean_steps_i * nominal_dt)`."""
        return {"duration.rate": 1.0 / (mean_steps * nominal_dt)}
