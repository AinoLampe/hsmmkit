"""Shifted negative-binomial duration (`r_i` phases per regime), ADR-0006: E3-T03."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np

from regimekit.model.dists import negbin_shifted_logpmf
from regimekit.model.duration.base import (
    Params,
    advance_prob_core,
    advance_prob_entry,
    coef_spec,
    cov_idx_from_names,
    rate_spec_from_cfg,
)
from regimekit.model.params.spec import ParamSpec
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.config.schema import DurationCfg


@dataclass(frozen=True)
class NegBinDuration:
    """Regime `i` has `r_i >= 1` phases; on a constant grid `d - r_i ~ NegBin(r_i, a_i)`.

    `E[d] = r_i / a_i`, `Var[d] = r_i (1 - a_i) / a_i**2`, minimum duration `r_i`. `r` is a static,
    per-state hyperparameter tuple from config (`DurationCfg.r`), never differentiated.
    """

    name: str = "negbin"
    representation: Literal["phase", "explicit"] = "phase"
    supports_explicit: bool = True
    r: tuple[int, ...] = ()
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
    ) -> tuple[NegBinDuration, tuple[ParamSpec, ...]]:
        """Map `model.duration` config to a configured `NegBinDuration` and its `ParamSpec`s."""
        if cfg.r is None or len(cfg.r) != n_states:
            raise ConfigError(f"negbin requires r: list[int] of length {n_states}", path="model.duration.r")
        cov_idx = cov_idx_from_names(cfg.covariates, cov_names, "model.duration.covariates")
        family = cls(r=tuple(cfg.r), cov_idx=cov_idx)
        specs = [rate_spec_from_cfg(cfg.params["rate"], n_states, nominal_dt, family.init_from_mean_sojourn)]
        if cov_idx:
            specs.append(coef_spec("duration.c", n_states, len(cov_idx)))
        return family, tuple(specs)

    def n_phases(self, i: int) -> int:
        """`r_i`."""
        return self.r[i]

    def advance_prob(self, theta: Params, dt: Array, z_d: Array | None) -> Array:
        """`(T, N)` advance probability `a_i(t) = 1 - exp(-lambda_i(t) dt_t)` (independent of `r`)."""
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
        """`(N, D_max)` shifted-negative-binomial log-pmf at a constant (entry-time) covariate value."""
        if z_entry is not None and self.cov_idx:
            z, c = z_entry[jnp.asarray(self.cov_idx)], theta.get("duration.c")
        else:
            z, c = None, None
        a = advance_prob_entry(theta["duration.rate"], nominal_dt, c, z)
        support = jnp.arange(1, d_max + 1, dtype=jnp.float64)
        rows = [negbin_shifted_logpmf(support, float(r_i), a[i], d_max) for i, r_i in enumerate(self.r)]
        return jnp.stack(rows, axis=0)

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`(N,)` mean sojourn `(r_i / a_i(nominal_dt)) * nominal_dt`, in `nominal_dt`'s clock units."""
        a = advance_prob_entry(theta["duration.rate"], nominal_dt, None, None)
        r = jnp.asarray(self.r, dtype=jnp.float64)
        return (r / a) * nominal_dt

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """`rate_i = r_i / (mean_steps_i * nominal_dt)`."""
        r = jnp.asarray(self.r, dtype=jnp.float64)
        return {"duration.rate": r / (mean_steps * nominal_dt)}
