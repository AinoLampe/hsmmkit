"""Poisson emission for counts: `y_tm | i ~ Poisson(exp(eta_im + gamma_im . z_t) s(dt))` (E11-T04).

`s(dt) = dt` with `scale_by_dt` (the default: `dt` is the exposure), else 1. The rate per unit exposure is
`lambda = exp(eta)`; `params.rate.init` is given in natural units and converted to `eta = log(rate)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from hsmmkit.model.conjugate import GammaConjugate
from hsmmkit.model.emission._discrete import (
    CLIP,
    DiscreteBase,
    check_free,
    eta_spec,
    gamma_spec,
    natural_init,
    safe_ratio,
)
from hsmmkit.model.emission.base import Support
from hsmmkit.model.params.spec import ParamSpec
from hsmmkit.plugins import register
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EmissionCfg


@register("emission", "poisson")
@dataclass(frozen=True)
class PoissonFamily(DiscreteBase):
    """Per-regime, per-series Poisson counts with a log link and an optional exposure `dt`."""

    name: str = "poisson"
    native_time: Literal["continuous", "discrete"] = field(default="discrete", init=False)
    support: Support = "count"
    scale_by_dt: bool = True
    mean_idx: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """`native_time` follows `scale_by_dt`."""
        object.__setattr__(self, "native_time", "continuous" if self.scale_by_dt else "discrete")

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """`emission.eta (N, M)` (rate 1 per unit exposure)."""
        del cov_names
        return (eta_spec(n_states, n_series, None),)

    @classmethod
    def from_config(
        cls, cfg: EmissionCfg, n_states: int, n_series: int, cov_names: tuple[str, ...]
    ) -> tuple[PoissonFamily, tuple[ParamSpec, ...]]:
        """`params.rate` (natural units) -> `eta`; covariates under `covariates.mean`."""
        if "rate" not in cfg.params:
            raise ConfigError("poisson emission requires params.rate", path="model.emission.params.rate")
        check_free(cfg.params["rate"], "model.emission.params.rate")
        rate = natural_init(cfg.params["rate"].init, n_states, n_series, "model.emission.params.rate.init")
        if rate is not None and bool(np.any(rate <= 0)):
            raise ConfigError("rates must be positive", path="model.emission.params.rate.init")
        try:
            idx = tuple(cov_names.index(c) for c in cfg.covariates.mean)
        except ValueError as exc:
            raise ConfigError(str(exc), path="model.emission.covariates.mean") from exc
        family = cls(scale_by_dt=True if cfg.scale_by_dt is None else bool(cfg.scale_by_dt), mean_idx=idx)
        specs = [eta_spec(n_states, n_series, None if rate is None else np.log(rate))]
        if idx:
            specs.append(gamma_spec(n_states, n_series, len(idx)))
        return family, tuple(specs)

    def _rate(self, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` Poisson means `exp(eta_t) s(dt)`."""
        lam = jnp.exp(self._linear(theta, z, dt.shape[0]))
        return lam * dt[:, None, None] if self.scale_by_dt else lam

    def log_prob_series(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` log-pmf `y log(lambda) - lambda - log(y!)`."""
        lam = self._rate(dt, theta, z)
        yy = y[:, None, :]
        return yy * jnp.log(lam) - lam - jsp.gammaln(yy + 1.0)

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` log-pmf summed over the series."""
        return jnp.sum(self.log_prob_series(y, dt, theta, z), axis=-1)

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` `P(Y <= y)` (regularized upper incomplete gamma)."""
        lam = self._rate(dt, theta, z)
        out: Array = jsp.gammaincc(jnp.floor(y[:, None, :]) + 1.0, lam)
        return out

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, M)` Poisson draws from the active regime."""
        lam = self._rate(dt, theta, z)[jnp.arange(dt.shape[0]), regime, :]
        return jax.random.poisson(key, lam).astype(lam.dtype)

    def moments(
        self, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> tuple[Array, Array]:
        """Mean = variance = `lambda`."""
        lam = self._rate(dt, theta, z)[jnp.arange(dt.shape[0]), regime, :]
        return lam, lam

    def sufficient_stats(
        self, y: Array, dt: Array, gamma: Array, z: Array | None = None
    ) -> dict[str, Array] | None:
        """`w_s = sum gamma s`, `wy = sum gamma y` (`None` with covariate links: ECM)."""
        del z
        if self.mean_idx:
            return None
        s = dt if self.scale_by_dt else jnp.ones_like(dt)
        return {"w_s": jnp.sum(gamma * s[:, None], axis=0), "wy": gamma.T @ y}

    def m_step(self, stats: object) -> dict[str, Array] | None:
        """`lambda = sum gamma y / sum gamma s` (clipped to `>= 1e-12`), returned as `eta = log lambda`."""
        if not isinstance(stats, dict):
            return None
        lam = safe_ratio(stats["wy"], stats["w_s"][:, None])
        return {"emission.eta": jnp.log(jnp.maximum(lam, CLIP))}

    def conjugate_prior(self) -> GammaConjugate | None:
        """`Gamma(1, 1)` on the rate per unit exposure (`None` with covariate links)."""
        return None if self.mean_idx else GammaConjugate()

    def expected_log_prob(self, y: Array, dt: Array, q_theta: object, z: Array | None) -> Array | None:
        """`E_q[log Poisson(y | lambda s)]`, `(T, N)`, under `Gamma(a, b)` rates (`a`, `b`: `(N, M)`)."""
        if self.mean_idx or not isinstance(q_theta, dict):
            return None
        del z
        log_rate = jsp.digamma(q_theta["a"]) - jnp.log(q_theta["b"])  # (N, M)
        rate = q_theta["a"] / q_theta["b"]
        s = dt[:, None, None] if self.scale_by_dt else 1.0
        yy = y[:, None, :]
        out: Array = jnp.sum(
            yy * (log_rate[None] + jnp.log(s)) - rate[None] * s - jsp.gammaln(yy + 1.0), axis=-1
        )
        return out
