"""Bernoulli emission for binary sequences (E11-T04).

`y_tm | i ~ Bernoulli(sigmoid(eta_im + gamma_im . z_t))`; `params.prob.init` is given as probabilities and
converted to `eta = logit(p)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.model.conjugate import BetaConjugate
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


@register("emission", "bernoulli")
@dataclass(frozen=True)
class BernoulliFamily(DiscreteBase):
    """Per-regime, per-series Bernoulli outcomes with a logit link."""

    name: str = "bernoulli"
    native_time: Literal["continuous", "discrete"] = field(default="discrete", init=False)
    support: Support = "binary"
    scale_by_dt: bool = False
    mean_idx: tuple[int, ...] = ()

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """`emission.eta (N, M)` (probability 1/2)."""
        del cov_names
        return (eta_spec(n_states, n_series, None),)

    @classmethod
    def from_config(
        cls, cfg: EmissionCfg, n_states: int, n_series: int, cov_names: tuple[str, ...]
    ) -> tuple[BernoulliFamily, tuple[ParamSpec, ...]]:
        """`params.prob` (probabilities) -> `eta = logit(p)`; covariates under `covariates.mean`."""
        if "prob" not in cfg.params:
            raise ConfigError("bernoulli emission requires params.prob", path="model.emission.params.prob")
        check_free(cfg.params["prob"], "model.emission.params.prob")
        prob = natural_init(cfg.params["prob"].init, n_states, n_series, "model.emission.params.prob.init")
        if prob is not None and bool(np.any((prob <= 0) | (prob >= 1))):
            raise ConfigError("probabilities must be in (0, 1)", path="model.emission.params.prob.init")
        if cfg.scale_by_dt:
            raise ConfigError("the bernoulli emission has no time scaling", path="model.emission.scale_by_dt")
        try:
            idx = tuple(cov_names.index(c) for c in cfg.covariates.mean)
        except ValueError as exc:
            raise ConfigError(str(exc), path="model.emission.covariates.mean") from exc
        init = None if prob is None else np.log(prob) - np.log1p(-prob)
        specs = [eta_spec(n_states, n_series, init)]
        if idx:
            specs.append(gamma_spec(n_states, n_series, len(idx)))
        return cls(mean_idx=idx), tuple(specs)

    def log_prob_series(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` log-pmf `y * logit - softplus(logit)`."""
        logit = self._linear(theta, z, dt.shape[0])
        return y[:, None, :] * logit - jax.nn.softplus(logit)

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` log-pmf summed over the series."""
        return jnp.sum(self.log_prob_series(y, dt, theta, z), axis=-1)

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` `P(Y <= y)`: `1 - p` below 1, 1 from 1 on."""
        p = jax.nn.sigmoid(self._linear(theta, z, dt.shape[0]))
        return jnp.where(y[:, None, :] >= 1.0, 1.0, 1.0 - p)

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, M)` Bernoulli draws from the active regime."""
        p = jax.nn.sigmoid(self._linear(theta, z, dt.shape[0]))[jnp.arange(dt.shape[0]), regime, :]
        return jax.random.bernoulli(key, p).astype(p.dtype)

    def moments(
        self, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> tuple[Array, Array]:
        """Mean `p` and variance `p (1 - p)`."""
        p = jax.nn.sigmoid(self._linear(theta, z, dt.shape[0]))[jnp.arange(dt.shape[0]), regime, :]
        return p, p * (1.0 - p)

    def sufficient_stats(
        self, y: Array, dt: Array, gamma: Array, z: Array | None = None
    ) -> dict[str, Array] | None:
        """`w = sum gamma`, `wy = sum gamma y` (`None` with covariate links: ECM)."""
        del dt, z
        if self.mean_idx:
            return None
        return {"w": jnp.sum(gamma, axis=0), "wy": gamma.T @ y}

    def m_step(self, stats: object) -> dict[str, Array] | None:
        """`p = sum gamma y / sum gamma` clipped to `[1e-12, 1 - 1e-12]`, returned as `eta = logit p`."""
        if not isinstance(stats, dict):
            return None
        p = jnp.clip(safe_ratio(stats["wy"], stats["w"][:, None]), CLIP, 1.0 - CLIP)
        return {"emission.eta": jnp.log(p) - jnp.log1p(-p)}

    def conjugate_prior(self) -> BetaConjugate:
        """`Beta(1, 1)`."""
        return BetaConjugate()
