"""Categorical emission, `K` categories, one series (E11-T04).

`y_t | i ~ Cat(softmax([0, eta_i + Gamma_i z_t]))`, category 0 as the reference; `params.probs.init` is the
list of `K` probabilities every regime starts from, converted to `eta = log(p[1:] / p[0])`. The number of
categories comes from that length (the preparer's data-driven rule applies to the data, not to the build).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from hsmmkit.model.conjugate import DirichletConjugate
from hsmmkit.model.emission._discrete import CLIP, DiscreteBase, check_free, eta_spec, gamma_spec, safe_ratio
from hsmmkit.model.emission.base import Support
from hsmmkit.model.params.spec import ParamSpec
from hsmmkit.plugins import register
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EmissionCfg


@register("emission", "categorical")
@dataclass(frozen=True)
class CategoricalFamily(DiscreteBase):
    """Categorical outcomes in `{0, .., K-1}` with a multinomial-logit link and category 0 as reference."""

    name: str = "categorical"
    native_time: Literal["continuous", "discrete"] = field(default="discrete", init=False)
    support: Support = "categorical"
    scale_by_dt: bool = False
    n_categories: int = 2
    mean_idx: tuple[int, ...] = ()

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """`emission.eta (N, K - 1)` (uniform categories)."""
        del cov_names
        self._check_series(n_series)
        return (eta_spec(n_states, self.n_categories - 1, None),)

    @staticmethod
    def _check_series(n_series: int) -> None:
        if n_series != 1:
            raise ConfigError(
                f"the categorical emission has one series, not {n_series}",
                path="data.source.observation_columns",
            )

    @classmethod
    def from_config(
        cls, cfg: EmissionCfg, n_states: int, n_series: int, cov_names: tuple[str, ...]
    ) -> tuple[CategoricalFamily, tuple[ParamSpec, ...]]:
        """`params.probs.init` (probabilities) -> `eta`; covariates under `covariates.mean`."""
        cls._check_series(n_series)
        if "probs" not in cfg.params:
            raise ConfigError(
                "categorical emission requires params.probs", path="model.emission.params.probs"
            )
        check_free(cfg.params["probs"], "model.emission.params.probs")
        raw = cfg.params["probs"].init
        path = "model.emission.params.probs.init"
        if raw is None or isinstance(raw, (str, float)):
            raise ConfigError("params.probs.init must list the category probabilities", path=path)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ConfigError("params.probs.init needs at least 2 probabilities", path=path)
        n_cat = arr.size
        probs = np.tile(arr, (n_states, 1))  # the same probabilities in every regime
        if bool(np.any(probs <= 0)) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
            raise ConfigError("probabilities must be positive and sum to 1", path=path)
        if cfg.scale_by_dt:
            raise ConfigError(
                "the categorical emission has no time scaling", path="model.emission.scale_by_dt"
            )
        try:
            idx = tuple(cov_names.index(c) for c in cfg.covariates.mean)
        except ValueError as exc:
            raise ConfigError(str(exc), path="model.emission.covariates.mean") from exc
        init = np.log(probs[:, 1:]) - np.log(probs[:, :1])
        specs = [eta_spec(n_states, n_cat - 1, init)]
        if idx:
            specs.append(gamma_spec(n_states, n_cat - 1, len(idx)))
        return cls(n_categories=n_cat, mean_idx=idx), tuple(specs)

    def _log_probs(self, theta: dict[str, Array], z: Array | None, t: int) -> Array:
        """`(T, N, K)` log category probabilities."""
        lin = self._linear(theta, z, t)  # (T, N, K-1)
        logits = jnp.concatenate([jnp.zeros((*lin.shape[:-1], 1), dtype=lin.dtype), lin], axis=-1)
        return jax.nn.log_softmax(logits, axis=-1)

    def log_prob_series(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, 1)` log `p_y`."""
        logp = self._log_probs(theta, z, dt.shape[0])
        idx = y[:, 0].astype(jnp.int32)
        return jnp.take_along_axis(logp, idx[:, None, None], axis=-1)

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` log `p_y`."""
        return self.log_prob_series(y, dt, theta, z)[..., 0]

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, 1)` `P(Y <= y)`."""
        cum = jnp.cumsum(jnp.exp(self._log_probs(theta, z, dt.shape[0])), axis=-1)
        idx = y[:, 0].astype(jnp.int32)
        return jnp.take_along_axis(cum, idx[:, None, None], axis=-1)

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, 1)` category draws from the active regime."""
        logp = self._log_probs(theta, z, dt.shape[0])[jnp.arange(dt.shape[0]), regime, :]
        return jax.random.categorical(key, logp)[:, None].astype(logp.dtype)

    def moments(
        self, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> tuple[Array, Array]:
        """Mean and variance of the category index `k` (one-hot moments are `p` and `p (1 - p)`)."""
        p = jnp.exp(self._log_probs(theta, z, dt.shape[0]))[jnp.arange(dt.shape[0]), regime, :]
        k = jnp.arange(self.n_categories, dtype=p.dtype)
        mean = jnp.sum(p * k, axis=-1, keepdims=True)
        return mean, jnp.sum(p * k**2, axis=-1, keepdims=True) - mean**2

    def sufficient_stats(
        self, y: Array, dt: Array, gamma: Array, z: Array | None = None
    ) -> dict[str, Array] | None:
        """`w = sum gamma`, `counts_ik = sum gamma 1[y = k]` (`None` with covariate links: ECM)."""
        del dt, z
        if self.mean_idx:
            return None
        onehot = jax.nn.one_hot(y[:, 0].astype(jnp.int32), self.n_categories, dtype=gamma.dtype)
        return {"w": jnp.sum(gamma, axis=0), "counts": gamma.T @ onehot}

    def m_step(self, stats: object) -> dict[str, Array] | None:
        """`p = counts / w` clipped to `>= 1e-12`, returned as `eta = log(p[1:] / p[0])`."""
        if not isinstance(stats, dict):
            return None
        p = jnp.maximum(safe_ratio(stats["counts"], stats["w"][:, None]), CLIP)
        return {"emission.eta": jnp.log(p[:, 1:]) - jnp.log(p[:, :1])}

    def conjugate_prior(self) -> DirichletConjugate | None:
        """Symmetric `Dirichlet(1)` (`None` with covariate links)."""
        return None if self.mean_idx else DirichletConjugate()

    def expected_log_prob(self, y: Array, dt: Array, q_theta: object, z: Array | None) -> Array | None:
        """`E_q[log p_y]`, `(T, N)`, under `Dirichlet(alpha)` category probabilities `q_theta` (`(N, K)`)."""
        if self.mean_idx or not isinstance(q_theta, dict):
            return None
        del dt, z
        alpha = q_theta["alpha"]
        log_p = jsp.digamma(alpha) - jsp.digamma(jnp.sum(alpha, axis=-1, keepdims=True))  # (N, K)
        idx = y[:, 0].astype(jnp.int32)
        out: Array = jnp.take(log_p, idx, axis=1).T  # (T, N)
        return out
