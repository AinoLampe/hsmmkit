"""Nonparametric sojourn pmf over `1..D_max` (E3-T04): `duration.logits (N, D_max)` on the `pmf` domain.

The `pmf` bijector returns the constrained simplex values, so `theta["duration.logits"]` already holds
the probabilities (the ticket's "softmax" is the bijector itself, cf. `initial.logits`); `log_pmf` is
their log, renormalized defensively. There is no covariate link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
import numpy as np

from regimekit.model.duration.base import Params, broadcast_per_state, prior_from_cfg
from regimekit.model.duration.explicit import ExplicitDurationBase, ParamDef
from regimekit.model.params.spec import ParamSpec
from regimekit.plugins import register
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.config.schema import DurationCfg

_MIN_PROB = 1e-12


@register("duration", "nonparametric")
@dataclass(frozen=True)
class NonparametricDuration(ExplicitDurationBase):
    """Free pmf `p_i(d)`, `d = 1..d_max`, per state (`d_max` is a static hyperparameter)."""

    name: str = "nonparametric"
    d_max: int = 0
    param_defs: ClassVar[tuple[ParamDef, ...]] = ()
    primary: ClassVar[str] = "logits"

    def param_specs(self, n_states: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """Default (config-free) spec: `duration.logits`, free, uniform pmf init."""
        del cov_names
        return (self._spec(n_states, "free", None, None),)

    def _spec(self, n_states: int, mode: object, init: np.ndarray | None, prior: object) -> ParamSpec:
        shape = (n_states, self.d_max)
        return ParamSpec(
            name="duration.logits",
            shape=shape,
            domain="pmf",
            mode=mode,  # type: ignore[arg-type]
            init=np.full(shape, 1.0 / self.d_max) if init is None else init,
            prior=prior,  # type: ignore[arg-type]
        )

    @classmethod
    def from_config(
        cls,
        cfg: DurationCfg,
        n_states: int,
        cov_names: tuple[str, ...],
        nominal_dt: float,
        d_max: int | None = None,
    ) -> tuple[NonparametricDuration, tuple[ParamSpec, ...]]:
        """Build from `model.duration`; `d_max` (`model.d_max`) is required."""
        if d_max is None or d_max < 1:
            raise ConfigError("nonparametric duration requires model.d_max >= 1", path="model.d_max")
        if cfg.covariates:
            raise ConfigError(
                "nonparametric duration has no covariate link", path="model.duration.covariates"
            )
        if "logits" not in cfg.params:
            raise ConfigError("nonparametric requires params.logits", path="model.duration.params.logits")
        del cov_names
        pcfg = cfg.params["logits"]
        if pcfg.values is not None or pcfg.init is not None:
            raise ConfigError(
                "per-state values/init cannot describe an (N, D_max) pmf; use init_mean_sojourn_steps",
                path="model.duration.params.logits",
            )
        if getattr(pcfg, "init_mean_sojourn_days", None) is not None:
            raise ConfigError(
                "init_mean_sojourn_days needs a clock-aware days->steps conversion (model/build.py); "
                "pre-convert to init_mean_sojourn_steps",
                path="model.duration.params.logits.init_mean_sojourn_days",
            )
        family = cls(d_max=d_max)
        init = None
        if pcfg.init_mean_sojourn_steps is not None:
            mean_steps = broadcast_per_state(pcfg.init_mean_sojourn_steps, n_states)
            assert mean_steps is not None
            init = np.asarray(
                family.init_from_mean_sojourn(jnp.asarray(mean_steps), nominal_dt)["duration.logits"]
            )
        prior = prior_from_cfg(pcfg.prior, None) if pcfg.prior is not None else None
        return family, (family._spec(n_states, pcfg.mode, init, prior),)

    def _log_pmf_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        p = theta["duration.logits"]
        if p.shape[-1] != d_max:
            raise ConfigError(
                f"duration.logits has {p.shape[-1]} bins but d_max={d_max}", path="duration.logits"
            )
        p = jnp.maximum(p, _MIN_PROB)
        return jnp.log(p) - jnp.log(jnp.sum(p, axis=-1, keepdims=True))

    def _tail_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        return jnp.zeros(theta["duration.logits"].shape[0])

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`sum_d d p(d) * nominal_dt` (exact: the pmf is supported on `1..D_max`)."""
        d_max = theta["duration.logits"].shape[-1]
        return self.mean_sojourn_steps_truncated(theta, d_max, nominal_dt) * nominal_dt

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """Geometric pmf on `1..d_max` with mean `mean_steps` (clipped away from a point mass)."""
        del nominal_dt
        a = jnp.clip(1.0 / mean_steps, 1e-6, 0.999)[:, None]
        d = jnp.arange(1, self.d_max + 1, dtype=jnp.float64)[None, :]
        p = jnp.maximum(a * (1.0 - a) ** (d - 1.0), _MIN_PROB)
        return {"duration.logits": p / jnp.sum(p, axis=-1, keepdims=True)}
