"""Shared machinery for the explicit-duration families (E3-T04).

An explicit family gives the sojourn pmf `p_i(d)`, `d = 1..D_max`, on a regular grid of step `nominal_dt`
(clock units: years under `trading`/`calendar`, steps under `steps`, CLAUDE.md §3.1). Continuous
sojourn laws are discretized as `p(d) = [F(d dt) - F((d-1) dt)] / F(D_max dt)`. The covariate link is
entry-time: the scale-like parameter is multiplied by `exp(c_i . z_entry)`.

No phase representation exists: `n_phases`/`advance_prob`/`advance_prob_t` raise `NotImplementedError`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np

from regimekit.model.duration.base import (
    Params,
    broadcast_per_state,
    coef_spec,
    cov_idx_from_names,
    spec_from_paramcfg,
)
from regimekit.model.params.spec import ParamSpec
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.config.schema import DurationCfg

# Floors keep `log` (and its gradient) finite when a grid cell underflows; the induced mass error is
# below 1e-300 per cell, far under the 1e-10 row-sum tolerance.
_TINY = 1e-300
D_MAX_TAIL_TOLERANCE = 1e-6
D_MAX_SOJOURN_MULTIPLIER = 2.0

ParamDef = tuple[str, Literal["real", "positive"], float]  # (config key, domain, default init)


def grid_log_pmf(cdf: Callable[[Array], Array], dt: float, d_max: int) -> Array:
    """`(N, D_max)` log of `[F(d dt) - F((d-1) dt)] / F(D_max dt)`; `cdf` maps `(D,)` edges to `(N, D)`."""
    edges = jnp.arange(1, d_max + 1, dtype=jnp.float64) * dt
    f = cdf(edges)  # (N, D): F(d dt), F(0) = 0 implied
    f_prev = jnp.concatenate([jnp.zeros_like(f[:, :1]), f[:, :-1]], axis=1)
    numerator = jnp.maximum(f - f_prev, _TINY)
    denominator = jnp.maximum(f[:, -1:], _TINY)
    return jnp.log(numerator) - jnp.log(denominator)


@dataclass(frozen=True)
class ExplicitDurationBase:
    """Common behaviour of explicit families; subclasses supply `_log_pmf_one`, `_tail_one` and the params."""

    name: str = ""
    representation: Literal["phase", "explicit"] = "explicit"
    supports_explicit: bool = True
    cov_idx: tuple[int, ...] = ()

    param_defs: ClassVar[tuple[ParamDef, ...]] = ()
    primary: ClassVar[str] = ""

    # --- phase-representation API: not available --------------------------------------------

    def n_phases(self, i: int) -> int:
        """Not defined for an explicit family."""
        raise NotImplementedError(f"{self.name}: explicit duration family has no phase representation")

    def advance_prob(self, theta: Params, dt: Array, z_d: Array | None) -> Array:
        """Not defined for an explicit family."""
        raise NotImplementedError(f"{self.name}: explicit duration family has no phase advance probability")

    def advance_prob_t(self, theta: Params, dt: Array | float, z_d_t: Array | None = None) -> Array:
        """Not defined for an explicit family."""
        raise NotImplementedError(f"{self.name}: explicit duration family has no phase advance probability")

    # --- parameters -------------------------------------------------------------------------

    def param_specs(self, n_states: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """Default (config-free) specs: every parameter free at its default init."""
        del cov_names
        return tuple(
            ParamSpec(
                name=f"duration.{key}",
                shape=(n_states,),
                domain=domain,
                mode="free",
                init=np.full((n_states,), default),
            )
            for key, domain, default in self.param_defs
        )

    @classmethod
    def from_config(
        cls, cfg: DurationCfg, n_states: int, cov_names: tuple[str, ...], nominal_dt: float
    ) -> tuple[ExplicitDurationBase, tuple[ParamSpec, ...]]:
        """Map `model.duration` config to a configured family and its `ParamSpec`s.

        `init_mean_sojourn_steps` on the family's primary parameter initializes all parameters that have
        no explicit `init` through `init_from_mean_sojourn` (CLAUDE.md §3.1, §3.3).
        """
        cov_idx = cov_idx_from_names(cfg.covariates, cov_names, "model.duration.covariates")
        family = cls(cov_idx=cov_idx)
        for key, _, _ in cls.param_defs:
            if key not in cfg.params:
                raise ConfigError(
                    f"{cls.__name__} requires params.{key}", path=f"model.duration.params.{key}"
                )
        primary_cfg = cfg.params[cls.primary]
        if getattr(primary_cfg, "init_mean_sojourn_days", None) is not None:
            raise ConfigError(
                "init_mean_sojourn_days needs a clock-aware days->steps conversion (model/build.py, "
                "CLAUDE.md §3.3); pre-convert to init_mean_sojourn_steps",
                path=f"model.duration.params.{cls.primary}.init_mean_sojourn_days",
            )
        theta0: Params | None = None
        if primary_cfg.init_mean_sojourn_steps is not None:
            mean_steps = broadcast_per_state(primary_cfg.init_mean_sojourn_steps, n_states)
            assert mean_steps is not None
            theta0 = family.init_from_mean_sojourn(jnp.asarray(mean_steps), nominal_dt)
        specs: list[ParamSpec] = []
        for key, domain, default in cls.param_defs:
            spec = spec_from_paramcfg(
                f"duration.{key}", cfg.params[key], n_states=n_states, domain=domain, default_init=default
            )
            if theta0 is not None and cfg.params[key].init is None and spec.mode != "fixed":
                spec = replace(spec, init=np.asarray(theta0[f"duration.{key}"]))
            specs.append(spec)
        if cov_idx:
            specs.append(coef_spec("duration.c", n_states, len(cov_idx)))
        return family, tuple(specs)

    # --- numerics ---------------------------------------------------------------------------

    def _adj(self, theta: Params, z_entry: Array | None) -> Array | None:
        """Entry-time link `c_i . z_entry`: `(N,)` for one entry, `(T, N)` for `(T, K)`; `None` if unused."""
        if z_entry is None or not self.cov_idx or "duration.c" not in theta:
            return None
        return z_entry[..., jnp.asarray(self.cov_idx)] @ theta["duration.c"].T

    def _log_pmf_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        raise NotImplementedError

    def _tail_one(self, theta: Params, adj: Array | None, d_max: int, dt: float) -> Array:
        raise NotImplementedError

    def log_pmf(self, theta: Params, d_max: int, nominal_dt: float, z_entry: Array | None) -> Array:
        """`(N, D_max)` log-pmf (`z_entry (K_d,)` or `None`), or `(T, N, D_max)` for `z_entry (T, K_d)`."""
        adj = self._adj(theta, z_entry)
        if adj is None or adj.ndim == 1:
            return self._log_pmf_one(theta, adj, d_max, nominal_dt)
        return jax.vmap(lambda a: self._log_pmf_one(theta, a, d_max, nominal_dt))(adj)

    def tail_mass(self, theta: Params, d_max: int, nominal_dt: float, z_entry: Array | None = None) -> Array:
        """Untruncated probability beyond `D_max`, `1 - F(D_max dt)`: `(N,)`, or `(T, N)`."""
        adj = self._adj(theta, z_entry)
        if adj is None or adj.ndim == 1:
            return self._tail_one(theta, adj, d_max, nominal_dt)
        return jax.vmap(lambda a: self._tail_one(theta, a, d_max, nominal_dt))(adj)

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`(N,)` analytic untruncated mean sojourn in clock units."""
        raise NotImplementedError

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """`theta` whose analytic mean sojourn is `mean_steps` steps (per state)."""
        raise NotImplementedError

    def mean_sojourn_steps_truncated(
        self, theta: Params, d_max: int, dt: float, z_entry: Array | None = None
    ) -> Array:
        """`(N,)` mean of the truncated, renormalized pmf, in steps: `sum_d d p(d)`."""
        p = jnp.exp(self.log_pmf(theta, d_max, dt, z_entry))
        d = jnp.arange(1, d_max + 1, dtype=jnp.float64)
        return jnp.sum(p * d, axis=-1)


def check_d_max(
    family: ExplicitDurationBase, theta: Params, d_max: int, nominal_dt: float, *, warn: bool = True
) -> list[str]:
    """`D_max` rule: flag truncated mass `> 1e-6` or `d_max < 2 x` mean sojourn (in steps).

    Returns the messages and emits each as a `UserWarning` unless `warn=False`. Runs outside jit
    (it concretizes values).
    """
    messages: list[str] = []
    tail = float(jnp.max(family.tail_mass(theta, d_max, nominal_dt)))
    if tail > D_MAX_TAIL_TOLERANCE:
        messages.append(
            f"model.d_max={d_max}: truncated duration mass {tail:.3g} exceeds {D_MAX_TAIL_TOLERANCE:g}"
        )
    mean_steps = float(jnp.max(family.mean_sojourn(theta, nominal_dt))) / nominal_dt
    if d_max < D_MAX_SOJOURN_MULTIPLIER * mean_steps:
        messages.append(
            f"model.d_max={d_max} is below {D_MAX_SOJOURN_MULTIPLIER:g} x the mean sojourn "
            f"({mean_steps:.1f} steps)"
        )
    if warn:
        for message in messages:
            warnings.warn(message, UserWarning, stacklevel=2)
    return messages
