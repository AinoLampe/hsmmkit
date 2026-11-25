"""`DurationFamily` protocol (E3-T03; replaces the E1-T05 signature stub) and shared helpers.

Concept refs: §4.3, §4.4 Durations, §5.5, ADR-0006. `mean_sojourn(theta, nominal_dt)` (renamed
from `mean_sojourn_years`, CLAUDE.md §3.1); `supports_explicit`/`init_from_mean_sojourn`/
`advance_prob_t` are CLAUDE.md §3.3 additions to the concept's protocol text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

import jax.numpy as jnp
import numpy as np

from regimekit.model.params import DEFAULT_COEF_PRIOR, DirichletPrior, LogNormalPrior, NormalPrior, Prior
from regimekit.model.params.spec import Mode, ParamSpec
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.config.schema import ParamCfg, PriorCfg

Params = dict[str, Array]


class DurationFamily(Protocol):
    """Structural type for sojourn-duration families (geometric, negbin, explicit, ...)."""

    name: str
    representation: Literal["phase", "explicit"]
    supports_explicit: bool

    def param_specs(self, n_states: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """The `ParamSpec`s this family contributes for a model of `n_states` regimes."""
        ...

    def n_phases(self, i: int) -> int:
        """`r_i`: the (static) number of phases regime `i` chains through."""
        ...

    def advance_prob(self, theta: Params, dt: Array, z_d: Array | None) -> Array:
        """`(T, N)` per-step, per-regime phase-advance probability `a_i(t)`."""
        ...

    def advance_prob_t(self, theta: Params, dt: Array | float, z_d_t: Array | None = None) -> Array:
        """Per-step wrapper of `advance_prob` (`(N,)`), for use inside a `jax.lax.scan` body."""
        ...

    def log_pmf(self, theta: Params, d_max: int, nominal_dt: float, z_entry: Array | None) -> Array:
        """`(N, D_max)` sojourn-length log-pmf at a constant (entry-time) covariate value."""
        ...

    def mean_sojourn(self, theta: Params, nominal_dt: float) -> Array:
        """`(N,)` mean sojourn length in clock units (`nominal_dt`'s units: years or steps)."""
        ...

    def init_from_mean_sojourn(self, mean_steps: Array, nominal_dt: float) -> Params:
        """`theta` reproducing `mean_steps` (per state, in steps) as this family's mean sojourn."""
        ...


# --- shared numerics: advance probability, independent of phase count r ------------------------


def advance_prob_core(rate: Array, dt: Array, c: Array | None, z_d: Array | None) -> Array:
    """`(T, N)` phase-advance probability `a_i(t) = 1 - exp(-lambda_i(t) * dt_t)`.

    `lambda_i(t) = rate_i * exp(c_i . z_d_t)` when both `c` and `z_d` are given, else `rate_i`.
    """
    t = dt.shape[0]
    n = rate.shape[0]
    log_rate_adj = z_d @ c.T if (c is not None and z_d is not None) else jnp.zeros((t, n))
    rate_t = rate[None, :] * jnp.exp(log_rate_adj)  # (T, N)
    return -jnp.expm1(-rate_t * dt[:, None])


def advance_prob_entry(
    rate: Array, nominal_dt: Array | float, c: Array | None, z_entry: Array | None
) -> Array:
    """`(N,)` advance probability at a fixed (entry-time) covariate value `z_entry`."""
    n = rate.shape[0]
    log_rate_adj = c @ z_entry if (c is not None and z_entry is not None) else jnp.zeros((n,))
    rate_i = rate * jnp.exp(log_rate_adj)
    return -jnp.expm1(-rate_i * nominal_dt)


# --- shared config mapping: `ParamCfg` -> `ParamSpec` (rate/c), mirrors GBMFamily (E3-T01) -----


def prior_from_cfg(cfg: PriorCfg, init: np.ndarray | None) -> Prior:
    """`PriorCfg` -> a `Prior` object (same rule as `GBMFamily`: lognormal centers on `log(init)`)."""
    if cfg.type == "normal":
        return NormalPrior(sd=cfg.sd if cfg.sd is not None else 1.0)
    if cfg.type == "lognormal":
        mean_log = float(np.log(init).mean()) if init is not None else 0.0
        return LogNormalPrior(mean_log=mean_log, sd=cfg.sd if cfg.sd is not None else 1.0)
    if cfg.type == "dirichlet":
        return DirichletPrior(alpha=cfg.alpha if cfg.alpha is not None else 1.0)
    raise ConfigError(f"unknown prior type {cfg.type!r}", path="prior.type")


def broadcast_per_state(values: list[float] | None, n_states: int) -> np.ndarray | None:
    """`values` (one per state) -> `(n_states,)`."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n_states,):
        raise ConfigError(f"expected {n_states} per-state values, got shape {arr.shape}", path="values")
    return arr


def spec_from_paramcfg(
    name: str, cfg: ParamCfg, *, n_states: int, domain: Literal["real", "positive"], default_init: float
) -> ParamSpec:
    """`ParamCfg` -> a `(n_states,)` `ParamSpec` (plain `values`/`init`; no `increasing` handling)."""
    shape = (n_states,)
    value = broadcast_per_state(cfg.values, n_states)
    init = broadcast_per_state(cfg.init, n_states)
    if init is None and cfg.mode != "fixed":
        init = np.full(shape, default_init, dtype=np.float64)
    prior = prior_from_cfg(cfg.prior, init) if cfg.prior is not None else None
    mode: Mode | tuple[Mode, ...] = cfg.mode
    return ParamSpec(
        name=name,
        shape=shape,
        domain=domain,
        mode=mode,
        value=value,
        init=init,
        bounds=cfg.bounds,
        prior=prior,
        tie_group=cfg.tie,
        constraint=cfg.constraint,
    )


def coef_spec(name: str, n_states: int, n_coef: int) -> ParamSpec:
    """Default free coefficient spec (`duration.c`), `NormalPrior(0, coef_prior_sd)`-regularized."""
    shape = (n_states, n_coef)
    return ParamSpec(
        name=name,
        shape=shape,
        domain="real",
        mode="free",
        init=np.zeros(shape, dtype=np.float64),
        prior=DEFAULT_COEF_PRIOR,
    )


def cov_idx_from_names(names: list[str], cov_names: tuple[str, ...], path: str) -> tuple[int, ...]:
    """Column indices of `names` within `cov_names`, or `ConfigError` for an unknown name."""
    try:
        return tuple(cov_names.index(c) for c in names)
    except ValueError as exc:
        raise ConfigError(str(exc), path=path) from exc


def rate_spec_from_cfg(
    rate_cfg: ParamCfg,
    n_states: int,
    nominal_dt: float,
    init_from_mean_sojourn: object,
) -> ParamSpec:
    """`duration.rate` spec from `ParamCfg`, handling `init_mean_sojourn_steps` via `init_from_mean_sojourn`.

    `init_mean_sojourn_days` needs a clock-aware days->steps conversion that only
    `model/build.py` has (CLAUDE.md §3.3: "`init_mean_sojourn_days` is converted here with
    `days_to_steps`"); callers must pre-convert it to `init_mean_sojourn_steps` before this ticket's
    scope. `init_from_mean_sojourn` is `DurationFamily.init_from_mean_sojourn` (typed as `object`
    here to avoid a `Callable[[Array, float], Params]` import cycle with the concrete families).
    """
    if rate_cfg.init_mean_sojourn_steps is not None:
        mean_steps = broadcast_per_state(rate_cfg.init_mean_sojourn_steps, n_states)
        assert mean_steps is not None
        theta0 = init_from_mean_sojourn(jnp.asarray(mean_steps), nominal_dt)  # type: ignore[operator]
        init = np.asarray(theta0["duration.rate"])
        return ParamSpec(
            name="duration.rate",
            shape=(n_states,),
            domain="positive",
            mode=rate_cfg.mode,
            init=init,
            value=broadcast_per_state(rate_cfg.values, n_states),
            bounds=rate_cfg.bounds,
            prior=prior_from_cfg(rate_cfg.prior, init) if rate_cfg.prior is not None else None,
            tie_group=rate_cfg.tie,
            constraint=rate_cfg.constraint,
        )
    if getattr(rate_cfg, "init_mean_sojourn_days", None) is not None:
        raise ConfigError(
            "init_mean_sojourn_days needs a clock-aware days->steps conversion (model/build.py, "
            "CLAUDE.md §3.3); pre-convert to init_mean_sojourn_steps before DurationFamily.from_config",
            path="model.duration.params.rate.init_mean_sojourn_days",
        )
    return spec_from_paramcfg(
        "duration.rate", rate_cfg, n_states=n_states, domain="positive", default_init=0.1
    )
