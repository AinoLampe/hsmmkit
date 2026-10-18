"""GBM emission: per-state Gaussian log-returns scaled by `dt`, with covariate links (E3-T01).

`y_t | i ~ N((mu_i(t) - sigma_i(t)**2 / 2) * dt_t, sigma_i(t)**2 * dt_t)` with

    mu_i(t)     = mu_i + gamma_i . z^e_t          (drift link)
    log sigma_i(t) = log sigma_i + delta_i . z^e_t  (log-vol link)

Parameters are in clock units (CLAUDE.md §3.1): annualized under `trading`/`calendar`, per step
under `steps` (`dt == 1`, so the density is simply `N(mu_i - sigma_i**2 / 2, sigma_i**2)`) -- no
code here branches on the clock name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import jax.scipy.stats
import numpy as np

from regimekit.model.dists import normal_logpdf, sample_normal
from regimekit.model.emission.base import Support
from regimekit.model.params import DEFAULT_COEF_PRIOR, DirichletPrior, LogNormalPrior, NormalPrior, Prior
from regimekit.model.params.spec import Domain, Mode, ParamSpec
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.config.schema import EmissionCfg, ParamCfg, PriorCfg


def _prior_from_cfg(cfg: PriorCfg, init: np.ndarray | None) -> Prior:
    if cfg.type == "normal":
        return NormalPrior(sd=cfg.sd if cfg.sd is not None else 1.0)
    if cfg.type == "lognormal":
        mean_log = float(np.log(init).mean()) if init is not None else 0.0
        return LogNormalPrior(mean_log=mean_log, sd=cfg.sd if cfg.sd is not None else 1.0)
    if cfg.type == "dirichlet":
        return DirichletPrior(alpha=cfg.alpha if cfg.alpha is not None else 1.0)
    raise ConfigError(f"unknown prior type {cfg.type!r}", path="prior.type")


def _broadcast_per_state(
    values: list[float] | None, n_states: int, n_series: int, *, flat: bool
) -> np.ndarray | None:
    """`values` (one per state) -> `(n_states,)` (`flat`) or `(n_states, n_series)`."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n_states,):
        raise ConfigError(f"expected {n_states} per-state values, got shape {arr.shape}", path="values")
    return arr if flat else np.tile(arr[:, None], (1, n_series))


def _spec_from_paramcfg(
    name: str,
    cfg: ParamCfg,
    *,
    n_states: int,
    n_series: int,
    domain: Domain,
    default_init: float,
    flat: bool = False,
) -> ParamSpec:
    # The `increasing` bijector (ParamManager, E1-T03b) only supports a 1-D `(N,)` cell; a joint
    # ordering across a per-series `(N, M)` grid has no clean semantics, so `constraint="increasing"`
    # collapses the spec to `(N,)` (matches Appendix A: single-series, M=1) and is rejected for M>1,
    # rather than silently sharing one ordered value across every series.
    # `flat=True` forces an `(N,)` spec regardless (per-state parameters such as Student-t `nu`).
    increasing = cfg.constraint == "increasing"
    flat = flat or increasing
    if increasing and n_series != 1:
        raise ConfigError(
            f"constraint='increasing' is only supported for a single series (n_series={n_series})",
            path=name,
        )
    shape: tuple[int, ...] = (n_states,) if flat else (n_states, n_series)
    value = _broadcast_per_state(cfg.values, n_states, n_series, flat=flat)
    init = _broadcast_per_state(cfg.init, n_states, n_series, flat=flat)
    if init is None and cfg.mode != "fixed":
        init = np.full(shape, default_init, dtype=np.float64)
    prior = _prior_from_cfg(cfg.prior, init) if cfg.prior is not None else None
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


def _coef_spec(name: str, n_states: int, n_coef: int) -> ParamSpec:
    """Default free coefficient spec (`gamma`/`delta`), `NormalPrior(0, coef_prior_sd)`-regularized."""
    shape = (n_states, n_coef)
    init = np.zeros(shape, dtype=np.float64)
    return ParamSpec(
        name=name,
        shape=shape,
        domain="real",
        mode="free",
        init=init,
        prior=DEFAULT_COEF_PRIOR,
    )


@dataclass(frozen=True)
class GBMFamily:
    """Per-state geometric Brownian motion emission (Gaussian log-returns).

    `drift_idx`/`logvol_idx` select, by column, which entries of the `(T, K_e)` emission
    covariate block `z` feed the drift/log-vol links (built by `from_config`).
    """

    name: str = "gbm"
    native_time: Literal["continuous", "discrete"] = "continuous"
    support: Support = "real"
    scale_by_dt: bool = True
    drift_idx: tuple[int, ...] = ()
    logvol_idx: tuple[int, ...] = ()

    # --- protocol: parameter layout ---------------------------------------------------------

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """Default (config-free) specs: `mu`/`sigma` only, both free.

        The real, config-driven mapping (modes, values, priors, covariate links) is
        `GBMFamily.from_config` (CLAUDE.md §3.3's "every family gets a `from_config`" pattern);
        `param_specs` exists to satisfy `EmissionFamily` for callers with no project config.
        """
        del cov_names
        mu = ParamSpec(
            name="emission.mu",
            shape=(n_states, n_series),
            domain="real",
            mode="free",
            init=np.zeros((n_states, n_series)),
        )
        sigma = ParamSpec(
            name="emission.sigma",
            shape=(n_states, n_series),
            domain="positive",
            mode="free",
            init=np.full((n_states, n_series), 0.1),
        )
        return (mu, sigma)

    @classmethod
    def from_config(
        cls,
        cfg: EmissionCfg,
        n_states: int,
        n_series: int,
        cov_names: tuple[str, ...],
    ) -> tuple[GBMFamily, tuple[ParamSpec, ...]]:
        """Map `model.emission` config to a configured `GBMFamily` and its `ParamSpec`s."""
        try:
            drift_idx = tuple(cov_names.index(c) for c in cfg.covariates.drift)
        except ValueError as exc:
            raise ConfigError(str(exc), path="model.emission.covariates.drift") from exc
        try:
            logvol_idx = tuple(cov_names.index(c) for c in cfg.covariates.log_vol)
        except ValueError as exc:
            raise ConfigError(str(exc), path="model.emission.covariates.log_vol") from exc

        family = cls(drift_idx=drift_idx, logvol_idx=logvol_idx)

        specs = [
            _spec_from_paramcfg(
                "emission.mu",
                cfg.params["mu"],
                n_states=n_states,
                n_series=n_series,
                domain="real",
                default_init=0.0,
            ),
            _spec_from_paramcfg(
                "emission.sigma",
                cfg.params["sigma"],
                n_states=n_states,
                n_series=n_series,
                domain="positive",
                default_init=0.1,
            ),
        ]
        if drift_idx:
            if "gamma" in cfg.params:
                specs.append(
                    _spec_from_paramcfg(
                        "emission.gamma",
                        cfg.params["gamma"],
                        n_states=n_states,
                        n_series=len(drift_idx),
                        domain="real",
                        default_init=0.0,
                    )
                )
            else:
                specs.append(_coef_spec("emission.gamma", n_states, len(drift_idx)))
        if logvol_idx:
            if "delta" in cfg.params:
                specs.append(
                    _spec_from_paramcfg(
                        "emission.delta",
                        cfg.params["delta"],
                        n_states=n_states,
                        n_series=len(logvol_idx),
                        domain="real",
                        default_init=0.0,
                    )
                )
            else:
                specs.append(_coef_spec("emission.delta", n_states, len(logvol_idx)))
        return family, tuple(specs)

    # --- protocol: numerics ------------------------------------------------------------------

    def _mean_sd(self, dt: Array, theta: dict[str, Array], z: Array | None) -> tuple[Array, Array]:
        """`(T, N, M)` mean and sd of the per-step, per-regime log-return density."""
        mu = theta["emission.mu"]
        sigma = theta["emission.sigma"]
        mu2 = mu[:, None] if mu.ndim == 1 else mu  # (N, M)
        sigma2 = sigma[:, None] if sigma.ndim == 1 else sigma  # (N, M)
        n_states, n_series = mu2.shape
        t = dt.shape[0]

        drift_adj = jnp.zeros((t, n_states))
        if self.drift_idx and "emission.gamma" in theta and z is not None:
            z_drift = z[:, jnp.asarray(self.drift_idx)]  # (T, K_drift)
            drift_adj = z_drift @ theta["emission.gamma"].T  # (T, N)

        logvol_adj = jnp.zeros((t, n_states))
        if self.logvol_idx and "emission.delta" in theta and z is not None:
            z_logvol = z[:, jnp.asarray(self.logvol_idx)]  # (T, K_logvol)
            logvol_adj = z_logvol @ theta["emission.delta"].T  # (T, N)

        mu_t = mu2[None, :, :] + drift_adj[:, :, None]  # (T, N, M)
        sigma_t = jnp.exp(jnp.log(sigma2)[None, :, :] + logvol_adj[:, :, None])  # (T, N, M)

        dt_ = dt[:, None, None]
        mean_t = (mu_t - 0.5 * sigma_t**2) * dt_
        sd_t = sigma_t * jnp.sqrt(dt_)
        del n_series
        return mean_t, sd_t

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` log-density, summed over the `M` series."""
        mean_t, sd_t = self._mean_sd(dt, theta, z)
        logdens = normal_logpdf(y[:, None, :], mean_t, sd_t)  # (T, N, M)
        return jnp.sum(logdens, axis=-1)

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` per-regime, per-series predictive CDF at `y` (E5-T04's PIT diagnostic)."""
        mean_t, sd_t = self._mean_sd(dt, theta, z)
        return jax.scipy.stats.norm.cdf((y[:, None, :] - mean_t) / sd_t)

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, M)` samples, one draw per step from the active regime's density."""
        mean_t, sd_t = self._mean_sd(dt, theta, z)
        t = dt.shape[0]
        idx = jnp.arange(t)
        mean_sel = mean_t[idx, regime, :]  # (T, M)
        sd_sel = sd_t[idx, regime, :]
        return sample_normal(key, mean_sel, sd_sel, shape=mean_sel.shape)

    def init_carry(self) -> object:
        """GBM has no path-dependent state."""
        return None

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Identity: GBM has no recursive carry to advance."""
        del y_t, theta
        return carry

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array) -> object | None:
        """No EM sufficient statistics in v1 (hook for E5-T06)."""
        del y, dt, gamma
        return None
