"""Gaussian emission with a diagonal covariance (E10-T06, FR-EM01 diag part, FR-EM06).

    y_t | i ~ N( m_i(t) s(dt_t),  diag(sigma_i(t)**2) s(dt_t) ),   s(dt) = dt if scale_by_dt else 1

    m_i(t)        = mu_i + gamma_i . z^e_t - (sigma_i(t)**2 / 2 if ito_correction else 0)
    log sigma_i(t) = log sigma_i + delta_i . z^e_t

With the defaults (`scale_by_dt=False`, `ito_correction=False`) this is the plain per-step Gaussian
`N(mean_i, scale_i**2)`. With `scale_by_dt=True, ito_correction=True` the operations are the ones of the
Part I GBM density, in the same order (a finance application registers that case as `gbm`).

Parameters are in clock units (CLAUDE.md §3.1); no code here branches on the clock name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

import jax.numpy as jnp
import jax.scipy.stats
import numpy as np

from hsmmkit.model.dists import normal_logpdf, sample_normal
from hsmmkit.model.emission.base import Support
from hsmmkit.model.params import DEFAULT_COEF_PRIOR, DirichletPrior, LogNormalPrior, NormalPrior, Prior
from hsmmkit.model.params.spec import Domain, Mode, ParamSpec
from hsmmkit.plugins import register
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EmissionCfg, ParamCfg, PriorCfg


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
    values: list[float] | float | str | None, n_states: int, n_series: int, *, flat: bool
) -> np.ndarray | None:
    """`values` (one per state) -> `(n_states,)` (`flat`) or `(n_states, n_series)`."""
    if values is None:
        return None
    if isinstance(values, str):
        raise ConfigError(f"init: {values!r} is not available before E12-T06", path="values")
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
    # `flat=True` forces an `(N,)` spec regardless (per-state parameters such as Student-t `nu`, or one
    # value shared by all series under `model.emission.per_series: false`, which also allows `increasing`).
    increasing = cfg.constraint == "increasing"
    if increasing and n_series != 1 and not flat:
        raise ConfigError(
            f"constraint='increasing' is only supported for a single series (n_series={n_series}); "
            "set model.emission.per_series: false to share one ordered value across the series",
            path=name,
        )
    flat = flat or increasing
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


@register("emission", "gaussian")
@dataclass(frozen=True)
class GaussianFamily:
    """Per-state Gaussian emission with diagonal covariance and optional covariate links.

    `mean_idx`/`scale_idx` select, by column, which entries of the `(T, K_e)` emission covariate block `z`
    feed the mean / log-scale links (built by `from_config`).

    The class attributes name the family's vocabulary, so that a subclass (the finance `gbm`) keeps its own
    parameter and config names on top of the same numerics.
    """

    name: str = "gaussian"
    native_time: Literal["continuous", "discrete"] = field(default="discrete", init=False)
    support: Support = "real"
    scale_by_dt: bool = False
    ito_correction: bool = False
    mean_idx: tuple[int, ...] = ()
    scale_idx: tuple[int, ...] = ()

    # Names of the family's parameters / config keys; subclasses override them.
    mean_param: ClassVar[str] = "mean"
    scale_param: ClassVar[str] = "scale"
    mean_cov_key: ClassVar[str] = "mean"
    scale_cov_key: ClassVar[str] = "log_scale"
    default_scale: ClassVar[float] = 1.0
    fixed_options: ClassVar[bool] = False
    """`True`: `scale_by_dt` and `ito_correction` are the class defaults and the config cannot change them."""

    def __post_init__(self) -> None:
        """`native_time` follows `scale_by_dt`."""
        object.__setattr__(self, "native_time", "continuous" if self.scale_by_dt else "discrete")

    @classmethod
    def name_for_errors(cls) -> str:
        """The family's registry name, for messages."""
        return str(cls.__dataclass_fields__["name"].default)

    @classmethod
    def mean_name(cls) -> str:
        """Full name of the location parameter (`emission.mean`)."""
        return f"emission.{cls.mean_param}"

    @classmethod
    def scale_name(cls) -> str:
        """Full name of the scale parameter (`emission.scale`)."""
        return f"emission.{cls.scale_param}"

    @property
    def series_param_names(self) -> tuple[str, ...]:
        """Parameters that carry a series axis `(N, M)`, or `(N,)` when shared across series."""
        return (self.mean_name(), self.scale_name())

    # --- protocol: parameter layout ---------------------------------------------------------

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """Default (config-free) specs: location and scale only, both free.

        The config-driven mapping (modes, values, priors, covariate links) is `from_config`.
        """
        del cov_names
        mean = ParamSpec(
            name=self.mean_name(),
            shape=(n_states, n_series),
            domain="real",
            mode="free",
            init=np.zeros((n_states, n_series)),
        )
        scale = ParamSpec(
            name=self.scale_name(),
            shape=(n_states, n_series),
            domain="positive",
            mode="free",
            init=np.full((n_states, n_series), self.default_scale),
        )
        return (mean, scale)

    @classmethod
    def from_config(
        cls,
        cfg: EmissionCfg,
        n_states: int,
        n_series: int,
        cov_names: tuple[str, ...],
    ) -> tuple[GaussianFamily, tuple[ParamSpec, ...]]:
        """Map `model.emission` config to a configured family and its `ParamSpec`s."""
        for required in (cls.mean_param, cls.scale_param):
            if required not in cfg.params:
                raise ConfigError(
                    f"{cls.name_for_errors()} emission requires params.{required}",
                    path=f"model.emission.params.{required}",
                )
        covs = cfg.covariates
        for foreign in {"mean", "log_scale", "drift", "log_vol"} - {cls.mean_cov_key, cls.scale_cov_key}:
            if getattr(covs, foreign):
                raise ConfigError(
                    f"the {cls.name_for_errors()} emission takes the covariate keys "
                    f"'{cls.mean_cov_key}' and '{cls.scale_cov_key}', not '{foreign}'",
                    path=f"model.emission.covariates.{foreign}",
                )
        mean_cov = getattr(covs, cls.mean_cov_key)
        scale_cov = getattr(covs, cls.scale_cov_key)
        try:
            mean_idx = tuple(cov_names.index(c) for c in mean_cov)
        except ValueError as exc:
            raise ConfigError(str(exc), path=f"model.emission.covariates.{cls.mean_cov_key}") from exc
        try:
            scale_idx = tuple(cov_names.index(c) for c in scale_cov)
        except ValueError as exc:
            raise ConfigError(str(exc), path=f"model.emission.covariates.{cls.scale_cov_key}") from exc

        if cls.fixed_options:
            family = cls(mean_idx=mean_idx, scale_idx=scale_idx)
        else:
            family = cls(
                scale_by_dt=bool(cfg.scale_by_dt),
                ito_correction=bool(cfg.ito_correction),
                mean_idx=mean_idx,
                scale_idx=scale_idx,
            )

        flat = not cfg.per_series
        specs = [
            _spec_from_paramcfg(
                cls.mean_name(),
                cfg.params[cls.mean_param],
                n_states=n_states,
                n_series=n_series,
                domain="real",
                default_init=0.0,
                flat=flat,
            ),
            _spec_from_paramcfg(
                cls.scale_name(),
                cfg.params[cls.scale_param],
                n_states=n_states,
                n_series=n_series,
                domain="positive",
                default_init=cls.default_scale,
                flat=flat,
            ),
        ]
        for coef, idx in (("gamma", mean_idx), ("delta", scale_idx)):
            if not idx:
                continue
            if coef in cfg.params:
                specs.append(
                    _spec_from_paramcfg(
                        f"emission.{coef}",
                        cfg.params[coef],
                        n_states=n_states,
                        n_series=len(idx),
                        domain="real",
                        default_init=0.0,
                    )
                )
            else:
                specs.append(_coef_spec(f"emission.{coef}", n_states, len(idx)))
        return family, tuple(specs)

    # --- protocol: numerics ------------------------------------------------------------------

    def _mean_sd(self, dt: Array, theta: dict[str, Array], z: Array | None) -> tuple[Array, Array]:
        """`(T, N, M)` mean and sd of the per-step, per-regime density."""
        mu = theta[self.mean_name()]
        sigma = theta[self.scale_name()]
        mu2 = mu[:, None] if mu.ndim == 1 else mu  # (N, M)
        sigma2 = sigma[:, None] if sigma.ndim == 1 else sigma  # (N, M)
        n_states = mu2.shape[0]
        t = dt.shape[0]

        mean_adj = jnp.zeros((t, n_states))
        if self.mean_idx and "emission.gamma" in theta and z is not None:
            z_mean = z[:, jnp.asarray(self.mean_idx)]  # (T, K_mean)
            mean_adj = z_mean @ theta["emission.gamma"].T  # (T, N)

        logscale_adj = jnp.zeros((t, n_states))
        if self.scale_idx and "emission.delta" in theta and z is not None:
            z_scale = z[:, jnp.asarray(self.scale_idx)]  # (T, K_logscale)
            logscale_adj = z_scale @ theta["emission.delta"].T  # (T, N)

        mu_t = mu2[None, :, :] + mean_adj[:, :, None]  # (T, N, M)
        sigma_t = jnp.exp(jnp.log(sigma2)[None, :, :] + logscale_adj[:, :, None])  # (T, N, M)

        loc = mu_t - 0.5 * sigma_t**2 if self.ito_correction else mu_t
        if self.scale_by_dt:
            dt_ = dt[:, None, None]
            return loc * dt_, sigma_t * jnp.sqrt(dt_)
        return loc, sigma_t

    def log_prob_series(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` per-regime, per-series log-density (E9-T01: `ProductEmission` masks per series)."""
        mean_t, sd_t = self._mean_sd(dt, theta, z)
        return normal_logpdf(y[:, None, :], mean_t, sd_t)

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` log-density, summed over the `M` series."""
        return jnp.sum(self.log_prob_series(y, dt, theta, z), axis=-1)

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
        """No path-dependent state."""
        return None

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Identity: no recursive carry to advance."""
        del y_t, theta
        return carry

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array) -> dict[str, Array]:
        """Weighted sums for the closed-form M-step (E5-T06), covariate-free, single series.

        `y (T, 1)` (or `(T,)`), `dt (T,)`, `gamma (T, N)` regime weights (zero at unobserved steps).
        With `s = dt` if `scale_by_dt` else `1` and `m_i = mean_i - scale_i**2 / 2` (Itô) or `mean_i`, the
        maximum-likelihood estimates are `m_i = w_y / w_s` and
        `scale_i**2 = (w_y2_s - 2 m_i w_y + m_i**2 w_s) / w`.

        Returns:
            `(N,)` arrays `w = sum w_t`, `w_dt = sum w_t s_t`, `w_y = sum w_t y_t`,
            `w_y2_dt = sum w_t y_t**2 / s_t`.
        """
        y1 = y[:, 0] if y.ndim == 2 else y
        s = dt if self.scale_by_dt else jnp.ones_like(dt)
        return {
            "w": jnp.sum(gamma, axis=0),
            "w_dt": jnp.sum(gamma * s[:, None], axis=0),
            "w_y": jnp.sum(gamma * y1[:, None], axis=0),
            "w_y2_dt": jnp.sum(gamma * (y1**2 / s)[:, None], axis=0),
        }
