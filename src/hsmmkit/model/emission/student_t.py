"""Student-t emission: heavy-tailed per-state observations scaled by `dt`, with covariate links (E3-T02).

`y_t | i ~ m_i(t) * dt_t + s_i(t) * sqrt(dt_t) * t_{nu_i}` with `nu_i > 2` and

    m_i(t)     = m_i + gamma_i . z^e_t          (location link)
    log s_i(t) = log s_i + delta_i . z^e_t      (log-scale link)

`s_i` is a *scale*, not the standard deviation: `sd = s * sqrt(nu / (nu - 2))` (per unit `sqrt(dt)`).
`m_i` is the per-clock-unit location itself (no Itô term, unlike `gaussian(ito_correction=True)`).
Parameters are in clock units (CLAUDE.md §3.1); no code here branches on the clock name.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Literal

import jax
import jax.numpy as jnp
import jax.scipy.stats
import numpy as np

from hsmmkit.model.dists import mvt_logpdf, sample_student_t, student_t_logpdf, unpack_lower
from hsmmkit.model.emission.base import BaseEmission, Support
from hsmmkit.model.emission.gaussian import (
    _chol_spec,
    _chol_spec_from_paramcfg,
    _coef_spec,
    _spec_from_paramcfg,
)
from hsmmkit.model.params.spec import ParamSpec
from hsmmkit.plugins import register
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.config.schema import EmissionCfg

NU_TIE_GROUP = "nu"
_DEFAULT_NU_INIT = 8.0


@register("emission", "student_t")
@dataclass(frozen=True)
class StudentTFamily(BaseEmission):
    """Per-state location-scale Student-t emission.

    `loc_idx`/`logscale_idx` select, by column, which entries of the `(T, K_e)` emission covariate
    block `z` feed the location / log-scale links (built by `from_config`).
    """

    name: str = "student_t"
    native_time: Literal["continuous", "discrete"] = field(default="discrete", init=False)
    support: Support = "real"
    scale_by_dt: bool = False
    loc_idx: tuple[int, ...] = ()
    logscale_idx: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """`native_time` follows `scale_by_dt`."""
        object.__setattr__(self, "native_time", "continuous" if self.scale_by_dt else "discrete")

    # Parameters that carry a series axis `(N, M)`, or `(N,)` when shared across series (`ProductEmission`).
    series_param_names: ClassVar[tuple[str, ...]] = ("emission.m", "emission.s")

    # --- protocol: parameter layout ---------------------------------------------------------

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """Default (config-free) specs: `m`, `s` free and one free `nu` per state."""
        del cov_names
        return (
            ParamSpec(
                name="emission.m",
                shape=(n_states, n_series),
                domain="real",
                mode="free",
                init=np.zeros((n_states, n_series)),
            ),
            ParamSpec(
                name="emission.s",
                shape=(n_states, n_series),
                domain="positive",
                mode="free",
                init=np.full((n_states, n_series), 0.1),
            ),
            ParamSpec(
                name="emission.nu",
                shape=(n_states,),
                domain="nu_gt_2",
                mode="free",
                init=np.full((n_states,), _DEFAULT_NU_INIT),
            ),
        )

    @classmethod
    def from_config(
        cls,
        cfg: EmissionCfg,
        n_states: int,
        n_series: int,
        cov_names: tuple[str, ...],
    ) -> tuple[StudentTFamily, tuple[ParamSpec, ...]]:
        """Map `model.emission` config to a configured `StudentTFamily` and its `ParamSpec`s.

        A configuration with `covariance: full` or with the multivariate parameter names (`mean`, `scale` or
        `chol`) selects the joint `MultivariateStudentT` (E11-T03); `m` / `s` keep the Part I code path.
        """
        if cfg.covariance == "full" or "mean" in cfg.params:
            return MultivariateStudentT.from_config(cfg, n_states, n_series, cov_names)  # type: ignore[return-value]
        try:
            loc_idx = tuple(cov_names.index(c) for c in cfg.covariates.drift)
        except ValueError as exc:
            raise ConfigError(str(exc), path="model.emission.covariates.drift") from exc
        try:
            logscale_idx = tuple(cov_names.index(c) for c in cfg.covariates.log_vol)
        except ValueError as exc:
            raise ConfigError(str(exc), path="model.emission.covariates.log_vol") from exc

        for required in ("m", "s", "nu"):
            if required not in cfg.params:
                raise ConfigError(
                    f"student_t emission requires params.{required}",
                    path=f"model.emission.params.{required}",
                )

        nu_spec = _spec_from_paramcfg(
            "emission.nu",
            cfg.params["nu"],
            n_states=n_states,
            n_series=1,
            domain="nu_gt_2",
            default_init=_DEFAULT_NU_INIT,
            flat=True,
        )
        if nu_spec.tie_group is None and nu_spec.mode == "tied":
            nu_spec = replace(nu_spec, tie_group=NU_TIE_GROUP)

        specs = [
            _spec_from_paramcfg(
                "emission.m",
                cfg.params["m"],
                n_states=n_states,
                n_series=n_series,
                domain="real",
                default_init=0.0,
                flat=not cfg.per_series,
            ),
            _spec_from_paramcfg(
                "emission.s",
                cfg.params["s"],
                n_states=n_states,
                n_series=n_series,
                domain="positive",
                default_init=0.1,
                flat=not cfg.per_series,
            ),
            nu_spec,
        ]
        for name, idx in (("gamma", loc_idx), ("delta", logscale_idx)):
            if not idx:
                continue
            if name in cfg.params:
                specs.append(
                    _spec_from_paramcfg(
                        f"emission.{name}",
                        cfg.params[name],
                        n_states=n_states,
                        n_series=len(idx),
                        domain="real",
                        default_init=0.0,
                    )
                )
            else:
                specs.append(_coef_spec(f"emission.{name}", n_states, len(idx)))
        return cls(scale_by_dt=bool(cfg.scale_by_dt), loc_idx=loc_idx, logscale_idx=logscale_idx), tuple(
            specs
        )

    # --- protocol: numerics ------------------------------------------------------------------

    def _loc_scale_nu(
        self, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> tuple[Array, Array, Array]:
        """`(T, N, M)` location and scale, `(N,)` degrees of freedom of the per-step density."""
        m = theta["emission.m"]
        s = theta["emission.s"]
        m2 = m[:, None] if m.ndim == 1 else m  # (N, M)
        s2 = s[:, None] if s.ndim == 1 else s  # (N, M)
        n_states = m2.shape[0]
        t = dt.shape[0]

        loc_adj = jnp.zeros((t, n_states))
        if self.loc_idx and "emission.gamma" in theta and z is not None:
            loc_adj = z[:, jnp.asarray(self.loc_idx)] @ theta["emission.gamma"].T  # (T, N)

        logscale_adj = jnp.zeros((t, n_states))
        if self.logscale_idx and "emission.delta" in theta and z is not None:
            logscale_adj = z[:, jnp.asarray(self.logscale_idx)] @ theta["emission.delta"].T  # (T, N)

        loc_t = m2[None, :, :] + loc_adj[:, :, None]
        scale_t = jnp.exp(jnp.log(s2)[None, :, :] + logscale_adj[:, :, None])
        if self.scale_by_dt:
            dt_ = dt[:, None, None]
            loc_t = loc_t * dt_
            scale_t = scale_t * jnp.sqrt(dt_)
        return loc_t, scale_t, theta["emission.nu"]

    def log_prob_series(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` per-regime, per-series log-density (E9-T01: `ProductEmission` masks per series)."""
        loc_t, scale_t, nu = self._loc_scale_nu(dt, theta, z)
        return student_t_logpdf(y[:, None, :], loc_t, scale_t, nu[None, :, None])

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` log-density, summed over the `M` series."""
        return jnp.sum(self.log_prob_series(y, dt, theta, z), axis=-1)

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` per-regime, per-series predictive CDF at `y` (E5-T04's PIT diagnostic)."""
        loc_t, scale_t, nu = self._loc_scale_nu(dt, theta, z)
        z_score = (y[:, None, :] - loc_t) / scale_t
        out: Array = jax.scipy.stats.t.cdf(z_score, df=nu[None, :, None])  # type: ignore[attr-defined]
        return out

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, M)` samples, one draw per step from the active regime's density."""
        loc_t, scale_t, nu = self._loc_scale_nu(dt, theta, z)
        idx = jnp.arange(dt.shape[0])
        loc_sel = loc_t[idx, regime, :]  # (T, M)
        scale_sel = scale_t[idx, regime, :]
        nu_sel = nu[regime][:, None]  # (T, 1)
        return sample_student_t(key, loc_sel, scale_sel, nu_sel, shape=loc_sel.shape)

    def moments(
        self, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> tuple[Array, Array]:
        """Mean and variance `(T, M)`; variance `inf` where `nu <= 2`, mean `nan` where `nu <= 1`."""
        loc_t, scale_t, nu = self._loc_scale_nu(dt, theta, z)
        idx = jnp.arange(dt.shape[0])
        loc, scale = loc_t[idx, regime, :], scale_t[idx, regime, :]
        nu_t = nu[regime][:, None]
        var = jnp.where(nu_t > 2.0, scale**2 * nu_t / jnp.where(nu_t > 2.0, nu_t - 2.0, 1.0), jnp.inf)
        return jnp.where(nu_t > 1.0, loc, jnp.nan), var

    def init_carry(self) -> object:
        """Student-t has no path-dependent state."""
        return None

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Identity: no recursive carry to advance."""
        del y_t, theta
        return carry

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array, z: Array | None = None) -> object | None:
        """No EM sufficient statistics in v1 (hook for E5-T06)."""
        del y, dt, gamma, z
        return None


@dataclass(frozen=True)
class MultivariateStudentT(BaseEmission):
    """Joint multivariate Student-t with a diagonal or full scale matrix (selected by `student_t` configs).

    `y_t | i ~ t_nu_i(m_i(t) s(dt), Sigma_i(t) s(dt))`, `Sigma_i = L_i L_i^T` (`diag`: `L_i = diag(scale_i)`),
    `Sigma_i(t) = e^{2 delta_i . z_t} Sigma_i`, one `nu_i > 2` per regime shared by all series. Built by
    `StudentTFamily.from_config` (the config name is `student_t`; it has no registry entry of its own).
    """

    name: str = "student_t"
    native_time: Literal["continuous", "discrete"] = field(default="discrete", init=False)
    support: Support = "real"
    scale_by_dt: bool = False
    covariance: Literal["diag", "full"] = "diag"
    mean_idx: tuple[int, ...] = ()
    scale_idx: tuple[int, ...] = ()
    joint: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """`native_time` follows `scale_by_dt`."""
        object.__setattr__(self, "native_time", "continuous" if self.scale_by_dt else "discrete")

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """Default specs: `mean`, `scale` (diag) or `chol` (full), one `nu` per regime."""
        del cov_names
        mean = ParamSpec(
            name="emission.mean",
            shape=(n_states, n_series),
            domain="real",
            mode="free",
            init=np.zeros((n_states, n_series)),
        )
        nu = ParamSpec(
            name="emission.nu",
            shape=(n_states,),
            domain="nu_gt_2",
            mode="free",
            init=np.full((n_states,), _DEFAULT_NU_INIT),
        )
        if self.covariance == "full":
            scale = _chol_spec(n_states, n_series, None, mode="free", prior=None)
        else:
            scale = ParamSpec(
                name="emission.scale",
                shape=(n_states, n_series),
                domain="positive",
                mode="free",
                init=np.full((n_states, n_series), 0.1),
            )
        return (mean, scale, nu)

    @classmethod
    def from_config(
        cls, cfg: EmissionCfg, n_states: int, n_series: int, cov_names: tuple[str, ...]
    ) -> tuple[MultivariateStudentT, tuple[ParamSpec, ...]]:
        """Map `model.emission` to the family and its specs (`ito_correction` is not allowed)."""
        if cfg.ito_correction:
            raise ConfigError(
                "the student_t emission has no Ito correction", path="model.emission.ito_correction"
            )
        full = cfg.covariance == "full"
        scale_key = "chol" if full else "scale"
        for required in ("mean", scale_key, "nu"):
            if required not in cfg.params:
                raise ConfigError(
                    f"student_t emission requires params.{required}", path=f"model.emission.params.{required}"
                )
        idx: dict[str, tuple[int, ...]] = {}
        for key, names in (("drift", cfg.covariates.drift), ("log_vol", cfg.covariates.log_vol)):
            try:
                idx[key] = tuple(cov_names.index(c) for c in names)
            except ValueError as exc:
                raise ConfigError(str(exc), path=f"model.emission.covariates.{key}") from exc
        nu_spec = _spec_from_paramcfg(
            "emission.nu",
            cfg.params["nu"],
            n_states=n_states,
            n_series=1,
            domain="nu_gt_2",
            default_init=_DEFAULT_NU_INIT,
            flat=True,
        )
        if nu_spec.tie_group is None and nu_spec.mode == "tied":
            nu_spec = replace(nu_spec, tie_group=NU_TIE_GROUP)
        specs = [
            _spec_from_paramcfg(
                "emission.mean",
                cfg.params["mean"],
                n_states=n_states,
                n_series=n_series,
                domain="real",
                default_init=0.0,
            ),
            _chol_spec_from_paramcfg(cfg.params["chol"], n_states, n_series)
            if full
            else _spec_from_paramcfg(
                "emission.scale",
                cfg.params["scale"],
                n_states=n_states,
                n_series=n_series,
                domain="positive",
                default_init=0.1,
            ),
            nu_spec,
        ]
        for name, key in (("gamma", "drift"), ("delta", "log_vol")):
            if not idx[key]:
                continue
            if name in cfg.params:
                specs.append(
                    _spec_from_paramcfg(
                        f"emission.{name}",
                        cfg.params[name],
                        n_states=n_states,
                        n_series=len(idx[key]),
                        domain="real",
                        default_init=0.0,
                    )
                )
            else:
                specs.append(_coef_spec(f"emission.{name}", n_states, len(idx[key])))
        family = cls(
            scale_by_dt=bool(cfg.scale_by_dt),
            covariance=cfg.covariance,
            mean_idx=idx["drift"],
            scale_idx=idx["log_vol"],
        )
        return family, tuple(specs)

    def _parts(self, dt: Array, theta: dict[str, Array], z: Array | None) -> tuple[Array, Array, Array]:
        """`(T, N, M)` location, `(T, N, M, M)` scale factor `L~`, `(N,)` degrees of freedom."""
        mu = theta["emission.mean"]
        n_states, m = mu.shape
        if self.covariance == "full":
            chol = unpack_lower(theta["emission.chol"], m)
        else:
            chol = jax.vmap(jnp.diag)(theta["emission.scale"])
        t = dt.shape[0]
        mean_adj = jnp.zeros((t, n_states))
        if self.mean_idx and "emission.gamma" in theta and z is not None:
            mean_adj = z[:, jnp.asarray(self.mean_idx)] @ theta["emission.gamma"].T
        logscale_adj = jnp.zeros((t, n_states))
        if self.scale_idx and "emission.delta" in theta and z is not None:
            logscale_adj = z[:, jnp.asarray(self.scale_idx)] @ theta["emission.delta"].T
        mult = jnp.exp(logscale_adj)
        loc = mu[None] + mean_adj[:, :, None]
        if self.scale_by_dt:
            loc = loc * dt[:, None, None]
            mult = mult * jnp.sqrt(dt)[:, None]
        return loc, mult[:, :, None, None] * chol[None], theta["emission.nu"]

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` joint log-density."""
        loc, chol_t, nu = self._parts(dt, theta, z)
        return mvt_logpdf(y[:, None, :], loc, chol_t, nu[None, :])

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N, M)` marginal predictive CDFs (the marginals of a multivariate t are Student-t)."""
        loc, chol_t, nu = self._parts(dt, theta, z)
        sd = jnp.sqrt(jnp.sum(chol_t**2, axis=-1))
        out: Array = jax.scipy.stats.t.cdf((y[:, None, :] - loc) / sd, df=nu[None, :, None])  # type: ignore[attr-defined]
        return out

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`mu + L z / sqrt(g / nu)` with `z ~ N(0, I)`, `g ~ chi2_nu`."""
        loc, chol_t, nu = self._parts(dt, theta, z)
        idx = jnp.arange(dt.shape[0])
        k1, k2 = jax.random.split(key)
        eps = jax.random.normal(k1, (dt.shape[0], loc.shape[-1]), dtype=loc.dtype)
        g = 2.0 * jax.random.gamma(k2, nu[regime] / 2.0, shape=(dt.shape[0],), dtype=loc.dtype)
        shock = jnp.einsum("tij,tj->ti", chol_t[idx, regime], eps)
        return loc[idx, regime, :] + shock / jnp.sqrt(g / nu[regime])[:, None]

    def moments(
        self, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> tuple[Array, Array]:
        """Mean `loc` and marginal variances `nu / (nu - 2) diag(Sigma)` (`nu > 2` by construction)."""
        loc, chol_t, nu = self._parts(dt, theta, z)
        idx = jnp.arange(dt.shape[0])
        diag = jnp.sum(chol_t[idx, regime] ** 2, axis=-1)
        n = nu[regime][:, None]
        return loc[idx, regime, :], diag * n / (n - 2.0)

    def init_carry(self) -> object:
        """No path-dependent state."""
        return None

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Identity."""
        del y_t, theta
        return carry

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array, z: Array | None = None) -> object | None:
        """None (the multivariate t is estimated by ECM, E13-T02b)."""
        del y, dt, gamma, z
        return None
