"""Student-t emission: heavy-tailed per-state log-returns scaled by `dt`, with covariate links (E3-T02).

`y_t | i ~ m_i(t) * dt_t + s_i(t) * sqrt(dt_t) * t_{nu_i}` with `nu_i > 2` and

    m_i(t)     = m_i + gamma_i . z^e_t          (location link)
    log s_i(t) = log s_i + delta_i . z^e_t      (log-scale link)

`s_i` is a *scale*, not the standard deviation: `sd = s * sqrt(nu / (nu - 2))` (per unit `sqrt(dt)`).
`m_i` is the per-clock-unit location of the log-return itself (no `-sigma^2 / 2` term, unlike GBM).
Parameters are in clock units (CLAUDE.md §3.1); no code here branches on the clock name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Literal

import jax.numpy as jnp
import jax.scipy.stats
import numpy as np

from regimekit.model.dists import sample_student_t, student_t_logpdf
from regimekit.model.emission.base import Support
from regimekit.model.emission.gbm import _coef_spec, _spec_from_paramcfg
from regimekit.model.params.spec import ParamSpec
from regimekit.plugins import register
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

if TYPE_CHECKING:
    from regimekit.config.schema import EmissionCfg

NU_TIE_GROUP = "nu"
_DEFAULT_NU_INIT = 8.0


@register("emission", "student_t")
@dataclass(frozen=True)
class StudentTFamily:
    """Per-state location-scale Student-t emission.

    `loc_idx`/`logscale_idx` select, by column, which entries of the `(T, K_e)` emission covariate
    block `z` feed the location / log-scale links (built by `from_config`).
    """

    name: str = "student_t"
    native_time: Literal["continuous", "discrete"] = "continuous"
    support: Support = "real"
    scale_by_dt: bool = True
    loc_idx: tuple[int, ...] = ()
    logscale_idx: tuple[int, ...] = ()

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
        """Map `model.emission` config to a configured `StudentTFamily` and its `ParamSpec`s."""
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
        return cls(loc_idx=loc_idx, logscale_idx=logscale_idx), tuple(specs)

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

        dt_ = dt[:, None, None]
        loc_t = (m2[None, :, :] + loc_adj[:, :, None]) * dt_
        scale_t = jnp.exp(jnp.log(s2)[None, :, :] + logscale_adj[:, :, None]) * jnp.sqrt(dt_)
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

    def init_carry(self) -> object:
        """Student-t has no path-dependent state."""
        return None

    def step_carry(self, carry: object, y_t: Array, theta: dict[str, Array]) -> object:
        """Identity: no recursive carry to advance."""
        del y_t, theta
        return carry

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array) -> object | None:
        """No EM sufficient statistics in v1 (hook for E5-T06)."""
        del y, dt, gamma
        return None
