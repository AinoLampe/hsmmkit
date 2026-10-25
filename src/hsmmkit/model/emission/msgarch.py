"""EXPERIMENTAL: Markov-switching GARCH(1,1) emission, Haas et al. (2004) form (E9-T03, ADR-0022).

Every regime `i` runs its own variance recursion on the *observed* series, in parallel for all regimes,

    eps_{i,t} = y_t - m_i * dt_t
    h_{i,t+1} = omega_i + alpha_i * eps_{i,t}**2 + beta_i * h_{i,t},        y_t | i ~ N(m_i * dt_t, h_{i,t}),

with `h_{i,1} = omega_i / (1 - alpha_i - beta_i)` (the unconditional variance, denominator floored). Because
`h_{i,t}` depends on `y` only and not on the regime path, the recursion can run in one `lax.scan` before the
forward pass; the engine does exactly that when `family.init_carry() is not None`
(`engine/expanded_hmm.py::_regime_log_emis`). A step with no observation feeds `eps**2 -> h` (its
expectation) into the recursion.

Units (ADR-0022, the annualization exception): the family is `native_time="discrete"`. `omega`, `alpha`,
`beta` are per step and dimensionless-per-step; `dt` is only used for the mean, `m_i` is per clock unit
(`m_i * dt_t` per step). Single series (`M = 1`) only. `sample`/`cdf` (hence forecasting and PIT) are not
implemented: forecasting with a GARCH carry is out of scope for the spike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from regimekit.model.dists import normal_logpdf
from regimekit.model.emission.base import Support
from regimekit.model.params.spec import ParamSpec
from regimekit.utils.typing import Array

EXPERIMENTAL = True
_MIN_GAP = 1e-3  # floor of `1 - alpha - beta` in the unconditional variance


@dataclass(frozen=True)
class MSGARCHFamily:
    """Regime-wise GARCH(1,1) Gaussian emission (single series)."""

    name: str = "msgarch"
    native_time: Literal["continuous", "discrete"] = "discrete"
    support: Support = "real"
    scale_by_dt: bool = False

    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]:
        """`emission.{m,omega,alpha,beta}`, shape `(N,)`; `omega` is `increasing` (labelling guard)."""
        del cov_names
        if n_series != 1:
            raise ValueError("MSGARCHFamily supports a single series")
        return (
            ParamSpec("emission.m", (n_states,), "real", "free", init=np.zeros(n_states)),
            ParamSpec(
                "emission.omega",
                (n_states,),
                "positive",
                "free",
                init=np.linspace(2e-6, 2e-5, n_states),
                constraint="increasing",
            ),
            ParamSpec(
                "emission.alpha",
                (n_states,),
                "interval",
                "free",
                init=np.full(n_states, 0.08),
                bounds=(1e-6, 0.5),
            ),
            ParamSpec(
                "emission.beta",
                (n_states,),
                "interval",
                "free",
                init=np.full(n_states, 0.85),
                bounds=(1e-6, 0.995),
            ),
        )

    # --- carry hooks -------------------------------------------------------------------------

    def init_carry(self) -> object:
        """A non-`None` marker: the engine switches to the carry (scan) emission path."""
        return ()

    def initial_variance(self, theta: dict[str, Array]) -> Array:
        """`(N,)` unconditional variances `omega / (1 - alpha - beta)` (gap floored at `1e-3`)."""
        gap = jnp.maximum(1.0 - theta["emission.alpha"] - theta["emission.beta"], _MIN_GAP)
        return theta["emission.omega"] / gap

    def step_carry(
        self, carry: object, y_t: Array, theta: dict[str, Array], dt_t: Array | None = None
    ) -> object:
        """`h_{t+1}` from `h_t` and the observed `y_t` (`dt_t` defaults to `1`)."""
        h = jnp.asarray(carry)
        step = 1.0 if dt_t is None else dt_t
        eps = y_t - theta["emission.m"] * step
        return theta["emission.omega"] + theta["emission.alpha"] * eps**2 + theta["emission.beta"] * h

    # --- numerics ----------------------------------------------------------------------------

    def log_prob_scan(
        self, y: Array, mask: Array, dt: Array, theta: dict[str, Array], z: Array | None
    ) -> Array:
        """`(T, N)` log-densities with the variance recursion threaded through `lax.scan`.

        `y (T, 1)` must be finite (engines pass `where(mask, y, 0)`); a masked step contributes its
        density but feeds `h` forward instead of `eps**2`; the engine zeroes such rows afterwards.
        """
        del z
        m, alpha, beta, omega = (
            theta[k] for k in ("emission.m", "emission.alpha", "emission.beta", "emission.omega")
        )
        observed = jnp.all(mask, axis=-1)

        def body(h: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
            y_t, obs_t, dt_t = xs
            mean = m * dt_t
            lp = normal_logpdf(y_t, mean, jnp.sqrt(h))
            h_seen = jnp.asarray(self.step_carry(h, y_t, theta, dt_t))
            h_masked = omega + (alpha + beta) * h  # eps^2 -> h for an unobserved step
            return jnp.where(obs_t, h_seen, h_masked), lp

        _, lps = jax.lax.scan(body, self.initial_variance(theta), (y[:, 0], observed, dt))
        return lps

    def log_prob(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """`(T, N)` log-density of a *full* series, every step observed.

        The recursion starts at the unconditional variance, so calling this on a single step (one-step
        scoring, `evaluation.metrics.predictive`) gives the wrong density; use a full series.
        """
        return self.log_prob_scan(y, jnp.ones(y.shape, dtype=bool), dt, theta, z)

    def cdf(self, y: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """Not implemented (PIT with a GARCH carry is out of scope)."""
        raise NotImplementedError("MSGARCHFamily.cdf is not implemented (E9-T03 spike)")

    def sample(self, key: Array, regime: Array, dt: Array, theta: dict[str, Array], z: Array | None) -> Array:
        """Not implemented (forecasting with a GARCH carry is out of scope)."""
        raise NotImplementedError("MSGARCHFamily.sample is not implemented (E9-T03 spike)")

    def sufficient_stats(self, y: Array, dt: Array, gamma: Array) -> object | None:
        """No EM statistics."""
        return None
