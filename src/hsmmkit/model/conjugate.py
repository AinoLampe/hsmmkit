"""Conjugate priors of the emission families (E11-T04, E13-T05a): posterior updates and VB expectations.

Each prior has `update(stats)` (posterior hyperparameters from the family's `sufficient_stats`; the prior
itself for zero statistics), `zero_stats(...)`, `expected_log_params(q)` (digamma forms for VB),
`kl(q)` (the Kullback-Leibler divergence `KL(q || prior)`, summed over regimes and series) and
`mean_params(q)` (the posterior means as parameters of the family, constrained scale). Shapes follow the
statistics: `(N, M)` for Poisson / Bernoulli / Gaussian, `(N, K)` for Categorical.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from hsmmkit.utils.typing import Array

__all__ = [
    "BetaConjugate",
    "DirichletConjugate",
    "GammaConjugate",
    "NormalGamma",
    "beta_kl",
    "dirichlet_kl",
    "gamma_kl",
    "posterior",
]


def gamma_kl(a: Array, b: Array, a0: Array | float, b0: Array | float) -> Array:
    """Elementwise `KL(Gamma(a, b) || Gamma(a0, b0))` (shape, rate)."""
    return (
        (a - a0) * jsp.digamma(a)
        - jsp.gammaln(a)
        + jsp.gammaln(a0)
        + a0 * (jnp.log(b) - jnp.log(b0))
        + a * (b0 - b) / b
    )


def beta_kl(a: Array, b: Array, a0: Array | float, b0: Array | float) -> Array:
    """Elementwise `KL(Beta(a, b) || Beta(a0, b0))`."""
    psi_sum = jsp.digamma(a + b)
    return (
        jsp.gammaln(a0)
        + jsp.gammaln(b0)
        - jsp.gammaln(a0 + b0)
        - jsp.gammaln(a)
        - jsp.gammaln(b)
        + jsp.gammaln(a + b)
        + (a - a0) * (jsp.digamma(a) - psi_sum)
        + (b - b0) * (jsp.digamma(b) - psi_sum)
    )


def dirichlet_kl(alpha: Array, alpha0: Array | float, mask: Array | None = None) -> Array:
    """`KL(Dirichlet(alpha) || Dirichlet(alpha0))` over the last axis, one value per leading index.

    `mask` (same shape as `alpha`) selects the categories in the support (the others carry no mass and are
    left out of every sum).
    """
    a0 = jnp.broadcast_to(jnp.asarray(alpha0, dtype=alpha.dtype), alpha.shape)
    m = jnp.ones(alpha.shape, dtype=bool) if mask is None else mask
    a = jnp.where(m, alpha, 1.0)
    a0 = jnp.where(m, a0, 1.0)
    total, total0 = jnp.sum(jnp.where(m, a, 0.0), axis=-1), jnp.sum(jnp.where(m, a0, 0.0), axis=-1)
    elog = jsp.digamma(a) - jsp.digamma(total)[..., None]
    terms = jnp.where(m, -jsp.gammaln(a) + jsp.gammaln(a0) + (a - a0) * elog, 0.0)
    return jsp.gammaln(total) - jsp.gammaln(total0) + jnp.sum(terms, axis=-1)


@dataclass(frozen=True)
class GammaConjugate:
    """`Gamma(a0, b0)` (shape, rate) on a Poisson rate per unit exposure."""

    a0: float = 1.0
    b0: float = 1.0

    def zero_stats(self, n_states: int, n_series: int) -> dict[str, Array]:
        """Statistics of no data."""
        return {"w_s": jnp.zeros((n_states,)), "wy": jnp.zeros((n_states, n_series))}

    def update(self, stats: dict[str, Array]) -> dict[str, Array]:
        """`a = a0 + sum(gamma y)`, `b = b0 + sum(gamma s)`."""
        return {"a": self.a0 + stats["wy"], "b": self.b0 + stats["w_s"][:, None] + 0.0 * stats["wy"]}

    def expected_log_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """`E[log lambda] = digamma(a) - log b`, `E[lambda] = a / b`."""
        return {"log_rate": jsp.digamma(q["a"]) - jnp.log(q["b"]), "rate": q["a"] / q["b"]}

    def kl(self, q: dict[str, Array]) -> Array:
        """`KL(Gamma(a, b) || Gamma(a0, b0))` summed over regimes and series."""
        return jnp.sum(gamma_kl(q["a"], q["b"], self.a0, self.b0))

    def mean_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """Posterior-mean rate as `emission.eta = log rate`."""
        return {"emission.eta": jnp.log(q["a"] / q["b"])}

    def sample_params(self, key: Array, q: dict[str, Array]) -> dict[str, Array]:
        """One draw of the rate from `Gamma(a, b)` as `emission.eta = log rate` (Gibbs)."""
        return {"emission.eta": jnp.log(jax.random.gamma(key, q["a"]) / q["b"])}

    def sd_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """Posterior SD of `emission.eta = log rate` (delta method): `1 / sqrt(a)`."""
        return {"emission.eta": 1.0 / jnp.sqrt(q["a"])}


@dataclass(frozen=True)
class BetaConjugate:
    """`Beta(a0, b0)` on a Bernoulli probability."""

    a0: float = 1.0
    b0: float = 1.0

    def zero_stats(self, n_states: int, n_series: int) -> dict[str, Array]:
        """Statistics of no data."""
        return {"w": jnp.zeros((n_states,)), "wy": jnp.zeros((n_states, n_series))}

    def update(self, stats: dict[str, Array]) -> dict[str, Array]:
        """`a = a0 + sum(gamma y)`, `b = b0 + sum(gamma (1 - y))`."""
        return {"a": self.a0 + stats["wy"], "b": self.b0 + stats["w"][:, None] - stats["wy"]}

    def expected_log_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """`E[log p]` and `E[log (1 - p)]` (digamma forms)."""
        total = jsp.digamma(q["a"] + q["b"])
        return {"log_p": jsp.digamma(q["a"]) - total, "log_1mp": jsp.digamma(q["b"]) - total}

    def kl(self, q: dict[str, Array]) -> Array:
        """`KL(Beta(a, b) || Beta(a0, b0))` summed over regimes and series."""
        return jnp.sum(beta_kl(q["a"], q["b"], self.a0, self.b0))

    def mean_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """Posterior-mean probability as `emission.eta = logit p`."""
        return {"emission.eta": jnp.log(q["a"]) - jnp.log(q["b"])}

    def sample_params(self, key: Array, q: dict[str, Array]) -> dict[str, Array]:
        """One draw of the probability from `Beta(a, b)` as `emission.eta = logit p` (Gibbs)."""
        p = jax.random.beta(key, q["a"], q["b"])
        return {"emission.eta": jnp.log(p) - jnp.log1p(-p)}

    def sd_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """Posterior SD of `emission.eta = logit p` (delta method): `sqrt(1/a + 1/b)`."""
        return {"emission.eta": jnp.sqrt(1.0 / q["a"] + 1.0 / q["b"])}


@dataclass(frozen=True)
class DirichletConjugate:
    """`Dirichlet(alpha0)` (symmetric) on categorical probabilities."""

    alpha0: float = 1.0

    def zero_stats(self, n_states: int, n_categories: int) -> dict[str, Array]:
        """Statistics of no data."""
        return {"w": jnp.zeros((n_states,)), "counts": jnp.zeros((n_states, n_categories))}

    def update(self, stats: dict[str, Array]) -> dict[str, Array]:
        """`alpha = alpha0 + sum(gamma 1[y = k])`."""
        return {"alpha": self.alpha0 + stats["counts"]}

    def expected_log_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """`E[log p_k] = digamma(alpha_k) - digamma(sum alpha)`."""
        return {"log_p": jsp.digamma(q["alpha"]) - jsp.digamma(jnp.sum(q["alpha"], axis=-1, keepdims=True))}

    def kl(self, q: dict[str, Array]) -> Array:
        """`KL(Dirichlet(alpha) || Dirichlet(alpha0))` summed over regimes."""
        return jnp.sum(dirichlet_kl(q["alpha"], self.alpha0))

    def mean_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """Posterior-mean probabilities as `emission.eta = log(p[1:] / p[0])`."""
        p = q["alpha"] / jnp.sum(q["alpha"], axis=-1, keepdims=True)
        return {"emission.eta": jnp.log(p[:, 1:]) - jnp.log(p[:, :1])}

    def sample_params(self, key: Array, q: dict[str, Array]) -> dict[str, Array]:
        """One draw of the probabilities from `Dirichlet(alpha)` as `emission.eta` (Gibbs)."""
        p = jax.random.dirichlet(key, q["alpha"])
        return {"emission.eta": jnp.log(p[:, 1:]) - jnp.log(p[:, :1])}

    def sd_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """Posterior SD of `emission.eta = log(p[1:] / p[0])` (delta method)."""
        alpha = q["alpha"]
        return {"emission.eta": jnp.sqrt(1.0 / alpha[:, 1:] + 1.0 / alpha[:, :1])}


@dataclass(frozen=True)
class NormalGamma:
    """Normal-Gamma prior `(m0, kappa0, a0, b0)` on the mean and precision of every series of every regime.

    `mean | tau ~ N(m0, 1 / (kappa0 tau))`, `tau ~ Gamma(a0, b0)` (shape, rate). The defaults are weakly
    informative for standardized data; `with_data_scale` rescales `b0` and `m0` to the data.
    """

    m0: float = 0.0
    kappa0: float = 0.01
    a0: float = 1.0
    b0: float = 1.0

    def with_data_scale(self, mean: float, var: float) -> NormalGamma:
        """The prior centred on the data: `m0 = mean`, `E[tau] = a0 / b0 = 1 / var`."""
        return NormalGamma(float(mean), self.kappa0, self.a0, self.a0 * float(var))

    def zero_stats(self, n_states: int, n_series: int) -> dict[str, Array]:
        """The statistics of no data (`update(zero_stats)` returns the prior)."""
        z = jnp.zeros((n_states, n_series))
        return {"w": jnp.zeros((n_states,)), "w_y": z, "w_y2_dt": z}

    @staticmethod
    def _nm(x: Array, n: int) -> Array:
        return x[:, None] if x.ndim == 1 else jnp.reshape(x, (n, -1))

    def update(self, stats: dict[str, Array]) -> dict[str, Array]:
        """Posterior `(m, kappa, a, b)`, each `(N, M)`, from the family's `sufficient_stats`."""
        n_states = stats["w"].shape[0]
        w = stats["w"][:, None]
        s1, s2 = self._nm(stats["w_y"], n_states), self._nm(stats["w_y2_dt"], n_states)
        n_safe = jnp.where(w > 0, w, 1.0)
        ybar = s1 / n_safe
        scatter = jnp.maximum(s2 - w * ybar**2, 0.0)
        kappa = self.kappa0 + w + 0.0 * s1
        m = (self.kappa0 * self.m0 + s1) / kappa
        b = self.b0 + 0.5 * (scatter + self.kappa0 * w * (ybar - self.m0) ** 2 / kappa)
        return {"m": m, "kappa": kappa, "a": self.a0 + 0.5 * w + 0.0 * s1, "b": b}

    def expected_log_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """`E[log tau] = digamma(a) - log b`, `E[tau] = a / b`."""
        return {"log_tau": jsp.digamma(q["a"]) - jnp.log(q["b"]), "tau": q["a"] / q["b"]}

    def kl(self, q: dict[str, Array]) -> Array:
        """`KL(NG(m, kappa, a, b) || NG(m0, kappa0, a0, b0))` summed over regimes and series."""
        m, kappa, a, b = q["m"], q["kappa"], q["a"], q["b"]
        k0 = self.kappa0
        normal = 0.5 * (k0 / kappa - 1.0 + k0 * (a / b) * (m - self.m0) ** 2 + jnp.log(kappa / k0))
        return jnp.sum(gamma_kl(a, b, self.a0, self.b0) + normal)

    def mean_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """Posterior mean `m` and `scale = 1 / sqrt(E[tau])` as `emission.mean` / `emission.scale`."""
        return {"emission.mean": q["m"], "emission.scale": jnp.sqrt(q["b"] / q["a"])}

    def sample_params(self, key: Array, q: dict[str, Array]) -> dict[str, Array]:
        """One draw `tau ~ Gamma(a, b)`, `mean | tau ~ N(m, 1 / (kappa tau))` (Gibbs)."""
        k1, k2 = jax.random.split(key)
        tau = jax.random.gamma(k1, q["a"]) / q["b"]
        mean = q["m"] + jax.random.normal(k2, tau.shape) / jnp.sqrt(q["kappa"] * tau)
        return {"emission.mean": mean, "emission.scale": 1.0 / jnp.sqrt(tau)}

    def sd_params(self, q: dict[str, Array]) -> dict[str, Array]:
        """Posterior SD of the mean (`sqrt(b / (kappa (a - 1)))`) and, by the delta method, of the scale."""
        a, b, kappa = q["a"], q["b"], q["kappa"]
        sd_mean = jnp.sqrt(b / (kappa * jnp.maximum(a - 1.0, 0.5)))
        tau = a / b
        sd_scale = 0.5 * tau**-1.5 * jnp.sqrt(a) / b
        return {"emission.mean": sd_mean, "emission.scale": sd_scale}


def posterior(
    stats: dict[str, Array], prior: GammaConjugate | BetaConjugate | DirichletConjugate | NormalGamma
) -> dict[str, Array]:
    """The posterior hyperparameters of `prior` given `stats` (`prior.update`)."""
    return prior.update(stats)
