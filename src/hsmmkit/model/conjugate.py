"""Conjugate priors of the discrete emission families (E11-T04): posterior updates and VB expectations.

Each prior has `update(stats)` (posterior hyperparameters from the family's `sufficient_stats`; the prior
itself for zero statistics), `zero_stats(...)` and `expected_log_params(q)` (digamma forms for VB). Shapes
follow the statistics: `(N, M)` for Poisson / Bernoulli, `(N, K)` for Categorical.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy.special as jsp

from hsmmkit.utils.typing import Array

__all__ = ["BetaConjugate", "DirichletConjugate", "GammaConjugate", "posterior"]


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


def posterior(
    stats: dict[str, Array], prior: GammaConjugate | BetaConjugate | DirichletConjugate
) -> dict[str, Array]:
    """The posterior hyperparameters of `prior` given `stats` (`prior.update`)."""
    return prior.update(stats)
