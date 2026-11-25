"""Log-prior densities evaluated on constrained `theta`, summed to a scalar (E1-T03a)."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special

from hsmmkit.model.params.bijectors import lower_diag_mask
from hsmmkit.utils.typing import Array

_LOG_2PI = math.log(2.0 * math.pi)


class NormalPrior(eqx.Module):
    """`Normal(mean, sd)` on `theta`."""

    sd: float = eqx.field(static=True)
    mean: float = eqx.field(static=True, default=0.0)

    def log_prob(self, theta: Array) -> Array:
        """`sum(-0.5 * ((theta - mean) / sd) ** 2 - log(sd) - 0.5 * log(2 * pi))`."""
        z = (theta - self.mean) / self.sd
        return jnp.sum(-0.5 * z**2 - jnp.log(self.sd) - 0.5 * _LOG_2PI)


class LogNormalPrior(eqx.Module):
    """`LogNormal(mean_log, sd)` on `theta` (i.e. `Normal(mean_log, sd)` on `log(theta)`)."""

    mean_log: float = eqx.field(static=True)
    sd: float = eqx.field(static=True)

    def log_prob(self, theta: Array) -> Array:
        """`sum(-0.5 * ((log(theta) - mean_log) / sd) ** 2 - log(sd) - 0.5 * log(2 * pi) - log(theta))`."""
        log_theta = jnp.log(theta)
        z = (log_theta - self.mean_log) / self.sd
        return jnp.sum(-0.5 * z**2 - jnp.log(self.sd) - 0.5 * _LOG_2PI - log_theta)


class DirichletPrior(eqx.Module):
    """Symmetric `Dirichlet(alpha)` on each row of a jump matrix (or a single row)."""

    alpha: float = eqx.field(static=True)

    def log_prob(self, p: Array) -> Array:
        """`sum((alpha - 1) * log(p))` per row, plus the Dirichlet normalizing constant."""
        n = p.shape[-1]
        const = jsp_special.gammaln(n * self.alpha) - n * jsp_special.gammaln(self.alpha)
        per_row = jnp.sum((self.alpha - 1.0) * jnp.log(p), axis=-1) + const
        return jnp.sum(per_row)


class CholeskyDiagPrior(eqx.Module):
    """A scalar prior (`LogNormal`, ...) on the diagonal of packed Cholesky factors `(..., M(M+1)/2)`."""

    base: LogNormalPrior | NormalPrior

    def log_prob(self, theta: Array) -> Array:
        """`base.log_prob` of the diagonal entries."""
        diag = jnp.asarray(lower_diag_mask(theta.shape[-1]))
        return self.base.log_prob(theta[..., diag])


class LkjPrior(eqx.Module):
    """`LKJ(eta)` on the correlation matrix of packed Cholesky covariance factors `(..., M(M+1)/2)`."""

    eta: float = eqx.field(static=True, default=1.0)

    def log_prob(self, theta: Array) -> Array:
        """Sum over the leading axes of `lkj_logpdf` of the correlation factor of each covariance factor."""
        from hsmmkit.model.dists import lkj_logpdf, unpack_lower

        m = round((math.sqrt(8 * theta.shape[-1] + 1) - 1) / 2)
        chol = unpack_lower(theta, m)  # (..., M, M)
        rows = chol / jnp.sqrt(jnp.sum(chol**2, axis=-1, keepdims=True))  # unit-norm rows: Cholesky of R
        flat = rows.reshape(-1, m, m)
        return jnp.sum(jax.vmap(lambda c: lkj_logpdf(c, self.eta))(flat))


DEFAULT_COEF_PRIOR = NormalPrior(mean=0.0, sd=1.0)
"""Default coefficient prior (`coef_prior_sd = 1.0`), per overview Appendix A defaults."""
