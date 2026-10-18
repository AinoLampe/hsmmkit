"""In-house distributions used by emission and duration families (ADR-0003, no TFP).

Every function is pure, vectorized, float64, and jittable. Duration pmfs are truncated at `D_max`
and renormalized to sum to 1 on the support `d = 1..D_max`; `d_max` (and, for the negative-binomial
family, `r`) are static (shape-defining, overview §4.5), not differentiated.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special

from regimekit.utils.typing import Array

_LOG_2PI = math.log(2.0 * math.pi)


# --- Continuous log-densities ---------------------------------------------------------------


def normal_logpdf(y: Array, mean: Array, sd: Array) -> Array:
    """Normal log-density."""
    z = (y - mean) / sd
    return -0.5 * z**2 - jnp.log(sd) - 0.5 * _LOG_2PI


def student_t_logpdf(y: Array, loc: Array, scale: Array, nu: Array) -> Array:
    """Student-t log-density (`nu > 2` degrees of freedom, location-scale form)."""
    z = (y - loc) / scale
    return (
        jsp_special.gammaln((nu + 1.0) / 2.0)
        - jsp_special.gammaln(nu / 2.0)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(scale)
        - (nu + 1.0) / 2.0 * jnp.log1p(z**2 / nu)
    )


# --- Duration log-pmfs (truncated at D_max, renormalized) ------------------------------------


def _poisson_logpmf(k: Array, kappa: Array) -> Array:
    return jnp.where(k >= 0, k * jnp.log(kappa) - kappa - jsp_special.gammaln(k + 1.0), -jnp.inf)


def poisson_shifted_logpmf(d: Array, kappa: Array, d_max: int) -> Array:
    """`d = 1 + Poisson(kappa)`, truncated at `D_max` and renormalized."""
    support = jnp.arange(1, d_max + 1, dtype=jnp.float64)
    log_z = jsp_special.logsumexp(_poisson_logpmf(support - 1.0, kappa))
    return _poisson_logpmf(d - 1.0, kappa) - log_z


def negbin_mean(r: float, a: Array) -> Array:
    """Untruncated mean of the shifted negative binomial, `r / a`."""
    return r / a


def negbin_var(r: float, a: Array) -> Array:
    """Untruncated variance of the shifted negative binomial, `r * (1 - a) / a**2`."""
    return r * (1.0 - a) / a**2


def _negbin_shifted_logpmf_raw(d: Array, r: float, a: Array) -> Array:
    return jnp.where(
        d >= r,
        jsp_special.gammaln(d)
        - jsp_special.gammaln(r)
        - jsp_special.gammaln(d - r + 1.0)
        + r * jnp.log(a)
        + (d - r) * jnp.log1p(-a),
        -jnp.inf,
    )


def negbin_shifted_logpmf(d: Array, r: float, a: Array, d_max: int) -> Array:
    """`d - r ~ NegBin(r, a)` (failures before `r` successes), support `d >= r`.

    Truncated at `D_max` and renormalized.
    """
    support = jnp.arange(1, d_max + 1, dtype=jnp.float64)
    log_z = jsp_special.logsumexp(_negbin_shifted_logpmf_raw(support, r, a))
    return _negbin_shifted_logpmf_raw(d, r, a) - log_z


def _gamma_cdf(x: Array | float, shape: Array, scale: Array) -> Array:
    return jsp_special.gammainc(shape, x / scale)


def disc_gamma_logpmf(d: Array, shape: Array, scale: Array, dt: float, d_max: int) -> Array:
    """`p(d) = [F(d*dt) - F((d-1)*dt)] / F(D_max*dt)`, `F` the Gamma cdf (in years)."""
    numerator = _gamma_cdf(d * dt, shape, scale) - _gamma_cdf((d - 1.0) * dt, shape, scale)
    denominator = _gamma_cdf(d_max * dt, shape, scale)
    return jnp.log(numerator) - jnp.log(denominator)


def _weibull_cdf(x: Array | float, shape: Array, scale: Array) -> Array:
    return -jnp.expm1(-((x / scale) ** shape))


def disc_weibull_logpmf(d: Array, shape: Array, scale: Array, dt: float, d_max: int) -> Array:
    """`p(d) = [F(d*dt) - F((d-1)*dt)] / F(D_max*dt)`, `F` the Weibull cdf (closed form)."""
    numerator = _weibull_cdf(d * dt, shape, scale) - _weibull_cdf((d - 1.0) * dt, shape, scale)
    denominator = _weibull_cdf(d_max * dt, shape, scale)
    return jnp.log(numerator) - jnp.log(denominator)


def _lognormal_cdf(x: Array | float, mu_log: Array, sd_log: Array) -> Array:
    z = (jnp.log(x) - mu_log) / sd_log
    # `ndtr` (not `0.5 * (1 + erf(z / sqrt(2)))`) avoids catastrophic cancellation for very
    # negative z, where erf(z) -> -1 and `1 + erf(...)` loses precision.
    return jsp_special.ndtr(z)


def disc_lognormal_logpmf(d: Array, mu_log: Array, sd_log: Array, dt: float, d_max: int) -> Array:
    """`p(d) = [F(d*dt) - F((d-1)*dt)] / F(D_max*dt)`, `F` the log-normal cdf."""
    lower = jnp.where(d > 1.0, _lognormal_cdf((d - 1.0) * dt, mu_log, sd_log), 0.0)
    numerator = _lognormal_cdf(d * dt, mu_log, sd_log) - lower
    denominator = _lognormal_cdf(d_max * dt, mu_log, sd_log)
    return jnp.log(numerator) - jnp.log(denominator)


def nonparametric_logpmf(logits: Array, d_max: int) -> Array:
    """`log softmax(logits)` over `1..D_max` (`len(logits) == d_max`)."""
    del d_max
    return jax.nn.log_softmax(logits)


def geometric_logpmf(d: Array, a: Array, d_max: int) -> Array:
    """`(1 - a)^(d - 1) * a`, truncated at `D_max` and renormalized."""
    raw = (d - 1.0) * jnp.log1p(-a) + jnp.log(a)
    log_z = jnp.log1p(-jnp.exp(d_max * jnp.log1p(-a)))
    return raw - log_z


# --- Samplers ---------------------------------------------------------------------------------


def sample_normal(key: Array, mean: Array, sd: Array, shape: tuple[int, ...] = ()) -> Array:
    """Sample from `Normal(mean, sd)`."""
    return mean + sd * jax.random.normal(key, shape=shape)


def sample_student_t(key: Array, loc: Array, scale: Array, nu: Array, shape: tuple[int, ...] = ()) -> Array:
    """Sample from the location-scale Student-t distribution."""
    return loc + scale * jax.random.t(key, df=nu, shape=shape)


def sample_categorical(key: Array, log_probs: Array) -> Array:
    """Sample a category index from `log_probs` (unnormalized log-probabilities)."""
    return jax.random.categorical(key, log_probs)


def sample_pmf(key: Array, log_pmf: Array) -> Array:
    """Sample `d in {1, ..., len(log_pmf)}` from a duration log-pmf."""
    return sample_categorical(key, log_pmf) + 1
