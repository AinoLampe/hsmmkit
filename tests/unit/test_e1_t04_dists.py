"""E1-T04: distributions vs scipy references, normalization, and analytic means."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from regimekit.model import dists

D_MAX = 20
DT = 1.0 / 252.0


# --- AC1: log-pdf/pmf vs scipy, <= 1e-10 -----------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_e1_t04_normal_logpdf_matches_scipy(seed: int) -> None:
    """`normal_logpdf` matches `scipy.stats.norm.logpdf` within 1e-10."""
    rng = np.random.default_rng(seed)
    mean, sd = rng.uniform(-3, 3), rng.uniform(0.1, 4.0)
    y = rng.normal(size=8)
    mine = np.asarray(dists.normal_logpdf(jnp.asarray(y), mean, sd))
    ref = stats.norm(mean, sd).logpdf(y)
    np.testing.assert_allclose(mine, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_e1_t04_student_t_logpdf_matches_scipy(seed: int) -> None:
    """`student_t_logpdf` matches `scipy.stats.t.logpdf` within 1e-10."""
    rng = np.random.default_rng(seed)
    loc, scale, nu = rng.uniform(-3, 3), rng.uniform(0.1, 4.0), rng.uniform(2.1, 30.0)
    y = rng.normal(size=8)
    mine = np.asarray(dists.student_t_logpdf(jnp.asarray(y), loc, scale, nu))
    ref = stats.t(df=nu, loc=loc, scale=scale).logpdf(y)
    np.testing.assert_allclose(mine, ref, atol=1e-10)


def _support(d_max: int) -> np.ndarray:
    return np.arange(1, d_max + 1)


@pytest.mark.parametrize("seed", range(5))
def test_e1_t04_poisson_shifted_logpmf_matches_scipy(seed: int) -> None:
    """`poisson_shifted_logpmf` matches a truncated/renormalized `scipy.stats.poisson`."""
    rng = np.random.default_rng(seed)
    kappa = rng.uniform(0.5, 10.0)
    ds = _support(D_MAX)
    raw = stats.poisson(kappa).logpmf(ds - 1)
    ref = raw - np.log(np.sum(np.exp(raw)))
    mine = np.asarray(dists.poisson_shifted_logpmf(jnp.asarray(ds, dtype=jnp.float64), kappa, D_MAX))
    np.testing.assert_allclose(mine, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_e1_t04_negbin_shifted_logpmf_matches_scipy(seed: int) -> None:
    """`negbin_shifted_logpmf` matches a truncated/renormalized `scipy.stats.nbinom`."""
    rng = np.random.default_rng(seed)
    r = int(rng.integers(1, 5))
    a = rng.uniform(0.1, 0.9)
    ds = _support(D_MAX)
    raw = np.where(ds >= r, stats.nbinom(r, a).logpmf(ds - r), -np.inf)
    ref = raw - np.log(np.sum(np.exp(raw)))
    mine = np.asarray(dists.negbin_shifted_logpmf(jnp.asarray(ds, dtype=jnp.float64), r, a, D_MAX))
    np.testing.assert_allclose(mine, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_e1_t04_disc_gamma_logpmf_matches_scipy(seed: int) -> None:
    """`disc_gamma_logpmf` matches a truncated/renormalized `scipy.stats.gamma` cdf."""
    rng = np.random.default_rng(seed)
    shape = rng.uniform(1.5, 6.0)
    mean_days = rng.uniform(0.5 * D_MAX, 1.5 * D_MAX)
    scale = (mean_days / 252) / shape
    ds = _support(D_MAX)

    def cdf(x: float) -> float:
        return float(stats.gamma(a=shape, scale=scale).cdf(x))

    raw = np.log([cdf(d * DT) - cdf((d - 1) * DT) for d in ds])
    ref = raw - np.log(cdf(D_MAX * DT))
    mine = np.asarray(dists.disc_gamma_logpmf(jnp.asarray(ds, dtype=jnp.float64), shape, scale, DT, D_MAX))
    np.testing.assert_allclose(mine, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_e1_t04_disc_weibull_logpmf_matches_scipy(seed: int) -> None:
    """`disc_weibull_logpmf` matches a truncated/renormalized `scipy.stats.weibull_min` cdf."""
    rng = np.random.default_rng(seed)
    shape = rng.uniform(1.0, 4.0)
    mean_days = rng.uniform(0.5 * D_MAX, 1.5 * D_MAX)
    scale = mean_days / 252
    ds = _support(D_MAX)

    def cdf(x: float) -> float:
        return float(stats.weibull_min(c=shape, scale=scale).cdf(x))

    raw = np.log([cdf(d * DT) - cdf((d - 1) * DT) for d in ds])
    ref = raw - np.log(cdf(D_MAX * DT))
    mine = np.asarray(dists.disc_weibull_logpmf(jnp.asarray(ds, dtype=jnp.float64), shape, scale, DT, D_MAX))
    np.testing.assert_allclose(mine, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_e1_t04_disc_lognormal_logpmf_matches_scipy(seed: int) -> None:
    """`disc_lognormal_logpmf` matches a truncated/renormalized `scipy.stats.lognorm` cdf."""
    rng = np.random.default_rng(seed)
    sd_log = rng.uniform(0.2, 0.8)
    mean_days = rng.uniform(0.5 * D_MAX, 1.5 * D_MAX)
    mu_log = np.log(mean_days / 252)
    ds = _support(D_MAX)

    def cdf(x: float) -> float:
        return float(stats.lognorm(s=sd_log, scale=np.exp(mu_log)).cdf(x))

    raw = np.log([cdf(d * DT) - cdf((d - 1) * DT) for d in ds])
    ref = raw - np.log(cdf(D_MAX * DT))
    mine = np.asarray(
        dists.disc_lognormal_logpmf(jnp.asarray(ds, dtype=jnp.float64), mu_log, sd_log, DT, D_MAX)
    )
    np.testing.assert_allclose(mine, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_e1_t04_geometric_logpmf_matches_scipy(seed: int) -> None:
    """`geometric_logpmf` matches a truncated/renormalized `scipy.stats.geom`."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.05, 0.9)
    ds = _support(D_MAX)
    raw = stats.geom(a).logpmf(ds)
    ref = raw - np.log(np.sum(np.exp(raw)))
    mine = np.asarray(dists.geometric_logpmf(jnp.asarray(ds, dtype=jnp.float64), a, D_MAX))
    np.testing.assert_allclose(mine, ref, atol=1e-10)


def test_e1_t04_nonparametric_logpmf_matches_log_softmax() -> None:
    """`nonparametric_logpmf` is exactly `log_softmax(logits)`."""
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=D_MAX))
    mine = np.asarray(dists.nonparametric_logpmf(logits, D_MAX))
    ref = np.asarray(jax.nn.log_softmax(logits))
    np.testing.assert_allclose(mine, ref, atol=1e-12)


# --- AC2: every pmf sums to 1 over 1..D_max; NB is -inf for d < r ----------------------------


@pytest.mark.parametrize("seed", range(5))
def test_e1_t04_pmfs_sum_to_one(seed: int) -> None:
    """Every duration pmf sums to 1 over `1..D_max` within 1e-10."""
    rng = np.random.default_rng(seed)
    ds = jnp.asarray(_support(D_MAX), dtype=jnp.float64)

    kappa = rng.uniform(0.5, 10.0)
    assert abs(float(jnp.sum(jnp.exp(dists.poisson_shifted_logpmf(ds, kappa, D_MAX)))) - 1.0) <= 1e-10

    r, a = int(rng.integers(1, 5)), rng.uniform(0.1, 0.9)
    assert abs(float(jnp.sum(jnp.exp(dists.negbin_shifted_logpmf(ds, r, a, D_MAX)))) - 1.0) <= 1e-10

    a_geo = rng.uniform(0.05, 0.9)
    assert abs(float(jnp.sum(jnp.exp(dists.geometric_logpmf(ds, a_geo, D_MAX)))) - 1.0) <= 1e-10

    logits = jnp.asarray(rng.normal(size=D_MAX))
    assert abs(float(jnp.sum(jnp.exp(dists.nonparametric_logpmf(logits, D_MAX)))) - 1.0) <= 1e-10

    shape_g = rng.uniform(1.5, 6.0)
    scale_g = (rng.uniform(0.5 * D_MAX, 1.5 * D_MAX) / 252) / shape_g
    assert (
        abs(float(jnp.sum(jnp.exp(dists.disc_gamma_logpmf(ds, shape_g, scale_g, DT, D_MAX)))) - 1.0) <= 1e-10
    )

    shape_w = rng.uniform(1.0, 4.0)
    scale_w = rng.uniform(0.5 * D_MAX, 1.5 * D_MAX) / 252
    assert (
        abs(float(jnp.sum(jnp.exp(dists.disc_weibull_logpmf(ds, shape_w, scale_w, DT, D_MAX)))) - 1.0)
        <= 1e-10
    )

    sd_log = rng.uniform(0.2, 0.8)
    mu_log = np.log(rng.uniform(0.5 * D_MAX, 1.5 * D_MAX) / 252)
    assert (
        abs(float(jnp.sum(jnp.exp(dists.disc_lognormal_logpmf(ds, mu_log, sd_log, DT, D_MAX)))) - 1.0)
        <= 1e-10
    )


def test_e1_t04_negbin_shifted_logpmf_is_neg_inf_below_r() -> None:
    """`negbin_shifted_logpmf(d, r, a, D_max)` is `-inf` for `d < r`."""
    r = 3
    for d in (0.0, 1.0, 2.0):
        assert dists.negbin_shifted_logpmf(jnp.asarray(d), r, 0.4, D_MAX) == -jnp.inf


# --- AC3: analytic NB mean reproduced by the pmf mean when truncation mass is negligible -----


@pytest.mark.parametrize("seed", range(20))
def test_e1_t04_negbin_pmf_mean_matches_analytic_mean(seed: int) -> None:
    """The truncated pmf's mean matches `negbin_mean(r, a)` when truncation mass < 1e-10.

    `negbin_shifted_logpmf` renormalizes over `1..D_max` by construction, so its own pmf always
    sums to 1 regardless of how much of the *untruncated* distribution's mass lies beyond `D_max`
    -- that untruncated tail mass (via `scipy.stats.nbinom`'s cdf) is the quantity this AC's
    "truncation mass" refers to, and is what must be negligible for the truncated mean to
    reproduce the untruncated analytic mean. Rejection-sample `(r, a)` within each case (rather
    than skipping seeds that don't happen to land in the negligible-tail-mass regime) so every
    parametrized case actually exercises the assertion.
    """
    rng = np.random.default_rng(seed)
    r = int(rng.integers(1, 4))
    for _ in range(100):
        a = rng.uniform(0.5, 0.95)
        untruncated_tail_mass = float(stats.nbinom(r, a).sf(D_MAX - r))
        if untruncated_tail_mass < 1e-10:
            break
    else:
        pytest.fail("could not find (r, a) with negligible untruncated tail mass in 100 draws")
    ds = jnp.asarray(_support(D_MAX), dtype=jnp.float64)
    pmf = jnp.exp(dists.negbin_shifted_logpmf(ds, r, a, D_MAX))
    pmf_mean = float(jnp.sum(ds * pmf))
    assert abs(pmf_mean - float(dists.negbin_mean(r, a))) <= 1e-6


def test_e1_t04_negbin_mean_and_var_formulas() -> None:
    """`negbin_mean`/`negbin_var` match `scipy.stats.nbinom` moments (shifted by `r`)."""
    r, a = 3, 0.4
    ref_mean = r + stats.nbinom(r, a).mean()
    ref_var = stats.nbinom(r, a).var()
    assert abs(float(dists.negbin_mean(r, a)) - ref_mean) <= 1e-10
    assert abs(float(dists.negbin_var(r, a)) - ref_var) <= 1e-10


# --- AC5: jittable and differentiable w.r.t. continuous parameters --------------------------


def test_e1_t04_continuous_logpdfs_are_jittable_and_differentiable() -> None:
    """`normal_logpdf`/`student_t_logpdf` are jittable and differentiable w.r.t. their parameters."""
    y = jnp.array(0.3)

    def normal_sum(mean: jax.Array, sd: jax.Array) -> jax.Array:
        return dists.normal_logpdf(y, mean, sd)

    grads = jax.grad(jax.jit(normal_sum), argnums=(0, 1))(jnp.array(0.1), jnp.array(0.9))
    assert all(bool(jnp.isfinite(g)) for g in grads)

    def t_sum(loc: jax.Array, scale: jax.Array, nu: jax.Array) -> jax.Array:
        return dists.student_t_logpdf(y, loc, scale, nu)

    grads_t = jax.grad(jax.jit(t_sum), argnums=(0, 1, 2))(jnp.array(0.0), jnp.array(1.0), jnp.array(6.0))
    assert all(bool(jnp.isfinite(g)) for g in grads_t)


def test_e1_t04_duration_logpmfs_are_jittable_and_differentiable() -> None:
    """Duration log-pmfs are jittable and differentiable w.r.t. their continuous parameters."""
    d = jnp.array(5.0)

    def poisson_at_d(kappa: jax.Array) -> jax.Array:
        return dists.poisson_shifted_logpmf(d, kappa, D_MAX)

    assert jnp.isfinite(jax.grad(jax.jit(poisson_at_d))(jnp.array(3.0)))

    def negbin_at_d(a: jax.Array) -> jax.Array:
        return dists.negbin_shifted_logpmf(d, 2, a, D_MAX)

    assert jnp.isfinite(jax.grad(jax.jit(negbin_at_d))(jnp.array(0.4)))

    def gamma_at_d(shape: jax.Array, scale: jax.Array) -> jax.Array:
        return dists.disc_gamma_logpmf(d, shape, scale, DT, D_MAX)

    grads = jax.grad(jax.jit(gamma_at_d), argnums=(0, 1))(jnp.array(2.0), jnp.array(0.05))
    assert all(bool(jnp.isfinite(g)) for g in grads)

    def weibull_at_d(shape: jax.Array, scale: jax.Array) -> jax.Array:
        return dists.disc_weibull_logpmf(d, shape, scale, DT, D_MAX)

    grads_w = jax.grad(jax.jit(weibull_at_d), argnums=(0, 1))(jnp.array(1.5), jnp.array(0.05))
    assert all(bool(jnp.isfinite(g)) for g in grads_w)

    def lognormal_at_d(mu_log: jax.Array, sd_log: jax.Array) -> jax.Array:
        return dists.disc_lognormal_logpmf(d, mu_log, sd_log, DT, D_MAX)

    grads_ln = jax.grad(jax.jit(lognormal_at_d), argnums=(0, 1))(jnp.array(-2.0), jnp.array(0.4))
    assert all(bool(jnp.isfinite(g)) for g in grads_ln)

    def geometric_at_d(a: jax.Array) -> jax.Array:
        return dists.geometric_logpmf(d, a, D_MAX)

    assert jnp.isfinite(jax.grad(jax.jit(geometric_at_d))(jnp.array(0.3)))

    def nonparametric_sum(logits: jax.Array) -> jax.Array:
        return jnp.sum(dists.nonparametric_logpmf(logits, D_MAX))

    grads_np = jax.grad(jax.jit(nonparametric_sum))(jnp.zeros(D_MAX))
    assert bool(jnp.all(jnp.isfinite(grads_np)))
