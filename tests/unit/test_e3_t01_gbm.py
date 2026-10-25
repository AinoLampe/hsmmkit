"""E3-T01: GBM emission -- log_prob/sample vs analytic/scipy references, covariate links."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from regimekit.model.emission.gbm import GBMFamily

FAMILY = GBMFamily()


def _theta(mu: np.ndarray, sigma: np.ndarray) -> dict[str, jax.Array]:
    return {"emission.mu": jnp.asarray(mu), "emission.sigma": jnp.asarray(sigma)}


# --- AC1: log_prob vs scipy.stats.norm.logpdf, <= 1e-12 --------------------------------------


@pytest.mark.parametrize("dt_value", [1.0 / 252.0, 1.0 / 52.0])
def test_e3_t01_log_prob_matches_scipy_reference(dt_value: float) -> None:
    """`log_prob` matches `scipy.stats.norm.logpdf` with the same mean/variance, T=50, N=3, M=1."""
    rng = np.random.default_rng(0)
    t, n, m = 50, 3, 1
    mu = rng.uniform(-0.2, 0.2, size=(n, m))
    sigma = rng.uniform(0.05, 0.4, size=(n, m))
    y = rng.normal(size=(t, m)) * 0.01
    dt = jnp.full((t,), dt_value)

    theta = _theta(mu, sigma)
    out = np.asarray(FAMILY.log_prob(jnp.asarray(y), dt, theta, None))
    assert out.shape == (t, n)

    for i in range(n):
        mean = (mu[i, 0] - 0.5 * sigma[i, 0] ** 2) * dt_value
        sd = sigma[i, 0] * np.sqrt(dt_value)
        ref = stats.norm(mean, sd).logpdf(y[:, 0])
        np.testing.assert_allclose(out[:, i], ref, atol=1e-12)


# --- AC2: Appendix E reproduction (4 significant digits) -------------------------------------


@pytest.mark.parametrize(
    ("mu", "sigma", "expected_mean_pct", "expected_sd_pct"),
    [
        (0.08, 0.12, 0.029, 0.756),
        (-0.10, 0.35, -0.064, 2.205),
    ],
)
def test_e3_t01_appendix_e_daily_mean_sd(
    mu: float, sigma: float, expected_mean_pct: float, expected_sd_pct: float
) -> None:
    """Daily (dt=1/252) mean/sd reproduce Appendix E's calm/turbulent numbers."""
    dt_value = 1.0 / 252.0
    theta = _theta(np.array([[mu]]), np.array([[sigma]]))
    dt = jnp.full((1,), dt_value)

    # Read the family's own computed mean/sd directly, rather than recomputing the formula.
    mean_t, sd_t = FAMILY._mean_sd(dt, theta, None)
    mean = float(mean_t[0, 0, 0])
    sd = float(sd_t[0, 0, 0])

    ref = float(stats.norm(mean, sd).logpdf(0.0))
    out = float(np.asarray(FAMILY.log_prob(jnp.zeros((1, 1)), dt, theta, None))[0, 0])
    assert out == pytest.approx(ref, abs=1e-12)

    assert round(mean * 100, 3) == pytest.approx(expected_mean_pct, abs=5e-4)
    assert round(sd * 100, 3) == pytest.approx(expected_sd_pct, abs=5e-4)


# --- AC3: covariate links ----------------------------------------------------------------------


def test_e3_t01_zero_coefficients_reproduce_covariate_free_density() -> None:
    """`gamma=0, delta=0` reproduce the plain (no-covariate) density."""
    rng = np.random.default_rng(1)
    t, n, m, k = 10, 2, 1, 2
    mu = rng.uniform(-0.1, 0.1, size=(n, m))
    sigma = rng.uniform(0.1, 0.3, size=(n, m))
    y = rng.normal(size=(t, m)) * 0.01
    dt = jnp.full((t,), 1.0 / 252.0)
    z = jnp.asarray(rng.normal(size=(t, k)))

    family = GBMFamily(drift_idx=(0, 1), logvol_idx=(0, 1))
    theta_plain = _theta(mu, sigma)
    theta_cov = {
        **theta_plain,
        "emission.gamma": jnp.zeros((n, k)),
        "emission.delta": jnp.zeros((n, k)),
    }

    plain = family.log_prob(jnp.asarray(y), dt, theta_plain, None)
    with_zero_cov = family.log_prob(jnp.asarray(y), dt, theta_cov, z)
    np.testing.assert_allclose(np.asarray(plain), np.asarray(with_zero_cov), atol=1e-12)


def test_e3_t01_nonzero_delta_scales_variance_by_exp_2_delta_z() -> None:
    """Non-zero `delta` scales the per-step variance by `exp(2 * delta . z)`."""
    t, n, m, k = 5, 1, 1, 1
    mu = np.zeros((n, m))
    sigma = np.full((n, m), 0.2)
    dt_value = 1.0 / 252.0
    dt = jnp.full((t,), dt_value)
    z = jnp.asarray(np.full((t, k), 0.5))
    delta = np.full((n, k), 0.3)

    family = GBMFamily(logvol_idx=(0,))
    theta = {**_theta(mu, sigma), "emission.delta": jnp.asarray(delta)}
    mean_t, sd_t = family._mean_sd(dt, theta, z)

    scale = np.exp(2.0 * 0.3 * 0.5)
    expected_var = (sigma[0, 0] ** 2 * dt_value) * scale
    expected_mean = (0.0 - 0.5 * sigma[0, 0] ** 2 * scale) * dt_value
    np.testing.assert_allclose(np.asarray(sd_t[:, 0, 0]) ** 2, expected_var, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(mean_t[:, 0, 0]), expected_mean, rtol=1e-10)


# --- AC4: deterministic per key (moment check lives in tests/property, per E1-T04 convention) --


def test_e3_t01_sample_deterministic_per_key() -> None:
    """Same key -> identical samples; different key -> different samples."""
    theta = _theta(np.array([[0.1]]), np.array([[0.2]]))
    dt = jnp.full((5,), 1.0 / 252.0)
    regime = jnp.zeros((5,), dtype=jnp.int32)

    s1 = FAMILY.sample(jax.random.key(0), regime, dt, theta, None)
    s2 = FAMILY.sample(jax.random.key(0), regime, dt, theta, None)
    s3 = FAMILY.sample(jax.random.key(1), regime, dt, theta, None)

    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
    assert not np.allclose(np.asarray(s1), np.asarray(s3))


# --- AC5: M > 1, jittable, differentiable -------------------------------------------------------


def test_e3_t01_m_greater_than_1_jittable_and_differentiable() -> None:
    """`M > 1` shapes are accepted, `log_prob` is jittable and differentiable in `theta`."""
    rng = np.random.default_rng(2)
    t, n, m = 8, 2, 3
    mu = rng.uniform(-0.1, 0.1, size=(n, m))
    sigma = rng.uniform(0.1, 0.3, size=(n, m))
    y = jnp.asarray(rng.normal(size=(t, m)) * 0.01)
    dt = jnp.full((t,), 1.0 / 252.0)
    theta = _theta(mu, sigma)

    jitted = jax.jit(lambda th: FAMILY.log_prob(y, dt, th, None))
    out = jitted(theta)
    assert out.shape == (t, n)

    def total(th: dict[str, jax.Array]) -> jax.Array:
        return jnp.sum(FAMILY.log_prob(y, dt, th, None))

    grads = jax.grad(total)(theta)
    assert np.all(np.isfinite(np.asarray(grads["emission.mu"])))
    assert np.all(np.isfinite(np.asarray(grads["emission.sigma"])))


# --- AC6: support/scale_by_dt; steps-clock equivalence -----------------------------------------


def test_e3_t01_support_and_scale_by_dt() -> None:
    """`GBMFamily.support == "real"` and `scale_by_dt is True`."""
    assert FAMILY.support == "real"
    assert FAMILY.scale_by_dt is True


def test_e3_t01_steps_clock_matches_scipy_no_dt_scaling() -> None:
    """With `dt = ones(T)` (steps clock), density equals `N(mu - sigma**2/2, sigma)`."""
    rng = np.random.default_rng(3)
    t, n, m = 20, 2, 1
    mu = rng.uniform(-0.1, 0.1, size=(n, m))
    sigma = rng.uniform(0.1, 0.4, size=(n, m))
    y = rng.normal(size=(t, m))
    dt = jnp.ones((t,))
    theta = _theta(mu, sigma)

    out = np.asarray(FAMILY.log_prob(jnp.asarray(y), dt, theta, None))
    for i in range(n):
        ref = stats.norm(mu[i, 0] - 0.5 * sigma[i, 0] ** 2, sigma[i, 0]).logpdf(y[:, 0])
        np.testing.assert_allclose(out[:, i], ref, atol=1e-12)


# --- carry / sufficient_stats hooks --------------------------------------------------------------


def test_e3_t01_carry_and_sufficient_stats_are_noops() -> None:
    """GBM has no path-dependent state: carry hooks are identity; zero weights give zero stats."""
    assert FAMILY.init_carry() is None
    assert FAMILY.step_carry("carry", jnp.zeros((1,)), {}) == "carry"
    stats = FAMILY.sufficient_stats(jnp.zeros((1, 1)), jnp.ones((1,)), jnp.zeros((1, 1)))  # E5-T06
    assert set(stats) == {"w", "w_dt", "w_y", "w_y2_dt"}  # type: ignore[arg-type]
    assert all(float(v[0]) == 0.0 for v in stats.values())  # type: ignore[attr-defined]


# --- param_specs (config-free default) ----------------------------------------------------------


def test_e3_t01_default_param_specs_shapes_and_domains() -> None:
    """The config-free `param_specs` returns free `mu` (real) and `sigma` (positive) specs."""
    mu, sigma = FAMILY.param_specs(n_states=3, n_series=2, cov_names=("x",))
    assert mu.name == "emission.mu"
    assert mu.shape == (3, 2)
    assert mu.domain == "real"
    assert mu.mode == "free"
    assert sigma.name == "emission.sigma"
    assert sigma.shape == (3, 2)
    assert sigma.domain == "positive"
