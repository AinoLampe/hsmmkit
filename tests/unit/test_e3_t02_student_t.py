"""E3-T02: Student-t emission -- log_prob vs scipy, GBM limit, tied nu, sampler moments."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from regimekit.config.schema import EmissionCfg
from regimekit.model.emission.gaussian import GaussianFamily
from regimekit.model.emission.student_t import StudentTFamily
from regimekit.model.params import ParamManager
from regimekit.utils.errors import ConfigError

FAMILY = StudentTFamily(scale_by_dt=True)


def _theta(m: np.ndarray, s: np.ndarray, nu: np.ndarray) -> dict[str, jax.Array]:
    return {"emission.m": jnp.asarray(m), "emission.s": jnp.asarray(s), "emission.nu": jnp.asarray(nu)}


# --- AC1: log_prob vs scipy.stats.t.logpdf, <= 1e-10 -----------------------------------------


@pytest.mark.parametrize("dt_value", [1.0 / 252.0, 1.0 / 52.0, 1.0])
def test_e3_t02_log_prob_matches_scipy_reference(dt_value: float) -> None:
    """`log_prob` matches `scipy.stats.t.logpdf` (same loc/scale/df), T=50, N=3, M=1."""
    rng = np.random.default_rng(0)
    t, n, m = 50, 3, 1
    loc = rng.uniform(-0.2, 0.2, size=(n, m))
    scale = rng.uniform(0.05, 0.4, size=(n, m))
    nu = rng.uniform(2.5, 30.0, size=(n,))
    y = rng.normal(size=(t, m)) * 0.01
    dt = jnp.full((t,), dt_value)

    out = np.asarray(FAMILY.log_prob(jnp.asarray(y), dt, _theta(loc, scale, nu), None))
    assert out.shape == (t, n)
    for i in range(n):
        ref = stats.t(df=nu[i], loc=loc[i, 0] * dt_value, scale=scale[i, 0] * np.sqrt(dt_value)).logpdf(
            y[:, 0]
        )
        np.testing.assert_allclose(out[:, i], ref, atol=1e-10)


def test_e3_t02_log_prob_sums_over_series() -> None:
    """With `M = 2` the density is the sum of the two per-series scipy log-densities."""
    rng = np.random.default_rng(3)
    t, n, m = 20, 2, 2
    loc = rng.uniform(-0.1, 0.1, size=(n, m))
    scale = rng.uniform(0.1, 0.3, size=(n, m))
    nu = np.array([4.0, 12.0])
    y = rng.normal(size=(t, m)) * 0.02
    out = np.asarray(FAMILY.log_prob(jnp.asarray(y), jnp.full((t,), 1.0), _theta(loc, scale, nu), None))
    for i in range(n):
        ref = sum(stats.t(df=nu[i], loc=loc[i, j], scale=scale[i, j]).logpdf(y[:, j]) for j in range(m))
        np.testing.assert_allclose(out[:, i], ref, atol=1e-10)


# --- AC2: nu -> infinity recovers GBM with mu - sigma^2/2 -> m, sigma -> s --------------------


def test_e3_t02_large_nu_matches_gbm_with_ito_shift() -> None:
    """`nu = 1e6` (fixed): density equals GBM at `mu = m + s^2/2`, `sigma = s` within 1e-6.

    The fixture draws `y` within `|z| <= 1.5` of each regime's centre: the t-vs-normal log-density gap
    is `O(z^4 / (4 nu))`, so the 1e-6 bound is a statement about the bulk, not the far tails.
    """
    rng = np.random.default_rng(1)
    t, n, m = 200, 3, 1
    dt_value = 1.0 / 252.0
    m_loc = rng.uniform(-0.2, 0.2, size=(n, m))
    s = rng.uniform(0.05, 0.4, size=(n, m))
    dt = jnp.full((t,), dt_value)
    z = np.clip(rng.normal(size=(t, m)), -1.5, 1.5)
    # centre on regime 0's density; regimes 1, 2 see the same y, so bound |y - loc_i| directly per regime
    y = m_loc[0, 0] * dt_value + z * s[0, 0] * np.sqrt(dt_value)
    keep = np.all(
        [np.abs(y[:, 0] - m_loc[i, 0] * dt_value) <= 1.5 * s[i, 0] * np.sqrt(dt_value) for i in range(n)],
        axis=0,
    )
    assert keep.sum() >= 20
    y = y[keep]
    dt = dt[: y.shape[0]]

    st = FAMILY.log_prob(jnp.asarray(y), dt, _theta(m_loc, s, np.full((n,), 1e6)), None)
    gbm = GaussianFamily(scale_by_dt=True, ito_correction=True).log_prob(
        jnp.asarray(y),
        dt,
        {"emission.mean": jnp.asarray(m_loc + 0.5 * s**2), "emission.scale": jnp.asarray(s)},
        None,
    )
    np.testing.assert_allclose(np.asarray(st), np.asarray(gbm), atol=1e-6)


# --- AC3: tied nu -> a single u coordinate ---------------------------------------------------


def _cfg(nu: dict[str, object], **extra: object) -> EmissionCfg:
    return EmissionCfg.model_validate(
        {
            "family": "student_t",
            "scale_by_dt": True,
            "params": {
                "m": {"mode": "free"},
                "s": {"mode": "free"},
                "nu": nu,
            },
            **extra,
        }
    )


def test_e3_t02_tied_nu_is_one_free_coordinate() -> None:
    """`nu` tied -> one `u` entry; free -> N; fixed -> 0 (`ParamManager.n_free`, N=3, M=1)."""
    n = 3
    base = 2 * n  # m and s free
    for nu_cfg, expected_nu in (
        ({"mode": "tied", "init": [6.0, 6.0, 6.0]}, 1),
        ({"mode": "free"}, n),
        ({"mode": "fixed", "values": [5.0, 5.0, 5.0]}, 0),
    ):
        _, specs = StudentTFamily.from_config(_cfg(nu_cfg), n, 1, ())
        assert ParamManager(specs).n_free == base + expected_nu, nu_cfg


def test_e3_t02_tied_nu_uses_one_shared_value() -> None:
    """A tied `nu` maps one `u` to the same `theta` value for every state."""
    _, specs = StudentTFamily.from_config(_cfg({"mode": "tied", "init": [7.0, 7.0, 7.0]}), 3, 1, ())
    manager = ParamManager(specs)
    theta = manager.unflatten(manager.init_u())
    nu = np.asarray(theta["emission.nu"])
    np.testing.assert_allclose(nu, 7.0, atol=1e-10)
    u_shift = manager.init_u() + jnp.linspace(0.1, 0.9, manager.n_free)  # a different offset per coordinate
    nu_shift = np.asarray(manager.unflatten(u_shift)["emission.nu"])
    assert nu_shift.shape == (3,)
    assert np.unique(nu_shift).size == 1
    assert abs(nu_shift[0] - 7.0) > 1e-3  # the shared value did move


def test_e3_t02_nu_domain_keeps_nu_above_2() -> None:
    """Any `u` yields `nu > 2` through the `nu_gt_2` bijector."""
    _, specs = StudentTFamily.from_config(_cfg({"mode": "free"}), 2, 1, ())
    manager = ParamManager(specs)
    for scale in (-30.0, 0.0, 30.0):
        theta = manager.unflatten(jnp.full((manager.n_free,), scale))
        assert np.all(np.asarray(theta["emission.nu"]) > 2.0)


def test_e3_t02_missing_nu_raises_with_path() -> None:
    """A config without `params.nu` is rejected with the config path."""
    cfg = EmissionCfg.model_validate(
        {"family": "student_t", "params": {"m": {"mode": "free"}, "s": {"mode": "free"}}}
    )
    with pytest.raises(ConfigError) as info:
        StudentTFamily.from_config(cfg, 2, 1, ())
    assert info.value.path == "model.emission.params.nu"


# --- AC4: sampler moments within 3 SE (nu = 8) -----------------------------------------------


def test_e3_t02_sampler_moments_within_3_se() -> None:
    """`nu = 8`: sample mean and variance match `m dt` and `s^2 dt nu/(nu-2)` within 3 SE."""
    n_draw, dt_value = 400_000, 1.0 / 52.0
    m_val, s_val, nu = 0.3, 0.25, 8.0
    theta = _theta(np.array([[m_val]]), np.array([[s_val]]), np.array([nu]))
    dt = jnp.full((n_draw,), dt_value)
    regime = jnp.zeros((n_draw,), dtype=jnp.int32)
    x = np.asarray(FAMILY.sample(jax.random.key(7), regime, dt, theta, None))[:, 0]

    mean_true = m_val * dt_value
    var_true = s_val**2 * dt_value * nu / (nu - 2.0)
    se_mean = np.sqrt(var_true / n_draw)
    kurt = 3.0 + 6.0 / (nu - 4.0)
    se_var = var_true * np.sqrt((kurt - 1.0) / n_draw)
    assert abs(x.mean() - mean_true) < 3.0 * se_mean
    assert abs(x.var() - var_true) < 3.0 * se_var


def test_e3_t02_sample_uses_active_regime_parameters() -> None:
    """Each step draws from its own regime's location and degrees of freedom."""
    n_draw = 100_000
    theta = _theta(np.array([[0.0], [5.0]]), np.array([[0.01], [0.01]]), np.array([5.0, 9.0]))
    dt = jnp.ones((n_draw,))
    regime = jnp.asarray(np.arange(n_draw) % 2, dtype=jnp.int32)
    x = np.asarray(FAMILY.sample(jax.random.key(0), regime, dt, theta, None))[:, 0]
    assert abs(x[::2].mean()) < 0.01
    assert abs(x[1::2].mean() - 5.0) < 0.01
    # heavier tails in regime 0 (nu=5): larger variance ratio than the nu=9 regime
    assert x[::2].var() > x[1::2].var()


def test_e3_t02_sample_deterministic_per_key() -> None:
    """Same key -> identical samples; different key -> different samples."""
    theta = _theta(np.array([[0.1]]), np.array([[0.2]]), np.array([6.0]))
    dt = jnp.full((5,), 1.0 / 252.0)
    regime = jnp.zeros((5,), dtype=jnp.int32)
    s1 = FAMILY.sample(jax.random.key(0), regime, dt, theta, None)
    s2 = FAMILY.sample(jax.random.key(0), regime, dt, theta, None)
    s3 = FAMILY.sample(jax.random.key(1), regime, dt, theta, None)
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
    assert not np.allclose(np.asarray(s1), np.asarray(s3))


# --- covariate links, jit/grad, protocol attributes -------------------------------------------


def test_e3_t02_covariate_links_shift_location_and_scale() -> None:
    """`gamma`/`delta` enter as `m + gamma z` and `log s + delta z`; zero coefficients change nothing."""
    cfg = _cfg({"mode": "free"}, covariates={"drift": ["a"], "log_vol": ["b"]})
    family, specs = StudentTFamily.from_config(cfg, 2, 1, ("a", "b"))
    assert {sp.name for sp in specs} == {
        "emission.m",
        "emission.s",
        "emission.nu",
        "emission.gamma",
        "emission.delta",
    }
    t = 6
    rng = np.random.default_rng(5)
    z = jnp.asarray(rng.normal(size=(t, 2)))
    y = jnp.asarray(rng.normal(size=(t, 1)) * 0.01)
    dt = jnp.full((t,), 1.0 / 252.0)
    base = _theta(np.array([[0.1], [0.2]]), np.array([[0.2], [0.3]]), np.array([5.0, 8.0]))

    zero = {**base, "emission.gamma": jnp.zeros((2, 1)), "emission.delta": jnp.zeros((2, 1))}
    np.testing.assert_allclose(
        np.asarray(family.log_prob(y, dt, zero, z)),
        np.asarray(FAMILY.log_prob(y, dt, base, None)),
        atol=1e-12,
    )

    gamma, delta = np.array([[0.5], [-0.4]]), np.array([[0.3], [0.1]])
    full = {**base, "emission.gamma": jnp.asarray(gamma), "emission.delta": jnp.asarray(delta)}
    out = np.asarray(family.log_prob(y, dt, full, z))
    zz = np.asarray(z)
    for i in range(2):
        loc = (np.asarray(base["emission.m"])[i, 0] + gamma[i, 0] * zz[:, 0]) * (1.0 / 252.0)
        scale = np.exp(np.log(np.asarray(base["emission.s"])[i, 0]) + delta[i, 0] * zz[:, 1]) * np.sqrt(
            1.0 / 252.0
        )
        ref = stats.t(df=[5.0, 8.0][i], loc=loc, scale=scale).logpdf(np.asarray(y)[:, 0])
        np.testing.assert_allclose(out[:, i], ref, atol=1e-10)


def test_e3_t02_jittable_and_differentiable_in_theta() -> None:
    """`log_prob` is jittable and has finite gradients w.r.t. `m`, `s`, `nu`."""
    theta = _theta(np.array([[0.1], [0.2]]), np.array([[0.2], [0.3]]), np.array([5.0, 8.0]))
    y = jnp.asarray(np.random.default_rng(0).normal(size=(10, 1)) * 0.01)
    dt = jnp.full((10,), 1.0 / 252.0)

    def loss(th: dict[str, jax.Array]) -> jax.Array:
        return jnp.sum(FAMILY.log_prob(y, dt, th, None))

    grads = jax.jit(jax.grad(loss))(theta)
    for g in grads.values():
        assert np.all(np.isfinite(np.asarray(g)))


def test_e3_t02_protocol_attributes_and_noop_hooks() -> None:
    """`support`/`scale_by_dt` are declared; carry and sufficient-statistics hooks are no-ops."""
    assert FAMILY.name == "student_t"
    assert FAMILY.support == "real"
    assert FAMILY.scale_by_dt is True
    assert FAMILY.init_carry() is None
    assert FAMILY.step_carry(None, jnp.zeros(1), {}) is None
    assert FAMILY.sufficient_stats(jnp.zeros(1), jnp.ones(1), jnp.ones((1, 1))) is None


def test_e3_t02_default_param_specs_shapes_and_domains() -> None:
    """Config-free specs: `m`, `s` are `(N, M)`, `nu` is `(N,)` on the `nu_gt_2` domain."""
    specs = {sp.name: sp for sp in FAMILY.param_specs(3, 2, ())}
    assert specs["emission.m"].shape == (3, 2)
    assert specs["emission.s"].domain == "positive"
    assert specs["emission.nu"].shape == (3,)
    assert specs["emission.nu"].domain == "nu_gt_2"
