"""E3-T04: explicit duration families -- pmf normalization, means, hazard, entry-time link, D_max rule."""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from regimekit.finance.clocks import TRADING_DAYS_PER_YEAR
from regimekit.finance.config import FinanceDurationCfg as DurationCfg  # finance params allow the `days` key
from regimekit.model.duration import (
    GammaDuration,
    GeometricDuration,
    LogNormalDuration,
    NonparametricDuration,
    PoissonDuration,
    WeibullDuration,
    hazard,
)
from regimekit.model.duration.explicit import check_d_max
from regimekit.model.params import ParamManager
from regimekit.utils.errors import ConfigError

N = 3
CONTINUOUS = ["gamma", "weibull", "lognormal", "poisson"]
FAMILIES = {
    "gamma": GammaDuration,
    "weibull": WeibullDuration,
    "lognormal": LogNormalDuration,
    "poisson": PoissonDuration,
}


def _theta(kind: str, rng: np.random.Generator, dt: float) -> dict[str, jax.Array]:
    """Random parameters with mean sojourn of roughly 3-25 steps (in clock units)."""
    mean = rng.uniform(3.0, 25.0, size=N) * dt
    if kind == "gamma":
        shape = rng.uniform(1.0, 4.0, size=N)
        return {"duration.shape": jnp.asarray(shape), "duration.scale": jnp.asarray(mean / shape)}
    if kind == "weibull":
        shape = rng.uniform(0.8, 3.0, size=N)
        scale = mean / np.exp(np.vectorize(math.lgamma)(1.0 + 1.0 / shape))
        return {"duration.shape": jnp.asarray(shape), "duration.scale": jnp.asarray(scale)}
    if kind == "lognormal":
        sd = rng.uniform(0.3, 0.9, size=N)
        return {"duration.mu_log": jnp.asarray(np.log(mean) - sd**2 / 2), "duration.sd_log": jnp.asarray(sd)}
    return {"duration.kappa": jnp.asarray(np.maximum(mean - dt, 0.5 * dt))}


def _scipy_pmf(kind: str, theta: dict[str, jax.Array], i: int, d_max: int, dt: float) -> np.ndarray:
    """Independent scipy reference: cdf differences (or shifted Poisson pmf), truncated + renormalized."""
    edges = np.arange(0, d_max + 1) * dt
    if kind == "poisson":
        lam = float(theta["duration.kappa"][i]) / dt
        p = stats.poisson(lam).pmf(np.arange(0, d_max))
        return p / p.sum()
    if kind == "gamma":
        dist = stats.gamma(a=float(theta["duration.shape"][i]), scale=float(theta["duration.scale"][i]))
    elif kind == "weibull":
        dist = stats.weibull_min(c=float(theta["duration.shape"][i]), scale=float(theta["duration.scale"][i]))
    else:
        dist = stats.lognorm(
            s=float(theta["duration.sd_log"][i]), scale=float(np.exp(theta["duration.mu_log"][i]))
        )
    cdf = dist.cdf(edges)
    return np.diff(cdf) / cdf[-1]


def _analytic_mean(kind: str, theta: dict[str, jax.Array], i: int, dt: float) -> float:
    if kind == "poisson":
        return dt + float(theta["duration.kappa"][i])
    if kind == "gamma":
        return float(
            stats.gamma(a=float(theta["duration.shape"][i]), scale=float(theta["duration.scale"][i])).mean()
        )
    if kind == "weibull":
        return float(
            stats.weibull_min(
                c=float(theta["duration.shape"][i]), scale=float(theta["duration.scale"][i])
            ).mean()
        )
    return float(
        stats.lognorm(
            s=float(theta["duration.sd_log"][i]), scale=float(np.exp(theta["duration.mu_log"][i]))
        ).mean()
    )


# --- AC1: rows sum to 1 (and equal an independent scipy pmf) ------------------------------------


@pytest.mark.parametrize("kind", CONTINUOUS)
@pytest.mark.parametrize("d_max", [50, 250])
@pytest.mark.parametrize("dt", [1.0 / TRADING_DAYS_PER_YEAR, 1.0])
def test_e3_t04_log_pmf_rows_sum_to_one_and_match_scipy(kind: str, d_max: int, dt: float) -> None:
    """`sum_d p(d) = 1` to 1e-10 for random parameters; the pmf equals the scipy cdf-difference pmf."""
    rng = np.random.default_rng(d_max)
    for _ in range(5):
        theta = _theta(kind, rng, dt)
        log_pmf = np.asarray(FAMILIES[kind]().log_pmf(theta, d_max, dt, None))
        assert log_pmf.shape == (N, d_max)
        np.testing.assert_allclose(np.exp(log_pmf).sum(axis=1), 1.0, atol=1e-10)
        for i in range(N):
            np.testing.assert_allclose(np.exp(log_pmf[i]), _scipy_pmf(kind, theta, i, d_max, dt), atol=1e-10)


@pytest.mark.parametrize("d_max", [50, 250])
def test_e3_t04_nonparametric_rows_sum_to_one(d_max: int) -> None:
    """A nonparametric pmf (from a random simplex) normalizes to 1 and log_pmf is its log."""
    p = np.random.default_rng(0).dirichlet(np.ones(d_max), size=N)
    fam = NonparametricDuration(d_max=d_max)
    out = np.asarray(fam.log_pmf({"duration.logits": jnp.asarray(p)}, d_max, 1.0, None))
    np.testing.assert_allclose(np.exp(out).sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.exp(out), p, atol=1e-10)


# --- AC2: truncated mean vs analytic mean -------------------------------------------------------


@pytest.mark.parametrize("kind", CONTINUOUS)
@pytest.mark.parametrize("dt", [1.0 / TRADING_DAYS_PER_YEAR, 1.0])
def test_e3_t04_truncated_mean_matches_scipy_and_analytic_within_truncation(kind: str, dt: float) -> None:
    """Truncated pmf mean == scipy's truncated mean (1e-9); vs analytic within tail + ceil-discretization."""
    d_max = 400
    theta = _theta(kind, np.random.default_rng(5), dt)
    fam = FAMILIES[kind]()
    mean_steps = np.asarray(fam.mean_sojourn_steps_truncated(theta, d_max, dt))
    tail = np.asarray(fam.tail_mass(theta, d_max, dt))
    d = np.arange(1, d_max + 1)
    for i in range(N):
        ref = float((d * _scipy_pmf(kind, theta, i, d_max, dt)).sum())
        assert mean_steps[i] == pytest.approx(ref, abs=1e-9)
        analytic_steps = _analytic_mean(kind, theta, i, dt) / dt
        assert float(np.asarray(fam.mean_sojourn(theta, dt))[i]) / dt == pytest.approx(
            analytic_steps, rel=1e-9
        )
        # E[d] for d = ceil(x/dt) is sum_{k>=0} P(x > k dt) (Poisson: 1 + lam exactly); the truncated mean
        # differs from it by at most tail * E[d | d > D_max] <= tail * (d_max + mean)
        ks = np.arange(0, 20_000) * dt
        exact = _analytic_mean(kind, theta, i, dt) / dt if kind == "poisson" else _sf_sum(kind, theta, i, ks)
        assert abs(mean_steps[i] - exact) <= tail[i] * (d_max + exact) + 1e-8


def _sf_sum(kind: str, theta: dict[str, jax.Array], i: int, ks: np.ndarray) -> float:
    if kind == "gamma":
        dist = stats.gamma(a=float(theta["duration.shape"][i]), scale=float(theta["duration.scale"][i]))
    elif kind == "weibull":
        dist = stats.weibull_min(c=float(theta["duration.shape"][i]), scale=float(theta["duration.scale"][i]))
    else:
        dist = stats.lognorm(
            s=float(theta["duration.sd_log"][i]), scale=float(np.exp(theta["duration.mu_log"][i]))
        )
    return float(dist.sf(ks).sum())


def test_e3_t04_nonparametric_mean_is_exact() -> None:
    """The nonparametric mean is `sum d p(d) * dt` exactly."""
    p = np.random.default_rng(1).dirichlet(np.ones(30), size=N)
    fam = NonparametricDuration(d_max=30)
    theta = {"duration.logits": jnp.asarray(p)}
    np.testing.assert_allclose(
        np.asarray(fam.mean_sojourn(theta, 0.5)), (p * np.arange(1, 31)).sum(axis=1) * 0.5, atol=1e-12
    )


# --- AC3: hazard ------------------------------------------------------------------------------


def test_e3_t04_hazard_of_geometric_pmf_is_constant() -> None:
    """An (untruncated) geometric pmf has hazard `a` to 1e-10 wherever survival is not lost to round-off."""
    a = np.array([0.05, 0.1, 0.3])
    d = np.arange(1, 31)  # survival stays >= 0.7**29 ~ 3e-5, so `1 - cumsum` keeps ~11 digits
    log_pmf = (d[None, :] - 1.0) * np.log1p(-a[:, None]) + np.log(a[:, None])
    h = np.asarray(hazard(jnp.asarray(log_pmf)))
    np.testing.assert_allclose(h, np.broadcast_to(a[:, None], h.shape), atol=1e-10)


def test_e3_t04_hazard_of_geometric_family_is_constant_away_from_truncation() -> None:
    """`GeometricDuration.log_pmf` (truncated at 250): hazard equals `a` to 1e-10 over the first 30 steps."""
    a = np.array([0.1, 0.3])
    h = np.asarray(
        hazard(GeometricDuration().log_pmf({"duration.rate": jnp.asarray(-np.log1p(-a))}, 250, 1.0, None))
    )
    np.testing.assert_allclose(h[:, :30], np.broadcast_to(a[:, None], (2, 30)), atol=1e-10)


def test_e3_t04_hazard_of_truncated_family_pmf_is_finite_and_ends_at_one() -> None:
    """For a pmf renormalized over `1..D_max` the hazard is in [0, 1] and the last hazard is 1."""
    fam = GeometricDuration()
    theta = {"duration.rate": jnp.asarray([0.1, 0.4])}
    h = np.asarray(hazard(fam.log_pmf(theta, 30, 1.0, None)))
    assert np.all(np.isfinite(h))
    assert np.all((h >= 0) & (h <= 1))
    np.testing.assert_allclose(h[:, -1], 1.0, atol=1e-10)


# --- AC4: entry-time covariate link -----------------------------------------------------------


@pytest.mark.parametrize("kind", CONTINUOUS)
def test_e3_t04_entry_link_zero_coefficients_reproduce_base_and_shapes(kind: str) -> None:
    """Zero `c` gives the base pmf; `z_entry (K,)` -> `(N, D)`, `(T, K)` -> `(T, N, D)`."""
    dt, d_max, t, k = 1.0 / TRADING_DAYS_PER_YEAR, 60, 4, 2
    theta = _theta(kind, np.random.default_rng(2), dt)
    fam = FAMILIES[kind](cov_idx=(1, 0))
    base = np.asarray(fam.log_pmf(theta, d_max, dt, None))
    z_seq = jnp.asarray(np.random.default_rng(3).normal(size=(t, k)))
    zero = {**theta, "duration.c": jnp.zeros((N, 2))}
    np.testing.assert_allclose(np.asarray(fam.log_pmf(zero, d_max, dt, z_seq[0])), base, atol=1e-12)
    out = np.asarray(fam.log_pmf(zero, d_max, dt, z_seq))
    assert out.shape == (t, N, d_max)
    np.testing.assert_allclose(out, np.broadcast_to(base, out.shape), atol=1e-12)


@pytest.mark.parametrize("kind", CONTINUOUS)
def test_e3_t04_entry_link_scales_the_primary_parameter(kind: str) -> None:
    """Non-zero `c`: the pmf equals the base pmf at `scale * exp(c.z)` (`mu_log + c.z` for lognormal)."""
    dt, d_max = 1.0 / TRADING_DAYS_PER_YEAR, 80
    theta = _theta(kind, np.random.default_rng(4), dt)
    fam = FAMILIES[kind](cov_idx=(0,))
    c = np.array([[0.4], [-0.3], [0.2]])
    z = jnp.asarray([0.7])
    got = np.asarray(fam.log_pmf({**theta, "duration.c": jnp.asarray(c)}, d_max, dt, z))
    shift = (c @ np.asarray(z)).ravel()
    key = {"gamma": "duration.scale", "weibull": "duration.scale", "poisson": "duration.kappa"}.get(kind)
    moved = dict(theta)
    if key is not None:
        moved[key] = theta[key] * jnp.exp(jnp.asarray(shift))
    else:
        moved["duration.mu_log"] = theta["duration.mu_log"] + jnp.asarray(shift)
    ref = np.asarray(FAMILIES[kind]().log_pmf(moved, d_max, dt, None))
    np.testing.assert_allclose(got, ref, atol=1e-12)
    assert not np.allclose(got, np.asarray(fam.log_pmf(theta, d_max, dt, None)))


# --- AC5: D_max warning rule ------------------------------------------------------------------


def test_e3_t04_d_max_warning_triggers_for_small_d_max() -> None:
    """`d_max = 5` with a mean sojourn of 20 steps warns on both the mass rule and the 2x-mean rule."""
    fam = GammaDuration()
    theta = fam.init_from_mean_sojourn(jnp.full((N,), 20.0), 1.0)
    with pytest.warns(UserWarning, match="d_max=5") as record:
        messages = check_d_max(fam, theta, 5, 1.0)
    assert len(record) == 2 == len(messages)
    assert any("truncated duration mass" in m for m in messages)
    assert any("mean sojourn" in m for m in messages)


def test_e3_t04_d_max_warning_silent_for_generous_d_max() -> None:
    """A `d_max` far beyond the mean sojourn and the tail raises no warning."""
    fam = LogNormalDuration()
    theta = fam.init_from_mean_sojourn(jnp.full((N,), 10.0), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_d_max(fam, theta, 400, 1.0) == []


# --- init from mean sojourn, phase API, config, gradients -------------------------------------


@pytest.mark.parametrize("kind", CONTINUOUS)
@pytest.mark.parametrize("dt", [1.0 / TRADING_DAYS_PER_YEAR, 1.0])
def test_e3_t04_init_from_mean_sojourn_round_trips_mean(kind: str, dt: float) -> None:
    """`mean_sojourn(init_from_mean_sojourn(m)) == m * dt` in clock units (steps clock: dt = 1)."""
    mean_steps = jnp.asarray([2.0, 8.0, 40.0])
    fam = FAMILIES[kind]()
    theta = fam.init_from_mean_sojourn(mean_steps, dt)
    np.testing.assert_allclose(
        np.asarray(fam.mean_sojourn(theta, dt)), np.asarray(mean_steps) * dt, rtol=1e-12
    )


def test_e3_t04_documented_init_values() -> None:
    """Gamma shape 2, weibull shape 1.5, lognormal sd_log 0.5 with `mu = log(mean) - 0.125`, poisson kappa."""
    m = jnp.asarray([10.0])
    assert float(GammaDuration().init_from_mean_sojourn(m, 1.0)["duration.shape"][0]) == 2.0
    assert float(WeibullDuration().init_from_mean_sojourn(m, 1.0)["duration.shape"][0]) == 1.5
    ln = LogNormalDuration().init_from_mean_sojourn(m, 1.0)
    assert float(ln["duration.sd_log"][0]) == 0.5
    assert float(ln["duration.mu_log"][0]) == pytest.approx(np.log(10.0) - 0.125, abs=1e-14)
    assert float(PoissonDuration().init_from_mean_sojourn(m, 0.5)["duration.kappa"][0]) == pytest.approx(4.5)
    with pytest.raises(ConfigError, match="> 1 step"):
        PoissonDuration().init_from_mean_sojourn(jnp.asarray([1.0]), 1.0)
    np_theta = NonparametricDuration(d_max=200).init_from_mean_sojourn(m, 1.0)
    p = np.asarray(np_theta["duration.logits"])
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert (p * np.arange(1, 201)).sum() == pytest.approx(10.0, rel=1e-3)


def test_e3_t04_phase_api_raises_not_implemented() -> None:
    """Explicit families have no phase representation."""
    fam = GammaDuration()
    assert fam.representation == "explicit"
    assert fam.supports_explicit
    with pytest.raises(NotImplementedError):
        fam.n_phases(0)
    with pytest.raises(NotImplementedError):
        fam.advance_prob({}, jnp.ones(2), None)
    with pytest.raises(NotImplementedError):
        fam.advance_prob_t({}, 1.0)


def _cfg(family: str, params: dict[str, dict[str, object]], **extra: object) -> DurationCfg:
    return DurationCfg.model_validate({"family": family, "params": params, **extra})


def test_e3_t04_from_config_specs_and_mean_init() -> None:
    """`from_config`: spec names/shapes, `init_mean_sojourn_steps` seeds shape and scale."""
    cfg = _cfg(
        "gamma",
        {
            "shape": {"mode": "free"},
            "scale": {"mode": "free", "init_mean_sojourn_steps": [5.0, 10.0, 20.0]},
        },
        covariates=["vix"],
    )
    fam, specs = GammaDuration.from_config(cfg, 3, ("vix",), 1.0)
    by = {s.name: s for s in specs}
    assert set(by) == {"duration.shape", "duration.scale", "duration.c"}
    np.testing.assert_allclose(by["duration.shape"].init, 2.0)
    np.testing.assert_allclose(by["duration.scale"].init, [2.5, 5.0, 10.0])
    assert fam.cov_idx == (0,)
    with pytest.raises(ConfigError) as info:
        GammaDuration.from_config(_cfg("gamma", {"scale": {"mode": "free"}}), 3, (), 1.0)
    assert info.value.path == "model.duration.params.shape"
    with pytest.raises(ConfigError, match="clock-aware"):
        GammaDuration.from_config(
            _cfg(
                "gamma",
                {"shape": {"mode": "free"}, "scale": {"mode": "free", "init_mean_sojourn_days": [5, 5, 5]}},
            ),
            3,
            (),
            1.0,
        )


def test_e3_t04_nonparametric_from_config_and_param_count() -> None:
    """Nonparametric spec is `(N, D_max)` on `pmf` (`N (D_max-1)` free entries); covariates are rejected."""
    cfg = _cfg("nonparametric", {"logits": {"mode": "free", "init_mean_sojourn_steps": [4.0, 8.0]}})
    fam, specs = NonparametricDuration.from_config(cfg, 2, (), 1.0, d_max=20)
    assert specs[0].shape == (2, 20)
    assert specs[0].domain == "pmf"
    assert ParamManager(specs).n_free == 2 * 19
    manager = ParamManager(specs)
    theta = manager.unflatten(manager.init_u())
    np.testing.assert_allclose(np.asarray(theta["duration.logits"]), specs[0].init, atol=1e-10)
    assert fam.d_max == 20
    with pytest.raises(ConfigError, match="d_max"):
        NonparametricDuration.from_config(cfg, 2, (), 1.0)
    with pytest.raises(ConfigError, match="covariate"):
        NonparametricDuration.from_config(
            _cfg("nonparametric", {"logits": {"mode": "free"}}, covariates=["x"]), 2, ("x",), 1.0, d_max=5
        )
    with pytest.raises(ConfigError, match="bins"):
        fam.log_pmf({"duration.logits": jnp.full((2, 7), 1 / 7)}, 20, 1.0, None)


@pytest.mark.parametrize("kind", CONTINUOUS)
def test_e3_t04_log_pmf_is_jittable_and_differentiable_with_finite_gradients(kind: str) -> None:
    """Gradients w.r.t. every parameter (and the link coefficients) are finite."""
    dt, d_max = 1.0 / TRADING_DAYS_PER_YEAR, 60
    theta = _theta(kind, np.random.default_rng(6), dt)
    fam = FAMILIES[kind](cov_idx=(0,))
    theta = {**theta, "duration.c": jnp.full((N, 1), 0.1)}
    z = jnp.asarray([0.5])
    weights = jnp.asarray(np.random.default_rng(0).normal(size=(N, d_max)))

    def loss(th: dict[str, jax.Array]) -> jax.Array:
        return jnp.sum(weights * fam.log_pmf(th, d_max, dt, z))

    grads = jax.jit(jax.grad(loss))(theta)
    for name, g in grads.items():
        assert np.all(np.isfinite(np.asarray(g))), name


def test_e3_t04_extreme_parameters_keep_log_pmf_and_gradient_finite() -> None:
    """A cell whose probability underflows is floored, not `-inf`/NaN (CLAUDE.md §5)."""
    fam = WeibullDuration()
    theta = {"duration.shape": jnp.asarray([40.0]), "duration.scale": jnp.asarray([30.0])}
    log_pmf = fam.log_pmf(theta, 50, 1.0, None)
    assert np.all(np.isfinite(np.asarray(log_pmf)))
    grads = jax.grad(lambda th: jnp.sum(fam.log_pmf(th, 50, 1.0, None)[:, 25:]))(theta)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grads.values())
