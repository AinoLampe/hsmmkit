"""E6-T04b: `MonteCarloForecaster` aggregation, analytic h=1 check, covariates, explicit engine."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.engine.explicit_duration import ExplicitDurationEngine
from regimekit.inference.covariate_forecasters import ColumnMap, FreezeForecaster, make_forecaster
from regimekit.inference.forecast import forecast_paths
from regimekit.utils.clock import future_grid
from regimekit.utils.errors import ConfigError, NumericalError
from regimekit.utils.rng import FORECAST_PATHS, PURPOSE_FORECAST, key_for
from tests.helpers.forecast_fixtures import (
    DT,
    QUANTILES,
    fit_of,
    forecast_cfg,
    forecaster_for,
    with_named_covariate,
)
from tests.helpers.objective_fixtures import make_case


def test_e6_t04b_h1_mean_and_var_match_the_analytic_mixture() -> None:
    """AC1: at h=1 the cumulative mean/var equal the predictive GBM mixture within 3 SE (n_sim=20000)."""
    case = make_case("gbm", "negbin", 1, t_len=60, seed=11)
    fit = fit_of(case)
    n_sim = 20000
    res = forecaster_for(case).forecast(fit, case.data, 1, n_sim, seed=0)

    log_alpha, _ = case.engine.forward(fit.params, case.spec, case.space, case.data)
    _, log_a = case.space.transition_fn(fit.params, jnp.zeros((0,)), jnp.zeros((0,)), jnp.asarray(DT))
    w = (
        np.asarray(jax.nn.softmax(log_alpha[-1]))
        @ np.exp(np.asarray(log_a))
        @ np.asarray(case.space.index.collapse)
    )
    mu = np.asarray(fit.params["emission.mean"]).reshape(-1)
    sigma = np.asarray(fit.params["emission.scale"]).reshape(-1)
    m, v = (mu - 0.5 * sigma**2) * DT, sigma**2 * DT
    mean = float(w @ m)
    var = float(w @ v + w @ m**2 - (w @ m) ** 2)

    y = np.asarray(res.samples[:, 0, 0])
    assert abs(float(res.mean[0, 0]) - mean) <= 3 * np.sqrt(var / n_sim)
    se_var = np.sqrt(np.var((y - y.mean()) ** 2) / n_sim)
    assert abs(float(res.var[0, 0]) - var) <= 3 * se_var
    np.testing.assert_allclose(np.asarray(res.regime_probs[0]).sum(), 1.0, atol=1e-12)


def test_e6_t04b_result_shapes_normalisation_and_monotone_quantiles() -> None:
    """AC2: `regime_probs` rows sum to 1 (1e-12), `cum_quantiles` is monotone in Q; metadata is filled in."""
    case = make_case("gbm", "negbin", 1, t_len=60, seed=12)
    h, n_sim = 6, 400
    res = forecaster_for(case).forecast(fit_of(case), case.data, h, n_sim, seed=3)

    assert res.samples.shape == (n_sim, h, 1)
    assert res.regime_probs.shape == (h, case.spec.n_states)
    assert res.cum_quantiles.shape == (h, len(QUANTILES), 1)
    assert res.mean.shape == (h, 1)
    assert res.var.shape == (h, 1)
    np.testing.assert_allclose(np.asarray(res.regime_probs).sum(axis=1), 1.0, atol=1e-12)
    assert np.all(np.diff(np.asarray(res.cum_quantiles), axis=1) >= 0)
    assert res.quantile_levels == tuple(QUANTILES)
    assert res.seed == 3
    assert res.origin_time == case.data.times[-1]
    assert res.horizon_times.shape == (h,)
    assert np.all(np.diff(res.horizon_times) > np.timedelta64(0, "ns"))
    np.testing.assert_allclose(
        np.asarray(res.mean[:, 0]), np.asarray(jnp.cumsum(res.samples, axis=1).mean(axis=0))[:, 0]
    )


def test_e6_t04b_horizon_times_skip_the_weekend() -> None:
    """The horizon continues on weekdays: the data end on a Friday, the first forecast step is Monday."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=13)  # 1970-01-01 (Thu) + 29 d == Fri
    res = forecaster_for(case).forecast(fit_of(case), case.data, 3, 10, seed=0)
    assert str(res.origin_time.astype("datetime64[D]")) == "1970-01-30"
    assert [str(t.astype("datetime64[D]")) for t in res.horizon_times] == [
        "1970-02-02",
        "1970-02-03",
        "1970-02-04",
    ]


def test_e6_t04b_covariates_with_horizon_above_one_require_a_forecaster() -> None:
    """AC4: `h > 1` with covariates and no forecaster is a `ConfigError`; `h == 1` needs none."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=14, cov_tr=True)
    data = with_named_covariate(case.data)
    mc = forecaster_for(case)
    with pytest.raises(ConfigError):
        mc.forecast(fit_of(case), data, 3, 50, seed=0)
    res = mc.forecast(fit_of(case), data, 1, 50, seed=0)
    assert res.regime_probs.shape == (1, 3)
    assert res.covariate_forecaster == "none"


def test_e6_t04b_freeze_equals_hard_coded_constant_covariates() -> None:
    """AC4: `freeze` reproduces constant covariates -- regime probabilities equal a hand-fed constant path."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=14, cov_tr=True)
    data = with_named_covariate(case.data)
    fit = fit_of(case)
    h, n_sim, seed = 4, 300, 5
    res = forecaster_for(case, FreezeForecaster()).forecast(fit, data, h, n_sim, seed)

    last = jnp.asarray(np.asarray(data.cov_transition)[-1])  # (1,)
    cov_paths = jnp.broadcast_to(last, (n_sim, h, 1))
    log_alpha, _ = case.engine.forward(fit.params, case.spec, case.space, data)
    _, dt = future_grid(np.asarray(data.times)[-1], h, "trading")
    colmap = ColumnMap(names=("z",), transition=(0,), duration=(), emission=())
    ref = forecast_paths(
        key_for(seed, PURPOSE_FORECAST, FORECAST_PATHS),
        fit.params,
        case.spec,
        case.space,
        log_alpha[-1],
        jnp.asarray(dt),
        n_sim,
        cov_paths,
        colmap,
    )
    ref_probs = np.asarray(jax.nn.one_hot(ref.regimes, 3).mean(axis=0))
    assert np.array_equal(np.asarray(res.regime_probs), ref_probs)
    assert res.covariate_forecaster == "freeze"


def test_e6_t04b_bootstrap_forecaster_runs_and_is_named() -> None:
    """A stochastic covariate forecaster produces a finite result and is recorded by name."""
    case = make_case("gbm", "geometric", 1, t_len=40, seed=15, cov_tr=True)
    data = with_named_covariate(case.data)
    fc = make_forecaster(forecast_cfg("bootstrap").covariate_forecaster)
    res = forecaster_for(case, fc, forecast_cfg("bootstrap")).forecast(fit_of(case), data, 5, 200, seed=1)
    assert res.covariate_forecaster == "bootstrap"
    assert np.all(np.isfinite(np.asarray(res.samples)))


def test_e6_t04b_explicit_engine_gives_the_same_shapes() -> None:
    """AC5: an explicit-engine fit yields a `ForecastResult` of the same shapes."""
    case = make_case("gbm", "gamma", 1, t_len=40, seed=16)
    assert case.spec.engine == "explicit"
    assert isinstance(case.engine, ExplicitDurationEngine)
    h, n_sim = 4, 200
    res = forecaster_for(case).forecast(fit_of(case), case.data, h, n_sim, seed=2)
    assert res.samples.shape == (n_sim, h, 1)
    assert res.regime_probs.shape == (h, case.spec.n_states)
    assert res.cum_quantiles.shape == (h, len(QUANTILES), 1)
    np.testing.assert_allclose(np.asarray(res.regime_probs).sum(axis=1), 1.0, atol=1e-12)


def test_e6_t04b_memory_guard_and_argument_checks() -> None:
    """`n_sim * h * M > 2e8` is a `NumericalError`; non-positive horizon / n_sim a `ConfigError`."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=17)
    mc = forecaster_for(case)
    fit = fit_of(case)
    with pytest.raises(NumericalError):
        mc.forecast(fit, case.data, 3, 10**8, seed=0)
    with pytest.raises(ConfigError):
        mc.forecast(fit, case.data, 0, 10, seed=0)
    with pytest.raises(ConfigError):
        mc.forecast(fit, case.data, 3, 0, seed=0)


def test_e6_t04b_supplied_posterior_is_reused_and_chunked_one_rejected() -> None:
    """`posterior=` skips the filter pass (same result as the engine's own); a collapsed posterior raises."""
    case = make_case("gbm", "gamma", 1, t_len=40, seed=18)
    fit = fit_of(case)
    mc = forecaster_for(case)
    post = case.engine.forward_backward(fit.params, case.spec, case.space, case.data)
    own = mc.forecast(fit, case.data, 3, 100, seed=4)
    reused = mc.forecast(fit, case.data, 3, 100, seed=4, posterior=post)
    assert np.array_equal(np.asarray(own.samples), np.asarray(reused.samples))

    chunked = ExplicitDurationEngine(chunk_size=8).forward_backward(
        fit.params, case.spec, case.space, case.data
    )
    with pytest.raises(ConfigError):
        mc.forecast(fit, case.data, 3, 100, seed=4, posterior=chunked)


def test_e6_t04b_one_step_covariate_forecast_uses_the_last_observed_row() -> None:
    """h=1 with covariates needs no forecaster and feeds the last observed covariate row to `A_{T+1}`."""
    case = make_case("gbm", "geometric", 1, t_len=30, seed=19, cov_tr=True)
    data = with_named_covariate(case.data)
    fit = fit_of(case)
    theta = dict(fit.params)
    theta["transition.b"] = jnp.full_like(theta["transition.b"], 3.0)
    fit = dataclasses.replace(fit, params=theta)
    n_sim = 20000

    def run(last_value: float) -> np.ndarray:
        z = np.asarray(data.cov_transition).copy()
        z[-1, 0] = last_value
        res = forecaster_for(case).forecast(
            fit, dataclasses.replace(data, cov_transition=jnp.asarray(z)), 1, n_sim, 0
        )
        return np.asarray(res.regime_probs[0])

    plus, minus = run(2.0), run(-2.0)
    assert np.abs(plus - minus).max() > 0.02  # only the last row changed, and it drives step T+1
