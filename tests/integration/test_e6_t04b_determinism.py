"""E6-T04b AC3: the same (fit, data, seed) gives bitwise identical samples; another seed differs."""

from __future__ import annotations

import numpy as np

from regimekit.inference.covariate_forecasters import make_forecaster
from tests.helpers.forecast_fixtures import fit_of, forecast_cfg, forecaster_for, with_named_covariate
from tests.helpers.objective_fixtures import make_case


def test_e6_t04b_same_seed_gives_bitwise_identical_samples() -> None:
    """Covariates + bootstrap forecaster: two calls agree bit for bit on every output; a new seed differs."""
    case = make_case("gbm", "negbin", 1, t_len=40, seed=21, cov_tr=True)
    data = with_named_covariate(case.data)
    fit = fit_of(case)
    cfg = forecast_cfg("bootstrap")
    mc = forecaster_for(case, make_forecaster(cfg.covariate_forecaster), cfg)

    a = mc.forecast(fit, data, 5, 500, seed=7)
    b = mc.forecast(fit, data, 5, 500, seed=7)
    c = mc.forecast(fit, data, 5, 500, seed=8)

    for name in ("samples", "regime_probs", "cum_quantiles", "mean", "var"):
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name
    assert not np.array_equal(np.asarray(a.samples), np.asarray(c.samples))


def test_e6_t04b_explicit_engine_is_deterministic_too() -> None:
    """The explicit-engine sampler obeys the same determinism contract."""
    case = make_case("gbm", "gamma", 1, t_len=40, seed=22)
    fit = fit_of(case)
    mc = forecaster_for(case)
    a = mc.forecast(fit, case.data, 4, 300, seed=1)
    b = mc.forecast(fit, case.data, 4, 300, seed=1)
    assert np.array_equal(np.asarray(a.samples), np.asarray(b.samples))
