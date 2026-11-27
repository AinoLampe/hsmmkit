"""E15-T03 AC3: a forecast from the online state equals the offline forecast at the same origin."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hsmmkit.config.schema import ForecastSection
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.inference.forecast import MonteCarloForecaster
from hsmmkit.online import ExpandedOnlineFilter
from hsmmkit.utils.errors import ConfigError
from tests.helpers.engine_fixtures import fit_of, make_scenario


@pytest.mark.parametrize("horizon", [1, 7])
def test_e15_t03_forecast_from_the_online_state_equals_the_offline_forecast_bitwise(horizon: int) -> None:
    """Same origin vector and seed -> identical samples, regime probabilities and quantiles."""
    sc = make_scenario(3, [3, 3, 3], 1, 600, seed=4)
    data = dataclasses.replace(sc.data, clock="calendar")  # a clock registered in hsmmkit
    fit = fit_of(sc)
    filt = ExpandedOnlineFilter(fit)
    y = np.asarray(data.observations)
    state, _ = filt.update_many(filt.start(), y, np.asarray(data.dt))

    cfg = ForecastSection(horizon=horizon, n_sim=200, quantiles=[0.1, 0.5, 0.9])
    offline = MonteCarloForecaster(sc.spec, sc.space, ExpandedHMMEngine(), None, cfg).forecast(
        fit, data, horizon, 200, seed=11
    )
    online = filt.forecast(state, horizon, 200, data, seed=11)
    np.testing.assert_array_equal(np.asarray(online.samples), np.asarray(offline.samples))
    np.testing.assert_array_equal(np.asarray(online.regime_probs), np.asarray(offline.regime_probs))
    np.testing.assert_array_equal(np.asarray(online.cum_quantiles), np.asarray(offline.cum_quantiles))
    assert online.origin_time == offline.origin_time
    other = filt.forecast(state, horizon, 200, data, seed=12)
    assert not np.array_equal(np.asarray(other.samples), np.asarray(online.samples))


def test_e15_t03_forecast_needs_an_observation() -> None:
    """A cold state has no origin distribution."""
    sc = make_scenario(3, [3, 3, 3], 1, 50, seed=4)
    filt = ExpandedOnlineFilter(fit_of(sc))
    with pytest.raises(ConfigError, match="at least one observation"):
        filt.forecast(filt.start(), 3, 10, dataclasses.replace(sc.data, clock="calendar"))
