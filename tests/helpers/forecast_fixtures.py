"""Shared builders for the E6-T04b forecast tests (not collected)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from regimekit.config.schema import CovariateForecasterCfg, ForecastSection
from regimekit.data.contracts import PreparedDataset, PrepStats
from regimekit.estimation.results import FitResult
from regimekit.finance.clocks import TRADING_DAYS_PER_YEAR
from regimekit.inference.forecast import MonteCarloForecaster
from tests.helpers.objective_fixtures import Case

DT = 1.0 / TRADING_DAYS_PER_YEAR
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def forecast_cfg(kind: str = "freeze") -> ForecastSection:
    """A `forecast` config section with the shared quantile levels."""
    return ForecastSection(
        horizon=5,
        n_sim=100,
        covariate_forecaster=CovariateForecasterCfg(
            type=kind, mean_block_length=3 if kind == "bootstrap" else None
        ),  # type: ignore[arg-type]
        quantiles=QUANTILES,
    )


def fit_of(case: Case) -> FitResult:
    """A hand-built `FitResult` around the case's starting parameters."""
    return FitResult(
        estimator="map",
        params=case.manager.unflatten(case.u0),
        uparams=case.u0,
        objective=0.0,
        loglik=0.0,
        restarts=pd.DataFrame(),
        converged=True,
        n_iter=0,
        std_errors=None,
        diagnostics={},
        spec=case.spec,
        prep_stats=case.data.prep_stats,
        config_resolved={},
    )


def with_named_covariate(data: PreparedDataset) -> PreparedDataset:
    """Declare the single transition covariate column as `z` (unified `prep_stats.cov_names`)."""
    stats = PrepStats(
        cov_mean=np.zeros(1),
        cov_std=np.ones(1),
        train_window=("1970-01-01", "1970-03-01"),
        cov_names=("z",),
        transforms={},
        lags={},
    )
    return dataclasses.replace(
        data, cov_names={"transition": ("z",), "duration": (), "emission": ()}, prep_stats=stats
    )


def forecaster_for(
    case: Case, forecaster: object = None, cfg: ForecastSection | None = None
) -> MonteCarloForecaster:
    """A `MonteCarloForecaster` for `case`."""
    return MonteCarloForecaster(
        case.spec,
        case.space,
        case.engine,
        forecaster,
        cfg or forecast_cfg(),  # type: ignore[arg-type]
    )
