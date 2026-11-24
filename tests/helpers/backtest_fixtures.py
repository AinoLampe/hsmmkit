"""Shared builders for the E7-T02b walk-forward tests (not collected).

A two-regime geometric-duration GBM on the `steps` clock (`dt == 1`, parameters per step): 500 prices,
`initial_train_years: 260` steps, `refit_every_steps: 60`, horizons `[1, 5]` give four folds whose train
windows (260..440 rows) and filter windows (<= 494 rows) all pad to 512 rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from regimekit.config.yaml_io import load_config
from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.finance import FinanceConfig

ROOT = Path(__file__).resolve().parents[2]
_YAML = YAML(typ="safe")

N_PRICES = 500
SIGMA_TRUE = (0.006, 0.02)
MU_TRUE = (0.0005, -0.001)
MEAN_SOJOURN = 30.0

COV_NAME = "signal"
COV_SERIES = "SIGNAL"


def simulate_returns(n_returns: int, seed: int = 3) -> np.ndarray:
    """Per-step log-returns of a 2-regime Markov-switching GBM (geometric sojourns, mean 30 steps)."""
    rng = np.random.default_rng(seed)
    regime = np.empty(n_returns, dtype=int)
    state = 0
    for t in range(n_returns):
        regime[t] = state
        if rng.random() < 1.0 / MEAN_SOJOURN:
            state = 1 - state
    mu, sigma = np.asarray(MU_TRUE)[regime], np.asarray(SIGMA_TRUE)[regime]
    return mu + sigma * rng.standard_normal(n_returns)


def observations(returns: np.ndarray) -> ObservationSet:
    """Prices `100 exp(cumsum(returns))` on business days; `prices[0]` is the grid start tau_0."""
    times = pd.bdate_range("2020-01-01", periods=returns.shape[0] + 1).to_numpy().astype("datetime64[ns]")
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    return ObservationSet(
        symbols=("SYN",),
        times=times,
        prices=prices[:, None],
        interval="1d",
        adjusted=False,
        provider="csv",
        snapshot_id="synthetic",
        meta={},
    )


def covariate(obs: ObservationSet, values: np.ndarray, *, stamp_shift: int = 0) -> CovariateSeries:
    """A covariate with `values[i]` stamped at `obs.times[i + stamp_shift]`."""
    times = obs.times[stamp_shift : stamp_shift + values.shape[0]]
    return CovariateSeries(
        name=COV_NAME,
        times=times,
        values=np.asarray(values, dtype=np.float64),
        provider="csv",
        series_id=COV_SERIES,
        snapshot_id="synthetic",
        frequency="d",
    )


def make_config(
    tmp_path: Path,
    *,
    with_covariate: bool = False,
    metrics: tuple[str, ...] = ("log_score", "crps", "pit_coverage", "rmse", "bic", "regime_occupancy"),
    n_sim: int = 200,
    horizons: tuple[int, ...] = (1, 5),
    restarts: int = 2,
    scheme: str = "expanding",
    initial_train: int = 260,
    refit_every: int = 60,
    sharp: bool = False,
) -> FinanceConfig:
    """The shared 2-regime steps-clock config; `with_covariate` adds `signal` as an emission drift covariate.

    `sharp` fixes the drift at the truth and tightens the
    convergence tolerances: a well-conditioned fit, so that two runs which differ only by round-off (padding)
    end at the same optimum. With every parameter free the T=200 likelihood is flat along some directions and
    L-BFGS paths diverge on 1e-13 differences.
    """
    cfg: dict[str, Any] = _YAML.load(
        (ROOT / "tests" / "fixtures" / "configs" / "steps_clock.yaml").read_text(encoding="utf-8")
    )
    cfg["project"].update(seed=11, output_dir=str(tmp_path / "runs"))
    cfg["data"]["primary"].update(provider="csv", symbols=["SYN"], start="2019-01-01")
    cfg["data"]["cache"]["dir"] = str(tmp_path / "cache")
    cfg["data"]["covariates"] = (
        [
            {
                "name": COV_NAME,
                "provider": "csv",
                "series": COV_SERIES,
                "transform": [],
                "lag": 1,
                "missing": "ffill",
                "max_staleness": 5,
                "use_in": ["emission"],
            }
        ]
        if with_covariate
        else []
    )
    cfg["model"] = {
        "n_states": 2,
        "engine": "expanded",
        "initial_state": "uniform",
        "emission": {
            "family": "gbm",
            "params": {
                "mu": (
                    {"mode": "fixed", "values": list(MU_TRUE)}
                    if sharp
                    else {"mode": "free", "init": list(MU_TRUE)}
                ),
                "sigma": {
                    "mode": "free",
                    "init": [0.008, 0.018],
                    "constraint": "increasing",
                    "prior": {"type": "lognormal", "sd": 1.0},
                },
            },
            "covariates": {"drift": [COV_NAME] if with_covariate else [], "log_vol": []},
        },
        "duration": {
            "family": "geometric",
            "params": {
                "rate": {
                    "mode": "free",
                    "init_mean_sojourn_steps": [MEAN_SOJOURN, MEAN_SOJOURN],
                    "prior": {"type": "lognormal", "sd": 1.0},
                }
            },
            "covariates": [],
        },
        "transition": {"mode": "jump", "order": 1, "covariates": [], "coef_prior_sd": 1.0},
    }
    cfg["estimation"].update(restarts=restarts, max_iter=300, init=["rollvol_kmeans", "lattice_jitter"])
    cfg["estimation"]["device"] = "cpu"
    if sharp:
        cfg["estimation"].update(max_iter=2000, tol_f=1.0e-16, tol_g=1.0e-11)
    cfg["forecast"] = {
        "horizon": max(horizons),
        "n_sim": n_sim,
        "covariate_forecaster": {"type": "freeze"},
        "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
    }
    cfg["evaluation"] = {
        "metrics": list(metrics),
        "backtest": {
            "scheme": scheme,
            "initial_train_years": initial_train,
            "rolling_years": initial_train if scheme == "rolling" else None,
            "refit_every_steps": refit_every,
            "horizons": list(horizons),
            "embargo_steps": 0,
        },
    }
    cfg["output"] = {"write": ["backtest"]}
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as fh:
        _YAML.dump(cfg, fh)
    return cast(FinanceConfig, load_config(path, FinanceConfig))
