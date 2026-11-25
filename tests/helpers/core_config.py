"""A core (`ProjectConfig`) two-regime Gaussian backtest config on the `steps` clock (not collected).

No finance import: the observations come from `data_loader`, the config only needs a registered source kind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regimekit.config.schema import ProjectConfig, parse_project_config
from regimekit.data.contracts import ObservationSet

N_OBS = 300


def simulate_values(n: int = N_OBS, seed: int = 3) -> np.ndarray:
    """A 2-regime Gaussian series (per-step mean 0.0005 / -0.001, sd 0.006 / 0.02, mean sojourn 30 steps)."""
    rng = np.random.default_rng(seed)
    state, out = 0, np.empty(n)
    for t in range(n):
        out[t] = rng.normal((0.0005, -0.001)[state], (0.006, 0.02)[state])
        if rng.random() < 1 / 30:
            state = 1 - state
    return out


def observations(values: np.ndarray) -> ObservationSet:
    """`values` as a one-series `ObservationSet` on consecutive business days."""
    times = pd.bdate_range("2020-01-01", periods=values.shape[0]).to_numpy().astype("datetime64[ns]")
    return ObservationSet(("x",), times, values.reshape(-1, 1), "memory", "synthetic")


def core_config(tmp_path: Path, *, metrics: tuple[str, ...] = ("log_score", "bic")) -> ProjectConfig:
    """The shared core config: `gaussian` emission, geometric durations, 3 folds."""
    raw: dict[str, Any] = {
        "schema_version": 2,
        "project": {"name": "core_demo", "seed": 11, "output_dir": str(tmp_path / "runs")},
        "data": {
            "source": {"kind": "csv", "path": "unused.csv", "observation_columns": ["x"]},
            "clock": "steps",
        },
        "model": {
            "n_states": 2,
            "engine": "expanded",
            "initial_state": "uniform",
            "emission": {
                "family": "gaussian",
                "ito_correction": False,
                "params": {
                    "mean": {"mode": "free", "init": [0.0005, -0.001]},
                    "scale": {"mode": "free", "init": [0.006, 0.02], "constraint": "increasing"},
                },
            },
            "duration": {
                "family": "geometric",
                "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [30.0, 30.0]}},
                "covariates": [],
            },
            "transition": {"mode": "jump", "order": 1, "covariates": []},
        },
        "estimation": {"restarts": 1, "max_iter": 60, "init": ["rolling_kmeans"], "device": "cpu"},
        "forecast": {
            "horizon": 1,
            "n_sim": 50,
            "covariate_forecaster": {"type": "freeze"},
            "quantiles": [0.05, 0.5, 0.95],
        },
        "evaluation": {
            "metrics": list(metrics),
            "backtest": {
                "scheme": "expanding",
                "initial_train_steps": 160,
                "refit_every_steps": 60,
                "horizons": [1],
                "embargo_steps": 0,
            },
        },
        "output": {"write": ["backtest"]},
    }
    return parse_project_config(raw)
