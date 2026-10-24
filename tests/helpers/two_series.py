"""Two-series (`M = 2`) model and data builders for E9-T01 tests (not collected)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML

from regimekit.config.schema import ProjectConfig

APPENDIX_A = Path(__file__).resolve().parents[1] / "fixtures" / "configs" / "appendix_a.yaml"

SIGMA_STAR = np.array([[0.10, 0.15], [0.35, 0.50]])  # (N, M): regime 1 is the high-vol regime in both series
MU_STAR = np.array([[0.08, 0.05], [-0.10, -0.05]])
MEAN_SOJOURN_DAYS = (60.0, 30.0)


def two_series_config(
    *, per_series: bool = True, family: str = "gbm", symbols: tuple[str, ...] = ("A", "B")
) -> ProjectConfig:
    """A valid 2-regime, `len(symbols)`-series config: negbin durations, no covariates, free `mu`/`sigma`."""
    raw: dict[str, Any] = YAML(typ="safe").load(APPENDIX_A.read_text(encoding="utf-8"))
    raw["data"]["primary"]["symbols"] = list(symbols)
    raw["data"]["covariates"] = []
    model = raw["model"]
    model["n_states"] = 2
    model["d_max"] = 200
    if family == "gbm":
        model["emission"] = {
            "family": "gbm",
            "per_series": per_series,
            "params": {
                "mu": {"mode": "free", "init": [0.0, 0.0]},
                "sigma": {"mode": "free", "init": [0.1, 0.3]},
            },
        }
    else:
        model["emission"] = {
            "family": "student_t",
            "per_series": per_series,
            "params": {
                "m": {"mode": "free", "init": [0.0, 0.0]},
                "s": {"mode": "free", "init": [0.1, 0.3]},
                "nu": {"mode": "free", "init": [8.0, 8.0]},
            },
        }
    model["duration"] = {
        "family": "negbin",
        "r": [3, 3],
        "params": {"rate": {"mode": "free", "init_mean_sojourn_days": [60, 30]}},
        "covariates": [],
    }
    model["transition"] = {"mode": "jump", "order": 1, "covariates": []}
    raw["evaluation"]["metrics"] = ["log_score", "bic"]
    raw["evaluation"]["strategy"] = {"type": "regime_long_flat", "flat_if": {"regime": 0}, "cost_bps": 5}
    return ProjectConfig.model_validate(raw)
