"""E11-T08 AC1 (`-m slow`): recovery of a 2x2 covariance per state through a `product` block, 3 seeds.

Two regimes, `T = 10000`: a full-covariance Gaussian block on columns (0, 1) and an independent Poisson
block on column 2. Every entry of each state's 2x2 covariance within 15 % relative, in each of the 3 seeds.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import EstimationSection, parse_project_config
from hsmmkit.data.contracts import ObservationSet
from hsmmkit.data.prepare import Preparer
from hsmmkit.estimation import fit
from hsmmkit.model.build import build_model_spec
from hsmmkit.model.dists import unpack_lower
from tests.helpers.core_config import core_raw

pytestmark = pytest.mark.slow

T_LEN = 10_000
SIGMA = np.array([[[1.0, 0.5], [0.5, 1.5]], [[3.0, -1.0], [-1.0, 2.0]]])
MEAN = np.array([[0.0, 0.0], [1.5, -1.0]])
RATE = np.array([2.0, 6.0])
SOJOURN = 40.0


def _simulate(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(SIGMA)
    s, out = 0, np.empty((T_LEN, 3))
    for t in range(T_LEN):
        out[t, :2] = MEAN[s] + chol[s] @ rng.normal(size=2)
        out[t, 2] = rng.poisson(RATE[s])
        if rng.random() < 1.0 / SOJOURN:
            s = 1 - s
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e11_t08_two_series_covariance_recovery(seed: int, tmp_path: Path) -> None:
    """AC1: all covariance entries of both states within 15 % relative."""
    raw = core_raw(tmp_path)
    raw["data"]["source"]["observation_columns"] = ["a", "b", "c"]
    raw["model"]["emission"] = {
        "family": "product",
        "params": {},
        "blocks": [
            {
                "family": "gaussian",
                "columns": [0, 1],
                "covariance": "full",
                "params": {"mean": {"mode": "free", "init": [0.2, 1.0]}, "chol": {"mode": "free"}},
            },
            {
                "family": "poisson",
                "columns": [2],
                "scale_by_dt": False,
                "params": {"rate": {"mode": "free", "init": [3.0, 5.0]}},
            },
        ],
    }
    raw["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [30.0, 30.0]
    cfg = parse_project_config(raw)
    y = _simulate(seed)
    times = pd.bdate_range("2000-01-03", periods=T_LEN).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("a", "b", "c"), times, y, "memory", "synthetic")
    data = (
        Preparer(cfg.data, series_support=("real", "real", "count"))
        .fit(obs, [], train_end=times[-1])
        .transform(obs, [])
    )
    spec = build_model_spec(cfg, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = fit(
            spec, data, EstimationSection(restarts=2, max_iter=300, init=["random"], device="cpu"), seed=seed
        )  # type: ignore[call-arg]
    chol = np.asarray(unpack_lower(res.params["emission.b0.chol"], 2))
    sigma_hat = chol @ np.swapaxes(chol, -1, -2)
    rel = np.abs(sigma_hat - SIGMA) / np.abs(SIGMA)
    assert float(rel.max()) <= 0.15, f"seed {seed}: relative covariance errors\n{rel}\n{sigma_hat}"
