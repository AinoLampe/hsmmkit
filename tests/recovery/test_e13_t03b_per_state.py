"""E13-T03b AC3: a per-state duration mix (gamma / gamma / geometric / nonparametric) fitted by GEM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation.gem import GEMEstimator
from hsmmkit.model.params.manager import ParamManager
from tests.helpers import gem_models

D_MAX = 60


def _truth() -> np.ndarray:
    edges = np.arange(0, D_MAX + 1, dtype=float)
    g0 = np.diff(stats.gamma.cdf(edges, a=4.0, scale=1.5))  # mean 6
    g1 = np.diff(stats.gamma.cdf(edges, a=3.0, scale=5.0))  # mean 15
    geo = 0.125 * 0.875 ** np.arange(D_MAX)  # mean ~8
    npar = np.zeros(D_MAX)
    npar[[4, 5, 6, 17, 18, 19]] = [1, 2, 1, 1, 2, 1]  # bimodal, mean ~ 12
    pmf = np.stack([g0, g1, geo, npar])
    return pmf / pmf.sum(axis=1, keepdims=True)


@pytest.mark.slow
def test_e13_t03b_per_state_mean_sojourns_are_recovered(tmp_path: Path) -> None:
    """Mean sojourn of each regime within 25 % in >= 4 of 5 seeds."""
    pmf = _truth()
    d = np.arange(1, D_MAX + 1)
    true_mean = pmf @ d
    pi = (np.ones((4, 4)) - np.eye(4)) / 3.0
    cfg_dur = {
        "per_state": [
            {
                "family": "gamma",
                "params": {"shape": {"mode": "free", "init": 2.0}, "scale": {"mode": "free", "init": 3.0}},
            },
            {
                "family": "gamma",
                "params": {"shape": {"mode": "free", "init": 2.0}, "scale": {"mode": "free", "init": 6.0}},
            },
            {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 10.0}}},
            {
                "family": "nonparametric",
                "params": {"logits": {"mode": "free", "init_mean_sojourn_steps": 10.0}},
            },
        ]
    }
    spec = gem_models.build(
        tmp_path,
        4,
        D_MAX,
        duration_cfg=cfg_dur,
        means=[-2.7, -0.8, 0.8, 2.7],
        scales=[0.8, 0.8, 0.8, 0.8],
    )
    cfg = EstimationSection(method="gem", restarts=0, max_iter=300, tol_f=1e-9, use_priors=False)
    passed = 0
    for seed in range(5):
        y, _ = gem_models.simulate(
            12_000, pmf, pi, np.array([-3.0, -1.0, 1.0, 3.0]), np.array([0.6, 0.6, 0.6, 0.6]), 20 + seed
        )
        res = GEMEstimator().fit(
            spec, gem_models.dataset(y), cfg, 0, warm_start=ParamManager(spec.param_specs).init_u()
        )
        est = np.asarray(spec.duration.mean_sojourn_steps_truncated(res.params, D_MAX, 1.0))
        order = np.argsort(np.asarray(res.params["emission.mean"]).ravel())
        passed += int(np.all(np.abs(est[order] / true_mean - 1.0) <= 0.25))
    assert passed >= 4
