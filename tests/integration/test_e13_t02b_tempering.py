"""E13-T02b AC3: tempered EM finds the global optimum of a multimodal fixture where plain EM does not.

Fixture (chosen after a sweep of starts, recorded in STATUS): 3-state Gaussian data (means -1.2 / 0 / 1.2,
sd 0.5, mean sojourn 25 steps, T = 900, seeds 0-4). The start puts all three states at the left cluster
(means -1.4 / -1.3 / -1.2) with scales 0.05, so plain EM keeps the two state groups on the wrong clusters
(a local optimum far below the global one); the reference optimum is EM from a start at the truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hsmmkit.config.schema import EmCfg, EstimationSection, TemperingCfg, parse_project_config
from hsmmkit.estimation.em import EMEstimator
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.params.manager import ParamManager
from tests.helpers.core_config import core_raw
from tests.helpers.engine_fixtures import make_dataset


def _data(seed: int, t_len: int = 900):  # noqa: ANN202
    rng = np.random.default_rng(seed)
    z = np.empty(t_len, dtype=int)
    state = 0
    for t in range(t_len):
        z[t] = state
        if rng.random() < 1 / 25:
            state = int(rng.choice([k for k in range(3) if k != state]))
    y = rng.normal(np.array([-1.2, 0.0, 1.2])[z], 0.5)
    return make_dataset(y[:, None], np.ones(t_len))


def _spec(tmp_path: Path, means: list[float], scales: list[float]):  # noqa: ANN202
    raw = core_raw(tmp_path)
    raw["model"]["n_states"] = 3
    raw["model"]["emission"] = {
        "family": "gaussian",
        "scale_by_dt": False,
        "ito_correction": False,
        "params": {"mean": {"mode": "free", "init": means}, "scale": {"mode": "free", "init": scales}},
    }
    raw["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [15.0] * 3
    cfg = parse_project_config(raw)
    return build_model_spec_from(cfg, BuildInputs(n_series=1, cov_names={}, clock="steps", nominal_dt=1.0))


@pytest.mark.slow
def test_e13_t02b_tempering_reaches_the_global_optimum_where_plain_em_does_not(tmp_path: Path) -> None:
    """Tempered EM: >= 4/5 seeds at the global optimum; plain EM: <= 2/5."""
    good = _spec(tmp_path, [-1.0, 0.0, 1.0], [0.6, 0.6, 0.6])
    bad = _spec(tmp_path, [-1.4, -1.3, -1.2], [0.05, 0.05, 0.05])
    start = ParamManager(bad.param_specs).init_u()
    plain_cfg = EstimationSection(restarts=0, max_iter=3000, tol_f=1e-12, use_priors=False)
    temp_cfg = EstimationSection(
        restarts=0,
        max_iter=3000,
        tol_f=1e-12,
        use_priors=False,
        em=EmCfg(tempering=TemperingCfg(enabled=True, beta0=0.3, steps=30)),
    )
    hits = {"plain": 0, "tempered": 0}
    for seed in range(5):
        data = _data(seed)
        ref = (
            EMEstimator()
            .fit(good, data, plain_cfg, 0, warm_start=ParamManager(good.param_specs).init_u())
            .loglik
        )
        for name, cfg in (("plain", plain_cfg), ("tempered", temp_cfg)):
            ll = EMEstimator().fit(bad, data, cfg, 0, warm_start=start).loglik
            hits[name] += int(ll >= ref - 1e-3)
    assert hits["tempered"] >= 4
    assert hits["plain"] <= 2
