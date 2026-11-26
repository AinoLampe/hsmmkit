"""E12-T02 (`-m slow`): MAP on 50 sequences x T = 200 recovers what a single T = 10 000 sequence does.

Gaussian (diag), 3 regimes, negative-binomial durations with `r = 2` (expanded engine), 5 seeds. Tolerances of
Appendix D / G, each category in >= 4 of 5 seeds: sigma within 10 % relative, mean sojourn within 25 %
relative, the jump probabilities within 0.15 absolute. The batch fit and the single-sequence fit are held to
the same bar.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection, parse_project_config
from hsmmkit.estimation import fit
from hsmmkit.inference.simulate import simulate_batch
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.params.manager import ParamManager
from tests.helpers.core_config import core_raw

pytestmark = pytest.mark.slow

SIGMA = np.array([0.5, 1.0, 1.5])
PI = np.array([[0.7, 0.3], [0.4, 0.6], [0.3, 0.7]])
CFG = EstimationSection(restarts=3, max_iter=250, init=["rolling_kmeans", "random"], device="cpu")


def _spec_and_truth(tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    raw = core_raw(tmp_path)
    raw["model"]["n_states"] = 3
    raw["model"]["emission"]["params"] = {
        "mean": {"mode": "free", "init": [-2.0, 0.0, 2.0]},
        "scale": {"mode": "free", "init": SIGMA.tolist(), "constraint": "increasing"},
    }
    raw["model"]["duration"] = {
        "family": "negbin",
        "r": [2, 2, 2],
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [10.0, 15.0, 20.0]}},
    }
    spec = build_model_spec_from(parse_project_config(raw), BuildInputs(1, {}, "steps", 1.0))
    manager = ParamManager(spec.param_specs)
    theta = dict(manager.unflatten(manager.init_u()))
    theta["transition.pi_rows"] = np.asarray(PI)
    return spec, theta


def _categories(spec: Any, truth: dict[str, Any], hat: dict[str, Any]) -> dict[str, bool]:  # noqa: ANN401
    sigma = np.asarray(hat["emission.scale"]).ravel()
    soj_true = np.asarray(spec.duration.mean_sojourn(truth, 1.0))
    soj_hat = np.asarray(spec.duration.mean_sojourn(hat, 1.0))
    pi_hat = np.asarray(hat["transition.pi_rows"])
    return {
        "sigma": bool(np.all(np.abs(sigma / SIGMA - 1.0) <= 0.10)),
        "mean_sojourn": bool(np.all(np.abs(soj_hat / soj_true - 1.0) <= 0.25)),
        "jump": bool(np.all(np.abs(pi_hat - PI) <= 0.15)),
    }


@pytest.mark.parametrize("mode", ["batch_50x200", "single_10000"])
def test_e12_t02_batch_recovery_matches_single_sequence_recovery(mode: str, tmp_path: Path) -> None:
    """AC1: every tolerance category passes in >= 4 of 5 seeds, for the batch and for one long sequence."""
    spec, truth = _spec_and_truth(tmp_path)
    passed = {"sigma": 0, "mean_sojourn": 0, "jump": 0}
    for seed in range(5):
        lengths = [200] * 50 if mode == "batch_50x200" else [10_000]
        batch = simulate_batch(truth, spec, lengths, jax.random.key(100 + seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = fit(spec, batch, CFG, seed=seed)
        for name, ok in _categories(spec, truth, res.params).items():
            passed[name] += int(ok)
    assert all(v >= 4 for v in passed.values()), f"{mode}: seeds passing per category {passed} of 5"
