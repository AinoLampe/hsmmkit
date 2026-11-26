"""E12-T05 (`-m slow`): per-state duration families and duration-dependent jumps recover their truths.

* mixed durations (state 0 geometric, state 1 gamma, state 2 nonparametric): each regime's mean sojourn
  within 25 % in >= 4 of 5 seeds (`T = 20000`, explicit engine);
* duration-dependent jumps: with `pi_02(d) / pi_01(d) ∝ d^{c_02 - c_01}`, `c_01 = 1`, `c_02 = -1` the fitted
  coefficient of the free cell `(0 -> 2)` (identified only through `c_02 - c_01 = -2`) has the right sign in
  >= 4/5 seeds.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import EstimationSection, parse_project_config
from hsmmkit.data.contracts import ObservationSet
from hsmmkit.data.prepare import Preparer
from hsmmkit.estimation import fit
from hsmmkit.model.build import build_model_spec
from tests.helpers.core_config import core_raw

pytestmark = pytest.mark.slow

MEANS = np.array([-2.0, 0.0, 2.0])
SDS = np.array([0.5, 0.7, 0.9])


def _prepare(cfg: Any, y: np.ndarray) -> Any:  # noqa: ANN401
    times = pd.bdate_range("2000-01-03", periods=y.shape[0]).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y.reshape(-1, 1), "memory", "synthetic")
    return Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])


def _gamma_pmf(shape: float, scale: float, d_max: int) -> np.ndarray:
    from scipy import stats

    edges = np.arange(0, d_max + 1)
    pmf = np.diff(stats.gamma.cdf(edges, shape, scale=scale))
    return pmf / pmf.sum()


def _simulate_mixed(seed: int, t_len: int, pmfs: list[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, state, t = np.empty(t_len), int(rng.integers(3)), 0
    while t < t_len:
        d = int(rng.choice(np.arange(1, pmfs[state].shape[0] + 1), p=pmfs[state]))
        d = min(d, t_len - t)
        y[t : t + d] = rng.normal(MEANS[state], SDS[state], d)
        t += d
        state = int(rng.choice([s for s in range(3) if s != state]))
    return y


def test_e12_t05_mixed_per_state_durations_recover_the_mean_sojourns(tmp_path: Path) -> None:
    """AC1: geometric / gamma / nonparametric, `T = 20000`, 5 seeds: mean sojourns within 25 % in >= 4."""
    d_max = 60
    geo = (1 - 1 / 5.0) ** np.arange(0, d_max) * (1 / 5.0)
    pmfs = [geo / geo.sum(), _gamma_pmf(4.0, 2.5, d_max), _gamma_pmf(8.0, 2.5, d_max)]
    true_mean = np.array([(p * np.arange(1, d_max + 1)).sum() for p in pmfs])
    raw = core_raw(tmp_path)
    raw["model"].update({"n_states": 3, "engine": "explicit", "d_max": d_max})
    raw["model"]["emission"]["params"] = {
        "mean": {"mode": "free", "init": [-1.5, 0.3, 1.5]},
        "scale": {"mode": "free", "init": [0.6, 0.8, 1.0], "constraint": "increasing"},
    }
    raw["model"]["duration"] = {
        "per_state": [
            {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 8.0}}},
            {
                "family": "gamma",
                "params": {
                    "shape": {"mode": "free", "init": 2.0},
                    "scale": {"mode": "free", "init_mean_sojourn_steps": 8.0},
                },
            },
            {"family": "nonparametric", "params": {"logits": {"mode": "free"}}},
        ]
    }
    cfg = parse_project_config(raw)
    ok = np.zeros(3, dtype=int)
    for seed in range(5):
        data = _prepare(cfg, _simulate_mixed(seed, 20_000, pmfs))
        spec = build_model_spec(cfg, data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = fit(
                spec,
                data,
                EstimationSection(restarts=2, max_iter=150, init=["random"], device="cpu"),
                seed=seed,
            )  # type: ignore[call-arg]
        est = np.asarray(spec.duration.mean_sojourn_steps_truncated(res.params, d_max, data.nominal_dt))
        ok += np.abs(est / true_mean - 1.0) <= 0.25
    assert bool(np.all(ok >= 4)), f"mean sojourns within 25 %: {ok.tolist()} of 5 seeds"


def _simulate_dd(seed: int, t_len: int) -> np.ndarray:
    """3 regimes, geometric sojourns (mean 8); from regime 0 the jump goes to 2 with prob. `1 / (1 + d^2)`."""
    rng = np.random.default_rng(seed)
    y, state, t = np.empty(t_len), 0, 0
    while t < t_len:
        d = min(int(rng.geometric(1 / 8.0)), t_len - t)
        y[t : t + d] = rng.normal(MEANS[state], SDS[state], d)
        t += d
        if state == 0:
            state = 2 if rng.random() < 1.0 / (1.0 + d**2) else 1
        else:
            state = int(rng.choice([s for s in range(3) if s != state]))
    return y


def test_e12_t05_duration_dependent_jump_coefficient_has_the_right_sign(tmp_path: Path) -> None:
    """AC2: the fitted `transition.c` of cell `(0 -> 2)` is negative (truth -2) in >= 4/5 seeds."""
    d_max = 30
    raw = core_raw(tmp_path)
    raw["model"].update({"n_states": 3, "engine": "explicit", "d_max": d_max})
    raw["model"]["emission"]["params"] = {
        "mean": {"mode": "free", "init": [-1.5, 0.3, 1.5]},
        "scale": {"mode": "free", "init": [0.6, 0.8, 1.0], "constraint": "increasing"},
    }
    raw["model"]["duration"] = {
        "family": "geometric",
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [8.0, 8.0, 8.0]}},
        "duration_dependent_jumps": True,
    }
    cfg = parse_project_config(raw)
    signs = []
    for seed in range(5):
        data = _prepare(cfg, _simulate_dd(seed, 6_000))
        spec = build_model_spec(cfg, data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = fit(
                spec,
                data,
                EstimationSection(restarts=1, max_iter=120, init=["random"], device="cpu"),
                seed=seed,
            )  # type: ignore[call-arg]
        c = np.asarray(res.params["transition.c"])
        signs.append(float(c[0]) < 0.0)  # flat cells: (0,2), (1,2), (2,1): the first is row 0's free cell

    assert sum(signs) >= 4, f"signs {signs}"
