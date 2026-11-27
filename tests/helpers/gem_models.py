"""Explicit-engine GEM test models (not collected): Gaussian emissions, nonparametric / geometric duration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hsmmkit.config.schema import parse_project_config
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.spec import ModelSpec
from tests.helpers.core_config import core_raw
from tests.helpers.engine_fixtures import make_dataset


def simulate(
    t_len: int, pmf: np.ndarray, pi: np.ndarray, means: np.ndarray, sds: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """`(y (T, 1), regimes (T,))` of an explicit-duration chain (`pmf (N, D)` over `d = 1..D`)."""
    rng = np.random.default_rng(seed)
    n, d_max = pmf.shape
    z: list[int] = []
    state = int(rng.integers(n))
    while len(z) < t_len:
        d = int(rng.choice(d_max, p=pmf[state])) + 1
        z.extend([state] * d)
        state = int(rng.choice(n, p=pi[state]))
    path = np.asarray(z[:t_len])
    return rng.normal(means[path], sds[path])[:, None], path


def build(
    tmp_path: Path,
    n: int,
    d_max: int,
    *,
    duration: str = "nonparametric",
    left: bool = False,
    mean_sojourn: float = 6.0,
    means: list[float] | None = None,
    scales: list[float] | None = None,
    **model: Any,  # noqa: ANN401
) -> ModelSpec:
    """An `n`-regime explicit-engine model with Gaussian emissions."""
    raw = core_raw(tmp_path)
    m = raw["model"]
    m["n_states"] = n
    m["engine"] = "explicit"
    m["d_max"] = d_max
    m["left_censored"] = left
    m["initial_state"] = "uniform"
    m["emission"]["scale_by_dt"] = False
    m["emission"]["params"] = {
        "mean": {"mode": "free", "init": means or [0.3 * i for i in range(n)]},
        "scale": {"mode": "free", "init": scales or [0.6 + 0.2 * i for i in range(n)]},
    }
    init = {"init_mean_sojourn_steps": [mean_sojourn] * n}
    m["duration"] = (
        {"family": "nonparametric", "params": {"logits": {"mode": "free", **init}}}
        if duration == "nonparametric"
        else {"family": duration, "params": {"rate": {"mode": "free", **init}}}
    )
    if duration == "negbin":
        m["duration"]["r"] = [2] * n
    m.update(model)
    cfg = parse_project_config(raw)
    return build_model_spec_from(cfg, BuildInputs(n_series=1, cov_names={}, clock="steps", nominal_dt=1.0))


def dataset(y: np.ndarray):  # noqa: ANN201
    """A `steps`-grid dataset (`dt = 1`) of `y (T, 1)`."""
    return make_dataset(y, np.ones(y.shape[0]))
