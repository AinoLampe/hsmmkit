"""Small EM test models (not collected): Gaussian (diag / full), Poisson, Categorical; synthetic data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.contracts import PreparedDataset
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.model.spec import ModelSpec
from tests.helpers.core_config import core_raw
from tests.helpers.engine_fixtures import make_dataset


def regimes(t_len: int, seed: int, sojourn: float = 25.0) -> np.ndarray:
    """A 2-regime path with geometric sojourns (mean `sojourn` steps)."""
    rng = np.random.default_rng(seed)
    state, out = 0, np.empty(t_len, dtype=int)
    for t in range(t_len):
        out[t] = state
        if rng.random() < 1.0 / sojourn:
            state = 1 - state
    return out


def observations(family: str, t_len: int, seed: int) -> np.ndarray:
    """`(T, M)` observations of the family under a regime path."""
    rng = np.random.default_rng(seed + 1000)
    z = regimes(t_len, seed)
    if family == "gaussian":
        return rng.normal(np.where(z == 0, 0.0, 1.5), np.where(z == 0, 0.5, 1.2))[:, None]
    if family == "gaussian_full":
        mean = np.where(z[:, None] == 0, [0.0, 0.0], [1.5, -1.0])
        cov = [np.array([[0.4, 0.1], [0.1, 0.3]]), np.array([[1.5, -0.6], [-0.6, 1.0]])]
        out = np.empty((t_len, 2))
        for t in range(t_len):
            out[t] = rng.multivariate_normal(mean[t], cov[z[t]])
        return out
    if family == "poisson":
        return rng.poisson(np.where(z == 0, 2.0, 9.0)).astype(float)[:, None]
    if family == "categorical":
        probs = np.array([[0.7, 0.2, 0.1], [0.15, 0.35, 0.5]])
        return np.array([rng.choice(3, p=probs[k]) for k in z], dtype=float)[:, None]
    raise ValueError(family)


def emission_cfg(family: str) -> dict[str, Any]:
    """The `model.emission` section of the family, with starting values off the truth."""
    if family == "gaussian":
        return {
            "family": "gaussian",
            "scale_by_dt": False,
            "ito_correction": False,
            "params": {
                "mean": {"mode": "free", "init": [0.3, 1.0]},
                "scale": {"mode": "free", "init": [0.7, 1.0]},
            },
        }
    if family == "gaussian_full":
        return {
            "family": "gaussian",
            "covariance": "full",
            "scale_by_dt": False,
            "ito_correction": False,
            "params": {
                "mean": {"mode": "free", "init": [0.2, 1.0]},
                "chol": {"mode": "free", "init": [0.8, 0.0, 0.8]},
            },
        }
    if family == "poisson":
        return {
            "family": "poisson",
            "scale_by_dt": False,
            "params": {"rate": {"mode": "free", "init": [3.0, 6.0]}},
        }
    if family == "categorical":
        return {
            "family": "categorical",
            "params": {"probs": {"mode": "free", "init": [0.5, 0.3, 0.2]}},
        }
    raise ValueError(family)


def build(
    tmp_path: Path,
    family: str,
    duration: str = "geometric",
    **model: Any,  # noqa: ANN401
) -> tuple[ModelSpec, dict[str, jnp.ndarray]]:
    """`(spec, theta0)` of a 2-regime model; `theta0` holds the configured starting values."""
    raw = core_raw(tmp_path)
    raw["model"]["emission"] = emission_cfg(family)
    raw["model"]["duration"] = (
        {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [15.0, 30.0]}}}
        if duration == "geometric"
        else {
            "family": "negbin",
            "r": [2, 3],
            "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [15.0, 30.0]}},
        }
    )
    raw["model"].update(model)
    cfg = parse_project_config(raw)
    n_series = 2 if family == "gaussian_full" else 1
    spec = build_model_spec_from(
        cfg, BuildInputs(n_series=n_series, cov_names={}, clock="steps", nominal_dt=1.0)
    )
    manager = ParamManager(spec.param_specs)
    theta = dict(manager.unflatten(manager.init_u()))
    if family == "categorical":  # per-regime starting probabilities (the config takes one shared vector)
        start = np.array([[0.5, 0.3, 0.2], [0.3, 0.3, 0.4]])
        theta["emission.eta"] = jnp.asarray(np.log(start[:, 1:] / start[:, :1]))
    return spec, theta


def dataset(family: str, t_len: int, seed: int) -> PreparedDataset:
    """A `steps`-grid dataset of the family (`dt = 1`)."""
    y = observations(family, t_len, seed)
    return make_dataset(y, np.ones(t_len))
