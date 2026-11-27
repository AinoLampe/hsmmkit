"""E13-T06b AC2: post-hoc relabelling of posterior draws."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hsmmkit.config.schema import parse_project_config
from hsmmkit.estimation.relabel import location_features, relabel_draws, relabel_permutation
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.lattice import permute_states
from hsmmkit.model.params.manager import ParamManager
from tests.helpers.core_config import core_raw


def _spec(tmp_path: Path, n: int = 3):  # noqa: ANN202
    raw = core_raw(tmp_path)
    m = raw["model"]
    m["n_states"] = n
    m["emission"] = {
        "family": "gaussian",
        "scale_by_dt": False,
        "ito_correction": False,
        "params": {
            "mean": {"mode": "free", "init": [-2.0, 0.0, 2.0][:n]},
            "scale": {"mode": "free", "init": [0.5, 0.8, 1.1][:n]},
        },
    }
    m["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [10.0, 20.0, 30.0][:n]
    return build_model_spec_from(
        parse_project_config(raw), BuildInputs(n_series=1, cov_names={}, clock="steps", nominal_dt=1.0)
    )


def _draws(spec, pivot, n_chain: int, n_draw: int, rng):  # noqa: ANN001, ANN202
    """Draws near `pivot` (small noise), as arrays `(chain, draw, *shape)`."""
    return {
        k: np.asarray(v)[None, None] * (1.0 + 0.01 * rng.normal(size=(n_chain, n_draw, *np.shape(v))))
        + 0.01 * rng.normal(size=(n_chain, n_draw, *np.shape(v)))
        for k, v in pivot.items()
    }


def test_e13_t06b_random_permutations_are_undone(tmp_path: Path) -> None:
    """Randomly permuted draws return to the pivot labelling (>= 99 %; here every draw)."""
    spec = _spec(tmp_path)
    manager = ParamManager(spec.param_specs)
    pivot = dict(manager.unflatten(manager.init_u()))
    rng = np.random.default_rng(0)
    clean = _draws(spec, pivot, 2, 150, rng)
    shuffled = {k: v.copy() for k, v in clean.items()}
    for c in range(2):
        for d in range(150):
            perm = rng.permutation(3)
            theta = permute_states({k: v[c, d] for k, v in clean.items()}, spec, perm)
            for k, v in theta.items():
                shuffled[k][c, d] = np.asarray(v)
    fixed, fraction = relabel_draws(shuffled, spec, pivot)
    same = np.array(
        [
            all(np.allclose(fixed[k][c, d], clean[k][c, d], rtol=0.0, atol=1e-12) for k in clean)
            for c in range(2)
            for d in range(150)
        ]
    )
    assert same.mean() >= 0.99
    assert 0.7 < fraction < 0.95  # 5 of the 6 permutations move the labels


def test_e13_t06b_a_permuted_model_returns_its_original_posterior_means(tmp_path: Path) -> None:
    """Permuting all states of a posterior and relabelling gives back the original means (<= 1e-12)."""
    spec = _spec(tmp_path)
    manager = ParamManager(spec.param_specs)
    pivot = dict(manager.unflatten(manager.init_u()))
    clean = _draws(spec, pivot, 2, 80, np.random.default_rng(1))
    perm = np.array([2, 0, 1])
    permuted = {k: np.zeros_like(v) for k, v in clean.items()}
    for c in range(2):
        for d in range(80):
            theta = permute_states({k: v[c, d] for k, v in clean.items()}, spec, perm)
            for k, v in theta.items():
                permuted[k][c, d] = np.asarray(v)
    fixed, fraction = relabel_draws(permuted, spec, pivot)
    assert fraction == 1.0
    for k in clean:
        np.testing.assert_allclose(fixed[k].mean(axis=(0, 1)), clean[k].mean(axis=(0, 1)), atol=1e-12)


def test_e13_t06b_the_permutation_minimizes_the_standardized_distance() -> None:
    """Features `(N, K)`: the assignment matches near states; ties in scale fall back to a unit scale."""
    pivot = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    draw = pivot[[2, 0, 1]] + 0.1  # new state i is old state perm[i]; relabelling must undo it
    perm = relabel_permutation(draw, pivot, np.ones(2))
    np.testing.assert_array_equal(draw[perm], pivot + 0.1)
    feats = location_features({"emission.mean": np.zeros((3, 1)), "emission.scale": np.ones((3, 1))}, 3)
    assert feats.shape == (3, 2)  # location and log scale
