"""E13-T05b: pruning of unused regimes, the VB fit report, `decode` metadata."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection, parse_project_config
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.estimation.vb import VBEstimator
from hsmmkit.inference.decode import decode
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.persistence.store import ModelStore
from tests.helpers.core_config import core_raw
from tests.helpers.engine_fixtures import make_dataset


def _data(seed: int, t_len: int = 5000):  # noqa: ANN202
    """Three well-separated regimes (means -2 / 0 / 2, sd 0.5, mean sojourn 40)."""
    rng = np.random.default_rng(seed)
    z = np.empty(t_len, dtype=int)
    state = 0
    for t in range(t_len):
        z[t] = state
        if rng.random() < 1 / 40:
            state = int(rng.choice([k for k in range(3) if k != state]))
    return make_dataset(rng.normal(np.array([-2.0, 0.0, 2.0])[z], 0.5)[:, None], np.ones(t_len))


def _raw(tmp_path: Path, n_states: int, means: list[float]) -> dict:  # type: ignore[type-arg]
    raw = core_raw(tmp_path)
    m = raw["model"]
    m["n_states"] = n_states
    m["emission"] = {
        "family": "gaussian",
        "scale_by_dt": False,
        "ito_correction": False,
        "params": {
            "mean": {"mode": "free", "init": means},
            "scale": {"mode": "free", "init": [0.8] * n_states},
        },
    }
    m["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [20.0] * n_states
    return raw


def _spec(tmp_path: Path, n_states: int, means: list[float]):  # noqa: ANN202
    return build_model_spec_from(
        parse_project_config(_raw(tmp_path, n_states, means)),
        BuildInputs(n_series=1, cov_names={}, clock="steps", nominal_dt=1.0),
    )


def test_e13_t05b_a_four_state_fit_of_three_regime_data_prunes_one_state(tmp_path: Path) -> None:
    """AC1: exactly one pruned regime in >= 4 of 5 seeds; AC4: `decode` carries the pruned states."""
    spec = _spec(tmp_path, 4, [-2.5, -0.5, 0.5, 2.5])
    start = ParamManager(spec.param_specs).init_u()
    cfg = EstimationSection(method="vb", restarts=0, max_iter=300, tol_f=1e-10)
    hits = 0
    last = None
    for seed in range(5):
        data = _data(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = VBEstimator().fit(spec, data, cfg, 0, warm_start=start)
        assert any("smaller model.n_states" in str(w.message) for w in caught) == (len(res.pruned_states) > 0)
        hits += int(len(res.pruned_states) == 1)
        last = (res, data)
    assert hits >= 4
    res, data = last  # the fit report and `decode` metadata
    occupancy = np.asarray(res.diagnostics["vb"]["occupancy"])
    assert occupancy.shape == (4,)
    assert occupancy.sum() == pytest.approx(1.0)
    assert all(occupancy[i] < 0.01 for i in res.pruned_states)
    assert set(res.diagnostics["vb"]["treatment"].values()) == {"conjugate"}
    assert res.std_errors is not None
    assert set(res.std_errors) >= {"emission.mean", "emission.scale", "duration.rate"}
    assert np.all(np.isfinite(np.asarray(res.std_errors["emission.mean"])))
    post = decode(res, data)
    assert post.pruned_states == res.pruned_states
    # the pruned regime is the unused one in `params` (spurious wide scale, weakest posterior mean)
    (pruned,) = res.pruned_states
    assert int(np.argmax(np.asarray(res.params["emission.scale"]).ravel())) == pruned
    assert int(np.argmax(np.asarray(res.std_errors["emission.mean"]).ravel())) == pruned
    batch_post = decode(res, SequenceBatch((data, data), (8192,)))
    assert all(p.pruned_states == res.pruned_states for p in batch_post)
    bundle = tmp_path / "vb.hsmm"
    cfg = parse_project_config(_raw(tmp_path, 4, [-2.5, -0.5, 0.5, 2.5]))
    ModelStore.save(res, bundle, cfg, data)
    again = ModelStore.load(bundle)
    assert again.pruned_states == res.pruned_states
    assert again.estimator == "vb"
    np.testing.assert_allclose(
        np.asarray(again.std_errors["emission.mean"]), np.asarray(res.std_errors["emission.mean"])
    )


def test_e13_t05b_prune_threshold_comes_from_the_config(tmp_path: Path) -> None:
    """`estimation.vb.prune_below = 0` reports nothing; a large threshold reports the small regimes."""
    spec = _spec(tmp_path, 3, [-2.2, 0.3, 2.2])
    data = _data(1, 1500)
    start = ParamManager(spec.param_specs).init_u()
    base = {"method": "vb", "restarts": 0, "max_iter": 100, "tol_f": 1e-9}
    none = EstimationSection.model_validate({**base, "vb": {"prune_below": 0.0}})
    big = EstimationSection.model_validate({**base, "vb": {"prune_below": 0.6}})
    assert VBEstimator().fit(spec, data, none, 0, warm_start=start).pruned_states == ()
    with pytest.warns(UserWarning, match="pruned"):
        assert len(VBEstimator().fit(spec, data, big, 0, warm_start=start).pruned_states) >= 1
