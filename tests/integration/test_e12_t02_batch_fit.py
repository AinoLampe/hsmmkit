"""E12-T02 integration: MAP on a `SequenceBatch`, per-group initial distributions, score, batched simulate."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import EstimationSection, parse_project_config
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.batch import batched_loglik
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.estimation import fit
from hsmmkit.inference.simulate import simulate_batch
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.presets import get_preset
from hsmmkit.presets.fixtures import segmentation_gaussian_ltr
from hsmmkit.project import Project
from hsmmkit.statespace.builders import build_expanded
from tests.helpers.core_config import core_raw

CFG = EstimationSection(restarts=2, max_iter=60, init=["rolling_kmeans", "random"], device="cpu")
MEAN_SOJOURN = 12.0


def _raw(tmp_path: Path, *, per_group: bool = False, engine: str = "expanded") -> dict[str, Any]:
    raw = core_raw(tmp_path)
    m = raw["model"]
    m["engine"] = engine
    m["d_max"] = 40
    m["emission"]["params"] = {
        "mean": {"mode": "free", "init": [-1.0, 1.0]},
        "scale": {"mode": "free", "init": [0.5, 1.0], "constraint": "increasing"},
    }
    m["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [MEAN_SOJOURN, MEAN_SOJOURN]
    if per_group:
        m["initial_state"] = "free"
        m["initial_groups"] = "per_group"
    return raw


def _spec(tmp_path: Path, n_groups: int = 1, **kw: Any) -> Any:  # noqa: ANN401
    cfg = parse_project_config(_raw(tmp_path, **kw))
    return build_model_spec_from(cfg, BuildInputs(1, {}, "steps", 1.0, n_groups))


def _truth(spec: Any) -> dict[str, Any]:  # noqa: ANN401
    manager = ParamManager(spec.param_specs)
    theta = dict(manager.unflatten(manager.init_u()))
    return _set(theta, [-1.0, 1.5], [0.5, 1.0])


def _set(theta: dict[str, Any], mean: list[float], scale: list[float]) -> dict[str, Any]:
    """Overwrite the emission parameters (shapes follow the model: `(N, M)` here, `M = 1`)."""
    theta["emission.mean"] = jnp.asarray(mean).reshape(np.shape(theta["emission.mean"]))
    theta["emission.scale"] = jnp.asarray(scale).reshape(np.shape(theta["emission.scale"]))
    return theta


def test_e12_t02_per_group_initial_spec_has_one_row_per_group(tmp_path: Path) -> None:
    """`per_group` + `free` gives `initial.logits (G, S_phase1)`; `shared` + `free` gives `(S_phase1,)`."""
    spec = _spec(tmp_path, 3, per_group=True)
    shapes = {s.name: s.shape for s in spec.param_specs}
    assert shapes["initial.logits"] == (3, 2)
    assert spec.n_groups == 3
    raw = _raw(tmp_path)
    raw["model"]["initial_state"] = "free"
    shared = build_model_spec_from(parse_project_config(raw), BuildInputs(1, {}, "steps", 1.0))
    assert {s.name: s.shape for s in shared.param_specs}["initial.logits"] == (2,)
    assert shared.n_groups == 1


def test_e12_t02_per_group_needs_a_free_initial_distribution(tmp_path: Path) -> None:
    """`initial_groups: per_group` with `initial_state: uniform` is a `ConfigError` naming the key."""
    raw = _raw(tmp_path)
    raw["model"]["initial_groups"] = "per_group"
    with pytest.raises(Exception, match="initial_state: free") as info:
        build_model_spec_from(parse_project_config(raw), BuildInputs(1, {}, "steps", 1.0, 2))
    assert info.value.path == "model.initial_groups"  # type: ignore[attr-defined]


def test_e12_t02_simulate_batch_is_deterministic_and_sojourns_match(tmp_path: Path) -> None:
    """AC4: same key -> same batch; the mean sojourn (geometric, mean 12) is within 3 SE of the analytic."""
    spec = _spec(tmp_path)
    theta = _truth(spec)
    key = jax.random.key(5)
    a = simulate_batch(theta, spec, [400] * 40, key)
    b = simulate_batch(theta, spec, [400] * 40, key)
    for x, y in zip(a.sequences, b.sequences, strict=True):
        assert np.array_equal(np.asarray(x.observations), np.asarray(y.observations))
    other = simulate_batch(theta, spec, [400] * 40, jax.random.key(6))
    assert not np.array_equal(
        np.asarray(a.sequences[0].observations), np.asarray(other.sequences[0].observations)
    )
    # regimes are not stored: re-derive run lengths from the two well-separated emissions (sd 0.5 / 1.0)
    space = build_expanded(spec)
    runs: list[int] = []
    for k in range(40):
        res = _simulate_regimes(theta, spec, space, 400, jax.random.fold_in(key, k))
        change = np.flatnonzero(np.diff(res)) + 1
        edges = np.concatenate(([0], change, [res.shape[0]]))
        runs.extend(np.diff(edges)[1:-1].tolist())  # drop the censored first and last run
    runs_arr = np.asarray(runs, dtype=float)
    se = runs_arr.std(ddof=1) / np.sqrt(runs_arr.size)
    assert abs(runs_arr.mean() - MEAN_SOJOURN) <= 3 * se


def _simulate_regimes(theta: Any, spec: Any, space: Any, n: int, key: Any) -> np.ndarray:  # noqa: ANN401
    from hsmmkit.inference.simulate import simulate_paths

    return np.asarray(simulate_paths(key, theta, spec, space, jnp.ones(n), None, None, None, 1).regimes[0])


def test_e12_t02_batch_fit_recovers_the_truth_and_reports_per_sequence(tmp_path: Path) -> None:
    """A 20 x 150 batch fit recovers the emission means/scales loosely and returns the per-sequence table."""
    spec = _spec(tmp_path)
    theta = _truth(spec)
    batch = simulate_batch(theta, spec, [150] * 20, jax.random.key(1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = fit(spec, batch, CFG, seed=3)
    assert np.isfinite(res.loglik)
    np.testing.assert_allclose(np.asarray(res.params["emission.scale"]).ravel(), [0.5, 1.0], rtol=0.25)
    np.testing.assert_allclose(np.asarray(res.params["emission.mean"]).ravel(), [-1.0, 1.5], atol=0.3)
    table = res.diagnostics["per_sequence"]
    assert [r["T"] for r in table] == [150] * 20
    assert sum(r["loglik"] for r in table) == pytest.approx(res.loglik, rel=1e-12)
    pb = batch.padded()
    space = build_expanded(spec)
    total = float(jnp.sum(batched_loglik(ExpandedHMMEngine(), res.params, spec, space, pb)))
    assert total == pytest.approx(res.loglik, rel=1e-12)


def test_e12_t02_batch_fit_is_deterministic_per_seed(tmp_path: Path) -> None:
    """Same seed, same batch -> bitwise identical `uparams`."""
    spec = _spec(tmp_path)
    batch = simulate_batch(_truth(spec), spec, [100] * 6, jax.random.key(2))
    cfg = EstimationSection(restarts=1, max_iter=25, init=["random"], device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = fit(spec, batch, cfg, seed=4)
        b = fit(spec, batch, cfg, seed=4)
    assert np.array_equal(np.asarray(a.uparams), np.asarray(b.uparams))


def test_e12_t02_b1_batch_reproduces_the_single_sequence_fit_bitwise(tmp_path: Path) -> None:
    """AC3: a `B = 1` batch gives the single-sequence `uparams` exactly."""
    spec = _spec(tmp_path)
    batch = simulate_batch(_truth(spec), spec, [200], jax.random.key(3))
    cfg = EstimationSection(restarts=1, max_iter=30, init=["rolling_kmeans"], device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        single = fit(spec, batch.sequences[0], cfg, seed=7)
        via_batch = fit(spec, SequenceBatch.of(batch.sequences[0]), cfg, seed=7)
    assert np.array_equal(np.asarray(single.uparams), np.asarray(via_batch.uparams))
    assert single.loglik == via_batch.loglik


def test_e12_t02_per_group_initial_regimes_are_recovered(tmp_path: Path) -> None:
    """AC2: group `a` starts in regime 0, group `b` in regime 1; the fitted rows have the right argmax."""
    spec = _spec(tmp_path, 2, per_group=True)
    theta = _truth(spec)
    theta["initial.logits"] = jnp.asarray([[0.95, 0.05], [0.05, 0.95]])
    theta = _set(theta, [-2.0, 2.0], [0.5, 0.5])
    groups = ["a"] * 30 + ["b"] * 30
    batch = simulate_batch(theta, spec, [40] * 60, jax.random.key(8), groups=groups)
    assert batch.group_names == ("a", "b")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = fit(spec, batch, CFG, seed=1)
    rows = np.asarray(res.params["initial.logits"])
    assert rows.shape == (2, 2)
    assert res.diagnostics["initial_groups"] == ["a", "b"]
    means = np.asarray(res.params["emission.mean"]).ravel()
    low = int(np.argmin(means))
    assert int(np.argmax(rows[0])) == low
    assert int(np.argmax(rows[1])) == 1 - low


def test_e12_t02_score_gives_total_and_per_sequence_and_falls_back_for_unseen_groups(tmp_path: Path) -> None:
    """`Project.score`: per-sequence table sums to the total; an unseen group warns and uses the mean row."""
    cfg = parse_project_config(_raw(tmp_path, per_group=True))
    project = Project(cfg, out_dir=tmp_path / "out")
    spec = _spec(tmp_path, 2, per_group=True)
    theta = _truth(spec)
    theta["initial.logits"] = jnp.asarray([[0.9, 0.1], [0.1, 0.9]])
    batch = simulate_batch(theta, spec, [30] * 6, jax.random.key(4), groups=["a", "b"] * 3)
    cfgs = EstimationSection(restarts=1, max_iter=15, init=["random"], device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = fit(spec, batch, cfgs, seed=2)
    scored = project.score(res, batch)
    assert scored.per_sequence["loglik"].sum() == pytest.approx(scored.loglik)
    assert scored.loglik == pytest.approx(res.loglik, rel=1e-9)
    assert list(scored.per_sequence["T"]) == [30] * 6
    unseen = SequenceBatch(
        tuple(s.__class__(**{**s.__dict__, "group": "zzz"}) for s in batch.sequences[:2]), (), ("u0", "u1")
    )
    with pytest.warns(UserWarning, match="not seen"):
        fallback = project.score(res, unseen)
    assert np.isfinite(fallback.loglik)


def test_e12_t02_single_dataset_scores_as_a_one_sequence_batch(tmp_path: Path) -> None:
    """`Project.score(fit, PreparedDataset)` equals the fit's own log-likelihood."""
    cfg = parse_project_config(_raw(tmp_path))
    spec = _spec(tmp_path)
    ds = simulate_batch(_truth(spec), spec, [120], jax.random.key(9)).sequences[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = fit(spec, ds, EstimationSection(restarts=1, max_iter=20, init=["random"], device="cpu"), seed=1)
    out = Project(cfg, out_dir=tmp_path / "o").score(res, ds)
    assert out.loglik == pytest.approx(res.loglik, rel=1e-9)


def test_e12_t02_ltr_preset_data_loads_and_a_map_fit_runs(tmp_path: Path) -> None:
    """AC5: the ltr preset's 50-sequence data loads (`load_batch`); a MAP fit runs with the gated keys off."""
    import yaml

    raw = yaml.safe_load(get_preset("core/segmentation_gaussian_ltr").yaml)
    path = tmp_path / "data" / "sensor_windows.parquet"
    path.parent.mkdir(parents=True)
    frame = segmentation_gaussian_ltr(0)["data/sensor_windows.parquet"]
    frame.to_parquet(path)
    raw["data"]["source"]["path"] = str(path)
    raw["data"]["labels"] = None
    raw["data"]["sequences"]["bucket_edges"] = [256, 512]
    raw["project"]["output_dir"] = str(tmp_path / "runs")
    m = raw["model"]
    m.update({"d_max": 40, "left_censored": False, "engine": "expanded", "topology": "left_to_right"})
    m["emission"] = {
        "family": "gaussian",
        "covariance": "diag",
        "params": {
            "mean": {"mode": "free", "init": [0.0, 1.0, 0.0, -1.0]},
            "scale": {"mode": "free", "init": [0.8, 0.8, 0.8, 0.8]},
        },
    }
    m["duration"] = {
        "family": "geometric",
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 30.0}},
    }
    m["initial_state"] = "free"
    raw.pop("online", None)
    raw["output"] = {"write": ["report"]}
    raw["evaluation"] = {"metrics": ["log_score", "bic"]}
    raw["estimation"] = {"method": "map", "max_iter": 8, "restarts": 1, "init": ["rolling_kmeans"]}
    cfg = parse_project_config(raw)
    project = Project(cfg, out_dir=tmp_path / "out")
    batch = project.load_batch()
    assert len(batch.sequences) == 50
    assert batch.padded().observations.shape[0] == 64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = project.fit(batch)
    assert np.isfinite(res.loglik)
    assert len(res.diagnostics["per_sequence"]) == 50
    assert isinstance(pd.DataFrame(res.diagnostics["per_sequence"]), pd.DataFrame)
