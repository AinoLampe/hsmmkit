"""E12-T06 integration: `init: labels` recovers the emissions and sojourns without optimization."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import pytest

from hsmmkit.config.schema import EstimationSection, parse_project_config
from hsmmkit.data.prepare import Preparer
from hsmmkit.data.sources import load_data
from hsmmkit.estimation import fit
from hsmmkit.estimation.init import labels_init
from hsmmkit.model.build import build_model_spec
from tests.helpers.core_config import core_raw

MEANS = np.array([-3.0, 1.0, 4.0])
SDS = np.array([0.5, 1.0, 1.5])
SOJOURN = np.array([10.0, 25.0, 40.0])


def _simulate(t_len: int, seed: int, frac: float = 0.3, block: int = 400) -> pd.DataFrame:
    """A 3-regime Gaussian series with geometric sojourns; `frac` of the steps labelled in blocks."""
    rng = np.random.default_rng(seed)
    regime = np.empty(t_len, dtype=np.int64)
    t, k = 0, 0
    while t < t_len:
        d = int(rng.geometric(1.0 / SOJOURN[k]))
        regime[t : t + d] = k
        t += d
        k = int(rng.choice([j for j in range(3) if j != k]))
    y = rng.normal(MEANS[regime], SDS[regime])
    labelled = np.zeros(t_len, dtype=bool)
    n_blocks = int(frac * t_len / block)
    for start in rng.choice(t_len // block, size=n_blocks, replace=False) * block:
        labelled[start : start + block] = True
    return pd.DataFrame({"x": y, "reg": np.where(labelled, regime, np.nan)})


def _cfg(tmp_path: Path, init: list[str]) -> Any:  # noqa: ANN401
    raw = core_raw(tmp_path)
    raw["data"] = {
        "source": {"kind": "pandas", "observation_columns": ["x"]},
        "clock": "steps",
        "labels": {"source": "column", "column": "reg"},
    }
    raw["model"]["n_states"] = 3
    raw["model"]["emission"]["params"] = {
        "mean": {"mode": "free", "init": [0.0, 0.0, 0.0]},
        "scale": {"mode": "free", "init": [1.0, 1.0, 1.0], "constraint": "increasing"},
    }
    raw["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [20.0, 20.0, 20.0]
    raw["estimation"]["init"] = init
    return parse_project_config(raw)


def _dataset(cfg: Any, frame: pd.DataFrame) -> Any:  # noqa: ANN401
    obs, covs = load_data(cfg.data, obj=frame)
    return Preparer(cfg.data, n_regimes=3).fit(obs, covs, 10**9).transform(obs, covs)


def test_e12_t06_labels_init_recovers_means_scales_and_sojourns_within_15_percent(tmp_path: Path) -> None:
    """AC4: 30 % labelled steps, no optimization: means, scales and mean sojourns within 15 %."""
    cfg = _cfg(tmp_path, ["labels"])
    data = _dataset(cfg, _simulate(60_000, seed=0))
    spec = build_model_spec(cfg, data)
    theta = labels_init(data, spec, jax.random.key(0))
    mean = np.asarray(theta["emission.mean"]).ravel()
    scale = np.asarray(theta["emission.scale"]).ravel()
    np.testing.assert_allclose(mean, MEANS, rtol=0.15)
    np.testing.assert_allclose(scale, SDS, rtol=0.15)
    sojourn = np.asarray(spec.duration.mean_sojourn(theta, 1.0))
    np.testing.assert_allclose(sojourn, SOJOURN, rtol=0.15)
    pi = np.asarray(theta["transition.pi_rows"])
    np.testing.assert_allclose(pi.sum(axis=1), 1.0)
    assert pi.min() > 0.0


def test_e12_t06_an_unlabelled_regime_falls_back_to_random_init_with_a_warning(tmp_path: Path) -> None:
    """A regime without a single labelled step takes its parameters from the random init (warning)."""
    cfg = _cfg(tmp_path, ["labels"])
    frame = _simulate(20_000, seed=1)
    frame.loc[frame["reg"] == 2, "reg"] = np.nan
    data = _dataset(cfg, frame)
    spec = build_model_spec(cfg, data)
    with pytest.warns(UserWarning, match=r"regime\(s\) \[2\] have no labelled step"):
        theta = labels_init(data, spec, jax.random.key(0))
    mean = np.asarray(theta["emission.mean"]).ravel()
    np.testing.assert_allclose(mean[:2], MEANS[:2], rtol=0.15)
    assert np.isfinite(mean[2])


def test_e12_t06_a_map_fit_from_label_init_uses_the_labels_and_recovers(tmp_path: Path) -> None:
    """`fit` with `init: [labels]` runs, honours the labels in the likelihood and recovers the scales."""
    cfg = _cfg(tmp_path, ["labels"])
    data = _dataset(cfg, _simulate(6_000, seed=2, block=300))
    spec = build_model_spec(cfg, data)
    est = EstimationSection(restarts=1, max_iter=60, init=["labels"], device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = fit(spec, data, est, seed=3)
    assert np.isfinite(res.loglik)
    np.testing.assert_allclose(np.asarray(res.params["emission.scale"]).ravel(), SDS, rtol=0.15)


def test_e12_t06_a_batch_map_fit_with_labels_and_boundaries_runs_and_honours_them(tmp_path: Path) -> None:
    """MAP on a `SequenceBatch` with labels and boundaries: finite fit, Viterbi paths obey them."""
    from hsmmkit.engine.batch import batched_viterbi
    from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
    from hsmmkit.statespace.builders import build_expanded

    parts = []
    for sid in range(6):
        frame = _simulate(300, seed=10 + sid, frac=0.5, block=50)
        frame["sid"] = sid
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    cfg = _cfg(tmp_path, ["labels"])
    raw = cfg.model_dump(mode="json")
    raw["data"]["source"]["sequence_column"] = "sid"
    raw["data"]["labels"]["boundaries"] = {"must_switch_at": [], "must_not_switch_at": [5]}
    cfg = parse_project_config(raw)
    obs, covs = load_data(cfg.data, obj=frame)
    batch = Preparer(cfg.data, n_regimes=3).fit(obs, covs, 10**9).transform_batch(obs, covs)
    spec = build_model_spec(cfg, batch)
    est = EstimationSection(restarts=1, max_iter=40, init=["labels"], device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = fit(spec, batch, est, seed=1)
    assert np.isfinite(res.loglik)
    paths = np.asarray(
        batched_viterbi(ExpandedHMMEngine(), res.params, spec, build_expanded(spec), batch.padded())
    )
    regime_of = np.asarray(build_expanded(spec).index.regime_of)
    for b, seq in enumerate(batch.sequences):
        regimes = regime_of[paths[b, :300]]
        allowed = np.asarray(seq.labels)
        assert np.all(allowed[np.arange(300), regimes])
        assert regimes[5] == regimes[4]


def test_e12_t06_a_fit_with_labels_does_not_relabel_the_regimes_by_scale(tmp_path: Path) -> None:
    """Without an identifiability guard the regimes keep the numbering of the labels (warning, no sort)."""
    raw = _cfg(tmp_path, ["labels"]).model_dump(mode="json")
    raw["model"]["emission"]["params"]["scale"] = {"mode": "free", "init": [1.0, 1.0, 1.0]}
    cfg = parse_project_config(raw)
    frame = _simulate(6_000, seed=5, block=300)
    data = _dataset(cfg, frame)
    spec = build_model_spec(cfg, data)
    est = EstimationSection(restarts=1, max_iter=30, init=["labels"], device="cpu")
    with pytest.warns(UserWarning, match="not relabelled by scale"):
        res = fit(spec, data, est, seed=2)
    assert np.isfinite(res.loglik)
