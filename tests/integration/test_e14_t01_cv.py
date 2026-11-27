"""E14-T01: sequence K-fold on the segmentation fixture, preparation per fold, walk-forward CV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ruamel.yaml import YAML
from typer.testing import CliRunner

from hsmmkit.cli.app import app
from hsmmkit.data.prepare import Preparer
from hsmmkit.evaluation.backtest import run_backtest
from hsmmkit.model.build import required_own_lags, required_series_support
from hsmmkit.presets.fixtures import segmentation_gaussian_ltr
from hsmmkit.project import Project
from hsmmkit.selection.cv import cross_validate
from hsmmkit.selection.splitters import SequenceKFold, WalkForward
from tests.helpers import core_config as cc

_runner = CliRunner()


def _dump(raw: dict[str, Any], path: Path) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        YAML(typ="safe").dump(raw, fh)
    return path


@pytest.fixture(scope="module")
def sessions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The 50-sequence segmentation fixture (60 steps per session) and a diagonal-Gaussian MAP config."""
    root = tmp_path_factory.mktemp("cv_sessions")
    frame = segmentation_gaussian_ltr(0)["data/sensor_windows.parquet"]
    frame = frame.groupby("session_id", sort=False).head(60).reset_index(drop=True)
    frame.to_parquet(root / "sessions.parquet", index=False)
    raw = cc.core_raw(root)
    raw["project"].update(output_dir=str(root / "runs"), seed=7)
    raw["data"] = {
        "source": {
            "kind": "parquet",
            "path": str(root / "sessions.parquet"),
            "observation_columns": ["acc_x", "acc_y", "acc_z"],
            "sequence_column": "session_id",
        },
        "clock": "steps",
        "preparation": {"standardize_observations": True},
    }
    raw["model"]["emission"] = {
        "family": "gaussian",
        "scale_by_dt": False,
        "ito_correction": False,
        "params": {
            "mean": {"mode": "free", "init": [-0.5, 0.5]},
            "scale": {"mode": "free", "init": [1.0, 1.0]},
        },
    }
    raw["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [15.0, 15.0]
    raw["estimation"].update(restarts=1, max_iter=25, init=["rolling_kmeans"])
    raw["forecast"]["n_sim"] = 40
    raw["evaluation"]["metrics"] = ["log_score", "bic"]
    raw["evaluation"]["cv"] = {"splitter": "sequence_kfold", "folds": 5, "seed": 7}
    return {"root": root, "config": _dump(raw, root / "config.yaml")}


def test_e14_t01_cv_cli_writes_the_documented_artifacts_on_the_50_sequence_fixture(
    sessions: dict[str, Path],
) -> None:
    """AC4: `hsmmkit cv` -> scores, summary, folds and a cross-validation section of `report.md`."""
    out = sessions["root"] / "out"
    result = _runner.invoke(app, ["--out", str(out), "cv", str(sessions["config"])])
    assert result.exit_code == 0, result.output
    scores = pd.read_parquet(out / "cv_scores.parquet")
    summary = pd.read_parquet(out / "cv_summary.parquet")
    folds = pd.read_parquet(out / "cv_folds.parquet")
    assert list(scores.columns) == ["fold", "sequence", "horizon", "metric", "value"]
    assert set(scores["fold"]) == {0, 1, 2, 3, 4}
    assert scores["sequence"].nunique() == 50  # every sequence is scored exactly once per metric
    assert scores.groupby(["sequence", "metric"]).size().eq(1).all()
    assert {"log_score", "crps", "bic"} <= set(summary["metric"])
    assert np.all(np.isfinite(scores["value"]))
    assert list(folds["n_test"]) == [10] * 5
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Cross-validation" in report
    assert "log_score" in report


def test_e14_t01_the_preparation_is_fitted_on_the_training_sequences_of_each_fold(
    sessions: dict[str, Path],
) -> None:
    """AC3 (leakage): the standardization statistics come from the training rows only.

    The test sequences are transformed with them; fitting on all rows would give different statistics.
    """
    project = Project.from_yaml(sessions["config"], offline=True)
    cfg = project.config
    fetched = project.load_inputs()
    obs = fetched.obs

    def factory() -> Preparer:
        return Preparer(
            cfg.data,
            own_lags=required_own_lags(cfg),
            series_support=required_series_support(cfg),
            n_regimes=cfg.model.n_states,
        )

    splitter = SequenceKFold(5, seed=7)
    assign = splitter.assignment(obs)
    ids = np.asarray([str(s) for s in np.asarray(obs.sequence_ids)])
    all_mean = np.asarray(obs.values).mean(axis=0)
    for split in splitter.split((obs, fetched.covs), cfg.evaluation, factory):
        train_rows = np.asarray([assign[s] != split.fold for s in ids])
        expected = np.asarray(obs.values)[train_rows].mean(axis=0)
        stats = split.train.sequences[0].prep_stats
        np.testing.assert_allclose(np.asarray(stats.obs_mean), expected, rtol=1e-12)
        assert not np.allclose(np.asarray(stats.obs_mean), all_mean, rtol=1e-12, atol=0.0)
        assert split.test.sequences[0].prep_stats.obs_mean is not None
        np.testing.assert_allclose(
            np.asarray(split.test.sequences[0].prep_stats.obs_mean), np.asarray(stats.obs_mean), rtol=0.0
        )
        assert len(split.test.sequences) == 10
        assert len(split.train.sequences) == 40


@pytest.fixture(scope="module")
def series(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A single 300-step Gaussian series and a backtest configuration (the E10-T07b world)."""
    root = tmp_path_factory.mktemp("cv_series")
    values = cc.simulate_values()
    times = pd.bdate_range("2020-01-01", periods=values.shape[0])
    pd.DataFrame({"t": times.strftime("%Y-%m-%d"), "x": values}).to_csv(root / "data.csv", index=False)
    raw = cc.core_raw(root)
    raw["data"]["source"] = {
        "kind": "csv",
        "path": str(root / "data.csv"),
        "observation_columns": ["x"],
        "time_column": "t",
    }
    raw["project"]["output_dir"] = str(root / "runs")
    raw["evaluation"]["backtest"]["initial_train_steps"] = 200
    raw["forecast"].update(horizon=3, n_sim=40)
    raw["evaluation"]["cv"] = {"splitter": "walk_forward"}
    return {"root": root, "config": _dump(raw, root / "config.yaml")}


def test_e14_t01_walk_forward_cross_validation_is_bitwise_the_backtest(series: dict[str, Any]) -> None:
    """AC1: `cross_validate(WalkForward)` equals `run_backtest` (folds, metrics, summary; fit times aside)."""
    project = Project.from_yaml(series["config"], offline=True)
    cfg = project.config
    fetched = project.load_inputs()
    raw = (fetched.obs, fetched.covs)

    def factory() -> Preparer:
        return Preparer(
            cfg.data,
            own_lags=required_own_lags(cfg),
            series_support=required_series_support(cfg),
            n_regimes=cfg.model.n_states,
        )

    result = cross_validate(cfg, raw, WalkForward(), cfg.project.seed, factory)
    direct = run_backtest(cfg, lambda: raw, cfg.project.seed)
    drop = ["fit_time"]
    pd.testing.assert_frame_equal(result.folds.drop(columns=drop), direct.folds.drop(columns=drop))
    pd.testing.assert_frame_equal(result.summary, direct.summary)
    assert result.scores["value"].to_numpy().tolist() == direct.metrics["value"].to_numpy().tolist()
    # the splitter exposes the same folds and their origins
    splits = list(WalkForward().split(raw, cfg.evaluation, factory))
    assert len(splits) == len(direct.folds)
    for split, (_, row) in zip(splits, direct.folds.iterrows(), strict=True):
        assert len(split.train.sequences[0].times) == int(row["train_end"] - row["train_start"])
        assert split.origins is not None
        assert len(split.origins) == int(row["test_end"] - row["test_start"])
        assert split.origins[0] == 0
