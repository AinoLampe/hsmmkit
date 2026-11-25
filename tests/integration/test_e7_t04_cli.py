"""E7-T04: `backtest` and `compare` on a synthetic csv project (price-only, 2-state GBM, three folds)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from click.testing import Result
from typer.testing import CliRunner

from regimekit.cli.app import app
from regimekit.evaluation.report import SECTIONS
from regimekit.persistence.results import read_backtest
from tests.integration.test_e2_t08_fetch import _write_csv_project

_runner = CliRunner()

METRICS = ["log_score", "crps", "pit_coverage", "bic", "icl", "regime_occupancy", "sharpe", "max_drawdown"]


def _model(sojourn: float) -> dict[str, object]:
    return {
        "n_states": 2,
        "engine": "expanded",
        "initial_state": "uniform",
        "emission": {
            "family": "gbm",
            "params": {
                "mu": {"mode": "free", "init": [0.05, -0.05]},
                "sigma": {
                    "mode": "free",
                    "init": [0.1, 0.3],
                    "constraint": "increasing",
                    "prior": {"type": "lognormal", "sd": 1.0},
                },
            },
            "covariates": {"drift": [], "log_vol": []},
        },
        "duration": {
            "family": "geometric",
            "params": {
                "rate": {
                    "mode": "free",
                    "init_mean_sojourn_days": [sojourn, sojourn],
                    "prior": {"type": "lognormal", "sd": 1.0},
                }
            },
            "covariates": [],
        },
        "transition": {"mode": "jump", "order": 1, "covariates": [], "coef_prior_sd": 1.0},
    }


def _project(root: Path, name: str, sojourn: float = 30.0) -> Path:
    """A csv project under `root/name`: 300 trading days, 3 folds, a cheap 2-state model."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    return _write_csv_project(
        folder,
        data__covariates=[],
        model=_model(sojourn),
        estimation__restarts=1,
        estimation__max_iter=150,
        estimation__init=["rolling_kmeans"],
        estimation__device="cpu",
        forecast__horizon=1,
        forecast__n_sim=50,
        forecast__covariate_forecaster={"type": "freeze"},
        evaluation__metrics=METRICS,
        evaluation__backtest__horizons=[1],
    )


def _invoke(*args: str) -> Result:
    return _runner.invoke(app, list(args))


@pytest.fixture(scope="module")
def backtest_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """`(config, output dir)` of one `backtest` run into the config's own `project.output_dir`."""
    root = tmp_path_factory.mktemp("e7t04")
    config = _project(root, "a")
    result = _invoke("backtest", str(config))
    assert result.exit_code == 0, result.output
    return config, root / "a" / "runs"


# --- AC1 / AC3 -----------------------------------------------------------------------------------


def test_e7_t04_backtest_writes_all_artifacts_with_the_documented_schema(
    backtest_run: tuple[Path, Path],
) -> None:
    """AC1: metrics, folds, strategy and report; the metrics schema is `fold, horizon, metric, value`."""
    _, out = backtest_run
    for name in ("backtest_metrics.parquet", "backtest_folds.parquet", "strategy.parquet", "report.md"):
        assert (out / name).is_file(), name
    schema = pq.read_schema(out / "backtest_metrics.parquet")
    assert [(f.name, str(f.type)) for f in schema] == [
        ("fold", "int64"), ("horizon", "int64"), ("metric", "string"), ("value", "double"),
    ]  # fmt: skip
    metrics = pq.read_table(out / "backtest_metrics.parquet").to_pandas()
    assert set(metrics["horizon"]) == {0, 1}
    assert {"log_score", "crps", "pit_coverage.coverage_90", "bic", "icl", "sharpe", "max_drawdown"} <= set(
        metrics["metric"]
    )
    assert metrics.groupby("fold").size().nunique() == 1  # the same rows in every fold
    assert metrics["value"].notna().all()

    folds = pq.read_table(out / "backtest_folds.parquet").to_pandas()
    assert len(folds) == 3
    for col in (
        "fold",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "fit_time",
        "converged",
        "objective",
    ):
        assert col in folds.columns, col
    strategy = pq.read_table(out / "strategy.parquet").to_pandas()
    assert {"time", "position", "return", "equity", "turnover"} <= set(strategy.columns)
    assert len(strategy) == int((folds["test_end"] - folds["test_start"]).sum())

    result, meta = read_backtest(out)
    assert "strategy" in result.extras
    assert meta["config"] == "dax_3state_gbm_nb"
    assert len(meta["snapshot_ids"]) == 1
    assert isinstance(meta["n_free"], int)
    assert meta["n_free"] > 0


def test_e7_t04_report_contains_every_section_and_the_run_header(backtest_run: tuple[Path, Path]) -> None:
    """AC3: each heading of the report is present, with snapshot ids, folds and the strategy table."""
    _, out = backtest_run
    text = (out / "report.md").read_text(encoding="utf-8")
    for section in SECTIONS:
        assert f"\n## {section}\n" in text, section
    meta = json.loads((out / "backtest.json").read_text(encoding="utf-8"))
    assert meta["snapshot_ids"][0] in text
    assert "3 folds" in text
    assert "regime_long_flat" in text
    assert "fold fits converged" in text


def test_e7_t04_backtest_is_deterministic(backtest_run: tuple[Path, Path], tmp_path: Path) -> None:
    """The same config and seed reproduce the metric table exactly (a second run into another directory)."""
    config, out = backtest_run
    again = tmp_path / "again"
    result = _invoke("--out", str(again), "--offline", "backtest", str(config))
    assert result.exit_code == 0, result.output
    first = pq.read_table(out / "backtest_metrics.parquet").to_pandas()
    second = pq.read_table(again / "backtest_metrics.parquet").to_pandas()
    pd.testing.assert_frame_equal(first, second, check_exact=True)


# --- AC2: compare --------------------------------------------------------------------------------


def test_e7_t04_compare_two_configs_on_one_snapshot_gives_a_two_row_table(
    backtest_run: tuple[Path, Path], tmp_path: Path
) -> None:
    """AC2: `compare A B --reuse` reuses A's stored backtest, runs B, and tabulates both."""
    config_a, _ = backtest_run
    config_b = _project(tmp_path, "b", sojourn=15.0)
    out = tmp_path / "cmp"
    result = _invoke(
        "--out", str(out), "compare", str(config_a), str(config_b), "--reuse", "--metric", "log_score"
    )
    assert result.exit_code == 0, result.output
    table = pq.read_table(out / "compare.parquet").to_pandas()
    assert table["config"].nunique() == 2
    assert len(table) == 2
    assert {"n_free", "fit_time", "log_score@h1", "log_score@h1_sd"} <= set(table.columns)
    assert "crps@h1" not in table.columns  # --metric restricts the columns
    text = (out / "compare.md").read_text(encoding="utf-8")
    assert "## Data snapshots" in text
    assert text.count("**") >= 2  # the best log score is marked
    assert (tmp_path / "b" / "runs" / "report.md").is_file()  # B was actually run


def test_e7_t04_compare_refuses_different_snapshots_unless_allowed(
    backtest_run: tuple[Path, Path], tmp_path: Path
) -> None:
    """AC2: different data snapshots -> exit 3 before any fit; `--allow-mixed` compares anyway."""
    config_a, out_a = backtest_run
    config_c = _project(tmp_path, "c")
    prices = tmp_path / "c" / "PRIMARY.csv"
    frame = pd.read_csv(prices)
    frame.loc[len(frame) - 1, ["close", "adj_close"]] *= 1.01  # one different print -> a new snapshot id
    frame.to_csv(prices, index=False)

    out = tmp_path / "cmp"
    refused = _invoke("--out", str(out), "compare", str(config_a), str(config_c))
    assert refused.exit_code == 3, refused.output
    assert "allow-mixed" in refused.output
    assert not (out / "compare.parquet").exists()
    assert not (tmp_path / "c" / "runs" / "backtest_metrics.parquet").exists()  # nothing was fitted

    # reuse path: C carries a stored (copied) result with its own snapshot id
    shutil.copytree(out_a, tmp_path / "c" / "runs", dirs_exist_ok=True)
    meta_path = tmp_path / "c" / "runs" / "backtest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["snapshot_ids"] = ["f" * 64]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert _invoke("--out", str(out), "compare", str(config_a), str(config_c), "--reuse").exit_code == 3
    mixed = _invoke("--out", str(out), "compare", str(config_a), str(config_c), "--reuse", "--allow-mixed")
    assert mixed.exit_code == 0, mixed.output
    assert len(pq.read_table(out / "compare.parquet").to_pandas()) == 2
