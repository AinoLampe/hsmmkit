"""E7-T04 building blocks: backtest writers, comparison table, snapshot check and report sections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from regimekit.evaluation.compare import (
    CompareEntry,
    best_configs,
    check_snapshots,
    comparison_table,
    render_markdown,
)
from regimekit.evaluation.report import SECTIONS, build_report
from regimekit.evaluation.results import BacktestResult
from regimekit.persistence.results import read_backtest, write_backtest
from regimekit.utils.errors import ConfigError, DataError
from tests.helpers import backtest_fixtures as bf


def _result(
    log_score: float, crps: float, bic: float, fit_time: float, strategy: bool = True
) -> BacktestResult:
    summary = pd.DataFrame(
        {
            "horizon": [0, 0, 1, 1, 1, 1, 1],
            "metric": [
                "bic",
                "regime_occupancy.0",
                "crps",
                "log_score",
                "pit_ks",
                "pit_coverage.pit",
                "sharpe",
            ],
            "mean": [bic, 0.5, crps, log_score, 0.05, 0.5, 0.3],
            "sd": [10.0, 0.1, 0.001, 0.2, np.nan, 0.05, 0.1],
            "n": [3, 3, 3, 3, 120, 3, 3],
        }
    )
    metrics = pd.DataFrame(
        {
            "fold": [0, 1],
            "horizon": [1, 1],
            "metric": ["log_score", "log_score"],
            "value": [log_score, log_score],
        }
    )
    folds = pd.DataFrame(
        {
            "fold": [0, 1],
            "train_start": [0, 0],
            "train_end": [100, 130],
            "test_start": [100, 130],
            "test_end": [130, 160],
            "train_end_time": pd.to_datetime(["2020-01-01", "2020-02-01"]).astype("datetime64[ns]"),
            "fit_time": [fit_time / 2, fit_time / 2],
            "objective": [-1.0, -2.0],
            "loglik": [1.0, 2.0],
            "converged": [True, False],
            "n_iter": [10, 20],
            "restart_gap": [0.0, 0.5],
        }
    )
    trace = None
    if strategy:
        trace = pd.DataFrame(
            {
                "fold": [0, 0, 1],
                "time": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-02"]).astype("datetime64[ns]"),
                "position": [1.0, 0.0, 1.0],
                "return": [0.01, 0.0, -0.01],
                "gross": [0.01, 0.0, -0.01],
                "equity": np.exp(np.cumsum([0.01, 0.0, -0.01])),
                "turnover": [1.0, 1.0, 1.0],
                "dt": [1.0, 1.0, 1.0],
            }
        )
    return BacktestResult(folds=folds, metrics=metrics, strategy=trace, summary=summary)


def test_e7_t04_backtest_files_round_trip_and_the_stale_strategy_file_is_removed(tmp_path: Path) -> None:
    """Every table comes back equal; no strategy -> no `strategy.parquet` (an older one is deleted)."""
    result = _result(-1.0, 0.01, -500.0, 10.0)
    write_backtest(result, tmp_path, {"snapshot_ids": ["a", "b"], "n_free": 7})
    back, meta = read_backtest(tmp_path)
    assert meta == {"snapshot_ids": ["a", "b"], "n_free": 7}
    pd.testing.assert_frame_equal(back.metrics, result.metrics, check_dtype=False)
    pd.testing.assert_frame_equal(back.summary, result.summary, check_dtype=False)
    pd.testing.assert_frame_equal(back.folds, result.folds, check_dtype=False)
    assert result.strategy is not None
    assert back.strategy is not None
    pd.testing.assert_frame_equal(back.strategy, result.strategy, check_dtype=False)
    schema = pq.read_schema(tmp_path / "backtest_metrics.parquet")
    assert [(f.name, str(f.type)) for f in schema] == [
        ("fold", "int64"), ("horizon", "int64"), ("metric", "string"), ("value", "double"),
    ]  # fmt: skip
    write_backtest(_result(-1.0, 0.01, -500.0, 10.0, strategy=False), tmp_path, {"snapshot_ids": []})
    assert not (tmp_path / "strategy.parquet").exists()
    assert read_backtest(tmp_path)[0].strategy is None


def test_e7_t04_comparison_table_has_one_row_per_config_and_marks_the_best() -> None:
    """Columns `config, n_free, fit_time`, ranking metrics with sd; best value per ranked column."""
    a = CompareEntry("a", _result(-1.0, 0.020, -500.0, 10.0), {"n_free": 7})
    b = CompareEntry("b", _result(-0.8, 0.025, -450.0, 30.0), {"n_free": 12})
    table = comparison_table([a, b])
    assert list(table["config"]) == ["a", "b"]
    assert list(table["n_free"]) == [7, 12]
    assert list(table["fit_time"]) == [10.0, 30.0]
    assert {"bic", "crps@h1", "crps@h1_sd", "log_score@h1", "sharpe@h1"} <= set(table.columns)
    assert not any(c.startswith(("pit_ks", "regime_occupancy", "pit_coverage.pit")) for c in table.columns)
    best = best_configs(table)
    assert best["log_score@h1"] == "b"  # higher is better
    assert best["crps@h1"] == "a"  # lower is better
    assert best["bic"] == "a"
    assert best["fit_time"] == "a"
    assert "n_free" not in best
    md = render_markdown(table, {"a": ["s" * 64], "b": ["s" * 64]})
    assert "| a |" in md
    assert "**0.02" in md  # a's crps is bold
    assert comparison_table([a]).shape[0] == 1
    assert best_configs(comparison_table([a])) == {}


def test_e7_t04_metric_filter_keeps_only_the_requested_metrics_and_rejects_unknown_ones() -> None:
    """Metric filter keeps only the requested metrics and rejects unknown ones."""
    a = CompareEntry("a", _result(-1.0, 0.020, -500.0, 10.0), {"n_free": 7})
    table = comparison_table([a], ["log_score"])
    assert [c for c in table.columns if c not in ("config", "n_free", "fit_time")] == [
        "bic",
        "bic_sd",
        "log_score@h1",
        "log_score@h1_sd",
    ]  # the in-sample criteria always stay
    with pytest.raises(ConfigError, match="nope"):
        comparison_table([a], ["nope"])
    with pytest.raises(ConfigError):
        comparison_table([])


def test_e7_t04_different_snapshots_are_refused_unless_mixing_is_allowed() -> None:
    """Different snapshots are refused unless mixing is allowed."""
    same = {"a": ["x", "y"], "b": ["x", "y"]}
    check_snapshots(same, allow_mixed=False)
    differ = {"a": ["x", "y"], "b": ["x", "z"]}
    with pytest.raises(DataError, match="allow-mixed") as err:
        check_snapshots(differ, allow_mixed=False)
    assert err.value.path == "data.cache.snapshot"
    check_snapshots(differ, allow_mixed=True)


def test_e7_t04_report_has_every_section_and_lists_diagnostic_warnings(tmp_path: Path) -> None:
    """`build_report` renders each heading, the fold fit table, the strategy table and diagnostic warnings."""
    cfg = bf.make_config(tmp_path)
    meta = {"seed": 11, "snapshot_ids": ["abc123"], "n_free": 9, "versions": {"jax": "x"}, "wall_s": 12.0}
    diagnostics = {
        "regime_occupancy": {"value": [0.6, 0.4], "warning": None},
        "restart_dispersion": {"value": 0.3, "warning": "restarts disagree"},
    }
    text = build_report("demo", cfg, _result(-1.0, 0.01, -500.0, 10.0), meta, diagnostics)
    for section in SECTIONS:
        assert f"\n## {section}\n" in text, section
    assert "# Backtest report: demo" in text
    for needle in (
        "`abc123`",
        "1 of 2 fold fits converged",
        "**warning:** restarts disagree",
        "- restart_dispersion: restarts disagree",
        "total log-return",
        "sharpe",
    ):
        assert needle in text, needle
    no_strategy = build_report("demo", cfg, _result(-1.0, 0.01, -500.0, 10.0, strategy=False), meta, None)
    assert "No strategy was configured." in no_strategy
    assert "No diagnostics were computed." in no_strategy
    assert text == build_report(
        "demo", cfg, _result(-1.0, 0.01, -500.0, 10.0), meta, diagnostics
    )  # deterministic


def test_e7_t04_metric_filter_keeps_the_in_sample_criteria() -> None:
    """`--metric log_score` still yields `bic`, `n_free` and `fit_time` columns."""
    a = CompareEntry("a", _result(-1.0, 0.020, -500.0, 10.0), {"n_free": 7})
    table = comparison_table([a], ["log_score"])
    assert {"bic", "n_free", "fit_time", "log_score@h1"} <= set(table.columns)
    assert "crps@h1" not in table.columns


def test_e7_t04_stored_results_without_snapshot_ids_are_a_persistence_error(tmp_path: Path) -> None:
    """A `backtest.json` lacking `snapshot_ids` cannot be reused."""
    from regimekit.utils.errors import PersistenceError

    write_backtest(_result(-1.0, 0.01, -500.0, 10.0), tmp_path, {"n_free": 1})
    with pytest.raises(PersistenceError, match="snapshot_ids"):
        read_backtest(tmp_path)
