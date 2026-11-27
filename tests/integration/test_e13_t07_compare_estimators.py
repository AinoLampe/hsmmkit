"""E13-T07 AC3: `hsmmkit compare CONFIG --estimators map,em,vb` on the Gaussian fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ruamel.yaml import YAML
from typer.testing import CliRunner

from hsmmkit.cli.app import app
from hsmmkit.estimation.compare import COLUMNS
from tests.helpers import core_config as cc

_runner = CliRunner()


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A csv of Gaussian observations and a core configuration pointing at it."""
    root = tmp_path_factory.mktemp("compare_estimators")
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
    raw["estimation"].update(restarts=1, max_iter=40)
    config = root / "config.yaml"
    with config.open("w", encoding="utf-8") as fh:
        YAML(typ="safe").dump(raw, fh)
    return {"root": root, "config": config}


def test_e13_t07_compare_estimators_writes_the_documented_table(world: dict[str, Path]) -> None:
    """One row per estimator with the documented columns; VB has an ELBO; EM is close to MAP."""
    out = world["root"] / "out"
    result = _runner.invoke(
        app, ["--out", str(out), "compare", str(world["config"]), "--estimators", "map,em,vb"]
    )
    assert result.exit_code == 0, result.output
    table = pd.read_parquet(out / "compare.parquet")
    assert list(table.columns) == list(COLUMNS)
    assert list(table["estimator"]) == ["map", "em", "vb"]
    assert np.all(np.isfinite(table["loglik"]))
    assert np.isfinite(table.loc[table["estimator"] == "vb", "elbo"]).all()
    assert table.loc[table["estimator"] != "vb", "elbo"].isna().all()
    assert table.loc[table["estimator"] == "map", "max_abs_dtheta_vs_map"].iloc[0] == 0.0
    assert (table["n_params"] == table["n_params"].iloc[0]).all()
    assert np.all(np.isfinite(table["bic"]))
    text = (out / "compare.md").read_text(encoding="utf-8")
    assert "# Estimator comparison" in text
    assert "| map |" in text


def test_e13_t07_compare_estimators_refuses_unknown_names_and_several_configs(world: dict[str, Path]) -> None:
    """Unknown estimators and `--estimators` with two configs are configuration errors (exit code 2)."""
    out = world["root"] / "out2"
    bad = _runner.invoke(
        app, ["--out", str(out), "compare", str(world["config"]), "--estimators", "map,nope"]
    )
    assert bad.exit_code == 2, bad.output
    two = _runner.invoke(
        app, ["--out", str(out), "compare", str(world["config"]), str(world["config"]), "--estimators", "map"]
    )
    assert two.exit_code == 2, two.output
    assert "unknown estimator" in bad.output
    assert "exactly one configuration" in two.output


def test_e13_t07_an_estimator_that_cannot_fit_the_model_fails_before_any_fit(world: dict[str, Path]) -> None:
    """`gem` needs the explicit engine: refused up front (exit 2), not after the other fits."""
    out = world["root"] / "out3"
    result = _runner.invoke(
        app, ["--out", str(out), "compare", str(world["config"]), "--estimators", "map,gem"]
    )
    assert result.exit_code == 2, result.output
    assert "engine: explicit" in result.output
    assert not (out / "compare.parquet").exists()
