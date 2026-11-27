"""E14-T02: grid model selection on the multi-sequence 2-regime Gaussian fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ruamel.yaml import YAML
from typer.testing import CliRunner

from hsmmkit.cli.app import app
from hsmmkit.project import Project
from hsmmkit.selection.grid import COLUMNS, make_cells, recommend
from tests.helpers import core_config as cc

_runner = CliRunner()


def _dump(raw: dict[str, Any], path: Path) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        YAML(typ="safe").dump(raw, fh)
    return path


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """20 sequences of 150 steps (two Gaussian regimes, three channels) and a diagonal-Gaussian MAP config."""
    root = tmp_path_factory.mktemp("grid")
    rng = np.random.default_rng(5)
    rows = []
    for seq in range(20):  # two regimes, three independent channels, mean sojourn 20 steps
        state = int(rng.integers(2))
        for t in range(150):
            mean = (-1.0, 1.0)[state]
            sd = (0.6, 1.2)[state]
            rows.append((f"s{seq}", t, *rng.normal(mean, sd, 3)))
            if rng.random() < 1 / 20:
                state = 1 - state
    frame = pd.DataFrame(rows, columns=["session_id", "t", "acc_x", "acc_y", "acc_z"])
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
    raw["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [20.0, 20.0]
    raw["estimation"].update(restarts=1, max_iter=60, init=["rolling_kmeans"])
    raw["forecast"]["n_sim"] = 20
    raw["evaluation"]["metrics"] = ["log_score", "bic"]
    return {"root": root, "raw": raw}


def _project(base: dict[str, Any], name: str, **evaluation: Any) -> Project:  # noqa: ANN401
    raw = {**base["raw"], "evaluation": {**base["raw"]["evaluation"], **evaluation}}
    return Project.from_yaml(_dump(raw, base["root"] / f"{name}.yaml"), offline=True)


def test_e14_t02_cells_resize_lists_and_invalid_cells_keep_their_reason(base: dict[str, Any]) -> None:
    """The product of the axes; `n_states` resizes the per-regime lists; an unknown family is skipped."""
    project = _project(base, "cells", grid={"n_states": [2, 3], "duration": ["geometric", "nope"]})
    cells = make_cells(project.config)
    assert [c.axes["n_states"] for c in cells] == [2, 2, 3, 3]
    ok = [c for c in cells if c.raw is not None]
    assert [c.axes["duration"] for c in cells if c.raw is None] == ["nope", "nope"]
    assert all(c.skipped_reason for c in cells if c.raw is None)
    three = next(c for c in ok if c.axes["n_states"] == 3)
    assert three.raw is not None
    assert three.raw["model"]["emission"]["params"]["mean"]["init"] == pytest.approx([-0.5, 0.0, 0.5])
    assert len(three.raw["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"]) == 3


def test_e14_t02_two_cell_grid_table_schema_and_outputs(base: dict[str, Any]) -> None:
    """AC2: the documented columns, a skipped cell with a reason, the files and the CLI."""
    _project(base, "two", grid={"n_states": [2, 3], "duration": ["geometric", "gamma"]})
    out = base["root"] / "out_two"
    result = _runner.invoke(app, ["--out", str(out), "compare", str(base["root"] / "two.yaml"), "--grid"])
    assert result.exit_code == 0, result.output
    table = pd.read_parquet(out / "grid.parquet")
    assert list(table.columns) == list(COLUMNS)
    done = table[table["skipped_reason"] == ""]
    skipped = table[table["skipped_reason"] != ""]
    assert len(done) == 2
    assert len(skipped) == 2  # gamma durations need the explicit engine
    assert set(done["n_states"]) == {2, 3}
    assert np.all(np.isfinite(done[["loglik", "aic", "bic", "min_occupancy"]].to_numpy()))
    assert (done["n_free"].to_numpy() > 0).all()
    assert table["recommended"].sum() == 1
    text = (out / "grid.md").read_text(encoding="utf-8")
    assert "## Skipped cells" in text
    assert "**" in text


def test_e14_t02_the_grid_is_deterministic_and_parallel_equals_sequential(base: dict[str, Any]) -> None:
    """AC3/AC4: same seed, same table; the process pool gives the same table (wall time aside)."""
    seq = _project(base, "seq", grid={"n_states": [2, 3]})
    par = _project(base, "par", grid={"n_states": [2, 3], "parallel": {"backend": "process", "workers": 2}})
    a, b, c = seq.compare_grid(), seq.compare_grid(), par.compare_grid()
    drop = ["wall_s"]
    pd.testing.assert_frame_equal(a.drop(columns=drop), b.drop(columns=drop))
    pd.testing.assert_frame_equal(a.drop(columns=drop), c.drop(columns=drop))


def test_e14_t02_recommendation_rule_breaks_bic_ties_by_cv_log_score() -> None:
    """Lowest BIC; within 1e-9 the higher cross-validated log score wins; skipped cells never win."""
    table = pd.DataFrame(
        {
            "cell": [0, 1, 2, 3],
            "bic": [10.0, 10.0, 9.0, 8.0],
            "cv_log_score": [-1.0, -0.5, -2.0, -3.0],
            "skipped_reason": ["", "", "", "boom"],
        }
    )
    assert recommend(table) == 2
    table.loc[2, "bic"] = 10.0
    assert recommend(table) == 1


@pytest.mark.slow
def test_e14_t02_the_two_regime_fixture_selects_two_states_by_bic_and_cv(base: dict[str, Any]) -> None:
    """AC1: N in {2, 3, 4} on the two-regime fixture: BIC and the CV log score both pick N = 2."""
    project = _project(
        base,
        "full",
        grid={"n_states": [2, 3, 4]},
        cv={"splitter": "sequence_kfold", "folds": 3, "seed": 7},
    )
    table = project.compare_grid()
    assert (table["skipped_reason"] == "").all()
    assert table.loc[table["bic"].idxmin(), "n_states"] == 2
    assert table.loc[table["cv_log_score"].idxmax(), "n_states"] == 2
    assert table.loc[table["recommended"], "n_states"].tolist() == [2]
