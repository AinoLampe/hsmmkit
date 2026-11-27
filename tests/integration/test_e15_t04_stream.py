"""E15-T04: `hsmmkit stream` and `Project.online` (piped CSV -> JSONL), resume, exit codes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ruamel.yaml import YAML
from typer.testing import CliRunner

from hsmmkit.cli.app import app
from hsmmkit.config.schema import ProjectConfig, parse_project_config
from hsmmkit.inference.decode import decode
from hsmmkit.project import Project
from tests.helpers import core_config as cc

_runner = CliRunner()
N_ROWS = 300
CUT = 120


def _write_world(root: Path, *, covariate: bool) -> dict[str, Any]:
    values = cc.simulate_values(N_ROWS)
    times = pd.bdate_range("2020-01-01", periods=N_ROWS)
    frame = pd.DataFrame({"t": times.strftime("%Y-%m-%d"), "x": values})
    cov_cols: list[str] = []
    if covariate:
        rng = np.random.default_rng(5)
        frame["z"] = np.abs(values) * 50 + rng.normal(0, 0.1, N_ROWS)
        cov_cols = ["z"]
    frame.to_csv(root / "data.csv", index=False)
    raw = cc.core_raw(root)
    raw["data"]["source"] = {
        "kind": "csv",
        "path": str(root / "data.csv"),
        "observation_columns": ["x"],
        "time_column": "t",
        "covariate_columns": cov_cols,
    }
    if covariate:
        raw["data"]["covariates"] = [{"name": "z", "lag": 1, "use_in": ["duration"], "max_staleness": 5}]
        raw["model"]["duration"]["covariates"] = ["z"]
    raw["project"]["output_dir"] = str(root / "runs")
    raw["evaluation"]["backtest"]["initial_train_steps"] = 200
    raw["online"] = {"lag": 4, "alerts": [{"name": "turb", "regime": 1, "p_on": 0.8, "p_off": 0.5}]}
    config = root / "config.yaml"
    with config.open("w", encoding="utf-8") as fh:
        YAML(typ="safe").dump(raw, fh)
    out = root / "out"
    result = _runner.invoke(app, ["--out", str(out), "fit", str(config)])
    assert result.exit_code == 0, result.output
    return {
        "root": root,
        "config": config,
        "model": out / "core_demo.hsmm",
        "frame": frame,
        "out": out,
        "raw": raw,
    }


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A fitted two-regime Gaussian model on a csv (no covariates) and the csv to stream."""
    return _write_world(tmp_path_factory.mktemp("stream_plain"), covariate=False)


@pytest.fixture(scope="module")
def with_cov(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The same with one duration covariate `z`."""
    return _write_world(tmp_path_factory.mktemp("stream_cov"), covariate=True)


def _csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False)


def _stream(world: dict[str, Any], text: str, *extra: str) -> Any:  # noqa: ANN401
    args = ["stream", str(world["config"]), "--model", str(world["model"]), *extra]
    return _runner.invoke(app, ["--out", str(world["out"]), *args], input=text)


def _jsonl(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def _offline_filtered(world: dict[str, Any]) -> np.ndarray:
    project = Project(parse_project_config(world["raw"]), config_dir=world["root"], out_dir=world["out"])
    fit = Project.load(world["model"])
    data = project.load_data()
    return np.asarray(decode(fit, data).filtered)


def test_e15_t04_piped_csv_equals_push_many_and_the_offline_filter_bitwise(plain: dict[str, Any]) -> None:
    """AC1: JSONL filtered probabilities == `Project.online().push_many()` == offline `decode` filtered."""
    result = _stream(plain, _csv(plain["frame"]))
    assert result.exit_code == 0, result.output
    rows = _jsonl(result.stdout)
    assert len(rows) == N_ROWS
    assert [r["t"] for r in rows] == list(range(1, N_ROWS + 1))
    streamed = np.array([r["filtered"] for r in rows])
    assert rows[10]["smoothed_lag"] is not None
    assert (
        rows[1]["smoothed_lag"] == [None, None]
        or np.isnan(np.array(rows[1]["smoothed_lag"], dtype=float)).all()
    )

    config = parse_project_config(plain["raw"])
    project = Project(config, config_dir=plain["root"], out_dir=plain["out"])
    session = project.online(Project.load(plain["model"]))
    table = session.push_many(plain["frame"])
    pushed = table[[c for c in table.columns if c.startswith("filtered_")]].to_numpy()
    np.testing.assert_array_equal(streamed, pushed)
    np.testing.assert_array_equal(streamed, _offline_filtered(plain))
    # the events of the stream are the events of the session
    assert [e for r in rows for e in r["events"]] == [e.to_dict() for e in session.events]


def test_e15_t04_a_real_pipe_gives_the_same_lines(plain: dict[str, Any]) -> None:
    """The console entry point through a subprocess pipe prints the lines `CliRunner` captured."""
    text = _csv(plain["frame"].iloc[:60])
    done = subprocess.run(
        [sys.executable, "-m", "hsmmkit.cli.app", "--out", str(plain["out"]), "stream", str(plain["config"]),
         "--model", str(plain["model"])],
        input=text, capture_output=True, text=True, check=False, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert _jsonl(done.stdout) == _jsonl(_stream(plain, text).stdout)


def test_e15_t04_resume_from_a_state_directory_continues_exactly(
    plain: dict[str, Any], tmp_path: Path
) -> None:
    """AC2: stop after N rows with `--state`, restart on the rest: the remaining output is unchanged."""
    frame = plain["frame"]
    full = _jsonl(_stream(plain, _csv(frame)).stdout)
    state = tmp_path / "state"
    first = _stream(plain, _csv(frame.iloc[:CUT]), "--state", str(state))
    assert first.exit_code == 0, first.output
    assert (state / "state.json").exists()
    second = _stream(plain, _csv(frame.iloc[CUT:]), "--state", str(state))
    assert second.exit_code == 0, second.output
    assert _jsonl(first.stdout) + _jsonl(second.stdout) == full


def test_e15_t04_emit_every_and_events_always(plain: dict[str, Any]) -> None:
    """`--every k` thins the output, `--emit` selects fields; lines with events are always written."""
    text = _csv(plain["frame"])
    every = _jsonl(_stream(plain, text, "--every", "10", "--emit", "filtered").stdout)
    full = _jsonl(_stream(plain, text).stdout)
    event_rows = {r["t"] for r in full if r["events"]}
    assert {r["t"] for r in every} == {t for t in range(1, N_ROWS + 1) if t % 10 == 0} | event_rows
    assert all(r["smoothed_lag"] is None for r in every)
    assert all(r["filtered"] is not None for r in every)


def test_e15_t04_exit_codes(plain: dict[str, Any], tmp_path: Path) -> None:
    """AC3: missing model 2, missing column 3 (named), malformed row skipped or 3 with `--strict`."""
    config = str(plain["config"])
    missing = _runner.invoke(app, ["stream", config, "--model", str(tmp_path / "nope.hsmm")], input="t,x\n")
    assert missing.exit_code == 2
    no_col = _stream(plain, "t,y\n2020-01-01,0.1\n")
    assert no_col.exit_code == 3
    assert "'x'" in no_col.output
    text = _csv(plain["frame"].iloc[:30]).splitlines()
    bad = [*text[:10], "2020-02-03,not-a-number", *text[10:]]
    lenient = _stream(plain, "\n".join(bad) + "\n")
    assert lenient.exit_code == 0
    assert len(_jsonl(lenient.stdout)) == 30
    strict = _stream(plain, "\n".join(bad) + "\n", "--strict")
    assert strict.exit_code == 3
    assert _stream(plain, "t,x\n", "--emit", "bogus").exit_code == 2
    assert _stream(plain, "").exit_code == 3  # no header


def test_e15_t04_covariates_stream_equals_the_offline_pipeline(with_cov: dict[str, Any]) -> None:
    """Raw covariate columns are lagged and standardized online like the offline `Preparer` does (AC1)."""
    result = _stream(with_cov, _csv(with_cov["frame"]))
    assert result.exit_code == 0, result.output
    rows = _jsonl(result.stdout)
    offline = _offline_filtered(with_cov)
    assert len(rows) == offline.shape[0]
    np.testing.assert_allclose(np.array([r["filtered"] for r in rows]), offline, rtol=0, atol=1e-12)


def test_e15_t04_missing_covariate_column_exits_3_naming_it(with_cov: dict[str, Any]) -> None:
    """AC3: a missing covariate column is a data error naming the column."""
    frame = with_cov["frame"].drop(columns=["z"])
    result = _stream(with_cov, _csv(frame))
    assert result.exit_code == 3
    assert "'z'" in result.output


def test_e15_t04_project_online_is_usable_without_the_cli(plain: dict[str, Any]) -> None:
    """AC4: `Project.online` works as a plain library call, with forecast, save and load."""
    config: ProjectConfig = parse_project_config(plain["raw"])
    project = Project(config, config_dir=plain["root"], out_dir=plain["out"])
    fit = Project.load(plain["model"])
    session = project.online(fit)
    for _, row in plain["frame"].iloc[:50].iterrows():
        out = session.push(row)
    assert out["t"] == 50
    assert len(out["filtered"]) == 2
    assert len(out["smoothed_lag"]) == 2
    fc = session.forecast(3, n_sim=20)
    assert np.asarray(fc.regime_probs).shape == (3, 2)
    session.save(plain["root"] / "sess")
    other = project.online(fit)
    other.load(plain["root"] / "sess")
    assert other.push(plain["frame"].iloc[50]) == session.push(plain["frame"].iloc[50])
