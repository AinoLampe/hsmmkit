"""E10-T07b: the core CLI (`hsmmkit`) end to end on a gaussian csv, without the finance package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from ruamel.yaml import YAML
from typer.testing import CliRunner

from hsmmkit.cli.app import app, build_app, exit_code_of
from hsmmkit.config.schema import ProjectConfig
from hsmmkit.project import Project
from hsmmkit.utils.errors import ConfigError, DataError, NumericalError, PersistenceError
from tests.helpers import core_config as cc

_runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A csv of 300 gaussian observations and a core configuration pointing at it."""
    root = tmp_path_factory.mktemp("corecli")
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
    config = root / "config.yaml"
    with config.open("w", encoding="utf-8") as fh:
        YAML(typ="safe").dump(raw, fh)
    return {"root": root, "config": config}


def test_e10_t07b_core_cli_runs_fit_decode_forecast_backtest_end_to_end(world: dict[str, Path]) -> None:
    """AC2: fit, decode, forecast and backtest on the gaussian csv, no finance package involved."""
    out = world["root"] / "out"
    config = str(world["config"])
    result = _runner.invoke(app, ["--out", str(out), "validate-config", config])
    assert result.exit_code == 0, result.output
    result = _runner.invoke(app, ["--out", str(out), "fit", config])
    assert result.exit_code == 0, result.output
    model = out / "core_demo.hsmm"
    assert (model / "manifest.json").is_file()
    assert (out / "prepared.parquet").is_file()
    assert (out / "provenance.json").is_file()
    for command, expected in (
        (["decode", config, "--model", str(model)], "posterior.parquet"),
        (["forecast", config, "--model", str(model)], "forecast.json"),
        (["backtest", config], "backtest_metrics.parquet"),
    ):
        result = _runner.invoke(app, ["--out", str(out), *command])
        assert result.exit_code == 0, (command, result.output)
        assert any(out.glob(expected.split(".")[0] + "*")), (command, sorted(p.name for p in out.iterdir()))
    assert (out / "report.md").is_file()


def test_e10_t07b_core_app_has_no_fetch_and_the_documented_commands() -> None:
    """AC2: `fetch` does not exist in the core app; the eleven core commands do."""
    commands = {c.name for c in app.registered_commands}
    assert commands == {
        "validate-config",
        "config-schema",
        "fit",
        "decode",
        "forecast",
        "simulate",
        "backtest",
        "cv",
        "bootstrap",
        "compare",
        "selftest",
        "stream",
    }
    result = _runner.invoke(app, ["fetch", "config.yaml"])
    assert result.exit_code == 2  # no such command (Click's usage error)


def test_e10_t07b_build_app_registers_extra_commands_and_carries_the_context() -> None:
    """`build_app` runs each extra command function on the app; the app carries its classes."""

    def add_hello(typer_app: object) -> None:
        @typer_app.command("hello")  # type: ignore[attr-defined]
        def hello() -> None:
            """Say hello."""
            print("hello from the extension")

    custom = build_app(name="demo", config_cls=ProjectConfig, project_cls=Project, extra_commands=[add_hello])
    assert custom.context.project_cls is Project
    assert "hello" in {c.name for c in custom.registered_commands}
    assert _runner.invoke(custom, ["hello"]).stdout.strip() == "hello from the extension"


def test_e10_t07b_exit_codes_use_the_most_specific_class() -> None:
    """Core codes: config 2, data 3, numerical 4, persistence 1, anything else 1; subclasses inherit."""

    class SubDataError(DataError):
        pass

    assert [exit_code_of(e) for e in (ConfigError("x"), DataError("x"), NumericalError("x"))] == [2, 3, 4]
    assert exit_code_of(PersistenceError("x")) == 1
    assert exit_code_of(SubDataError("x")) == 3
    assert exit_code_of(RuntimeError("x")) == 1


def test_e10_t07b_the_core_app_in_a_fresh_process_has_no_provider_exit_code() -> None:
    """`ProviderError -> 5` is finance-only: a fresh process importing the core CLI has no such code."""
    code = (
        "import hsmmkit.cli.app as a, sys\n"
        "print(sorted(set(a.EXIT_CODES.values())))\n"
        "print('hsmmkit.finance' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert out == ["[1, 2, 3, 4]", "False"]


def test_e10_t07b_time_less_data_loads_with_a_step_index_train_end(tmp_path: Path) -> None:
    """`_train_end` of time-less observations is a step index (`initial_train_steps - 1`, else the last)."""
    import numpy as np

    from hsmmkit.config.schema import BacktestCfg
    from hsmmkit.data.contracts import ObservationSet
    from hsmmkit.project import _train_end

    obs = ObservationSet(("x",), None, np.zeros((50, 1)), "memory", "s")
    cfg = cc.core_config(tmp_path).evaluation.backtest
    assert isinstance(cfg, BacktestCfg)
    assert _train_end(obs, cfg.model_copy(update={"initial_train_steps": 30})) == 29
    assert _train_end(obs, cfg.model_copy(update={"initial_train_steps": None, "initial_train": 1.0})) == 49
