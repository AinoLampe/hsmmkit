"""E8-T02: the `Project` facade and the CLI produce the same artifacts, with the same exit codes.

One synthetic csv project (price-only, 2-state GBM, three backtest folds). Every command is run through
the CLI into one directory and through `Project(..., write=True)` into another; the written parquet
frames are compared exactly (`fit_time` columns are wall-clock timings and are left out).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from click.testing import Result
from typer.testing import CliRunner

from regimekit import Project
from regimekit.cli import app as app_module
from regimekit.cli.app import _selftest_lines, app
from regimekit.config.yaml_io import load_config
from regimekit.estimation import selftest as selftest_mod
from regimekit.finance import FinanceConfig
from regimekit.persistence.store import ModelStore
from regimekit.utils.errors import ConfigError, ProviderError
from tests.integration.test_e7_t04_cli import _project

_runner = CliRunner()
SEED = 7
GLOBALS = ["--seed", str(SEED), "--device", "cpu", "--log-level", "WARNING"]
OVERRIDES = {"project.seed": SEED, "estimation.device": "cpu", "project.log_level": "WARNING"}


def _cli(out: Path, *args: str, offline: bool = False) -> Result:
    extra = ["--offline"] if offline else []
    return _runner.invoke(app, ["--out", str(out), *GLOBALS, *extra, *args])


def _same_frames(cli_dir: Path, py_dir: Path, name: str) -> None:
    """The parquet `name` written by the CLI and by `Project` are exactly equal (timings excluded)."""
    frames = []
    for directory in (cli_dir, py_dir):
        frame = pd.read_parquet(directory / name)
        frames.append(frame.drop(columns=[c for c in frame.columns if "fit_time" in str(c)]))
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


def _same_json(cli_dir: Path, py_dir: Path, name: str) -> None:
    left = json.loads((cli_dir / name).read_text(encoding="utf-8"))
    right = json.loads((py_dir / name).read_text(encoding="utf-8"))
    assert left == right


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The CLI `fit` and `Project.fetch/fit` runs every other test builds on."""
    root = tmp_path_factory.mktemp("e8t02")
    config = _project(root, "a")
    cli_dir, py_dir = root / "cli", root / "py"
    result = _cli(cli_dir, "fit", str(config))
    assert result.exit_code == 0, result.output
    project = Project.from_yaml(config, OVERRIDES, out_dir=py_dir)
    data = project.fetch(write=True)
    fit = project.fit(data, write=True)
    return SimpleNamespace(
        root=root, config=config, cli=cli_dir, py=py_dir, project=project, data=data, fit=fit
    )


# --- AC1: parity, command by command -------------------------------------------------------------


def test_e8_t02_validate_config_parity(tmp_path: Path) -> None:
    """`validate-config`: same summary lines on stdout and an identical `config.resolved.yaml`."""
    config = _project(tmp_path, "v")
    cli_dir, py_dir = tmp_path / "cli", tmp_path / "py"
    result = _cli(cli_dir, "validate-config", str(config))
    assert result.exit_code == 0, result.output
    lines = Project.from_yaml(config, OVERRIDES, out_dir=py_dir).validate(write=True)
    assert result.output.splitlines() == lines
    assert (cli_dir / "config.resolved.yaml").read_text() == (py_dir / "config.resolved.yaml").read_text()


def test_e8_t02_fetch_parity(world: SimpleNamespace) -> None:
    """`fetch`: identical `prepared.parquet`, `prep_stats.json`, `snapshots.json`, quality report."""
    _same_frames(world.cli, world.py, "prepared.parquet")
    for name in ("prep_stats.json", "snapshots.json"):
        _same_json(world.cli, world.py, name)
    assert world.project.last_quality is not None
    _same_json(world.cli, world.py, "quality.json")
    assert world.project.last_fetched is not None


def test_e8_t02_fit_parity_is_bitwise(world: SimpleNamespace) -> None:
    """`fit`: the bundle the CLI wrote and the `FitResult` of `Project.fit` agree bit for bit."""
    name = world.project.config.project.name
    from_cli = ModelStore.load(world.cli / f"{name}.rkmodel", FinanceConfig)
    from_py = ModelStore.load(world.py / f"{name}.rkmodel", FinanceConfig)
    assert np.array_equal(np.asarray(from_cli.uparams), np.asarray(world.fit.uparams))
    assert np.array_equal(np.asarray(from_cli.uparams), np.asarray(from_py.uparams))
    assert from_cli.loglik == world.fit.loglik == from_py.loglik
    for out in (world.cli, world.py):
        report = json.loads((out / "fit_report.json").read_text(encoding="utf-8"))
        assert report["seed"] == SEED
        assert report["loglik"] == world.fit.loglik
        assert "regime_occupancy" in report["diagnostics"]
        assert (out / "config.resolved.yaml").is_file()


def test_e8_t02_save_load_round_trip(world: SimpleNamespace, tmp_path: Path) -> None:
    """`Project.save` / `Project.load`: the bundle round trip is bitwise (E8-T01 contract)."""
    path = world.project.save(world.fit, tmp_path / "model.rkmodel")
    loaded = Project.load(path)
    assert np.array_equal(np.asarray(loaded.uparams), np.asarray(world.fit.uparams))
    assert loaded.loglik == world.fit.loglik


def test_e8_t02_decode_parity(world: SimpleNamespace) -> None:
    """`decode`: identical `posterior.parquet`, `segments.json`, `criteria.json`."""
    model = world.cli / f"{world.project.config.project.name}.rkmodel"
    result = _cli(world.cli, "decode", str(world.config), "--model", str(model))
    assert result.exit_code == 0, result.output
    posterior = world.project.decode(world.fit, world.data, write=True)
    _same_frames(world.cli, world.py, "posterior.parquet")
    for name in ("segments.json", "criteria.json"):
        _same_json(world.cli, world.py, name)
    assert [
        (s["start"], s["end"], s["regime"]) for s in json.loads((world.py / "segments.json").read_text())
    ] == (posterior.segments)


def test_e8_t02_forecast_parity_is_bitwise(world: SimpleNamespace) -> None:
    """`forecast`: with the same seed the written forecast tables are exactly equal."""
    model = world.cli / f"{world.project.config.project.name}.rkmodel"
    result = _cli(
        world.cli, "forecast", str(world.config), "--model", str(model), "--horizon", "5", "--samples"
    )
    assert result.exit_code == 0, result.output
    world.project.forecast(world.fit, world.data, horizon=5, samples=True, write=True)
    _same_frames(world.cli, world.py, "forecast.parquet")
    _same_frames(world.cli, world.py, "forecast_samples.parquet")
    _same_json(world.cli, world.py, "forecast.json")


def test_e8_t02_simulate_parity(world: SimpleNamespace) -> None:
    """`simulate`: identical `simulation.parquet` and segments; the returned arrays match the file."""
    result = _cli(world.cli, "simulate", str(world.config), "--steps", "60", "--n-paths", "2")
    assert result.exit_code == 0, result.output
    out = world.project.simulate("config", steps=60, n_paths=2, data=world.data, write=True)
    _same_frames(world.cli, world.py, "simulation.parquet")
    _same_json(world.cli, world.py, "simulation_segments.json")
    assert out.regimes.shape == (2, 60)


def test_e8_t02_backtest_and_compare_parity(world: SimpleNamespace) -> None:
    """`backtest` and `compare --reuse`: identical metrics, strategy trace and comparison table."""
    result = _cli(world.cli, "backtest", str(world.config))
    assert result.exit_code == 0, result.output
    backtest = world.project.backtest(write=True)
    assert len(backtest.folds) == 3
    for name in ("backtest_metrics.parquet", "strategy.parquet"):
        _same_frames(world.cli, world.py, name)

    config_b = _project(world.root, "b", sojourn=15.0)
    for run in (world.root / "a" / "runs", world.root / "b" / "runs"):
        shutil.copytree(world.cli, run, dirs_exist_ok=True)
    cmp_cli, cmp_py = world.root / "cmp_cli", world.root / "cmp_py"
    result = _cli(cmp_cli, "compare", str(world.config), str(config_b), "--reuse")
    assert result.exit_code == 0, result.output
    table = Project.compare(
        [world.config, config_b], reuse=True, overrides=OVERRIDES, out_dir=cmp_py, write=True
    )
    _same_frames(cmp_cli, cmp_py, "compare.parquet")
    assert (cmp_cli / "compare.md").read_text() == (cmp_py / "compare.md").read_text()
    assert len(table) == 2
    quiet = world.root / "cmp_quiet"
    Project.compare([world.config, config_b], reuse=True, overrides=OVERRIDES, out_dir=quiet)
    assert not quiet.exists()


def test_e8_t02_selftest_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """`selftest`: the CLI prints exactly the lines of `Project.selftest` (same seed)."""
    monkeypatch.setattr(selftest_mod, "QUICK_T", 300)
    monkeypatch.setattr(selftest_mod, "QUICK_RESTARTS", 1)
    result = _runner.invoke(app, ["--seed", str(SEED), "selftest"])
    assert result.exit_code == 0, result.output
    expected = _selftest_lines(Project.selftest(seed=SEED))
    lines = result.output.splitlines()  # log lines (console or, with CI set, JSON) precede the report
    assert lines[lines.index(expected[0]) :] == expected


def test_e8_t02_methods_write_nothing_without_write(world: SimpleNamespace, tmp_path: Path) -> None:
    """Without `write=True` no method touches the output directory."""
    out = tmp_path / "nowrite"
    project = Project.from_yaml(world.config, OVERRIDES, out_dir=out)
    data = project.fetch()
    project.decode(world.fit, data)
    project.forecast(world.fit, data, horizon=3)
    project.simulate(steps=20, data=data)
    project.validate()
    project.fit(data)
    assert not out.exists()


# --- AC2: global options -------------------------------------------------------------------------


def test_e8_t02_global_options_are_reflected_in_the_resolved_config_and_the_logs(
    world: SimpleNamespace, tmp_path: Path
) -> None:
    """`--seed` / `--device` / `--log-level` land in `config.resolved.yaml`; the level filters the logs."""
    resolved = load_config(world.cli / "config.resolved.yaml", FinanceConfig)
    assert resolved.project.seed == SEED
    assert resolved.estimation.device == "cpu"
    assert resolved.project.log_level == "WARNING"

    quiet = _runner.invoke(
        app, ["--out", str(tmp_path / "q"), "--log-level", "ERROR", "fetch", str(world.config)]
    )
    loud = _runner.invoke(
        app, ["--out", str(tmp_path / "l"), "--log-level", "INFO", "fetch", str(world.config)]
    )
    assert quiet.exit_code == loud.exit_code == 0
    assert "fetch_done" not in quiet.output
    assert "fetch_done" in loud.output


def test_e8_t02_unknown_override_is_a_config_error(world: SimpleNamespace) -> None:
    """A dotted override of a key the schema does not have raises `ConfigError` with the key as path."""
    with pytest.raises(ConfigError) as info:
        Project.from_yaml(world.config, {"project.nope": 1})
    assert info.value.path == "project.nope"
    with pytest.raises(ConfigError):
        Project.from_yaml(world.config, {"estimation.device": "bogus"})


# --- AC3: exit codes -----------------------------------------------------------------------------


def test_e8_t02_config_error_in_project_fit_exits_2(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `ConfigError` raised inside `Project.fit` is exit code 2 through the CLI, with its path."""

    def _boom(self: Project, *args: object, **kwargs: object) -> None:
        raise ConfigError("bad restart count", path="estimation.restarts")

    monkeypatch.setattr(Project, "fit", _boom)
    result = _runner.invoke(app, ["--out", str(world.root / "e2"), "fit", str(world.config)])
    assert result.exit_code == 2
    assert "estimation.restarts" in result.output


def test_e8_t02_provider_error_exits_5(world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `ProviderError` from the fetch inside `fit` is exit code 5."""

    def _boom(self: Project, *args: object, **kwargs: object) -> None:
        raise ProviderError("provider down", path="data.primary")

    monkeypatch.setattr(Project, "fetch_inputs", _boom)
    result = _runner.invoke(app, ["--out", str(world.root / "e5"), "fit", str(world.config)])
    assert result.exit_code == 5


def test_e8_t02_unexpected_exception_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`main()` maps any other exception to exit code 1 and a one-line message."""

    def _explode() -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(app_module, "app", _explode)
    with pytest.raises(SystemExit) as info:
        app_module.main()
    assert info.value.code == 1
    assert "unexpected error" in capsys.readouterr().err


# --- AC4: importing is side-effect free ----------------------------------------------------------


def test_e8_t02_importing_project_needs_no_network_and_writes_no_files(tmp_path: Path) -> None:
    """`from regimekit import Project` with sockets disabled leaves the working directory empty."""
    code = (
        "import socket\n"
        "def _blocked(*a, **k):\n"
        "    raise RuntimeError('network access')\n"
        "socket.socket.connect = _blocked\n"
        "socket.getaddrinfo = _blocked\n"
        "socket.create_connection = _blocked\n"
        "from regimekit import Project\n"
        "assert Project.__name__ == 'Project'\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert list(tmp_path.iterdir()) == []
