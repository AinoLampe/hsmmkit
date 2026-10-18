"""E6-T05: `forecast` / `simulate` CLI, `forecast.parquet` / `simulation.parquet` writers and schemas."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import Result
from typer.testing import CliRunner

from regimekit.cli.app import _covariate_forecaster, app
from regimekit.config.yaml_io import load_config
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.inference.covariate_forecasters import make_forecaster
from regimekit.inference.forecast import MonteCarloForecaster
from regimekit.inference.results import ForecastResult
from regimekit.inference.simulate import segments_of
from regimekit.model.build import build_model_spec
from regimekit.model.params.manager import ParamManager
from regimekit.persistence.results import read_forecast, read_prepared, write_forecast, write_simulation
from regimekit.persistence.temp_model import load_temp_model
from regimekit.statespace.builders import build_expanded
from regimekit.utils.errors import ConfigError, PersistenceError
from tests.integration.test_e2_t08_fetch import _fetch, _write_csv_project
from tests.integration.test_e6_t02_decode import _write_model

_runner = CliRunner()


def _cli(out: Path, *args: str) -> Result:
    return _runner.invoke(app, ["--out", str(out), *args])


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """`(config, out_dir, model_dir)` for a fetched csv project with an init-valued model."""
    config = _write_csv_project(tmp_path)
    out = tmp_path / "out"
    assert _fetch(config, out).exit_code == 0
    model_dir = tmp_path / "model"
    _write_model(config, out, model_dir)
    return config, out, model_dir


def _synthetic_result(n_sim: int = 6, h: int = 4, m: int = 2, n: int = 3) -> ForecastResult:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(n), size=h)
    levels = (0.05, 0.5, 0.95)
    return ForecastResult(
        origin_time=np.datetime64("2024-03-29T00:00:00", "ns"),
        horizon_times=(np.datetime64("2024-04-01", "ns") + np.arange(h) * np.timedelta64(1, "D")).astype(
            "datetime64[ns]"
        ),
        regime_probs=jnp.asarray(probs),
        samples=jnp.asarray(rng.normal(size=(n_sim, h, m))),
        cum_quantiles=jnp.asarray(rng.normal(size=(h, len(levels), m))),
        mean=jnp.asarray(rng.normal(size=(h, m))),
        var=jnp.asarray(rng.uniform(size=(h, m))),
        quantile_levels=levels,
        covariate_forecaster="ar",
        seed=11,
    )


# --- AC1: round trip -----------------------------------------------------------------------------


def test_e6_t05_write_then_read_forecast_is_bitwise(tmp_path: Path) -> None:
    """AC1: `read_forecast` reconstructs every array bit for bit (two series, three regimes, samples kept)."""
    res = _synthetic_result()
    write_forecast(res, tmp_path, samples=True, snapshot_ids=("a", "b"))
    back = read_forecast(tmp_path)
    for name in ("regime_probs", "cum_quantiles", "mean", "var", "samples"):
        assert np.array_equal(np.asarray(getattr(back, name)), np.asarray(getattr(res, name))), name
    assert back.quantile_levels == res.quantile_levels
    assert back.origin_time == res.origin_time
    assert np.array_equal(back.horizon_times, res.horizon_times)
    assert (back.covariate_forecaster, back.seed) == ("ar", 11)
    summary = json.loads((tmp_path / "forecast.json").read_text(encoding="utf-8"))
    assert summary["snapshot_ids"] == ["a", "b"]
    assert summary["n_sim"] == 6


def test_e6_t05_samples_file_is_optional_and_stale_ones_are_removed(tmp_path: Path) -> None:
    """No `samples=True`: no samples file (an older one is deleted) and `samples` reads back empty."""
    res = _synthetic_result()
    write_forecast(res, tmp_path, samples=True)
    assert (tmp_path / "forecast_samples.parquet").is_file()
    write_forecast(res, tmp_path, samples=False)
    assert not (tmp_path / "forecast_samples.parquet").exists()
    assert read_forecast(tmp_path).samples.shape == (0, 4, 2)


def test_e6_t05_forecast_cli_matches_a_direct_library_call(tmp_path: Path) -> None:
    """The CLI output, read back, equals `MonteCarloForecaster.forecast` with the same inputs and seed."""
    config, out, model_dir = _project(tmp_path)
    result = _cli(
        out,
        "forecast",
        str(config),
        "--model",
        str(model_dir),
        "--horizon",
        "5",
        "--n-sim",
        "200",
        "--samples",
    )
    assert result.exit_code == 0, result.output

    cfg = load_config(config)
    dataset = read_prepared(out / "prepared.parquet")
    fit = load_temp_model(model_dir, dataset)
    mc = MonteCarloForecaster(
        fit.spec,
        build_expanded(fit.spec),
        ExpandedHMMEngine(parallel_scan=cfg.estimation.parallel_scan),
        make_forecaster(cfg.forecast.covariate_forecaster),
        cfg.forecast,
        bars_per_day=cfg.data.bars_per_day,
    )
    direct = mc.forecast(fit, dataset, 5, 200, cfg.project.seed)
    back = read_forecast(out)
    for name in ("regime_probs", "cum_quantiles", "mean", "var", "samples"):
        assert np.array_equal(np.asarray(getattr(back, name)), np.asarray(getattr(direct, name))), name
    assert back.seed == cfg.project.seed
    summary = json.loads((out / "forecast.json").read_text(encoding="utf-8"))
    assert summary["horizon"] == 5
    assert summary["n_sim"] == 200
    assert summary["snapshot_ids"] == list(dataset.snapshot_ids)
    assert summary["covariate_forecaster"] == direct.covariate_forecaster


def test_e6_t05_forecast_seed_override_changes_the_forecast(tmp_path: Path) -> None:
    """`--seed` overrides `project.seed` and is recorded in `forecast.json`."""
    config, out, model_dir = _project(tmp_path)
    base = [
        "forecast",
        str(config),
        "--model",
        str(model_dir),
        "--horizon",
        "3",
        "--n-sim",
        "100",
        "--samples",
    ]
    assert _cli(out, *base).exit_code == 0
    first = read_forecast(out)
    assert _cli(out, "--seed", "999", *base).exit_code == 0
    second = read_forecast(out)
    assert second.seed == 999
    assert not np.array_equal(np.asarray(first.samples), np.asarray(second.samples))


def test_e6_t05_scenario_csv_drives_the_covariates(tmp_path: Path) -> None:
    """`--scenario` switches to the scenario forecaster; a wrong-length file is a config error (exit 2)."""
    config, out, model_dir = _project(tmp_path)
    names = list(read_prepared(out / "prepared.parquet").prep_stats.cov_names)
    good = tmp_path / "scenario.csv"
    pd.DataFrame({n: np.linspace(0.0, 1.0, 4) for n in names}).to_csv(good, index=False)
    args = ["forecast", str(config), "--model", str(model_dir), "--horizon", "4", "--n-sim", "50"]
    ok = _cli(out, *args, "--scenario", str(good))
    assert ok.exit_code == 0, ok.output
    assert (
        json.loads((out / "forecast.json").read_text(encoding="utf-8"))["covariate_forecaster"] == "scenario"
    )

    bad = tmp_path / "bad.csv"
    pd.DataFrame({n: np.zeros(3) for n in names}).to_csv(bad, index=False)
    assert _cli(out, *args, "--scenario", str(bad)).exit_code == 2


def test_e6_t05_forecast_errors_map_to_exit_codes(tmp_path: Path) -> None:
    """A bad horizon is exit 2; offline without a prepared dataset is exit 3; a non-model directory exit 1."""
    config, out, model_dir = _project(tmp_path)
    assert _cli(out, "forecast", str(config), "--model", str(model_dir), "--horizon", "0").exit_code == 2
    empty = tmp_path / "empty_out"
    assert _cli(empty, "--offline", "forecast", str(config), "--model", str(model_dir)).exit_code == 3
    assert _cli(out, "forecast", str(config), "--model", str(tmp_path / "nope")).exit_code == 1


# --- AC2: simulate -------------------------------------------------------------------------------


def test_e6_t05_simulate_config_writes_1500_valid_rows(tmp_path: Path) -> None:
    """AC2: `simulate --params config --steps 500 --n-paths 3` -> 1500 rows with valid regimes and phases."""
    config, out, _ = _project(tmp_path)
    result = _cli(out, "simulate", str(config), "--params", "config", "--steps", "500", "--n-paths", "3")
    assert result.exit_code == 0, result.output

    table = pd.read_parquet(out / "simulation.parquet")
    cfg = load_config(config)
    assert len(table) == 1500
    assert set(table["path"]) == {0, 1, 2}
    assert table["regime"].between(0, cfg.model.n_states - 1).all()
    dataset = read_prepared(out / "prepared.parquet")
    spec = build_model_spec(cfg, dataset)
    max_phase = np.asarray([spec.duration.n_phases(i) for i in range(cfg.model.n_states)])
    assert table["phase"].ge(1).all()
    assert (table["phase"].to_numpy() <= max_phase[table["regime"].to_numpy()]).all()
    assert np.isfinite(table["return"]).all()
    assert (table.groupby("path")["step"].apply(lambda s: list(s) == list(range(1, 501)))).all()
    assert table.groupby("path")["time"].apply(lambda s: s.is_monotonic_increasing).all()

    segments = json.loads((out / "simulation_segments.json").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in segments] == [0, 1, 2]
    for entry in segments:
        rows = table[table["path"] == entry["path"]]["regime"].to_numpy()
        covered = sum(s["end"] - s["start"] + 1 for s in entry["segments"])
        assert covered == 500
        for s in entry["segments"]:
            assert (rows[s["start"] : s["end"] + 1] == s["regime"]).all()


def test_e6_t05_simulate_is_deterministic_and_model_params_work(tmp_path: Path) -> None:
    """The same seed gives identical simulations; `--params model` uses the stored model; misuse is exit 2."""
    config, out, model_dir = _project(tmp_path)
    base = ["simulate", str(config), "--steps", "50", "--n-paths", "2"]
    assert _cli(out, *base).exit_code == 0
    first = pd.read_parquet(out / "simulation.parquet")
    assert _cli(out, *base).exit_code == 0
    assert first.equals(pd.read_parquet(out / "simulation.parquet"))

    from_model = _cli(out, *base, "--params", "model", "--model", str(model_dir))
    assert from_model.exit_code == 0, from_model.output
    assert _cli(out, *base, "--params", "model").exit_code == 2
    assert _cli(out, *base, "--params", "bogus").exit_code == 2
    assert _cli(out, "simulate", str(config), "--n-paths", "0").exit_code == 2


# --- AC3: documented schemas ---------------------------------------------------------------------


def test_e6_t05_parquet_schemas_are_the_documented_ones(tmp_path: Path) -> None:
    """AC3: column names and dtypes of the three parquet outputs."""
    config, out, model_dir = _project(tmp_path)
    assert (
        _cli(
            out,
            "forecast",
            str(config),
            "--model",
            str(model_dir),
            "--horizon",
            "3",
            "--n-sim",
            "20",
            "--samples",
        ).exit_code
        == 0
    )
    assert _cli(out, "simulate", str(config), "--steps", "10", "--n-paths", "2").exit_code == 0

    def schema(name: str) -> list[tuple[str, pa.DataType]]:
        s = pq.read_schema(out / name)
        return [(f.name, f.type) for f in s]

    assert schema("forecast.parquet") == [
        ("origin_time", pa.timestamp("ns")),
        ("horizon_time", pa.timestamp("ns")),
        ("step", pa.int64()),
        ("kind", pa.string()),
        ("key", pa.string()),
        ("series", pa.int64()),
        ("value", pa.float64()),
    ]
    assert schema("forecast_samples.parquet") == [
        ("path", pa.int64()),
        ("step", pa.int64()),
        ("series", pa.int64()),
        ("value", pa.float64()),
    ]
    assert schema("simulation.parquet") == [
        ("path", pa.int64()),
        ("step", pa.int64()),
        ("time", pa.timestamp("ns")),
        ("regime", pa.int64()),
        ("phase", pa.int64()),
        ("return", pa.float64()),
    ]
    forecast = pd.read_parquet(out / "forecast.parquet")
    assert set(forecast["kind"]) == {"regime_prob", "quantile", "mean", "var"}
    n_states = load_config(config).model.n_states
    n_quantiles = len(load_config(config).forecast.quantiles)
    per_step = n_states + n_quantiles + 2
    assert len(forecast) == 3 * per_step


def test_e6_t05_scenario_values_reach_the_forecast(tmp_path: Path) -> None:
    """With non-trivial coefficients, different scenario paths give different forecasts (same seed)."""
    config, out, _ = _project(tmp_path)
    cfg = load_config(config)
    dataset = read_prepared(out / "prepared.parquet")
    manager = ParamManager(build_model_spec(cfg, dataset).param_specs)
    base_u = manager.init_u()
    u = base_u + jnp.asarray(np.random.default_rng(0).normal(0.0, 1.0, base_u.shape))
    model_dir = tmp_path / "model_random"
    _write_model(config, out, model_dir, u=u)
    names = list(dataset.prep_stats.cov_names)

    def run(value: float) -> np.ndarray:
        path = tmp_path / f"scenario_{value}.csv"
        pd.DataFrame({n: np.full(4, value) for n in names}).to_csv(path, index=False)
        args = ["forecast", str(config), "--model", str(model_dir), "--horizon", "4", "--n-sim", "3000"]
        assert _cli(out, *args, "--scenario", str(path)).exit_code == 0
        return np.asarray(read_forecast(out).regime_probs)

    assert np.abs(run(-3.0) - run(3.0)).max() > 0.02


def test_e6_t05_scenario_without_covariates_is_a_config_error(tmp_path: Path) -> None:
    """`--scenario` on a dataset without covariates is rejected instead of silently ignored."""
    config, out, _ = _project(tmp_path)
    dataset = read_prepared(out / "prepared.parquet")
    bare = dataclasses.replace(dataset, prep_stats=dataclasses.replace(dataset.prep_stats, cov_names=()))
    cfg = load_config(config)
    assert _covariate_forecaster(cfg, tmp_path, bare, None) is None
    with pytest.raises(ConfigError):
        _covariate_forecaster(cfg, tmp_path, bare, tmp_path / "scenario.csv")


def test_e6_t05_non_numeric_scenario_column_is_exit_2(tmp_path: Path) -> None:
    """A scenario CSV with a text column is a config error (exit 2), not a traceback."""
    config, out, model_dir = _project(tmp_path)
    names = list(read_prepared(out / "prepared.parquet").prep_stats.cov_names)
    bad = tmp_path / "text.csv"
    pd.DataFrame({n: ["a", "b", "c", "d"] for n in names}).to_csv(bad, index=False)
    args = ["forecast", str(config), "--model", str(model_dir), "--horizon", "4", "--n-sim", "50"]
    assert _cli(out, *args, "--scenario", str(bad)).exit_code == 2


def test_e6_t05_malformed_forecast_dir_is_a_persistence_error(tmp_path: Path) -> None:
    """A forecast.json with a missing key is a `PersistenceError`, not a `KeyError`."""
    write_forecast(_synthetic_result(), tmp_path)
    meta = json.loads((tmp_path / "forecast.json").read_text(encoding="utf-8"))
    del meta["n_states"]
    (tmp_path / "forecast.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(PersistenceError):
        read_forecast(tmp_path)


def test_e6_t05_write_simulation_splits_returns_for_several_series(tmp_path: Path) -> None:
    """`M > 1` -> `return_0`, `return_1`; segments json is written per path."""
    rng = np.random.default_rng(1)
    regimes = np.asarray([[0, 0, 1, 1], [1, 0, 0, 0]])
    returns = rng.normal(size=(2, 4, 2))
    write_simulation(
        tmp_path,
        times=np.arange(4).astype("datetime64[D]").astype("datetime64[ns]"),
        regimes=regimes,
        phases=np.ones_like(regimes),
        returns=returns,
        segments=segments_of(regimes),
    )
    table = pd.read_parquet(tmp_path / "simulation.parquet")
    assert list(table.columns) == ["path", "step", "time", "regime", "phase", "return_0", "return_1"]
    assert np.array_equal(table["return_1"].to_numpy(), returns[:, :, 1].reshape(-1))
    segs = json.loads((tmp_path / "simulation_segments.json").read_text(encoding="utf-8"))
    assert segs[0]["segments"] == [{"start": 0, "end": 1, "regime": 0}, {"start": 2, "end": 3, "regime": 1}]
