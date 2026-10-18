"""E6-T02: `decode` API and CLI -- filtered/smoothed/Viterbi/segments, `posterior.parquet`."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from click.testing import Result
from typer.testing import CliRunner

from regimekit.cli.app import app
from regimekit.config.yaml_io import load_config
from regimekit.estimation.results import FitResult
from regimekit.inference.decode import decode
from regimekit.model.build import build_model_spec
from regimekit.model.params.manager import ParamManager
from regimekit.persistence.results import read_prepared
from regimekit.persistence.store import ModelStore
from tests.integration.test_e2_t08_fetch import _fetch, _write_csv_project

_runner = CliRunner()


def _empty_restarts() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["restart", "init_strategy", "objective", "loglik", "n_iter", "converged", "wall_s"]
    )


def _write_model(config_path: Path, out_dir: Path, model_dir: Path, *, u: jnp.ndarray | None = None) -> None:
    """Build a `ModelSpec` from the resolved config + prepared data, write a model bundle."""
    cfg = load_config(config_path)
    dataset = read_prepared(out_dir / "prepared.parquet")
    spec = build_model_spec(cfg, dataset)
    manager = ParamManager(spec.param_specs)
    u0 = manager.init_u() if u is None else u
    fit = FitResult(
        estimator="map",
        params=manager.unflatten(u0),
        uparams=u0,
        objective=0.0,
        loglik=0.0,
        restarts=_empty_restarts(),
        converged=True,
        n_iter=0,
        std_errors=None,
        diagnostics={},
        spec=spec,
        prep_stats=dataset.prep_stats,
        config_resolved={},
    )
    ModelStore.save(fit, model_dir, cfg, dataset)


def _decode(config: Path, out: Path, model: Path, *extra: str) -> Result:
    return _runner.invoke(app, ["--out", str(out), "decode", str(config), "--model", str(model), *extra])


# --- AC1: posterior.parquet shape + row normalization -------------------------------------------


def test_e6_t02_decode_writes_posterior_with_correct_shape_and_normalization(tmp_path: Path) -> None:
    """AC1: `T` rows, `2N+2` columns; `filtered_*`/`smoothed_*` rows each sum to 1."""
    config = _write_csv_project(tmp_path)
    out = tmp_path / "out"
    assert _fetch(config, out).exit_code == 0
    model_dir = tmp_path / "model"
    _write_model(config, out, model_dir)

    result = _decode(config, out, model_dir)
    assert result.exit_code == 0, result.output

    table = pd.read_parquet(out / "posterior.parquet")
    n_states = load_config(config).model.n_states
    assert table.shape[1] == 2 * n_states + 2
    filtered_cols = [f"filtered_{i}" for i in range(n_states)]
    smoothed_cols = [f"smoothed_{i}" for i in range(n_states)]
    assert set(filtered_cols + smoothed_cols + ["time", "viterbi"]) == set(table.columns)

    dataset = read_prepared(out / "prepared.parquet")
    assert table.shape[0] == dataset.times.shape[0]
    np.testing.assert_allclose(table[filtered_cols].to_numpy().sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(table[smoothed_cols].to_numpy().sum(axis=1), 1.0, atol=1e-6)
    assert set(table["viterbi"].unique()) <= set(range(n_states))

    criteria = json.loads((out / "criteria.json").read_text(encoding="utf-8"))
    assert np.isfinite(criteria["loglik"])
    assert np.isfinite(criteria["aic"])
    assert np.isfinite(criteria["bic"])
    assert np.isfinite(criteria["icl"])


def test_e6_t02_decode_augmented_flag_writes_extra_parquet(tmp_path: Path) -> None:
    """`--augmented` also writes `augmented.parquet` (`time` x `S` smoothed expanded posterior)."""
    config = _write_csv_project(tmp_path)
    out = tmp_path / "out"
    assert _fetch(config, out).exit_code == 0
    model_dir = tmp_path / "model"
    _write_model(config, out, model_dir)

    assert not (out / "augmented.parquet").exists()
    result = _decode(config, out, model_dir, "--augmented")
    assert result.exit_code == 0, result.output
    assert (out / "augmented.parquet").is_file()
    aug = pd.read_parquet(out / "augmented.parquet")
    row_sums = aug.drop(columns=["time"]).to_numpy().sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# --- AC2: segments.json is the RLE of the viterbi column -----------------------------------------


def test_e6_t02_segments_json_matches_rle_of_viterbi_column(tmp_path: Path) -> None:
    """AC2: `segments.json` is exactly the run-length encoding of `posterior.parquet`'s `viterbi`."""
    config = _write_csv_project(tmp_path)
    out = tmp_path / "out"
    assert _fetch(config, out).exit_code == 0
    model_dir = tmp_path / "model"
    _write_model(config, out, model_dir)
    assert _decode(config, out, model_dir).exit_code == 0

    viterbi = pd.read_parquet(out / "posterior.parquet")["viterbi"].to_numpy()
    segments = json.loads((out / "segments.json").read_text(encoding="utf-8"))

    rebuilt = np.empty_like(viterbi)
    for seg in segments:
        rebuilt[seg["start"] : seg["end"] + 1] = seg["regime"]
    np.testing.assert_array_equal(rebuilt, viterbi)
    # Every segment is a maximal run: not mergeable with its neighbour.
    for i in range(len(segments) - 1):
        assert segments[i]["regime"] != segments[i + 1]["regime"]
        assert segments[i]["end"] + 1 == segments[i + 1]["start"]


# --- AC3: non-finite loglik -> exit code 4 -------------------------------------------------------


def test_e6_t02_non_finite_loglik_exits_4(tmp_path: Path) -> None:
    """AC3: a crafted (NaN) `u` gives a non-finite log-likelihood and the CLI exits 4."""
    config = _write_csv_project(tmp_path)
    out = tmp_path / "out"
    assert _fetch(config, out).exit_code == 0
    model_dir = tmp_path / "model"
    cfg = load_config(config)
    dataset = read_prepared(out / "prepared.parquet")
    spec = build_model_spec(cfg, dataset)
    manager = ParamManager(spec.param_specs)
    nan_u = jnp.full_like(manager.init_u(), jnp.nan)
    _write_model(config, out, model_dir, u=nan_u)

    result = _decode(config, out, model_dir)
    assert result.exit_code == 4, result.output


def test_e6_t02_decode_api_matches_cli_output(tmp_path: Path) -> None:
    """The `decode()` API and the CLI command agree on filtered/smoothed/Viterbi/segments/IC."""
    config = _write_csv_project(tmp_path)
    out = tmp_path / "out"
    assert _fetch(config, out).exit_code == 0
    model_dir = tmp_path / "model"
    _write_model(config, out, model_dir)
    assert _decode(config, out, model_dir).exit_code == 0

    dataset = read_prepared(out / "prepared.parquet")
    fit = ModelStore.load(model_dir)
    api_result = decode(fit, dataset)

    table = pd.read_parquet(out / "posterior.parquet")
    n_states = load_config(config).model.n_states
    np.testing.assert_allclose(
        table[[f"filtered_{i}" for i in range(n_states)]].to_numpy(),
        np.asarray(api_result.filtered),
        atol=1e-10,
    )
    np.testing.assert_array_equal(table["viterbi"].to_numpy(), api_result.viterbi)
    segments_json = json.loads((out / "segments.json").read_text(encoding="utf-8"))
    assert [(s["start"], s["end"], s["regime"]) for s in segments_json] == api_result.segments
    criteria_json = json.loads((out / "criteria.json").read_text(encoding="utf-8"))
    assert criteria_json["loglik"] == pytest.approx(api_result.loglik)
    assert criteria_json["aic"] == pytest.approx(api_result.aic)
    assert criteria_json["bic"] == pytest.approx(api_result.bic)
    assert criteria_json["icl"] == pytest.approx(api_result.icl)


def test_e6_t02_run_decode_builds_engine_from_estimation_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_decode` builds the engine from `config.estimation` (`parallel_scan`), not a bare default.

    `decode()` itself defaults to a plain `ExpandedHMMEngine()`/`ExplicitDurationEngine()` when no
    `engine` is passed; `estimation.map.fit` configures its engine from `cfg.estimation.chunk_size`/
    `parallel_scan`, so `run_decode` must do the same (otherwise decoding a model fit with
    `parallel_scan: true`/a non-default `chunk_size` would silently use different engine settings
    than the fit did).
    """
    from regimekit.cli import app as app_module

    config = _write_csv_project(tmp_path, estimation__parallel_scan=True)
    out = tmp_path / "out"
    assert _fetch(config, out).exit_code == 0
    model_dir = tmp_path / "model"
    _write_model(config, out, model_dir)

    captured: dict[str, object] = {}
    real_decode = app_module.decode

    def _spy_decode(fit: object, data: object, engine: object = None) -> object:
        captured["engine"] = engine
        return real_decode(fit, data, engine)  # type: ignore[arg-type]

    monkeypatch.setattr(app_module, "decode", _spy_decode)
    result = _decode(config, out, model_dir)
    assert result.exit_code == 0, result.output
    assert captured["engine"].parallel_scan is True  # type: ignore[attr-defined]
