"""E14-T04: parametric bootstrap SE and nested-model LRT on small Gaussian / geometric models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection, parse_project_config
from hsmmkit.estimation import fit as run_fit
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.selection.bootstrap import (
    _simulate_like,
    bootstrap_lrt,
    bootstrap_se,
    render_bootstrap_section,
)
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.rng import PURPOSE_BOOT, key_for
from tests.helpers import em_models
from tests.helpers.core_config import core_raw
from tests.helpers.engine_fixtures import make_dataset

_EST = EstimationSection.model_validate(
    {"method": "map", "restarts": 1, "init": ["rolling_kmeans"], "max_iter": 80}
)


def _spec(tmp_path: Path, n: int) -> Any:  # noqa: ANN401
    """A Gaussian / geometric model with `n` regimes (starting values spread over the data range)."""
    raw = core_raw(tmp_path)
    raw["model"]["n_states"] = n
    raw["model"]["emission"] = {
        "family": "gaussian",
        "scale_by_dt": False,
        "ito_correction": False,
        "params": {
            "mean": {"mode": "free", "init": list(np.linspace(0.0, 2.0, n))},
            "scale": {"mode": "free", "init": [1.0] * n},
        },
    }
    raw["model"]["duration"] = {
        "family": "geometric",
        "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": [20.0] * n}},
    }
    cfg = parse_project_config(raw)
    return build_model_spec_from(cfg, BuildInputs(n_series=1, cov_names={}, clock="steps", nominal_dt=1.0))


def _truth(spec: Any, means: list[float], scales: list[float], sojourn: float) -> dict[str, Any]:  # noqa: ANN401
    theta = dict(ParamManager(spec.param_specs).unflatten(ParamManager(spec.param_specs).init_u()))
    theta["emission.mean"] = jnp.asarray(means).reshape(-1, 1)
    theta["emission.scale"] = jnp.asarray(scales).reshape(-1, 1)
    theta["duration.rate"] = jnp.full(len(means), -np.log(1 - 1 / sojourn))
    return theta


def _warm_fit(spec: Any, data: Any, theta: dict[str, Any], seed: int) -> Any:  # noqa: ANN401
    u = ParamManager(spec.param_specs).flatten(theta)
    return run_fit(spec, data, _EST.model_copy(update={"restarts": 0}), seed, u)


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any, Any]:
    """`(spec, data, fit)` of a 2-regime Gaussian / geometric model, T = 800."""
    spec, _ = em_models.build(tmp_path_factory.mktemp("boot"), "gaussian", "geometric")
    data = em_models.dataset("gaussian", 800, 3)
    return spec, data, run_fit(spec, data, _EST, 0)


def test_e14_t04_the_bootstrap_is_deterministic_and_reports_failures(small: tuple[Any, Any, Any]) -> None:
    """AC3: same seed, same table (also with a process pool); a refit that cannot converge is listed."""
    _, data, fit = small
    a = bootstrap_se(fit, data, 4, 5, estimation=_EST)
    b = bootstrap_se(fit, data, 4, 5, estimation=_EST)
    c = bootstrap_se(fit, data, 4, 6, estimation=_EST)
    np.testing.assert_array_equal(a.table["se"].to_numpy(), b.table["se"].to_numpy())
    assert not np.array_equal(a.table["se"].to_numpy(), c.table["se"].to_numpy())
    assert a.n_requested == 4
    assert a.n_failed == 0
    assert {"emission.mean", "emission.scale", "duration.rate"} == set(a.table["parameter"])
    assert (a.table["lower"] <= a.table["upper"]).all()
    assert a.compare_hessian is not None
    assert "## Bootstrap" in render_bootstrap_section(a)
    par = bootstrap_se(fit, data, 4, 5, estimation=_EST, workers=2)
    np.testing.assert_allclose(par.table["se"].to_numpy(), a.table["se"].to_numpy(), rtol=1e-12)
    one_iter = _EST.model_copy(update={"max_iter": 1})
    with pytest.raises(ConfigError, match="only 0 of 3 refits succeeded"):
        bootstrap_se(fit, data, 3, 5, estimation=one_iter, require_converged=True)


def test_e14_t04_unsupported_fits_and_small_r_are_refused(small: tuple[Any, Any, Any]) -> None:
    """`R < 2` and non-warm-startable estimators raise a `ConfigError`."""
    _, data, fit = small
    with pytest.raises(ConfigError, match="R must be"):
        bootstrap_se(fit, data, 1, 0)
    import dataclasses

    with pytest.raises(ConfigError, match="refits with"):
        bootstrap_se(dataclasses.replace(fit, estimator="vb"), data, 4, 0)


def test_e14_t04_simulated_data_keep_dt_mask_and_covariates(small: tuple[Any, Any, Any]) -> None:
    """The replicate has the lengths, `Delta_t` and mask of the data and different observations."""
    _, data, fit = small
    sim = _simulate_like(fit.params, fit.spec, data, key_for(1, PURPOSE_BOOT, 0))
    assert sim.observations.shape == data.observations.shape
    np.testing.assert_array_equal(np.asarray(sim.dt), np.asarray(data.dt))
    np.testing.assert_array_equal(np.asarray(sim.mask), np.asarray(data.mask))
    assert not np.allclose(np.asarray(sim.observations), np.asarray(data.observations))


@pytest.mark.slow
def test_e14_t04_bootstrap_se_matches_the_simulation_sd_at_t_5000(tmp_path: Path) -> None:
    """AC1: bootstrap SE within 20 % of the SD of the estimates over 50 independent datasets (T = 5000)."""
    spec = _spec(tmp_path, 2)
    truth = _truth(spec, [0.0, 1.5], [0.5, 1.2], 25.0)
    template = make_dataset(np.zeros((5000, 1)), np.ones(5000))
    estimates = []
    for k in range(51):
        data = _simulate_like(truth, spec, template, key_for(11, PURPOSE_BOOT, 1000 + k))
        res = _warm_fit(spec, data, truth, k)
        estimates.append(res)
    reference = {
        n: np.std(np.stack([np.asarray(r.params[n]) for r in estimates[1:]]), axis=0, ddof=1)
        for n in ("emission.mean", "emission.scale", "duration.rate")
    }
    data0 = _simulate_like(truth, spec, template, key_for(11, PURPOSE_BOOT, 1000))
    boot = bootstrap_se(estimates[0], data0, 100, 3, estimation=_EST)
    for name, sd in reference.items():
        ratio = boot.se[name].reshape(-1) / sd.reshape(-1)
        assert np.all(np.abs(ratio - 1.0) <= 0.2), (name, ratio)


def _lrt_pair(tmp_path: Path, data: Any) -> tuple[Any, Any]:  # noqa: ANN401
    """`(fit0 with N = 2, fit1 with N = 3)` on `data`."""
    fits = []
    for n in (2, 3):
        est = _EST.model_copy(update={"restarts": 2})
        fits.append(run_fit(_spec(tmp_path, n), data, est, 0))
    return fits[0], fits[1]


@pytest.mark.slow
@pytest.mark.parametrize("truth_n", [2, 3])
def test_e14_t04_bootstrap_lrt_two_versus_three_states(tmp_path: Path, truth_n: int) -> None:
    """AC2: N = 2 vs 3 (R = 99): p > 0.05 on 2-state, p < 0.05 on 3-state data, each in >= 4 of 5 seeds."""
    spec = _spec(tmp_path, truth_n)
    if truth_n == 2:
        truth = _truth(spec, [0.0, 2.0], [0.6, 1.0], 20.0)
    else:
        truth = _truth(spec, [-2.5, 0.0, 2.5], [0.6, 0.6, 0.6], 20.0)
    template = make_dataset(np.zeros((600, 1)), np.ones(600))
    passed = 0
    for seed in range(5):
        data = _simulate_like(truth, spec, template, key_for(21, PURPOSE_BOOT, seed))
        fit0, fit1 = _lrt_pair(tmp_path, data)
        res = bootstrap_lrt(fit0, fit1, data, 99, seed, estimation=_EST.model_copy(update={"restarts": 1}))
        passed += int(res.p_value > 0.05) if truth_n == 2 else int(res.p_value < 0.05)
    assert passed >= 4


def test_e14_t04_cli_bootstrap_writes_the_table_and_a_report_section(tmp_path: Path) -> None:
    """`hsmmkit bootstrap CONFIG --model PATH --R 3`: table, `bootstrap.md` and the report section."""
    import pandas as pd
    from ruamel.yaml import YAML
    from typer.testing import CliRunner

    from hsmmkit.cli.app import app

    values = em_models.observations("gaussian", 400, 3)[:, 0]
    pd.DataFrame({"t": pd.bdate_range("2020-01-01", periods=400).strftime("%Y-%m-%d"), "x": values}).to_csv(
        tmp_path / "data.csv", index=False
    )
    raw = core_raw(tmp_path)
    raw["data"]["source"] = {
        "kind": "csv",
        "path": str(tmp_path / "data.csv"),
        "observation_columns": ["x"],
        "time_column": "t",
    }
    raw["project"]["output_dir"] = str(tmp_path / "runs")
    raw["estimation"].update(restarts=1, max_iter=40)
    config = tmp_path / "config.yaml"
    with config.open("w", encoding="utf-8") as fh:
        YAML(typ="safe").dump(raw, fh)
    runner, out = CliRunner(), tmp_path / "out"
    assert runner.invoke(app, ["--out", str(out), "fit", str(config)]).exit_code == 0
    (out / "report.md").write_text("# Report\n\n## Other\n\ntext\n", encoding="utf-8")
    model = out / "core_demo.hsmm"
    result = runner.invoke(
        app, ["--out", str(out), "bootstrap", str(config), "--model", str(model), "--R", "3"]
    )
    assert result.exit_code == 0, result.output
    table = pd.read_parquet(out / "bootstrap.parquet")
    assert {"parameter", "estimate", "se", "lower", "upper", "hessian_se"} <= set(table.columns)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Bootstrap" in report
    assert "## Other" in report
    bad = runner.invoke(app, ["--out", str(out), "bootstrap", str(config), "--model", str(model), "--R", "1"])
    assert bad.exit_code != 0
