"""E14-T05: pseudo-residuals (continuous, mid-distribution, Rosenblatt), seed sweeps, fit-report sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation import fit as run_fit
from hsmmkit.selection.pseudo_residuals import pseudo_residuals
from tests.helpers import em_models
from tests.helpers.engine_fixtures import make_dataset

_EST = EstimationSection.model_validate(
    {"method": "map", "restarts": 1, "init": ["rolling_kmeans"], "max_iter": 80}
)


def _gaussian_data(kind: str, seed: int, t_len: int = 1500) -> Any:  # noqa: ANN401
    """A 2-regime series with Gaussian (`kind="gauss"`) or Student-t (`nu = 3`) noise."""
    z = em_models.regimes(t_len, seed)
    rng = np.random.default_rng(seed + 99)
    noise = rng.normal(size=t_len) if kind == "gauss" else rng.standard_t(3, size=t_len)
    y = np.where(z == 0, 0.0, 1.5) + np.where(z == 0, 0.5, 1.2) * noise
    return make_dataset(y[:, None], np.ones(t_len))


@pytest.fixture(scope="module")
def gauss_spec(tmp_path_factory: pytest.TempPathFactory) -> Any:  # noqa: ANN401, D103
    spec, _ = em_models.build(tmp_path_factory.mktemp("res"), "gaussian", "geometric")
    return spec


def _shapiro_p(spec: Any, kind: str, seed: int) -> float:  # noqa: ANN401
    data = _gaussian_data(kind, seed)
    res = pseudo_residuals(run_fit(spec, data, _EST, seed), data)
    return float(res.table["shapiro_p"].iloc[0])


def test_e14_t05_a_correct_and_a_misspecified_gaussian_are_told_apart(gauss_spec: Any) -> None:  # noqa: ANN401
    """AC1 (single seeds): correct model passes Shapiro-Wilk, a Gaussian fitted to Student-t data fails."""
    assert _shapiro_p(gauss_spec, "gauss", 0) > 0.05
    assert _shapiro_p(gauss_spec, "t3", 0) < 0.05


def test_e14_t05_the_result_has_the_documented_shape(gauss_spec: Any) -> None:  # noqa: ANN401
    """Residuals `(T, M)`, `u` in [0, 1], a table with the tests and QQ data sorted against Blom scores."""
    data = _gaussian_data("gauss", 1, 600)
    res = pseudo_residuals(run_fit(gauss_spec, data, _EST, 1), data)
    assert res.residuals.shape == (600, 1)
    assert np.all((res.u >= 0.0) & (res.u <= 1.0))
    assert res.kind == "continuous"
    row = res.table.iloc[0]
    assert row["n"] == 600
    assert {"shapiro_w", "shapiro_p", "lb_resid_10_p", "lb_resid_20_p", "lb_sq_10_p", "lb_sq_20_p"} <= set(
        res.table.columns
    )
    assert np.all(np.diff(res.qq["sample"].to_numpy()) >= 0.0)
    assert np.all(np.diff(res.qq["theoretical"].to_numpy()) > 0.0)
    assert abs(float(np.mean(res.residuals))) < 0.2


def test_e14_t05_masked_steps_have_no_residual(gauss_spec: Any) -> None:  # noqa: ANN401
    """Unobserved steps give NaN residuals and are left out of the tests."""
    import dataclasses

    data = _gaussian_data("gauss", 2, 500)
    mask = np.ones((500, 1), dtype=bool)
    mask[100:110] = False
    masked = dataclasses.replace(data, mask=mask)
    res = pseudo_residuals(run_fit(gauss_spec, masked, _EST, 2), masked)
    assert np.isnan(res.residuals[100:110]).all()
    assert int(res.table["n"].iloc[0]) == 490


def _poisson_data(seed: int, t_len: int = 250) -> Any:  # noqa: ANN401
    """Poisson counts with means 30 / 120 (large means keep the mid-distribution lattice fine)."""
    z = em_models.regimes(t_len, seed)
    y = np.random.default_rng(seed).poisson(np.where(z == 0, 30, 120)).astype(float)
    return make_dataset(y[:, None], np.ones(t_len))


def test_e14_t05_poisson_mid_distribution_residuals_are_approximately_normal(tmp_path: Path) -> None:
    """AC3 (single seed): a correct Poisson model gives finite, nearly normal mid-distribution residuals."""
    spec, _ = em_models.build(tmp_path, "poisson", "geometric")
    data = _poisson_data(0)
    res = pseudo_residuals(run_fit(spec, data, _EST, 0), data)
    assert res.kind == "mid-distribution"
    assert np.all(np.isfinite(res.residuals))
    assert float(res.table["shapiro_w"].iloc[0]) > 0.98
    assert abs(float(np.std(res.residuals)) - 1.0) < 0.1


def test_e14_t05_rosenblatt_residuals_of_a_full_covariance_gaussian_are_independent_normals(
    tmp_path: Path,
) -> None:
    """`M = 2`, `full` covariance: sequential residuals, ~N(0, 1) and uncorrelated across the series."""
    spec, _ = em_models.build(tmp_path, "gaussian_full", "geometric")
    data = em_models.dataset("gaussian_full", 1500, 0)
    res = pseudo_residuals(run_fit(spec, data, _EST, 0), data)
    assert res.kind == "rosenblatt"
    assert res.residuals.shape == (1500, 2)
    assert (res.table["shapiro_w"] > 0.99).all()
    assert np.all(np.abs(np.std(res.residuals, axis=0) - 1.0) < 0.1)
    assert abs(float(np.corrcoef(res.residuals[:, 0], res.residuals[:, 1])[0, 1])) < 0.1


@pytest.mark.parametrize("error", ["not_implemented", "config"])
def test_e14_t05_families_without_a_cdf_are_skipped_with_a_note(gauss_spec: Any, error: str) -> None:  # noqa: ANN401
    """A family whose `cdf` raises (also a `ConfigError`, as MS-AR does) is noted; nothing crashes."""
    import dataclasses

    from hsmmkit.utils.errors import ConfigError

    class _NoCdf(type(gauss_spec.emission)):  # type: ignore[misc]
        def cdf(self, *args: object, **kwargs: object) -> object:
            if error == "config":
                raise ConfigError("no cdf for this form", path="model.emission")
            raise NotImplementedError("no cdf")

    emission = _NoCdf(
        **{
            f.name: getattr(gauss_spec.emission, f.name)
            for f in dataclasses.fields(gauss_spec.emission)
            if f.init
        }
    )
    data = _gaussian_data("gauss", 3, 300)
    fit = run_fit(gauss_spec, data, _EST, 3)
    res = pseudo_residuals(
        dataclasses.replace(fit, spec=dataclasses.replace(fit.spec, emission=emission)), data
    )
    assert res.notes
    assert np.isnan(res.residuals).all()


@pytest.mark.slow
def test_e14_t05_shapiro_wilk_pass_rates_over_50_seeds(gauss_spec: Any) -> None:  # noqa: ANN401
    """AC1: correct model passes at 5 % on >= 90 % of 50 seeds, the Student-t misfit fails on >= 90 %."""
    ok = [_shapiro_p(gauss_spec, "gauss", s) > 0.05 for s in range(50)]
    bad = [_shapiro_p(gauss_spec, "t3", s) < 0.05 for s in range(50)]
    assert np.mean(ok) >= 0.9
    assert np.mean(bad) >= 0.9


@pytest.mark.slow
def test_e14_t05_poisson_pass_rate_over_seeds(tmp_path: Path) -> None:
    """AC3: Poisson mid-distribution residuals pass Shapiro-Wilk (5 %) on >= 80 % of 30 seeds (T = 250).

    The mid-distribution has atoms: with means 2 / 9 or at T = 1500 the test detects the lattice in every seed
    (measured), so the pass rate depends on the means and the length; 30 / 120 at T = 250 measured 93 %.
    """
    spec, _ = em_models.build(tmp_path, "poisson", "geometric")
    passed = []
    for seed in range(30):
        data = _poisson_data(seed)
        res = pseudo_residuals(run_fit(spec, data, _EST, seed), data)
        passed.append(float(res.table["shapiro_p"].iloc[0]) > 0.05)
    assert np.mean(passed) >= 0.8


def test_e14_t05_the_fit_report_has_the_configured_adequacy_sections(tmp_path: Path) -> None:
    """`estimation.diagnostics`: `pseudo_residuals` by default, `ppc` on request, in `fit_report.json`."""
    import json

    import pandas as pd
    from ruamel.yaml import YAML
    from typer.testing import CliRunner

    from hsmmkit.cli.app import app
    from tests.helpers.core_config import core_raw

    values = em_models.observations("gaussian", 400, 3)[:, 0]
    pd.DataFrame({"t": pd.bdate_range("2020-01-01", periods=400).strftime("%Y-%m-%d"), "x": values}).to_csv(
        tmp_path / "data.csv", index=False
    )
    runner = CliRunner()
    sections: dict[str, set[str]] = {}
    for name, diagnostics in (("default", None), ("both", ["pseudo_residuals", "ppc"])):
        raw = core_raw(tmp_path)
        raw["data"]["source"] = {
            "kind": "csv",
            "path": str(tmp_path / "data.csv"),
            "observation_columns": ["x"],
            "time_column": "t",
        }
        raw["project"]["output_dir"] = str(tmp_path / "runs")
        raw["estimation"].update(restarts=1, max_iter=40, ppc_replicates=6)
        if diagnostics is not None:
            raw["estimation"]["diagnostics"] = diagnostics
        config = tmp_path / f"{name}.yaml"
        with config.open("w", encoding="utf-8") as fh:
            YAML(typ="safe").dump(raw, fh)
        out = tmp_path / f"out_{name}"
        assert runner.invoke(app, ["--out", str(out), "fit", str(config)]).exit_code == 0
        report = json.loads((out / "fit_report.json").read_text(encoding="utf-8"))
        sections[name] = set(report["diagnostics"])
        if name == "both":
            assert report["diagnostics"]["ppc"]["table"]
            assert report["diagnostics"]["pseudo_residuals"]["table"][0]["n"] == 400
    assert "pseudo_residuals" in sections["default"]
    assert "ppc" not in sections["default"]
    assert {"pseudo_residuals", "ppc"} <= sections["both"]
