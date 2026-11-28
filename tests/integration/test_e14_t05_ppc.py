"""E14-T05: posterior predictive checks: statistics, tail probabilities, determinism, MCMC draws."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation import fit as run_fit
from hsmmkit.plugins import REGISTRY
from hsmmkit.selection.ppc import ppc
from hsmmkit.utils.errors import ConfigError
from tests.helpers import em_models
from tests.helpers.engine_fixtures import make_dataset

_EST = EstimationSection.model_validate(
    {"method": "map", "restarts": 1, "init": ["rolling_kmeans"], "max_iter": 80}
)


def _data(kind: str, seed: int, t_len: int = 1500) -> Any:  # noqa: ANN401
    """A 2-regime series with Gaussian or Student-t (`nu = 3`) noise."""
    z = em_models.regimes(t_len, seed)
    rng = np.random.default_rng(seed + 99)
    noise = rng.normal(size=t_len) if kind == "gauss" else rng.standard_t(3, size=t_len)
    y = np.where(z == 0, 0.0, 1.5) + np.where(z == 0, 0.5, 1.2) * noise
    return make_dataset(y[:, None], np.ones(t_len))


@pytest.fixture(scope="module")
def gauss_spec(tmp_path_factory: pytest.TempPathFactory) -> Any:  # noqa: ANN401
    """A 2-regime Gaussian / geometric model spec."""
    spec, _ = em_models.build(tmp_path_factory.mktemp("ppc"), "gaussian", "geometric")
    return spec


def _kurtosis_p(spec: Any, kind: str, seed: int, reps: int = 60) -> float:  # noqa: ANN401
    data = _data(kind, seed)
    res = ppc(run_fit(spec, data, _EST, seed), data, reps, seed, statistics=("kurtosis",))
    return float(res.table["p"].iloc[0])


def test_e14_t05_ppc_flags_the_misspecified_model_and_not_the_correct_one(gauss_spec: Any) -> None:  # noqa: ANN401
    """AC2 (single seeds): the kurtosis of Student-t data is far in the tail of a Gaussian model."""
    assert _kurtosis_p(gauss_spec, "gauss", 0) > 0.05
    assert _kurtosis_p(gauss_spec, "t3", 0) < 0.05


def test_e14_t05_default_statistics_deterministic_and_documented_table(gauss_spec: Any) -> None:  # noqa: ANN401
    """The three default statistics, per-element p in [0, 1], deterministic per seed."""
    data = _data("gauss", 4, 600)
    fit = run_fit(gauss_spec, data, _EST, 4)
    a = ppc(fit, data, 8, 1)
    b = ppc(fit, data, 8, 1)
    c = ppc(fit, data, 8, 2)
    assert set(a.table["statistic"]) == {"acf_abs", "sojourn_mean", "kurtosis"}
    assert list(a.table.columns) == [
        "statistic",
        "element",
        "observed",
        "rep_mean",
        "rep_lower",
        "rep_upper",
        "p",
    ]
    assert (a.table["element"].str.startswith("m0.lag")).sum() == 20
    assert {"regime0", "regime1"} <= set(a.table["element"])
    assert ((a.table["p"].dropna() >= 0.0) & (a.table["p"].dropna() <= 1.0)).all()
    np.testing.assert_array_equal(a.replicates["kurtosis"], b.replicates["kurtosis"])
    assert not np.array_equal(a.replicates["kurtosis"], c.replicates["kurtosis"])


def test_e14_t05_user_statistics_are_registered_by_name_and_unknown_names_fail(gauss_spec: Any) -> None:  # noqa: ANN401
    """Kind `ppc_statistic`: a registered function is used; an unknown name lists the registered ones."""
    data = _data("gauss", 5, 400)
    fit = run_fit(gauss_spec, data, _EST, 5)
    REGISTRY.register(
        "ppc_statistic",
        "e14_t05_max_abs",
        lambda y, ctx: {"m0": float(np.max(np.abs(y[:, 0])))},
        provenance="tests",
        replace=True,
    )
    res = ppc(fit, data, 6, 0, statistics=("e14_t05_max_abs",))
    assert list(res.table["element"]) == ["m0"]
    with pytest.raises(ConfigError, match="e14_t05_nope"):
        ppc(fit, data, 6, 0, statistics=("e14_t05_nope",))
    with pytest.raises(ConfigError, match="R must be"):
        ppc(fit, data, 1, 0)


def test_e14_t05_an_mcmc_fit_draws_its_parameters_from_the_posterior(gauss_spec: Any) -> None:  # noqa: ANN401
    """A fit with posterior samples gives replicates that differ from the plug-in ones."""
    pytest.importorskip("arviz")
    from hsmmkit.estimation.mcmc import MCMCEstimator

    data = _data("gauss", 6, 400)
    cfg = EstimationSection.model_validate(
        {
            "method": "mcmc",
            "restarts": 1,
            "init": ["rolling_kmeans"],
            "mcmc": {"sampler": "gibbs", "chains": 2, "warmup": 40, "draws": 40},
        }
    )
    mc = MCMCEstimator().fit(gauss_spec, data, cfg, 0)
    assert mc.posterior_samples is not None
    plug = dataclasses.replace(mc, posterior_samples=None)
    a = ppc(mc, data, 12, 3, statistics=("kurtosis",))
    b = ppc(plug, data, 12, 3, statistics=("kurtosis",))
    assert np.all(np.isfinite(a.replicates["kurtosis"]))
    assert not np.array_equal(a.replicates["kurtosis"], b.replicates["kurtosis"])


@pytest.mark.slow
def test_e14_t05_ppc_kurtosis_over_five_seeds(gauss_spec: Any) -> None:  # noqa: ANN401
    """AC2: kurtosis p < 0.05 for the misspecified model in >= 4/5 seeds, > 0.05 for the correct one."""
    bad = [_kurtosis_p(gauss_spec, "t3", s, 100) < 0.05 for s in range(5)]
    ok = [_kurtosis_p(gauss_spec, "gauss", s, 100) > 0.05 for s in range(5)]
    assert sum(bad) >= 4
    assert sum(ok) >= 4
