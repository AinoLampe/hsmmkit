"""E13-T06b: blocked Gibbs sampler, NUTS agreement, relabelling in the estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("arviz")

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation.mcmc import MCMCEstimator
from hsmmkit.utils.errors import ConfigError
from tests.helpers import em_models

pytestmark = pytest.mark.bayes


def _cfg(sampler: str = "gibbs", **mcmc: Any) -> EstimationSection:  # noqa: ANN401
    base = {"sampler": sampler, "chains": 2, "warmup": 50, "draws": 80}
    return EstimationSection.model_validate(
        {"method": "mcmc", "restarts": 1, "init": ["rolling_kmeans"], "mcmc": {**base, **mcmc}}
    )


def _model(tmp_path: Path, family: str = "gaussian", t_len: int = 300, seed: int = 3):  # noqa: ANN202
    spec, _ = em_models.build(tmp_path, family, "geometric")
    return spec, em_models.dataset(family, t_len, seed)


def test_e13_t06b_gibbs_draws_are_reproducible_per_seed(tmp_path: Path) -> None:
    """AC3: same seed, same draws; another seed, other draws; the fit report names the sampler."""
    spec, data = _model(tmp_path)
    a = MCMCEstimator().fit(spec, data, _cfg(), 4)
    b = MCMCEstimator().fit(spec, data, _cfg(), 4)
    c = MCMCEstimator().fit(spec, data, _cfg(), 5)
    key = ("posterior", "emission.mean")
    pa, pb, pc = (np.asarray(r.posterior_samples[key[0]][key[1]]) for r in (a, b, c))
    np.testing.assert_array_equal(pa, pb)
    assert not np.array_equal(pa, pc)
    info = a.diagnostics["mcmc"]
    assert info["sampler"] == "gibbs"
    assert 0.0 <= info["relabelled_fraction"] <= 1.0
    assert a.estimator == "mcmc"
    assert set(a.params) == {s.name for s in spec.param_specs}


def test_e13_t06b_a_non_conjugate_model_with_gibbs_lists_the_components(tmp_path: Path) -> None:
    """AC3: Student-t emissions are not conjugate: `ConfigError` names the component and suggests nuts."""
    spec, data = _model(tmp_path, "student_t")
    with pytest.raises(ConfigError, match=r"emission.*nuts"):
        MCMCEstimator().fit(spec, data, _cfg(), 0)


def test_e13_t06b_sampled_paths_are_kept_on_request(tmp_path: Path) -> None:
    """`keep_paths_every` adds a `path_samples` group of regime paths (thinned)."""
    spec, data = _model(tmp_path, t_len=200)
    res = MCMCEstimator().fit(spec, data, _cfg(keep_paths_every=20), 1)
    paths = np.asarray(res.posterior_samples["path_samples"]["regime_path"])
    assert paths.shape == (2, 4, 1, 200)  # 80 draws, every 20th
    assert set(np.unique(paths)) <= {0, 1}


def test_e13_t06b_nuts_draws_are_relabelled_when_nothing_identifies_the_states(tmp_path: Path) -> None:
    """The NUTS estimator reports the relabelled fraction (zero for a well-separated posterior)."""
    spec, data = _model(tmp_path, t_len=400)
    res = MCMCEstimator().fit(spec, data, _cfg("nuts", warmup=40, draws=40), 0)
    assert res.diagnostics["mcmc"]["relabelled_fraction"] == 0.0


@pytest.mark.slow
def test_e13_t06b_gibbs_and_nuts_agree_on_the_two_regime_fixture(tmp_path: Path) -> None:
    """AC1 (Appendix E, Gaussian + geometric): means within 2 MC SE of each other, 2 SD of truth, R-hat."""
    spec, data = _model(tmp_path, t_len=5000, seed=4)
    results = {}
    for sampler in ("gibbs", "nuts"):
        cfg = EstimationSection.model_validate(
            {"method": "mcmc", "restarts": 1, "init": ["rolling_kmeans"], "mcmc": {"sampler": sampler}}
        )
        results[sampler] = MCMCEstimator().fit(spec, data, cfg, 0)
    truth = {
        "emission.mean": np.array([[0.0], [1.5]]),
        "emission.scale": np.array([[0.5], [1.2]]),
        "duration.rate": np.full(2, -np.log(1 - 1 / 25)),
    }
    for name, res in results.items():
        info = res.diagnostics["mcmc"]
        assert info["max_r_hat"] <= 1.05, name
        assert res.std_errors is not None
        for pname, value in truth.items():
            z = np.abs(np.asarray(res.params[pname]) - value) / np.asarray(res.std_errors[pname])
            assert np.all(z <= 2.0), (name, pname, z)
    for pname in truth:
        mc_se = []
        for res in results.values():
            ess = np.array(
                [v for k, v in res.diagnostics["mcmc"]["ess_bulk"].items() if k.startswith(pname)]
            ).reshape(np.shape(res.params[pname]))
            mc_se.append(np.asarray(res.std_errors[pname]) / np.sqrt(ess))  # type: ignore[index]
        gap = np.abs(np.asarray(results["gibbs"].params[pname]) - np.asarray(results["nuts"].params[pname]))
        assert np.all(gap <= 2.0 * np.sqrt(mc_se[0] ** 2 + mc_se[1] ** 2)), (pname, gap)
