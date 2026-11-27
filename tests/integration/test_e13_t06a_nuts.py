"""E13-T06a: NUTS posterior sampling (`bayes` extra): reproducibility, ArviZ output, refusals, Appendix E."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("numpyro")
pytest.importorskip("arviz")

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation import fit as dispatch_fit
from hsmmkit.estimation.mcmc import MCMCEstimator, improper_parameters, with_default_priors
from hsmmkit.model.params.manager import ParamManager
from tests.helpers import em_models

pytestmark = pytest.mark.bayes


def _cfg(**mcmc: Any) -> EstimationSection:  # noqa: ANN401
    base = {"chains": 2, "warmup": 60, "draws": 60}
    return EstimationSection.model_validate(
        {"method": "mcmc", "restarts": 1, "init": ["rolling_kmeans"], "mcmc": {**base, **mcmc}}
    )


def _model(tmp_path: Path, t_len: int = 300, seed: int = 3):  # noqa: ANN202
    spec, _ = em_models.build(tmp_path, "gaussian", "geometric")
    return spec, em_models.dataset("gaussian", t_len, seed)


def test_e13_t06a_seeded_runs_are_reproducible_bitwise(tmp_path: Path) -> None:
    """AC2: same seed, backend and versions give identical draws; another seed gives other draws."""
    spec, data = _model(tmp_path)
    a = MCMCEstimator().fit(spec, data, _cfg(), 5)
    b = MCMCEstimator().fit(spec, data, _cfg(), 5)
    c = MCMCEstimator().fit(spec, data, _cfg(), 6)
    pa, pb, pc = (np.asarray(r.posterior_samples["posterior"]["emission.mean"]) for r in (a, b, c))
    np.testing.assert_array_equal(pa, pb)
    assert not np.array_equal(pa, pc)
    assert a.diagnostics["mcmc"]["versions"]["numpyro"]


def test_e13_t06a_inference_data_is_named_after_the_parameter_specs(tmp_path: Path) -> None:
    """AC3: dims / coords from the specs; `az.summary` runs; the fit report is filled."""
    az = pytest.importorskip("arviz")
    spec, data = _model(tmp_path)
    res = MCMCEstimator().fit(spec, data, _cfg(), 1)
    post = res.posterior_samples["posterior"]
    names = {s.name for s in spec.param_specs}
    assert set(post.data_vars) == names
    assert post["emission.mean"].dims == ("chain", "draw", "regime", "series")
    assert post["emission.mean"].shape == (2, 60, 2, 1)
    assert set(res.posterior_samples["log_likelihood"].data_vars) == {"loglik"}
    assert res.posterior_samples["log_likelihood"]["loglik"].shape == (2, 60, 1)
    assert set(res.posterior_samples["sample_stats"].data_vars) >= {"diverging", "tree_depth", "energy"}
    table = az.summary(res.posterior_samples)
    assert {"r_hat", "ess_bulk", "ess_tail"} <= set(table.columns)
    info = res.diagnostics["mcmc"]
    assert set(info) >= {"r_hat", "ess_bulk", "ess_tail", "divergences", "versions"}
    assert res.estimator == "mcmc"
    assert res.std_errors is not None
    assert np.isfinite(res.loglik)
    assert set(res.params) == names


def test_e13_t06a_without_the_bayes_extra_the_error_names_the_extra(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """AC4: `method: mcmc` without NumPyro raises an actionable ImportError."""
    spec, data = _model(tmp_path)
    monkeypatch.setitem(sys.modules, "numpyro", None)
    with pytest.raises(ImportError, match=r"hsmmkit\[bayes\]"):
        dispatch_fit(spec, data, _cfg(), 0)


def test_e13_t06a_improper_posteriors_and_unsupported_options_are_refused(tmp_path: Path) -> None:
    """Improper posteriors, flat `use_priors: false` and the Gibbs sampler are refused."""
    from hsmmkit.utils.errors import ConfigError

    spec, data = _model(tmp_path)
    assert improper_parameters(spec)  # the Gaussian model has free unbounded parameters
    assert improper_parameters(with_default_priors(spec)) == []
    with pytest.raises(ConfigError, match="improper"):
        MCMCEstimator().fit(spec, data, _cfg(default_priors=False), 0)
    flat = _cfg().model_copy(update={"use_priors": False})
    with pytest.raises(ConfigError, match="use_priors"):
        MCMCEstimator().fit(spec, data, flat, 0)
    with pytest.raises(ConfigError, match="E13-T06b"):
        MCMCEstimator().fit(spec, data, _cfg(sampler="gibbs"), 0)


def test_e13_t06a_vectorized_chains_run(tmp_path: Path) -> None:
    """`chain_method: vectorized` gives the same output structure."""
    spec, data = _model(tmp_path, 200)
    res = MCMCEstimator().fit(spec, data, _cfg(chain_method="vectorized"), 2)
    assert res.posterior_samples["posterior"]["duration.rate"].shape[:2] == (2, 60)


def test_e13_t06a_ordered_parameters_get_a_plain_prior_mode(tmp_path: Path) -> None:
    """An `increasing` constraint (an identifiability guard) takes the default prior as a plain mode."""
    import dataclasses

    spec, data = _model(tmp_path, 200)
    ordered = tuple(
        dataclasses.replace(s, constraint="increasing", init=np.array([0.02, 0.05]))
        if s.name == "duration.rate"
        else s
        for s in spec.param_specs
    )
    spec = dataclasses.replace(spec, param_specs=ordered)
    start = ParamManager(spec.param_specs).init_u()
    cfg = _cfg(chains=1, warmup=40, draws=40).model_copy(update={"restarts": 0})
    res = MCMCEstimator().fit(spec, data, cfg, 0, start)
    rate = np.asarray(res.posterior_samples["posterior"]["duration.rate"])
    assert np.all(np.diff(rate, axis=-1) > 0)  # ordered in every draw
    assert res.diagnostics["mcmc"]["max_r_hat"] != res.diagnostics["mcmc"]["max_r_hat"]  # one chain: NaN


def test_e13_t06a_a_batch_gives_one_loglik_column_per_sequence(tmp_path: Path) -> None:
    """`log_likelihood` has a `sequence` dimension named after the batch ids; the values are finite."""
    from hsmmkit.data.batch import SequenceBatch

    spec, _ = _model(tmp_path)
    a, b = em_models.dataset("gaussian", 200, 1), em_models.dataset("gaussian", 150, 2)
    batch = SequenceBatch((a, b), (256,))
    res = MCMCEstimator().fit(spec, batch, _cfg(warmup=30, draws=30), 0)
    ll = res.posterior_samples["log_likelihood"]["loglik"]
    assert ll.shape == (2, 30, 2)
    assert list(ll["sequence"].values) == list(batch.ids)
    assert np.all(np.isfinite(np.asarray(ll)))


@pytest.mark.slow
def test_e13_t06a_two_regime_fixture_recovers_the_truth(tmp_path: Path) -> None:
    """AC1 (Appendix E, Gaussian form): T = 5000, 4 chains x 1000 draws: 2 posterior SD, R-hat, ESS."""
    spec, data = _model(tmp_path, 5000, 4)
    cfg = EstimationSection.model_validate({"method": "mcmc", "restarts": 1, "init": ["rolling_kmeans"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = MCMCEstimator().fit(spec, data, cfg, 0)
    info = res.diagnostics["mcmc"]
    assert info["max_r_hat"] <= 1.05
    assert info["min_ess_bulk"] >= 200
    truth = {
        "emission.mean": np.array([[0.0], [1.5]]),
        "emission.scale": np.array([[0.5], [1.2]]),
        "duration.rate": np.full(2, -np.log(1 - 1 / 25)),
    }
    assert res.std_errors is not None
    for name, value in truth.items():
        z = np.abs(np.asarray(res.params[name]) - value) / np.asarray(res.std_errors[name])
        assert np.all(z <= 2.0), (name, z)
