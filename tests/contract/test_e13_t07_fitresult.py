"""E13-T07 AC2: every estimator returns a uniform `FitResult`, and its bundle round-trips."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation import fit as dispatch_fit
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.persistence.store import ModelStore
from tests.helpers import em_models, gem_models

_METHODS = ["map", "em", "gem", "vb", "mcmc"]


def _setup(tmp_path: Path, method: str) -> tuple[Any, Any, Any, EstimationSection]:
    """`(cfg, spec, data, estimation section)` of a small model every estimator can fit."""
    mcmc = {"chains": 2, "warmup": 40, "draws": 40}
    base: dict[str, Any] = {
        "method": method,
        "restarts": 1,
        "init": ["rolling_kmeans"],
        "max_iter": 25,
        "mcmc": mcmc,
    }
    if method == "gem":
        cfg, spec = gem_models.build(tmp_path, 2, 12, duration="geometric", return_cfg=True)
        pmf = np.zeros((2, 12))
        pmf[:, :6] = 1.0 / 6.0
        y, _ = gem_models.simulate(
            400, pmf, np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([-1.0, 1.0]), np.array([0.6, 0.9]), 1
        )
        return cfg, spec, gem_models.dataset(y), EstimationSection.model_validate(base)
    cfg, spec, _ = em_models.build_with_cfg(tmp_path, "gaussian", "geometric")
    return cfg, spec, em_models.dataset("gaussian", 400, 3), EstimationSection.model_validate(base)


@pytest.mark.parametrize("method", _METHODS)
def test_e13_t07_fitresult_is_uniform_and_the_bundle_round_trips(tmp_path: Path, method: str) -> None:
    """Fields, constrained params, `uparams = flatten(params)`, `elbo` iff VB, samples iff MCMC, bundles."""
    if method == "mcmc":
        pytest.importorskip("numpyro")
        pytest.importorskip("arviz")
    cfg, spec, data, est = _setup(tmp_path, method)
    result = dispatch_fit(spec, data, est, 0)
    assert result.estimator == method
    assert np.isfinite(result.loglik)
    assert isinstance(result.pruned_states, tuple)
    assert (result.elbo is not None) == (method == "vb")
    assert (result.posterior_samples is not None) == (method == "mcmc")
    names = {s.name for s in result.spec.param_specs}
    assert set(result.params) == names
    for name in ("emission.scale", "duration.rate"):  # positive parameters are on the constrained scale
        assert np.all(np.asarray(result.params[name]) > 0)
    manager = ParamManager(result.spec.param_specs)
    np.testing.assert_allclose(
        np.asarray(result.uparams), np.asarray(manager.flatten(result.params)), atol=1e-10
    )
    bundle = tmp_path / f"{method}.hsmm"
    ModelStore.save(result, bundle, cfg, data)
    again = ModelStore.load(bundle)
    assert again.estimator == method
    for name, value in result.params.items():
        np.testing.assert_array_equal(np.asarray(again.params[name]), np.asarray(value))
    assert again.loglik == pytest.approx(result.loglik, rel=0, abs=0)
