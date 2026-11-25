"""E9-T01 AC1: two-series synthetic recovery (shared regimes, sigma per series) and a default-suite smoke."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.config.schema import EstimationSection
from regimekit.data.contracts import PreparedDataset
from regimekit.estimation.map import fit
from regimekit.inference.simulate import simulate_dataset
from regimekit.model.build import build_model_spec
from regimekit.model.emission.product import ProductEmission
from regimekit.model.params import ParamManager
from regimekit.model.spec import ModelSpec
from regimekit.statespace.builders import build_expanded
from regimekit.utils.rng import key_for
from tests.helpers.engine_fixtures import DT, make_dataset
from tests.helpers.two_series import MEAN_SOJOURN_DAYS, MU_STAR, SIGMA_STAR, two_series_config

SIGMA_REL_TOL = 0.10  # overview 4.7: sigma <= 10 % relative
SEEDS = (0, 1, 2)
SIM_PURPOSE = 1  # RNG purpose id of the simulated path (tests only)


def _simulate(seed: int, t_len: int) -> tuple[ModelSpec, PreparedDataset]:
    """A two-series dataset drawn from the true model (shared regime path, independent emissions)."""
    stub = make_dataset(np.zeros((t_len, 2)), np.full(t_len, DT))
    spec = build_model_spec(two_series_config(), stub)
    assert isinstance(spec.emission, ProductEmission)
    manager = ParamManager(spec.param_specs)
    theta = dict(manager.unflatten(manager.init_u()))
    theta["emission.mean"], theta["emission.scale"] = jnp.asarray(MU_STAR), jnp.asarray(SIGMA_STAR)
    theta.update(spec.duration.init_from_mean_sojourn(jnp.asarray(MEAN_SOJOURN_DAYS), DT))
    data = simulate_dataset(
        key_for(seed, SIM_PURPOSE), theta, spec, build_expanded(spec), jnp.full((t_len,), DT), clock="trading"
    )
    return spec, data


def _fit_sigma(spec: ModelSpec, data: PreparedDataset, seed: int, restarts: int) -> np.ndarray:
    cfg = EstimationSection(restarts=restarts, max_iter=400, init=["rolling_kmeans", "lattice_jitter"])
    res = fit(spec, data, cfg, seed)
    assert np.isfinite(res.loglik)
    return np.asarray(res.params["emission.scale"])


def test_e9_t01_two_series_fit_runs_and_has_one_sigma_per_series_and_regime() -> None:
    """Default-suite smoke at T=1500: finite fit, `(N, M)` sigma, high-vol regime last in both series."""
    spec, data = _simulate(seed=0, t_len=1500)
    sigma = _fit_sigma(spec, data, seed=0, restarts=2)
    assert sigma.shape == (2, 2)
    assert (sigma[1] > sigma[0]).all()


def test_e9_t01_fit_with_scattered_per_series_gaps_stays_close_to_truth() -> None:
    """3 % of series 2's prints masked at random: the fit uses series 1 at those steps (sigma within 25 %)."""
    spec, data = _simulate(seed=1, t_len=6000)
    rng = np.random.default_rng(5)
    mask = np.ones((6000, 2), dtype=bool)
    mask[rng.random(6000) < 0.03, 1] = False
    holed = dataclasses.replace(data, mask=jnp.asarray(mask))
    sigma = _fit_sigma(spec, holed, seed=1, restarts=3)
    np.testing.assert_allclose(sigma, SIGMA_STAR, rtol=0.25)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_e9_t01_two_series_sigma_recovery_meets_the_appendix_d_tolerance(seed: int) -> None:
    """AC1: T=20000, shared regimes, sigma per series within 10 % relative in every one of the 3 seeds."""
    spec, data = _simulate(seed=seed, t_len=20000)
    sigma = _fit_sigma(spec, data, seed=seed, restarts=4)
    rel = np.abs(sigma - SIGMA_STAR) / SIGMA_STAR
    assert rel.max() <= SIGMA_REL_TOL, f"sigma {sigma.tolist()} vs {SIGMA_STAR.tolist()}"
