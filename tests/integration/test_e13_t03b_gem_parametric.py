"""E13-T03b: GEM with parametric durations (damped Newton), per-state mixes, Newton fallback."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation.gem import GEMEstimator
from hsmmkit.estimation.map import fit as map_fit
from hsmmkit.estimation.newton import newton_minimize
from hsmmkit.model.duration.base import coef_spec
from hsmmkit.model.params.manager import ParamManager
from tests.helpers import gem_models

_FAMILIES: dict[str, dict[str, Any]] = {
    "gamma": {
        "family": "gamma",
        "params": {
            "shape": {"mode": "free", "init": [2.0, 2.0]},
            "scale": {"mode": "free", "init": [2.5, 3.0]},
        },
    },
    "weibull": {
        "family": "weibull",
        "params": {
            "shape": {"mode": "free", "init": [1.5, 1.5]},
            "scale": {"mode": "free", "init": [5.0, 6.0]},
        },
    },
    "lognormal": {
        "family": "lognormal",
        "params": {
            "mu_log": {"mode": "free", "init": [1.2, 1.4]},
            "sd_log": {"mode": "free", "init": [0.5, 0.5]},
        },
    },
    "poisson": {"family": "poisson", "params": {"kappa": {"mode": "free", "init": [4.0, 6.0]}}},
}


def _data(t_len: int = 500, seed: int = 1):  # noqa: ANN202
    pmf = np.zeros((2, 25))
    d = np.arange(1, 26)
    pmf[0] = np.exp(-0.5 * ((d - 5) / 2.0) ** 2)
    pmf[1] = np.exp(-0.5 * ((d - 9) / 3.0) ** 2)
    pmf /= pmf.sum(axis=1, keepdims=True)
    pi = np.array([[0.0, 1.0], [1.0, 0.0]])
    y, _ = gem_models.simulate(t_len, pmf, pi, np.array([-1.0, 1.0]), np.array([0.6, 0.9]), seed)
    return gem_models.dataset(y)


def _spec(tmp_path: Path, name: str, **kw: Any):  # noqa: ANN202, ANN401
    return gem_models.build(
        tmp_path, 2, 25, duration_cfg=_FAMILIES[name], means=[-0.5, 0.5], scales=[0.8, 0.8], **kw
    )


@pytest.mark.parametrize("name", sorted(_FAMILIES))
def test_e13_t03b_parametric_gem_is_monotone(tmp_path: Path, name: str, monkeypatch) -> None:  # noqa: ANN001
    """AC1: the log-likelihood never decreases (violations <= 1e-10)."""
    monkeypatch.setenv("HSMMKIT_DEBUG", "1")
    spec = _spec(tmp_path, name)
    cfg = EstimationSection(method="gem", restarts=0, max_iter=40, tol_f=1e-14, use_priors=False)
    res = GEMEstimator().fit(spec, _data(), cfg, 0, warm_start=ParamManager(spec.param_specs).init_u())
    trace = np.asarray(res.restarts["loglik_trace"].iloc[0])
    assert trace[-1] > trace[0]
    assert np.all(np.diff(trace[:-1]) >= -(1e-10 + 1e-13 * np.abs(trace[:-2])))
    assert res.diagnostics["gem"]["duration"].endswith("(newton)")


def test_e13_t03b_entry_time_covariates_are_monotone(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """AC1 (covariates): Newton on `(phi, c)` with the per-start segment counts."""
    monkeypatch.setenv("HSMMKIT_DEBUG", "1")
    from tests.helpers.engine_fixtures import make_dataset

    t_len = 400
    rng = np.random.default_rng(4)
    z = np.cumsum(rng.normal(size=(t_len, 1)) * 0.2, axis=0)
    base, _ = gem_models.simulate(
        t_len,
        np.full((2, 25), 1.0 / 25),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([-1.0, 1.0]),
        np.array([0.6, 0.9]),
        2,
    )
    data = make_dataset(base, np.ones(t_len), z_d=z)
    spec = _spec(tmp_path, "gamma")
    gamma = dataclasses.replace(spec.duration, cov_idx=(0,))
    specs = (*spec.param_specs, coef_spec("duration.c", 2, 1))
    spec = dataclasses.replace(spec, duration=gamma, param_specs=specs)
    cfg = EstimationSection(method="gem", restarts=0, max_iter=15, tol_f=1e-14, use_priors=False)
    res = GEMEstimator().fit(spec, data, cfg, 0, warm_start=ParamManager(spec.param_specs).init_u())
    trace = np.asarray(res.restarts["loglik_trace"].iloc[0])
    assert "duration.c" in res.params
    assert trace[-1] > trace[0]
    assert np.all(np.diff(trace[:-1]) >= -(1e-10 + 1e-13 * np.abs(trace[:-2])))


def test_e13_t03b_gamma_parity_with_map(tmp_path: Path) -> None:
    """AC2: no priors, same start: GEM and MAP reach one optimum (theta rel 1e-4, loglik rel 1e-6)."""
    spec = _spec(tmp_path, "gamma")
    data = _data(900, 3)
    start = ParamManager(spec.param_specs).init_u()
    gem = GEMEstimator().fit(
        spec,
        data,
        EstimationSection(restarts=0, max_iter=6000, tol_f=1e-15, use_priors=False),
        0,
        warm_start=start,
    )
    mp = map_fit(
        spec,
        data,
        EstimationSection(restarts=0, max_iter=3000, tol_f=1e-15, tol_g=1e-9, use_priors=False),
        0,
        warm_start=start,
    )
    assert abs(gem.loglik - mp.loglik) <= 1e-6 * abs(mp.loglik)
    for name, value in mp.params.items():
        np.testing.assert_allclose(
            np.asarray(gem.params[name]), np.asarray(value), rtol=1e-4, atol=1e-6, err_msg=name
        )
    # from the MAP optimum GEM stays
    stay = GEMEstimator().fit(
        spec,
        data,
        EstimationSection(restarts=0, max_iter=50, tol_f=1e-15, use_priors=False),
        0,
        warm_start=mp.uparams,
    )
    for name, value in mp.params.items():
        np.testing.assert_allclose(
            np.asarray(stay.params[name]), np.asarray(value), rtol=1e-4, atol=1e-6, err_msg=name
        )


def test_e13_t03b_newton_falls_back_to_bfgs_and_still_improves() -> None:
    """AC4: Newton cannot move on an ill-conditioned problem; the BFGS fallback still lowers the objective."""
    scales = jnp.asarray([1.0, 1e4])

    def fun(u: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(0.5 * scales * (u - 1.0) ** 2)

    u0 = jnp.asarray([-3.0, 2.0])
    _, f, used = newton_minimize(fun, u0, lm=1e30)  # damping so large that Newton cannot move
    assert bool(used)
    assert float(f) < float(fun(u0))
    _, f2, used2 = newton_minimize(fun, u0)  # the default: Newton alone solves the quadratic
    assert not bool(used2)
    assert float(f2) <= float(f) + 1e-12


def test_e13_t03b_per_state_mix_runs_and_is_monotone(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Each state's family is updated on its own rows (gamma / geometric here)."""
    monkeypatch.setenv("HSMMKIT_DEBUG", "1")
    cfg_dur = {
        "per_state": [
            {
                "family": "gamma",
                "params": {"shape": {"mode": "free", "init": 2.0}, "scale": {"mode": "free", "init": 2.5}},
            },
            {"family": "geometric", "params": {"rate": {"mode": "free", "init_mean_sojourn_steps": 8.0}}},
        ]
    }
    spec = gem_models.build(tmp_path, 2, 25, duration_cfg=cfg_dur, means=[-0.5, 0.5], scales=[0.8, 0.8])
    cfg = EstimationSection(method="gem", restarts=0, max_iter=30, tol_f=1e-14, use_priors=False)
    res = GEMEstimator().fit(spec, _data(), cfg, 0, warm_start=ParamManager(spec.param_specs).init_u())
    trace = np.asarray(res.restarts["loglik_trace"].iloc[0])
    assert trace[-1] > trace[0]
    assert np.all(np.diff(trace[:-1]) >= -(1e-10 + 1e-13 * np.abs(trace[:-2])))
