"""E5-T04: `estimation/report.py` diagnostics (`diagnose`) -- AC1, AC3."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from regimekit.data.contracts import PreparedDataset
from regimekit.engine.base import InferenceEngine
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.estimation.objective import make_objective
from regimekit.estimation.report import diagnose, std_errors
from regimekit.estimation.results import FitResult
from regimekit.inference.simulate import simulate_dataset
from regimekit.model.duration.geometric import GeometricDuration
from regimekit.model.emission.gbm import GBMFamily
from regimekit.model.params.manager import ParamManager, Params
from regimekit.model.spec import ModelSpec
from regimekit.model.transition import JumpTransition
from regimekit.statespace.builders import build_expanded
from regimekit.statespace.index import StateSpace
from regimekit.utils.rng import key_for
from regimekit.utils.typing import Array
from tests.helpers.engine_fixtures import DT, make_dataset
from tests.helpers.objective_fixtures import make_case


def _fake_fit(
    spec: ModelSpec, theta: Params, uparams: Array, restarts_rows: list[dict[str, float]]
) -> FitResult:
    rows = [{"wall_s": 0.0, **row} for row in restarts_rows]
    return FitResult(
        estimator="map",
        params=theta,
        uparams=uparams,
        objective=float(restarts_rows[0]["objective"]),
        loglik=0.0,
        restarts=pd.DataFrame(rows),
        converged=True,
        n_iter=0,
        std_errors=None,
        diagnostics={},
        spec=spec,
        prep_stats=None,  # type: ignore[arg-type]
        config_resolved={},
    )


def _four_state_case(
    t_len: int = 200,
) -> tuple[ModelSpec, StateSpace, InferenceEngine, PreparedDataset, Params, Array]:
    """A 4-state GBM/geometric model where states 2, 3 fit the (2-state-like) data very poorly."""
    n = 4
    emis = GBMFamily()
    dur = GeometricDuration()
    trans = JumpTransition()
    emis_specs = list(emis.param_specs(n, 1, ()))
    dur_specs = list(dur.param_specs(n, ()))
    trans_specs = list(trans.param_specs(n, 1, ()))

    mu = np.array([[0.0], [0.0], [10.0], [-10.0]])
    sigma = np.array([[0.15], [0.15], [5.0], [5.0]])
    emis_specs = [dataclasses.replace(s, init=mu) if s.name == "emission.mu" else s for s in emis_specs]
    emis_specs = [dataclasses.replace(s, init=sigma) if s.name == "emission.sigma" else s for s in emis_specs]

    spec = ModelSpec(
        n_states=n,
        n_series=1,
        emission=emis,
        duration=dur,
        transition=trans,
        order=1,
        duration_conditioning="current",
        engine="expanded",
        d_max=10,
        initial="uniform",
        param_specs=tuple(emis_specs + dur_specs + trans_specs),
    )
    space = build_expanded(spec)
    manager = ParamManager(spec.param_specs)
    u0 = manager.init_u()
    theta = manager.unflatten(u0)

    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.01, size=(t_len, 1))
    data = make_dataset(y, np.full(t_len, DT))
    return spec, space, ExpandedHMMEngine(), data, theta, u0


def test_e5_t04_occupancy_warning_on_a_4_state_fit_of_2_state_data() -> None:
    """AC1: a 4-state fit on 2-state-like data triggers the occupancy warning for the unused states."""
    spec, space, engine, data, theta, u0 = _four_state_case()
    fit = _fake_fit(spec, theta, u0, [{"objective": 1.0, "loglik": -1.0}])

    diag = diagnose(fit, spec, space, engine, data)

    occupancy = diag["regime_occupancy"]["value"]
    assert occupancy[2] < 0.02
    assert occupancy[3] < 0.02
    assert diag["regime_occupancy"]["warning"] is not None


def test_e5_t04_restart_dispersion_warns_on_a_2_percent_gap() -> None:
    """AC1: two restarts with objectives differing by 2% trigger the dispersion warning."""
    spec, space, engine, data, theta, u0 = _four_state_case(t_len=60)
    rows = [
        {"objective": -100.0, "loglik": 100.0},
        {"objective": -98.0, "loglik": 98.0},
        {"objective": -90.0, "loglik": 90.0},
    ]
    fit = _fake_fit(spec, theta, u0, rows)

    diag = diagnose(fit, spec, space, engine, data)

    assert diag["restart_dispersion"]["value"] == pytest.approx(0.02, abs=1e-9)
    assert diag["restart_dispersion"]["warning"] is not None


def test_e5_t04_restart_dispersion_silent_on_a_tiny_gap() -> None:
    """No dispersion warning when the best two restarts are within 1%."""
    spec, space, engine, data, theta, u0 = _four_state_case(t_len=60)
    rows = [{"objective": -100.0, "loglik": 100.0}, {"objective": -100.5, "loglik": 100.5}]
    fit = _fake_fit(spec, theta, u0, rows)

    diag = diagnose(fit, spec, space, engine, data)

    assert diag["restart_dispersion"]["warning"] is None


def test_e5_t04_ordering_check_flags_shuffled_sigma() -> None:
    """AC1: a shuffled (non-ascending) sigma sets `ordering_check=False` and flags label-switching risk."""
    spec, space, engine, data, theta, u0 = _four_state_case(t_len=60)
    theta = dict(theta)
    theta["emission.sigma"] = jnp.asarray(theta["emission.sigma"])[::-1]

    fit = _fake_fit(spec, theta, u0, [{"objective": 1.0, "loglik": -1.0}])
    diag = diagnose(fit, spec, space, engine, data)

    assert diag["ordering_check"]["value"] is False
    assert diag["label_switching_risk"]["value"] is True


def test_e5_t04_pit_true_model_passes_wrong_sigma_fails() -> None:
    """AC1: the true model's PIT passes (p > 0.05); a wrong-sigma model's PIT fails (p < 0.01)."""
    case = make_case("gbm", "geometric", 1, t_len=3000, seed=5)
    theta_true = dict(case.manager.unflatten(case.u0))
    # PIT needs data actually generated by `theta_true` (independent noise, as `make_case`'s own
    # fixture data is, would fail PIT for *any* theta -- it was never fit to that data).
    data = simulate_dataset(key_for(0, 1), theta_true, case.spec, case.space, case.data.dt, clock="trading")

    fit_true = _fake_fit(case.spec, theta_true, case.u0, [{"objective": 1.0, "loglik": -1.0}])
    diag_true = diagnose(fit_true, case.spec, case.space, case.engine, data)
    assert diag_true["pit_ks_pvalue"]["value"] > 0.05

    theta_wrong = dict(theta_true)
    theta_wrong["emission.sigma"] = jnp.asarray(theta_true["emission.sigma"]) * 5.0
    fit_wrong = _fake_fit(case.spec, theta_wrong, case.u0, [{"objective": 1.0, "loglik": -1.0}])
    diag_wrong = diagnose(fit_wrong, case.spec, case.space, case.engine, data)
    assert diag_wrong["pit_ks_pvalue"]["value"] < 0.01


def test_e5_t04_diagnose_handles_masked_observations_without_nan() -> None:
    """A dataset with some masked (unobserved) rows must not leak `NaN` into the PIT/Ljung-Box entries."""
    case = make_case("gbm", "geometric", 1, t_len=500, seed=7)
    theta = dict(case.manager.unflatten(case.u0))
    data = simulate_dataset(key_for(0, 1), theta, case.spec, case.space, case.data.dt, clock="trading")
    mask = np.asarray(data.mask).copy()
    mask[10::37] = False  # scatter some missing observations through the series
    y = np.asarray(data.observations).copy()
    y[~mask] = np.nan  # masked rows carry NaN in the raw array, matching real prepared data
    data = dataclasses.replace(data, observations=jnp.asarray(y), mask=jnp.asarray(mask))

    fit = _fake_fit(case.spec, theta, case.u0, [{"objective": 1.0, "loglik": -1.0}])
    diag = diagnose(fit, case.spec, case.space, case.engine, data)

    assert np.isfinite(diag["pit_ks_statistic"]["value"])
    assert np.isfinite(diag["pit_ks_pvalue"]["value"])
    for _lag, (q_stat, p_value) in diag["ljung_box"]["value"].items():
        assert np.isfinite(q_stat)
        assert np.isfinite(p_value)
    assert not np.any(np.isnan(diag["pit_hist"]["value"]))


def test_e5_t04_non_pd_hessian_returns_none_with_no_raise() -> None:
    """AC3: a crafted non-PD Hessian returns `None` instead of raising."""
    case = make_case("gbm", "geometric", 1, t_len=20, seed=6)
    objective = make_objective(case.spec, case.space, case.engine, case.manager, case.data, use_priors=False)
    # far from the optimum: a short, noisy series with a small T makes the Hessian at u0 indefinite.
    se = std_errors(objective, case.u0, case.manager)
    assert se is None
