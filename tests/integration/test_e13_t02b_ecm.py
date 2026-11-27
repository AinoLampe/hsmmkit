"""E13-T02b: ECM blocks, per-step counts, `diagnostics["em"]`, `warm_start_from`, the new config keys."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.config.schema import EmCfg, EstimationSection, TemperingCfg
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.estep import expanded_estep
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.estimation.em import EMEstimator, run_em, tempering_schedule
from hsmmkit.estimation.mstep import block_plan
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.statespace.builders import build_expanded
from tests.helpers import em_models
from tests.helpers.engine_fixtures import make_scenario
from tests.helpers.objective_fixtures import make_case


def test_e13_t02b_per_step_counts_sum_to_the_totals() -> None:
    """`keep_per_t`: `stay_t`, `advance_t`, `jump_t` sum over steps and sequences to the totals."""
    sc = make_scenario(3, [2, 1, 2], 1, 80, n_cov_pi=2, n_cov_d=2, seed=3)
    pb = SequenceBatch((sc.data.window(0, 80), sc.data.window(10, 50)), (128,)).padded()
    plain = expanded_estep(ExpandedHMMEngine(), sc.theta, sc.spec, sc.space, pb)
    kept = expanded_estep(ExpandedHMMEngine(), sc.theta, sc.spec, sc.space, pb, keep_per_t=True)
    assert plain.stay_t is None
    assert kept.stay_t is not None
    assert kept.advance_t is not None
    assert kept.jump_t is not None
    np.testing.assert_allclose(np.asarray(kept.stay_t).sum(axis=(0, 1)), np.asarray(kept.stay), atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(kept.advance_t).sum(axis=(0, 1)), np.asarray(kept.advance), atol=1e-9
    )
    np.testing.assert_allclose(np.asarray(kept.jump_t).sum(axis=(0, 1)), np.asarray(kept.jump), atol=1e-9)
    assert np.all(np.asarray(kept.stay_t)[:, 0] == 0.0)  # row 0 is no transition
    np.testing.assert_allclose(np.asarray(kept.stay), np.asarray(plain.stay), rtol=1e-12)


def test_e13_t02b_the_plan_names_closed_form_and_ecm_blocks() -> None:
    """Covariate links, irregular steps and Student-t move their block to ECM; the others stay closed-form."""
    base = make_case("gbm", "negbin", 1, t_len=60)
    pb = SequenceBatch.of(base.data).padded()
    assert dict(block_plan(base.spec, pb)) == {
        "emission": "closed-form",
        "duration": "closed-form",
        "transition": "closed-form",
        "initial": "fixed",
    }
    cov = make_case("gbm", "negbin", 1, t_len=60, cov_em=True, cov_du=True)
    plan = dict(block_plan(cov.spec, SequenceBatch.of(cov.data).padded()))
    assert plan["emission"] == "ecm"
    assert plan["duration"] == "ecm"
    student = make_case("student_t", "negbin", 1, t_len=60)
    assert dict(block_plan(student.spec, SequenceBatch.of(student.data).padded()))["emission"] == "ecm"
    irregular = dataclasses.replace(base.data, dt=base.data.dt * jnp.asarray([1.0, 3.0] * 30))
    assert dict(block_plan(base.spec, SequenceBatch.of(irregular).padded()))["duration"] == "ecm"


def test_e13_t02b_fit_diagnostics_list_the_method_per_component(tmp_path: Path) -> None:
    """AC4: `FitResult.diagnostics["em"]` records closed-form / ECM per block and the tempering used."""
    spec, theta0 = em_models.build(tmp_path, "poisson")
    data = em_models.dataset("poisson", 200, 1)
    start = ParamManager(spec.param_specs).flatten(theta0)
    cfg = EstimationSection(restarts=0, max_iter=15, use_priors=False)
    result = EMEstimator().fit(spec, data, cfg, 0, warm_start=start)
    info = result.diagnostics["em"]
    assert info["emission"] == info["duration"] == info["transition"] == "closed-form"
    assert info["tempering"]["enabled"] is False
    case = make_case("student_t", "negbin", 1, t_len=120)
    res2 = EMEstimator().fit(
        case.spec, case.data, EstimationSection(restarts=1, init=["random"], max_iter=5), 0
    )
    assert res2.diagnostics["em"]["emission"] == "ecm"


def test_e13_t02b_tempering_schedule_and_monotonicity_after_it(tmp_path: Path) -> None:
    """`beta_k = beta0^(1 - k/steps)` for `steps` iterations; monotone once `beta = 1`."""
    em = EmCfg(tempering=TemperingCfg(enabled=True, beta0=0.3, steps=4))
    assert tempering_schedule(em, 100) == pytest.approx([0.3, 0.3**0.75, 0.3**0.5, 0.3**0.25])
    assert tempering_schedule(EmCfg(), 100) == []
    spec, theta0 = em_models.build(tmp_path, "gaussian")
    pb = SequenceBatch.of(em_models.dataset("gaussian", 400, 2)).padded()
    cfg = EstimationSection(max_iter=60, tol_f=1e-14, use_priors=False, em=em)
    _, trace, _, n_iter = run_em(theta0, spec, build_expanded(spec), pb, cfg, debug=True)
    after = np.asarray(trace[4:])  # the first `steps` entries ran at beta < 1
    assert n_iter > 10
    assert np.all(np.diff(after) >= -(1e-10 + 1e-13 * np.abs(after[:-1])))


def test_e13_t02b_warm_start_from_map_starts_at_the_map_optimum(tmp_path: Path) -> None:
    """`estimation.em.warm_start_from: map` runs EM from the MAP fit: one restart, labelled `map`."""
    spec, _ = em_models.build(tmp_path, "poisson")
    data = em_models.dataset("poisson", 300, 3)
    cfg = EstimationSection(
        restarts=1, init=["rolling_kmeans"], max_iter=30, use_priors=False, em=EmCfg(warm_start_from="map")
    )
    result = EMEstimator().fit(spec, data, cfg, 3)
    assert list(result.restarts["init_method"]) == ["map"]
    trace = result.restarts["loglik_trace"].iloc[0]
    assert np.all(np.diff(trace) >= -1e-10)


def test_e13_t02b_ecm_on_a_multi_sequence_batch_with_covariates() -> None:
    """ECM pools its per-step counts over a padded batch (covariate links in every block)."""
    sc = make_scenario(3, [2, 1, 2], 1, 120, n_cov_pi=2, n_cov_d=2, seed=5)
    batch = SequenceBatch((sc.data.window(0, 70), sc.data.window(40, 120), sc.data.window(30, 61)), (128,))
    spec = dataclasses.replace(sc.spec, param_specs=_specs_of(sc))
    cfg = EstimationSection(restarts=0, max_iter=8, tol_f=1e-14, use_priors=False)
    manager = ParamManager(spec.param_specs)
    result = EMEstimator().fit(spec, batch, cfg, 0, warm_start=manager.flatten(sc.theta))
    trace = np.asarray(result.restarts["loglik_trace"].iloc[0])
    assert np.all(np.diff(trace) >= -1e-10)
    assert trace[-1] > trace[0]
    info = result.diagnostics["em"]
    assert info["duration"] == "ecm"
    assert info["transition"] == "ecm"


def _specs_of(sc):  # noqa: ANN001, ANN202
    """`ParamSpec`s of a `make_scenario` model (the fixture leaves `param_specs` empty)."""
    from hsmmkit.model.params.spec import ParamSpec

    out = []
    for name, value in sc.theta.items():
        arr = np.asarray(value)
        domain = "positive" if name in ("emission.scale", "duration.rate") else "real"
        if name == "transition.pi_rows":
            domain = "simplex_ref"
        out.append(ParamSpec(name=name, shape=arr.shape, domain=domain, mode="free", init=arr))  # type: ignore[arg-type]
    return tuple(out)
