"""E13-T05b: VB-EM hybrid (point blocks by ECM), treatment table, fit-report fields."""

from __future__ import annotations

import dataclasses

import numpy as np

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation.vb import VBEstimator
from hsmmkit.model.emission.student_t import StudentTFamily
from hsmmkit.model.params.manager import ParamManager
from tests.helpers.engine_fixtures import make_scenario
from tests.integration.test_e13_t02b_ecm import _specs_of


def _hybrid_model(t_len: int = 400, seed: int = 4, **kw):  # noqa: ANN003, ANN202
    """3 regimes, transition covariates (point block), Student-t emissions (point block), NB durations."""
    sc = make_scenario(3, [2, 1, 2], 1, t_len, n_cov_pi=2, seed=seed, **kw)
    family = StudentTFamily(scale_by_dt=False)
    others = tuple(s for s in _specs_of(sc) if not s.name.startswith("emission."))
    specs = (*family.param_specs(3, 1, ()), *others)
    spec = dataclasses.replace(sc.spec, emission=family, param_specs=specs)
    rng = np.random.default_rng(seed)
    y = rng.standard_t(5, size=(t_len, 1)) * np.where(rng.random((t_len, 1)) < 0.5, 0.5, 1.5)
    data = dataclasses.replace(sc.data, observations=np.asarray(y))
    return spec, data


def test_e13_t05b_hybrid_runs_with_a_monotone_elbo_and_a_correct_treatment_table(monkeypatch) -> None:  # noqa: ANN001
    """AC2: transition covariates + Student-t emissions; duration stays conjugate; ELBO monotone (1e-8)."""
    monkeypatch.setenv("HSMMKIT_DEBUG", "1")
    spec, data = _hybrid_model()
    cfg = EstimationSection(method="vb", restarts=0, max_iter=40, tol_f=1e-12)
    res = VBEstimator().fit(spec, data, cfg, 0, warm_start=ParamManager(spec.param_specs).init_u())
    info = res.diagnostics["vb"]
    assert info["treatment"] == {"emission": "point", "duration": "conjugate", "transition": "point"}
    trace = np.asarray(res.restarts["elbo_trace"].iloc[0])
    assert len(trace) > 3
    assert np.all(np.diff(trace) >= -1e-8)
    assert trace[-1] > trace[0]
    assert set(res.params) >= {"emission.nu", "transition.b", "duration.rate"}
    assert res.std_errors is not None
    assert "duration.rate" in res.std_errors  # posterior SD of the conjugate duration block
    assert "emission.nu" not in res.std_errors  # point blocks have none
    assert res.pruned_states == ()


def test_e13_t05b_a_fixed_conjugate_parameter_turns_its_block_into_a_point_block(monkeypatch) -> None:  # noqa: ANN001
    """Blocks that a conjugate update cannot honour (here a `fixed` rate) fall back to ECM and stay fixed."""
    monkeypatch.setenv("HSMMKIT_DEBUG", "1")
    spec, data = _hybrid_model(300, 2)
    base = next(s for s in spec.param_specs if s.name == "duration.rate")
    specs = tuple(
        dataclasses.replace(s, mode="fixed", value=np.asarray(base.init)) if s.name == "duration.rate" else s
        for s in spec.param_specs
    )
    spec = dataclasses.replace(spec, param_specs=specs)
    cfg = EstimationSection(method="vb", restarts=0, max_iter=15, tol_f=1e-12)
    res = VBEstimator().fit(spec, data, cfg, 0, warm_start=ParamManager(spec.param_specs).init_u())
    assert res.diagnostics["vb"]["treatment"]["duration"] == "point"
    # the regimes may be renumbered; the fixed values themselves must be unchanged
    np.testing.assert_allclose(
        np.sort(np.asarray(res.params["duration.rate"])), np.sort(np.asarray(base.init))
    )


def test_e13_t05b_a_point_duration_with_a_conjugate_transition_is_monotone(monkeypatch) -> None:  # noqa: ANN001
    """Irregular steps make the duration a point block (ECM on per-step counts); jumps stay conjugate."""
    monkeypatch.setenv("HSMMKIT_DEBUG", "1")
    sc = make_scenario(3, [2, 1, 2], 1, 300, seed=6, irregular=True)
    spec = dataclasses.replace(sc.spec, param_specs=_specs_of(sc))
    rng = np.random.default_rng(6)
    data = dataclasses.replace(sc.data, observations=np.asarray(rng.normal(0.0, 0.01, (300, 1))))
    cfg = EstimationSection(method="vb", restarts=0, max_iter=25, tol_f=1e-12)
    res = VBEstimator().fit(spec, data, cfg, 0, warm_start=ParamManager(spec.param_specs).init_u())
    assert res.diagnostics["vb"]["treatment"]["duration"] == "point"
    assert res.diagnostics["vb"]["treatment"]["transition"] == "conjugate"
    trace = np.asarray(res.restarts["elbo_trace"].iloc[0])
    assert np.all(np.diff(trace) >= -1e-8)
    assert trace[-1] > trace[0]
