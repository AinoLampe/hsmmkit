"""E13-T02a: `EMEstimator` parity with MAP (AC2), batches / topology / labels (AC4), the registry."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.estimation import fit as dispatch_fit
from hsmmkit.estimation.em import EMEstimator
from hsmmkit.estimation.map import fit as map_fit
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.plugins import REGISTRY
from hsmmkit.statespace.builders import build_expanded
from hsmmkit.utils.errors import ConfigError
from tests.golden.test_e12_t03_absorbing import _model
from tests.helpers import em_models


def _same_start(spec: Any, theta0: dict[str, jnp.ndarray]) -> jnp.ndarray:  # noqa: ANN401
    return ParamManager(spec.param_specs).flatten(theta0)


@pytest.mark.parametrize(
    ("family", "duration"),
    [("gaussian", "geometric"), ("gaussian", "negbin"), ("poisson", "geometric"), ("categorical", "negbin")],
)
def test_e13_t02a_em_agrees_with_map_from_the_same_start(tmp_path: Path, family: str, duration: str) -> None:
    """AC2: no priors, same start: parameters rel <= 1e-4, loglik rel <= 1e-6 (the same optimum)."""
    spec, theta0 = em_models.build(tmp_path, family, duration)
    data = em_models.dataset(family, 1200, 4)
    start = _same_start(spec, theta0)
    em_cfg = EstimationSection(restarts=0, max_iter=20000, tol_f=1e-15, use_priors=False)
    map_cfg = EstimationSection(restarts=0, max_iter=3000, tol_f=1e-15, tol_g=1e-9, use_priors=False)
    em = EMEstimator().fit(spec, data, em_cfg, 0, warm_start=start)
    mp = map_fit(spec, data, map_cfg, 0, warm_start=start)
    assert em.estimator == "em"
    assert abs(em.loglik - mp.loglik) <= 1e-6 * abs(mp.loglik)
    for name, value in mp.params.items():
        np.testing.assert_allclose(
            np.asarray(em.params[name]), np.asarray(value), rtol=1e-4, atol=1e-6, err_msg=name
        )


def test_e13_t02a_the_method_key_selects_the_estimator(tmp_path: Path) -> None:
    """`estimation.fit` dispatches on `cfg.method`; the registry lists both estimators."""
    assert {"map", "em"} <= set(REGISTRY.names("estimator"))
    spec, _ = em_models.build(tmp_path, "poisson")
    data = em_models.dataset("poisson", 300, 1)
    cfg = EstimationSection(method="em", restarts=1, max_iter=30, init=["rolling_kmeans"], use_priors=False)
    result = dispatch_fit(spec, data, cfg, 1)
    assert result.estimator == "em"
    assert list(result.restarts.columns)[:7] == [
        "restart", "init_method", "objective", "loglik", "n_iter", "converged", "wall_s"
    ]  # fmt: skip
    trace = result.restarts["loglik_trace"].iloc[0]
    assert len(trace) >= 2
    assert trace[-1] >= trace[0]
    assert result.uparams.shape == (ParamManager(spec.param_specs).n_free,)


def test_e13_t02a_tie_groups_across_blocks_are_refused(tmp_path: Path) -> None:
    """A tie group linking the emission and the duration block cannot be updated one block at a time."""
    spec, _ = em_models.build(tmp_path, "gaussian")
    tied = tuple(
        dataclasses.replace(s, tie_group="g") if s.name in ("emission.scale", "duration.rate") else s
        for s in spec.param_specs
    )
    with pytest.raises(ConfigError, match="tie groups"):
        EMEstimator().fit(
            dataclasses.replace(spec, param_specs=tied),
            em_models.dataset("gaussian", 50, 0),
            EstimationSection(restarts=1, init=["random"]),
            0,
        )


def test_e13_t02a_batches_and_fixed_parameters(tmp_path: Path) -> None:
    """A padded batch is fitted (pooled counts); `fixed` entries keep their value."""
    spec, theta0 = em_models.build(tmp_path, "poisson")
    seqs = tuple(em_models.dataset("poisson", n, seed) for n, seed in ((300, 1), (200, 2), (120, 3)))
    batch = SequenceBatch(seqs, (128, 256, 512))
    cfg = EstimationSection(restarts=0, max_iter=40, tol_f=1e-12, use_priors=False)
    result = EMEstimator().fit(spec, batch, cfg, 0, warm_start=_same_start(spec, theta0))
    table = result.diagnostics["per_sequence"]
    assert [row["T"] for row in table] == [300, 200, 120]
    assert result.loglik == pytest.approx(sum(row["loglik"] for row in table))
    trace = result.restarts["loglik_trace"].iloc[0]
    assert np.all(np.diff(trace) >= -1e-9)
    assert trace[-1] > trace[0]
    # a fixed emission rate is not touched
    fixed = tuple(
        dataclasses.replace(s, mode="fixed", value=np.asarray(theta0["emission.eta"]))
        if s.name == "emission.eta"
        else s
        for s in spec.param_specs
    )
    spec_fixed = dataclasses.replace(spec, param_specs=fixed)
    res = EMEstimator().fit(spec_fixed, seqs[0], cfg, 0, warm_start=ParamManager(fixed).flatten(theta0))
    np.testing.assert_allclose(np.asarray(res.params["emission.eta"]), np.asarray(theta0["emission.eta"]))


def test_e13_t02a_topology_masks_and_labels(tmp_path: Path) -> None:
    """A left-to-right topology keeps forbidden jumps at probability 0; labels hold; EM is monotone."""
    _, data, spec = _model(tmp_path, topology={"matrix": [[0, 1, 1], [0, 0, 1], [1, 0, 0]]})
    manager = ParamManager(spec.param_specs)
    theta0 = dict(manager.unflatten(manager.init_u()))
    labels = np.ones((data.observations.shape[0], 3), dtype=bool)
    labels[10:20] = [True, False, False]
    labelled = dataclasses.replace(data, labels=jnp.asarray(labels))
    cfg = EstimationSection(restarts=0, max_iter=25, tol_f=1e-14, use_priors=False)
    for d in (data, labelled):
        res = EMEstimator().fit(spec, d, cfg, 0, warm_start=manager.flatten(theta0))
        trace = res.restarts["loglik_trace"].iloc[0]
        assert np.all(np.diff(trace) >= -(1e-10 + 1e-13 * np.abs(trace[:-1])))
        layout = build_expanded(spec).layout
        pi = np.asarray(spec.transition.jump_probs_t(res.params, None, layout))  # type: ignore[union-attr]
        assert np.all(pi[~layout.target_mask] == 0.0)


def test_e13_t02a_the_em_gate_row_is_removed(tmp_path: Path) -> None:
    """`estimation.method: em` builds; `gem` was ungated by E13-T03a, `vb` stays gated."""
    from hsmmkit.config import gates
    from hsmmkit.config.schema import parse_project_config
    from tests.helpers.core_config import core_raw

    raw = core_raw(tmp_path)
    raw["estimation"]["method"] = "em"
    gates.check(parse_project_config(raw))
    assert "estimation.method=em" not in gates.GATES
    raw["estimation"]["method"] = "vb"
    with pytest.raises(ConfigError, match="E13-T05a"):
        gates.check(parse_project_config(raw))


def test_e13_t02a_stationary_initial_law_is_refused(tmp_path: Path) -> None:
    """Review: `initial_state: stationary` has no EM update (monotonicity would not hold): refused."""
    spec, _ = em_models.build(tmp_path, "poisson", initial_state="stationary")
    with pytest.raises(ConfigError, match="stationary"):
        EMEstimator().fit(
            spec, em_models.dataset("poisson", 60, 0), EstimationSection(restarts=1, init=["random"]), 0
        )


def test_e13_t02a_dirichlet_pseudo_counts_reach_the_default_jump_rows(tmp_path: Path) -> None:
    """Review: a Dirichlet prior on `transition.pi_rows` adds `alpha - 1` to the counts under MAP-EM."""
    from hsmmkit.estimation.mstep import transition_mstep
    from hsmmkit.model.params.priors import DirichletPrior

    _, _, spec = _model(tmp_path)
    specs = tuple(
        dataclasses.replace(s, mode="prior", prior=DirichletPrior(alpha=3.0))
        if s.name == "transition.pi_rows"
        else s
        for s in spec.param_specs
    )
    spec = dataclasses.replace(spec, param_specs=specs)
    space = build_expanded(spec)
    theta = dict(ParamManager(spec.param_specs).unflatten(ParamManager(spec.param_specs).init_u()))
    jump = np.zeros((3, 3))
    jump[0, 1], jump[0, 2] = 8.0, 2.0
    jump[1, 0], jump[1, 2] = 5.0, 5.0
    jump[2, 0], jump[2, 1] = 1.0, 9.0
    stats = type("S", (), {"jump": jnp.asarray(jump)})()
    ml = np.asarray(transition_mstep(spec, space, theta, stats, False)["transition.pi_rows"])  # type: ignore[arg-type]
    map_ = np.asarray(transition_mstep(spec, space, theta, stats, True)["transition.pi_rows"])  # type: ignore[arg-type]
    np.testing.assert_allclose(ml[0], [0.8, 0.2])
    np.testing.assert_allclose(map_[0], [10.0 / 14.0, 4.0 / 14.0])  # counts + (alpha - 1) = + 2
    np.testing.assert_allclose(map_.sum(axis=1), 1.0)
