"""E13-T05a: `VBEstimator`: ELBO monotonicity, agreement with MAP, the evidence bound, refusals."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from hsmmkit.config.schema import EstimationSection
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.estimation import fit as dispatch_fit
from hsmmkit.estimation.map import fit as map_fit
from hsmmkit.estimation.vb import VBEstimator
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.plugins import REGISTRY
from hsmmkit.statespace.builders import build_expanded
from hsmmkit.utils.errors import ConfigError
from tests.helpers import em_models


def _start(spec, theta0):  # noqa: ANN001, ANN202
    return ParamManager(spec.param_specs).flatten(theta0)


@pytest.mark.parametrize("family", ["gaussian", "gaussian_full", "poisson", "categorical"])
@pytest.mark.parametrize("duration", ["geometric", "negbin"])
def test_e13_t05a_elbo_is_monotone(tmp_path: Path, family: str, duration: str, monkeypatch) -> None:  # noqa: ANN001
    """AC1: ELBO violations <= 1e-8 (debug mode raises on a decrease)."""
    monkeypatch.setenv("HSMMKIT_DEBUG", "1")
    spec, theta0 = em_models.build(tmp_path, family, duration)
    data = em_models.dataset(family, 500, 2)
    cfg = EstimationSection(method="vb", restarts=0, max_iter=80, tol_f=1e-13)
    res = VBEstimator().fit(spec, data, cfg, 0, warm_start=_start(spec, theta0))
    trace = np.asarray(res.restarts["elbo_trace"].iloc[0])
    assert res.estimator == "vb"
    assert len(trace) > 3
    assert np.all(np.diff(trace) >= -1e-8)
    assert trace[-1] > trace[0]
    info = res.diagnostics["vb"]
    assert set(info) >= {"elbo_trace", "q_params"}
    assert res.objective == pytest.approx(-trace[-1])
    assert np.isfinite(res.loglik)


def test_e13_t05a_posterior_means_agree_with_map_for_flat_priors(tmp_path: Path) -> None:
    """AC3: T = 20 000, diagonal Gaussian: posterior means within 2 % of the MAP estimates."""
    spec, theta0 = em_models.build(tmp_path, "gaussian", "geometric")
    data = em_models.dataset("gaussian", 20_000, 5)
    start = _start(spec, theta0)
    vb = VBEstimator().fit(
        spec, data, EstimationSection(method="vb", restarts=0, max_iter=200, tol_f=1e-12), 0, warm_start=start
    )
    mp = map_fit(
        spec,
        data,
        EstimationSection(restarts=0, max_iter=3000, tol_f=1e-14, tol_g=1e-9, use_priors=False),
        0,
        warm_start=start,
    )
    for name in ("emission.mean", "emission.scale", "duration.rate"):
        np.testing.assert_allclose(
            np.asarray(vb.params[name]), np.asarray(mp.params[name]), rtol=0.02, err_msg=name
        )


def test_e13_t05a_the_elbo_lower_bounds_the_log_evidence(tmp_path: Path) -> None:
    """AC4: T = 10 Poisson / geometric model: ELBO <= log p(y) (prior importance sampling, 4-D parameter)."""
    spec, theta0 = em_models.build(tmp_path, "poisson", "geometric")
    data = em_models.dataset("poisson", 10, 7)
    res = VBEstimator().fit(
        spec,
        data,
        EstimationSection(method="vb", restarts=0, max_iter=200, tol_f=1e-14),
        0,
        warm_start=_start(spec, theta0),
    )
    elbo = -res.objective
    space = build_expanded(spec)
    engine = ExpandedHMMEngine()
    n = 200_000
    rng = np.random.default_rng(0)
    a = rng.uniform(size=(n, 2))  # Beta(1, 1) advance probabilities
    lam = rng.gamma(1.0, 1.0, size=(n, 2))  # Gamma(1, 1) rates

    @jax.jit
    def one(a_i: jnp.ndarray, lam_i: jnp.ndarray) -> jnp.ndarray:
        th = dict(theta0)
        th["duration.rate"] = -jnp.log1p(-a_i)
        th["emission.eta"] = jnp.log(lam_i)[:, None]
        return engine.loglik(th, spec, space, data)

    ll = np.asarray(jax.vmap(one)(jnp.asarray(a), jnp.asarray(lam)))
    log_ev = special.logsumexp(ll) - np.log(n)
    w = np.exp(ll - ll.max())
    se = np.sqrt(np.var(w / w.mean()) / n)  # relative SE of the estimate
    assert elbo <= log_ev + 3 * se + 1e-9
    assert elbo > log_ev - 3.0  # and it is not vacuous


def test_e13_t05a_refusals_and_registration(tmp_path: Path) -> None:
    """`vb` is registered; priors are required; non-conjugate emissions are refused."""
    assert REGISTRY.get("estimator", "vb").name == "vb"
    spec, theta0 = em_models.build(tmp_path, "poisson", "geometric")
    data = em_models.dataset("poisson", 100, 1)
    start = _start(spec, theta0)
    with pytest.raises(ConfigError, match="use_priors"):
        VBEstimator().fit(
            spec, data, EstimationSection(method="vb", restarts=0, use_priors=False), 0, warm_start=start
        )
    res = dispatch_fit(spec, data, EstimationSection(method="vb", restarts=0, max_iter=5), 0, start)
    assert res.estimator == "vb"
    spec_t, theta_t = em_models.build(tmp_path, "student_t", "geometric")
    hybrid = VBEstimator().fit(
        spec_t,
        em_models.dataset("student_t", 100, 1),
        EstimationSection(method="vb", restarts=0, max_iter=5),
        0,
        warm_start=_start(spec_t, theta_t),
    )
    assert hybrid.diagnostics["vb"]["treatment"]["emission"] == "point"  # E13-T05b: no longer refused


def test_e13_t05a_multi_start_picks_the_best_elbo_and_batches_work(tmp_path: Path) -> None:
    """`vb` is registered; priors are required; non-conjugate emissions are refused."""
    spec, _ = em_models.build(tmp_path, "gaussian", "geometric")
    a, b = em_models.dataset("gaussian", 300, 1), em_models.dataset("gaussian", 250, 2)
    cfg = EstimationSection(
        method="vb", restarts=3, init=["rolling_kmeans", "random"], max_iter=60, tol_f=1e-10
    )
    res = VBEstimator().fit(spec, SequenceBatch((a, b), (512,)), cfg, 4)
    elbos = [-float(o) for o in res.restarts["objective"]]
    assert len(elbos) == 3
    assert -res.objective == pytest.approx(max(elbos))


def test_e13_t05a_the_elbo_bounds_the_evidence_with_three_regimes(tmp_path: Path) -> None:
    """N = 3: the Dirichlet jump block (two allowed targets per row) enters the ELBO and the weights."""
    from hsmmkit.config.schema import parse_project_config
    from hsmmkit.model.build import BuildInputs, build_model_spec_from
    from tests.helpers.core_config import core_raw

    raw = core_raw(tmp_path)
    m = raw["model"]
    m["n_states"] = 3
    m["emission"] = {
        "family": "poisson",
        "scale_by_dt": False,
        "params": {"rate": {"mode": "free", "init": [2.0, 5.0, 9.0]}},
    }
    m["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [4.0, 4.0, 4.0]
    spec = build_model_spec_from(
        parse_project_config(raw), BuildInputs(n_series=1, cov_names={}, clock="steps", nominal_dt=1.0)
    )
    manager = ParamManager(spec.param_specs)
    theta0 = dict(manager.unflatten(manager.init_u()))
    rng = np.random.default_rng(3)
    y = rng.poisson([2, 2, 6, 6, 9, 9, 3]).astype(float)[:, None]
    from tests.helpers.engine_fixtures import make_dataset

    data = make_dataset(y, np.ones(7))
    cfg = EstimationSection(method="vb", restarts=0, max_iter=300, tol_f=1e-14)
    res = VBEstimator().fit(spec, data, cfg, 0, warm_start=manager.flatten(theta0))
    elbo = -res.objective
    space, engine, n = build_expanded(spec), ExpandedHMMEngine(), 300_000
    a = rng.uniform(size=(n, 3))
    lam = rng.gamma(1.0, 1.0, size=(n, 3))
    pi = rng.dirichlet(np.ones(2), size=(n, 3))  # each row: Dirichlet(1, 1) over its two targets

    @jax.jit
    def one(a_i: jnp.ndarray, lam_i: jnp.ndarray, pi_i: jnp.ndarray) -> jnp.ndarray:
        th = dict(theta0)
        th["duration.rate"] = -jnp.log1p(-a_i)
        th["emission.eta"] = jnp.log(lam_i)[:, None]
        th["transition.pi_rows"] = pi_i
        return engine.loglik(th, spec, space, data)

    ll = np.asarray(jax.vmap(one)(jnp.asarray(a), jnp.asarray(lam), jnp.asarray(pi)))
    log_ev = special.logsumexp(ll) - np.log(n)
    w = np.exp(ll - ll.max())
    se = np.sqrt(np.var(w / w.mean()) / n)
    assert elbo <= log_ev + 3 * se + 1e-9
    assert elbo > log_ev - 4.0


def test_e13_t05a_unsupported_parameter_modes_make_the_block_a_point_block(tmp_path: Path) -> None:
    """Bounded / prior parameters (ignored by a conjugate update) turn their block into a point block."""
    import dataclasses

    spec, theta0 = em_models.build(tmp_path, "poisson", "geometric")
    bounded = tuple(
        dataclasses.replace(s, mode="bounded", domain="interval", bounds=(-3.0, 3.0))
        if s.name == "emission.eta"
        else s
        for s in spec.param_specs
    )
    bad = dataclasses.replace(spec, param_specs=bounded)
    res = VBEstimator().fit(
        bad,
        em_models.dataset("poisson", 80, 1),
        EstimationSection(method="vb", restarts=0, max_iter=5),
        0,
        warm_start=_start(spec, theta0),
    )
    assert res.diagnostics["vb"]["treatment"]["emission"] == "point"
