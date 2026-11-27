"""E13-T03a: GEM on the explicit engine: monotonicity, censoring, geometric parity with EM, refusals."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.config.schema import EstimationSection
from hsmmkit.estimation import fit as dispatch_fit
from hsmmkit.estimation.em import EMEstimator
from hsmmkit.estimation.gem import GEMEstimator, censored_pmf, pmf_objective
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.plugins import REGISTRY
from hsmmkit.utils.errors import ConfigError
from tests.helpers import gem_models


def _chain(t_len: int, seed: int, d_max: int = 12):  # noqa: ANN202
    pmf = np.zeros((2, d_max))
    pmf[0, :8] = [0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.03, 0.02]
    pmf[1, :10] = [0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01]
    pmf /= pmf.sum(axis=1, keepdims=True)
    pi = np.array([[0.0, 1.0], [1.0, 0.0]])
    y, _ = gem_models.simulate(t_len, pmf, pi, np.array([-1.0, 1.0]), np.array([0.6, 0.9]), seed)
    return gem_models.dataset(y), pmf


@pytest.mark.parametrize(
    ("left", "seed"), [(False, 1), (False, 2), (True, 1), (True, 2), (True, 3), (True, 4)]
)
def test_e13_t03a_nonparametric_gem_is_monotone(
    tmp_path: Path,
    left: bool,
    seed: int,
    monkeypatch,  # noqa: ANN001
) -> None:
    """AC1: the log-likelihood never decreases beyond 1e-10 (with and without left-censoring)."""
    monkeypatch.setenv("HSMMKIT_DEBUG", "1")  # the run raises on a decrease
    data, _ = _chain(300 if left else 600, seed)
    extra = {"initial_state": "free"} if left else {}
    spec = gem_models.build(tmp_path, 2, 12, left=left, means=[-0.5, 0.5], scales=[0.8, 0.8], **extra)
    cfg = EstimationSection(method="gem", restarts=0, max_iter=60, tol_f=1e-14, use_priors=False)
    start = ParamManager(spec.param_specs).init_u()
    res = GEMEstimator().fit(spec, data, cfg, 0, warm_start=start)
    trace = np.asarray(res.restarts["loglik_trace"].iloc[0])
    assert res.estimator == "gem"
    assert trace[-1] > trace[0]
    assert np.all(np.diff(trace[:-1]) >= -(1e-10 + 1e-13 * np.abs(trace[:-2])))
    assert res.diagnostics["gem"]["duration"].startswith("closed-form")
    pmf = np.asarray(res.params["duration.logits"])
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-10)


def test_e13_t03a_censoring_beats_the_naive_update() -> None:
    """AC3: with a long right-censored final segment the self-consistency pmf has a higher Q than p ~ eta."""
    d = 10
    eta = np.array([5.0, 8.0, 9.0, 6.0, 3.0, 1.0, 0.5, 0.2, 0.0, 0.0])
    cens = np.zeros(d)
    cens[8] = 6.0  # six final segments still running after 9 steps
    p0 = np.full(d, 1.0 / d)
    naive = eta / eta.sum()
    better = np.asarray(censored_pmf(jnp.asarray(p0), jnp.asarray(eta), jnp.asarray(cens), jnp.asarray(0.0)))
    q = lambda p: float(  # noqa: E731
        pmf_objective(
            jnp.asarray(np.maximum(p, 1e-12)), jnp.asarray(eta), jnp.asarray(cens), jnp.asarray(0.0)
        )
    )
    assert q(better) > q(naive) + 1e-3
    assert better[8:].sum() > naive[8:].sum()  # mass moves to the long durations
    # without censoring the maximizer is p ~ eta
    free = np.asarray(censored_pmf(jnp.asarray(p0), jnp.asarray(eta + 1e-12), jnp.zeros(d), jnp.asarray(0.0)))
    np.testing.assert_allclose(free, eta / eta.sum(), atol=1e-6)


@pytest.mark.parametrize("n_first", [0.0, 7.0])
def test_e13_t03a_the_minorize_maximize_step_never_decreases_q(n_first: float) -> None:
    """Every inner step raises Q (also with the residual-life normalizer `-n_first log mu`)."""
    from hsmmkit.estimation.gem import _mm_step

    rng = np.random.default_rng(3)
    d = 15
    c, w = jnp.asarray(rng.gamma(1.0, 3.0, d)), jnp.asarray(rng.gamma(1.0, 2.0, d))
    p = jnp.asarray(rng.dirichlet(np.ones(d)))
    nf = jnp.asarray(n_first)
    q_prev = float(pmf_objective(p, c, w, nf))
    for _ in range(40):
        p = _mm_step(p, c, w, nf)
        assert abs(float(jnp.sum(p)) - 1.0) < 1e-12
        q = float(pmf_objective(p, c, w, nf))
        assert q >= q_prev - 1e-10
        q_prev = q


def test_e13_t03a_geometric_gem_matches_em_on_the_expanded_engine(tmp_path: Path) -> None:
    """AC4: geometric durations under GEM reach the EM (expanded) optimum (D_max mass loss ~ 1e-13)."""
    rng = np.random.default_rng(5)
    z = np.zeros(900, dtype=int)
    s = 0
    for t in range(900):
        z[t] = s
        if rng.random() < 0.25:
            s = 1 - s
    y = rng.normal(np.where(z == 0, -1.0, 1.0), np.where(z == 0, 0.6, 0.9))[:, None]
    data = gem_models.dataset(y)
    kw = {"means": [-0.4, 0.4], "scales": [0.8, 0.8], "duration": "geometric", "mean_sojourn": 4.0}
    spec_x = gem_models.build(tmp_path, 2, 100, **kw)
    spec_e = gem_models.build(tmp_path, 2, 100, engine="expanded", **kw)
    cfg = EstimationSection(restarts=0, max_iter=4000, tol_f=1e-15, use_priors=False)
    start = ParamManager(spec_x.param_specs).init_u()
    gem = GEMEstimator().fit(spec_x, data, cfg, 0, warm_start=start)
    em = EMEstimator().fit(spec_e, data, cfg, 0, warm_start=ParamManager(spec_e.param_specs).init_u())
    assert abs(gem.loglik - em.loglik) <= 1e-6 * abs(em.loglik)
    for name, value in em.params.items():
        if name == "transition.pi_rows" or name not in gem.params:
            continue
        np.testing.assert_allclose(np.asarray(gem.params[name]), np.asarray(value), rtol=1e-4, atol=1e-6)


def test_e13_t03a_the_method_key_and_refusals(tmp_path: Path) -> None:
    """`gem` is a registered estimator; the gate row is gone; unsupported structures are refused."""
    assert REGISTRY.get("estimator", "gem").name == "gem"
    data, _ = _chain(200, 2)
    cfg = EstimationSection(method="gem", restarts=0, max_iter=3, use_priors=False)
    spec = gem_models.build(tmp_path, 2, 12)
    res = dispatch_fit(spec, data, cfg, 0, ParamManager(spec.param_specs).init_u())
    assert res.estimator == "gem"
    with pytest.raises(ConfigError, match="E13-T03b"):
        GEMEstimator().fit(gem_models.build(tmp_path, 2, 12, duration="negbin"), data, cfg, 0)
    with pytest.raises(ConfigError, match="explicit"):
        GEMEstimator().fit(
            gem_models.build(tmp_path, 2, 12, duration="geometric", engine="expanded"), data, cfg, 0
        )
    tempered = EstimationSection.model_validate(
        {"method": "gem", "restarts": 0, "max_iter": 3, "gem": {"tempering": {"enabled": True}}}
    )
    with pytest.raises(ConfigError, match="tempering"):
        GEMEstimator().fit(spec, data, tempered, 0, ParamManager(spec.param_specs).init_u())


def test_e13_t03a_left_censoring_needs_a_free_initial_law(tmp_path: Path) -> None:
    """With `uniform` the engine ties the first segment to the jump matrix: refused (monotonicity)."""
    data, _ = _chain(100, 1)
    spec = gem_models.build(tmp_path, 2, 12, left=True)
    with pytest.raises(ConfigError, match="initial_state: free"):
        GEMEstimator().fit(spec, data, EstimationSection(method="gem", restarts=0), 0)
    per_state = gem_models.build(tmp_path, 2, 12)
    object.__setattr__(per_state, "duration", type("PerStateDuration", (), {})())
    with pytest.raises(ConfigError, match="per-state"):
        GEMEstimator().fit(per_state, data, EstimationSection(method="gem", restarts=0), 0)
