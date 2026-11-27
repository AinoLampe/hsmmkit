"""E5-T06: EM (Baum-Welch) for the covariate-free GBM / geometric-or-negbin model vs MAP."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import hsmmkit.estimation.em as em_module
from hsmmkit.config.schema import EstimationSection
from hsmmkit.data.contracts import PreparedDataset
from hsmmkit.estimation.em import EMResult, em_fit
from hsmmkit.estimation.map import fit
from hsmmkit.inference.simulate import simulate_dataset
from hsmmkit.model.emission.gaussian import GaussianFamily
from hsmmkit.model.spec import ModelSpec
from hsmmkit.utils.errors import ConfigError, NumericalError
from hsmmkit.utils.rng import key_for
from tests.helpers.engine_fixtures import DT
from tests.helpers.objective_fixtures import make_case

SIM_PURPOSE = 1
TRUE_MU = np.array([[0.10], [-0.20]])
TRUE_SIGMA = np.array([[0.12], [0.40]])
MEAN_SOJOURN = jnp.asarray([40.0, 25.0])


def _problem(duration: str, seed: int, t_len: int) -> tuple[ModelSpec, PreparedDataset]:
    """A 2-regime GBM model of the requested duration family and data simulated from known parameters."""
    case = make_case("gbm", duration, 1, t_len=t_len, seed=seed)
    theta = dict(case.manager.unflatten(case.manager.init_u()))
    theta["emission.mean"], theta["emission.scale"] = jnp.asarray(TRUE_MU), jnp.asarray(TRUE_SIGMA)
    theta.update(case.spec.duration.init_from_mean_sojourn(MEAN_SOJOURN, DT))
    data = simulate_dataset(
        key_for(seed, SIM_PURPOSE), theta, case.spec, case.space, jnp.full((t_len,), DT), clock="trading"
    )
    return case.spec, data


def _cfg(max_iter: int = 400, tol_f: float = 1e-10) -> EstimationSection:
    return EstimationSection(max_iter=max_iter, tol_f=tol_f, use_priors=False)


@pytest.mark.parametrize(("duration", "seed"), [("geometric", 1), ("negbin", 2), ("negbin", 3)])
def test_e5_t06_loglik_is_non_decreasing_across_iterations(duration: str, seed: int) -> None:
    """AC1: monotone EM on 3 synthetic datasets (and `debug=True` would raise on a decrease)."""
    spec, data = _problem(duration, seed, t_len=1500)
    res = em_fit(spec, data, _cfg(max_iter=60), seed, debug=True)
    ll = np.asarray(res.logliks)
    assert len(ll) >= 4
    assert np.all(np.diff(ll[:3]) > 0)  # the first iterations make real progress
    assert np.all(np.diff(ll) >= -1e-8 * np.abs(ll[:-1]))
    assert ll[-1] > ll[0]


def test_e5_t06_masked_steps_keep_em_monotone() -> None:
    """Unobserved steps carry no emission weight; EM stays monotone with a gap block."""
    spec, data = _problem("negbin", 5, t_len=1200)
    mask = np.ones(data.mask.shape, dtype=bool)
    mask[300:340] = False
    gappy = dataclasses.replace(data, mask=jnp.asarray(mask))
    ll = np.asarray(em_fit(spec, gappy, _cfg(max_iter=40), 5, debug=True).logliks)
    assert np.all(np.diff(ll) >= -1e-8 * np.abs(ll[:-1]))


def test_e5_t06_pair_conditioning_is_refused_and_increasing_sigma_runs_by_ecm() -> None:
    """Pair conditioning has no EM update; an `increasing` scale ordering is kept by ECM (E13-T02b)."""
    case = make_case("gbm", "negbin", 1, t_len=200)
    pair = dataclasses.replace(case.spec, duration_conditioning="pair")
    with pytest.raises(ConfigError, match="conditioning"):
        em_fit(pair, case.data, _cfg(), 0)
    ordered = tuple(
        dataclasses.replace(s, constraint="increasing", shape=(2,), init=np.array([0.1, 0.3]))
        if s.name == "emission.scale"
        else s
        for s in case.spec.param_specs
    )
    res = em_fit(
        dataclasses.replace(case.spec, param_specs=ordered), case.data, _cfg(max_iter=10), 0, debug=True
    )
    scale = np.asarray(res.params["emission.scale"]).ravel()
    assert np.all(np.diff(scale) > 0)
    assert np.all(np.diff(np.asarray(res.logliks)) >= -1e-8)


def test_e5_t06_debug_mode_raises_when_the_loglik_decreases(monkeypatch: pytest.MonkeyPatch) -> None:
    """`debug` guards monotonicity: a corrupted M-step trips `NumericalError`."""
    spec, data = _problem("geometric", 4, t_len=800)
    good = em_fit(spec, data, _cfg(max_iter=3), 4)
    assert isinstance(good, EMResult)
    original = em_module.em_update

    def broken(*args: object, **kwargs: object) -> dict[str, jnp.ndarray]:
        out = dict(original(*args, **kwargs))  # type: ignore[arg-type]
        out["emission.scale"] = out["emission.scale"] * 3.0
        return out

    monkeypatch.setattr(em_module, "em_update", broken)
    em_module._em_step.clear_cache()  # the compiled step closes over the (patched) function at trace time
    try:
        with pytest.raises(NumericalError, match="decreased"):
            em_fit(spec, data, _cfg(max_iter=20), 4, debug=True)
    finally:
        monkeypatch.undo()
        em_module._em_step.clear_cache()


@pytest.mark.parametrize("duration", ["geometric", "negbin"])
def test_e5_t06_em_agrees_with_map_on_a_two_state_set(duration: str) -> None:
    """AC2: EM optimum vs MAP (`use_priors=False`): loglik 1e-4 relative, sigma 1e-2 relative."""
    spec, data = _problem(duration, 7, t_len=4000)
    em = em_fit(spec, data, _cfg(max_iter=3000, tol_f=1e-12), 7)
    cfg = EstimationSection(
        restarts=3, max_iter=600, use_priors=False, init=["rolling_kmeans", "lattice_jitter"]
    )
    map_fit = fit(spec, data, cfg, 7)
    assert abs(em.loglik - map_fit.loglik) <= 1e-4 * abs(map_fit.loglik)
    sig_em = np.sort(np.asarray(em.params["emission.scale"]).ravel())
    sig_map = np.sort(np.asarray(map_fit.params["emission.scale"]).ravel())
    np.testing.assert_allclose(sig_em, sig_map, rtol=1e-2)


def test_e5_t06_sufficient_stats_give_the_gaussian_mle() -> None:
    """The weighted sums reproduce the direct weighted Gaussian MLE of (m, sigma^2)."""
    rng = np.random.default_rng(0)
    t = 500
    dt = DT * rng.integers(1, 4, size=t).astype(float)
    y = rng.normal(0.001, 0.01, size=(t, 1))
    w = rng.dirichlet([1.0, 2.0], size=t)
    st = GaussianFamily(scale_by_dt=True, ito_correction=True).sufficient_stats(
        jnp.asarray(y), jnp.asarray(dt), jnp.asarray(w)
    )
    m = np.asarray(st["w_y"] / st["w_dt"])
    var = np.asarray((st["w_y2_dt"] - 2 * m * st["w_y"] + m**2 * st["w_dt"]) / st["w"])
    for i in range(2):
        m_ref = (w[:, i] * y[:, 0]).sum() / (w[:, i] * dt).sum()
        var_ref = (w[:, i] * (y[:, 0] - m_ref * dt) ** 2 / dt).sum() / w[:, i].sum()
        assert m[i] == pytest.approx(m_ref, rel=1e-12)
        assert var[i] == pytest.approx(var_ref, rel=1e-10)


@pytest.mark.parametrize("kwargs", [{"cov_em": True}, {"cov_du": True}, {"cov_tr": True}])
def test_e5_t06_covariate_models_run_by_ecm_and_stay_monotone(kwargs: dict[str, bool]) -> None:
    """Covariate links have no closed form: ECM updates them and the log-likelihood does not decrease."""
    case = make_case("gbm", "negbin", 1, t_len=150, **kwargs)  # type: ignore[arg-type]
    ll = np.asarray(em_fit(case.spec, case.data, _cfg(max_iter=8), 0, debug=True).logliks)
    assert np.all(np.diff(ll) >= -1e-8 * np.abs(ll[:-1]))
    assert ll[-1] > ll[0]


def test_e5_t06_student_t_irregular_steps_and_the_explicit_engine() -> None:
    """Student-t (no closed form) and irregular steps run by ECM; the explicit engine is refused."""
    student = make_case("student_t", "negbin", 1, t_len=150)
    ll = np.asarray(em_fit(student.spec, student.data, _cfg(max_iter=6), 0, debug=True).logliks)
    assert np.all(np.diff(ll) >= -1e-8 * np.abs(ll[:-1]))
    base = make_case("gbm", "negbin", 1, t_len=150)
    irregular = dataclasses.replace(base.data, dt=base.data.dt * jnp.asarray([1.0, 3.0] * 75))
    ll_irr = np.asarray(em_fit(base.spec, irregular, _cfg(max_iter=6), 0, debug=True).logliks)
    assert np.all(np.diff(ll_irr) >= -1e-8 * np.abs(ll_irr[:-1]))
    explicit = make_case("gbm", "gamma", 1, t_len=60)
    with pytest.raises(ConfigError, match="expanded engine"):
        em_fit(explicit.spec, explicit.data, _cfg(), 0)


def test_e5_t06_order_two_memory_is_supported_and_monotone() -> None:
    """Order-2 memory (refused by the GBM-only EM of E5-T06) now runs and is monotone."""
    case = make_case("gbm", "negbin", 2, t_len=400, seed=3)
    ll = np.asarray(em_fit(case.spec, case.data, _cfg(max_iter=15), 3, debug=True).logliks)
    assert np.all(np.diff(ll) >= -1e-8 * np.abs(ll[:-1]))
