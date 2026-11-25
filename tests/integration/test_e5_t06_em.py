"""E5-T06: EM (Baum-Welch) for the covariate-free GBM / geometric-or-negbin model vs MAP."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import regimekit.estimation.em as em_module
from regimekit.config.schema import EstimationSection
from regimekit.data.contracts import PreparedDataset
from regimekit.estimation.em import EMResult, em_fit
from regimekit.estimation.map import fit
from regimekit.inference.simulate import simulate_dataset
from regimekit.model.emission.gaussian import GaussianFamily
from regimekit.model.spec import ModelSpec
from regimekit.utils.errors import ConfigError, NumericalError
from regimekit.utils.rng import key_for
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


def test_e5_t06_pair_conditioning_and_increasing_sigma_are_refused() -> None:
    """Unsupported structure raises `ConfigError` instead of a shape error or a mis-ordered fit."""
    case = make_case("gbm", "negbin", 1, t_len=60)
    pair = dataclasses.replace(case.spec, duration_conditioning="pair")
    with pytest.raises(ConfigError, match="conditioning"):
        em_fit(pair, case.data, _cfg(), 0)
    ordered = tuple(
        dataclasses.replace(s, constraint="increasing", shape=(2,), init=np.array([0.1, 0.3]))
        if s.name == "emission.scale"
        else s
        for s in case.spec.param_specs
    )
    with pytest.raises(ConfigError, match="increasing"):
        em_fit(dataclasses.replace(case.spec, param_specs=ordered), case.data, _cfg(), 0)


def test_e5_t06_debug_mode_raises_when_the_loglik_decreases(monkeypatch: pytest.MonkeyPatch) -> None:
    """`debug` guards monotonicity: a corrupted M-step trips `NumericalError`."""
    spec, data = _problem("geometric", 4, t_len=800)
    good = em_fit(spec, data, _cfg(max_iter=3), 4)
    assert isinstance(good, EMResult)
    original = em_module._m_step

    def broken(*args: object) -> dict[str, jnp.ndarray]:
        out = dict(original(*args))
        out["emission.scale"] = out["emission.scale"] * 3.0
        return out

    monkeypatch.setattr(em_module, "_m_step", broken)
    with pytest.raises(NumericalError, match="decreased"):
        em_fit(spec, data, _cfg(max_iter=20), 4, debug=True)


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


def test_e5_t06_unsupported_models_raise_config_error() -> None:
    """AC3: covariates, non-GBM families, the explicit engine and irregular grids are refused."""
    for kwargs in ({"cov_em": True}, {"cov_du": True}, {"cov_tr": True}):
        case = make_case("gbm", "negbin", 1, t_len=60, **kwargs)  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="covariates"):
            em_fit(case.spec, case.data, _cfg(), 0)
    student = make_case("student_t", "negbin", 1, t_len=60)
    with pytest.raises(ConfigError, match="gaussian"):
        em_fit(student.spec, student.data, _cfg(), 0)
    explicit = make_case("gbm", "gamma", 1, t_len=60)
    with pytest.raises(ConfigError, match="expanded engine"):
        em_fit(explicit.spec, explicit.data, _cfg(), 0)
    order2 = make_case("gbm", "negbin", 2, t_len=60)
    with pytest.raises(ConfigError, match="order 1"):
        em_fit(order2.spec, order2.data, _cfg(), 0)
    base = make_case("gbm", "negbin", 1, t_len=60)
    irregular = dataclasses.replace(base.data, dt=base.data.dt * jnp.asarray([1.0, 3.0] * 30))
    with pytest.raises(ConfigError, match="constant time step"):
        em_fit(base.spec, irregular, _cfg(), 0)
