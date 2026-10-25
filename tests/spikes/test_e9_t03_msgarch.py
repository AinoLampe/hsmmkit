"""E9-T03: MS-GARCH emission through the engine carry branch (ADR-0022)."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.config.schema import EstimationSection
from regimekit.engine.expanded_hmm import ExpandedHMMEngine, _regime_log_emis
from regimekit.estimation.map import fit
from regimekit.model.duration.negbin import NegBinDuration
from regimekit.model.emission.gbm import GBMFamily
from regimekit.model.emission.msgarch import MSGARCHFamily
from regimekit.model.params import ParamManager, ParamSpec
from regimekit.model.params.priors import DirichletPrior
from regimekit.model.spec import ModelSpec
from regimekit.model.transition import JumpTransition
from regimekit.statespace.builders import build_expanded
from tests.helpers.engine_fixtures import DT, make_dataset
from tests.reference.ref_garch import garch11_loglik, simulate_garch11

PARAMS = {"m": 0.05, "omega": 4e-6, "alpha": 0.08, "beta": 0.88}


def _spec(emission: object, extra: tuple[ParamSpec, ...] = ()) -> ModelSpec:
    return ModelSpec(
        n_states=2,
        n_series=1,
        emission=emission,  # type: ignore[arg-type]
        duration=NegBinDuration(r=(1, 1)),
        transition=JumpTransition(),
        order=1,
        duration_conditioning="current",
        engine="expanded",
        d_max=10,
        initial="uniform",
        param_specs=extra,
    )


def _theta(
    m: list[float], omega: list[float], alpha: list[float], beta: list[float]
) -> dict[str, jnp.ndarray]:
    return {
        "emission.m": jnp.asarray(m),
        "emission.omega": jnp.asarray(omega),
        "emission.alpha": jnp.asarray(alpha),
        "emission.beta": jnp.asarray(beta),
        "duration.rate": jnp.asarray([10.0, 20.0]),
        "transition.pi_rows": jnp.ones((2, 1)),
    }


def _two_regimes() -> dict[str, jnp.ndarray]:
    return _theta([0.0, 0.1], [2e-6, 8e-6], [0.05, 0.15], [0.9, 0.7])


def test_e9_t03_one_state_garch_matches_the_reference_loglik() -> None:
    """AC2: two identical regimes make the HMM likelihood equal the single GARCH(1,1) likelihood (1e-6)."""
    t = 1500
    y = simulate_garch11(t, DT, **PARAMS, seed=3)
    data = make_dataset(y[:, None], np.full(t, DT))
    theta = _theta(*([PARAMS[k]] * 2 for k in ("m", "omega", "alpha", "beta")))
    spec = _spec(MSGARCHFamily())
    got = float(ExpandedHMMEngine().loglik(theta, spec, build_expanded(spec), data))
    assert abs(got - garch11_loglik(y, np.full(t, DT), **PARAMS)) <= 1e-6


def test_e9_t03_carry_is_per_regime_and_a_masked_step_feeds_the_variance_forward() -> None:
    """Each regime has its own recursion; a masked step uses `eps^2 = h` and everything stays finite."""
    t = 40
    y = simulate_garch11(t, DT, **PARAMS, seed=5)[:, None]
    fam = MSGARCHFamily()
    theta = _two_regimes()
    full = np.asarray(fam.log_prob(jnp.asarray(y), jnp.full((t,), DT), theta, None))
    for i in range(2):
        ref = garch11_loglik(
            y[:, 0],
            np.full(t, DT),
            float(theta["emission.m"][i]),
            float(theta["emission.omega"][i]),
            float(theta["emission.alpha"][i]),
            float(theta["emission.beta"][i]),
        )
        assert full[:, i].sum() == pytest.approx(ref, abs=1e-9)
    mask = np.ones((t, 1), dtype=bool)
    mask[10] = False
    held = np.asarray(fam.log_prob_scan(jnp.asarray(y), jnp.asarray(mask), jnp.full((t,), DT), theta, None))
    np.testing.assert_allclose(held[:11], full[:11], rtol=0, atol=1e-12)
    assert np.isfinite(held).all()
    assert not np.allclose(held[12:], full[12:])


def test_e9_t03_carry_scan_is_differentiable_through_the_engine() -> None:
    """AC1 (gradient): a finite, non-zero gradient w.r.t. every GARCH parameter through the scan."""
    t = 200
    y = simulate_garch11(t, DT, **PARAMS, seed=7)
    data = make_dataset(y[:, None], np.full(t, DT))
    spec = _spec(MSGARCHFamily())
    space = build_expanded(spec)
    theta = _two_regimes()
    names = ["emission.m", "emission.omega", "emission.alpha", "emission.beta"]

    def f(vals: list[jnp.ndarray]) -> jnp.ndarray:
        th = dict(theta)
        th.update(dict(zip(names, vals, strict=True)))
        return ExpandedHMMEngine().loglik(th, spec, space, data)

    grads = jax.grad(f)([theta[k] for k in names])
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
    assert all(float(jnp.max(jnp.abs(g))) > 0 for g in grads)


def test_e9_t03_gbm_keeps_the_non_carry_path() -> None:
    """AC1 (GBM unchanged): no carry, so the engine takes the old `log_prob` path bit for bit."""
    assert GBMFamily().init_carry() is None
    t = 30
    y = np.random.default_rng(1).normal(0, 0.01, (t, 1))
    data = make_dataset(y, np.full(t, DT))
    spec = _spec(GBMFamily())
    theta = {"emission.mu": jnp.asarray([[0.1], [-0.1]]), "emission.sigma": jnp.asarray([[0.15], [0.3]])}
    table = np.asarray(_regime_log_emis(theta, spec, data))
    direct = np.asarray(GBMFamily().log_prob(jnp.asarray(y), data.dt, theta, jnp.zeros((t, 0))))
    assert table.tobytes() == direct.tobytes()


def test_e9_t03_two_state_msgarch_fits_without_numerical_failure() -> None:
    """AC3: a 2-regime MS-GARCH MAP fit on synthetic data (restarts=2) returns a finite optimum."""
    t = 1200
    y = np.concatenate(
        [
            simulate_garch11(t // 2, DT, 0.05, 2e-6, 0.06, 0.90, 21),
            simulate_garch11(t // 2, DT, -0.05, 2e-5, 0.10, 0.80, 22),
        ]
    )
    data = make_dataset(y[:, None], np.full(t, DT))
    fam = MSGARCHFamily()
    extra = (
        *fam.param_specs(2, 1, ()),
        ParamSpec("duration.rate", (2,), "positive", "free", init=np.array([10.0, 20.0])),
        ParamSpec(
            "transition.pi_rows",
            (2, 1),
            "simplex_ref",
            "prior",
            init=np.ones((2, 1)),
            prior=DirichletPrior(alpha=1.5),
        ),
    )
    spec = _spec(fam, extra)
    assert ParamManager(spec.param_specs).n_free > 0
    cfg = EstimationSection(restarts=2, max_iter=150, init=["lattice_jitter"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit(spec, data, cfg, seed=0)
    assert np.isfinite(res.loglik)
    assert all(bool(np.all(np.isfinite(np.asarray(v)))) for v in res.params.values())
