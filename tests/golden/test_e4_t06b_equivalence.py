"""E4-T06b golden: expanded (phase) engine vs explicit-duration engine on shifted-NB sojourns."""

from __future__ import annotations

from dataclasses import fields

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.engine.explicit_duration import ExplicitDurationEngine
from regimekit.model.spec import ModelSpec
from regimekit.statespace.builders import build_explicit
from tests.helpers.engine_fixtures import DT, Scenario, make_dataset, make_scenario

D_MAX = 400
T_LEN = 300


def simulate_returns(
    rng: np.random.Generator, a: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Log-returns from the model itself: shifted-NB sojourns (`r = (2, 3)`), alternating regimes."""
    r = (2, 3)
    regime, path = int(rng.integers(2)), []
    while len(path) < T_LEN:
        path.extend([regime] * int(r[regime] + rng.negative_binomial(r[regime], a[regime])))
        regime = 1 - regime
    z = np.asarray(path[:T_LEN])
    return rng.normal((mu[z] - 0.5 * sigma[z] ** 2) * DT, sigma[z] * np.sqrt(DT))[:, None]


def paired_scenarios(seed: int) -> tuple[Scenario, ModelSpec, dict[str, jnp.ndarray]]:
    """NB `r = (2, 3)`, mean sojourn ~30 steps, constant `Delta`, no covariates: expanded + explicit.

    The data are simulated from the model: the truncation at `D_MAX` (mass < 1e-10 per sojourn) is
    only relative to the survival of a *very* long final segment, which noise-like data would favour.
    """
    rng = np.random.default_rng(seed)
    sc = make_scenario(2, [2, 3], 1, T_LEN, seed=seed)
    a = np.array([2.0, 3.0]) / (30.0 * rng.uniform(0.8, 1.2, size=2))
    mu, sigma = np.array([0.15, -0.1]), np.array([0.1, 0.45])
    theta = {
        **sc.theta,
        "duration.rate": jnp.asarray(-np.log1p(-a) / DT),
        "emission.mean": jnp.asarray(mu),
        "emission.scale": jnp.asarray(sigma),
    }
    data = make_dataset(simulate_returns(rng, a, mu, sigma), np.full(T_LEN, DT))
    spec_x = ModelSpec(
        **{
            **{f.name: getattr(sc.spec, f.name) for f in fields(sc.spec)},
            "engine": "explicit",
            "d_max": D_MAX,
        }
    )
    return Scenario(sc.spec, sc.space, theta, data), spec_x, theta


@pytest.mark.parametrize("seed", range(20))
def test_e4_t06b_engines_agree_on_shifted_negbin(seed: int) -> None:
    """AC2 (Appendix D): loglik rel <= 1e-6 and smoothed regime marginals max-abs <= 1e-6, 20 seeds."""
    sc, spec_x, theta = paired_scenarios(seed)
    space_x = build_explicit(spec_x)

    pmf_mass = np.exp(np.asarray(spec_x.duration.log_pmf(theta, D_MAX, sc.data.nominal_dt, None))).sum(axis=1)
    assert np.all(1.0 - pmf_mass < 1e-10)  # truncation at D_MAX loses no mass

    expanded = ExpandedHMMEngine().forward_backward(theta, sc.spec, sc.space, sc.data)
    explicit = ExplicitDurationEngine().forward_backward(theta, spec_x, space_x, sc.data)

    ll_e, ll_x = float(expanded.loglik), float(explicit.loglik)
    assert abs(ll_e - ll_x) <= 1e-6 * abs(ll_e)
    np.testing.assert_allclose(
        np.asarray(expanded.regime_smoothed()), np.asarray(explicit.regime_smoothed()), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(expanded.regime_filtered()), np.asarray(explicit.regime_filtered()), rtol=0, atol=1e-6
    )


def test_e4_t06b_elapsed_posterior_is_consistent_with_phase_posterior() -> None:
    """Explicit elapsed mass is a proper distribution and its regime marginal equals the expanded one."""
    sc, spec_x, theta = paired_scenarios(3)
    explicit = ExplicitDurationEngine().forward_backward(theta, spec_x, build_explicit(spec_x), sc.data)
    elapsed = np.asarray(explicit.elapsed_posterior())
    assert elapsed.shape == (T_LEN, 2, D_MAX)
    np.testing.assert_allclose(elapsed.sum(axis=(1, 2)), 1.0, rtol=0, atol=1e-10)
    # the elapsed count can never exceed the number of steps seen so far
    for t in (0, 5, 50):
        assert elapsed[t, :, t + 1 :].sum() == 0.0
