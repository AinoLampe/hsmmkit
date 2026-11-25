"""E4-T03a golden: JAX forward vs the NumPy reference driven with E4-T02b's `A_t`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import norm

from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.statespace.builders import initial_log_rho
from tests.helpers.engine_fixtures import Scenario, make_scenario
from tests.reference import ref_hmm


def reference_inputs(
    sc: Scenario, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(log_rho, log_a (T, S, S), log_emis (T, S))` for `ref_hmm`: scipy emissions, E4-T02b `A_t`."""
    d, sp, th = sc.data, sc.space, sc.theta
    y, dt = np.asarray(d.observations)[:, 0], np.asarray(d.dt)
    t_len = y.shape[0]
    mu, sig = np.asarray(th["emission.mean"]), np.asarray(th["emission.scale"])
    mean = (mu[None, :] - 0.5 * sig[None, :] ** 2) * dt[:, None]
    sd = sig[None, :] * np.sqrt(dt)[:, None]
    log_emis_n = norm.logpdf(y[:, None], mean, sd)  # (T, N)
    if mask is not None:
        log_emis_n = np.where(mask[:, None], log_emis_n, 0.0)
    log_emis = log_emis_n[:, np.asarray(sp.index.regime_of)]
    z_pi = None if d.cov_transition is None else np.asarray(d.cov_transition)
    z_d = None if d.cov_duration is None else np.asarray(d.cov_duration)
    # Test-only: `vmap` over steps (the engine itself never materializes `(T, S, S)`).
    z_pi_j = jnp.zeros((t_len, 0)) if z_pi is None else jnp.asarray(z_pi)
    z_d_j = jnp.zeros((t_len, 0)) if z_d is None else jnp.asarray(z_d)
    _, log_a = jax.vmap(lambda zp, zd, dti: sp.transition_fn(th, zp, zd, dti))(z_pi_j, z_d_j, jnp.asarray(dt))
    log_a = np.asarray(log_a)
    log_rho = np.asarray(initial_log_rho(sc.spec, sp, th, d.nominal_dt))
    return log_rho, log_a, log_emis


def reference_full(
    sc: Scenario, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """`ref_hmm.forward_backward` on `reference_inputs`."""
    return ref_hmm.forward_backward(*reference_inputs(sc, mask))


def reference(sc: Scenario, mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """`(log_alpha, loglik)` of `reference_full`."""
    log_alpha, _, _, loglik = reference_full(sc, mask)
    return log_alpha, loglik


def assert_matches(log_alpha: np.ndarray, loglik: float, ref_alpha: np.ndarray, ref_loglik: float) -> None:
    """Same `-inf` pattern; finite entries within 1e-8 abs; loglik within 1e-8 rel."""
    finite = np.isfinite(ref_alpha)
    assert np.array_equal(finite, np.isfinite(log_alpha))
    np.testing.assert_allclose(log_alpha[finite], ref_alpha[finite], rtol=0, atol=1e-8)
    assert abs(loglik - ref_loglik) <= 1e-8 * abs(ref_loglik)


@given(
    n_states=st.integers(2, 4),
    order=st.integers(1, 2),
    t_len=st.integers(5, 200),
    seed=st.integers(0, 10_000),
    data=st.data(),
)
def test_e4_t03a_forward_matches_reference(
    n_states: int, order: int, t_len: int, seed: int, data: st.DataObject
) -> None:
    """AC1: `log_alpha` max-abs <= 1e-8, `loglik` rel <= 1e-8, random N/T/r_i/order/covariates."""
    r = data.draw(st.lists(st.integers(1, 3), min_size=n_states, max_size=n_states))
    n_cov = data.draw(st.integers(0, 2))
    sc = make_scenario(
        n_states, r, order, t_len, n_cov_pi=n_cov, n_cov_d=n_cov, seed=seed, irregular=bool(seed % 2)
    )
    log_alpha, loglik = ExpandedHMMEngine().forward(sc.theta, sc.spec, sc.space, sc.data)
    ref_alpha, ref_loglik = reference(sc)
    assert_matches(np.asarray(log_alpha), float(loglik), ref_alpha, ref_loglik)


def test_e4_t03a_free_initial_matches_reference() -> None:
    """AC3 (free): `initial="free"` uses `theta["initial.logits"]`; loglik equals the reference."""
    sc = make_scenario(3, [2, 1, 3], 1, 60, initial="free", seed=3)
    _, loglik = ExpandedHMMEngine().forward(sc.theta, sc.spec, sc.space, sc.data)
    _, ref_loglik = reference(sc)
    assert abs(float(loglik) - ref_loglik) <= 1e-8 * abs(ref_loglik)
    other = dict(sc.theta)
    other["initial.logits"] = sc.theta["initial.logits"][::-1]
    _, changed = ExpandedHMMEngine().forward(other, sc.spec, sc.space, sc.data)
    assert abs(float(changed) - float(loglik)) > 1e-6  # the logits actually enter the result
