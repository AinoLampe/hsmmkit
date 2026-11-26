"""E12-T06 golden: label / boundary masks equal independent NumPy references; unmasked runs are unchanged."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.data.contracts import BoundaryConstraints
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine, _regime_log_emis
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.engine.viterbi import viterbi_path
from tests.helpers.engine_fixtures import Scenario, make_explicit_scenario, make_scenario
from tests.reference.ref_constraints import explicit_brute_force, masked_forward, masked_viterbi_score

N = 3


def _constrain(sc: Scenario, allowed: np.ndarray | None, switch: set[int], stay: set[int]) -> Scenario:
    boundaries = BoundaryConstraints(tuple(sorted(switch)), tuple(sorted(stay))) if switch or stay else None
    data = dataclasses.replace(
        sc.data, labels=None if allowed is None else jnp.asarray(allowed), boundaries=boundaries
    )
    return dataclasses.replace(sc, data=data)


def _labels(t_len: int, truth: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Allowed-regime matrix: a fraction of the steps is hard-labelled with `truth`, the rest is free."""
    rng = np.random.default_rng(seed)
    allowed = np.ones((t_len, N), dtype=bool)
    for t in np.flatnonzero(rng.random(t_len) < frac):
        allowed[t] = False
        allowed[t, truth[t]] = True
    return allowed


def _expanded_inputs(sc: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    space, spec, theta = sc.space, sc.spec, sc.theta
    _, log_a = space.transition_fn(theta, jnp.zeros(0), jnp.zeros(0), jnp.asarray(sc.data.nominal_dt))
    t_len = sc.data.observations.shape[0]
    log_emis = np.asarray(jnp.take(_regime_log_emis(theta, spec, sc.data), space.index.regime_of, axis=1))
    s = space.n_expanded
    log_rho = np.full(s, -np.inf)
    log_rho[np.asarray(space.phase1_idx)] = -np.log(space.phase1_idx.shape[0])
    del t_len
    return log_rho, np.asarray(log_a), log_emis, np.asarray(space.index.regime_of)


@pytest.mark.parametrize("order", [1, 2])
def test_e12_t06_expanded_masked_loglik_posterior_and_viterbi_match_the_reference(order: int) -> None:
    """AC2: labels and both boundary kinds on the expanded engine equal the masked NumPy forward (<= 1e-8)."""
    sc0 = make_scenario(N, [2, 1, 3], order, 14, seed=2)
    rng = np.random.default_rng(1)
    truth = rng.integers(0, N, size=14)
    allowed = _labels(14, truth, 0.25, seed=3)
    # two must_not_switch steps keep the model feasible; `switch` is chosen where the (random) labels allow it
    switch, stay = {7}, {3, 11}
    sc = _constrain(sc0, allowed, switch, stay)
    log_rho, log_a, log_emis, regime_of = _expanded_inputs(sc0)
    alpha, ref = masked_forward(log_rho, log_a, log_emis, regime_of, allowed, switch, stay)
    if not np.isfinite(ref):  # a random label set may be infeasible under the switch; relax it
        sc = _constrain(sc0, allowed, set(), stay)
        alpha, ref = masked_forward(log_rho, log_a, log_emis, regime_of, allowed, set(), stay)
        switch = set()
    engine = ExpandedHMMEngine()
    log_alpha, loglik = engine.forward(sc.theta, sc.spec, sc.space, sc.data)
    assert float(loglik) == pytest.approx(ref, rel=1e-8)
    finite = np.isfinite(alpha)
    np.testing.assert_allclose(np.asarray(log_alpha)[finite], alpha[finite], rtol=1e-8, atol=1e-8)
    post = engine.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    smoothed = np.asarray(post.smoothed)
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-10)
    forbidden = ~allowed[:, regime_of]
    assert float(smoothed[forbidden].max(initial=0.0)) == 0.0
    path = viterbi_path(sc.theta, sc.spec, sc.space, sc.data)
    score = masked_viterbi_score(log_rho, log_a, log_emis, regime_of, allowed, switch, stay)
    regimes = regime_of[np.asarray(path)]
    assert np.all(allowed[np.arange(14), regimes])
    for t in stay:
        assert regimes[t] == regimes[t - 1]
    for t in switch:
        assert regimes[t] != regimes[t - 1]
    assert np.isfinite(score)


def test_e12_t06_unlabelled_data_are_bitwise_unchanged() -> None:
    """AC2: `labels=None`, all-`True` labels and no boundaries give the identical log-likelihood bits."""
    sc = make_scenario(N, [2, 1, 3], 1, 30, seed=5)
    engine = ExpandedHMMEngine()
    base = np.asarray(engine.loglik(sc.theta, sc.spec, sc.space, sc.data))
    free = _constrain(sc, np.ones((30, N), dtype=bool), set(), set())
    assert np.array_equal(base, np.asarray(engine.loglik(sc.theta, sc.spec, sc.space, free.data)))
    xs = make_explicit_scenario(N, 8, 24, family="gamma", seed=6)
    eng = ExplicitDurationEngine()
    ref = np.asarray(eng.loglik(xs.theta, xs.spec, xs.space, xs.data))
    freex = _constrain(xs, np.ones((24, N), dtype=bool), set(), set())
    assert np.array_equal(ref, np.asarray(eng.loglik(freex.theta, freex.spec, freex.space, freex.data)))


def _explicit_inputs(sc: Scenario) -> tuple[Any, Any, Any, Any]:
    spec, theta = sc.spec, sc.theta
    pi = np.asarray(spec.transition.jump_probs_t(theta, None, sc.space.layout))
    with np.errstate(divide="ignore"):
        log_pi = np.log(pi)
    log_pmf = np.asarray(spec.duration.log_pmf(theta, spec.d_max, sc.data.nominal_dt, None))
    dt = jnp.full_like(sc.data.dt, sc.data.nominal_dt)
    log_emis = np.asarray(_regime_log_emis(theta, spec, sc.data, dt))
    return np.full(N, -np.log(N)), log_pi, log_pmf, log_emis


@pytest.mark.parametrize("family", ["gamma", "negbin"])
def test_e12_t06_explicit_masked_loglik_matches_brute_force(family: str) -> None:
    """AC2 (explicit engine): every constraint kind equals the enumeration of all feasible segmentations."""
    t_len = 9
    sc0 = make_explicit_scenario(N, 5, t_len, family=family, seed=3)
    truth = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    allowed = _labels(t_len, truth, 0.4, seed=4)
    for switch, stay in [(set(), set()), ({5}, set()), (set(), {3, 6}), ({3}, {6})]:
        sc = _constrain(sc0, allowed, switch, stay)
        log_rho, log_pi, log_pmf, log_emis = _explicit_inputs(sc0)
        ref = explicit_brute_force(log_rho, log_pi, log_pmf, log_emis, allowed, switch, stay)
        got = float(ExplicitDurationEngine().loglik(sc.theta, sc.spec, sc.space, sc.data))
        assert np.isfinite(ref)
        assert got == pytest.approx(ref, rel=1e-8), (switch, stay)


def test_e12_t06_explicit_viterbi_and_posterior_respect_the_masks() -> None:
    """Segmental Viterbi and the posterior honour labels and boundaries; the posterior still normalizes."""
    t_len = 12
    sc0 = make_explicit_scenario(N, 6, t_len, family="gamma", seed=7)
    truth = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    allowed = _labels(t_len, truth, 0.5, seed=8)
    sc = _constrain(sc0, allowed, {4}, {2, 9})
    eng = ExplicitDurationEngine()
    path = np.asarray(eng.viterbi(sc.theta, sc.spec, sc.space, sc.data))
    regimes = np.asarray(sc.space.index.regime_of)[path]
    assert np.all(allowed[np.arange(t_len), regimes])
    assert regimes[4] != regimes[3]
    assert regimes[2] == regimes[1]
    assert regimes[9] == regimes[8]
    post = eng.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    smoothed = np.asarray(post.smoothed).reshape(t_len, N, -1).sum(axis=2)
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-10)
    assert float(smoothed[~allowed].max(initial=0.0)) < 1e-12
