"""E4-T06a golden: explicit-duration forward-backward vs `tests/reference/ref_hsmm`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import logsumexp
from scipy.stats import norm

from regimekit.engine.explicit_duration import ExplicitDurationEngine, _forward_backward_core
from tests.helpers.engine_fixtures import Scenario, make_explicit_scenario
from tests.reference import ref_hsmm

ENGINE = ExplicitDurationEngine()


def reference_inputs(
    sc: Scenario, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(log_rho, log_trans, log_dur_pmf, log_emis)` for `ref_hsmm` (constant pmf/jump, nominal `dt`)."""
    d, th, spec = sc.data, sc.theta, sc.spec
    y = np.asarray(d.observations)[:, 0]
    mu, sig = np.asarray(th["emission.mean"]), np.asarray(th["emission.scale"])
    dt = d.nominal_dt
    log_emis = norm.logpdf(y[:, None], (mu - 0.5 * sig**2) * dt, sig * np.sqrt(dt))
    if mask is not None:
        log_emis = np.where(mask[:, None], log_emis, 0.0)
    pi = np.asarray(spec.transition.jump_probs_t(th, None, sc.space.layout))
    with np.errstate(divide="ignore"):
        log_trans = np.log(pi)
    log_pmf = np.asarray(spec.duration.log_pmf(th, spec.d_max, dt, None))
    if spec.initial == "free":
        log_rho = np.log(np.asarray(th["initial.logits"]))
    else:
        log_rho = np.full(spec.n_states, -np.log(spec.n_states))
    return log_rho, log_trans, log_pmf, log_emis


def elapsed_of(post: object) -> np.ndarray:
    """`(T, N, D)` elapsed posterior from a `Posterior`."""
    return np.asarray(post.elapsed_posterior())  # type: ignore[attr-defined]


@given(
    n_states=st.integers(2, 4),
    d_max=st.integers(2, 20),
    t_len=st.integers(5, 120),
    family=st.sampled_from(["negbin", "geometric", "gamma", "weibull"]),
    seed=st.integers(0, 10_000),
)
def test_e4_t06a_forward_backward_matches_reference(
    n_states: int, d_max: int, t_len: int, family: str, seed: int
) -> None:
    """AC1: loglik rel <= 1e-8; smoothed regime marginals and elapsed posterior max-abs <= 1e-8."""
    sc = make_explicit_scenario(n_states, d_max, t_len, family=family, seed=seed)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    inputs = reference_inputs(sc)
    ref_loglik, ref_smoothed, ref_elapsed = ref_hsmm.forward_backward(*inputs, d_max)

    assert abs(float(post.loglik) - ref_loglik) <= 1e-8 * abs(ref_loglik)
    np.testing.assert_allclose(np.asarray(post.regime_smoothed()), ref_smoothed, rtol=0, atol=1e-8)
    np.testing.assert_allclose(elapsed_of(post), ref_elapsed, rtol=0, atol=1e-8)
    assert post.engine == "explicit"


@given(seed=st.integers(0, 10_000), family=st.sampled_from(["negbin", "geometric", "gamma"]))
def test_e4_t06a_posterior_normalization_and_filter(seed: int, family: str) -> None:
    """AC3: rows sum to 1 (1e-10); `logsumexp(log_alpha[-1]) == loglik`; alpha * beta is the joint."""
    sc = make_explicit_scenario(3, 12, 60, family=family, seed=seed)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    elapsed = elapsed_of(post)
    np.testing.assert_allclose(elapsed.sum(axis=(1, 2)), 1.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(post.smoothed).sum(axis=1), 1.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(post.regime_smoothed()).sum(axis=1), 1.0, rtol=0, atol=1e-10)
    log_alpha = np.asarray(post.log_alpha)
    assert abs(logsumexp(log_alpha[-1]) - float(post.loglik)) <= 1e-9
    np.testing.assert_allclose(np.asarray(post.regime_filtered()).sum(axis=1), 1.0, rtol=0, atol=1e-10)
    # alpha * beta reproduces the joint: exp(alpha + beta - loglik) == smoothed wherever both are finite
    with np.errstate(invalid="ignore"):
        joint = np.exp(log_alpha + np.asarray(post.log_beta) - float(post.loglik))
    np.testing.assert_allclose(joint, np.asarray(post.smoothed), rtol=0, atol=1e-10)


@given(seed=st.integers(0, 10_000))
def test_e4_t06a_forward_and_backward_logliks_agree(seed: int) -> None:
    """The backward recursion's total equals the forward log-likelihood (1e-10)."""
    sc = make_explicit_scenario(3, 8, 40, family="negbin", seed=seed)
    *_, loglik, loglik_b = _forward_backward_core(sc.theta, sc.spec, sc.space, sc.data)
    assert abs(float(loglik) - float(loglik_b)) <= 1e-10 * abs(float(loglik))
    assert abs(float(ENGINE.loglik(sc.theta, sc.spec, sc.space, sc.data)) - float(loglik)) <= 1e-12


@given(seed=st.integers(0, 10_000), family=st.sampled_from(["negbin", "gamma"]))
def test_e4_t06a_right_censoring_matches_brute_force(seed: int, family: str) -> None:
    """AC2: T=6, D_max=3 loglik equals the enumeration over all segmentations (<= 1e-10)."""
    sc = make_explicit_scenario(3, 3, 6, family=family, seed=seed, mean_steps=2.0)
    inputs = reference_inputs(sc)
    brute = ref_hsmm.brute_force_loglik(*inputs, 3)
    loglik = float(ENGINE.loglik(sc.theta, sc.spec, sc.space, sc.data))
    assert abs(loglik - brute) <= 1e-10 * max(1.0, abs(brute))


def test_e4_t06a_free_initial_matches_reference() -> None:
    """`initial="free"` uses `theta["initial.logits"]` over the first segment's regime."""
    sc = make_explicit_scenario(3, 6, 30, family="gamma", initial="free", seed=4)
    inputs = reference_inputs(sc)
    ref_loglik, ref_smoothed, _ = ref_hsmm.forward_backward(*inputs, 6)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    assert abs(float(post.loglik) - ref_loglik) <= 1e-8 * abs(ref_loglik)
    np.testing.assert_allclose(np.asarray(post.regime_smoothed()), ref_smoothed, rtol=0, atol=1e-8)


def test_e4_t06a_masked_steps_match_reference() -> None:
    """Masked steps contribute a zero log-density in engine and reference alike."""
    sc = make_explicit_scenario(3, 6, 40, family="weibull", seed=8)
    mask = np.ones((40, 1), dtype=bool)
    mask[7:12] = False
    fields = {f: getattr(sc.data, f) for f in sc.data.__dataclass_fields__}
    data = sc.data.__class__(**{**fields, "mask": mask})
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, data)
    ref_loglik, ref_smoothed, _ = ref_hsmm.forward_backward(*reference_inputs(sc, mask[:, 0]), 6)
    assert abs(float(post.loglik) - ref_loglik) <= 1e-8 * abs(ref_loglik)
    np.testing.assert_allclose(np.asarray(post.regime_smoothed()), ref_smoothed, rtol=0, atol=1e-8)


@pytest.mark.parametrize("t_len", [1, 2])
def test_e4_t06a_tiny_sequences_are_finite(t_len: int) -> None:
    """T=1 and T=2 (empty / one-step scans) give a normalized posterior."""
    sc = make_explicit_scenario(2, 4, t_len, family="geometric", seed=1)
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    np.testing.assert_allclose(np.asarray(post.smoothed).sum(axis=1), 1.0, rtol=0, atol=1e-10)
    assert np.isfinite(float(post.loglik))
