"""E4-T03b golden: `log_beta` and smoothed posterior vs the NumPy reference."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from tests.golden.test_e4_t03a_forward import assert_matches, reference_full
from tests.helpers.engine_fixtures import make_scenario


def to_probs(log_gamma: np.ndarray) -> np.ndarray:
    """`exp` of a log-posterior with structural `-inf` mapped to exactly 0."""
    finite = np.isfinite(log_gamma)
    return np.where(finite, np.exp(np.where(finite, log_gamma, 0.0)), 0.0)


@given(
    n_states=st.integers(2, 4),
    order=st.integers(1, 2),
    t_len=st.integers(5, 200),
    seed=st.integers(0, 10_000),
    data=st.data(),
)
def test_e4_t03b_beta_and_smoothed_match_reference(
    n_states: int, order: int, t_len: int, seed: int, data: st.DataObject
) -> None:
    """AC1: `log_beta`, `smoothed` max-abs <= 1e-8 (and alpha/loglik as in E4-T03a)."""
    r = data.draw(st.lists(st.integers(1, 3), min_size=n_states, max_size=n_states))
    n_cov = data.draw(st.integers(0, 2))
    sc = make_scenario(
        n_states, r, order, t_len, n_cov_pi=n_cov, n_cov_d=n_cov, seed=seed, irregular=bool(seed % 2)
    )
    post = ExpandedHMMEngine().forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    ref_alpha, ref_beta, ref_gamma, ref_loglik = reference_full(sc)

    assert_matches(np.asarray(post.log_alpha), float(post.loglik), ref_alpha, ref_loglik)
    beta = np.asarray(post.log_beta)
    finite = np.isfinite(ref_beta)
    assert np.array_equal(finite, np.isfinite(beta))
    np.testing.assert_allclose(beta[finite], ref_beta[finite], rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(post.smoothed), to_probs(ref_gamma), rtol=0, atol=1e-8)
