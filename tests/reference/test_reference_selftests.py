"""E4-T01: self-tests for the pure NumPy reference implementations themselves.

No `hsmmkit` import (CLAUDE.md §6) -- these tests validate `tests/reference/` in isolation, so
later golden tests (E4-T03a, E4-T06a, ...) can trust it as ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.reference import ref_dists, ref_expansion, ref_hmm, ref_hsmm

REFERENCE_DIR = Path(__file__).parent


# --- AC1: hand-computed 2-state/3-step HMM, alpha/beta/loglik to 1e-12 ---------------------------


def test_e4_t01_ref_hmm_hand_example_alpha_beta_loglik() -> None:
    """`ref_hmm.forward_backward` reproduces the hand-computed 2-state/3-step example, <= 1e-12."""
    log_alpha, log_beta, _log_gamma, loglik = ref_hmm.forward_backward(
        ref_hmm.HAND_EXAMPLE_LOG_RHO, ref_hmm.HAND_EXAMPLE_LOG_A, ref_hmm.HAND_EXAMPLE_LOG_EMIS
    )
    np.testing.assert_allclose(np.exp(log_alpha), ref_hmm.HAND_EXAMPLE_ALPHA, atol=1e-12)
    np.testing.assert_allclose(np.exp(log_beta), ref_hmm.HAND_EXAMPLE_BETA, atol=1e-12)
    assert loglik == pytest.approx(ref_hmm.HAND_EXAMPLE_LOGLIK, abs=1e-12)


def test_e4_t01_ref_hmm_viterbi_hand_example() -> None:
    """`ref_hmm.viterbi` on the hand example picks the (visually obvious) all-state-0 path."""
    path = ref_hmm.viterbi(
        ref_hmm.HAND_EXAMPLE_LOG_RHO, ref_hmm.HAND_EXAMPLE_LOG_A, ref_hmm.HAND_EXAMPLE_LOG_EMIS
    )
    np.testing.assert_array_equal(path, [0, 0, 0])


# --- AC2: ref_hsmm.forward_backward loglik == brute_force_loglik, N=2, T=6, D_max=3, 1e-12 -------


@pytest.mark.parametrize("seed", range(10))
def test_e4_t01_ref_hsmm_forward_backward_matches_brute_force(seed: int) -> None:
    """`forward_backward`'s loglik equals `brute_force_loglik`'s, random `N=2, T=6, D_max=3`."""
    rng = np.random.default_rng(seed)
    n_states, t_len, d_max = 2, 6, 3

    log_rho = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.full((n_states, n_states), -np.inf)
    for i in range(n_states):
        targets = [j for j in range(n_states) if j != i]
        probs = rng.dirichlet(np.ones(len(targets)))
        for k, j in enumerate(targets):
            log_trans[i, j] = np.log(probs[k])
    log_dur_pmf = np.log(rng.dirichlet(np.ones(d_max), size=n_states))
    log_emis = -rng.uniform(0.1, 2.0, size=(t_len, n_states))

    loglik_fb, smoothed, elapsed = ref_hsmm.forward_backward(log_rho, log_trans, log_dur_pmf, log_emis, d_max)
    loglik_bf = ref_hsmm.brute_force_loglik(log_rho, log_trans, log_dur_pmf, log_emis, d_max)

    assert loglik_fb == pytest.approx(loglik_bf, abs=1e-12)
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(elapsed.sum(axis=(1, 2)), 1.0, atol=1e-10)


def test_e4_t01_ref_hsmm_backward_loglik_matches_forward() -> None:
    """`backward`'s loglik agrees with `forward`'s (computed independently, `<= 1e-10`)."""
    rng = np.random.default_rng(0)
    n_states, t_len, d_max = 3, 5, 4
    log_rho = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.full((n_states, n_states), -np.inf)
    for i in range(n_states):
        targets = [j for j in range(n_states) if j != i]
        probs = rng.dirichlet(np.ones(len(targets)))
        for k, j in enumerate(targets):
            log_trans[i, j] = np.log(probs[k])
    log_dur_pmf = np.log(rng.dirichlet(np.ones(d_max), size=n_states))
    log_emis = -rng.uniform(0.1, 2.0, size=(t_len, n_states))

    _, loglik_f = ref_hsmm.forward(log_rho, log_trans, log_dur_pmf, log_emis, d_max)
    _, loglik_b = ref_hsmm.backward(log_rho, log_trans, log_dur_pmf, log_emis, d_max)
    assert loglik_f == pytest.approx(loglik_b, abs=1e-10)


def test_e4_t01_ref_hsmm_segmental_viterbi_no_crash_and_valid_path() -> None:
    """`segmental_viterbi` returns a valid `(T,)` regime path (sanity, no independent oracle)."""
    rng = np.random.default_rng(2)
    n_states, t_len, d_max = 2, 6, 3
    log_rho = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.full((n_states, n_states), -np.inf)
    for i in range(n_states):
        targets = [j for j in range(n_states) if j != i]
        probs = rng.dirichlet(np.ones(len(targets)))
        for k, j in enumerate(targets):
            log_trans[i, j] = np.log(probs[k])
    log_dur_pmf = np.log(rng.dirichlet(np.ones(d_max), size=n_states))
    log_emis = -rng.uniform(0.1, 2.0, size=(t_len, n_states))

    path = ref_hsmm.segmental_viterbi(log_rho, log_trans, log_dur_pmf, log_emis, d_max)
    assert path.shape == (t_len,)
    assert set(np.unique(path)).issubset(set(range(n_states)))


def test_e4_t01_ref_hsmm_brute_force_2state_dmax3_hand_case() -> None:
    """Hand-set-up 2-state HSMM, `D_max=3`: brute force and `forward_backward` agree exactly."""
    d_max = 3
    log_rho = np.log(np.array([0.7, 0.3]))
    with np.errstate(divide="ignore"):
        log_trans = np.log(np.array([[0.0, 1.0], [1.0, 0.0]]))
    log_dur_pmf = np.log(np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]]))
    log_emis = np.log(np.array([[0.4, 0.1], [0.3, 0.2], [0.5, 0.1], [0.2, 0.4], [0.3, 0.3]]))

    loglik_fb, _smoothed, _elapsed = ref_hsmm.forward_backward(
        log_rho, log_trans, log_dur_pmf, log_emis, d_max
    )
    loglik_bf = ref_hsmm.brute_force_loglik(log_rho, log_trans, log_dur_pmf, log_emis, d_max)
    assert loglik_fb == pytest.approx(loglik_bf, abs=1e-12)


# --- AC3: ref_hmm on ref_expansion's r=1/order-1 chain == ref_hsmm w/ geometric, <= 1e-10 --------


@pytest.mark.parametrize("seed", range(5))
def test_e4_t01_phase_explicit_equivalence_r1_order1(seed: int) -> None:
    """`ref_hmm` on the `r=1`, order-1 expansion equals `ref_hsmm` with a geometric pmf, <= 1e-10."""
    rng = np.random.default_rng(seed)
    n_states, t_len, d_max = 3, 10, 300  # D_max large: truncation must be negligible at this tolerance
    order = 1
    r = [1] * n_states

    rate = rng.uniform(0.3, 0.8, size=n_states)  # moderate rate -> short typical durations
    a_of = 1.0 - np.exp(-rate * 1.0)
    log_dur_pmf = np.stack(
        [ref_dists.geometric_logpmf(np.arange(1, d_max + 1), a_of[j], d_max) for j in range(n_states)]
    )

    pi_rows = rng.dirichlet(np.ones(n_states - 1), size=n_states)
    log_trans = np.full((n_states, n_states), -np.inf)
    for i in range(n_states):
        targets = [j for j in range(n_states) if j != i]
        for k, j in enumerate(targets):
            log_trans[i, j] = np.log(pi_rows[i, k])

    log_rho = np.log(rng.dirichlet(np.ones(n_states)))
    log_emis = -rng.uniform(0.1, 2.0, size=(t_len, n_states))

    _, loglik_hsmm = ref_hsmm.forward(log_rho, log_trans, log_dur_pmf, log_emis, d_max)

    a_mat, states = ref_expansion.build_expanded_a(n_states, order, r, rate, pi_rows, 1.0)
    s_len = len(states)
    log_rho_s = np.full(s_len, -np.inf)
    for idx, (_memory, current, phase) in enumerate(states):
        if phase == 1:
            log_rho_s[idx] = log_rho[current]
    log_emis_s = np.stack([log_emis[:, current] for (_memory, current, _phase) in states], axis=1)
    log_a = np.log(np.stack([a_mat] * t_len))
    _, _, _, loglik_hmm = ref_hmm.forward_backward(log_rho_s, log_a, log_emis_s)

    assert loglik_hsmm == pytest.approx(loglik_hmm, abs=1e-10)


# --- AC4: each file < 300 lines, no JAX -----------------------------------------------------------


@pytest.mark.parametrize(
    "filename", ["ref_hmm.py", "ref_hsmm.py", "ref_expansion.py", "ref_dists.py", "__init__.py"]
)
def test_e4_t01_reference_files_under_300_lines_no_jax(filename: str) -> None:
    """Every `tests/reference/*.py` module is < 300 lines and never imports `jax` or `hsmmkit`."""
    text = (REFERENCE_DIR / filename).read_text(encoding="utf-8")
    n_lines = len(text.splitlines())
    assert n_lines < 300, f"{filename} has {n_lines} lines"
    assert "import jax" not in text
    assert "import hsmmkit" not in text
    assert "from hsmmkit" not in text
