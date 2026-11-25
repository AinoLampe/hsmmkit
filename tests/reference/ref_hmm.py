"""Pure NumPy reference first-order HMM forward-backward and Viterbi.

No `hsmmkit` import (CLAUDE.md §6). Index convention (CLAUDE.md §3.7): code index `t` (0-based)
is math step `t+1`; `log_A[t]` moves from code index `t-1` to `t` (`t >= 1`; `log_A[0]` is unused).

Hand-computed example (2-state, 3-step, doctest-style ground truth for AC1):
`log_rho = log([0.6, 0.4])`, `log_A[1] = log_A[2] = log([[0.7,0.3],[0.4,0.6]])` (both steps use the
same matrix), `log_emis = log([[0.5,0.1],[0.4,0.2],[0.6,0.3]])`. By hand:

    alpha_1 = rho * b_1 = [0.6*0.5, 0.4*0.1] = [0.30, 0.04]
    alpha_2(1) = b_2(1) * (alpha_1(1)*A(1,1) + alpha_1(2)*A(2,1))
               = 0.4 * (0.30*0.7 + 0.04*0.4) = 0.4 * 0.226 = 0.0904
    alpha_2(2) = b_2(2) * (alpha_1(1)*A(1,2) + alpha_1(2)*A(2,2))
               = 0.2 * (0.30*0.3 + 0.04*0.6) = 0.2 * 0.114 = 0.0228
    alpha_3(1) = b_3(1) * (alpha_2(1)*A(1,1) + alpha_2(2)*A(2,1))
               = 0.6 * (0.0904*0.7 + 0.0228*0.4) = 0.6 * 0.07240 = 0.043440
    alpha_3(2) = b_3(2) * (alpha_2(1)*A(1,2) + alpha_2(2)*A(2,2))
               = 0.3 * (0.0904*0.3 + 0.0228*0.6) = 0.3 * 0.04080 = 0.012240
    loglik = log(0.043440 + 0.012240) = log(0.055680)

    beta_3 = [1, 1]
    beta_2(1) = A(1,1)*b_3(1)*beta_3(1) + A(1,2)*b_3(2)*beta_3(2) = 0.7*0.6 + 0.3*0.3 = 0.51
    beta_2(2) = A(2,1)*b_3(1)*beta_3(1) + A(2,2)*b_3(2)*beta_3(2) = 0.4*0.6 + 0.6*0.3 = 0.42
    beta_1(1) = A(1,1)*b_2(1)*beta_2(1) + A(1,2)*b_2(2)*beta_2(2) = 0.7*0.4*0.51 + 0.3*0.2*0.42
              = 0.1428 + 0.0252 = 0.1680
    beta_1(2) = A(2,1)*b_2(1)*beta_2(1) + A(2,2)*b_2(2)*beta_2(2) = 0.4*0.4*0.51 + 0.6*0.2*0.42
              = 0.0816 + 0.0504 = 0.1320
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

HAND_EXAMPLE_LOG_RHO = np.log(np.array([0.6, 0.4]))
with np.errstate(divide="ignore"):
    # log_A[0] (identity, unused) has off-diagonal zeros; log(0) = -inf is fine and expected.
    HAND_EXAMPLE_LOG_A = np.log(
        np.stack([np.eye(2), np.array([[0.7, 0.3], [0.4, 0.6]]), np.array([[0.7, 0.3], [0.4, 0.6]])])
    )
HAND_EXAMPLE_LOG_EMIS = np.log(np.array([[0.5, 0.1], [0.4, 0.2], [0.6, 0.3]]))
HAND_EXAMPLE_ALPHA = np.array([[0.30, 0.04], [0.0904, 0.0228], [0.043440, 0.012240]])
HAND_EXAMPLE_BETA = np.array([[0.1680, 0.1320], [0.51, 0.42], [1.0, 1.0]])
HAND_EXAMPLE_LOGLIK = float(np.log(0.055680))


def forward_backward(
    log_rho: np.ndarray, log_a: np.ndarray, log_emis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """`(log_alpha, log_beta, log_gamma, loglik)`.

    `log_rho (S,)`, `log_a (T, S, S)` (`log_a[t]` moves `t-1 -> t`), `log_emis (T, S)`.
    """
    t_len, s_len = log_emis.shape
    log_alpha = np.zeros((t_len, s_len))
    log_alpha[0] = log_rho + log_emis[0]
    for t in range(1, t_len):
        for s in range(s_len):
            log_alpha[t, s] = log_emis[t, s] + logsumexp(log_alpha[t - 1] + log_a[t][:, s])
    loglik = float(logsumexp(log_alpha[-1]))

    log_beta = np.zeros((t_len, s_len))
    log_beta[-1] = 0.0
    for t in range(t_len - 2, -1, -1):
        for s in range(s_len):
            log_beta[t, s] = logsumexp(log_a[t + 1][s, :] + log_emis[t + 1] + log_beta[t + 1])

    log_gamma = log_alpha + log_beta - loglik
    return log_alpha, log_beta, log_gamma, loglik


def viterbi(log_rho: np.ndarray, log_a: np.ndarray, log_emis: np.ndarray) -> np.ndarray:
    """Most-likely state path `(T,)`, via max-product (log-space) dynamic programming."""
    t_len, s_len = log_emis.shape
    delta = np.zeros((t_len, s_len))
    psi = np.zeros((t_len, s_len), dtype=np.int64)
    delta[0] = log_rho + log_emis[0]
    for t in range(1, t_len):
        for s in range(s_len):
            scores = delta[t - 1] + log_a[t][:, s]
            psi[t, s] = int(np.argmax(scores))
            delta[t, s] = scores[psi[t, s]] + log_emis[t, s]

    path = np.zeros(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(t_len - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path
