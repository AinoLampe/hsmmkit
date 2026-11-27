"""Pure NumPy reference E-step with explicit pairwise posteriors `xi` (E13-T01a).

No `hsmmkit` import (CLAUDE.md §6). Small instances: the `(T-1, S, S)` tensor `xi` is formed in full.
Index convention (CLAUDE.md §3.7): `log_a[t]` moves code index `t-1 -> t`. The reductions mirror the
definitions of `ExpandedEStats`: `stay` / `advance` / `occupancy` per duration-rate row, `jump` per
jump-matrix row and target regime, regime marginals `gamma_regime`, first-step posterior `gamma0`.
"""

from __future__ import annotations

import numpy as np

from tests.reference import ref_hmm


def pairwise_posteriors(
    log_rho: np.ndarray, log_a: np.ndarray, log_emis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """`(xi (T-1, S, S), gamma (T, S), loglik)` of one sequence; `xi[t-1]` is the pair `(t-1, t)`."""
    log_alpha, log_beta, log_gamma, loglik = ref_hmm.forward_backward(log_rho, log_a, log_emis)
    t_len, s_len = log_emis.shape
    xi = np.zeros((max(t_len - 1, 0), s_len, s_len))
    for t in range(1, t_len):
        with np.errstate(invalid="ignore"):
            log_xi = log_alpha[t - 1][:, None] + log_a[t] + (log_emis[t] + log_beta[t])[None, :] - loglik
        xi[t - 1] = np.where(np.isfinite(log_xi), np.exp(np.where(np.isfinite(log_xi), log_xi, 0.0)), 0.0)
    gamma = np.where(np.isfinite(log_gamma), np.exp(np.where(np.isfinite(log_gamma), log_gamma, 0.0)), 0.0)
    return xi, gamma, loglik


def estep_reference(
    log_rho: np.ndarray,
    log_a: np.ndarray,
    log_emis: np.ndarray,
    *,
    regime_of: np.ndarray,
    rate_index_of: np.ndarray,
    is_last_phase: np.ndarray,
    exit_targets: np.ndarray,
    row_of_state: np.ndarray,
    n_rates: int,
    n_rows: int,
    n_regimes: int,
) -> dict[str, np.ndarray | float]:
    """All accumulators of one sequence, from the explicit `xi` tensor."""
    xi, gamma, loglik = pairwise_posteriors(log_rho, log_a, log_emis)
    t_len, s_len = log_emis.shape
    stay, advance, occupancy = np.zeros(n_rates), np.zeros(n_rates), np.zeros(n_rates)
    jump = np.zeros((n_rows, n_regimes))
    for t in range(1, t_len):
        x = xi[t - 1]
        for s in range(s_len):
            k = rate_index_of[s]
            stay[k] += x[s, s]
            advance[k] += x[s].sum() - x[s, s]
            occupancy[k] += gamma[t - 1, s]
            if is_last_phase[s]:
                for j in range(n_regimes):
                    if exit_targets[s, j] >= 0:
                        jump[row_of_state[s], j] += x[s, exit_targets[s, j]]
    gamma_regime = np.zeros((t_len, n_regimes))
    for s in range(s_len):
        gamma_regime[:, regime_of[s]] += gamma[:, s]
    return {
        "loglik": loglik,
        "gamma_regime": gamma_regime,
        "gamma0": gamma[0],
        "stay": stay,
        "advance": advance,
        "occupancy": occupancy,
        "jump": jump,
    }
