"""Enumeration reference for the explicit-duration HSMM with per-segment-start rows (E4-T06a).

Pure NumPy (no `hsmmkit` import, CLAUDE.md §6). Kept apart from `ref_hsmm.py` (E4-T01 limits that
file to 300 lines); reuses its cumulative-emission and segment helpers.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from tests.reference.ref_hsmm import _cum_log_emis, _seg_loglik


def brute_force_by_start(
    log_rho: np.ndarray, log_pi_t: np.ndarray, log_pmf_t: np.ndarray, log_emis: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Enumeration reference with per-segment-start jump matrix and duration pmf (entry-time covariates).

    `log_pi_t (T, N, N)` and `log_pmf_t (T, N, D_max)` are indexed by the step at which a segment
    starts (`log_pi_t[s][i, j]` is the jump `i -> j` that begins a segment at `s`; row `0` is unused,
    `log_rho` starts the first segment). The last segment is right-censored (survival of its own start
    row). Returns `(loglik, elapsed (T, N, D_max), best regime path (T,))`, where
    `elapsed[t, j, e-1] = P(z_t = j, elapsed = e | y)`.
    """
    t_len, n_states = log_emis.shape
    d_max = log_pmf_t.shape[2]
    cum = _cum_log_emis(log_emis)
    total: list[float] = []
    bookkeeping: list[list[tuple[int, int, int]]] = []  # per segmentation: (start, regime, length)

    def recurse(pos: int, prev: int | None, logprob: float, segs: list[tuple[int, int, int]]) -> None:
        if pos == t_len:
            total.append(logprob)
            bookkeeping.append(list(segs))
            return
        for j in range(n_states):
            if prev is not None and j == prev:
                continue
            trans = log_rho[j] if prev is None else log_pi_t[pos, prev, j]
            if not np.isfinite(trans):
                continue
            for d in range(1, min(d_max, t_len - pos) + 1):
                if pos + d == t_len:
                    dur = float(logsumexp(log_pmf_t[pos, j, d - 1 :]))  # P(duration >= d)
                else:
                    dur = float(log_pmf_t[pos, j, d - 1])
                seg = _seg_loglik(cum, pos, pos + d - 1, j)
                recurse(pos + d, j, logprob + trans + dur + seg, [*segs, (pos, j, d)])

    recurse(0, None, 0.0, [])
    log_weights = np.asarray(total)
    loglik = float(logsumexp(log_weights))
    elapsed = np.zeros((t_len, n_states, d_max))
    for lw, segs in zip(log_weights, bookkeeping, strict=True):
        weight = float(np.exp(lw - loglik))
        for start, j, length in segs:
            for k in range(length):
                elapsed[start + k, j, min(k, d_max - 1)] += weight
    best = bookkeeping[int(np.argmax(log_weights))]
    path = np.zeros(t_len, dtype=np.int64)
    for start, j, length in best:
        path[start : start + length] = j
    return loglik, elapsed, path
