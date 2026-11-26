"""Pure NumPy duration-dependent-jump HSMM reference (E12-T05): enumeration of every segmentation.

No `hsmmkit` import (CLAUDE.md §6). Uses the helpers of `ref_hsmm`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from tests.reference.ref_hsmm import _cum_log_emis, _log_survival, _seg_loglik


def brute_force_dd(
    log_rho: np.ndarray,
    log_trans_d: np.ndarray,
    log_dur_pmf: np.ndarray,
    log_emis: np.ndarray,
    d_max: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """`(loglik, smoothed (T, N), best regime path (T,))` with duration-dependent jumps, by enumeration.

    `log_trans_d[d - 1, i, j] = log pi_ij(d)`: the jump out of a segment of regime `i` that lasted `d` steps.
    Every segmentation of `[0, T)` is enumerated (right-censored last segment), so use `T <= 10`.
    """
    t_len, n_states = log_emis.shape
    cum = _cum_log_emis(log_emis)
    surv = _log_survival(log_dur_pmf, d_max)
    segmentations: list[tuple[float, list[tuple[int, int, int]]]] = []

    def recurse(
        pos: int, prev: tuple[int, int] | None, logprob: float, segs: list[tuple[int, int, int]]
    ) -> None:
        if pos == t_len:
            segmentations.append((logprob, segs))
            return
        for j in range(n_states):
            if prev is None:
                trans_lp = float(log_rho[j])
            else:
                if j == prev[0]:
                    continue
                trans_lp = float(log_trans_d[prev[1] - 1, prev[0], j])
            if not np.isfinite(trans_lp):
                continue
            for d in range(1, min(d_max, t_len - pos) + 1):
                dur_lp = surv[j, d] if pos + d == t_len else log_dur_pmf[j, d - 1]
                recurse(
                    pos + d,
                    (j, d),
                    logprob + trans_lp + dur_lp + _seg_loglik(cum, pos, pos + d - 1, j),
                    [*segs, (pos, pos + d, j)],
                )

    recurse(0, None, 0.0, [])
    logps = np.asarray([lp for lp, _ in segmentations])
    total = float(logsumexp(logps))
    smoothed = np.zeros((t_len, n_states))
    for lp, segs in segmentations:
        w = np.exp(lp - total)
        for a, b, j in segs:
            smoothed[a:b, j] += w
    best = segmentations[int(np.argmax(logps))][1]
    path = np.empty(t_len, dtype=np.int64)
    for a, b, j in best:
        path[a:b] = j
    return total, smoothed, path
