"""NumPy references for supervision masks (E12-T06): labels and boundaries applied to the joint likelihood.

Pure NumPy/SciPy; never imports `hsmmkit`. Conventions: `switch` / `stay` are sets of steps `t`
(`1 <= t <= T-1`)
at which a regime change between `t-1` and `t` is required / forbidden; `allowed (T, N)` bool are the labels.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def masked_forward(
    log_rho: np.ndarray,
    log_a: np.ndarray,
    log_emis: np.ndarray,
    regime_of: np.ndarray,
    allowed: np.ndarray | None,
    switch: set[int],
    stay: set[int],
) -> tuple[np.ndarray, float]:
    """`(log_alpha (T, S), loglik)` of an expanded HMM with a constant transition matrix and the masks."""
    t_len, s = log_emis.shape
    ell = log_emis.copy()
    if allowed is not None:
        ell = np.where(allowed[:, regime_of], ell, -np.inf)
    change = regime_of[:, None] != regime_of[None, :]
    alpha = np.full((t_len, s), -np.inf)
    alpha[0] = log_rho + ell[0]
    for t in range(1, t_len):
        a = log_a.copy()
        if t in switch:
            a = np.where(change, a, -np.inf)
        if t in stay:
            a = np.where(change, -np.inf, a)
        with np.errstate(invalid="ignore"):
            alpha[t] = ell[t] + logsumexp(alpha[t - 1][:, None] + a, axis=0)
    return alpha, float(logsumexp(alpha[-1]))


def masked_viterbi_score(
    log_rho: np.ndarray,
    log_a: np.ndarray,
    log_emis: np.ndarray,
    regime_of: np.ndarray,
    allowed: np.ndarray | None,
    switch: set[int],
    stay: set[int],
) -> float:
    """Best joint log-probability of one state path under the masks (max-product)."""
    t_len = log_emis.shape[0]
    ell = log_emis if allowed is None else np.where(allowed[:, regime_of], log_emis, -np.inf)
    change = regime_of[:, None] != regime_of[None, :]
    delta = log_rho + ell[0]
    for t in range(1, t_len):
        a = log_a.copy()
        if t in switch:
            a = np.where(change, a, -np.inf)
        if t in stay:
            a = np.where(change, -np.inf, a)
        delta = ell[t] + np.max(delta[:, None] + a, axis=0)
    return float(np.max(delta))


def _log_survival(log_pmf: np.ndarray) -> np.ndarray:
    """`surv[j, d] = log P(duration >= d)` for `d = 1..D` (index `d`, column 0 unused)."""
    n, d_max = log_pmf.shape
    out = np.full((n, d_max + 2), -np.inf)
    for d in range(d_max, 0, -1):
        out[:, d] = np.logaddexp(out[:, d + 1], log_pmf[:, d - 1])
    return out


def explicit_brute_force(
    log_rho: np.ndarray,
    log_trans: np.ndarray,
    log_pmf: np.ndarray,
    log_emis: np.ndarray,
    allowed: np.ndarray | None,
    switch: set[int],
    stay: set[int],
) -> float:
    """Log-likelihood of the explicit-duration model by enumerating every segmentation.

    The last segment is right-censored (survival). A segmentation is feasible iff every segment's regime is
    allowed at each of its steps, every `switch` step starts a segment and no `stay` step does.
    """
    t_len, n = log_emis.shape
    d_max = log_pmf.shape[1]
    surv = _log_survival(log_pmf)
    results: list[float] = []

    def recurse(pos: int, prev: int | None, logprob: float, starts: tuple[int, ...]) -> None:
        if pos == t_len:
            if switch <= set(starts) and not (stay & set(starts)):
                results.append(logprob)
            return
        for j in range(n):
            if prev is not None and j == prev:
                continue
            trans = log_rho[j] if prev is None else log_trans[prev, j]
            if not np.isfinite(trans):
                continue
            for d in range(1, min(d_max, t_len - pos) + 1):
                if allowed is not None and not bool(np.all(allowed[pos : pos + d, j])):
                    break
                dur = surv[j, d] if pos + d == t_len else log_pmf[j, d - 1]
                recurse(
                    pos + d,
                    j,
                    logprob + trans + dur + float(log_emis[pos : pos + d, j].sum()),
                    (*starts, pos),
                )

    recurse(0, None, 0.0, ())
    return float(logsumexp(results)) if results else -np.inf
