"""Pure NumPy explicit-duration HSMM forward-backward (Guedon 2003 / Yu 2010), Viterbi, brute force.

No `regimekit` import (CLAUDE.md §6). Segment-based (not elapsed-time-expanded) forward/backward,
with right-censoring of the final, possibly-incomplete segment.

Inputs: `log_rho (N,)` (initial state distribution), `log_trans (N, N)` (segment-to-segment jump
log-probabilities; diagonal `-inf`, since a "segment" already represents staying in a state --
there is no explicit self-transition here), `log_dur_pmf (N, D_max)` (`log_dur_pmf[j, d-1]` is
`log P(duration = d | state j)`, truncated/renormalized over `d = 1..D_max`), `log_emis (T, N)`
(per-step, per-state emission log-densities).

Recursion (concept refs §6.4.1, Appendix C): for segment boundary `tau in {0, ..., T}` (`tau = 0` is
"before the first observation"), `inner(tau, j) = log_rho[j]` if `tau == 0`, else
`logsumexp_i(end_prob(tau, i) + log_trans[i, j])`. `end_prob(t, j) = logsumexp_d(inner(t - d, j) +
log_dur_pmf[j, d-1] + sum_{u=t-d+1}^{t} log_emis[u, j])` for `d = 1..min(D_max, t)`. The (censored)
total log-likelihood sums, over the *last* segment's `(j, d)`, `inner(T-d, j) + log_survival[j, d]
+ seg_loglik`, using `P(duration >= d)` instead of `P(duration = d)` since the true length of the
final segment need not be fully observed.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _cum_log_emis(log_emis: np.ndarray) -> np.ndarray:
    """`(T+1, N)`: `cum[e+1, j] - cum[s, j] == sum_{u=s}^{e} log_emis[u, j]` (0-indexed inclusive)."""
    return np.concatenate([np.zeros((1, log_emis.shape[1])), np.cumsum(log_emis, axis=0)], axis=0)


def _log_survival(log_dur_pmf: np.ndarray, d_max: int) -> np.ndarray:
    """`(N, D_max + 2)`: `surv[:, d] = log P(duration >= d)`, `d = 1..D_max`; `surv[:, D_max+1] = -inf`."""
    n_states = log_dur_pmf.shape[0]
    pmf = np.exp(log_dur_pmf)  # (N, D_max)
    cum_from_end = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]  # cum_from_end[:, k] = sum_{i>=k} pmf[:, i]
    surv = np.zeros((n_states, d_max + 2))
    surv[:, 1 : d_max + 1] = cum_from_end
    with np.errstate(divide="ignore"):
        return np.log(surv)


def _seg_loglik(cum: np.ndarray, s0: int, e0: int, j: int) -> float:
    """`sum_{u=s0}^{e0} log_emis[u, j]` (0-indexed inclusive)."""
    return float(cum[e0 + 1, j] - cum[s0, j])


def forward(
    log_rho: np.ndarray, log_trans: np.ndarray, log_dur_pmf: np.ndarray, log_emis: np.ndarray, d_max: int
) -> tuple[np.ndarray, float]:
    """`(end_prob (T+1, N), loglik)`.

    `end_prob[t, j]` is the (uncensored) log-prob of a segment ending exactly at boundary `t` in
    state `j`; `loglik` is the total (right-censored) log-likelihood.
    """
    t_len, n_states = log_emis.shape
    cum = _cum_log_emis(log_emis)
    end_prob = np.full((t_len + 1, n_states), -np.inf)

    def inner(tau: int, j: int) -> float:
        if tau == 0:
            return float(log_rho[j])
        return float(logsumexp(end_prob[tau] + log_trans[:, j]))

    for t in range(1, t_len + 1):
        for j in range(n_states):
            terms = [
                inner(t - d, j) + log_dur_pmf[j, d - 1] + _seg_loglik(cum, t - d, t - 1, j)
                for d in range(1, min(d_max, t) + 1)
            ]
            end_prob[t, j] = logsumexp(terms) if terms else -np.inf

    surv = _log_survival(log_dur_pmf, d_max)
    censored_terms = [
        inner(t_len - d, j) + surv[j, d] + _seg_loglik(cum, t_len - d, t_len - 1, j)
        for j in range(n_states)
        for d in range(1, min(d_max, t_len) + 1)
    ]
    loglik = float(logsumexp(censored_terms))
    return end_prob, loglik


def _continuation(
    tau: int,
    j: int,
    beta: np.ndarray,
    surv: np.ndarray,
    log_dur_pmf: np.ndarray,
    cum: np.ndarray,
    t_len: int,
    d_max: int,
) -> float:
    """`cont(tau, j)`: total log-prob of everything after boundary `tau`.

    Given a segment `j` starts right after it (sums over every duration `d`, closed segments ending
    before `T` plus one right-censored branch for durations that would run past `T`).
    """
    terms = [
        log_dur_pmf[j, d - 1] + _seg_loglik(cum, tau, tau + d - 1, j) + beta[tau + d, j]
        for d in range(1, min(d_max, t_len - tau) + 1)
    ]
    censored_d = t_len - tau + 1
    if censored_d <= d_max:
        terms.append(surv[j, censored_d] + _seg_loglik(cum, tau, t_len - 1, j))
    return float(logsumexp(terms)) if terms else -np.inf


def backward(
    log_rho: np.ndarray, log_trans: np.ndarray, log_dur_pmf: np.ndarray, log_emis: np.ndarray, d_max: int
) -> tuple[np.ndarray, float]:
    """`(beta (T+1, N), loglik)`.

    `beta[tau, i]` is the log-prob of everything after boundary `tau`, given the segment ending
    there was in state `i` (`beta[T, :] = 0`); `loglik` matches `forward`'s.
    """
    t_len, n_states = log_emis.shape
    cum = _cum_log_emis(log_emis)
    surv = _log_survival(log_dur_pmf, d_max)
    beta = np.full((t_len + 1, n_states), -np.inf)
    beta[t_len] = 0.0

    for tau in range(t_len - 1, -1, -1):
        cont = np.array(
            [_continuation(tau, j, beta, surv, log_dur_pmf, cum, t_len, d_max) for j in range(n_states)]
        )
        for i in range(n_states):
            terms = log_trans[i, :] + cont
            beta[tau, i] = logsumexp(terms[np.isfinite(terms)]) if np.any(np.isfinite(terms)) else -np.inf

    cont0 = np.array(
        [_continuation(0, j, beta, surv, log_dur_pmf, cum, t_len, d_max) for j in range(n_states)]
    )
    loglik = float(logsumexp(log_rho + cont0))
    return beta, loglik


def forward_backward(
    log_rho: np.ndarray, log_trans: np.ndarray, log_dur_pmf: np.ndarray, log_emis: np.ndarray, d_max: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """`(loglik, smoothed (T, N), elapsed_posterior (T, N, D_max))`.

    `smoothed[t, j] = P(z_t = j | y)`; `elapsed_posterior[t, j, e-1] = P(z_t = j, elapsed = e | y)`
    (`elapsed`: steps since the current segment started, `1..D_max`). Computed by enumerating every
    segment `(state, start, duration)` -- closed (duration observed exactly) or right-censored (the
    final segment, duration only known to be `>= ` its observed length) -- weighted by
    `inner(start-1, state) + duration-term + segment-loglik + continuation`, normalized by `loglik`.
    """
    t_len, n_states = log_emis.shape
    cum = _cum_log_emis(log_emis)
    end_prob, loglik_f = forward(log_rho, log_trans, log_dur_pmf, log_emis, d_max)
    beta, loglik_b = backward(log_rho, log_trans, log_dur_pmf, log_emis, d_max)
    assert abs(loglik_f - loglik_b) < 1e-8, (loglik_f, loglik_b)
    surv = _log_survival(log_dur_pmf, d_max)

    def inner(tau: int, j: int) -> float:
        if tau == 0:
            return float(log_rho[j])
        return float(logsumexp(end_prob[tau] + log_trans[:, j]))

    smoothed = np.full((t_len, n_states), -np.inf)
    elapsed = np.full((t_len, n_states, d_max), -np.inf)
    for s0 in range(t_len):
        for j in range(n_states):
            prior = inner(s0, j)
            for d in range(1, min(d_max, t_len - s0) + 1):
                e0 = s0 + d - 1
                weight = prior + log_dur_pmf[j, d - 1] + _seg_loglik(cum, s0, e0, j) + beta[e0 + 1, j]
                for t in range(s0, e0 + 1):
                    smoothed[t, j] = np.logaddexp(smoothed[t, j], weight)
                    elapsed[t, j, t - s0] = np.logaddexp(elapsed[t, j, t - s0], weight)
            censored_d = t_len - s0 + 1
            if censored_d <= d_max:
                weight = prior + surv[j, censored_d] + _seg_loglik(cum, s0, t_len - 1, j)
                for t in range(s0, t_len):
                    smoothed[t, j] = np.logaddexp(smoothed[t, j], weight)
                    elapsed[t, j, t - s0] = np.logaddexp(elapsed[t, j, t - s0], weight)

    return loglik_f, np.exp(smoothed - loglik_f), np.exp(elapsed - loglik_f)


def brute_force_loglik(
    log_rho: np.ndarray, log_trans: np.ndarray, log_dur_pmf: np.ndarray, log_emis: np.ndarray, d_max: int
) -> float:
    """Total log-likelihood by literal enumeration of every segmentation of `[0, T)`."""
    t_len, n_states = log_emis.shape
    cum = _cum_log_emis(log_emis)
    surv = _log_survival(log_dur_pmf, d_max)
    results: list[float] = []

    def recurse(pos: int, prev_state: int | None, logprob: float) -> None:
        if pos == t_len:
            results.append(logprob)
            return
        for j in range(n_states):
            if prev_state is not None and j == prev_state:
                continue
            trans_lp = log_rho[j] if prev_state is None else log_trans[prev_state, j]
            if not np.isfinite(trans_lp):
                continue
            remaining = t_len - pos
            for d in range(1, min(d_max, remaining) + 1):
                seg_lp = _seg_loglik(cum, pos, pos + d - 1, j)
                dur_lp = surv[j, d] if pos + d == t_len else log_dur_pmf[j, d - 1]
                recurse(pos + d, j, logprob + trans_lp + dur_lp + seg_lp)

    recurse(0, None, 0.0)
    return float(logsumexp(results))


def segmental_viterbi(
    log_rho: np.ndarray, log_trans: np.ndarray, log_dur_pmf: np.ndarray, log_emis: np.ndarray, d_max: int
) -> np.ndarray:
    """Most-likely per-step regime path `(T,)`, right-censored last segment.

    Max-product over segmentations, via segmental dynamic programming (max in place of
    `logsumexp` in `forward`).
    """
    t_len, n_states = log_emis.shape
    cum = _cum_log_emis(log_emis)
    best_end = np.full((t_len + 1, n_states), -np.inf)  # best[t, j]: best score of a segment ending at t in j
    back_d = np.zeros((t_len + 1, n_states), dtype=np.int64)
    back_prev = np.full((t_len + 1, n_states), -1, dtype=np.int64)

    def best_inner(tau: int, j: int) -> tuple[float, int]:
        if tau == 0:
            return float(log_rho[j]), -1
        scores = best_end[tau] + log_trans[:, j]
        i = int(np.argmax(scores))
        return float(scores[i]), i

    for t in range(1, t_len + 1):
        for j in range(n_states):
            best_score, best_d = -np.inf, 1
            best_prev = -1
            for d in range(1, min(d_max, t) + 1):
                tau = t - d
                inner_score, prev = best_inner(tau, j)
                score = inner_score + log_dur_pmf[j, d - 1] + _seg_loglik(cum, t - d, t - 1, j)
                if score > best_score:
                    best_score, best_d, best_prev = score, d, prev
            best_end[t, j] = best_score
            back_d[t, j] = best_d
            back_prev[t, j] = best_prev

    surv = _log_survival(log_dur_pmf, d_max)
    final_score, final_j, final_d, final_prev = -np.inf, 0, 1, -1
    for j in range(n_states):
        for d in range(1, min(d_max, t_len) + 1):
            tau = t_len - d
            inner_score, prev = best_inner(tau, j)
            score = inner_score + surv[j, d] + _seg_loglik(cum, t_len - d, t_len - 1, j)
            if score > final_score:
                final_score, final_j, final_d, final_prev = score, j, d, prev

    path = np.zeros(t_len, dtype=np.int64)
    t, j, prev, d = t_len, final_j, final_prev, final_d
    while True:
        path[t - d : t] = j
        t = t - d
        if prev == -1:
            break
        j = prev
        d = int(back_d[t, j])
        prev = int(back_prev[t, j])
    return path
