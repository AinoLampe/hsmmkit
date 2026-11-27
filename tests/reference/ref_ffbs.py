"""Exact posterior path probabilities of tiny models by enumeration (E13-T04).

No `hsmmkit` import (CLAUDE.md §6).

`expanded_path_probs`: every state path of a first-order chain,
`P(z | y) ~ rho[z_0] prod_t A_t[z_{t-1}, z_t] ell`.
`explicit_path_probs`: every regime path of an explicit-duration chain (consecutive segments differ, so a
regime path fixes the segmentation):
`P(z, y) = rho[z_0] prod_seg pmf(d) * emissions * prod pi * survival(last)`.
"""

from __future__ import annotations

import itertools

import numpy as np


def expanded_path_probs(rho: np.ndarray, trans: np.ndarray, ell: np.ndarray) -> dict[tuple[int, ...], float]:
    """`{path: posterior}`; `trans[t]` (`(T, S, S)`) moves `t-1 -> t`, `ell (T, S)` are log-emissions."""
    t_len, s_len = ell.shape
    weights: dict[tuple[int, ...], float] = {}
    for path in itertools.product(range(s_len), repeat=t_len):
        w = rho[path[0]] * np.exp(ell[0, path[0]])
        for t in range(1, t_len):
            w *= trans[t][path[t - 1], path[t]] * np.exp(ell[t, path[t]])
        weights[path] = float(w)
    total = sum(weights.values())
    return {p: w / total for p, w in weights.items()}


def explicit_path_probs(
    rho: np.ndarray, pi: np.ndarray, pmf: np.ndarray, ell: np.ndarray
) -> dict[tuple[int, ...], float]:
    """`{regime path: posterior}` of the right-censored explicit-duration model (`pmf (N, D)`)."""
    t_len, n = ell.shape
    d_max = pmf.shape[1]
    weights: dict[tuple[int, ...], float] = {}
    for path in itertools.product(range(n), repeat=t_len):
        runs: list[tuple[int, int]] = []  # (regime, length)
        for z in path:
            if runs and runs[-1][0] == z:
                runs[-1] = (z, runs[-1][1] + 1)
            else:
                runs.append((z, 1))
        w, start = rho[runs[0][0]], 0
        for k, (j, length) in enumerate(runs):
            w *= np.exp(ell[start : start + length, j].sum())
            if k > 0:
                w *= pi[runs[k - 1][0], j]
            if k + 1 < len(runs):
                w *= pmf[j, length - 1] if length <= d_max else 0.0
            else:  # the segment covering the last step: survival
                w *= pmf[j, length - 1 :].sum() if length <= d_max else 0.0
            start += length
        weights[path] = float(w)
    total = sum(weights.values())
    return {p: w / total for p, w in weights.items()}
