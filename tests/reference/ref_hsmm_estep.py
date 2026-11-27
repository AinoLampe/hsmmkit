"""Pure NumPy explicit-duration E-step references: brute force and a probability-space FB (E13-T01b).

No `hsmmkit` import (CLAUDE.md §6). Model: initial regime `rho`, jump matrix `pi` (zero diagonal;
`(T, N, N)` when indexed by segment start), duration pmf `pmf (N, D)` or `(T, N, D)` per start, regime-wise
emissions `ell (T, N)` (log). A segment of regime `j` that starts at `s` and lasts `d` steps has weight
`pmf[j, d-1] * prod ell`. The segment covering the last step is right-censored: it carries the survival
`P(duration >= T - s)` (`0` if `T - s > D`).
Outputs: `loglik`, `eta` (closed segments by `(j, d)`), `cens` (final segment by elapsed `d = T - s`), `first`
(closed first segments when `first_residual`), `xi` (jumps), `gamma` (regime marginals), `rho` (first regime).
"""

from __future__ import annotations

import numpy as np


def _tab(x: np.ndarray, s: int) -> np.ndarray:
    return x[s] if x.ndim == 3 else x


def _surv(pmf_s: np.ndarray, r: int) -> np.ndarray:
    """`(N,)` `P(duration >= r)`, `0` beyond `D`."""
    d = pmf_s.shape[1]
    return pmf_s[:, r - 1 :].sum(axis=1) if r <= d else np.zeros(pmf_s.shape[0])


def _emit(ell: np.ndarray, s: int, d: int) -> np.ndarray:
    return np.exp(ell[s : s + d].sum(axis=0))


def _empty(t: int, n: int, d: int) -> dict[str, np.ndarray | float]:
    return {
        "eta": np.zeros((n, d)),
        "cens": np.zeros((n, d)),
        "first": np.zeros((n, d)),
        "xi": np.zeros((n, n)),
        "gamma": np.zeros((t, n)),
        "rho": np.zeros(n),
    }


def brute_force(
    rho: np.ndarray, pi: np.ndarray, pmf: np.ndarray, ell: np.ndarray, *, first_residual: bool = False
) -> dict[str, np.ndarray | float]:
    """Enumerate every segmentation (`T <= 8`): exact expectations by direct summation."""
    t_len, n = ell.shape
    d_max = pmf.shape[-1]
    runs: list[tuple[float, list[tuple[int, int, int, bool]]]] = []

    def rec(s: int, prev: int | None, prob: float, segs: list[tuple[int, int, int, bool]]) -> None:
        for j in range(n):
            w = rho[j] if prev is None else _tab(pi, s)[prev, j]
            if w == 0.0:
                continue
            pmf_s = _tab(pmf, s)
            for d in range(1, d_max + 1):
                if s + d < t_len:  # a closed segment, followed by another one
                    p = prob * w * pmf_s[j, d - 1] * _emit(ell, s, d)[j]
                    if p > 0:
                        rec(s + d, j, p, [*segs, (j, s, d, True)])
            r = t_len - s
            if r <= d_max:  # the segment covering the last step (survival)
                p = prob * w * _surv(pmf_s, r)[j] * _emit(ell, s, r)[j]
                if p > 0:
                    runs.append((p, [*segs, (j, s, r, False)]))

    rec(0, None, 1.0, [])
    z = sum(p for p, _ in runs)
    out = _empty(t_len, n, d_max)
    out["loglik"] = float(np.log(z))
    for p, segs in runs:
        w = p / z
        out["rho"][segs[0][0]] += w  # type: ignore[index]
        for k, (j, s, d, closed) in enumerate(segs):
            out["gamma"][s : s + d, j] += w  # type: ignore[index]
            if closed:
                key = "first" if (k == 0 and first_residual) else "eta"
                out[key][j, d - 1] += w  # type: ignore[index]
            else:
                out["cens"][j, min(d, d_max) - 1] += w  # type: ignore[index]
            if k + 1 < len(segs):
                out["xi"][j, segs[k + 1][0]] += w  # type: ignore[index]
    return out


def forward_backward(
    rho: np.ndarray, pi: np.ndarray, pmf: np.ndarray, ell: np.ndarray, *, first_residual: bool = False
) -> dict[str, np.ndarray | float]:
    """Guedon-style forward-backward in probability space (`T <= ~150`)."""
    t_len, n = ell.shape
    d_max = pmf.shape[-1]
    # alpha[t, j, d-1]: P(y_{<t}, a segment of j with duration d ends at boundary t), t = 1 .. T - 1
    alpha = np.zeros((t_len + 1, n, d_max))
    start = np.zeros((t_len, n))  # P(y_{<s}, a segment of j starts at s)
    for s in range(t_len):
        if s == 0:
            start[0] = rho
        else:
            ended = alpha[s].sum(axis=1)  # (N,) segments ending at s
            start[s] = ended @ _tab(pi, s)
        pmf_s = _tab(pmf, s)
        for d in range(1, d_max + 1):
            if s + d < t_len:
                alpha[s + d, :, d - 1] += start[s] * pmf_s[:, d - 1] * _emit(ell, s, d)
    # cont[s, j]: P(y_{>=s} | a segment of j starts at s)
    beta = np.zeros((t_len + 1, n))  # P(y_{>=t} | a segment ended at t)
    cont = np.zeros((t_len, n))
    for s in range(t_len - 1, -1, -1):
        pmf_s = _tab(pmf, s)
        c = _surv(pmf_s, t_len - s) * _emit(ell, s, t_len - s) if t_len - s <= d_max else np.zeros(n)
        for d in range(1, d_max + 1):
            if s + d < t_len:
                c = c + pmf_s[:, d - 1] * _emit(ell, s, d) * beta[s + d]
        cont[s] = c
        beta[s] = _tab(pi, s) @ c if s > 0 else 0.0
    z = float(rho @ cont[0])
    out = _empty(t_len, n, d_max)
    out["loglik"] = float(np.log(z))
    out["rho"] = rho * cont[0] / z
    for s in range(t_len):
        pmf_s = _tab(pmf, s)
        for d in range(1, d_max + 1):
            if s + d < t_len:
                w = start[s] * pmf_s[:, d - 1] * _emit(ell, s, d) * beta[s + d] / z
                out["first" if (s == 0 and first_residual) else "eta"][:, d - 1] += w  # type: ignore[index]
                out["gamma"][s : s + d] += w[None, :]  # type: ignore[index]
        r = t_len - s
        if r <= d_max:
            w = start[s] * _surv(pmf_s, r) * _emit(ell, s, r) / z
            out["cens"][:, r - 1] += w  # type: ignore[index]
            out["gamma"][s:] += w[None, :]  # type: ignore[index]
    for tau in range(1, t_len):
        ended = alpha[tau].sum(axis=1)
        out["xi"] += (ended[:, None] * _tab(pi, tau) * cont[tau][None, :]) / z  # type: ignore[operator]
    return out


__all__ = ["brute_force", "forward_backward"]
