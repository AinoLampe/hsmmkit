"""Optimal assignment (Hungarian algorithm) and the label matching built on it (E14-T03, ADR-0020).

`hungarian(cost)` is a vendored `O(n^2 m)` shortest-augmenting-path implementation (Kuhn-Munkres with
potentials), so that evaluating segmentations needs no scipy at run time (`import scipy.optimize` costs
~370 ms, ADR-0020). It has the contract of `scipy.optimize.linear_sum_assignment`: a minimum-cost assignment
of every row to a distinct column of a rectangular matrix (or of every column to a distinct row if there are
more rows).
"""

from __future__ import annotations

import numpy as np

from hsmmkit.utils.errors import ConfigError

__all__ = ["confusion", "hungarian", "match_labels"]


def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment: `(row_ind, col_ind)` with `row_ind` ascending.

    `min(n, m)` pairs are returned for an `(n, m)` cost matrix; the sum of `cost[row_ind, col_ind]` is
    minimal.

    Raises:
        ConfigError: a non-2-D or non-finite cost matrix.
    """
    a = np.asarray(cost, dtype=np.float64)
    if a.ndim != 2 or not bool(np.all(np.isfinite(a))):
        raise ConfigError("hungarian needs a finite 2-D cost matrix", path="cost")
    n, m = a.shape
    if min(n, m) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    transposed = n > m
    if transposed:
        a, (n, m) = a.T, (m, n)
    inf = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    owner = np.zeros(
        m + 1, dtype=np.int64
    )  # owner[j] = row (1-based) assigned to column j (1-based), 0 = free
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = owner[j0]
            free = ~used[1:]
            cur = np.where(free, a[i0 - 1] - u[i0] - v[1:], inf)
            better = cur < minv[1:]
            minv[1:] = np.where(better, cur, minv[1:])
            way[1:] = np.where(better, j0, way[1:])
            candidates = np.where(free, minv[1:], inf)
            j1 = int(np.argmin(candidates)) + 1
            delta = candidates[j1 - 1]
            u[owner[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    cols = np.flatnonzero(owner[1:] > 0)
    rows = owner[1:][cols] - 1
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    if transposed:  # `rows` are the original columns: report ascending original rows
        rows, cols = cols, rows
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
    return rows, cols


def confusion(predicted: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(C, pred_ids, ref_ids)` with `C[i, k] = #{t: predicted_t = pred_ids[i], reference_t = ref_ids[k]}`."""
    p, r = np.asarray(predicted), np.asarray(reference)
    if p.shape != r.shape or p.ndim != 1:
        raise ConfigError(
            f"predicted {p.shape} and reference {r.shape} must be equal-length (T,) arrays",
            path="evaluation.labels",
        )
    pred_ids, pi = np.unique(p, return_inverse=True)
    ref_ids, ri = np.unique(r, return_inverse=True)
    c = np.zeros((pred_ids.shape[0], ref_ids.shape[0]))
    np.add.at(c, (pi, ri), 1.0)
    return c, pred_ids, ref_ids


def match_labels(predicted: np.ndarray, reference: np.ndarray) -> dict[int, int]:
    """Map predicted ids to reference ids maximizing the number of agreeing steps (Hungarian).

    Only `min(#pred ids, #ref ids)` predicted ids are mapped; the rest have no match.
    """
    c, pred_ids, ref_ids = confusion(predicted, reference)
    rows, cols = hungarian(-c)
    return {int(pred_ids[i]): int(ref_ids[k]) for i, k in zip(rows, cols, strict=True)}
