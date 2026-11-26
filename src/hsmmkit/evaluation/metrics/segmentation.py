"""Boundary- and segment-based metrics of a decoded segmentation against reference labels (E14-T03).

All metrics work on one sequence of integer regime ids `q_true (T,)`, `q_pred (T,)` (the Viterbi regimes); the
registry entries read `ctx["posterior"].viterbi` and `ctx["labels"]` (and `ctx["boundary_tolerance"]`, default
`evaluation.boundary_tolerance` = 5). Per-sequence values are aggregated over sequences by the
**length-weighted mean** (`aggregate_sequences`); `nan` entries (a `hausdorff` with exactly one empty boundary
set) are skipped, and the aggregate is `nan` only if every entry is.

Boundaries are `B(q) = {t >= 1 : q_t != q_{t-1}}` (code index `t`, the first step of a new segment).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from hsmmkit.evaluation.metrics.registry import metric
from hsmmkit.inference.results import PosteriorResult
from hsmmkit.utils.errors import ConfigError

__all__ = [
    "aggregate_sequences",
    "boundaries",
    "boundary_scores",
    "count_matches",
    "covering_score",
    "hausdorff_distance",
]

DEFAULT_TOLERANCE = 5


def boundaries(q: np.ndarray) -> np.ndarray:
    """`(K,)` ascending steps `t >= 1` where `q_t != q_{t-1}`."""
    q = np.asarray(q)
    return np.flatnonzero(q[1:] != q[:-1]) + 1


def count_matches(pred: np.ndarray, true: np.ndarray, tolerance: int) -> int:
    """Size of a maximum one-to-one matching of predicted to true boundaries with `|p - t| <= tolerance`.

    A single sweep over both sorted lists is optimal: if the smallest unmatched predicted boundary `p` and the
    smallest unmatched true boundary `t` are within the tolerance, matching them is never worse (any optimal
    matching can be exchanged to contain the pair: every boundary `p` could still reach is `>= p`, every one
    `t` could still reach is `>= t`, and the compatibility intervals `[x - tol, x + tol]` are ordered); if
    `p < t - tol`, `p` can match nothing that remains and is dropped, symmetrically for `t`.

    Note: a greedy by *globally* smallest distance is not optimal (`pred = {1, 2}`, `true = {2, 3}`,
    tolerance 1 would match `2 <-> 2` and leave `1` and `3` unmatched), so the sweep is used.
    """
    p, t = np.sort(np.asarray(pred)), np.sort(np.asarray(true))
    i = j = matched = 0
    while i < p.shape[0] and j < t.shape[0]:
        if abs(int(p[i]) - int(t[j])) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif p[i] < t[j]:
            i += 1
        else:
            j += 1
    return matched


def boundary_scores(q_pred: np.ndarray, q_true: np.ndarray, tolerance: int) -> tuple[float, float, float]:
    """`(precision, recall, f1)` of the predicted boundaries at `tolerance`.

    Both sets empty: `(1, 1, 1)`; otherwise an empty side scores 0 (precision with no predicted boundary,
    recall with no true boundary); `f1 = 2 p r / (p + r)` (0 if `p + r = 0`).
    """
    pred, true = boundaries(q_pred), boundaries(q_true)
    if pred.size == 0 and true.size == 0:
        return 1.0, 1.0, 1.0
    matched = count_matches(pred, true, tolerance)
    precision = matched / pred.size if pred.size else 0.0
    recall = matched / true.size if true.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def hausdorff_distance(q_pred: np.ndarray, q_true: np.ndarray) -> float:
    """Hausdorff distance `max(max_p min_t |p - t|, max_t min_p |p - t|)` between the boundary sets.

    `nan` if exactly one set is empty (the distance is undefined), `0` if both are.
    """
    pred, true = boundaries(q_pred), boundaries(q_true)
    if pred.size == 0 and true.size == 0:
        return 0.0
    if pred.size == 0 or true.size == 0:
        return float("nan")
    d = np.abs(pred[:, None] - true[None, :])
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _segments(q: np.ndarray) -> list[tuple[int, int]]:
    edges = np.concatenate(([0], boundaries(q), [q.shape[0]]))
    return [(int(a), int(b)) for a, b in itertools.pairwise(edges)]


def covering_score(q_pred: np.ndarray, q_true: np.ndarray) -> float:
    """Covering `C = (1/T) sum_{A in segs(q_true)} |A| max_{B in segs(q_pred)} |A cap B| / |A cup B|`.

    Segments are runs by position, so equal ids in separate runs are separate segments.
    """
    q_pred, q_true = np.asarray(q_pred), np.asarray(q_true)
    t_len = q_true.shape[0]
    pred = _segments(q_pred)
    total = 0.0
    for a0, a1 in _segments(q_true):
        best = 0.0
        for b0, b1 in pred:
            inter = max(0, min(a1, b1) - max(a0, b0))
            if inter:
                best = max(best, inter / ((a1 - a0) + (b1 - b0) - inter))
        total += (a1 - a0) * best
    return float(total / t_len)


def aggregate_sequences(values: Sequence[float], lengths: Sequence[int]) -> float:
    """Length-weighted mean of per-sequence values, skipping `nan` entries (`nan` if all are)."""
    v, w = np.asarray(values, dtype=float), np.asarray(lengths, dtype=float)
    ok = ~np.isnan(v)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def _arrays(ctx: dict[str, object], name: str) -> tuple[np.ndarray, np.ndarray]:
    post, labels = ctx.get("posterior"), ctx.get("labels")
    if not isinstance(post, PosteriorResult):
        raise ConfigError(f"{name} needs ctx['posterior']", path="evaluation.metrics")
    if labels is None:
        raise ConfigError(f"{name} needs reference labels (ctx['labels'])", path="evaluation.labels")
    pred, true = np.asarray(post.viterbi), np.asarray(labels)
    if pred.shape != true.shape or pred.ndim != 1:
        raise ConfigError(
            f"predicted {pred.shape} and labels {true.shape} must be equal-length (T,) arrays",
            path="evaluation.labels",
        )
    return pred, true


def _tolerance(ctx: dict[str, object]) -> int:
    tol = ctx.get("boundary_tolerance", DEFAULT_TOLERANCE)
    if isinstance(tol, bool) or not isinstance(tol, int | np.integer) or tol < 0:
        raise ConfigError("boundary_tolerance must be an integer >= 0", path="evaluation.boundary_tolerance")
    return int(tol)


@metric("boundary_precision", {"posterior", "labels"}, better="higher")
def boundary_precision(**ctx: object) -> float:
    """Fraction of predicted boundaries matched to a true one within `boundary_tolerance`."""
    pred, true = _arrays(ctx, "boundary_precision")
    return boundary_scores(pred, true, _tolerance(ctx))[0]


@metric("boundary_recall", {"posterior", "labels"}, better="higher")
def boundary_recall(**ctx: object) -> float:
    """Fraction of true boundaries matched to a predicted one within `boundary_tolerance`."""
    pred, true = _arrays(ctx, "boundary_recall")
    return boundary_scores(pred, true, _tolerance(ctx))[1]


@metric("boundary_f1", {"posterior", "labels"}, better="higher")
def boundary_f1(**ctx: object) -> float:
    """Harmonic mean of boundary precision and recall at `boundary_tolerance`."""
    pred, true = _arrays(ctx, "boundary_f1")
    return boundary_scores(pred, true, _tolerance(ctx))[2]


@metric("hausdorff", {"posterior", "labels"}, better="lower")
def hausdorff(**ctx: object) -> float:
    """Hausdorff distance between the predicted and true boundary sets (steps); `nan` if one is empty."""
    pred, true = _arrays(ctx, "hausdorff")
    return hausdorff_distance(pred, true)


@metric("covering", {"posterior", "labels"}, better="higher")
def covering(**ctx: object) -> float:
    """Covering of the true segmentation by the predicted one (1 = identical segmentations)."""
    pred, true = _arrays(ctx, "covering")
    return covering_score(pred, true)
