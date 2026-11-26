"""Partition-based metrics of decoded regimes against reference labels (E14-T03).

`ari` and `nmi` are label-free (no matching, invariant to any relabelling); `hamming_matched` is the accuracy
after the optimal (Hungarian) matching of the predicted ids to the reference ids, `confusion_matched` the
matched confusion matrix as named fractions of the steps. Definitions and edge cases follow scikit-learn
(`adjusted_rand_score`; `normalized_mutual_info_score` with arithmetic-mean normalization).
"""

from __future__ import annotations

import numpy as np

from hsmmkit.evaluation.metrics.matching import confusion, hungarian
from hsmmkit.evaluation.metrics.registry import metric
from hsmmkit.evaluation.metrics.segmentation import _arrays

__all__ = ["adjusted_rand", "hamming_matched_score", "matched_confusion", "normalized_mutual_info"]


def _comb2(x: np.ndarray) -> np.ndarray:
    return np.asarray(x * (x - 1) / 2.0)


def adjusted_rand(pred: np.ndarray, true: np.ndarray) -> float:
    """Adjusted Rand index (Hubert-Arabie); 1.0 for identical partitions up to relabelling, ~0 for random."""
    c, _, _ = confusion(pred, true)
    n = c.sum()
    n_pred, n_true = c.shape
    if n_pred == n_true and (n_pred in (0, 1) or n_pred == n):
        return 1.0
    sum_cells = _comb2(c).sum()
    sum_rows, sum_cols = _comb2(c.sum(axis=1)).sum(), _comb2(c.sum(axis=0)).sum()
    expected = sum_rows * sum_cols / _comb2(np.asarray(n))
    maximum = (sum_rows + sum_cols) / 2.0
    if (
        maximum == expected
    ):  # degenerate partitions (e.g. one side a single cluster, the other all singletons)
        return 1.0
    return float((sum_cells - expected) / (maximum - expected))


def normalized_mutual_info(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalized mutual information with the arithmetic mean of the two entropies."""
    c, _, _ = confusion(pred, true)
    n = c.sum()
    if c.shape[0] == c.shape[1] == 1 or c.shape[0] == c.shape[1] == 0:
        return 1.0
    p_pred, p_true = c.sum(axis=1) / n, c.sum(axis=0) / n
    h_pred = -float(np.sum(p_pred * np.log(p_pred)))
    h_true = -float(np.sum(p_true * np.log(p_true)))
    joint = c / n
    nz = joint > 0
    outer = p_pred[:, None] * p_true[None, :]
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / outer[nz])))
    if mi <= 0.0:
        return 0.0
    return float(mi / np.mean([h_pred, h_true]))


def hamming_matched_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of steps whose predicted id equals the reference id after the optimal matching of ids."""
    c, _, _ = confusion(pred, true)
    rows, cols = hungarian(-c)
    return float(c[rows, cols].sum() / c.sum())


def matched_confusion(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """The confusion matrix as `{"true{k}/pred{i}": fraction of steps}`.

    The predicted ids are renamed to their matched reference ids (an unmatched predicted id keeps its own
    name, prefixed `u`).
    """
    c, pred_ids, ref_ids = confusion(pred, true)
    rows, cols = hungarian(-c)
    name = {int(i): f"{int(ref_ids[k])}" for i, k in zip(rows, cols, strict=True)}
    n = c.sum()
    out: dict[str, float] = {}
    for i in range(c.shape[0]):
        label = name.get(i, f"u{int(pred_ids[i])}")
        for k in range(c.shape[1]):
            if c[i, k] > 0:
                out[f"true{int(ref_ids[k])}/pred{label}"] = float(c[i, k] / n)
    return out


@metric("ari", {"posterior", "labels"}, better="higher")
def ari(**ctx: object) -> float:
    """Adjusted Rand index of the Viterbi regimes against `labels` (no matching needed)."""
    pred, true = _arrays(ctx, "ari")
    return adjusted_rand(pred, true)


@metric("nmi", {"posterior", "labels"}, better="higher")
def nmi(**ctx: object) -> float:
    """Normalized mutual information (arithmetic-mean normalization) of regimes and `labels`."""
    pred, true = _arrays(ctx, "nmi")
    return normalized_mutual_info(pred, true)


@metric("hamming_matched", {"posterior", "labels"}, better="higher")
def hamming_matched(**ctx: object) -> float:
    """Accuracy of the Viterbi regimes after Hungarian matching to `labels`."""
    pred, true = _arrays(ctx, "hamming_matched")
    return hamming_matched_score(pred, true)


@metric("confusion_matched", {"posterior", "labels"})
def confusion_matched(**ctx: object) -> dict[str, float]:
    """The matched confusion matrix as named fractions of the steps."""
    pred, true = _arrays(ctx, "confusion_matched")
    return matched_confusion(pred, true)
