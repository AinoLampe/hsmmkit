"""Label-based metrics (E7-T01): `balanced_accuracy` of the Viterbi regimes against reference labels."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from regimekit.evaluation.metrics.registry import metric
from regimekit.inference.results import PosteriorResult
from regimekit.utils.errors import ConfigError

MAX_CLASSES = 6


def balanced_accuracy_matched(predicted: np.ndarray, labels: np.ndarray) -> float:
    """Balanced accuracy (mean per-label recall) under the best relabelling of `predicted`.

    Regime ids are arbitrary, so the predicted ids are matched to the label ids by brute force over all
    permutations (scipy is dev-only, so no Hungarian algorithm). The value is invariant to any
    permutation of either id set.

    Raises:
        ConfigError: more than 6 distinct ids (the brute force would be too slow), or a length mismatch.
    """
    predicted, labels = np.asarray(predicted), np.asarray(labels)
    if predicted.shape != labels.shape or predicted.ndim != 1:
        raise ConfigError(
            f"predicted {predicted.shape} and labels {labels.shape} must be equal-length (T,) arrays",
            path="evaluation.labels",
        )
    ids = np.unique(np.concatenate([predicted, labels]))
    if ids.shape[0] > MAX_CLASSES:
        raise ConfigError(
            f"balanced_accuracy supports at most {MAX_CLASSES} distinct regimes/labels, got {ids.shape[0]}",
            path="evaluation.labels",
        )
    k = ids.shape[0]
    pred_idx, lab_idx = np.searchsorted(ids, predicted), np.searchsorted(ids, labels)
    confusion = np.zeros((k, k))
    np.add.at(confusion, (lab_idx, pred_idx), 1.0)
    support = confusion.sum(axis=1)
    present = support > 0
    best = 0.0
    for perm in permutations(range(k)):
        recall = confusion[np.arange(k), np.asarray(perm)][present] / support[present]
        best = max(best, float(recall.mean()))
    return best


@metric("balanced_accuracy", {"posterior", "labels"})
def balanced_accuracy(**ctx: object) -> float:
    """Viterbi regimes vs `ctx['labels']`, up to a relabelling of the regimes."""
    post, labels = ctx.get("posterior"), ctx.get("labels")
    if not isinstance(post, PosteriorResult):
        raise ConfigError("balanced_accuracy needs ctx['posterior']", path="evaluation.metrics")
    if labels is None:
        raise ConfigError(
            "balanced_accuracy needs reference labels (ctx['labels'])", path="evaluation.labels"
        )
    return balanced_accuracy_matched(np.asarray(post.viterbi), np.asarray(labels))
