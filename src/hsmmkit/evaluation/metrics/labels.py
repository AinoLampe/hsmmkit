"""Label-based metrics (E7-T01): `balanced_accuracy` of the Viterbi regimes against reference labels."""

from __future__ import annotations

import numpy as np

from hsmmkit.evaluation.metrics.matching import confusion, hungarian
from hsmmkit.evaluation.metrics.registry import metric
from hsmmkit.inference.results import PosteriorResult
from hsmmkit.utils.errors import ConfigError


def balanced_accuracy_matched(predicted: np.ndarray, labels: np.ndarray) -> float:
    """Balanced accuracy (mean per-label recall) under the best relabelling of `predicted`.

    Regime ids are arbitrary, so the predicted ids are matched to the label ids by the Hungarian algorithm
    (`evaluation.metrics.matching`, ADR-0020): maximize `sum_k C[pred(k), k] / support_k`. Valid for any
    number of ids; the value is invariant to any permutation of either id set.

    Raises:
        ConfigError: a length mismatch.
    """
    c, _, _ = confusion(predicted, labels)  # (pred ids, label ids)
    support = c.sum(axis=0)
    recall = c / np.where(support > 0, support, 1.0)[None, :]
    rows, cols = hungarian(-recall)
    return float(recall[rows, cols].sum() / np.count_nonzero(support > 0))


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
