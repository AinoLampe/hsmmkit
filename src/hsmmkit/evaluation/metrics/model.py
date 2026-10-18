"""Model-quality metrics from a decoded posterior (E7-T01): information criteria, occupancy, persistence."""

from __future__ import annotations

import numpy as np

from regimekit.evaluation.metrics.registry import metric
from regimekit.inference.results import PosteriorResult
from regimekit.utils.errors import ConfigError


def _posterior(ctx: dict[str, object]) -> PosteriorResult:
    post = ctx.get("posterior")
    if not isinstance(post, PosteriorResult):
        raise ConfigError("this metric needs ctx['posterior'] (a PosteriorResult)", path="evaluation.metrics")
    return post


@metric("aic", {"posterior"})
def aic(**ctx: object) -> float:
    """Akaike information criterion of the decoded fit."""
    return float(_posterior(ctx).aic)


@metric("bic", {"posterior"})
def bic(**ctx: object) -> float:
    """Bayesian information criterion of the decoded fit."""
    return float(_posterior(ctx).bic)


@metric("icl", {"posterior"})
def icl(**ctx: object) -> float:
    """Integrated completed likelihood criterion of the decoded fit."""
    return float(_posterior(ctx).icl)


@metric("regime_occupancy", {"posterior"})
def regime_occupancy(**ctx: object) -> dict[str, float]:
    """Fraction of steps the Viterbi path spends in each regime (`{"0": share, ...}`; sums to 1)."""
    post = _posterior(ctx)
    path = np.asarray(post.viterbi)
    n_states = int(np.asarray(post.smoothed).shape[1])
    return {str(i): float(np.mean(path == i)) for i in range(n_states)}


@metric("regime_persistence", {"posterior"})
def regime_persistence(**ctx: object) -> dict[str, float]:
    """Mean Viterbi segment length (steps) per regime; regimes that never occur are omitted."""
    post = _posterior(ctx)
    lengths: dict[int, list[int]] = {}
    for start, end, regime in post.segments:
        lengths.setdefault(int(regime), []).append(int(end) - int(start) + 1)
    return {str(r): float(np.mean(v)) for r, v in sorted(lengths.items())}
