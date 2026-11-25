"""`PosteriorResult` and `ForecastResult` (E1-T05): plain result containers, not pytrees.

Neither is jitted or traced directly (both are post-hoc, Python-level consumption/serialization
objects assembled once inference/forecasting has finished), so no pytree registration is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hsmmkit.engine.base import Posterior
from hsmmkit.utils.typing import Array


@dataclass(frozen=True)
class PosteriorResult:
    """Decoded posterior: filtered/smoothed regime probabilities, Viterbi path, segments, IC."""

    times: np.ndarray
    filtered: Array  # (T, N)
    smoothed: Array  # (T, N)
    viterbi: np.ndarray  # (T,) regime path
    segments: list[tuple[int, int, int]]  # (start, end, regime), end inclusive
    augmented: Posterior
    loglik: float
    aic: float
    bic: float
    icl: float


@dataclass(frozen=True)
class ForecastResult:
    """A Monte Carlo forecast from a given origin: samples, cumulative quantiles, moments."""

    origin_time: np.datetime64
    horizon_times: np.ndarray  # (h,)
    regime_probs: Array  # (h, N)
    samples: Array  # (n_sim, h, M) per-step log-values
    cum_quantiles: Array  # (h, Q, M)
    mean: Array  # (h, M) of cumulative log-return
    var: Array  # (h, M) of cumulative log-return
    quantile_levels: tuple[float, ...]
    covariate_forecaster: str
    seed: int
