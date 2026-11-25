"""Information criteria (E4-T05): AIC, BIC and ICL.

`p` is `ParamManager.n_free`, `T` the number of unmasked steps and `ell` the log-likelihood:
`AIC = 2p - 2 ell`, `BIC = p log T - 2 ell`, `ICL = BIC + 2 H(gamma)` with
`H(gamma) = -sum_t sum_s gamma_t(s) log gamma_t(s)` the entropy of the smoothed expanded posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hsmmkit.utils.errors import NumericalError


@dataclass(frozen=True)
class InformationCriteria:
    """`aic`, `bic`, `icl` (lower is better) and the inputs `n_params`, `n_obs`, `entropy`."""

    aic: float
    bic: float
    icl: float
    n_params: int
    n_obs: int
    entropy: float


def posterior_entropy(smoothed: np.ndarray, mask: np.ndarray | None = None) -> float:
    """`H(gamma) = -sum_t sum_s gamma_t(s) log gamma_t(s)` over unmasked steps (`0 log 0 = 0`)."""
    gamma = np.asarray(smoothed, dtype=float)
    if mask is not None:
        gamma = gamma[np.asarray(mask).all(axis=-1)]
    positive = gamma > 0
    return float(-np.sum(gamma[positive] * np.log(gamma[positive])))


def information_criteria(
    loglik: float, n_params: int, smoothed: np.ndarray, mask: np.ndarray | None = None
) -> InformationCriteria:
    """AIC/BIC/ICL for a fit.

    Args:
        loglik: log-likelihood `ell`.
        n_params: `p = ParamManager.n_free`.
        smoothed: `(T, S)` smoothed expanded posterior.
        mask: optional `(T, M)` observation mask; `T` counts steps where every series is observed.

    Raises:
        NumericalError: on a non-finite `loglik` or no unmasked step.
    """
    if not math.isfinite(loglik):
        raise NumericalError("non-finite log-likelihood", path="loglik")
    t_obs = int(np.asarray(smoothed).shape[0] if mask is None else np.asarray(mask).all(axis=-1).sum())
    if t_obs < 1:
        raise NumericalError("no unmasked observation", path="mask")
    entropy = posterior_entropy(smoothed, mask)
    bic = n_params * math.log(t_obs) - 2.0 * loglik
    return InformationCriteria(
        aic=2.0 * n_params - 2.0 * loglik,
        bic=bic,
        icl=bic + 2.0 * entropy,
        n_params=n_params,
        n_obs=t_obs,
        entropy=entropy,
    )
