"""Pure NumPy/SciPy reference distributions: shifted negative-binomial/geometric pmf, GBM density.

No `regimekit` import (CLAUDE.md §6). Slow, explicit-enumeration implementations -- ground truth
for `regimekit.model.dists` golden tests.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def negbin_shifted_pmf(d: np.ndarray, r: int, a: float, d_max: int) -> np.ndarray:
    """`d - r ~ NegBin(r, a)` via `scipy.stats.nbinom`, truncated at `d_max` and renormalized."""
    support = np.arange(1, d_max + 1)
    raw = np.where(support >= r, stats.nbinom.pmf(support - r, r, a), 0.0)
    z = raw.sum()
    d = np.asarray(d)
    return np.where(d >= r, stats.nbinom.pmf(d - r, r, a), 0.0) / z


def negbin_shifted_logpmf(d: np.ndarray, r: int, a: float, d_max: int) -> np.ndarray:
    """`log` of `negbin_shifted_pmf`."""
    with np.errstate(divide="ignore"):
        return np.log(negbin_shifted_pmf(d, r, a, d_max))


def geometric_pmf(d: np.ndarray, a: float, d_max: int) -> np.ndarray:
    """`d - 1 ~ Geometric(a)`, truncated at `d_max` and renormalized."""
    support = np.arange(1, d_max + 1)
    raw = stats.geom.pmf(support, a)
    z = raw.sum()
    d = np.asarray(d)
    return stats.geom.pmf(d, a) / z


def geometric_logpmf(d: np.ndarray, a: float, d_max: int) -> np.ndarray:
    """`log` of `geometric_pmf`."""
    with np.errstate(divide="ignore"):
        return np.log(geometric_pmf(d, a, d_max))


def gbm_log_density(y: np.ndarray, mu: float, sigma: float, dt: float) -> np.ndarray:
    """GBM log-return log-density: `N((mu - sigma**2/2) * dt, sigma**2 * dt)`."""
    mean = (mu - 0.5 * sigma**2) * dt
    sd = sigma * np.sqrt(dt)
    return stats.norm.logpdf(y, mean, sd)
