"""E5-T04 / E7-T01 shared helper: `evaluation/metrics/predictive.py` -- direct correctness checks.

`tests/reference/` normally holds the independent NumPy/SciPy comparisons (overview §4.7), but these
two functions are themselves already the NumPy-only "reference-grade" implementation (no JAX inside the
statistic itself); comparing them against `scipy.stats` here is the appropriate independent check.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from hsmmkit.evaluation.metrics.predictive import ljung_box, pit_histogram


def test_e5_t04_pit_histogram_ks_matches_scipy() -> None:
    """`pit_histogram`'s KS statistic and p-value match `scipy.stats.kstest` on the same sample."""
    rng = np.random.default_rng(0)
    uniform_sample = rng.uniform(size=500)

    _, d_stat, p_value = pit_histogram(uniform_sample)
    ref = stats.kstest(uniform_sample, "uniform")

    assert d_stat == ref.statistic
    assert abs(p_value - ref.pvalue) < 1e-3


def test_e5_t04_pit_histogram_rejects_a_non_uniform_sample() -> None:
    """A clearly non-uniform sample (all mass near 0) gives a small p-value."""
    skewed = np.random.default_rng(1).beta(0.5, 5.0, size=500)
    _, _, p_value = pit_histogram(skewed)
    assert p_value < 1e-3


def test_e5_t04_ljung_box_matches_statsmodels_style_formula_by_hand() -> None:
    """`ljung_box`'s Q statistic matches a hand-computed textbook formula at lag 2."""
    rng = np.random.default_rng(2)
    r = rng.normal(size=200)

    result = ljung_box(r, lags=(2,))
    q_stat, p_value = result[2]

    n = r.size
    x = r - r.mean()
    denom = np.sum(x**2)
    acf = [np.sum(x[k:] * x[:-k]) / denom for k in (1, 2)]
    expected_q = n * (n + 2) * sum(a**2 / (n - k) for k, a in zip((1, 2), acf, strict=True))

    assert abs(q_stat - expected_q) < 1e-9
    ref_p = 1.0 - stats.chi2.cdf(expected_q, df=2)
    assert abs(p_value - ref_p) < 1e-6


def test_e5_t04_ljung_box_high_p_value_on_white_noise() -> None:
    """Ljung-Box on IID white noise should not reject the no-autocorrelation null (large p)."""
    r = np.random.default_rng(3).normal(size=2000)
    result = ljung_box(r, lags=(10, 20))
    for _lag, (_q, p_value) in result.items():
        assert p_value > 0.05


def test_e5_t04_ljung_box_low_p_value_on_strong_autocorrelation() -> None:
    """Ljung-Box on a strongly autocorrelated (AR(1), rho=0.9) series should reject the null."""
    rng = np.random.default_rng(4)
    n = 2000
    x = np.zeros(n)
    eps = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + eps[t]

    result = ljung_box(x, lags=(10,))
    _q, p_value = result[10]
    assert p_value < 1e-6


def test_e5_t04_ljung_box_nan_when_lag_exceeds_sample_size() -> None:
    """A lag `>=` the sample length reports `nan` rather than crashing."""
    result = ljung_box(np.arange(5.0), lags=(10,))
    q_stat, p_value = result[10]
    assert np.isnan(q_stat)
    assert np.isnan(p_value)


def test_e5_t04_pit_histogram_drops_nan_entries() -> None:
    """Interspersed `NaN` (masked/unobserved steps) don't change the KS statistic vs. dropping them first."""
    rng = np.random.default_rng(5)
    clean = rng.uniform(size=300)
    with_gaps = np.insert(clean, [10, 100, 250], np.nan)

    _, d_clean, p_clean = pit_histogram(clean)
    _, d_gaps, p_gaps = pit_histogram(with_gaps)

    assert d_clean == d_gaps
    assert p_clean == p_gaps


def test_e5_t04_ljung_box_drops_nan_entries() -> None:
    """Interspersed `NaN` don't leak into the Q statistic or its p-value."""
    rng = np.random.default_rng(6)
    clean = rng.normal(size=300)
    with_gaps = np.insert(clean, [5, 150], np.nan)

    result_clean = ljung_box(clean, lags=(10,))
    result_gaps = ljung_box(with_gaps, lags=(10,))

    assert np.isfinite(result_gaps[10][0])
    assert np.isfinite(result_gaps[10][1])
    assert result_clean[10] == result_gaps[10]
