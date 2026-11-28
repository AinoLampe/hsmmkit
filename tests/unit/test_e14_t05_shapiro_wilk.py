"""E14-T05 AC4: the own Shapiro-Wilk equals `scipy.stats.shapiro` (scipy is a dev dependency)."""

from __future__ import annotations

import numpy as np
import pytest

from hsmmkit.selection.pseudo_residuals import shapiro_wilk

stats = pytest.importorskip("scipy.stats")


@pytest.mark.parametrize("n", [3, 4, 5, 6, 11, 12, 30, 200, 1000, 5000])
def test_e14_t05_shapiro_wilk_equals_scipy(n: int) -> None:
    """W within 1e-6 (and p within 1e-6) of scipy on normal, heavy-tailed and skewed samples."""
    rng = np.random.default_rng(n)
    for sample in (rng.normal(size=n), rng.standard_t(3, size=n), rng.exponential(size=n)):
        w, p = shapiro_wilk(sample)
        w0, p0 = stats.shapiro(sample)
        assert abs(w - w0) <= 1e-6
        assert abs(p - p0) <= 1e-6


def test_e14_t05_shapiro_wilk_edge_cases() -> None:
    """Constant sample: NaN; fewer than three values raise; longer samples use 5000 order statistics."""
    w, p = shapiro_wilk(np.ones(10))
    assert np.isnan(w)
    assert np.isnan(p)
    with pytest.raises(ValueError, match="at least 3"):
        shapiro_wilk(np.array([1.0, 2.0]))
    rng = np.random.default_rng(0)
    w_long, p_long = shapiro_wilk(rng.normal(size=20_000))
    assert 0.99 < w_long <= 1.0
    assert 0.0 <= p_long <= 1.0
