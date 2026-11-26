"""E14-T03 (dev-only, needs scikit-learn): ARI / NMI equal scikit-learn within 1e-12 on random labelings."""

from __future__ import annotations

import numpy as np
import pytest

from hsmmkit.evaluation.metrics.clustering import adjusted_rand, normalized_mutual_info

sklearn_metrics = pytest.importorskip("sklearn.metrics")


@pytest.mark.parametrize("seed", range(25))
def test_e14_t03_ari_nmi_match_scikit_learn(seed: int) -> None:
    """AC2: random labelings of random sizes and cluster counts, including degenerate ones."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 300))
    k1, k2 = (int(x) for x in rng.integers(1, 9, size=2))
    a, b = rng.integers(0, k1, size=n), rng.integers(0, k2, size=n)
    assert abs(adjusted_rand(a, b) - sklearn_metrics.adjusted_rand_score(b, a)) <= 1e-12
    assert abs(normalized_mutual_info(a, b) - sklearn_metrics.normalized_mutual_info_score(b, a)) <= 1e-12


def test_e14_t03_degenerate_cases_match_scikit_learn() -> None:
    """One cluster against one cluster, one against many, and all singletons."""
    one, many, singletons = np.zeros(6, dtype=int), np.arange(6) % 3, np.arange(6)
    for a, b in [(one, one), (one, many), (many, one), (singletons, singletons), (one, singletons)]:
        assert abs(adjusted_rand(a, b) - sklearn_metrics.adjusted_rand_score(b, a)) <= 1e-12
        assert abs(normalized_mutual_info(a, b) - sklearn_metrics.normalized_mutual_info_score(b, a)) <= 1e-12
