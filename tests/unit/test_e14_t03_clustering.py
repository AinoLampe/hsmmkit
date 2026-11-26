"""E14-T03 unit: Hungarian matching, ARI / NMI / matched accuracy, and the Hungarian `balanced_accuracy`."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hsmmkit.evaluation.metrics.clustering import (
    adjusted_rand,
    hamming_matched_score,
    matched_confusion,
    normalized_mutual_info,
)
from hsmmkit.evaluation.metrics.labels import balanced_accuracy_matched
from hsmmkit.evaluation.metrics.matching import hungarian, match_labels
from hsmmkit.utils.errors import ConfigError


def _brute_force_min(cost: np.ndarray) -> float:
    n, m = cost.shape
    if n <= m:
        return min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
    return _brute_force_min(cost.T)


@given(st.integers(1, 6), st.integers(1, 6), st.integers(0, 10_000))
def test_e14_t03_hungarian_equals_the_brute_force_optimum(n: int, m: int, seed: int) -> None:
    """AC3: for random `N <= 6` (square and rectangular, with ties) the cost is the brute-force optimum."""
    rng = np.random.default_rng(seed)
    cost = rng.integers(0, 5, size=(n, m)).astype(float) if seed % 2 else rng.normal(size=(n, m))
    rows, cols = hungarian(cost)
    assert rows.shape[0] == min(n, m)
    assert len(set(cols.tolist())) == cols.shape[0]
    assert np.all(np.diff(rows) > 0)
    assert cost[rows, cols].sum() == pytest.approx(_brute_force_min(cost), abs=1e-9)


def test_e14_t03_hungarian_rejects_bad_input_and_handles_empty() -> None:
    """Non-finite or non-2-D costs raise; an empty matrix gives empty indices."""
    with pytest.raises(ConfigError):
        hungarian(np.array([[1.0, np.nan]]))
    with pytest.raises(ConfigError):
        hungarian(np.zeros(3))
    rows, cols = hungarian(np.zeros((0, 4)))
    assert rows.shape == cols.shape == (0,)


def test_e14_t03_match_labels_maps_predicted_ids_to_the_reference() -> None:
    """The predicted ids are renamed to the reference ids that they overlap most."""
    true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    pred = np.array([7, 7, 7, 4, 4, 9, 9, 4])
    assert match_labels(pred, true) == {7: 0, 4: 1, 9: 2}
    assert hamming_matched_score(pred, true) == pytest.approx(7 / 8)
    conf = matched_confusion(pred, true)
    assert conf["true0/pred0"] == pytest.approx(3 / 8)
    assert conf["true2/pred1"] == pytest.approx(1 / 8)
    assert sum(conf.values()) == pytest.approx(1.0)


def test_e14_t03_closed_form_partitions() -> None:
    """Identical partitions score 1 (any relabelling); independent ones score 0 (ARI exactly, NMI exactly)."""
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand(a, 2 - a) == 1.0
    assert normalized_mutual_info(a, a + 10) == pytest.approx(1.0)
    left = np.array([0, 0, 1, 1])
    right = np.array([0, 1, 0, 1])  # independent of `left`
    assert adjusted_rand(left, right) == pytest.approx(-0.5)
    assert normalized_mutual_info(left, right) == 0.0
    assert adjusted_rand(np.zeros(5, dtype=int), np.zeros(5, dtype=int)) == 1.0
    assert normalized_mutual_info(np.zeros(5, dtype=int), np.zeros(5, dtype=int)) == 1.0


@given(
    st.lists(st.integers(0, 3), min_size=2, max_size=40), st.lists(st.integers(0, 3), min_size=2, max_size=40)
)
def test_e14_t03_metrics_are_invariant_to_relabelling(a: list[int], b: list[int]) -> None:
    """Permuting the predicted ids changes neither ARI, NMI, matched accuracy nor balanced accuracy."""
    n = min(len(a), len(b))
    pred, true = np.asarray(a[:n]), np.asarray(b[:n])
    perm = np.asarray([2, 0, 3, 1])
    for fn in (adjusted_rand, normalized_mutual_info, hamming_matched_score, balanced_accuracy_matched):
        assert fn(perm[pred], true) == pytest.approx(fn(pred, true), abs=1e-12)


def test_e14_t03_balanced_accuracy_is_unchanged_and_no_longer_limited_to_six_ids() -> None:
    """AC3: Part I values (brute force over permutations) are reproduced; 8 ids now work."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        true = rng.integers(0, 4, size=60)
        pred = np.where(rng.random(60) < 0.3, rng.integers(0, 4, size=60), true)
        confusion = np.zeros((4, 4))
        np.add.at(confusion, (true, pred), 1.0)
        support = confusion.sum(axis=1)
        present = support > 0
        best = max(
            (confusion[np.arange(4), np.asarray(p)][present] / support[present]).mean()
            for p in itertools.permutations(range(4))
        )
        assert balanced_accuracy_matched(pred, true) == pytest.approx(best, abs=1e-12)
    big = np.repeat(np.arange(8), 5)
    assert balanced_accuracy_matched((big + 3) % 8, big) == 1.0
    with pytest.raises(ConfigError, match="equal-length"):
        balanced_accuracy_matched(np.zeros(3, dtype=int), np.zeros(4, dtype=int))


def test_e14_t03_the_metric_modules_do_not_import_scipy() -> None:
    """ADR-0020: matching is vendored; importing the metrics never loads scipy (~370 ms)."""
    import subprocess
    import sys

    code = (
        "import sys, hsmmkit.evaluation.metrics.clustering, hsmmkit.evaluation.metrics.segmentation, "
        "hsmmkit.evaluation.metrics.labels; sys.exit(int('scipy' in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-I", "-c", code], check=False).returncode == 0
