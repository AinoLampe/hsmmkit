"""E14-T03 unit: boundary metrics, Hausdorff, covering on closed-form cases (exact)."""

from __future__ import annotations

import numpy as np
import pytest

from hsmmkit.evaluation.metrics import registry
from hsmmkit.evaluation.metrics.segmentation import (
    aggregate_sequences,
    boundaries,
    boundary_scores,
    count_matches,
    covering_score,
    hausdorff_distance,
)
from hsmmkit.inference.results import PosteriorResult
from hsmmkit.utils.errors import ConfigError


def _path(change_points: list[int], t_len: int) -> np.ndarray:
    """A regime path that alternates 0/1 at the given change points."""
    q = np.zeros(t_len, dtype=int)
    for k, c in enumerate(change_points):
        q[c:] = (k + 1) % 2
    return q


def test_e14_t03_boundaries_are_the_first_steps_of_new_segments() -> None:
    """`B(q) = {t >= 1 : q_t != q_{t-1}}`."""
    assert boundaries(np.array([0, 0, 1, 1, 1, 2, 0])).tolist() == [2, 5, 6]
    assert boundaries(np.zeros(5, dtype=int)).size == 0


def test_e14_t03_perfect_segmentation_scores_one_zero_one() -> None:
    """AC1: identical segmentations (also under relabelling): F1 = 1, Hausdorff = 0, covering = 1."""
    true = _path([20, 50, 80], 100)
    pred = (true + 1) % 2 + 5  # different ids, same segmentation
    assert boundary_scores(pred, true, 3) == (1.0, 1.0, 1.0)
    assert hausdorff_distance(pred, true) == 0.0
    assert covering_score(pred, true) == 1.0


@pytest.mark.parametrize(("shift", "f1"), [(0, 1.0), (3, 1.0), (-3, 1.0), (4, 0.0), (-4, 0.0)])
def test_e14_t03_shifting_every_boundary_by_tau_keeps_f1_one_and_by_tau_plus_one_drops_it(
    shift: int, f1: float
) -> None:
    """AC1: shifts of `tau = 3` match, shifts of `tau + 1 = 4` do not."""
    true = _path([20, 50, 80], 100)
    pred = _path([20 + shift, 50 + shift, 80 + shift], 100)
    assert boundary_scores(pred, true, 3)[2] == f1
    assert hausdorff_distance(pred, true) == abs(shift)


def test_e14_t03_matching_is_one_to_one_and_optimal() -> None:
    """Two predictions near one true boundary match once; the sweep finds the maximum matching."""
    assert count_matches(np.array([10, 11]), np.array([10]), 2) == 1
    # a greedy by global distance would take (2, 2) and strand 1 and 3: the optimum matches both pairs
    assert count_matches(np.array([1, 2]), np.array([2, 3]), 1) == 2
    p, r, f = boundary_scores(_path([10, 11], 30), _path([10], 30), 2)
    assert (p, r) == (0.5, 1.0)
    assert f == pytest.approx(2 * 0.5 / 1.5)


def test_e14_t03_empty_boundary_sets() -> None:
    """Both empty: perfect; one empty: 0 for the empty side, `nan` Hausdorff, covering of a single segment."""
    flat = np.zeros(20, dtype=int)
    many = _path([5, 10, 15], 20)
    assert boundary_scores(flat, flat, 2) == (1.0, 1.0, 1.0)
    assert hausdorff_distance(flat, flat) == 0.0
    assert boundary_scores(flat, many, 2) == (0.0, 0.0, 0.0)  # no prediction: precision 0 by convention
    assert boundary_scores(many, flat, 2) == (0.0, 0.0, 0.0)
    assert np.isnan(hausdorff_distance(flat, many))
    assert np.isnan(hausdorff_distance(many, flat))


def test_e14_t03_covering_of_single_segment_against_many_segments() -> None:
    """One predicted segment against `K` equal true segments: covering `1/K`; and the reverse."""
    flat = np.zeros(40, dtype=int)
    four = _path([10, 20, 30], 40)
    assert covering_score(flat, four) == pytest.approx(0.25)
    # four true... one true segment of 40 vs best predicted segment of 10: IoU 0.25, weight 1
    assert covering_score(four, flat) == pytest.approx(0.25)


def test_e14_t03_hausdorff_is_the_symmetric_worst_case_distance() -> None:
    """One far-away spurious boundary dominates."""
    true = _path([20, 50], 100)
    pred = _path([21, 50, 90], 100)  # note: `_path` alternates, so 90 is a third boundary
    assert hausdorff_distance(pred, true) == 40.0


def test_e14_t03_length_weighted_aggregation_skips_nan() -> None:
    """Sequences are combined by the length-weighted mean; `nan` entries are skipped."""
    assert aggregate_sequences([1.0, 0.0], [10, 30]) == pytest.approx(0.25)
    assert aggregate_sequences([1.0, float("nan")], [10, 30]) == 1.0
    assert np.isnan(aggregate_sequences([float("nan")], [5]))


def _posterior(viterbi: np.ndarray) -> PosteriorResult:
    t = viterbi.shape[0]
    return PosteriorResult(
        times=np.arange(t),
        filtered=np.zeros((t, 2)),
        smoothed=np.zeros((t, 2)),
        viterbi=viterbi,
        segments=[],
        loglik=0.0,
        aic=0.0,
        bic=0.0,
        icl=0.0,
        augmented=None,  # type: ignore[arg-type]  # the metrics only read `viterbi`
    )


def test_e14_t03_registered_metrics_read_the_posterior_and_labels_context() -> None:
    """The metrics are registered with `needs={posterior, labels}` and honour `boundary_tolerance`."""
    names = {"boundary_precision", "boundary_recall", "boundary_f1", "hausdorff", "covering"}
    assert names <= set(registry.available())
    true = _path([20, 50], 100)
    ctx = {"posterior": _posterior(_path([22, 50], 100)), "labels": true}
    for name, needs in ((n, registry.resolve([n])[0].needs) for n in sorted(names)):
        assert needs == {"posterior", "labels"}, name
    f1 = registry.resolve(["boundary_f1"])[0]
    assert f1(**ctx, boundary_tolerance=2) == 1.0
    assert f1(**ctx, boundary_tolerance=1) == 0.5
    assert registry.resolve(["hausdorff"])[0](**ctx) == 2.0
    with pytest.raises(ConfigError, match="reference labels"):
        f1(posterior=ctx["posterior"])
    with pytest.raises(ConfigError, match="boundary_tolerance"):
        f1(**ctx, boundary_tolerance=-1)
