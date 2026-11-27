"""E14-T01 AC2: fold assignment of `SequenceKFold` (grouped, exactly-once, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest

from hsmmkit.config.schema import CvCfg
from hsmmkit.data.contracts import ObservationSet
from hsmmkit.plugins import REGISTRY
from hsmmkit.selection.splitters import SequenceKFold, WalkForward, subset_rows
from hsmmkit.utils.errors import ConfigError


def _obs(n_seq: int, per: int = 5, groups: list[str] | None = None) -> ObservationSet:
    ids = np.repeat(np.arange(n_seq), per).astype(str)
    meta = {}
    if groups is not None:
        meta["group_values"] = [groups[int(s)] for s in ids]
    return ObservationSet(
        ("x",), None, np.arange(n_seq * per, dtype=np.float64)[:, None], "memory", "s", meta, ids
    )


def test_e14_t01_every_sequence_is_in_exactly_one_test_fold_and_the_assignment_is_balanced() -> None:
    """AC2: a partition into `folds` test sets of near-equal size."""
    obs = _obs(23)
    assign = SequenceKFold(5, seed=3).assignment(obs)
    assert set(assign) == {str(i) for i in range(23)}
    sizes = np.bincount(list(assign.values()), minlength=5)
    assert sizes.sum() == 23
    assert sizes.max() - sizes.min() <= 1


def test_e14_t01_the_assignment_is_deterministic_per_seed_and_changes_with_it() -> None:
    """Same seed, same folds; another seed, other folds."""
    obs = _obs(30)
    a, b, c = (SequenceKFold(5, seed=s).assignment(obs) for s in (1, 1, 2))
    assert a == b
    assert a != c


def test_e14_t01_grouped_folds_never_split_a_group() -> None:
    """With `group_by`, all sequences of a group share one test fold."""
    groups = [f"g{i % 7}" for i in range(28)]  # 7 groups of 4 sequences
    obs = _obs(28, groups=groups)
    assign = SequenceKFold(3, seed=0, group_by="subject").assignment(obs)
    for g in set(groups):
        folds = {assign[str(i)] for i in range(28) if groups[i] == g}
        assert len(folds) == 1, (g, folds)
    assert set(assign.values()) == {0, 1, 2}


def test_e14_t01_too_few_sequences_or_groups_and_missing_ids_are_configuration_errors() -> None:
    """`ConfigError` for B < folds, groups < folds and data without sequence ids."""
    with pytest.raises(ConfigError, match="cannot fill 5 folds"):
        SequenceKFold(5).assignment(_obs(4))
    with pytest.raises(ConfigError, match="cannot fill 3 folds"):
        SequenceKFold(3, group_by="g").assignment(_obs(8, groups=["a", "a", "b", "b", "a", "b", "a", "b"]))
    with pytest.raises(ConfigError, match="sequence_column"):
        SequenceKFold(2).assignment(ObservationSet(("x",), None, np.zeros((4, 1)), "memory", "s"))


def test_e14_t01_subset_rows_keeps_the_alignment_of_values_ids_and_covariates() -> None:
    """Values, ids, group values and covariates are cut with the same rows."""
    from hsmmkit.data.contracts import CovariateSeries

    obs = _obs(3, 2, groups=["a", "b", "c"])
    cov = CovariateSeries("c", None, np.arange(6.0), "memory", "c", "", "daily")
    sub, subs = subset_rows(obs, [cov], np.array([1, 4, 5]))
    np.testing.assert_array_equal(sub.values[:, 0], [1.0, 4.0, 5.0])
    assert list(sub.sequence_ids) == ["0", "2", "2"]
    assert sub.meta["group_values"] == ["a", "c", "c"]
    np.testing.assert_array_equal(subs[0].values, [1.0, 4.0, 5.0])


def test_e14_t01_the_splitters_are_registered_and_built_from_the_cv_section() -> None:
    """Registry kind `splitter`: both splitters build from `evaluation.cv`."""
    cv = CvCfg(splitter="sequence_kfold", folds=4, seed=9)
    kfold = REGISTRY.get("splitter", "sequence_kfold").from_config(cv, "g")
    assert kfold == SequenceKFold(4, 9, "g")
    assert isinstance(REGISTRY.get("splitter", "walk_forward").from_config(cv), WalkForward)
