"""`Splitter`: one interface for cross-validation over sequences and over time (E14-T01).

A splitter works on **raw** data `(ObservationSet, covariates)` and a `preparer_factory` (a callable returning
an unfitted preparer), so that the preparation statistics (standardization, lags) are fitted on each training
part only and applied to the test part: nothing leaks from the test data into the fit.

* `SequenceKFold(folds, seed, group_by)`: whole sequences go to the test fold; with `group_by` whole groups
  (the value of `data.sequences.group_by` per sequence) stay together. The units are shuffled with
  `key_for(seed, PURPOSE_CV)` and dealt to the folds in turn, so the assignment is deterministic per seed and
  every sequence is in exactly one test fold.
* `WalkForward`: the calendar folds of `evaluation.backtest` (`schedule_folds`); one sequence only. The
  preparation per fold is the one of `run_backtest` (train-window statistics, whole-sample transform).

Imports from `hsmmkit.data` are limited to `contracts` and `batch` (layer rule): the preparer is an argument.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import jax
import numpy as np

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.evaluation.backtest import schedule_folds
from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.rng import PURPOSE_CV, key_for

if TYPE_CHECKING:
    from hsmmkit.config.schema import CvCfg, EvaluationSection
    from hsmmkit.data.contracts import CovariateSeries, ObservationSet

__all__ = ["PreparerFactory", "RawData", "SequenceKFold", "Split", "Splitter", "WalkForward", "subset_rows"]

RawData = tuple["ObservationSet", Sequence["CovariateSeries"]]
PreparerFactory = Callable[[], Any]


@dataclass(frozen=True)
class Split:
    """One fold: the data the estimator sees and the data it is evaluated on.

    Attributes:
        fold: fold index.
        train: training sequences (the preparer was fitted on them).
        test: evaluation sequences (prepared with the training statistics).
        origins: walk-forward only: forecast origins as step indices into `test.sequences[0]`.
    """

    fold: int
    train: SequenceBatch
    test: SequenceBatch
    origins: np.ndarray | None = None


class Splitter(Protocol):
    """Anything that cuts raw data into `Split`s."""

    name: str

    def split(
        self, raw: RawData, cfg: EvaluationSection, preparer_factory: PreparerFactory
    ) -> Iterator[Split]:
        """Yield the folds."""
        ...


def subset_rows(
    obs: ObservationSet, covs: Sequence[CovariateSeries], rows: np.ndarray
) -> tuple[ObservationSet, list[CovariateSeries]]:
    """The rows `rows` (indices) of a multi-sequence data set; covariates are row-aligned columns."""
    sub = dataclasses.replace(
        obs,
        times=None if obs.times is None else obs.times[rows],
        values=obs.values[rows],
        sequence_ids=None if obs.sequence_ids is None else np.asarray(obs.sequence_ids)[rows],
    )
    meta = dict(obs.meta)
    for key in ("label_values", "group_values"):
        values = obs.meta.get(key)
        if values is not None:
            meta[key] = [values[int(r)] for r in rows]
    sub = dataclasses.replace(sub, meta=meta)
    out = [
        dataclasses.replace(
            c,
            times=None if c.times is None else np.asarray(c.times)[rows],
            values=np.asarray(c.values)[rows],
        )
        for c in covs
    ]
    return sub, out


def _end(obs: ObservationSet) -> Any:  # noqa: ANN401
    return obs.values.shape[0] - 1 if obs.times is None else np.max(obs.require_times())


@dataclass(frozen=True)
class SequenceKFold:
    """K-fold cross-validation over whole sequences (grouped when `group_by` is set)."""

    folds: int
    seed: int = 0
    group_by: str | None = None
    name: str = "sequence_kfold"

    @classmethod
    def from_config(cls, cv: CvCfg, group_by: str | None = None) -> SequenceKFold:
        """From `evaluation.cv` (and `data.sequences.group_by`)."""
        return cls(cv.folds, cv.seed, group_by)

    def assignment(self, obs: ObservationSet) -> dict[str, int]:
        """`{sequence id: test fold}` (first-appearance order of the ids).

        Raises:
            ConfigError: no sequence ids, or fewer sequences (groups) than folds.
        """
        if obs.sequence_ids is None:
            raise ConfigError(
                "sequence cross-validation needs data.source.sequence_column",
                path="data.source.sequence_column",
            )
        ids = np.asarray(obs.sequence_ids)
        uniq, first = np.unique(ids, return_index=True)
        order = [str(s) for s in uniq[np.argsort(first)]]
        if self.group_by is not None:
            values = obs.meta.get("group_values")
            if values is None:
                raise ConfigError(
                    "grouped folds need data.sequences.group_by", path="data.sequences.group_by"
                )
            group_of = {str(s): str(values[int(first[list(uniq).index(s)])]) for s in uniq[np.argsort(first)]}
            units = list(dict.fromkeys(group_of[s] for s in order))
        else:
            group_of = {s: s for s in order}
            units = order
        if len(units) < self.folds:
            kind = "groups" if self.group_by is not None else "sequences"
            raise ConfigError(
                f"{len(units)} {kind} cannot fill {self.folds} folds", path="evaluation.cv.folds"
            )
        perm = np.asarray(jax.random.permutation(key_for(self.seed, PURPOSE_CV), len(units)))
        fold_of_unit = {units[int(u)]: i % self.folds for i, u in enumerate(perm)}
        return {s: fold_of_unit[group_of[s]] for s in order}

    def split(
        self, raw: RawData, cfg: EvaluationSection, preparer_factory: PreparerFactory
    ) -> Iterator[Split]:
        """One `Split` per fold: the preparer is fitted on the training sequences only."""
        del cfg
        obs, covs = raw
        fold_of = self.assignment(obs)
        ids = np.asarray([str(s) for s in np.asarray(obs.sequence_ids)])
        row_fold = np.asarray([fold_of[s] for s in ids])
        for k in range(self.folds):
            train_obs, train_covs = subset_rows(obs, covs, np.flatnonzero(row_fold != k))
            test_obs, test_covs = subset_rows(obs, covs, np.flatnonzero(row_fold == k))
            prep = preparer_factory().fit(train_obs, train_covs, _end(train_obs))
            yield Split(
                k, prep.transform_batch(train_obs, train_covs), prep.transform_batch(test_obs, test_covs)
            )


@dataclass(frozen=True)
class WalkForward:
    """The walk-forward calendar folds of `evaluation.backtest` (single sequence)."""

    name: str = "walk_forward"

    @classmethod
    def from_config(cls, cv: CvCfg, group_by: str | None = None) -> WalkForward:
        """From `evaluation.cv` (the schedule comes from `evaluation.backtest`)."""
        del cv, group_by
        return cls()

    def split(
        self, raw: RawData, cfg: EvaluationSection, preparer_factory: PreparerFactory
    ) -> Iterator[Split]:
        """One `Split` per fold: `train` the train window, `test` the test window, `origins` its rows."""
        bt = cfg.backtest
        if bt is None:
            raise ConfigError("evaluation.backtest is required for walk_forward", path="evaluation.backtest")
        obs, covs = raw
        if obs.sequence_ids is not None and np.unique(np.asarray(obs.sequence_ids)).shape[0] > 1:
            raise ConfigError(
                "walk_forward cross-validation supports one sequence", path="data.source.sequence_column"
            )
        grid = preparer_factory().fit(obs, covs, _end(obs)).transform(obs, covs)
        horizons = sorted(set(bt.horizons))
        for fold in schedule_folds(np.asarray(grid.times), np.asarray(grid.dt), bt, horizons[-1]):
            prep = preparer_factory().fit(
                obs,
                covs,
                train_end=grid.times[fold.train_end - 1],
                train_start=grid.times[fold.train_start],
            )
            data = prep.transform(obs, covs)
            origins = np.arange(fold.test_start, fold.test_end) - fold.test_start
            yield Split(
                fold.fold,
                SequenceBatch.of(data.window(fold.train_start, fold.train_end)),
                SequenceBatch.of(data.window(fold.test_start, fold.test_end)),
                origins,
            )


REGISTRY.register("splitter", SequenceKFold.name, SequenceKFold, provenance=provenance_of(__name__))
REGISTRY.register("splitter", WalkForward.name, WalkForward, provenance=provenance_of(__name__))
