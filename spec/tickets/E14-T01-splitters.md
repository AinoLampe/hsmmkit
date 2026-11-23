# E14-T01 — `Splitter` interface: `SequenceKFold`, walk-forward re-based on it, CV aggregation

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-EV03, D21 | M11 | Part II §14.12 (Splitters), §15.3 (`Splitter`), §15.4 (`evaluation.cv`); Part I E7-T02a/b, E10-T07a (origin hooks) |

## Goal
Unify cross-validation over sequences and over time behind one interface so selection, grid search and reporting work the same for both.

## Context (self-contained)
```python
@dataclass(frozen=True)
class Split:
    fold: int
    train: SequenceBatch            # data the estimator sees (preparer re-fitted on it)
    test: SequenceBatch             # evaluation data (prepared with the train PrepStats)
    origins: np.ndarray | None      # walk-forward only: forecast origins (step indices in test sequence 0)
class Splitter(Protocol):
    name: str
    def split(self, raw: RawData, cfg: EvaluationSection, preparer_factory) -> Iterator[Split]: ...
```
`RawData` = `(ObservationSet, covs)` before preparation so that `Preparer.fit` runs on each training part only (no leakage).
- `SequenceKFold(folds, seed, group_by=None)`: shuffles sequence ids with `key_for(seed, "cv")`, assigns to `folds` folds; with `data.sequences.group_by` set, whole groups stay together (grouped K-fold). Requires `B ≥ folds` (else `ConfigError`).
- `WalkForward`: wraps E7-T02a `schedule_folds` (single sequence; `B > 1` → applies the same calendar split to every sequence and evaluates origins per sequence, documented) and exposes origins; `run_backtest` (E10-T07a) is rewritten as `cross_validate(splitter=WalkForward, …)` — results must stay bitwise identical.
- `cross_validate(project_cfg, raw, splitter, seed, hooks=()) -> CVResult(folds: DataFrame, scores: DataFrame long (fold, sequence, horizon, metric, value), summary)`:
  - sequence CV: per test sequence, out-of-sample **log score** = `ℓ_b(θ̂_train)` per step (`/ T_b`), **CRPS** of one-step predictive distributions (sample-based, E7-T01 CRPS on `n_sim` draws from the one-step predictive mixture), plus any configured metrics needing `posterior`/`labels` (decode on test with `θ̂_train`);
  - walk-forward: E7-T02b per-origin forecasts and metrics (unchanged).
- CLI `hsmmkit cv CONFIG` writes `cv_scores.parquet`, `cv_summary.parquet`, and a CV section in `report.md`; `evaluation.cv.splitter` selects the splitter (registry kind `splitter`, added to `plugins.Kind`).
- Gate `evaluation.cv` removed.

## Creates
- `src/hsmmkit/selection/splitters.py`, `selection/cv.py`, CLI `cv`, edits in `evaluation/backtest.py`
- `tests/integration/test_e14_t01_cv.py`, `tests/unit/test_e14_t01_sequence_kfold.py`

## Acceptance criteria
1. Walk-forward results (folds, metrics, summary, hook extras) bitwise unchanged after re-basing (Part I E7-T02b tests and the `regimekit` strategy backtest via the `downstream` job).
2. Grouped folds never split a group; every sequence is in exactly one test fold; deterministic per seed.
3. Look-ahead test of E7-T02b retained and passing; a sequence-CV leakage test (standardization fitted on test sequences would change scores) proves preparation is per fold.
4. `hsmmkit cv` on the 50-sequence segmentation fixture (MAP, diag Gaussian, no labels) writes the documented artifacts.

## Tests
Unit (fold assignment), integration (CV runs, bitwise walk-forward), leakage tests.

## Out of scope
Grid selection (E14-T02); segmentation metrics (E14-T03).

## Connections
- Depends on: E12-T02
- Blocks: E14-T02, E16-T05
- Relates to: E7-T02a/b, E10-T07a, E13-T07
