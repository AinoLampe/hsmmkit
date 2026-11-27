# Model selection: `compare --grid`

`hsmmkit compare CONFIG --grid` fits the configured model on every cell of `evaluation.grid` and tabulates
information criteria, cross-validated scores, occupancy and run health, so that the number of regimes, the
emission and duration families and the estimator can be chosen from one table.

```yaml
evaluation:
  cv: {splitter: sequence_kfold, folds: 5, seed: 7}   # optional: adds cv_log_score / cv_crps
  grid:
    n_states: [2, 3, 4]
    duration: [geometric, negbin]
    r_same_for_all: [2, 3]       # or `r: [[2, 3], ...]` for explicit per-regime counts
    estimator: [map, em]
    parallel: {backend: process, workers: 4}   # optional
```

## Cells

The cells are the Cartesian product of the axes; an absent axis keeps the base configuration's value. Every cell
is the base configuration with its overrides and is validated on its own.

* `n_states` resizes the per-regime lists of the base configuration (`init`, `values`,
  `init_mean_sojourn_steps`) by linear interpolation between their ends: a base with initial means
  `[-0.5, 0.5]` gives `[-0.5, 0.0, 0.5]` for three regimes. An explicit topology matrix cannot be resized.
* A different `emission` or `duration` family starts from the family's default parameters (`FAMILY_PARAMS`); the
  base duration's initial mean sojourn is carried over. Families without defaults are skipped.
* A cell that does not validate or cannot be built or fitted (for example `gamma` durations on the expanded
  engine) is **skipped**: it keeps its axis values and the reason in `skipped_reason`; the other cells run.

## Output

`grid.parquet` and `grid.md` (output directory, default `runs/compare`). Columns:

| Column | Meaning |
|---|---|
| `cell`, `n_states`, `r`, `emission`, `duration`, `estimator` | the cell and its axis values |
| `loglik`, `n_free`, `aic`, `bic`, `icl` | in-sample fit (ICL is NaN for several sequences) |
| `cv_log_score`, `cv_crps` | cross-validated scores (`evaluation.cv`), NaN otherwise |
| `min_occupancy` | smallest mean smoothed regime probability: near 0 flags an unused regime |
| `objective_spread` | range of the finite restart objectives (restart dispersion) |
| `wall_s`, `converged`, `n_iter` | run health |
| `skipped_reason` | empty for fitted cells |
| `recommended` | the recommended cell |

**Recommendation rule:** the lowest BIC; ties (within 1e-9) go to the highest cross-validated log score. In
`grid.md` the best value of each criterion is bold.

## Determinism and parallelism

Cell `i` fits with the seed `derive_seed(seed, PURPOSE_GRID, i)`, so a cell's result does not depend on the
order or the worker it ran on. With `evaluation.grid.parallel` the cells run in a `spawn` process pool; the table
equals the sequential one except for `wall_s`. Each worker compiles its own JIT caches, so use it for grids
whose cells take longer than a few seconds.

The in-sample criteria use the preparation statistics of the whole sample; the cross-validated columns fit the
preparation on the training part of each fold.
