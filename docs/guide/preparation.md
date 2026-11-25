# Preparation

The `Preparer` turns observations and covariates into a `PreparedDataset`. It is domain-neutral: observations
pass through unchanged unless a registered transform is configured.

```yaml
data:
  clock: steps
  preparation:
    observation_transform: null        # identity; or diff (core) / log_returns (finance)
    missing_observations: mask         # mask | drop | error
    standardize_observations: false    # real observations only
    standardize_covariates: true
```

## Observation transforms

Registry kind `transform`. A transform maps `(values, times)` to `(values, times)` and may drop leading rows
(`T = T0 - 1`) but never changes the others. The core registers `identity` (the default) and `diff`
(`y_t = x_t - x_{t-1}`); the finance application registers `log_returns` (`y_t = log(P_t / P_{t-1})`).
A transform that keeps every row (`identity`) gets one virtual grid time before the first row, so every
observation has a spacing `dt` (one median spacing; 1 under `steps`).

## Missing observations

- `mask`: every row is kept; non-finite values become `mask=False` with value 0.
- `drop`: rows where the primary (first) series is missing are removed, jointly with the covariates; other
  series that are missing stay masked.
- `error`: the first missing value raises a `DataError` naming the row and the series.

## Supports

The support comes from the emission family (`family.support`), not from the preparation section:

| support | observations after the transform | rule |
|---|---|---|
| `real` | `float64` | finite where `mask` |
| `count` | `int64` (floats accepted if integral) | `>= 0` |
| `binary` | `int64` | in `{0, 1}` |
| `categorical` | `int64`, one series | in `[0, n_categories)`; without `n_categories`: `max + 1` and a warning |

A violation is a `DataError(path="data.source.observation_columns")` that names the support, the row and the
series. Masked values are not checked. `standardize_observations` is only allowed for `real`.

## Standardization

Covariates (and, with `standardize_observations`, the observations) are standardized with mean and standard
deviation (ddof 0) of the training window only (`fit(..., train_end)`); `PrepStats.obs_mean` / `obs_std` hold
the observation statistics (`None` when not standardized; a `prep_stats.json` written by 0.1.x / 0.2.x loads
with `None`).

## Time-less data

Without timestamps the step index is the time axis (`PreparedDataset.times` is `int64`, `train_end` an index).
Covariates without timestamps are aligned by row; lag and staleness count rows.

## Quality report

`hsmmkit.data.quality` holds the generic checks (duplicated timestamps, index gaps `> 1.5 x` the median
spacing, `|y - median| > 8 MAD` outliers, covariate staleness). An application adds its own checks on top.

## Several sequences

A data set with `data.source.sequence_column` holds many independent sequences. `Preparer.fit` pools the training
rows of all sequences for the standardization statistics; `Preparer.transform_batch(obs, covs)` prepares each
sequence on its own (own lags, lagged covariates and first-step masks never cross a boundary) and returns a
`SequenceBatch` (first-appearance order of the ids). `data.sequences.bucket_edges` bound the compiled shapes:
`SequenceBatch.padded()` stacks the sequences into `(B, T_max, ...)` arrays with `T_max` the smallest edge not below
the longest sequence (else the longest length rounded up to a multiple of the last edge) and `B` padded to the next
power of two with empty sequences; `valid` marks the real steps and `unpad` returns the per-sequence arrays.
`data.sequences.group_by` names a source column that is constant within each sequence; it becomes
`PreparedDataset.group` / `PaddedBatch.group_index` (a separate initial distribution per group, E12-T02).
Covariates of a multi-sequence set must be row-aligned columns of the source. Fitting a model on `B > 1` sequences
stays gated until E12-T02 (ADR-0021).
