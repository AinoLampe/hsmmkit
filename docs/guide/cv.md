# Cross-validation

`hsmmkit cv CONFIG` scores the configured model out of sample. `evaluation.cv` picks the splitter (registry kind
`splitter`):

```yaml
evaluation:
  metrics: [bic, icl]
  cv:
    splitter: sequence_kfold   # or walk_forward
    folds: 5
    seed: 7
```

## Splitters

* **`sequence_kfold`** needs `data.source.sequence_column`. The sequences are shuffled with the key
  `key_for(seed, PURPOSE_CV)` and dealt to `folds` folds, so every sequence is in exactly one test fold and the
  assignment is deterministic per seed. With `data.sequences.group_by` whole groups stay together (a grouped
  K-fold). Fewer sequences (groups) than folds is a `ConfigError`.
* **`walk_forward`** is the calendar scheme of `evaluation.backtest` (one sequence). `cross_validate` runs
  `run_backtest`, so its folds, metrics, summary and hook frames are those of `hsmmkit backtest`.

The preparer is fitted on the **training** part of each fold only (standardization statistics, own lags) and
applied to the test part: nothing of the test data reaches the fit.

## What is scored

Per test sequence, with the parameters fitted on the training sequences:

| metric | meaning |
|---|---|
| `log_score` | the out-of-sample log-likelihood of the sequence per observed step (higher is better); labels and boundaries of the test data never enter it |
| `crps` | sample CRPS of the one-step-ahead predictive distribution (a mixture over the predictive regime probabilities), `forecast.n_sim` draws, averaged over the observed steps (lower is better; `NaN` for the explicit engine) |
| posterior metrics | every configured metric that needs only a decoded posterior (`aic`, `bic`, `icl`, `regime_*`), and the segmentation metrics when the test sequences carry single-regime labels (stored with `horizon = 0`) |

## Outputs

`cv_scores.parquet` (long: `fold, sequence, horizon, metric, value`), `cv_summary.parquet` (`horizon, metric, mean,
sd, n` across folds), `cv_folds.parquet` (sizes, fit time and fit summary per fold) and a *Cross-validation* section
in `report.md`. From Python: `Project.cross_validate(write=True)` or `hsmmkit.selection.cross_validate`.
