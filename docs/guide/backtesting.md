# Backtesting

!!! note
    Stub page (E0-T04). Filled in by E7-T01…E7-T04 (metrics/walk-forward/strategies/CLI); completed in E8-T04.

See [Math → Evaluation](../math/evaluation.md) for the governing metric formulas.

## Metrics

`regimekit.evaluation.metrics` holds a name-resolved registry (`resolve(names)`; an unknown name raises
`ConfigError` at `evaluation.metrics`). A metric is a function of a context (`forecast`, `realized`,
`horizon`, `posterior`, `fit`, `labels`, `strategy`) with declared `needs`; a runner skips metrics whose
needs it cannot meet. Predictive metrics compare the realized cumulative return `Y_h = sum_{k<=h} y_{t+k}`
with the forecast samples of the cumulative return at step `h`.

| Name | Needs | Value |
|---|---|---|
| `log_score` | forecast | log of a Gaussian-KDE density (Silverman bandwidth) of the samples at `Y_h`; higher is better |
| `log_score_mixture` | forecast, fit | `horizon == 1` only: `log sum_i w_i f_i(y)` with the emission family's exact densities (`dt_next`, optional `z_emission_next` in the context). Weights: `mixture_weights` from the context when given (exact, `one_step_mixture_weights`), else the forecast's one-step regime probabilities (Monte-Carlo estimate) |
| `crps` | forecast | sample CRPS `E|X - Y| - 1/2 E|X - X'|`, computed in `O(n log n)` from the sorted samples; lower is better |
| `pit_coverage` | forecast | `pit` (share of samples `<= Y_h`) plus `coverage_<level>` hit indicators for each central interval `(q, 1 - q)` among the configured quantiles; average the indicators for the hit rate, pool `pit` for a KS test |
| `rmse`, `mae` | forecast | `abs(mean[h-1] - Y_h)` per origin; `rmse` declares `aggregate = "rms"` |
| `aic`, `bic`, `icl` | posterior | information criteria of the decoded fit |
| `regime_occupancy` | posterior | share of steps per regime on the Viterbi path |
| `regime_persistence` | posterior | mean Viterbi segment length per regime (regimes that never occur are omitted) |
| `balanced_accuracy` | posterior, labels | mean per-label recall of the Viterbi regimes against `labels`, maximised over relabellings (at most 6 distinct regimes/labels) |
| `sharpe`, `max_drawdown` | strategy | registered by the strategy ticket (E7-T03); until then calling them raises |

The forecast metrics read `forecast.samples`, so the forecast must keep its samples. With `M > 1` series,
scalar metrics return `<name>_<m>` entries.

## Walk-forward folds

`regimekit.evaluation.backtest.schedule_folds(times, dt, cfg, max_horizon)` turns `evaluation.backtest`
into a fold table (pure index arithmetic; indices are grid rows, ends exclusive, like
`PreparedDataset.window`).

* The initial training window is the shortest prefix whose elapsed time `cumsum(dt)` reaches
  `initial_train_years` (504 rows for 2 years on the trading clock; under `clock: steps` a "year" is a row).
* Fold `k` is fitted on rows `[train_start, train_end)` with `train_end = initial_end + k * refit_every_steps`;
  its forecast origins are `[test_start, test_end)` with `test_start = train_end + embargo_steps` and
  `test_end = min(test_start + refit_every_steps, T - max_horizon)`, so every origin has `max_horizon`
  realized future rows. Folds without origins are dropped; a schedule without any fold is a `ConfigError`.
* `expanding` folds start at row 0. `rolling` folds start at the latest row whose window still spans
  `rolling_years` of elapsed time (constant length up to one row on a regular grid); `rolling_years` may not
  exceed `initial_train_years`.
* For an origin `o` and horizon `h` the realized return is `Y_h = sum(y[o+1 .. o+h])`.

`fold_table(folds, times)` adds the timestamps of the first/last training row and the first/last origin.

## Running a walk-forward backtest

`run_backtest(cfg, obs, covs, seed) -> BacktestResult` executes the folds of `schedule_folds` end to end.
For every fold:

1. `Preparer.fit(train_end=<last train row>)` and `transform` of the whole sample. Standardization statistics
   come from the fold's train window only, `[train_start, train_end)` (`PrepStats.train_window` records it); a covariate with `lag: 1`
   reaches step `t` only through prints stamped at or before `tau_{t-1}`.
2. `fit` (E5-T03) on `window(train_start, train_end)`. The previous fold's optimum is added as one extra
   restart (`init_strategy == "warm_start"`) next to the `estimation.restarts` regular ones.
3. One forward pass over `window(0, test_end)` gives the filtered state at every origin. The filter is
   causal, so row `o` is exactly the filter of `window(0, o + 1)`; the Monte-Carlo forecast of horizon
   `max(horizons)` starts from it with the seed `derive_seed(seed, PURPOSE_BACKTEST_FORECAST, fold, o)`
   (`evaluation.backtest.n_sim` simulated paths, default `forecast.n_sim`).
4. Every configured forecast metric is evaluated per horizon against the realized `Y_h`. The first forecast
   step uses the last *observed* covariate row (see [Forecasting](forecasting.md)).

Metrics that need a decoded posterior (`aic`, `bic`, `icl`, `regime_*`) are evaluated once per fold on the
train window and stored with `horizon == 0`; metrics that need labels or a strategy trace are skipped
(strategies: E7-T03). Dict-valued metrics become `<metric>.<key>` rows (`pit_coverage.coverage_90`).

`BacktestResult.folds` is the `fold_table` plus `fit_time` (seconds). `metrics` has one row per
`(fold, horizon, metric)` (mean over the fold's origins; `rmse` is the root mean square). `summary` has one
row per `(horizon, metric)` with `mean`, `sd` across folds and `n` folds (`rmse`: root mean square over
folds). `pit_ks` / `pit_ks_pvalue` rows pool the PIT values of all origins per horizon (`n` = pooled
origins); for horizons above 1 consecutive origins overlap, so the p-value is optimistic there.

### Padding and compilation

Every distinct window length `T` is a new compilation of the jitted engine functions. Fit and filter windows
are therefore padded at the end to the next multiple of 256 rows (`evaluation.padding.pad_dataset`): the
extra rows have `mask=False`, `observations=0`, the last `dt` and zero covariates, so they add no
log-density and leave the filter rows `0..T-1` untouched. Folds that pad to the same length share one
compilation of the objective and of the engine. The data-driven initializations (`rollvol_kmeans`, `random`)
read only the observed rows (`estimation.init.observed_rows`), so padding cannot bias the starting values.

The posterior metrics are the exception to padding: they decode the *unpadded* train window, because the
Viterbi path of the last rows is not invariant to trailing masked rows. Their decode therefore compiles once
per distinct train length (per fold under the expanding scheme); the fit and the filter compile once per
padded length.
