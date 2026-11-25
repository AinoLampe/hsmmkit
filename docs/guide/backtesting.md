# Backtesting

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
| `sharpe`, `max_drawdown`, `turnover`, `hit_rate` | strategy | see [Strategies](#strategies) |

The forecast metrics read `forecast.samples`, so the forecast must keep its samples. With `M > 1` series,
scalar metrics return `<name>_<m>` entries.

## Walk-forward folds

`regimekit.evaluation.backtest.schedule_folds(times, dt, cfg, max_horizon)` turns `evaluation.backtest`
into a fold table (pure index arithmetic; indices are grid rows, ends exclusive, like
`PreparedDataset.window`).

* The initial training window is the shortest prefix whose elapsed time `cumsum(dt)` reaches
  `initial_train` (504 rows for 2 years on the trading clock; under `clock: steps` a "year" is a row).
* Fold `k` is fitted on rows `[train_start, train_end)` with `train_end = initial_end + k * refit_every_steps`;
  its forecast origins are `[test_start, test_end)` with `test_start = train_end + embargo_steps` and
  `test_end = min(test_start + refit_every_steps, T - max_horizon)`, so every origin has `max_horizon`
  realized future rows. Folds without origins are dropped; a schedule without any fold is a `ConfigError`.
* `expanding` folds start at row 0. `rolling` folds start at the latest row whose window still spans
  `rolling_window` of elapsed time (constant length up to one row on a regular grid); `rolling_window` may not
  exceed `initial_train`. `initial_train_steps` and `rolling_window_steps` give the same lengths in rows (exactly one of `initial_train` / `initial_train_steps`).
* For an origin `o` and horizon `h` the realized return is `Y_h = sum(y[o+1 .. o+h])`.

`fold_table(folds, times)` adds the timestamps of the first/last training row and the first/last origin.

## Running a walk-forward backtest

`run_backtest(cfg, obs, covs, seed) -> BacktestResult` executes the folds of `schedule_folds` end to end.
For every fold:

1. `Preparer.fit(train_end=<last train row>)` and `transform` of the whole sample. Standardization statistics
   come from the fold's train window only, `[train_start, train_end)` (`PrepStats.train_window` records it); a covariate with `lag: 1`
   reaches step `t` only through prints stamped at or before `tau_{t-1}`.
2. `fit` (E5-T03) on `window(train_start, train_end)`. The previous fold's optimum is added as one extra
   restart (`init_method == "warm_start"`) next to the `estimation.restarts` regular ones.
3. One forward pass over `window(0, test_end)` gives the filtered state at every origin. The filter is
   causal, so row `o` is exactly the filter of `window(0, o + 1)`; the Monte-Carlo forecast of horizon
   `max(horizons)` starts from it with the seed `derive_seed(seed, PURPOSE_BACKTEST_FORECAST, fold, o)`
   (`evaluation.backtest.n_sim` simulated paths, default `forecast.n_sim`).
4. Every configured forecast metric is evaluated per horizon against the realized `Y_h`. The first forecast
   step uses the last *observed* covariate row (see [Forecasting](forecasting.md)).

Metrics that need a decoded posterior (`aic`, `bic`, `icl`, `regime_*`) are evaluated once per fold on the
train window and stored with `horizon == 0`; metrics that need labels or a strategy trace are skipped
(see [Strategies](#strategies)). Dict-valued metrics become `<metric>.<key>` rows (`pit_coverage.coverage_90`).

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
compilation of the objective and of the engine. The data-driven initializations (`rolling_kmeans`, `random`)
read only the observed rows (`estimation.init.observed_rows`), so padding cannot bias the starting values.

The posterior metrics are the exception to padding: they decode the *unpadded* train window, because the
Viterbi path of the last rows is not invariant to trailing masked rows. Their decode therefore compiles once
per distinct train length (per fold under the expanding scheme); the fit and the filter compile once per
padded length.

## Strategies

`evaluation.strategy` maps the state of knowledge at an origin to positions in `[-1, 1]`
(`regimekit.finance.strategies.make_strategy(cfg, nominal_dt, n_series)`). A position is *decided* at an
origin from its filtered regime probabilities and forecast and *held* over the next step; the strategy
return of a step is, in the log-return approximation,

`r_t = pos_{t-1} . y_t - cost_bps * 1e-4 * sum_m |pos_{t-1,m} - pos_{t-2,m}|`  (flat before the first trade),

and the equity curve is `exp(cumsum(r))`.

* `regime_long_flat` (`flat_if: {regime, prob_above, use}`): position 1, flat (0) when
  `P(regime) > prob_above` (strict). `use: smoothed` is look-ahead: it warns, and a backtest uses the filtered
  probabilities of the origin instead.
* `regime_target_vol` (`target_vol`, `cap = 1.0`, `allow_short = False`): magnitude
  `min(cap, target_vol / sigma_pred)` with `sigma_pred = sqrt(var[0] / nominal_dt)` from the origin's
  one-step forecast (annualized on the trading/calendar clocks, per step on `steps`); long when the
  predictive mean is non-negative, otherwise flat, or short with the same magnitude when `allow_short`.

`run_backtest(cfg, data_loader, seed, hooks=[StrategyHook(...)])` returns the trace in
`BacktestResult.extras["strategy"]` (`regimekit.finance.strategy_frame(result)`): `fold, time, position`
(`position_<m>` for `M > 1`), `return`, `gross` (`position . y`, before costs), `equity`, `turnover`
(`sum_m |change|`) and `dt`; `time` is the end of the step the position was held over. Where an origin's
next return is missing no position is recorded. The metrics are evaluated on each fold's slice (equity restarts
at 1) and stored with `horizon == 0`:

| Metric | Value |
|---|---|
| `sharpe` | `mean(r) / std(r) * sqrt(1 / mean(dt))`; a constant non-zero return gives `+-inf`, an all-zero one 0 |
| `max_drawdown` | largest peak-to-trough fall of the equity curve as a fraction of the peak, `>= 0` (equity starts at 1) |
| `turnover` | `sum |change in position|` per unit of clock time (per year on the trading/calendar clocks) |
| `hit_rate` | share of the invested periods with `position . y > 0` |

## Command line

`regimekit backtest CONFIG` fetches the data (respecting `--offline`), runs `run_backtest` with the
configured strategy and writes the files listed in [Outputs](outputs.md#backtest-files), including
`report.md`. `regimekit compare CONFIG... [--reuse] [--allow-mixed] [--metric M]` backtests several
configurations on the same data snapshot and writes `compare.parquet` / `compare.md`.

## Origin hooks

`run_backtest(cfg, data_loader, seed, hooks=(), on_fold=None)` takes `data_loader() -> (ObservationSet, covariates)`
and a sequence of `OriginHook`s (E10-T07a). A hook has a `name`, an `on_origin(fold, origin, posterior,
forecast, data)` that gives the values of one row (or `{}` for none) and a `finalize(rows)` that builds the frame
stored in `BacktestResult.extras[name]`. Metrics whose `needs` name a hook are evaluated per fold on that fold's
slice of the frame (`horizon == 0`, context `extras`). `build_report(..., extra_sections=[...])` appends the
sections the hooks contribute (the finance application adds the strategy table).
