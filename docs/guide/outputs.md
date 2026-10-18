# Output files

Every command writes into the output directory (`project.output_dir`, or `--out`). Tables are parquet
with the schemas below; `time` columns are `timestamp[ns]` (UTC, naive).

## `forecast.parquet` (long)

Written by `regimekit forecast CONFIG --model DIR [--horizon H] [--n-sim N] [--scenario PATH] [--samples]`.

| Column | Type | Meaning |
|---|---|---|
| `origin_time` | timestamp[ns] | last observed time |
| `horizon_time` | timestamp[ns] | time of the forecast step (weekday calendar) |
| `step` | int64 | 1-based forecast step |
| `kind` | string | `regime_prob`, `quantile`, `mean` or `var` |
| `key` | string | regime index (`regime_prob`), quantile level as `repr(float)` (`quantile`), series index (`mean`/`var`) |
| `series` | int64 | series index `m` (0 for `regime_prob`) |
| `value` | float64 | the value; `quantile`/`mean`/`var` refer to the **cumulative** log-return from the origin |

`forecast.json` holds the summary (`origin_time`, `horizon`, `n_sim`, `n_series`, `n_states`,
`covariate_forecaster`, `seed`, `quantile_levels`, `snapshot_ids`). With `--samples`,
`forecast_samples.parquet` (`path`, `step`, `series`, `value`; all int64 except `value` float64) holds the
per-step simulated log-returns; without it any older samples file is removed. `read_forecast(dir)`
(`regimekit.persistence.results`) restores a `ForecastResult` bit for bit (`samples` is empty without the
samples file).

`--scenario PATH` is a CSV with one column per covariate (names as in `prepared.parquet`, `h` rows; a
`time`/`date` column is ignored) in the *prepared* units (transformed and standardized); it selects the
`scenario` covariate forecaster, columns not listed stay frozen at their last value.

## `simulation.parquet` (long)

Written by `regimekit simulate CONFIG [--params config|model] [--model DIR] [--steps T] [--n-paths P]`.

| Column | Type | Meaning |
|---|---|---|
| `path` | int64 | path index |
| `step` | int64 | 1-based step |
| `time` | timestamp[ns] | the prepared dataset's times, extended with their median spacing beyond its length |
| `regime` | int64 | regime index |
| `phase` | int64 | 1-based phase within the regime |
| `return` | float64 | simulated log-return (`return_0 .. return_{M-1}` when `M > 1`) |

`simulation_segments.json` lists, per path, the run-length-encoded regime segments (`start`, `end`
inclusive, `regime`). `--params config` uses the parameters' `init`/`values` from the configuration;
`dt` and the covariates come from the prepared dataset and cycle through its rows when `--steps` exceeds
its length. The explicit-duration engine cannot be simulated (it has no `A_t`).

## Backtest files

Written by `regimekit backtest CONFIG` into the output directory (the data come through the cache, so
`--offline` replays a run). `horizon == 0` marks metrics that are not forecast metrics: the in-sample
criteria and regime shares of each fold's fit, and the strategy metrics of each fold's slice.

| File | Content |
|---|---|
| `backtest_metrics.parquet` | long: `fold` (int64), `horizon` (int64), `metric` (string), `value` (float64); one row per `(fold, horizon, metric)`, dict-valued metrics as `<metric>.<key>` |
| `backtest_folds.parquet` | the fold table: row indices (`train_start .. test_end`, ends exclusive), their timestamps, `fit_time` (s) and the fit summary `objective`, `loglik`, `converged`, `n_iter`, `restart_gap` (gap between the two best restarts) |
| `backtest_summary.parquet` | `horizon, metric, mean, sd, n`: across folds; `pit_ks` / `pit_ks_pvalue` pool the origins (`n` = origins) |
| `strategy.parquet` | with `evaluation.strategy` only: `fold, time, position` (`position_<m>` for `M > 1`), `return`, `gross`, `equity`, `turnover`, `dt` |
| `backtest.json` | `config`, `seed`, `snapshot_ids` (primary first, then the covariates), `n_free`, `versions`, `wall_s`; read by `compare --reuse` |
| `report.md` | header, model, fit, diagnostics (with warnings), backtest summary, strategy, timings |

`read_backtest(dir)` (`regimekit.persistence.results`) restores the `BacktestResult` and the metadata.
`output.write` is not consulted: `backtest` always writes these files.

## `compare.parquet` and `compare.md`

Written by `regimekit compare CONFIG... [--reuse] [--allow-mixed] [--metric M]...` into the output directory
(`--out`, default `runs/compare`). Each configuration is backtested into its own `project.output_dir`
(`--reuse` takes a stored result instead of re-running). The runs must share their data snapshot ids,
otherwise the command exits 3 before fitting anything, unless `--allow-mixed` is given. One row per
configuration: `config`, `n_free`, `fit_time`, then `<metric>` (in-sample, horizon 0: `bic`, `icl`) and
`<metric>@h<k>` (out-of-sample) with their across-fold sd in `<column>_sd`. `compare.md` shows
`mean ± sd` and marks the best value in each column whose direction is known (lower: `crps`, `rmse`, `mae`,
`aic`, `bic`, `icl`, `max_drawdown`, `fit_time`; higher: `log_score`, `log_score_mixture`, `sharpe`,
`hit_rate`). `--metric` keeps only the named metrics.
