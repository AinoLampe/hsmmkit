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
