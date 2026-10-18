# Troubleshooting

Errors carry the configuration path or array name they concern (`model.duration.r[2]: ...`). The exit code
tells the family:

| Exit | Error | Typical cause |
|---|---|---|
| 2 | `ConfigError` | invalid or inconsistent configuration, an unknown key, a bad `--device` or `--params` value |
| 3 | `DataError` | too few rows, an unusable gap, snapshots that differ in `compare`, no cached data with `--offline` |
| 4 | `NumericalError` | non-finite values, x64 disabled, `--device gpu` without a GPU, failed fits |
| 5 | `ProviderError` | the provider failed, a missing API key, a quota or rate limit |
| 1 | `PersistenceError` or anything unexpected | unwritable output directory, a malformed or newer model bundle; rerun with `--log-level DEBUG` for the traceback |

## `NumericalError`

- **Non-finite log-likelihood in `decode`/`forecast`.** The model has a parameter combination the data
  cannot support, or the bundle holds NaNs. Refit; check `fit_report.json` for `converged` and the restart
  table.
- **No restart produced a finite fit.** Too few observations for the model size (regimes, phases `r_i`,
  covariates), or starting values far from the data. Reduce `model.n_states` or `model.duration.r`, add
  restarts (`estimation.restarts`) or initializations (`estimation.init`).
- **`x64 disabled`.** Something initialized JAX in 32-bit mode before regimekit was imported. Import
  `regimekit` first; regimekit requires float64.
- **`--device gpu` fails.** No CUDA backend is installed. Use `--device cpu` or `auto`, or install the `gpu`
  extra on Linux.
- **Forecast too large.** `n_sim * horizon * M > 2e8`; reduce `forecast.n_sim` or the horizon.

## Data

- `no prepared dataset at ... and --offline forbids fetching one`: run `fetch` once without `--offline`.
- `a pinned snapshot id ... cannot pin covariates`: a pinned `data.cache.snapshot` identifies one cache entry;
  use `snapshot: latest` with `--offline` to replay a run with covariates.
- `quality.json` / `quality.md` list gaps, duplicates, outliers, staleness and the rows dropped by the
  preparation policies (`missing`, `max_staleness`, `data.preparation.missing_returns`).

## Fits that look wrong

- **`restart_dispersion` warns** (the two best restarts differ by more than 1 %): the likelihood is
  multimodal; raise `estimation.restarts` and compare `loglik` across restarts.
- **A regime holds less than 2 % of the steps** (`regime_occupancy` warns): too many states or fixed
  parameters that do not match the data.
- **Sojourn estimates look unstable.** Mean sojourn and jump probabilities need many regime visits: a few
  hundred visits for a 25 % relative error, which is thousands of daily observations for slow regimes (see
  [Validation](validation.md)). `regimekit selftest` is a smoke test, not a precision check.

## Backtests

- A schedule without any fold is a `ConfigError` (`evaluation.backtest`): the sample is too short for
  `initial_train_years`, `refit_every_steps` and the longest horizon; shorten them.
- A backtest refits the model in every fold; use fewer `estimation.restarts` while exploring.
