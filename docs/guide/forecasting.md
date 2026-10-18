# Forecasting

See [Math → Forecasting](../math/forecasting.md) for the governing Monte Carlo aggregation formula.

## Decode

`regimekit decode CONFIG --model DIR [--augmented]` turns a fitted model plus `CONFIG`'s prepared
dataset into a decoded posterior: filtered and smoothed regime probabilities, the Viterbi path, its
run-length-encoded segments, and the AIC/BIC/ICL information criteria.

`--model DIR` points at a model bundle (`<name>.rkmodel/`, see [Model bundle](bundle.md)) written by
`ModelStore.save`. `prepared.parquet` is read from the output directory, or fetched first if missing and
`--offline` is not set.

Outputs, written to the output directory:

| File | Contents |
|---|---|
| `posterior.parquet` | `time`, `filtered_0..N-1`, `smoothed_0..N-1`, `viterbi` (`T` rows, `2N+2` columns) |
| `augmented.parquet` | `time` x `S` smoothed expanded-state posterior (only with `--augmented`) |
| `segments.json` | Run-length encoding of the Viterbi regime path: `{"start", "end", "regime"}` (`end` inclusive) |
| `criteria.json` | `loglik`, `aic`, `bic`, `icl` |

The same logic is available as a library call: `regimekit.inference.decode.decode(fit, data) ->
PosteriorResult`.

## Covariate forecasters

Multi-step forecasting needs future values for every covariate the model conditions on
(`forecast.covariate_forecaster`). Four strategies (`regimekit.inference.covariate_forecasters`):

| `type` | Behaviour |
|---|---|
| `freeze` | Every forecast step repeats the last known value. |
| `scenario` | A user-supplied path for named columns (`path`, loaded by the caller -- `inference/` has no file I/O); unmentioned columns freeze. |
| `bootstrap` | A stationary block bootstrap of historical first differences, cumulated forward from the last known value (`mean_block_length`). |
| `ar` | A VAR(1) fit by OLS on the training window, with Gaussian innovations (`diagonal: true` fits each column independently). |

Every forecaster works on the *unified* set of covariate columns -- the union across the
transition/duration/emission blocks, de-duplicated by name (`ColumnMap`) -- since the same covariate
can drive more than one part of the model.

## Forecast paths

`regimekit.inference.forecast.forecast_paths` samples `n_sim` future paths of horizon `h` from the
filtered posterior at the origin: each path starts in a state drawn from `softmax(log_alpha_T)`, then
applies `A_{T+k}` (covariate row `k-1`, `Δ` from `future_grid`) and draws the return from the new
regime's emission. `forecast_paths_explicit` does the same for the explicit-duration engine: it draws the
origin segment's remaining sojourn from the pmf conditioned on `d >= elapsed` (right-censored at the
origin), then fresh durations for later segments. Durations use `nominal_dt` (regular-grid
assumption, ADR-0007).

`future_grid(last_time, h, clock, bars_per_day)` (in `regimekit.utils.clock`, re-exported by
`regimekit.data.calendar`) builds the horizon timestamps on a **weekday calendar** -- Saturdays and
Sundays are skipped, exchange holidays are *not* (no holiday calendar in 0.1.0). `Δ` is `1/252`
(trading), the real gap in days / 365.25 (calendar) or `1` (steps).

## `MonteCarloForecaster`

```python
mc = MonteCarloForecaster(spec, space, engine, forecaster, cfg.forecast)
result = mc.forecast(fit, data, horizon=20, n_sim=10_000, seed=0)
```

The filter pass (`engine.forward`, or a supplied `Posterior`) gives `log_alpha_T`; covariate paths come
from the configured forecaster (fitted on the training window only); `forecast_paths` /
`forecast_paths_explicit` sample the future. The `ForecastResult` holds:

| Field | Shape | Meaning |
|---|---|---|
| `regime_probs` | `(h, N)` | mean of the one-hot regime over paths; rows sum to 1 |
| `samples` | `(n_sim, h, M)` | per-step log-returns |
| `cum_quantiles` | `(h, Q, M)` | quantiles (`forecast.quantiles`) of the cumulative log-return |
| `mean`, `var` | `(h, M)` | mean / variance of the cumulative log-return |
| `horizon_times` | `(h,)` | weekday timestamps after `origin_time` |

Notes: the output is bitwise reproducible per `(fit, data, horizon, n_sim, seed)`; `n_sim * h * M > 2e8`
raises `NumericalError`; with covariates and `h > 1` a covariate forecaster is required, and the *last
observed* covariate row stands in for the (unobserved) row of the first forecast step. For the explicit
engine the unchunked `forward` is used, so very long samples (`T * N * D_max > 5e7`) are not supported.
