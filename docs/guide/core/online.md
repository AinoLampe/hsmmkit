# Online filtering

`hsmmkit.online.ExpandedOnlineFilter` (alias `OnlineFilter`) updates the filtered regime probabilities one
observation at a time, with the arithmetic of the offline forward pass.

```python
from hsmmkit.online import ExpandedOnlineFilter, load_state, save_state

filt = ExpandedOnlineFilter(fit)  # fit: a FitResult of an `engine: expanded` model
state = filt.start(history)  # warm start: offline forward on `history`, keep the last alpha
for y, dt, z in stream:  # y (M,) with NaN = unobserved; z = lagged covariate rows
    state, filtered = filt.update(state, y, dt, z)  # filtered: (N,) regime probabilities
state, filtered = filt.update_many(state, ys, dts, zs)  # lax.scan over update; filtered (T', N)
save_state(state, "run/state")  # state.safetensors + state.json
state = load_state("run/state", fit)  # PersistenceError if it belongs to another fit
```

* **Bitwise equality (NFR-15).** The step is `engine.expanded_hmm.forward_step`, the single body of the offline
  scan, and the emission row of one observation comes from the same code with `T = 1`. `log_alpha`, the filtered
  probabilities (`Posterior.regime_filtered`, evaluated row by row) and `log_norm` equal the offline results
  bitwise on the same backend when the emission's floating-point kernels round independently of the array
  shape (the tested Gaussian / NB scenarios do); a family whose log-density applies a vectorized transcendental
  (`exp`, `log`) to `(T, N, M)` arrays offline but to `(1, N, M)` online can differ in the last bits (about
  `1e-14`), as the fitted Gaussian of the stream tests does. `log_norm` is the running log-likelihood `log p(y_1:t)`.
* **Covariates.** `z` is a mapping `{"transition": (K,), "duration": (K,), "emission": (K,)}` of rows that are
  already lagged and standardized. `OnlineCovariatePreparer(fit.prep_stats, entry_points=...)` applies the stored
  transforms (`log`, `diff`, `pct`), the lag (`>= 1`), the stale limit and the standardization to raw prints:
  `push(prints)` once per grid index, then `z(t)` for step `t`.
* **Time.** `update_from_time(state, time, y, z)` computes `dt` with the configured clock from `state.last_time`.
* **State.** `OnlineState` holds `log_alpha (S,)`, `log_norm`, `t`, `last_time` and the `spec_hash` of the fit
  (sha256 of the model structure and the parameter names and shapes); `load_state` rejects another fit.
* **Fixed-lag smoothing.** With `lag=L` (or `online.lag` of the fit's config) the state carries a `RingBuffer`
  of the last `L + 1` steps (emission terms, covariate rows, `dt`, forward vectors; `O(L (S + K))` memory, `log A_u`
  is rebuilt on demand). `update` then returns `(state', filtered, smoothed)` with `smoothed (N,)` the marginals of
  step `t - L` given the data up to `t` (`NaN` until `L + 1` steps exist); `smoothed_at(state, lag)` evaluates any
  `lag <= L` (`O(lag S^2)`, cost linear in the lag). The result equals the offline smoother run on the *prefix*
  `y_{1:t}` (`data.window(0, t + 1)`) to 1e-12, and the full-data smoother at `t - L` when `t` is the last step; it
  only approximates the full-data smoother for earlier `t`. `hsmmkit.online.buffer.smoothed_series(fit, data, lag)`
  replays a whole dataset (row `t` = `gamma_{t-lag | t}`). The buffer is saved with the state.
* **Alerts.** `online.alerts` (or `alerts=` of the filter) are evaluated after each step in plain Python; the
  active flags live in `OnlineState.alerts` and are saved with it. `update` / `update_many` append the list of
  `AlertEvent(time, t, rule, kind, regime, value, threshold)` (`to_dict` / `from_dict`, JSON-serializable) as their
  last element when rules exist. A probability rule `{regime, p_on, p_off}` fires `on` at `P >= p_on` and `off`
  below `p_off`, so a path that hovers around `p_on` does not flap; `use: smoothed` takes the fixed-lag marginal
  (needs `lag > 0`). A remaining-duration rule `{regime, remaining_duration_below: d, p_min}` fires when
  `P(regime) >= p_min` and the expected remaining duration of the current segment
  (`sum_c P(phase c | regime) (r - c + 1) / a(t)` steps, times `nominal_dt`; phase-type durations) is below `d`; it
  switches off when the remaining duration exceeds `1.2 d` (fixed hysteresis factor) or `P(regime)` drops below
  `p_min`.
* **Forecasting.** `filt.forecast(state, horizon, n_sim, data, covariate_forecaster=..., seed=...)` samples from
  `softmax(state.log_alpha)` with the offline Monte-Carlo forecaster; with the same seed and origin it equals the
  offline `MonteCarloForecaster.forecast` bitwise. `data` is the prepared history ending at the origin (origin
  time, clock, last covariate row, covariate-forecaster training data).
* **Labels and boundaries** are not applied online: a warm-start prefix that carries `labels` / `boundaries` constrains the offline forward pass, but the streaming smoother (`active=False`) does not, so its lagged marginals can differ from the offline prefix smoother in that case.
* **Limits (0.4.0).** `engine: explicit`, time-step memory (`memory_mode: steps`) and families with a recurrent
  emission carry raise `ConfigError("online filtering requires engine: expanded")`; the stream CLI follows in E15-T04. The filter compiles its step once per instance; a step costs about 0.13 ms at `S = 12`
  on a cloud CPU.

## Command line and Python facade

`hsmmkit stream CONFIG --model PATH [--input - | FILE] [--follow] [--state DIR] [--emit filtered,smoothed,events]
[--every k] [--strict]` reads CSV rows (a header with the observation columns, the optional time column and the
raw covariate columns of the configuration) and writes one JSON line per processed row:
`{"t", "time", "filtered", "smoothed_lag", "events"}`. `--every k` thins the output (lines with events are always
written); `--state DIR` resumes from, and checkpoints into, a state directory (every `online.checkpoint_every`
rows, default 1000, and on exit or SIGINT / SIGTERM); `--follow` tails a growing file (poll 0.5 s). A missing
model exits with 2, a missing column with 3 (naming it); a malformed row is skipped with a message on stderr, or
exits with 3 under `--strict`.

`Project.online(fit, initial=None, lag=None)` returns an `OnlineSession` with `push(row)`, `push_many(df)`,
`events`, `save(dir)` / `load(dir)` and `forecast(horizon)`; the command uses the same session, so their outputs
are identical. Covariates are dated by row (print `t + 1 - lag`), leading rows without a covariate print are
skipped as the offline `Preparer` trims them. The identity observation transform and emissions without own lags
are supported. Against the offline filter the stream agrees to about `1e-14` (see above).
