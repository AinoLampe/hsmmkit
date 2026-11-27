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
  bitwise on the same backend. `log_norm` is the running log-likelihood `log p(y_1:t)`.
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
* **Labels and boundaries** are not applied online: a warm-start prefix that carries `labels` / `boundaries` constrains the offline forward pass, but the streaming smoother (`active=False`) does not, so its lagged marginals can differ from the offline prefix smoother in that case.
* **Limits (0.4.0).** `engine: explicit`, time-step memory (`memory_mode: steps`) and families with a recurrent
  emission carry raise `ConfigError("online filtering requires engine: expanded")`; alerts follow in E15-T03. The filter compiles its step once per instance; a step costs about 0.13 ms at `S = 12`
  on a cloud CPU.
