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
* **Limits (0.4.0).** `engine: explicit`, time-step memory (`memory_mode: steps`) and families with a recurrent
  emission carry raise `ConfigError("online filtering requires engine: expanded")`; fixed-lag smoothing and alerts
  follow in E15-T02/T03. The filter compiles its step once per instance; a step costs about 0.13 ms at `S = 12`
  on a cloud CPU.
