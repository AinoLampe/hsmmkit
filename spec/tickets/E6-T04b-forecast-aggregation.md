# E6-T04b — Forecast aggregation → `ForecastResult` (regime probs, quantiles, moments, future dates)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-I04, NFR-03 | M4 | §4.9 step 3, §5.4 (`ForecastResult`), §5.5 |

## Goal
Provide the user-facing `MonteCarloForecaster.forecast(...)` returning a complete, deterministic `ForecastResult`.

## Context (self-contained)
```python
class MonteCarloForecaster:
    def __init__(self, spec, space, engine, forecaster: CovariateForecaster, cfg: ForecastSection): ...
    def forecast(self, fit: FitResult, data: PreparedDataset, horizon: int, n_sim: int, seed: int) -> ForecastResult
```
Steps: run `engine.forward` on `data` (or reuse a supplied `Posterior`) → `log_alpha[-1]`; fit covariate forecaster on `data` (if any covariates and `h > 1`) → `cov_paths`; `future_grid` → `horizon_times, dt_future`; `forecast_paths` (E6-T04a) → aggregate:
- `regime_probs (h, N)` = mean over paths of one-hot regimes.
- `samples (n_sim, h, M)` per-step log-returns.
- `cum = cumsum(samples, axis=1)`; `cum_quantiles (h, Q, M)` at `cfg.quantiles`; `mean (h, M)`, `var (h, M)` of `cum`.
- `ForecastResult(origin_time=times[-1], horizon_times, regime_probs, samples, cum_quantiles, mean, var, quantile_levels, covariate_forecaster=name, seed)`.
Keys: `key_for(seed, "forecast")`-style ids (ints: use a fixed id constant per purpose). Memory guard: `n_sim·h·M > 2e8` floats → `NumericalError` suggesting fewer sims.
Analytic check for `h=1`: predictive mixture with weights `w = (softmax(α_T) A_{T+1}) C` and GBM components: `mean = Σ w_i (μ_i − σ_i²/2)Δ`, `var = Σ w_i σ_i²Δ + Σ w_i m_i² − (Σ w_i m_i)²`.

## Creates
- `src/regimekit/inference/forecast.py` (`MonteCarloForecaster`)
- `tests/unit/test_e6_t04b_forecast.py`, `tests/integration/test_e6_t04b_determinism.py`
- `docs/guide/forecasting.md`

## Acceptance criteria
1. `h=1`: `mean`/`var` match the analytic mixture within MC error (3 SE with `n_sim=20000`).
2. `regime_probs` rows sum to 1 ≤ 1e-12; `cum_quantiles` monotone in `Q`.
3. Same (fit, data, seed) ⇒ bitwise identical `samples` across two calls.
4. `h > 1` with covariates requires a forecaster; `freeze` reproduces constant covariates (assert via equality of regime probs against a run with covariates hard-coded).
5. Explicit-engine fit produces a result of the same shapes.

## Out of scope
CLI/writers (E6-T05); metrics (E7-T01).

## Connections
- Depends on: E6-T04a
- Blocks: E6-T05, E7-T01, E7-T02b, E8-T02
- Relates to: E5-T04 (PIT reuse)
