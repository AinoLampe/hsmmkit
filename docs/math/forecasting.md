# Forecasting

Forecast distributions are estimated by Monte Carlo simulation of `n_sim` future paths from the posterior
regime distribution at the forecast origin `t`:

$$
\hat{F}(y_{T+h}) = \frac{1}{n_{\text{sim}}} \sum_{s=1}^{n_{\text{sim}}} \mathbb{1}\!\left[y_{T+h}^{(s)} \le y\right]
$$

## The sampler

Inputs are a fitted model, the `PreparedDataset` up to the origin, the horizon `h`, `n_sim`, a covariate
forecaster and an integer seed.

1. `\alpha_t` is the **filtered** augmented posterior over the expanded states at the origin (one causal
   filter pass).
2. For each of the `n_sim` paths: draw `s \sim \alpha_t`; then for `k = 1..h`
    - obtain the covariates `z_{t+k}` (the known row for `k = 1`, the covariate forecaster afterwards),
    - build the transition `A_{t+k}(z_{t+k})` with the step size `\Delta_{t+k}` of the forecast grid and draw
      `s \sim A_{t+k}[s, \cdot]`,
    - draw `y_{t+k}` from the emission of the regime of `s` with `\Delta_{t+k}`.
3. Aggregate: regime probabilities `(h, N)`, per-step samples `(n_{sim}, h, M)`, quantiles, mean and variance
   of the **cumulative** log-return `Y_k = \sum_{u \le k} y_{t+u}`.

The step sizes follow the calendar: the next weekdays after the last observation (see
[Time conventions](time.md)). All randomness derives from one integer seed
(`key_for(seed, PURPOSE_FORECAST)`), so a forecast is bitwise reproducible for a fixed seed and backend.
The first forecast step uses the last *observed* covariate row; the covariate that becomes known at the
origin is not used yet.

## Covariate forecasters

| Type | Rule |
|---|---|
| `freeze` | `z_{t+k} = z_t` |
| `scenario` | a user-supplied path `(h, K)` in prepared units; missing columns fall back to `freeze` |
| `bootstrap` | stationary block bootstrap of the historical first differences (all covariates jointly), mean block length `forecast.covariate_forecaster.mean_block_length`, re-standardized with the training statistics |
| `ar` | VAR(1) fitted on the training window by OLS with Gaussian innovations; per-covariate AR(1) when `diagonal: true` |

## Accuracy

The Monte Carlo error of a quantile at level `\tau` is about `\sqrt{\tau(1-\tau)/n_{sim}}` divided by the
density at the quantile; moments have standard error `\sigma/\sqrt{n_{sim}}`. The simulation moments are
tested against analytic values within three standard errors. A request with `n_sim \cdot h \cdot M > 2 \cdot 10^8`
is refused with a `NumericalError`.

## Symbols and code names

| Math | Code |
|---|---|
| `\alpha_t` | filtered posterior of the expanded engine (`Posterior`) |
| `h`, `n_{sim}` | `forecast.horizon`, `forecast.n_sim` |
| `Y_k` quantiles | `ForecastResult.cum_quantiles` `(h, Q, M)` at `quantile_levels` |
| per-step samples | `ForecastResult.samples` `(n_sim, h, M)` |
| regime probabilities | `ForecastResult.regime_probs` `(h, N)` |
| sampler | `regimekit.inference.forecast.MonteCarloForecaster` |
