# Evaluation

Walk-forward evaluation compares realized outcomes against forecast distributions fold by fold. For an origin
`o` and horizon `h` the realized outcome is the cumulative log-return

$$
Y_h = \sum_{k=1}^{h} y_{o+k},
$$

scored against the forecast samples `X_1..X_n` of the cumulative return at step `h`.

## Scores

**Log score.** `\log \hat{f}(Y_h)`, where `\hat{f}` is a Gaussian kernel density estimate of the samples with
Silverman's bandwidth. Higher is better. At `h = 1` the exact mixture density
`\sum_i w_i f_i(Y_1)` over the emission densities is available as `log_score_mixture`.

**CRPS.** The sample estimate

$$
\mathrm{CRPS} = \frac1n\sum_i |X_i - Y_h| - \frac{1}{2n^2}\sum_{i,j} |X_i - X_j|,
$$

computed in `O(n \log n)` from the sorted samples. Lower is better.

**PIT and coverage.** `u = \hat F(Y_h)` is the share of samples not above `Y_h`; it is uniform when the
forecast is calibrated. A central interval between quantile levels `q` and `1-q` covers `Y_h` with
probability `1 - 2q`; the backtest reports the hit indicators and the pooled PIT with a KS test.

**Quantile (pinball) loss** at level `\tau`:

$$
\mathcal{L}_\tau(y, \hat{q}_\tau) = \max\big(\tau (y - \hat{q}_\tau),\, (\tau - 1)(y - \hat{q}_\tau)\big)
$$

**Point errors.** `\mathrm{RMSE}` and `\mathrm{MAE}` of the forecast mean against `Y_h`.

## Folds

An expanding (or rolling) training window is refitted every `refit_every_steps`; each fold forecasts from every
origin in its test block, which always leaves `\max(\text{horizons})` realized rows. The filter is causal,
so the filtered state at origin `o` is the same whether or not later rows exist in the window. Standardization
statistics come from the fold's training window only, so no fold sees the future through preprocessing. See the
[backtesting guide](../guide/backtesting.md) for the exact index arithmetic.

## Strategy accounting

A position decided at origin `o` from the filtered regime probabilities is held over the next step. In the
log-return approximation the strategy return is

$$
r_t = \mathrm{pos}_{t-1}^{\top} y_t - c \cdot 10^{-4} \sum_m |\mathrm{pos}_{t-1,m} - \mathrm{pos}_{t-2,m}|,
$$

with `c` = `cost_bps`. The annualized Sharpe ratio is `\overline{r}/\operatorname{sd}(r) \cdot \sqrt{1/\overline{\Delta}}`;
the maximum drawdown is the largest peak-to-trough fall of the equity curve `\exp(\sum r)` as a fraction of the
peak. Strategies built on the *smoothed* posterior would look ahead; the backtest therefore falls back to the
filtered probabilities and warns.

## Symbols and code names

| Math | Code |
|---|---|
| `Y_h` | cumulative realized return, built in `evaluation.backtest` |
| log score, CRPS, PIT | metrics `log_score`, `crps`, `pit_coverage` in `evaluation.metrics.predictive` |
| `c` | `evaluation.strategy.cost_bps` |
| positions, `r_t` | `BacktestResult.strategy` columns `position`, `return`, `equity`, `turnover` |
| folds | `evaluation.backtest.schedule_folds`, `BacktestResult.folds` |
