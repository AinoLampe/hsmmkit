# Forecasting

!!! note
    Stub page (E0-T04). Filled in by E6-T04a/b (Monte Carlo sampler, aggregation); completed in E8-T04.

Forecast distributions are estimated by Monte Carlo simulation of `n_sim` future paths from the posterior
regime distribution at the forecast origin:

$$
\hat{F}(y_{T+h}) = \frac{1}{n_{\text{sim}}} \sum_{s=1}^{n_{\text{sim}}} \mathbb{1}\!\left[y_{T+h}^{(s)} \le y\right]
$$
