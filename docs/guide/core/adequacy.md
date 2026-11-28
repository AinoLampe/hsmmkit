# Model adequacy: pseudo-residuals and posterior predictive checks

`estimation.diagnostics` selects the adequacy sections that `hsmmkit fit` adds to `fit_report.json`
(`diagnostics.pseudo_residuals`, `diagnostics.ppc`); the default is `[pseudo_residuals]`.

```yaml
estimation:
  diagnostics: [pseudo_residuals, ppc]
  ppc_replicates: 200
```

## Pseudo-residuals

```python
from hsmmkit.selection import pseudo_residuals, shapiro_wilk

res = pseudo_residuals(fit, data)
res.table  # per series: n, shapiro_w, shapiro_p, Ljung-Box p of r and r^2 at lags 10 / 20
res.qq  # theoretical vs sample quantiles, ready to plot
```

The one-step predictive distribution of series `m` is
`F_t(y) = Σ_i P(regime_t = i | y_{1:t−1}) F_{i,m}(y)` with the predictive regime probabilities of the filter.

| Support | Residual |
|---|---|
| `real` | `r_t = Φ⁻¹(F_t(y_t))` |
| `count`, `binary`, `categorical` | mid-distribution `u_t = ½ (F_t(y_t) + F_t(y_t − 1))`, `F_t(−1) = 0`, then `Φ⁻¹` |
| Gaussian with `full` covariance, `M > 1` | sequential Rosenblatt transform along the series order, with the regime weights updated by the earlier series of the same step |

A correct model gives i.i.d. standard normal residuals: look at the QQ data, the Shapiro–Wilk `W` and the
Ljung–Box tests of `r` (serial dependence left in the mean) and of `r²` (left in the variance). Unobserved steps
have no residual. Families without a usable `cdf` are skipped with a note.

!!! note "Discrete supports"
    Mid-distribution residuals have atoms, so a Shapiro–Wilk test on a long series detects the lattice even for a
    correct model (means below ~20, or thousands of steps). Judge discrete models by the QQ plot and `r²`
    autocorrelation; the category order of a categorical model is arbitrary, so its residuals are only reported.

`shapiro_wilk(x)` is an own implementation of Royston's AS R94 (3 ≤ n ≤ 5000; longer samples use 5000 evenly
spaced order statistics) and equals `scipy.stats.shapiro` to better than 1e-6 in `W` and `p`.

## Posterior predictive checks

```python
from hsmmkit.selection import ppc

chk = ppc(fit, data, R=200, seed=0, statistics=("acf_abs", "sojourn_mean", "kurtosis"))
chk.table  # statistic, element, observed, rep_mean, rep_lower, rep_upper, p
```

`R` datasets with the observed lengths, time steps and covariate paths are simulated from the fitted model (an
MCMC fit draws its parameters from the posterior for each replicate), a statistic `T(y)` is evaluated on the
data and on every replicate, and `p = 2 min(P(T_rep ≥ T_obs), P(T_rep ≤ T_obs))`. Defaults:

* `acf_abs`: autocorrelation of `|y|` at lags 1–20 (volatility clustering, regime persistence in the scale);
* `sojourn_mean`: mean decoded segment length per regime (the replicates are decoded with the **fitted**
  parameters);
* `kurtosis`: excess kurtosis (tails the emission cannot produce).

Statistics are registry kind `ppc_statistic`: a function `fn(y, ctx) -> {element: value}` registered under a
name (`REGISTRY.register("ppc_statistic", "my_stat", fn, provenance="my_pkg")`) is available by name. `ppc` takes
one sequence at a time and no autoregressive emissions.
