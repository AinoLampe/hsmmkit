# Model selection and diagnostics

## Information criteria

With `\ell` the maximized log-likelihood, `p` the number of free entries of `u` and `T` the number of observed
steps:

$$
\text{AIC} = 2p - 2\ell, \qquad \text{BIC} = p \log T - 2\ell, \qquad \text{ICL} = \text{BIC} + 2 H(\gamma)
$$

where `H(\gamma) = -\sum_t \sum_s \gamma_{t,s} \log \gamma_{t,s}` is the entropy of the smoothed posterior over the
expanded states `s`: ICL penalizes models whose regimes overlap. All three are computed by `decode` and by the
`aic`, `bic`, `icl` backtest metrics (lower is better). The number of regimes and the phase counts `r_i` are
hyperparameters, chosen by these criteria and by the out-of-sample scores, not by gradient.

## Out-of-sample criteria

The per-horizon log score and CRPS of the walk-forward backtest are the primary selection criteria (see
[Evaluation](evaluation.md)); `regimekit compare` tabulates them for several configurations on one data
snapshot.

## Diagnostics emitted with every fit

| Diagnostic | Meaning | Automatic warning |
|---|---|---|
| `regime_occupancy` | mean smoothed probability per regime | a regime below 2 %: misspecified fixed parameters or too many states |
| `posterior_mean_duration`, `_ratio` | mean Viterbi segment length per regime, and its ratio to the fitted mean sojourn (in steps) | none (reported) |
| `pit_hist`, `pit_ks_statistic`, `pit_ks_pvalue` | histogram and KS test of the one-step probability integral transform, uniform when calibrated | none (reported) |
| `ljung_box` | autocorrelation of the normal-score PIT residuals at lags 10 and 20 | none (reported) |
| `ordering_check`, `label_switching_risk` | the `sigma` ordering holds; risk if it does not and no identifiability guard is configured | none (reported) |
| `restart_dispersion` | relative gap between the two best restart objectives | gap above 1 %: likely multimodality |
| `timings` | wall time in total and per restart | none |

Hessian standard errors (`estimation.std_errors`, in `u`, mapped to `\theta` by the delta method) are separate
from the diagnostics; where the Hessian is not positive definite no standard errors are reported.

## Symbols and code names

| Math | Code |
|---|---|
| `\ell` | `FitResult.loglik` |
| `p` | `ParamManager.n_free` |
| `\gamma` | smoothed expanded posterior `(T, S)` (`Posterior`) |
| AIC, BIC, ICL | `engine.criteria.information_criteria`; exposed as `PosteriorResult.aic`, `.bic`, `.icl` (`inference.results`) |
| diagnostics | `regimekit.estimation.report.diagnose` |
