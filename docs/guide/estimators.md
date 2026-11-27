# Choosing an estimator

`estimation.method` selects one of five estimators. They share the model, the data contract and the `FitResult`
(`params` on the constrained scale, `uparams = ParamManager.flatten(params)`, `loglik`, `restarts`, `diagnostics`), so a
model fitted by any of them can be decoded, forecast, backtested and saved in a bundle. How they agree is shown in the
generated [parity table](../generated/estimator_parity.md); the mathematics is in
[Estimation](../math/estimation.md).

## Decision table

| | **MAP** (`map`) | **EM** (`em`) | **GEM** (`gem`) | **VB** (`vb`) | **MCMC** (`mcmc`) |
|---|---|---|---|---|---|
| What it returns | one posterior mode | one maximum (of the likelihood, or the posterior with `use_priors`) | the same, explicit engine | posterior approximation `q(theta)`, posterior means | posterior draws (ArviZ), posterior means / SDs |
| Engine | expanded, explicit | expanded | explicit (`engine: explicit`) | expanded | expanded, explicit (NUTS); expanded (Gibbs) |
| Needs | differentiable log-density | `m_step` or an ECM-able block | nonparametric / geometric / parametric durations | conjugate blocks, others stay point estimates (hybrid) | NUTS: priors on every free parameter (`default_priors`); Gibbs: all blocks conjugate |
| Covariates | all links | ECM blocks | duration covariates (entry-time); no transition covariates | point blocks (hybrid) | NUTS: all links; Gibbs: none |
| Cost per iteration | `O(T S^2)` forward-backward + backward pass for the gradient (L-BFGS: tens to hundreds of iterations) | one E-step `O(T S^2)`, closed-form M-step | one E-step `O(T N D_max)`, closed form / Newton M-step | one E-step `O(T S^2)`, closed-form `q` update | NUTS: `2^depth` gradient evaluations per draw; Gibbs: one FFBS `O(T S^2)` per sweep |
| Memory | `O(T S)` | `O(T S)` (no `(T, S, S)`) | `O(T N)` plus `(N, D_max)` | `O(T S)` | `O(T S)` per chain plus the draws |
| Outputs | `std_errors` (Hessian, optional) | `loglik_trace` | `loglik_trace` | `elbo`, `elbo_trace`, `q_params`, `pruned_states`, posterior SDs | `posterior_samples`, `r_hat`, `ess_bulk`, `ess_tail`, divergences |
| Failure modes | local optima (use restarts), flat likelihoods | local optima, slow near saddles (tempering helps), ECM stalls | the same; duration pmf not identifiable with short sequences | ELBO looseness, local optima, mean-field underestimates posterior SDs | label switching (relabelled afterwards), divergences, poor mixing (low ESS), cost |
| Use it when | you want fast point estimates and the Hessian | the model is (near-)conjugate and you want the monotone, derivative-free route, or MAP is slow | durations are not geometric / negative binomial and you want the segmental model | you want uncertainty quickly, or to prune unused regimes | you need honest uncertainty, small samples, or to check VB |

Rules of thumb:

* Start with `map` (the default). Add `restarts` when the likelihood is multimodal; `em` with `tempering` is the
  alternative when MAP's optimizer struggles.
* `vb` reports `pruned_states` (regimes with expected occupancy below `estimation.vb.prune_below`): fit a model with
  too many regimes once and read which ones the data do not use.
* `mcmc` with `sampler: gibbs` is fast but only for fully conjugate models; `sampler: nuts` handles everything the
  MAP objective handles. Both need the `bayes` extra (`pip install 'hsmmkit[bayes]'`).
* The estimators agree where theory says they must: EM / GEM reach MAP's optimum without priors, VB's posterior
  means approach MAP for large `T` and flat priors, Gibbs and NUTS agree under identical priors. The parity table
  checks these on every nightly run.

## Worked example: the two-regime fixture

The fixture is a two-regime Gaussian with geometric sojourns (`T = 5000`, means 0 and 1.5, scales 0.5 and 1.2, mean
sojourn 25 steps). Compare the estimators on your own configuration in one command:

```bash
hsmmkit --out runs/compare compare config.yaml --estimators map,em,vb
```

This fits the configuration with each estimator (same data, same seed) and writes `compare.parquet` and
`compare.md`:

| column | meaning |
|---|---|
| `estimator` | `map`, `em`, `gem`, `vb` or `mcmc` |
| `loglik` | log-likelihood at the returned parameters |
| `objective` / `elbo` | what the estimator optimizes (`-log_joint`, `-loglik`, `-ELBO`); `elbo` is set for VB only |
| `n_params`, `bic`, `icl` | free parameters and the information criteria (`icl` for a single sequence) |
| `wall_s`, `n_iter`, `converged` | cost and convergence |
| `max_abs_dtheta_vs_map` | largest parameter difference to the MAP fit (constrained scale) |

From Python: `Project.compare_estimators(["map", "em", "vb"], write=True)`.

For the sampler: `estimation.method: mcmc` with `estimation.mcmc: {sampler: nuts, chains: 4, warmup: 1000, draws: 1000}`
returns `FitResult.posterior_samples` (an ArviZ object) and `diagnostics["mcmc"]` (`r_hat`, `ess_bulk`, `ess_tail`,
`divergences`, library versions). On the fixture both samplers recover the truth within two posterior standard
deviations with `R-hat <= 1.05` and bulk ESS above 200 (rows `nuts vs truth`, `gibbs vs truth` of the parity table).

## Not covered

* Selecting among estimators or structures by cross-validation: the grid search of the model-selection layer.
* Persisting posterior draws in a model bundle: only the posterior means are saved (`params`).
