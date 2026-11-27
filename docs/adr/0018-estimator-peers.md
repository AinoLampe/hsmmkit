# ADR-0018: EM, GEM, VB and MCMC are peers of MAP behind one `Estimator` interface

Status: accepted (E13-T02a)

## Context

Part I has one estimator: MAP by L-BFGS (ADR-0005), plus a GBM-only EM kept as a cross-check (E5-T06). Part II adds
generalized EM (explicit durations, covariate links), variational Bayes and MCMC. Each could be wired into
`Project.fit` by name, or all could sit behind one interface so that the estimation method is configuration
(`estimation.method`) and so that results are comparable.

## Decision

- An estimator is a registered object (registry kind `estimator`) with
  `fit(spec, data, cfg, seed, warm_start=None) -> FitResult`. `estimation.fit` dispatches on
  `cfg.estimation.method` (`map` by default, `em` from E13-T02a, then `gem`, `vb`, `mcmc`).
- `FitResult.estimator` names the method; `uparams` is always `ParamManager.flatten(theta)`, so bundles, `decode`,
  `forecast` and the backtest work for every estimator unchanged. `objective` is the quantity the estimator
  optimizes (`-loglik` for EM) and `restarts` holds one row per start (EM adds `loglik_trace`).
- **MAP stays the reference and the parity oracle**: an EM fit without priors from the same start must reach MAP's
  optimum (parameters rel 1e-4, log-likelihood rel 1e-6, Part II Appendix G).
- **Capability discovery, not family special-casing.** An estimator never branches on a family's name: it calls the
  optional protocol methods (`sufficient_stats`, `m_step`, `from_advance_prob`, ...) and treats `None` as "no closed
  form" (a `ConfigError` naming the ticket that adds the fallback, ECM in E13-T02b).
- The expanded-engine E-step (`engine.estep.expanded_estep`) and the explicit-engine E-step
  (`engine.estep_explicit.explicit_estep`) are engine functions shared by EM, GEM and VB.

## Consequences

- A new estimator needs registration and tests, not changes in `Project`.
- EM uses maximum likelihood for emissions even when `use_priors` is set (only the Dirichlet pseudo-counts of the
  jump rows enter); full MAP-EM needs a family `map_step`, none of which exists yet.
- Tied parameters, covariate links, irregular steps and families without a closed form are refused until ECM.
