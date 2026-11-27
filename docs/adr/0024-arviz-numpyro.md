# ADR-0024: ArviZ as the Bayesian output format, NumPyro as the only sampler backend in 0.x

Status: accepted (E13-T06a)

## Context

`estimation.method: mcmc` needs a sampler and a result format. NumPyro (JAX-native NUTS) and BlackJAX are the two
candidates; ArviZ is the de-facto format for posterior draws, diagnostics (R-hat, ESS) and plots.

## Decision

- The posterior is sampled by **NumPyro's NUTS** on a backend-agnostic target: `log_joint(u)` is exactly the
  negated MAP objective of `estimation.objective` (log-likelihood with the states marginalized by the forward
  algorithm, plus `log p(theta(u))` and the log-Jacobian of the bijectors). No NumPyro model object is built, so
  another backend (BlackJAX) can be added without touching the model code. **NumPyro is the only backend in 0.x.**
- Results are **ArviZ** objects (`FitResult.posterior_samples`): groups `posterior` (one variable per
  `ParamSpec` name; dims `(chain, draw, regime, series, ...)`), `sample_stats` (divergences, tree depth, energy)
  and `log_likelihood` (per sequence). ArviZ 0.x (`InferenceData`) and 1.x (`DataTree`) are both supported by
  `hsmmkit.interop.arviz`.
- Both libraries live behind the `bayes` extra (`mcmc` is a deprecated alias until 0.5.0) and are imported lazily
  with `optional_import(..., extra="bayes")`: selecting `method: mcmc` without them raises an `ImportError` that
  names the extra. This adds no runtime dependency.
- Chains start from the MAP optimum plus Gaussian jitter, with keys `key_for(seed, PURPOSE_MCMC, chain)`: the same
  seed, backend and library versions give identical draws (NFR-17); the versions are recorded in
  `diagnostics["mcmc"]["versions"]`.
- Improper posteriors are refused: free unbounded parameters without a prior need `default_priors: true`
  (weakly informative defaults per domain, `docs/math/estimation.md`) or explicit priors.

## Consequences

- The sampler is exact for the marginal posterior of the parameters; state paths are drawn afterwards by FFBS
  (E13-T04) if needed. Label switching without an identifiability guard is only warned about; relabelling and the
  blocked Gibbs sampler are E13-T06b.
- `FitResult.posterior_samples` is not persisted in the model bundle.
