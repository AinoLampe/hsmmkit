# E13-T06a — `MCMCEstimator` (`bayes` extra): NUTS on `log_joint(u)`, ArviZ `InferenceData`, ADR-0024

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES05, FR-ES06, FR-IO03, NFR-17 | M10 | Part II §14.9 (a), §15.5 (`bayes`), §19 #15; Part I §4.8 (objective) |

## Goal
Sample the parameter posterior of any model with NUTS, states marginalized by the forward algorithm, and return results as ArviZ `InferenceData`.

## Context (self-contained)
- `estimation/mcmc.py`, estimator `mcmc` with `estimation.mcmc: {sampler: nuts, chains: 4, warmup: 1000, draws: 1000, target_accept: 0.8, max_tree_depth: 10}`; NumPyro and ArviZ imported via `optional_import(..., extra="bayes")` (E10-T11).
- Potential: `potential_fn(u) = −log_joint(u)`, `log_joint(u) = Σ_b ℓ_b(θ(u)) + log p(θ(u)) + log|det ∂θ/∂u|` — exactly the negated MAP objective of E5-T01/E12-T02 (`use_priors` must be true; improper flat priors are rejected with `ConfigError` listing the parameters without a prior — default weakly informative priors are offered by `estimation.mcmc.default_priors: true`, documented per domain).
- NumPyro: `numpyro.infer.MCMC(NUTS(potential_fn=…), num_warmup, num_samples, num_chains, chain_method="sequential" | "vectorized")`, init from MAP optimum + jitter (`init_params`) per chain, keys `key_for(seed, "mcmc", chain)`.
- Output: draws `u` → `θ` (vmapped `manager.unflatten`) → `InferenceData` with groups `posterior` (named parameters, dims `(chain, draw, state, series, …)` from `ParamSpec` shapes and names), `sample_stats` (divergences, tree depth, energy), `log_likelihood` (per sequence) and `observed_data` (optional). `FitResult(estimator="mcmc", params = posterior mean on the constrained scale, loglik at the posterior mean, posterior_samples = InferenceData, diagnostics["mcmc"] = {r_hat, ess_bulk, ess_tail, divergences})`.
- Label switching: ordering constraints of Part I are honoured through `u`; without them, post-hoc relabelling is E13-T06b (here: warning).
- Reproducibility (NFR-17): same seed/backend/NumPyro version → identical draws; versions recorded in diagnostics.
- **ADR-0024** (`docs/adr/0024-arviz-numpyro.md`, accepted): ArviZ as Bayesian output format; NumPyro only backend in 0.x; `log_joint(u)` backend-agnostic.
Gate `estimation.method: mcmc` (sampler `nuts`) removed.

## Creates
- `src/hsmmkit/estimation/mcmc.py`, `src/hsmmkit/interop/arviz.py` (`to_inference_data`)
- `tests/integration/test_e13_t06a_nuts.py` (marker `bayes`, `-m slow` for the full fixture)
- `docs/adr/0024-arviz-numpyro.md`

## Acceptance criteria
1. Appendix E two-regime fixture (Gaussian form, `T = 5000`, 4 chains × 1000 draws): posterior means within 2 posterior SD of truth, `R̂ ≤ 1.05`, bulk ESS ≥ 200 (Appendix G).
2. Seeded runs reproducible bitwise on CPU with the same NumPyro version.
3. `InferenceData` dims/coords named after parameter specs; `az.summary` runs.
4. Without the `bayes` extra, selecting `method: mcmc` raises the actionable `ImportError`.

## Tests
Integration (slow), reproducibility, extra-missing behaviour.

## Out of scope
Blocked Gibbs, relabelling (E13-T06b); BlackJAX (not in 0.x).

## Connections
- Depends on: E12-T02, E10-T11
- Blocks: E13-T06b
- Relates to: E5-T01, E16-T03 (trace plots)
