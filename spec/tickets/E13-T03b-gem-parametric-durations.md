# E13-T03b — GEM: parametric duration M-steps (weighted MLE), per-state families, parity with MAP

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES02, NFR-13 | M10 | Part II §14.6 (parametric durations by weighted MLE), Appendix G; Part I §4.3, E3-T04 |

## Goal
Complete GEM for all explicit duration families, including per-state mixes, and verify parity with MAP.

## Context (self-contained)
- **Weighted MLE** for each state's parametric family (`poisson`, `gamma`, `weibull`, `lognormal`, discretized on the grid as in Part I §4.3; shifted-NB via its explicit pmf):
```
max_φ  Σ_d eta[j, d] log p_j(d; φ) + Σ_d eta_censored[j, d] log S_j(d − 1; φ) (+ residual-life term if left_censored) + log p(φ)
```
  solved by Newton's method on the unconstrained 1–2 parameters (`jax.hessian` of the objective; damping `λ_LM = 1e-6`, step halving until increase; ≤ 30 iterations; fallback L-BFGS). Families may provide `m_step(stats)` (`DurationFamily` optional method, Part II §15.3) — `sufficient_stats(eta)` returns `(eta, eta_censored, eta_first)`; the default implementation is the Newton routine above.
- **Entry-time duration covariates**: Newton on `(φ, c_j)` using `eta_entry` (E13-T01b) with the covariate at segment start.
- **Per-state durations** (E12-T05): each state's family updated independently.
- **Emissions**: shared with E13-T02a/b (closed-form or ECM).
- Gate predicate for parametric durations under `gem` removed.
- **Parity**: from the MAP optimum, GEM stays (θ rel ≤ 1e-4); from the same random start, both reach the same optimum on gamma durations (Appendix G).

## Creates
- `estimation/gem.py` (parametric path), `model/duration/base.py` (default `m_step` via Newton), `estimation/newton.py`
- `tests/integration/test_e13_t03b_gem_parametric.py`, parity rows added to `tests/parity/` (`gem_map`)

## Acceptance criteria
1. Monotone `ℓ` for gamma, Weibull, lognormal, Poisson durations, with and without entry-time covariates (violations ≤ 1e-10).
2. Parity with MAP (`use_priors=false`) for gamma durations: `θ` rel ≤ 1e-4, loglik rel ≤ 1e-6.
3. A per-state model (gamma / gamma / geometric / nonparametric, explicit engine, diagonal Gaussian emissions, no labels) fits with `method: gem` on synthetic data and recovers each state's mean sojourn within 25 % (≥ 4/5 seeds, `-m slow`). (The full Appendix H run, with labels and full covariance, is the tutorial of E16-T05.)
4. Newton step falls back to L-BFGS on a deliberately ill-conditioned start and still increases `Q`.

## Tests
Integration, parity, robustness test of the Newton routine.

## Out of scope
VB for parametric durations (point estimates, E13-T05b).

## Connections
- Depends on: E13-T03a, E13-T02a, E12-T05
- Blocks: E13-T07, E14-T06, E16-T05
- Relates to: E3-T04, E12-T05
