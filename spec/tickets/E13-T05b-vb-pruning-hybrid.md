# E13-T05b — VB: automatic state pruning, VB-EM hybrid for non-conjugate components, fit-report integration, parity

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES04, NFR-13 | M10 | Part II §14.8 (pruning, hybrid), §19 #13, Appendix G |

## Goal
Make VB practical on real models: report and prune unused states, and treat non-conjugate components with point estimates instead of failing.

## Context (self-contained)
- **Pruning**: after convergence, state `i` is *pruned* if `E_q[occupancy_i] = Σ_b Σ_t γ_ti / Σ_b T_b < ε` with `ε = estimation.vb.prune_below` (default 0.01). Pruned states are reported in `FitResult.pruned_states` and `diagnostics["vb"]["occupancy"]`; they are **not** removed from the model (structure is static) — documented; `decode` marks them; a warning suggests refitting with smaller `N`.
- **VB-EM hybrid**: blocks without `conjugate_prior()` (covariate links, Student-t `ν`, parametric explicit durations, OU/Merton plug-ins, duration with irregular `Δ`) keep point estimates updated by the ECM conditional step of E13-T02b using the current `q`-expected statistics; their contribution to the ELBO is the expected complete-data log-likelihood at the point estimate plus `log p(θ̂)` (documented: the "ELBO" becomes a lower bound of a partially-collapsed objective; monotonicity still required). Config validation (E10-T10) already warns; here the fit report lists `diagnostics["vb"]["treatment"] = {component: "conjugate" | "point"}`.
- **Fit report**: ELBO trace, treatment table, pruned states, posterior SDs for conjugate blocks (`std_errors` field filled from `q`).
- **Parity rows** (`tests/parity/`): VB posterior means vs MAP within 2 % on `T ≥ 20 000` for Gaussian diag/full, Poisson, Categorical (flat priors) — feeds E13-T07.
Gate predicate for hybrid VB removed.

## Creates
- `estimation/vb.py` (pruning, hybrid), `estimation/report.py` (VB section)
- `tests/integration/test_e13_t05b_pruning.py`, `tests/integration/test_e13_t05b_hybrid.py`, parity rows `vb_map`

## Acceptance criteria
1. A 4-state VB fit on 3-state synthetic data (Gaussian, `T = 5000`) prunes exactly one state in ≥ 4/5 seeds.
2. Hybrid on a model with transition covariates and Student-t emissions runs, ELBO non-decreasing (≤ 1e-8), treatment table correct.
3. Parity rows within Appendix G (2 %).
4. `decode` on a VB fit marks pruned states in `PosteriorResult` metadata.

## Tests
Integration and parity as above.

## Out of scope
VB for parametric explicit durations (point estimates only, Part II §19 #13); MCMC (E13-T06a/b).

## Connections
- Depends on: E13-T05a, E13-T02b
- Blocks: E13-T07, E14-T06
- Relates to: E5-T04 (fit report)
