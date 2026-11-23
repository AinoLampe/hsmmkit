# E13-T02b — ECM for components without closed form, tempered EM, restarts and parity suite

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES01, FR-ES07, NFR-13 | M10 | Part II §14.6 (ECM, tempering), Appendix G; Ueda & Nakano 1998 |

## Goal
Make EM usable for every model the MAP estimator can fit (covariates, irregular grids, non-conjugate families) and less sensitive to local optima.

## Context (self-contained)
**ECM** (expectation conditional maximization): components whose `m_step` is `None` (Student-t, Merton/OU via plug-ins, covariate links, `bounded`/`prior` modes without closed form, irregular `Δ`, duration/transition covariates) are updated by maximizing the expected complete-data log-likelihood with the other blocks fixed:
```
Q_e(θ_e)  = Σ_t Σ_i γ_ti log f_i(y_t | θ_e, z_t)                         (emission block)
Q_d(θ_d)  = Σ_t Σ_k [ stay_tk log(1 − a_k(t)) + advance_tk log a_k(t) ]    (duration block; needs per-t counts → E13-T01a option keep_per_t=True)
Q_π(θ_π)  = Σ_t Σ_r Σ_j jump_trj log π_rj(t)                               (transition block; per-t counts when covariates)
```
plus `log p(θ)` when `use_priors`. Each block: a few L-BFGS iterations (`ecm.inner_iter`, default 5) on its slice of `u` (Optax, warm-started); ECM guarantees monotone `ℓ` only if each conditional step does not decrease `Q` — enforce by accepting a step only if `Q` increases (else keep old values and log). Per-t counts: extend `expanded_estep(…, keep_per_t: bool)` to return `(T, R_rate)` / `(T, R, N)` arrays when required (memory documented).
**Tempered EM** (FR-ES07, deterministic annealing): `em.tempering: {enabled, beta0 = 0.3, steps = 10}` — geometric schedule `β_k = beta0^{1 − k/steps}` for the first `steps` iterations, then `β = 1`; E-step via `beta_temper` (E13-T01a). Monotonicity is asserted only once `β = 1`.
**Restarts**: as MAP (keys `key_for(seed, restart)`), plus `em.warm_start_from: map | none` (run EM from the MAP optimum — used by the parity suite).
**Parity suite** (`tests/parity/test_e13_t02b_em_map.py`): table of configurations (Gaussian diag/full, Student-t, Poisson with covariates, Categorical, MS-AR(1), mixture-2; geometric/NB durations; topology full/LTR) × check: from the same start, EM converges to the MAP optimum within Appendix G (`use_priors=false`); written as a generated table (`docs/generated/em_parity.md`) consumed by E13-T07.

## Creates
- `estimation/em.py` (ECM, tempering), `engine/estep.py` (`keep_per_t`), `estimation/ecm.py`
- `tests/integration/test_e13_t02b_ecm.py`, `tests/parity/test_e13_t02b_em_map.py`, `tests/integration/test_e13_t02b_tempering.py`

## Acceptance criteria
1. ECM runs on every Part I/E11 family and on covariate models; `ℓ` monotone after tempering (violations ≤ 1e-10).
2. Parity table: every row within Appendix G (`θ` rel ≤ 1e-4, loglik rel ≤ 1e-6) or explicitly marked "not identifiable" with a reason.
3. Tempered EM reaches the MAP global optimum on a multimodal fixture (3-state Gaussian with two near-equal local optima, documented) in ≥ 4/5 seeds where plain EM does so in ≤ 2/5.
4. `FitResult.diagnostics["em"]` lists per component whether closed-form or ECM was used.

## Tests
Integration, parity suite, tempering experiment (`-m slow` for the multimodal case).

## Out of scope
Explicit engine GEM (E13-T03a/b); VB (E13-T05a/b).

## Connections
- Depends on: E13-T02a
- Blocks: E13-T05b, E13-T07, E13-T08, E14-T06
- Relates to: E11-T03, E11-T08 (`m_step_partial`)
