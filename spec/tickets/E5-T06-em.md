# E5-T06 — (v1.x) Generalized Baum–Welch (EM) for GBM emissions without covariates

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | D3, ADR-0005 | post-0.1.0 | §4.8 EM, §5.5 (`sufficient_stats`), §5.10 |

## Goal
Offer an EM estimator as a cross-check to MAP for the covariate-free GBM/geometric-or-NB model.

## Context (self-contained)
- E-step: `forward_backward` (expanded engine) → smoothed `γ_t(s)` and pairwise `ξ_t(s, s')` (compute via one extra scan: `ξ_t ∝ α_{t−1}(s) A_t(s,s') ℓ_t(s') β_t(s')`).
- M-step (closed form for GBM): per regime `i` with weights `w_t = Σ_{s: regime_of[s]=i} γ_t(s)`: `σ_i² = Σ w_t (y_t − m_i Δ_t)² / Σ w_t Δ_t` with `m_i = μ_i − σ_i²/2` solved jointly (iterate twice); `μ_i` from weighted mean of `y_t/Δ_t + σ_i²/2`. Advance probability `a_i` from expected advance/exit counts vs stay counts (`ξ` sums over the corresponding structural entries) → `λ_i = −log(1 − a_i)/Δ`; jump matrix rows from expected exit counts (normalized, zero diagonal). `sufficient_stats(y, dt, gamma)` on `GBMFamily` returns the weighted sums.
- Convergence on relative loglik change (`tol_f`), `max_iter`. Monotone loglik asserted in debug mode.

## Creates
- `src/regimekit/estimation/em.py`; `GBMFamily.sufficient_stats`
- `tests/integration/test_e5_t06_em.py`

## Acceptance criteria
1. Loglik is non-decreasing across EM iterations (assert on 3 synthetic datasets).
2. EM optimum agrees with MAP (`use_priors=False`) within 1e-4 relative loglik and 1e-2 relative in σ on a 2-state synthetic set.
3. Raises `ConfigError` when covariates or non-GBM families are configured.

## Out of scope
EM for covariates, Student-t, explicit engine.

## Connections
- Depends on: E5-T03
- Blocks: —
- Relates to: E3-T01 (hook)
