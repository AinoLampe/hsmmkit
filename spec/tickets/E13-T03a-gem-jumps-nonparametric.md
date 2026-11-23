# E13-T03a — `GEMEstimator` (explicit engine, Guédon): jump and nonparametric duration M-steps with censoring

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES02, FR-ES06 | M10 | Part II §14.6 (explicit engine), §14.4; Guédon 2003 |

## Goal
Provide generalized EM for the explicit-duration engine with closed-form jump-matrix and nonparametric duration updates, correctly accounting for censored segments.

## Context (self-contained)
`estimation/gem.py`, estimator `gem` (requires `engine: explicit`, else `ConfigError`). Iteration: E-step `ExplicitEStats` (E13-T01b); M-step:
- **Jumps** (no transition covariates, no duration-dependent jumps): `π_ij ∝ xi[i, j] + (α_dir − 1 if use_priors)` over allowed targets; with covariates or duration-dependent jumps → ECM block (reuse E13-T02b machinery with per-t/per-d counts `xi_by_d`).
- **Nonparametric durations** (`family: nonparametric` or per-state member): the censored last segment contributes survival mass, so the M-step is the Kaplan–Meier-type maximizer of
```
Q(p_j) = Σ_d eta[j, d] log p_j(d) + Σ_d eta_censored[j, d] log S_j(d − 1) (+ eta_first via residual life if left_censored)
```
  Without censoring: `p_j(d) ∝ eta[j, d]` (+ Dirichlet prior counts). With censoring: solve by the self-consistency (EM-within-M) iteration `p_j ← normalize(eta + Σ_{d' ≥ d} eta_censored[j, d'] · p_j(d) / S_j(d'−1))` until `‖Δp‖_1 < 1e-12` (max 200 inner iterations); left-censored contributions likewise with `p^{res}`.
- **Initial distribution**: `free` → `ρ_g ∝ rho_post[g]`.
- **Emissions**: closed-form via `family.m_step(sufficient_stats(…, gamma_regime, …))`; otherwise ECM (shared helper with E13-T02b).
- Parametric durations: raise `ConfigError("parametric durations under GEM available from E13-T03b")` in this ticket.
Convergence and monotonicity assertions as E13-T02a; tempering option `gem.tempering` (same schedule).
Gate `estimation.method: gem` removed for nonparametric/geometric durations (parametric remains gated until E13-T03b — implement as a gate predicate on the duration family).

## Creates
- `src/hsmmkit/estimation/gem.py`, `tests/integration/test_e13_t03a_gem.py`, `tests/recovery/test_e13_t03a_np_pmf.py` (`-m slow`)

## Acceptance criteria
1. `ℓ` non-decreasing (violations ≤ 1e-10) on synthetic data with nonparametric durations, with and without left-censoring.
2. Nonparametric pmf recovery at `T = 20 000` (3 states, `D_max = 60`): total variation ≤ 0.05 per state (Appendix G).
3. Censoring matters: on a fixture with a long right-censored final segment, the self-consistency M-step gives a higher `Q` than the naive `p ∝ eta` update (assert) and EM still increases `ℓ`.
4. Geometric durations under GEM match EM (expanded) within Appendix G (cross-engine parity).

## Tests
Integration, slow recovery, censoring unit test, cross-engine parity.

## Out of scope
Parametric duration M-steps (E13-T03b).

## Connections
- Depends on: E13-T01b, E12-T04
- Blocks: E13-T03b
- Relates to: E12-T05 (per-state, duration-dependent jumps)
