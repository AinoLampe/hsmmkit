# E9-T03 — MS-GARCH emission spike using carry hooks (Haas et al. 2004)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | §5.10 | post-0.1.0 | §4.2 (interface contract: carry), §5.10, Haas/Mittnik/Paolella 2004 |

## Goal
Verify that the engine's scan can thread recurrent emission state (per-regime GARCH variances) and that a 1-state GARCH reproduces a reference package.

## Context (self-contained)
- Haas et al. (2004): each regime `i` carries its own variance recursion `h_{i,t} = ω_i + α_i ε_{t−1}² + β_i h_{i,t−1}` with `ε_{t−1} = y_{t−1} − m_i Δ` (regime-specific), run in parallel for all regimes; emission `y_t | i ~ N(m_i Δ, h_{i,t})`. `native_time="discrete"`, so parameters are per step (document the deviation from annualization: `ω, α, β` are dimensionless/per-step, `m_i` per year).
- Threading: `init_carry() -> (N,) h_0` (unconditional variance); the expanded engine's scan carries `(α_{t−1}, carry)` and calls `step_carry(carry, y_t, θ) -> carry'` and `log_prob_step(y_t, dt_t, θ, carry)`; E4-T03a's `log_emis` precomputation is replaced by an in-scan emission when `family.init_carry() is not None` (branch on config, not data).
- Reference: 1-state GARCH(1,1) log-likelihood vs the `arch` package (dev dependency for the spike only, or a NumPy reference in `tests/reference/ref_garch.py` if `arch` is not added) within 1e-6 on a synthetic series.

## Creates
- `src/regimekit/model/emission/msgarch.py` (experimental), engine carry branch in `expanded_hmm.py`, `tests/spikes/test_e9_t03_msgarch.py`, ADR-0012 (in-scan emissions; annualization exception)

## Acceptance criteria
1. Carry threading works inside `lax.scan`, differentiable (finite gradient), and does not change GBM results (`init_carry() is None` path bitwise unchanged).
2. 1-state GARCH(1,1) loglik matches the reference within 1e-6.
3. 2-state MS-GARCH fits on synthetic data without numerical failure (`restarts=2`).

## Out of scope
Production support, forecasting with GARCH carry (design note only).

## Connections
- Depends on: E3-T01, E4-T03b
- Blocks: —
- Relates to: E4-T03a (emission precompute branch)
