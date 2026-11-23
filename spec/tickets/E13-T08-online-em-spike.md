# E13-T08 — Online/recursive EM spike (Cappé & Moulines): design note, prototype, ADR-0026

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | FR-ON05 | M12 | Part II §13.1 FR-ON05, D19, §14.6, §14.11; Cappé & Moulines 2009 |

## Goal
Decide how online parameter updates should work on top of the `OnlineFilter` and the EM sufficient statistics, and prove feasibility with a prototype.

## Context (self-contained)
- Online EM recursion (Cappé & Moulines 2009): keep running expected sufficient statistics `Ŝ_t` and auxiliary function `ρ_t(s)` (expected statistics conditional on the current state), update with step size `γ_t = t^{−κ}`, `κ ∈ (0.5, 1]`:
```
ρ_{t}(s') = Σ_s φ_{t|t−1}(s | s') [ (1 − γ_t) ρ_{t−1}(s) + γ_t s(s, s', y_t) ],   φ = backward kernel from the filter
Ŝ_t = Σ_{s'} ρ_t(s') P(s_t = s' | y_{1:t});   θ_t = m_step(Ŝ_t)  (only after a burn-in t_min)
```
  Cost `O(S² · dim(S))` per step. Applies to components with closed-form M-steps (E13-T02a); others stay fixed or use a stochastic-gradient step (to be decided).
- Questions the ADR must answer: (1) algorithm (Cappé–Moulines vs block-wise/mini-batch EM vs SGD on the filtered log-likelihood); (2) which components can be updated online (expanded engine only; explicit engine excluded?); (3) step-size schedule and burn-in defaults; (4) interaction with `OnlineState` serialization (state grows by `ρ`); (5) bitwise reproducibility requirements; (6) how drift detection/alerts consume parameter changes.
- Prototype (`spikes/online_em.py`, not packaged): expanded engine, Gaussian diag emissions with closed-form stats, geometric durations; tracks a synthetic series whose regime-1 `σ` drifts linearly by +50 % over `T = 20 000`.

## Creates
- `docs/adr/0026-online-em.md` (accepted with go/no-go and chosen algorithm), `spikes/online_em.py`, `tests/spikes/test_e13_t08_online_em.py` (not in the default suite)

## Acceptance criteria
1. ADR-0026 accepted: algorithm choice, supported components, defaults, state layout diff for `OnlineState`, complexity table.
2. Prototype tracks the drifting `σ` with mean absolute tracking error < 10 % after burn-in, and with a frozen step-size schedule (`γ_t → 0` fast) reproduces offline EM statistics within 1 %.
3. No code merged to `src/` beyond the ADR (spike).

## Tests
Spike test only.

## Out of scope
Production implementation (E15-T05).

## Connections
- Depends on: E13-T02b, E15-T01
- Blocks: E15-T05
- Relates to: E15-T02
