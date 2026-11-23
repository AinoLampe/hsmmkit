# E15-T05 — (v1.x) Online EM implementation following ADR-0026

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ON05 | post-0.5.0 (v1.x) | Part II FR-ON05, D19; ADR-0026 (E13-T08) |

## Goal
Update model parameters online during streaming with the algorithm chosen in ADR-0026, without breaking the bitwise guarantees of the filter when updates are disabled.

## Context (self-contained)
- Implements the algorithm, supported components, step-size schedule (`γ_t = t^{−κ}`, defaults from the ADR), burn-in `t_min` and state layout decided in ADR-0026 (E13-T08). Baseline expectation (Cappé & Moulines 2009): running conditional statistics `ρ_t(s)` updated from the filter's backward kernel and the per-step complete-data statistics; parameters re-estimated with the families' closed-form `m_step` every `online.em.update_every` steps after burn-in.
- Config `online.em: {enabled: false, kappa: 0.6, burn_in: 500, update_every: 1, components: [emission, duration, transition]}`; components without closed-form M-steps stay fixed (warning listing them).
- `OnlineState` gains `ρ` statistics and the current `θ_t`; `spec_hash` unchanged (structure unchanged), state file records `online_em: true`.
- With `enabled: false` the filter is bitwise identical to E15-T01 (regression).
- Parameters are streamed out in the JSONL (`--emit params`, every `k` rows) and can be saved as a new bundle (`OnlineSession.snapshot_fit()`).

## Creates
- `src/hsmmkit/online/em.py`, edits in `online/{filter,state,session}.py`, CLI flags
- `tests/integration/test_e15_t05_online_em.py`

## Acceptance criteria
1. Tracks a synthetic linear drift of one state's scale (+50 % over 20 000 steps) with a lag (time to 90 % of the step change in a step-change variant) < 500 steps.
2. With a frozen step-size schedule and a stationary series, online EM statistics converge to offline EM statistics within 1 %.
3. `enabled: false` → bitwise E15-T01 behaviour.

## Tests
Integration on synthetic drifts; regression.

## Out of scope
Online structure changes (adding/removing states).

## Connections
- Depends on: E13-T08
- Blocks: —
- Relates to: E15-T01, E15-T04
