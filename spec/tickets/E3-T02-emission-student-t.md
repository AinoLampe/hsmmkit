# E3-T02 — Student-t emission family

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-M02 | M1 | §4.2 Student-t |

## Goal
Add a heavy-tailed emission with per-state or tied degrees of freedom, same covariate links as GBM.

## Context (self-contained)
`y_t | i ~ m_i(t) Δ_t + s_i(t) √Δ_t · t_{ν_i}`, `ν_i > 2`. Links: `m_i(t) = m_i + γ_iᵀ z`, `log s_i(t) = log s_i + δ_iᵀ z`. Specs: `emission.m (N, M)` real, `emission.s (N, M)` positive (increasing constraint optional), `emission.nu (N,)` domain `nu_gt_2` (mode `free|tied|fixed` from config; `tied` uses one tie group `"nu"`), coefficient specs as in E3-T01. Log-density via `dists.student_t_logpdf(y, loc, scale, nu)` (E1-T04). Note: `s_i` is a scale, not the sd (`sd = s √(ν/(ν−2))`); document.

## Creates
- `src/regimekit/model/emission/student_t.py`
- `tests/unit/test_e3_t02_student_t.py`

## Acceptance criteria
1. `log_prob` matches `scipy.stats.t.logpdf` ≤ 1e-10.
2. With `ν = 1e6` (fixed), density equals GBM with `μ − σ²/2 → m`, `σ → s` within 1e-6 on a random fixture.
3. `nu` tied → one `u` coordinate (via `ParamManager.n_free`).
4. Sampler moments within 3 SE (use `ν = 8`).

## Out of scope
Skew-t, GARCH.

## Connections
- Depends on: E3-T01
- Blocks: —
- Relates to: E5-T05a (optional scenario), E9-T01
