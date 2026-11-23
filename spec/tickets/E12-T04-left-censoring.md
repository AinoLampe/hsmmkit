# E12-T04 — Left-censoring of the first segment in the explicit-duration engine (residual-life initial duration)

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | FR-DU04 | M9 | Part II §14.4; Part I §4.7.3, E4-T06a |

## Goal
Model sequences that start inside an ongoing segment, so the first segment's duration is not penalized as if it began at `t = 1`.

## Context (self-contained)
- Expanded engine: nothing to implement — `initial_state: stationary` (Part I FR-M13, E4-T02b) is the equilibrium over `(memory, regime, phase)` and is exactly left-censoring in the phase representation. `model.left_censored: true` with `engine: expanded` and `initial_state ≠ stationary` → warning "left_censored implies initial_state: stationary for the expanded engine" and the builder switches to `stationary`.
- Explicit engine, `left_censored: true`: for the **first** segment only, replace `ρ_j p_j(d)` in `α_d(j, d)` (segments ending at `t = d` that started at `t = 1`) by `ρ_j p_j^{res}(d)` with the residual-life pmf
```
p_j^{res}(d) = S_j(d − 1) / E[d_j],   S_j(d) = 1 − Σ_{u ≤ d} p_j(u),   E[d_j] = Σ_d d p_j(d)  (over 1..D_max, truncated pmf renormalized)
```
  and `ρ` the stationary regime occupancy `ρ_j ∝ ν_j E[d_j]`, `ν` the stationary distribution of the jump matrix `π` (eigenvector of `πᵀ` for eigenvalue 1, computed at `z = 0`), unless `initial_state: free` (then `ρ` is free and only `p^{res}` is used). The right-censored last segment (Part I) is unchanged; a sequence with a single segment is both left- and right-censored: its term is `ρ_j S^{res}_j(T − 1)` with `S^{res}` the survival of `p^{res}`.
- With entry-time duration covariates, `p^{res}` uses the covariates at `t = 1`.
- Batched engines (E12-T01b) apply this per sequence.
- Gate `model.left_censored` (explicit) removed.

## Creates
- edits in `engine/explicit_duration.py`, `model/duration/base.py` (`residual_log_pmf(theta, d_max, nominal_dt, z)` helper), `statespace/builders.py` (expanded switch)
- `tests/golden/test_e12_t04_left_censoring.py`; NumPy reference addition in `tests/reference/ref_hsmm.py` (`left_censored=True`)

## Acceptance criteria
1. Explicit-engine loglik with residual-life initial equals the expanded engine with `initial_state: stationary` on shifted-NB durations (no covariates, `D_max` mass loss < 1e-10) within Appendix D (rel ≤ 1e-6).
2. Golden vs the extended NumPy reference ≤ 1e-8 (random `N ∈ [2,4]`, `T ∈ [5, 120]`, `D_max ∈ [2, 20]`), including the single-segment case.
3. `p^{res}` sums to 1 ≤ 1e-12 for every Part I explicit family.
4. Gate removed; `left_censored: false` reproduces Part I bitwise.

## Tests
Golden vs reference, engine equivalence, pmf normalization.

## Out of scope
Left-censoring with segment memory of order ≥ 2 in the explicit engine (documented as unsupported → `ConfigError`).

## Connections
- Depends on: E10-T10
- Blocks: E13-T03a
- Relates to: E4-T06a, E4-T02b (stationary)
