# E13-T01b — Explicit-engine E-step: segment posteriors `η_t(j, d)` and jump posteriors `ξ_t(i, j)`

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES02 | M10 | Part II §14.6 (Guédon), §14.4; Part I §4.7.3, E4-T06a/b |

## Goal
Produce the expected segment and jump statistics of the explicit-duration engine that generalized EM needs, including censored first/last segments.

## Context (self-contained)
`engine/estep_explicit.py`:
```python
@dataclass(frozen=True)
class ExplicitEStats:
    loglik: Array          # (B,)
    eta: Array             # (N, D_max)   Σ_t η_t(j, d): expected number of COMPLETE segments of regime j with duration d
    eta_censored: Array    # (N, D_max)   expected right-censored last segments in progress with elapsed d (survival mass)
    eta_first: Array       # (N, D_max)   expected left-censored first segments (only if left_censored)
    eta_entry: Array | None   # (T, N, D_max) per segment start — only when entry-time duration covariates are used (else None)
    xi: Array              # (N, N)       Σ_t ξ_t(i, j) expected jumps i → j
    xi_by_d: Array | None  # (N, N, D_max) when duration-dependent jumps are on (E12-T05)
    gamma_regime: Array    # (B, T_max, N)
    rho_post: Array        # (G, N) first-segment regime posterior per initial group
```
Definitions (Guédon 2003, Yu 2010), with `α_t(j, d)` = log P(segment of `j` ends at `t` with duration `d`, `y_{1:t}`) and `β_t(j)` = log P(`y_{t+1:T}` | a segment of `j` ended at `t`):
```
η_t(j, d) = exp( α_t(j, d) + β_t(j) − ℓ ),            complete segments (t < T)
censored: segments covering T with elapsed d use survival S_j(d−1) in α (Part I right-censoring) → eta_censored
ξ_t(i, j) = exp( A_t(i) + log π_ij + log Σ_d p_j(d) e^{L[t+d,j] − L[t,j]} e^{β_{t+d}(j)} − ℓ )   (computed via the same entry buffer as the backward pass)
```
- Accumulate `eta` per `(j, d)` summed over `t` inside the backward scan (no `(T, N, D_max)` storage unless entry-time covariates require `eta_entry`; then chunking rules of E4-T06b apply).
- Padding/batching per E12-T01b; constraints per E12-T06; left-censoring per E12-T04 (`eta_first` uses `p^{res}`).
- Extend `tests/reference/ref_hsmm.py` with brute-force segment enumeration for `T ≤ 8`.

## Creates
- `src/hsmmkit/engine/estep_explicit.py`, reference extension
- `tests/golden/test_e13_t01b_segments.py`

## Acceptance criteria
1. `eta`, `eta_censored`, `xi` vs brute-force enumeration ≤ 1e-10 for `T ≤ 8, N ≤ 3, D_max ≤ 4`, and vs the NumPy forward-backward reference ≤ 1e-8 on random instances (`T ≤ 120`).
2. Mass balance: `Σ_{j,d} d·(eta + eta_censored + eta_first)[j, d]` equals `T` per sequence ≤ 1e-8; `Σ_j xi[i, j]` equals the expected number of completed segments of `i`.
3. `gamma_regime` equals E4-T06a smoothed marginals ≤ 1e-12.
4. No `(T, N, D_max)` allocation when `eta_entry` is not needed (jaxpr inspection).

## Tests
Brute-force and reference golden tests, mass balance, jaxpr inspection.

## Out of scope
M-steps (E13-T03a/b).

## Connections
- Depends on: E12-T01b
- Blocks: E13-T03a
- Relates to: E12-T04, E12-T05, E13-T04 (backward sampling uses `α_t(j, d)`)
