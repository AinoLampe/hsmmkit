# E13-T01a — Expanded-engine E-step accumulators computed inside the scan

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-ES01 | M10 | Part II §14.6 (E-step), §16.3 item 10; Part I §4.7.2, E4-T03a/b, E5-T06 |

## Goal
Compute all expected counts and sufficient statistics that EM and VB need from one forward-backward pass, without ever materializing pairwise posteriors over `(T, S, S)`.

## Context (self-contained)
`engine/estep.py`:
```python
@dataclass(frozen=True)
class ExpandedEStats:                          # pytree; all sums over t (and over sequences b for batches)
    loglik: Array                              # (B,)
    gamma_regime: Array                        # (B, T_max, N)  regime marginals (kept: needed by family sufficient_stats)
    gamma1: Array                              # (G, S) initial-state posterior summed per initial group
    stay: Array                                # (R_rate,)   Σ_t Σ_{s: rate_index_of[s]=k} ξ_t(s → s)
    advance: Array                             # (R_rate,)   Σ ξ_t(s → s+1 phase)  + exits from the last phase
    occupancy: Array                           # (R_rate,)   Σ_t γ_t(s) over states of rate row k (t < T)
    jump: Array                                # (R, N)      Σ_t ξ_t(exit from row r → target j)
def expanded_estep(engine, params, spec, space, pb: PaddedBatch, *, beta_temper: float = 1.0) -> ExpandedEStats
```
- Pairwise posterior for one step: `ξ_t(s, s') = exp(α_{t−1}(s) + log A_t(s, s') + ℓ_t(s') + β_t(s') − ℓ_b)`. Never form `(S, S)` for all `t` at once; inside a scan over `t` form the `(S, S)` matrix of the current step only, reduce it immediately with the static index arrays of E4-T02b (`next_phase_state`, `is_last_phase`, `exit_targets`, `row_of_state`, `rate_index_of`) into `stay`, `advance`, `jump` contributions, and add to the carry. Memory `O(S² + N R)` per sequence plus the stored `γ` regime marginals `(T, N)`.
- Implement as a third scan after forward/backward (reuses `log_alpha`, `log_beta` from `forward_backward`), vmapped over the batch; padding steps contribute 0 (`valid` mask).
- `beta_temper` (`β ∈ (0, 1]`, tempered EM, E13-T02b): E-step on `β ℓ_t(s)` — implemented by running forward-backward with scaled emissions; `loglik` reported at `β = 1` separately.
- Constraints (E12-T06) are applied through the engine (masked `ℓ`, `A_t`).
- Family sufficient statistics are **not** computed here (families consume `gamma_regime` via `sufficient_stats(y, dt, gamma, z)`); for very long `T`, `ExpandedEStats.gamma_regime` may be replaced by streaming accumulation in a later optimization (documented).
- NumPy reference: `tests/reference/ref_estep.py` with explicit `ξ` tensors on small instances.

## Creates
- `src/hsmmkit/engine/estep.py`, `tests/reference/ref_estep.py`
- `tests/golden/test_e13_t01a_estep.py`, `tests/unit/test_e13_t01a_memory.py`

## Acceptance criteria
1. Accumulators vs the NumPy reference ≤ 1e-8 (random `N ∈ [2,4]`, `r_i ∈ [1,3]`, order 1 and 2, covariates, topology masks, batches with padding).
2. Consistency: `Σ_s γ_t(s) = 1`; `stay + advance = occupancy` per rate row ≤ 1e-10; `Σ_j jump[r, j]` equals the expected exits from row `r`.
3. Memory test: peak device memory for `T = 20 000, S = 60` stays below `5 × (T·N + S²)` floats × 8 B (measured via `jax` compiled memory analysis); no array of shape `(T, S, S)` appears in the jaxpr (inspected programmatically).
4. Tempered E-step with `β = 1` equals the untempered E-step bitwise.

## Tests
Golden vs reference, invariants, memory/jaxpr inspection.

## Out of scope
M-steps (E13-T02a); explicit engine (E13-T01b).

## Connections
- Depends on: E12-T01b
- Blocks: E13-T02a, E13-T05a
- Relates to: E5-T06 (superseded), E4-T02b
