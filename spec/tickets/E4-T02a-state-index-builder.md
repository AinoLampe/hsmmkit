# E4-T02a — `StateIndex` construction: `build_expanded(spec)` for phases × memory (order k)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M10, FR-X01, D2, D5 | M2 | §4.5, §4.7.1, Appendix B (`S` formula) |

## Goal
Enumerate the expanded state space `S = (memory, regime, phase)` and its index arrays and collapse matrix from a `ModelSpec`.

## Context (self-contained)
- Memory tuples of order `k`: sequences of the previous `k−1` regimes with consecutive entries distinct and the last entry ≠ current regime. Order 1: memory is empty (one tuple). Order 2: memory `(i,)`, `i ≠` current regime. General `k`: `(N−1)^{k−1}` tuples per current regime after the "no repeats" rule. `k > 2` → `warnings.warn` on parameter count (config already warns; builder warns again with the actual `S`).
- Expanded states ordered lexicographically `(memory tuple, regime, phase)`; `regime_of (S,)`, `phase_of (S,)` (1..`r_i`), `memory_of (S, k−1)`, `collapse C (S, N)`. `S = Σ_{admissible (m, i)} r_i`; order 1 → `Σ_i r_i`; order 2 → `(N−1) Σ_i r_i`.
- Also produce `MemoryLayout` rows (E3-T05) consistent with the tuple enumeration: `row_of_state (S,)` = index of the jump-matrix row `(memory, regime)` that governs exits from `s`; `entry_state_of (R, N)` = expanded state index of `(shift(memory, current), target, phase 1)` for each row and target (`−1` where target = current), where `shift(m, i) = (m[1:], i)` for order ≥ 2 and empty for order 1.
- `duration_conditioning="pair"`: phases/rates indexed by `(memory[-1], regime)` instead of `regime`: expose `rate_index_of (S,)` mapping each state to the row of the duration parameter array (`N` rows for `current`, `N(N−1)` for `pair`); duration families receive `n_rates` accordingly (E3-T03 `param_specs(n_states=n_rates)`).
- `StateSpace(eqx.Module)`: `index: StateIndex`, `layout: MemoryLayout`, `n_expanded`, `rate_index_of`, `transition_fn` (filled by E4-T02b; here a placeholder raising `NotImplementedError`).
- Initial distribution helper `initial_log_rho(spec, space, theta) -> (S,)`: `uniform` over states with `phase == 1` (mass 0 elsewhere → `−inf`), `stationary` (stationary distribution of the mean `A` at `z=0`, computed by eigenvector or power iteration — E4-T02b provides `A`; here implement `uniform` and `free` (spec `initial.logits (S_phase1,)` domain `simplex_ref`), and leave `stationary` to E4-T02b).

## Creates
- `src/regimekit/statespace/builders.py`, extensions to `statespace/index.py`
- `tests/unit/test_e4_t02a_builder.py`, `tests/property/test_e4_t02a_builder.py`
- `docs/math/statespace.md`

## Acceptance criteria
1. `S` equals `Σ r_i` (order 1) and `(N−1) Σ r_i` (order 2) for random `N ∈ [2,4]`, `r_i ∈ [1,3]`.
2. Order-2 memory tuples never contain `memory[-1] == regime`; order-3 tuples have no adjacent repeats.
3. `C` rows are one-hot; `regime_of`, `phase_of`, `memory_of` agree with `tests/reference/ref_expansion.py` enumeration (same ordering).
4. `entry_state_of[r, j]` points to a state with `phase == 1`, `regime == j` and memory `shift(memory_r, current_r)`.
5. `pair` conditioning yields `rate_index_of` with `N(N−1)` distinct values for order 2.

## Out of scope
`A_t` construction (E4-T02b); factorial tuples (E9-T02).

## Connections
- Depends on: E1-T05, E3-T03, E3-T05
- Blocks: E4-T02b, E6-T01, E9-T02
- Relates to: E4-T01 (`ref_expansion`)
