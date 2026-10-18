# E4-T02b — Per-step expanded transition matrix `A_t` (jit-able, built inside scan)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M08, FR-M09, FR-M10, NFR-01 | M2 | §4.7.2 (stay/advance/exit), §4.3 Generator mode, Appendix C |

## Goal
Implement `StateSpace.transition_fn(θ, z_pi_t, z_d_t, Δ_t) → A_t (S, S)` and its log version, for one time step, using precomputed index arrays only (no Python loops on data).

## Context (self-contained)
For state `s = (m, i, c)` with `a_i(t)` from `duration.advance_prob` (row `rate_index_of[s]`) and `π_{(m,i) j}(t)` from `transition.jump_probs` (row `row_of_state[s]`):
```
stay:     A_t[s → s]                                  = 1 − a_i(t)
advance:  A_t[s → (m, i, c+1)]                        = a_i(t)                     if c < r_i
exit:     A_t[s → entry_state_of[row_of_state[s], j]] = a_i(t) · π_{(m,i) j}(t)     if c == r_i, j ≠ i
```
Implementation: precompute static index arrays in the builder: `next_phase_state (S,)` (`−1` at last phase), `is_last_phase (S,)` bool, `exit_targets (S, N)` = `entry_state_of[row_of_state[s], :]`. Build `A_t` with `jnp.zeros((S,S)).at[...]` scatter-adds using these arrays (three scatters: diag, advance, exit with `where(is_last_phase & target_mask)`). Return both `A_t` and `log A_t` (`log` with `−inf` at structural zeros via `jnp.where(A>0, log A, −inf)`). Signature per step so it can be called from the scan body with `z_pi_t (K_pi,)`, `z_d_t (K_d,)`, `dt_t` scalars; `advance_prob`/`jump_probs` are called on `T=1` slices (or the families expose per-step variants — choose per-step variants `advance_prob_t`, `jump_probs_t` to avoid reshaping; add them to E3-T03/T05 families as thin wrappers).
Generator mode (`transition.mode == "generator"`, requires geometric, `r = 1`, order 1): `A_t = expm(Q Δ_t)` from E3-T05; `S = N`.
`stationary` initial distribution: `A_bar = transition_fn(θ, z=0, Δ=nominal_dt)`; stationary vector via `jnp.linalg.eig` on `A_barᵀ` (take eigenvalue closest to 1, normalize, clip ≥ 0) — jittable; restricted to no-covariate or `z=0` semantics (document).

## Creates
- `src/regimekit/statespace/transitions.py`; `StateSpace.transition_fn` wired in `builders.py`; per-step wrappers in `duration/negbin.py`, `duration/geometric.py`, `transition.py`
- `tests/golden/test_e4_t02b_transitions.py`, `tests/property/test_e4_t02b_transitions.py`

## Acceptance criteria
1. `A_t` rows sum to 1 ≤ 1e-12; structural zeros exactly 0; `log A_t` is `−inf` exactly there.
2. Equals `tests/reference/ref_expansion.py` matrices ≤ 1e-12 for random `N ∈ [2,4]`, `r_i ∈ [1,3]`, order ∈ {1,2}, with random covariates.
3. `jax.jit(transition_fn)` compiles once per `StateSpace`; calling with different `θ`/`z` does not recompile (assert via `jax.jit(...).lower(...).compile()` count or `_cache_size`).
4. Generator mode reproduces Appendix E weekly numbers; `stationary` vector satisfies `π A = π` ≤ 1e-10.
5. Differentiable: `jax.grad` of `sum(A_t)` w.r.t. `θ` finite.

## Out of scope
Forward/backward (E4-T03a/b).

## Connections
- Depends on: E4-T02a
- Blocks: E4-T03a, E6-T01, E9-T02
- Relates to: E3-T03, E3-T05 (per-step wrappers)
