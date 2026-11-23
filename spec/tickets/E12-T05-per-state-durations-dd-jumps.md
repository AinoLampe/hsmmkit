# E12-T05 — Per-state duration families and duration-dependent jump probabilities (explicit engine)

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-DU03, FR-DU05, D16 | M9 | Part II §14.5 (duration-dependent jumps), §15.2 (`ModelSpec.per_state_durations`), Appendix H; Part I E3-T04, E4-T06a |

## Goal
Allow a different duration family per state and jump probabilities that depend on how long the segment being left lasted, both in the explicit-duration engine.

## Context (self-contained)
**Per-state durations** (`model/duration/per_state.py`): `PerStateDuration(families: tuple[DurationFamily, ...])` with `representation="explicit"`; `param_specs` = concatenation with prefixes `duration.s{i}.<name>`, each family built with `n_states = 1`; `log_pmf(theta, d_max, nominal_dt, z_entry) -> (N, D_max)` stacks per-state rows (or `(T, N, D_max)` with entry-time covariates); `mean_sojourn` per state from its family. Config `model.duration.per_state: [{family, params}, …]` (length `n_states`); requires `engine: explicit` (validated in E10-T10). Phase families (`geometric`, `negbin`) are allowed as members through their explicit `log_pmf`.
**Duration-dependent jumps**: `π_ij(d) ∝ Ω_ij exp(a_ij + b_ijᵀ z^π_t + c_ij log d)` evaluated with the duration `d` of the segment being **left**, at its end time `t`.
- Specs: `transition.c` with the same flat layout as the masked logits of E12-T03 (one coefficient per free logit), default `NormalPrior(0, coef_prior_sd)`.
- Explicit forward recursion change: the jump term into a new segment uses `log π_ij(d')` inside the sum over the previous segment's `(i, d')`:
```
α_t(j, d) = log Σ_i Σ_{d'} exp( α_{t−d}(i, d') + log π_ij(d'; t−d) ) + log p_j(d) + L[t, j] − L[t−d, j]
```
  This replaces the Part I buffer `A_u(i) = logsumexp_{d'} α_u(i, d')` by the entry term `B_u(j) = logsumexp_{i, d'} [α_u(i, d') + log π_ij(d'; u)]`, computed once when `α_u` is final (segments ending at `u`; covariates `z^π_u`). The rolling buffer keeps `B_u` with shape `(D_max, N)` as before; the extra cost is `O(N² D_max)` per step instead of `O(N²)`. Document the cost and keep the old path when `duration_dependent_jumps: false`. The chunking bound of E4-T06b is unchanged (buffer shape unchanged).
- Backward recursion and segmental Viterbi updated symmetrically.
- `engine: expanded` with `duration_dependent_jumps: true` → `ConfigError(path="model.duration.duration_dependent_jumps", "only the explicit engine can express duration-dependent jumps")` (exit happens only from phase `r_i`).
- Gate rows `per_state`, `duration_dependent_jumps` removed.

## Creates
- `src/hsmmkit/model/duration/per_state.py`, edits in `model/transition.py`, `engine/explicit_duration.py`, `engine/viterbi.py`; NumPy reference extension `tests/reference/ref_hsmm.py` (`pi_of_d`)
- `tests/golden/test_e12_t05_dd_jumps.py`, `tests/recovery/test_e12_t05_mixed_durations.py` (`-m slow`)

## Acceptance criteria
1. Mixed per-state model (state 0 geometric, state 1 gamma, state 2 nonparametric) recovers each family's mean sojourn within 25 % (`T = 20 000`, ≥ 4/5 seeds).
2. Duration-dependent jumps: with `c = 0` results equal the standard engine bitwise; on synthetic data with `c_01 = 1.0, c_02 = −1.0` the fitted signs are correct in ≥ 4/5 seeds.
3. Golden vs the extended NumPy reference ≤ 1e-8 (loglik, smoothed marginals, Viterbi path).
4. Expanded engine rejects the option with the documented `ConfigError`.
5. Appendix H duration block builds (other gates may still fire).

## Tests
Golden, slow recovery, config rejection.

## Out of scope
Duration-dependent jumps in the expanded engine (impossible by construction).

## Connections
- Depends on: E12-T03
- Blocks: E13-T03b, E16-T05
- Relates to: E3-T04, E4-T06a/b, E13-T03b (GEM with per-state families)
