# E4-T06b — Explicit engine: segmental Viterbi, chunked scan, expanded-vs-explicit equivalence

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-I01, NFR-01, §6.4.5 | M3 | §4.7.3, §4.7.4, §6.3 item 8, Appendix D |

## Goal
Finish the explicit-duration engine (decoding and memory bound) and prove numerical equivalence with the expanded engine on shifted-NB durations.

## Context (self-contained)
- Segmental Viterbi: `δ_t(j) = max_{i, d} [δ_{t−d}(i) + log π_ij + log p_j(d) + L[t,j] − L[t−d,j]]` with argmax `(i, d)` stored `(T, N, 2)`; backtrack by jumping `d` steps; output expanded path encoded as `(regime, elapsed)` pairs plus regime path and segments (same RLE as E4-T05). Right-censored last segment uses survival as in E4-T06a.
- Chunked scan: when `T·N·D_max > 5e7`, run the forward pass in chunks of `T_chunk` steps carrying the rolling `(D_max, N)` buffer; store only regime marginals `(T, N)` and, optionally, a decimated elapsed posterior; expose `chunk_size` in `EstimationSection`/engine kwargs (`None` = auto). Backward pass chunked symmetrically (two-pass: store chunk boundary carries, recompute inside).
- Equivalence test set-up: NB family `r = (2, 3)`, `λ` chosen so mean sojourn ≈ 30 steps, `D_max = 400` so truncated mass `< 1e-10` (assert on the pmf), constant `Δ`, no covariates, same `log_emis`; expanded engine (E4-T03b) vs explicit engine with `negbin_shifted_logpmf`. Note the initial distribution must match: expanded `uniform` over phase-1 states ⇔ explicit `uniform` over first-segment regimes.

## Creates
- `engine/explicit_duration.py` (viterbi, chunking); `engine/viterbi.py` (segmental variant)
- `tests/golden/test_e4_t06b_segmental_viterbi.py`, `tests/golden/test_e4_t06b_equivalence.py`, `tests/unit/test_e4_t06b_chunking.py`

## Acceptance criteria
1. Segmental Viterbi equals `ref_hsmm` segmental Viterbi on random instances.
2. Equivalence (Appendix D): loglik rel ≤ 1e-6; smoothed regime marginals max-abs ≤ 1e-6 between engines on the NB set-up, for 20 random seeds.
3. Chunked forward-backward equals unchunked ≤ 1e-10 on `T=1000` with `chunk_size=64`; a synthetic `T·N·D_max = 6e7` case runs without allocating `(T, N, D_max)` (assert peak memory via `jax.jit(...).lower().compile().memory_analysis()` temp size below the full-array size).
4. Irregular-grid warning emitted exactly once per call.

## Out of scope
Engine selection in config (already in E1-T02a); explicit-engine-specific forecasting (E6-T04a samples from the expanded engine; the explicit engine's filtered `(regime, elapsed)` is converted via `elapsed_posterior`).

## Connections
- Depends on: E4-T06a, E4-T03b
- Blocks: E4-T07, E5-T05a (explicit gamma scenario)
- Relates to: ADR-0006, ADR-0007
