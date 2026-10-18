# E4-T01 — NumPy reference implementations (`tests/reference/`)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | NFR-04 | M1 | §6.4.1, §4.7.2, §4.7.3, Appendix C |

## Goal
Write small, readable, slow NumPy implementations that serve as ground truth for golden tests of the JAX engines.

## Context (self-contained)
Each module < 300 lines, plain loops, log space via `scipy.special.logsumexp`, no JAX import. Inputs are plain arrays so they can be driven by hypothesis.

1. `ref_hmm.py`: `forward_backward(log_rho (S,), log_A (T, S, S) per step [log_A[t] used to move from t−1 to t, t ≥ 1], log_emis (T, S)) -> (log_alpha, log_beta, log_gamma, loglik)`; `viterbi(...) -> path (T,)`. Convention: `α_1 = log_rho + ℓ_1`; `α_t = ℓ_t + logsumexp_{s'}(α_{t−1}(s') + log A_t(s', s))`; `β_T = 0`; `β_t = logsumexp_{s'}(log A_{t+1}(s, s') + ℓ_{t+1}(s') + β_{t+1}(s'))`.
2. `ref_hsmm.py`: explicit-duration forward-backward (Guédon 2003 / Yu 2010) with right-censoring of the last segment and `D_max`: `α_t(j, d) = [Σ_i Σ_{d'} α_{t−d}(i, d') π_ij] · p_j(d) · Π_{u=t−d+1}^{t} f_j(y_u)` (with `t − d = 0` using the initial distribution), backward counterpart, smoothed regime marginals `(T, N)`, elapsed-duration posterior `(T, N, D_max)`, loglik including censored last segment (sum over survival of durations extending past `T`); segmental Viterbi.
3. `ref_expansion.py`: brute-force expansion of an order-`k`, NB-phase HSMM into a first-order HMM: enumerate states `(memory, regime, phase)`, build `A_t` from `a_i(t)` and `π_(m,i)j(t)` exactly as §4.7.2 (stay `1−a`, advance `a`, exit `a·π`), build collapse matrix. Used to test E4-T02a/b independently of the JAX builder.
4. `ref_dists.py`: NB shifted pmf by explicit enumeration (`scipy.stats.nbinom`), GBM log-density.
5. Hand-computed examples: 2-state, 3-step HMM with known α/β values in a docstring; 2-state HSMM with `D_max = 3` computed by enumerating all segmentations (`ref_hsmm.brute_force_loglik`).

## Creates
- `tests/reference/{__init__,ref_hmm,ref_hsmm,ref_expansion,ref_dists}.py`
- `tests/reference/test_reference_selftests.py`

## Acceptance criteria
1. `ref_hmm` reproduces the hand-computed 2-state/3-step α, β, loglik to 1e-12.
2. `ref_hsmm.forward_backward` loglik equals `brute_force_loglik` (enumeration of all segmentations) on `N=2, T=6, D_max=3` random instances to 1e-12.
3. `ref_hmm` on `ref_expansion`'s expanded chain with `r=1`, order 1 equals `ref_hsmm` with geometric pmf (same loglik ≤ 1e-10) — establishes the phase/explicit equivalence at reference level.
4. Each file < 300 lines, no JAX.

## Tests
The selftests above.

## Out of scope
Any production code.

## Connections
- Depends on: E1-T04
- Blocks: E4-T03a, E4-T06a
- Relates to: E4-T02a/b (use `ref_expansion`)
