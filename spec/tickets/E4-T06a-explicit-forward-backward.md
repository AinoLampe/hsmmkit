# E4-T06a — `ExplicitDurationEngine`: forward/backward with right-censoring and `D_max`

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-M06, FR-I01, FR-I02, NFR-01 | M3 | §4.7.3, §4.1 (regular grid), ADR-0007 |

## Goal
Implement the Guédon/Yu explicit-duration forward-backward in JAX for arbitrary duration pmfs, producing a `Posterior` with augmented state `(regime, elapsed duration)`.

## Context (self-contained)
Inputs: `log_emis (T, N)` (masked steps → 0), cumulative sums `L[t, j] = Σ_{u≤t} log f_j(y_u)` so that `Π_{u=t−d+1}^{t} f_j(y_u) = exp(L[t] − L[t−d])`; `log_pmf (N, D_max)` (or `(T, N, D_max)` with entry-time covariates, indexed by the segment *start* `t−d+1`); jump `log π (N, N)` (constant, or `(T, N, N)` indexed by segment start); initial distribution `ρ (N,)` over the regime of the first segment (uniform default).
Forward: `α_t(j, d) = log[ Σ_i Σ_{d'} α_{t−d}(i, d') + log π_ij ] + log p_j(d) + L[t, j] − L[t−d, j]` for `t − d ≥ 1`, and the `t − d = 0` term uses `log ρ_j`. Implement with a scan over `t` carrying the last `D_max` values of `A_t(i) = logsumexp_{d'} α_t(i, d')` (a rolling buffer `(D_max, N)`), so each step is `O(N² + N D_max)`. Right-censoring: the last segment may still be in progress at `T`, so the loglik sums over every regime `j` and every in-progress length `d ≤ D_max`, using the survival `log S_j(d−1) = log(1 − Σ_{u<d} p_j(u))` in place of `log p_j(d)` for that final segment (plus the completed-segment terms `α_T(j, d)`). The reference implementation (E4-T01 `ref_hsmm`) defines this exactly — mirror it.
Backward: symmetric recursion `β_t(i)` = probability of `y_{t+1:T}` given a segment of regime `i` ends at `t`; smoothed regime marginal `γ_t(j) = Σ_{segments covering t}`, computed via the standard "difference of cumulative" trick (Yu 2010 eqs.) in `O(T N D_max)`; elapsed-duration posterior `(T, N, D_max)`.
Regular grid required: if `data.irregular` (`max Δ/min Δ > 1.5`) → `warnings.warn`; use `nominal_dt` everywhere.
Memory: if `T·N·D_max > 5e7` raise `NumericalError("use chunked scan")` in this ticket (chunking implemented in E4-T06b).

## Creates
- `src/regimekit/engine/explicit_duration.py` (forward, backward, `forward_backward`, `loglik`)
- `tests/golden/test_e4_t06a_explicit.py`, `tests/property/test_e4_t06a_explicit.py`
- `docs/math/engine_explicit.md`

## Acceptance criteria
1. Golden vs `tests/reference/ref_hsmm`: loglik rel ≤ 1e-8, smoothed regime marginals and elapsed posterior max-abs ≤ 1e-8 (hypothesis `N ∈ [2,4]`, `T ∈ [5,120]`, `D_max ∈ [2,20]`, random pmfs including geometric and shifted-NB).
2. Right-censoring: loglik on a sequence equals the enumeration-based `brute_force_loglik` for `T=6, D_max=3` ≤ 1e-10.
3. Rows of smoothed marginals sum to 1 ≤ 1e-10; elapsed posterior sums over `(N, D_max)` to 1.
4. Entry-time covariates: with `(T, N, D_max)` pmfs the result equals a reference using per-segment-start pmfs.
5. Performance: ≤ 2 s at `T=5000, N=4, D_max=250` (8-core CPU); gradient of loglik finite and matches finite differences rel ≤ 1e-5 on a tiny instance.

## Out of scope
Segmental Viterbi, chunked scan, expanded-engine equivalence (E4-T06b).

## Connections
- Depends on: E4-T01, E3-T04
- Blocks: E4-T06b
- Relates to: E4-T04 (`elapsed_posterior`)
