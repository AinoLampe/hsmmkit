# E4-T03b — `ExpandedHMMEngine`: backward recursion, smoothing, `Posterior`, property tests

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-I01, FR-I02, NFR-01, NFR-02 | M2 | §4.7.2 Backward, Appendix C, §6.4.2 |

## Goal
Complete forward-backward: `log_beta`, smoothed posterior `γ`, and the `Posterior` object with regime marginals.

## Context (self-contained)
Backward (reverse `lax.scan`, `reverse=True`): `β_T = 0`; `β_t(s) = logsumexp_{s'}[log A_{t+1}(s, s') + ℓ_{t+1}(s') + β_{t+1}(s')]`, i.e. `logsumexp(logA_next + (ℓ_{t+1} + β_{t+1})[None, :], axis=1)`. `A_{t+1}` is rebuilt inside the reverse scan from `(z_pi_{t+1}, z_d_{t+1}, dt_{t+1})` — never stored for all `t`.
Smoothed: `γ_t = softmax(α_t + β_t)` (rows normalized with `logsumexp`); filtered `softmax(α_t)`. `Posterior(log_alpha, log_beta, smoothed, loglik, index)`; `regime_filtered = softmax(α) @ C`, `regime_smoothed = γ @ C`. Structural `−inf` entries: use `jnp.where` before `exp`. `forward_backward` is a single jitted function calling forward then backward (share `log_emis`).

## Creates
- `engine/expanded_hmm.py` (backward, `forward_backward`)
- `tests/golden/test_e4_t03b_forward_backward.py`, `tests/property/test_e4_t03b_posterior.py`, `tests/benchmark/test_e4_t03b_bench.py`
- `docs/math/engine.md` (Appendix C rendered)

## Acceptance criteria
1. Golden: `log_beta`, `smoothed` max-abs ≤ 1e-8 vs reference (same hypothesis strategy as E4-T03a).
2. Property: filtered and smoothed rows sum to 1 ≤ 1e-10; `logsumexp(α_t + β_t)` constant in `t` (≤ 1e-8 rel) and equal to `loglik`.
3. Property: loglik invariant under permuting equivalent phases — construct two `StateSpace`s differing only in state ordering (permutation of `S`), assert equal loglik ≤ 1e-10 and permuted posteriors.
4. Property: `M`-masked steps have smoothed regime marginals equal to the reference with zero emission.
5. Performance: jitted `forward_backward` ≤ 100 ms at NFR-02 size (`T=5000, N=4, r=5, order 1, 2 covariates`, 8-core CPU); baseline JSON entry added.
6. Non-finite outputs raise `NumericalError` at the public boundary.

## Out of scope
Elapsed/phase marginal utilities (E4-T04); Viterbi (E4-T05).

## Connections
- Depends on: E4-T03a
- Blocks: E4-T04, E4-T05, E4-T07, E5-T01, E9-T03
- Relates to: E4-T06b (equivalence test uses `smoothed`)
