# E4-T05 — Viterbi decoding on the expanded space, segment run-length encoding, AIC/BIC/ICL

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-I01, FR-I03 | M2 | §4.7.4, §4.10 |

## Goal
Provide the most likely expanded path, its regime segments, and the information criteria.

## Context (self-contained)
- Viterbi (log-space max-product) in `engine/viterbi.py`: forward scan `δ_t(s) = ℓ_t(s) + max_{s'}[δ_{t−1}(s') + log A_t(s', s)]`, store argmax `ψ_t (T, S)` (int32, acceptable memory `T·S`), backtrack with a reverse scan → `path (T,)` expanded states. Regime path `regime_of[path]`.
- `segments(regime_path) -> list[(start, end, regime)]` with `end` inclusive, RLE in NumPy (outside jit).
- Information criteria in `engine/criteria.py`: `p = ParamManager.n_free`; `AIC = 2p − 2ℓ`, `BIC = p log T − 2ℓ`, `ICL = BIC + 2·H(γ)` where `H(γ) = −Σ_t Σ_s γ_t(s) log γ_t(s)` (entropy of the smoothed expanded posterior; use `xlogy`), `T` = number of unmasked steps.

## Creates
- `src/regimekit/engine/viterbi.py`, `src/regimekit/engine/criteria.py`
- `tests/golden/test_e4_t05_viterbi.py`, `tests/unit/test_e4_t05_criteria.py`

## Acceptance criteria
1. Golden: path equals `tests/reference/ref_hmm.viterbi` on random instances (ties broken identically: lowest index).
2. Viterbi path log-probability ≥ log-probability of any of 100 posterior-sampled paths (sample from `softmax(α)` forward-filter backward-sample is not required; sampling from the smoothed marginals independently suffices as a weaker check) on random instances.
3. `segments` RLE verified on hand-built paths (single segment, alternating, ends).
4. `ICL ≥ BIC` always; AIC/BIC hand-checked on a tiny case; `p` matches Appendix B.

## Out of scope
Segmental Viterbi for the explicit engine (E4-T06b); CLI (E6-T02).

## Connections
- Depends on: E4-T03b
- Blocks: E5-T04, E6-T02
- Relates to: E1-T03b (`n_free`)
