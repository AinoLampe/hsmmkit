# E9-T02 — Factorial/coupled builder design spike and ADR-0008 finalization

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-X01, G6 | post-0.1.0 | §5.10, §4.7.1, ADR-0008, Ghahramani & Jordan 1997 |

## Goal
Decide and prototype how multiple regime chains compose into one `StateSpace` and where approximate inference plugs in.

## Context (self-contained)
- Prototype `build_factorial(chains: list[ModelSpec], coupling: CouplingSpec) -> StateSpace` with tuple `StateIndex` (`regime_of` becomes `(S, C)` for `C` chains; collapse per chain), `A_t` as Kronecker product of per-chain `A_t^{(c)}` for independent chains (`coupling=None`), plus a coupling hook that multiplies by a learnable interaction factor (design only).
- `ApproxInference` Protocol: `fit(posterior_init, ...)`, returning per-chain marginals; candidates: structured mean-field (per-chain forward-backward with expected log-emissions) vs blocked Gibbs; ADR-0008 records the choice with a complexity table (`O(T Π S_c ²)` exact vs `O(T Σ S_c ²)` per sweep approximate) and the memory bound.
- Exact-inference equality test: two independent chains with `N ≤ 3`, `r=1`, joint emission `= Σ` per-chain emissions → factorial `StateSpace` run through the *existing* expanded engine must equal the product of per-chain results (loglik sum, marginals outer product).
- No production wiring: prototype lives under `src/regimekit/statespace/factorial.py` behind `experimental` flag and `tests/spikes/`.

## Creates
- `src/regimekit/statespace/factorial.py` (prototype), `engine/approx.py` (Protocol), `docs/adr/0008-*.md` finalized (Status accepted), `tests/spikes/test_e9_t02_factorial.py`

## Acceptance criteria
1. ADR-0008 accepted with algorithm choice, complexity analysis and interface diff.
2. Prototype passes the exact-inference equality test for two independent chains (`loglik` ≤ 1e-10, marginals ≤ 1e-10).
3. `InferenceEngine.approx()` on existing engines still returns `None`; nothing changes for `M=1`, single-chain users (regression).

## Out of scope
Coupled-chain estimation; multi-series emissions (E9-T01).

## Connections
- Depends on: E4-T02b, E5-T01 (log_joint reuse)
- Blocks: —
- Relates to: E1-T05 (`StateIndex` generalization)
