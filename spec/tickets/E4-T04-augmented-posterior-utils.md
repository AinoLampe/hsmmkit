# E4-T04 — Augmented posterior utilities: phase/elapsed marginals, expected elapsed/remaining duration

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-I02, G1 | M2 | §4.7.1, §5.4 (`Posterior.elapsed_posterior`) |

## Goal
Expose interpretable summaries of the expanded posterior beyond regime marginals.

## Context (self-contained)
Functions in `engine/augmented.py` operating on `Posterior` + `StateIndex` (+ spec for `r_i`, `λ`):
- `phase_marginals(post) -> (T, N, r_max)`: sum of `smoothed` over memory for each `(regime, phase)`, zero-padded for `phase > r_i`.
- `memory_marginals(post) -> (T, R)`: mass per memory tuple row (order ≥ 2), else `(T, 1)`.
- `expected_remaining_steps(post, theta, dt) -> (T, N)`: for the NB phase chain, conditional on `(regime i, phase c)`, remaining phases `r_i − c + 1` each taking `1/a_i(t)` steps in expectation (using `a_i(t)` at step `t`, constant-rate approximation) → weighted by phase marginals.
- `expected_elapsed_steps(post) -> (T, N)`: computed from the smoothed posterior by a run-length recursion on regime marginals is **not** exact for HSMM; instead implement via the explicit-duration engine when available (`elapsed_posterior (T, N, D_max)` from E4-T06a) and for the expanded engine return the phase-based lower bound `(c − 1)·(1/a_i)` + note in docstring. `Posterior.elapsed_posterior()` dispatches on `index` type / engine tag.
- `collapse_to_regimes(x (T, S), C) -> (T, N)` generic helper.

## Creates
- `src/regimekit/engine/augmented.py`; `Posterior` methods wired
- `tests/unit/test_e4_t04_augmented.py`

## Acceptance criteria
1. `phase_marginals` sums over phases to the regime marginals ≤ 1e-10; `memory_marginals` rows sum to 1.
2. On simulated data from a known model (E6-T01 not yet available → use the test-side NumPy phase simulator from E3-T03), the mean of `expected_remaining_steps` at regime-entry steps is within 15 % of the analytic `r_i/a_i` over 2000 segments.
3. Functions are jittable.

## Out of scope
Explicit-engine elapsed posterior (E4-T06a); forecasting (E6-T04a uses `softmax(α_T)` directly).

## Connections
- Depends on: E4-T03b
- Blocks: E6-T04a
- Relates to: E4-T06a, E5-T04 (diagnostics use expected durations)
