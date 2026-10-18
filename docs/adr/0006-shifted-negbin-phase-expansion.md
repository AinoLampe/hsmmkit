# ADR-0006: Shifted negative-binomial semantics for the phase expansion

## Context

Representing a negative-binomial (or other explicit) sojourn duration inside the generic expanded-state
engine (ADR-0004) requires expanding each logical regime into a chain of "phase" sub-states that the engine's
ordinary geometric-sojourn machinery can traverse. The minimum sojourn length and the phase-transition
probabilities must together reproduce the target duration distribution exactly.

## Decision

The phase expansion uses shifted-negative-binomial semantics: a regime with negative-binomial duration
parameter `r_i` expands into a chain enforcing a **minimum sojourn of `r_i` steps** before the regime is
allowed to end, with per-phase continuation probabilities chosen so the aggregate sojourn-length distribution
matches the target negative-binomial pmf.

## Consequences

- `StateIndex`/`build_expanded` (E4-T02a) must size the expanded state space using `r_i` (or `N(N-1)` for
  pair-conditioned duration), not a fixed constant, and expose `row_of_state` for mapping expanded rows back
  to logical regimes.
- The explicit-duration engine (`engine/explicit_duration.py`) must agree with the expanded engine on this
  semantics to within the tolerance in overview §4.7 (shifted-NB `D_max` mass loss < 1e-10; loglik/posterior
  agreement 1e-6), which is exercised directly in E4-T06b's equivalence tests.
- Any future duration family with a hard minimum sojourn (e.g. Gamma-based ones under a shifted
  discretization) can reuse the same phase-expansion machinery.

## Status

accepted
