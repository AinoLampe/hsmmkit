# ADR-0004: One generic engine over a structured state index

## Context

Higher-order Markov chains, negative-binomial (and other explicit) sojourn durations, and future factorial /
coupled-chain extensions could each be implemented as a bespoke inference engine, but that multiplies the
amount of forward-backward/Viterbi code that must be kept numerically correct and performant.

## Decision

There is one generic expanded-state engine (`engine/expanded_hmm.py`) and one generic explicit-duration engine
(`engine/explicit_duration.py`). Higher-order memory, shifted-negative-binomial phase expansion, and other
structural variations are expressed as `statespace` **builders** that construct a `StateIndex` and a per-step
transition-matrix builder consumed by the generic engines (D2: "structure lives in builders, never as special
cases inside engines"). Factorial/coupled models (E9-T02) get an approximate-inference plug-in point
(`InferenceEngine.approx()`, ADR-0008) rather than a third bespoke exact engine.

## Consequences

- Adding a new duration family or memory structure never requires touching `engine/`; it only requires a new
  `statespace` builder and, for parametric duration families, a `DurationFamily` implementation.
- The generic engines must be written generally enough (arbitrary `n_expanded`, arbitrary `MemoryLayout`) to
  support every current and near-future builder, which is more work upfront but avoids engine-code duplication.
- Factorial models are explicitly out of exact-inference scope for 0.1.0; they get a documented approximate
  interface instead (see ADR-0008).

## Status

accepted
