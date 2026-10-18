# ADR-0005: MAP via L-BFGS is the primary estimator; EM optional (v1.x)

## Context

Both EM and direct gradient-based MAP optimization can fit HSMM parameters. EM has closed-form,
numerically stable M-steps for some families (notably GBM emissions) but requires a bespoke M-step derivation
per family, which does not generalize to the family-agnostic design of ADR-0004. Gradient-based MAP, using
JAX's autodiff through the forward recursion, generalizes to any differentiable family for free.

## Decision

The primary estimator is MAP via L-BFGS (with autodiff gradients through `engine.loglik`), implemented once in
`estimation/objective.py` and `estimation/optimizers.py`, family-agnostic. An EM implementation specific to the
GBM emission family is optional, deferred to v1.x (E5-T06), and never required for the 0.1.0 estimation path.

## Consequences

- One estimation code path works for every emission/duration family combination without per-family M-step
  code, consistent with ADR-0004's "no special cases" principle.
- Optimization quality depends on multi-start and initialization heuristics (`estimation/init.py`) rather than
  EM's monotonic-improvement guarantee; recovery-suite tolerances (Appendix D) are tuned accordingly.
- EM remains available as a future, opt-in alternative for the specific case where it is cheap to derive,
  without blocking 0.1.0.

## Status

accepted
