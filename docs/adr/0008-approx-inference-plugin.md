# ADR-0008: Approximate-inference plug-in interface reserved for factorial models

## Context

Factorial or coupled hidden Markov models (multiple interacting chains) make exact forward-backward
intractable in general (state space grows multiplicatively). regimekit does not need exact factorial inference
for 0.1.0, but the engine interface should not need to change shape when approximate inference is added later.

## Decision

`InferenceEngine` reserves an `approx()` method as part of its public interface from 0.1.0, alongside `loglik`,
`forward`, `forward_backward`, `viterbi`. It is unimplemented (or raises `NotImplementedError`) until the
factorial design spike (E9-T02, post-release) finalizes an approximate algorithm (e.g. structured
variational inference or a mean-field approximation) and provides a concrete implementation.

## Consequences

- The `InferenceEngine` protocol/base class is stable across 0.1.0 and the eventual factorial extension;
  downstream code that pattern-matches on engine methods does not need to change when `approx()` gains a body.
- E9-T02 is scoped as a design spike only; it does not have to ship a production-quality approximate engine
  for 0.1.0.
- No batch (`B`) dimension or factorial state space is introduced in the core for 0.1.0 (guardrail, overview
  §8); this ADR only reserves the interface shape.

## Status

accepted
