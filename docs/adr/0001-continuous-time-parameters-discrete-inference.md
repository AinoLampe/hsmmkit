# ADR-0001: Continuous-time parameters, discrete-time inference

## Context

regimekit models regime durations and rates (e.g. annualized drift/volatility, mean sojourn time) as
continuous-time quantities, but all inference (forward-backward, Viterbi, MAP) runs on a discrete grid of
observation times. The grid may be irregular (missing bars, holidays) and its native step `Δ_t` can be
expressed in trading days, calendar days, or synthetic steps.

## Decision

Parameters are specified and reported in continuous time (annualized under the `trading`/`calendar` clocks,
per-step under the `steps` clock — see ADR-0013), and converted to per-step quantities inside the engines at
each step using the grid's actual `Δ_t`. There is no intra-interval regime switching: exactly one state (or
duration-augmented state) is active for the whole of `[t-1, t)`.

## Consequences

- Every family (`EmissionFamily`, `DurationFamily`, `TransitionModel`) must implement a conversion from its
  continuous-time parameterization to the per-step quantity used inside a `jax.lax.scan` body.
- Irregular grids are supported without resampling, at the cost of every step-building function taking `Δ_t`
  as an explicit input rather than assuming a constant step.
- Reported parameters (e.g. `mean_sojourn`) are always comparable across differently-sampled datasets that
  share a clock, which is the concept's primary usability goal.

## Status

accepted
