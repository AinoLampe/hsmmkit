# ADR-0019: Supervision as engine-level masks

Status: accepted (E12-T06).

## Context

Labelled data (activity recognition, annotated segments) should be usable by every estimator without special
cases inside the estimators or engines (D2). Full labels, partial labels, label sets and known change points all
restrict which state paths are possible.

## Decision

- Supervision is data, not model structure: `PreparedDataset.labels (T, N)` bool (a regime is allowed at a step)
  and `PreparedDataset.boundaries` (`must_switch_at`, `must_not_switch_at`), produced by the `Preparer` from
  `data.labels`. An unlabelled step allows every regime.
- The engines multiply the constraints into the joint likelihood; rows are **not** renormalized, so the
  constrained likelihood is `p(y, constraints | theta)` and posteriors are conditioned on the constraints.
  - Expanded engine: the emission table gets `-inf` where a label forbids the regime; `A_t` keeps only the exit
    entries (a regime change) at a must-switch step and drops exactly those at a must-not-switch step.
  - Explicit engine: a segment `(start, regime, d)` is feasible iff the labels allow the regime for `d` steps, no
    must-switch step lies strictly inside it and it does not end right before a must-not-switch step; the masks are
    folded into the per-start duration table (the survival of a cut-off final segment still uses the
    unconstrained pmf), so the recursions are unchanged.
  - Padding in batches is unconstrained (`labels` is `True` there, the boundary code 0).
- Labels are dynamic arrays (a compiled engine serves any label pattern of one shape). Boundaries are encoded as an
  int8 code per step (`PaddedBatch.boundary_code` in batches); in the single-sequence engines they are static
  metadata of the data set, so a new boundary set compiles anew.
- A model whose constraints admit no path has likelihood exactly zero: the public engine functions raise
  `ConstraintError(step, sequence)` (exit code 3) outside jit, naming the first step at which every filtered state
  has probability zero (expanded) and the sequence of a batch.
- `init: labels` estimates parameters from the hard-labelled steps (moment matching, label-run lengths, label-change
  counts); regimes without labelled steps fall back to the `random` init.

## Consequences

- Every estimator that goes through the engines (MAP now; EM/GEM/VB later) is supervised for free.
- Constrained likelihoods are not comparable with unconstrained ones (information criteria of supervised fits are
  not defined).
- The explicit engine with constraints builds `(T, N, D_max)` tables, so it is limited by the usual
  `T * N * D_max` memory guard.
