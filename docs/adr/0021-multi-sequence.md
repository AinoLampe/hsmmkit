# ADR-0021: Multi-sequence data through `vmap` over single-sequence engines and length buckets

## Context

Applications such as activity recognition fit one model to many independent sequences of different lengths
(50 sessions of 150-400 steps). `PreparedDataset` is a single-sequence contract used by every layer; adding a
batch axis to it would touch each engine, estimator and metric, and variable lengths would force a recompilation
for every new length.

## Decision

- Contracts stay single-sequence: a data set with many sequences is a `SequenceBatch` of `PreparedDataset`s
  (`hsmmkit.data.batch`). `PreparedDataset` has no `B` dimension (Part II concept §19 #11).
- The batched engines (E12-T01b) `jax.vmap` the per-sequence engine functions over a `PaddedBatch`. A padded step
  has `valid == False`: its emission log-density is 0 and `A_t = I`, so padding changes neither the
  log-likelihood nor the posteriors of the real steps.
- Shapes are bucketed so that compilation does not follow the data: `T_max` is the smallest static bucket edge
  `>=` the longest sequence (the longest length rounded up to a multiple of the last edge beyond the edges) and `B`
  is padded to the next power of two with empty sequences (length 0, log-likelihood 0).
- The `Preparer` fits its standardization statistics on the pooled training rows of all sequences and prepares
  every sequence on its own (`Preparer.transform_batch`); own lags and lagged covariates are generated within a
  sequence, never across a boundary. A group column (`data.sequences.group_by`) selects an initial distribution
  per group.

## Consequences

- One compiled program per `(B_pad, T_bucket)` pair; a walk-forward over sequences or a growing data set reuses it.
- Padding costs time (the longest sequence of a bucket sets the work) and memory `B_pad * T_max`; the bucket
  edges are configurable (`data.sequences.bucket_edges`).
- Fitting with `B > 1` stays gated until E12-T02; this ADR covers the data side and the engine contract.

## Status

accepted
