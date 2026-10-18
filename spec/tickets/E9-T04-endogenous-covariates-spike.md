# E9-T04 — Endogenous covariates design spike (covariates as emission channels)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | FR-X02 | post-0.1.0 | §5.10 |

## Goal
Write the ADR describing how covariates such as VIX could be modelled jointly as observation channels rather than exogenous inputs, and what interface changes that implies.

## Context (self-contained)
- Idea: extend `y_t` from `(M,)` returns to `(M + K,)` channels, with a joint emission family (e.g. GBM for returns × Gaussian AR(1)-in-regime for log-VIX) — a `ProductEmission` (E9-T01) with heterogeneous per-channel families; covariates then no longer feed transition/duration links directly (or feed them with lag as before, but their own dynamics are modelled).
- Interface diff to document: `PreparedDataset` unchanged (channels are extra columns of `returns` with their own `mask`; the data layer supplies a `channel_kind` tuple); `EmissionFamily.param_specs` per channel; forecasting samples channels jointly, removing the need for a `CovariateForecaster` for endogenous channels.
- Deliverable is an ADR only; a small pseudo-code prototype may be attached as an appendix. No code merged to `main` beyond the ADR.

## Creates
- `docs/adr/0013-endogenous-covariates.md` (Status: proposed)

## Acceptance criteria
1. ADR contains: motivation, data-contract impact (none/what), emission interface diff, forecasting impact, identifiability caveats, migration path, and a decision on whether to schedule implementation.
2. Reviewed by the owner; Status set to accepted or rejected.

## Out of scope
Any implementation.

## Connections
- Depends on: E9-T01
- Blocks: —
- Relates to: E6-T03 (forecasters would become optional for endogenous channels)
