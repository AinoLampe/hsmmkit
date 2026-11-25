# ADR-0013: Three clocks, including `steps` (`Δ_t ≡ 1`, no annualization)

## Context

Part I supports two clocks for expressing `Δ_t`: `trading` (trading days per year, 252) and `calendar`
(calendar days per year, 365.25), both used to annualize continuous-time parameters (ADR-0001). A
domain-neutral core (ADR-0012) also needs a clock for data that has no natural annualization at all — e.g.
synthetic or non-financial step-indexed sequences — without forcing an arbitrary "days per year" onto it.

## Decision

A third clock, `steps`, is added: `Δ_t ≡ 1` for every step, and every family's parameters are interpreted
per-step with no annualization performed or possible. Every family (`EmissionFamily`, `DurationFamily`,
`TransitionModel`) must work correctly under all three clocks. Durations are always reported in the dataset's
own clock units via `mean_sojourn(theta, nominal_dt)` (never a clock-specific `mean_sojourn_years`).

## Consequences

- `dt` is in clock units throughout regimekit (years under `trading`/`calendar`, steps — always `1` — under
  `steps`); no function assumes a specific clock's meaning of `dt` outside `utils/clock.py`.
- `utils/clock.py` is the only place literal `252`/`365.25` may appear, and the only place code may branch on
  the clock's *name* (alongside `data/calendar.py`); every other module is clock-agnostic by construction.
- A dataset that provides no true annualization information (`clock="steps"`) still gets exact parameter
  recovery guarantees identical in form to the annualized clocks, just expressed per step.

## Status

accepted; amended by ADR-0015 (clock registry)
