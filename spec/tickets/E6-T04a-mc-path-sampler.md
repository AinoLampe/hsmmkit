# E6-T04a — Monte-Carlo forecast path sampler from the filtered augmented posterior

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-I04, G2 | M4 | §4.9 algorithm steps 1–2 |

## Goal
Sample `n_sim` future expanded-state and return paths of horizon `h` starting from `softmax(α_t)`.

## Context (self-contained)
```
1. α_t ← filtered augmented posterior at the origin t (last row of Posterior.log_alpha; explicit engine: convert (regime, elapsed) → an expanded-state start via the phase family is not possible → for the explicit engine sample (regime, elapsed) and continue with the explicit hazard; see below)
2. for each of n_sim paths:  s ← Categorical(softmax(α_t))
     for k = 1..h:  z_{t+k} ← covariate path row k−1 (E6-T03; row 0 known)
                    A_{t+k} ← transition_fn(θ, z^π_{t+k}, z^d_{t+k}, Δ_{t+k});  s ← Categorical(A_{t+k}[s])
                    y_{t+k} ← emission.sample(regime_of[s], Δ_{t+k}, z^e_{t+k})
```
- `forecast_paths(key, theta, spec, space, log_alpha_T (S,), dt_future (h,), cov_paths (n_sim, h, K) | None, colmap) -> (states (n_sim, h), regimes (n_sim, h), returns (n_sim, h, M))`: reuse `simulate_paths` (E6-T01) with `init_state` sampled per path and per-path covariates (vmap over the covariate path axis).
- `dt_future`: from `calendar.future_grid(times[-1], h, clock, bars_per_day)` — next `h` trading days (weekday calendar, no holiday calendar in v1; document) → `horizon_times (h,)`, `Δ` per clock; under `clock: steps` (Part II §15.7 amendment 2) `Δ ≡ 1` and `horizon_times` continue the grid with its median timestamp spacing.
- Explicit engine origin: sample `(regime j, elapsed d)` from `elapsed_posterior[T]`, then simulate with the explicit hazard `h_j(d)` (E3-T04 `hazard`) step by step until exit, then draw a fresh duration from `log_pmf` for the next regime — implement as a separate small scan `forecast_paths_explicit`.

## Creates
- `src/regimekit/inference/forecast.py` (path sampling part), `data/calendar.py::future_grid`
- `tests/unit/test_e6_t04a_paths.py`

## Acceptance criteria
1. `h=1`: empirical regime distribution of `regimes[:, 0]` within 3 SE of `(softmax(α_T) A_{T+1}) C`; mean/variance of `returns[:, 0]` within MC error of the analytic mixture.
2. Shapes `(n_sim, h)` / `(n_sim, h, M)`; deterministic per key.
3. `future_grid` skips weekends and yields `Δ = 1/252` under the trading clock, calendar gaps under the calendar clock, and `Δ = 1` under the steps clock.
4. Explicit-engine variant: with geometric pmf, regime-probability paths agree with the expanded variant within MC error.

## Out of scope
Aggregation (E6-T04b); CLI.

## Connections
- Depends on: E6-T01, E6-T03, E4-T04
- Blocks: E6-T04b
- Relates to: E4-T06b (explicit origin)
