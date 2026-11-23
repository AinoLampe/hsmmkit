# E15-T03 — Alert rules with hysteresis and remaining-duration triggers; structured events; forecasting from the online state

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | FR-ON03 | M12 | Part II §14.11 (alerts, forecasting from the online state), §15.4 (`online.alerts`); Part I §4.9, E4-T04, E6-T04a/b |

## Goal
Turn the live posterior into actionable, de-bounced events and allow forecasts from the current stream state.

## Context (self-contained)
- Config `online.alerts`: list of rules
  - `{name, regime: j, p_on, p_off, use: filtered | smoothed = filtered}` — fires `on` when `P(regime = j) ≥ p_on` while inactive, `off` when `< p_off` while active (`p_off < p_on`, validated in E10-T10);
  - `{name, regime: j, remaining_duration_below: d, p_min = 0.5}` — fires when `P(regime = j) ≥ p_min` and the expected remaining duration of the current `j`-segment (E4-T04 augmented posterior: phase marginals → `E[remaining] = Σ_c P(phase = c | regime = j) · E[remaining | phase c]`, with `E[remaining | c] = (r_j − c + 1) / a_j(t)` steps for NB) drops below `d` (clock units), with `off` when it rises above `d · 1.2` (fixed hysteresis factor, documented).
- Alert state lives in `OnlineState.alerts` (active flags, `(n_rules,)` bool) and is saved with it.
- Events: `AlertEvent(time, t, rule, kind: "on" | "off", regime, value, threshold)` returned from `update` as a list (Python side, after the jitted step; rules evaluated in plain Python on `(N,)` vectors — negligible cost). JSON-serializable (`to_dict`).
- Forecast from the online state: `OnlineFilter.forecast(state, horizon, n_sim, covariate_forecaster, seed) -> ForecastResult` uses `softmax(state.log_alpha)` as the origin distribution and the E6-T04a/b sampler/aggregation (covariate forecasters fitted on the training data stored in the fit's `PrepStats`/bundle).

## Creates
- `src/hsmmkit/online/alerts.py`, `online/filter.py` (forecast), `tests/unit/test_e15_t03_alerts.py`, `tests/integration/test_e15_t03_online_forecast.py`

## Acceptance criteria
1. A synthetic stream with one regime switch (calm → turbulent → calm) produces exactly one `on` and one `off` event for a hysteresis rule (`p_on = 0.8, p_off = 0.5`) and no flapping on a noisy probability path that crosses `p_on` repeatedly without reaching `p_off`.
2. Remaining-duration rule fires within ±3 steps of the analytic crossing on a deterministic NB example.
3. Forecast from the online state equals the offline `forecast` at the same origin with the same seed (bitwise samples).
4. Events round-trip through JSON; alert flags persist through save/load.

## Tests
Unit (rule logic), integration (forecast parity).

## Out of scope
Notification delivery (email, webhooks) — consumers of the JSONL stream.

## Connections
- Depends on: E15-T02
- Blocks: E15-T04, E16-T05
- Relates to: E4-T04, E6-T04a/b
