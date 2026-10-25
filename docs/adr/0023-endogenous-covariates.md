# ADR-0023: Endogenous covariates as emission channels

## Context

Covariates such as the VIX enter regimekit 0.1.0 as **exogenous** inputs: `PreparedDataset.cov_transition`,
`cov_duration` and `cov_emission` are lagged by at least one step (no look-ahead, `data/calendar.py`) and
enter the transition, duration and emission links. Their own dynamics are not modelled, so a forecast must
simulate them separately (`CovariateForecaster`: freeze, bootstrap, AR, scenario; E6-T03), and the regime
posterior ignores whatever information the covariate carries about the regime *at* time `t`.

The alternative is to treat such a series as an additional **observation channel**: the observation vector
grows from `(M,)` returns to `(M + K,)` channels, and the regime generates all of them. This ADR records the
design and what it would change. Nothing is implemented; this is the E9-T04 spike (status `proposed`, the
owner decides).

What exists after E9-T01 and E9-T03 and shapes the options:

- E9-T01: `ProductEmission` sums independent per-series log-densities with a **per-series mask**, and the
  master grid is the primary's timestamps with other series masked where missing. It wraps **one** family;
  channels with different families (GBM for returns, an AR(1) Gaussian for log-VIX) need per-channel
  parameter names, which E9-T01 left as a follow-up.
- E9-T03 / ADR-0022: emissions with recurrent state run in a scan before the forward pass when the family
  reports a carry. An in-regime AR(1) channel `x_t | i ~ N(c_i + phi_i x_{t-1}, s_i^2)` depends on the
  *observed* `x_{t-1}`, not on the regime path, so it needs no carry at all (a lagged column of the data), but
  a stateful variant (AR on the regime-specific residual) would use the same mechanism.

## Decision (proposed)

1. **Channels, not a new contract.** A covariate configured with `role: channel` (default `exogenous`, which
   keeps every 0.1.0 config valid) is appended as extra columns of `PreparedDataset.observations`, with its
   own column of `mask`. The `PreparedDataset` field set stays frozen (CLAUDE.md section 3.1). The channel
   layout (`channel_kind: tuple[Literal["return", "level", ...], ...]`, one entry per column) lives in
   `BuildInputs` / the model bundle, next to `n_series`, not in the dataset. Data layer: a channel is aligned
   to the primary grid with the same as-of rule as covariates (last print at or before `tau_t`), but with
   **lag 0** (it is observed at `t`), then transformed (`log` level or `diff` as configured) and masked where
   stale beyond `max_staleness`. `Preparer` standardization statistics are still train-window only.
2. **Heterogeneous product emission.** `ProductEmission` gains `families: tuple[EmissionFamily, ...]` with
   one family per channel group; parameter names get a channel prefix (`emission.<channel>.<key>`, for
   example `emission.log_vix.phi`), `param_specs` is concatenated, and `log_prob_masked` sums the families'
   per-channel densities. New families: `GaussianAR1Family` (levels, regime-wise intercept, persistence and
   scale) as the first endogenous-channel family. `EmissionFamily.support` already lets the builder check the
   column kind.
3. **Covariate links remain possible.** A channel may *also* drive transition/duration links with a lag, as
   an exogenous covariate does today (the column is then read twice); that is a configuration choice, not a
   new mechanism.
4. **Forecasting.** `MonteCarloForecaster` samples channels jointly with returns: the sampler's per-step
   emission draw returns `(M + K,)`, and an AR channel conditions on its own previous draw, so the sampler
   scan carries the last channel values. `CovariateForecaster` is then needed only for the remaining
   exogenous covariates. `ForecastResult` gains channel quantiles next to return quantiles; the aggregation
   of cumulative returns uses the return columns only.
5. **Estimation and inference.** No engine change: the emission table is `(T, N)` regardless of channel
   count. Gradients and the objective are unchanged. `ExpandedHMMEngine` and the explicit engine both
   consume the table.

## Interface diff

| Area | Today | Proposed |
|---|---|---|
| `PreparedDataset` | `observations (T, M)` | `observations (T, M + K)`, `mask (T, M + K)`; no new fields |
| Config | `data.covariates[i].use_in` | plus `role: exogenous \| channel` (default `exogenous`), `channel.transform`, `channel.family` |
| `BuildInputs` / bundle | `n_series` | plus `channel_kind` per column |
| `ProductEmission` | one base family | tuple of families, channel-prefixed parameters |
| `EmissionFamily` | `support` | new families declare `support`; no protocol change |
| Forecasting | `CovariateForecaster` for all covariates | only for exogenous ones; channels sampled jointly |
| Metrics | log score over returns | log score over returns marginal (see caveats) |

## Identifiability and modelling caveats

- **Channel dominance.** A channel with strong regime dependence and a small scale (log-VIX is smooth) can
  dominate the likelihood, so the regimes become "VIX regimes" and the return fit follows. Mitigations:
  `per_series` style tying, a tempering weight per channel (design option, not proposed here), and reporting
  the per-channel log-likelihood contribution in the fit report.
- **Likelihood comparability.** Models with and without channels have different data and cannot be compared
  by AIC/BIC/ICL; return-only scores (the marginal of returns, obtained by dropping the channel density from
  the same fitted model) are required for walk-forward comparison with exogenous-covariate models.
- **Contemporaneous dependence.** Independent channels given the regime cannot express a within-regime
  return/VIX correlation (leverage effect); correlated emissions are out of scope of `ProductEmission`.
- **Non-stationary levels.** AR(1) on a level requires stationarity within regime (`|phi| < 1`, bounded
  domain); unit-root-like series should enter as `diff`.
- **No look-ahead** remains guaranteed only if channel values enter `log f` at their own time and nothing
  else; links that read a channel must use the lagged copy.

## Migration path

1. Land the heterogeneous `ProductEmission` and channel-prefixed parameters (no behaviour change for
   single-family models).
2. Data layer: `role: channel` alignment with lag 0 and per-column masks, plus `channel_kind` in
   `BuildInputs` and the bundle manifest (`schema_version` bump, migration hook from E1-T02b).
3. `GaussianAR1Family` with recovery tests (a two-channel version of the E9-T01 recovery scenario).
4. Joint sampling in the forecaster and the return-marginal log score.
5. Docs: a guide section "exogenous versus endogenous covariates".

## Decision on scheduling

Recommend scheduling it after 0.1.x stabilises and after ADR-0022 is accepted, as one epic of roughly four
medium tickets (steps 1 to 4 above), because it removes the largest approximation in the forecasting path
(covariate forecasters) at the price of a model whose likelihood is no longer comparable with the current one.
It should not start before the owner has reviewed the channel-dominance risk on real data.

## Consequences

- Nothing changes in 0.1.x. Existing configs, bundles and results are unaffected.
- The design reuses E9-T01 (per-channel masks, primary-defined grid) and ADR-0022 (carry, if a stateful
  channel family is wanted).

## Status

proposed
