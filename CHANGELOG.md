# Changelog

Release notes are collected from `changelog.d/` fragments by [towncrier](https://towncrier.readthedocs.io)
(`<ticket>.<feature|bugfix|doc|removal|misc>.md`); see `towncrier.toml`.

<!-- towncrier release notes start -->

## 0.1.0 (2026-10-24)

### Features

- E2-T01: Add the `DataProvider` protocol, provider registry (`get_provider`), retry/throttle helpers and a local `csv` provider under `regimekit.finance.providers`.
- E3-T01: Add the `EmissionFamily` protocol and the GBM emission (per-state Gaussian log-returns scaled by
  `dt`, with linear drift and log-volatility covariate links).
- E6-T01: Add `regimekit.inference.simulate.simulate_paths`: deterministic Monte Carlo simulation of
  regime/phase paths and observations from any fitted (or hand-built) model, given a time grid and
  optional covariate paths. `segments_of` extracts per-path regime segments, and `simulate_dataset`
  wraps one simulated path as a synthetic `PreparedDataset` for tests and the recovery suite.
- E7-T01: Add the evaluation metric registry and metrics: log score (KDE and exact h=1 mixture), CRPS, PIT/interval coverage, RMSE/MAE, AIC/BIC/ICL, regime occupancy and persistence, balanced accuracy.
- E8-T01: Add `ModelStore` (`<name>.rkmodel/` bundle: manifest, resolved config, spec, safetensors parameters, prep stats, snapshots, fit report) and switch `decode`/`forecast`/`simulate --model` to it. New runtime dependency: `safetensors` (ADR-0011).
- E2-T02: Add the `yfinance` price provider (`regimekit.finance.providers.yfinance`) with retries, throttling, intraday window capping and a ToS log line; requires the `providers` extra.
- E3-T02: Add the Student-t emission (`student_t`): heavy-tailed per-state log-returns with per-state or tied degrees of
  freedom `nu > 2`, and the same location / log-scale covariate links as GBM.
- E6-T02: Add `regimekit decode CONFIG --model DIR [--augmented]` and the `inference.decode.decode(fit, data) ->
  PosteriorResult` API: filtered/smoothed regime probabilities, the Viterbi path and its segments, and
  AIC/BIC/ICL, written to `posterior.parquet`/`augmented.parquet`/`segments.json`/`criteria.json`. Also
  adds `model.build.build_model_spec(cfg, data) -> ModelSpec` (config -> model assembly, CLAUDE.md §3.3)
  and a temporary `params.npz`+`spec.json` model loader (`persistence.temp_model`) ahead of E8-T01's real
  bundle format.
- E8-T02: Add `regimekit.Project`, a Python facade that mirrors every CLI command (`fetch`, `fit`, `decode`, `forecast`, `backtest`, `simulate`, `compare`, `selftest`, `validate-config`) and writes nothing unless `write=True`, and the `regimekit fit` command (bundle, `fit_report.json`, `config.resolved.yaml`). The CLI now delegates to `Project`; unexpected exceptions exit with code 1.
- E4-T02a: Add `build_expanded(spec) -> StateSpace`: enumerates the expanded `(memory, regime, phase)` state
  space, its collapse matrix, jump-row/rate/entry-state index arrays, and the `uniform`/`free`
  initial distribution.
- E4-T02b: Add the per-step expanded transition matrix `A_t` (`StateSpace.transition_fn`): jump-mode
  stay/advance/exit scatter-adds and the generator-mode CTMC alternative, plus the `stationary`
  initial distribution (power iteration).
- E7-T02b: Add the walk-forward runner `regimekit.evaluation.backtest.run_backtest` (per-fold refit with warm start, per-origin Monte-Carlo forecasts, fold/metric/summary tables, pooled PIT KS), masked window padding, `estimation.fit(..., warm_start=)` and the config field `evaluation.backtest.n_sim`.
- E2-T03: Add the `stooq` daily-EOD fallback provider (`regimekit.finance.providers.stooq`) with `STOOQ_API_KEY` handling and a persisted daily request quota.
- E3-T03: Add the `DurationFamily` protocol and the phase-representation durations, `geometric` and shifted
  negative-binomial (`negbin`), with the log-rate covariate link and irregular `dt` handling.
- E5-T03: Add the MAP estimator: `regimekit.estimation.fit(spec, data, cfg, seed) -> FitResult` runs multi-start
  optimization (three initialization strategies -- rolling-volatility k-means, lattice/config-value
  jitter, scale-free random restarts) and returns the best converged fit, relabelling states by ascending
  volatility when the model has no built-in identifiability guard.
- E6-T03: Add covariate forecasters (`regimekit.inference.covariate_forecasters`): `freeze`, `scenario`,
  `bootstrap` (stationary block bootstrap of first differences) and `ar` (VAR(1), optionally diagonal)
  generate future covariate paths `(n_sim, h, K)` for multi-step forecasts, with a `ColumnMap` splitting
  the unified columns back into the transition/duration/emission blocks each model actually uses.
- E7-T03: Add the strategy layer: `regime_long_flat` and `regime_target_vol` rules, proportional costs, the strategy trace returned by `run_backtest(..., strategy=)`, the metrics `sharpe`, `max_drawdown`, `turnover` and `hit_rate`, and the config fields `evaluation.strategy.cap` and `allow_short`.
- E8-T03: Add `examples/`: the Appendix A configuration for `^GDAXI` and `^GSPC`, a synthetic look-alike data generator, and `examples/tutorial.ipynb` (the Python API end to end). CI replays the six-command flow offline on synthetic data and executes the notebook (`nox -s examples`); `tests/live/test_e8_t03_real_data.py` runs it on real data.
- E2-T04: Add the `fred` (macro series, `FRED_API_KEY`, 120 requests/minute limit, `"."` -> NaN) and `cboe` (VIX/VVIX daily history files, configurable URLs) covariate providers.
- E3-T04: Add the explicit duration families (`poisson`, `gamma`, `weibull`, `lognormal`, `nonparametric`) with an
  entry-time covariate link, the discrete `hazard` helper and the `D_max` truncation warning rule.
- E5-T04: Add post-fit diagnostics (`regimekit.estimation.report.diagnose`: regime occupancy, posterior mean
  duration, PIT calibration histogram and KS test, Ljung-Box residual autocorrelation, label-switching
  risk, restart dispersion, timings) and Hessian-based delta-method standard errors (`std_errors`).
  `evaluation.metrics.predictive` holds the shared PIT/Ljung-Box helpers for later reuse by backtest
  metrics; `EmissionFamily` gained a `cdf` method (GBM, Student-t) to support PIT.
- E7-T04: Add the `backtest` and `compare` commands: walk-forward metrics, fold table and strategy trace as parquet, a Markdown `report.md` (header, model, fit, diagnostics, summary, strategy, timings), and a multi-config comparison table that refuses mixed data snapshots.
- E8-T04: Complete the documentation (install, quickstart, troubleshooting, configuration reference generated from the pydantic schema by `scripts/gen_config_reference.py`, math pages with symbol-to-code tables, API reference) and add the release tooling: `towncrier.toml`, `python-semantic-release` configuration, the tag-triggered `release.yml` workflow and the `docs/releases/0.1.0.md` checklist.
- E6-T04b: Add `MonteCarloForecaster`: Monte-Carlo forecasts from the filtered posterior (expanded and explicit engines) with regime probabilities, cumulative-return quantiles and moments, plus `future_grid` for weekday forecast timestamps.
- E2-T05: Add the local parquet cache (`regimekit.data.cache`): reproducible snapshot ids, kept generations, `max_age_days`, pinned snapshots and offline replay for every provider.
- E3-T05: Add the `TransitionModel`/`JumpTransition`: multinomial-logit jump probabilities with a
  reference-target parametrization, memory-tuple layout for higher-order chains, and the exact CTMC
  generator-mode alternative (`transition_matrix_generator`, `expm`).
- E6-T05: Add `regimekit forecast` and `regimekit simulate`: long-format `forecast.parquet` / `simulation.parquet` outputs, optional `forecast_samples.parquet`, `forecast.json`, a `--scenario` CSV for future covariates, and `read_forecast`.
- E5-T05a: Add the parameter-recovery suite (`regimekit.estimation.recovery`): six scenarios (GBM/negative-binomial,
  transition and duration covariates, order-2 memory, explicit-duration gamma sojourns, a fixed-drift
  lattice) simulate from a known truth, MAP-fit, and check recovery against the concept's tolerances over
  5 seeds each. `tests/recovery/test_e5_t05a_recovery.py` runs it nightly (`-m slow`); see
  `docs/guide/validation.md`.
- E5-T05b: Add `regimekit selftest [--full]`: fits simulated data twice on the installed backend (`T=1500`, two restarts), prints the backend, a truth/estimate/tolerance table and the determinism result, and exits 4 unless all fits are finite and bitwise reproducible. The recovery rows are informational at this size.
- E2-T06: Add the master calendar (`regimekit.data.calendar`): observation grid from the primary instrument, `Δ_t` under the `trading`, `calendar` and `steps` clocks, and look-ahead-free covariate alignment with `lag >= 1` and `max_staleness`.
- E3-T06: Add `regimekit.model.lattice`: fixed-parameter state lattices from parameter grids, an ordering-constraint
  helper, label-order post-sorting of fitted parameters and an identifiability-guard check.
- E4-T06b: Add `estimation.chunk_size` (default `null` = automatic) to bound the memory of the explicit-duration
  engine, plus its segmental Viterbi decoder.
- E2-T07: Add `regimekit.data.prepare.Preparer`: log-return observations, covariate transforms, train-window-only standardization, `ffill`/`drop`/`error` missing policies and `PrepStats` JSON serialization.
- E4-T07: Add the opt-in `estimation.parallel_scan` flag (associative-scan smoothing for the expanded engine) and
  the committed benchmark baseline with GPU-skipping test variants.
- E2-T08: Add the `regimekit fetch` command: cached provider fetch, preparation, `prepared.parquet` / `prep_stats.json` / `snapshots.json`, and a data-quality report (`quality.json` / `quality.md`) with gaps, duplicates, non-positive prices, 8-sigma returns, adjustment discontinuities, covariate staleness and dropped rows.
- E2-T09: Add the optional `tiingo` US end-of-day price provider with `TIINGO_API_KEY` handling and persisted client-side free-tier limits (50/hour, 1000/day, 500 symbols/month).

### Bug fixes

- E7-T02b: `estimation.fit` now initializes from observed rows only (rows with `mask=False` no longer bias `rollvol_kmeans`/`random` starts; unchanged without masked rows). `build_expanded`/`build_explicit` are memoized per spec and `ParamManager` hashes by value, so repeated fits no longer recompile; `Preparer.fit` accepts an optional `train_start`.
- E5-T05a: Fix `estimation.init.random`'s `transition.pi_rows` initialization: it drew one logit column too
  many, so `ParamManager.flatten` crashed on any model with 3+ regimes and no transition covariates
  (silently tolerated at `N=2`, where the resulting free-coordinate count is 0). Found while adding the
  parameter-recovery suite's `N=3` scenarios.
