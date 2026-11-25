# Status

Progress ledger of the `hsmmkit` repository (restarted by E10-T08). The Part I ledger (E0–E9, `regimekit` 0.1.0 /
0.2.0) and the history of E10-T01…T07b live in the `RegimeDetectionAndForecast` repository.

## Tickets
<!-- Status: todo | in-progress | done | blocked | needs-owner | other-repo -->
| ID | Title | Size | Phase | Deps | Status | Commit | Notes |
|---|---|---|---|---|---|---|---|
| E10-T01 | Split guardrails: neutral-name scanner, `core-only` job, v0.2.0 golden values, ADR-0014 | S | 8 | Part I complete (E8-T04, E9-T01…T04) | done | 7fea2fd | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T02a | Component registries | M | 8 | E10-T01 | done | 6d5ea41 | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T02b | Config extension, `FinanceConfig`, schema v2 migration, ADR-0016 | M | 8 | E10-T02a | done | d032ae8 | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T03 | Clock registry; `trading` to finance; ADR-0015 | M | 8 | E10-T02b | done | c9ba730 | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T04a | Generic `ObservationSet`, `Source` protocol, numpy/pandas/csv/parquet sources, `source_contract` | M | 8 | E10-T02b | done | 9546ff9 | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T04b | xarray source; providers as source kind `provider`; parquet cache and snapshots moved to finance | M | 8 | E10-T04a | done | e106e04 | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T05 | Generic `Preparer`, observation-transform hook, supports and validation, quality split | M | 8 | E10-T03, E10-T04a | done | 6922464 | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T06 | Core `gaussian` emission (diagonal, `M ≥ 1`, `scale_by_dt`, `ito_correction`); `gbm` as a finance alias | M | 8 | E10-T02b | done | e9f3f3b | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T07a | Neutral init naming, walk-forward origin hooks, strategies to finance, bundle v2 | M | 8 | E10-T02b, E10-T03 | done | 3278c1a | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T07b | CLI app factory and `Project`/`FinanceProject` split; neutral-name allow-list empty | M | 8 | E10-T07a, E10-T04b, E10-T05, E10-T06 | done | 4252854 | done in RegimeDetectionAndForecast before the extraction (history: `git log`); commit refers to that repository |
| E10-T08 | Extract `hsmmkit` repository; release 0.1.0 | M | 9 | E10-T07b | done | 6a2838d | Extracted with history (three filter-repo passes), rename commit, docs/CLAUDE.md/STATUS, pushed as `main`. AC1 not attainable (source history already squashed per ticket, see Follow-ups); AC2 full suite 809 passed + known XLA aborts that pass alone (gradient student_t o2, segmental Viterbi, parallel scan), `nox` lint/types/docs equivalents green locally; AC3 `test_e10_t08_*`; AC4 `downstream` job in `nightly.yml` (red until E10-T09); AC5 wheel built, installed in a fresh venv, `hsmmkit validate-config` on a core config OK; tag/release/TestPyPI are owner actions. Found by the smoke install and fixed here: `beartype` (dev dependency) was imported at import time, now lazy. |
| E10-T09 | `regimekit` on top of `hsmmkit`; release 0.2.0 | M | 9 | E10-T08 | other-repo |  | RegimeDetectionAndForecast |
| E10-T10 | Complete schema with feature gates; JSON Schema; presets registry | M | 10 | E10-T08 | done | 37aa500 | See Spec fixes and Needs CI. Review: no blockers; findings fixed. |
| E10-T11 | Lazy optional imports, extras, import-time test | S | 10 | E10-T08 | done | bed5041 | AC1 `test_e10_t11_import_pulls_no_extra_package`; AC2 `extras` matrix job in `pr.yml` (runs only in CI: Needs CI) + `test_e10_t11_missing_module_names_the_extra`; AC3 `test_e10_t11_extras_table` (`mcmc` == `bayes`). The import-time test compares the cumulative time of the `hsmmkit` row (the ticket text says self-time; deviation) and is skipped under xdist. Import-time baseline recorded on a 4-core cloud VM (1.02 s cumulative, limit x1.3): machine dependent, may need a refresh on the CI runner. `optional_import(module, *, extra)` is keyword-only now. |
| E11-T01 | `family_contract` suite | M | 10 | E10-T08 | done | 1a89911 | branch feat/E11-T01-family-contract. Decisions: the suite is a pytest *module* (`test_family_contract` + `pytest_generate_tests` are imported into a test module of the user: a pytest plugin cannot contribute test functions); `moments` implemented for `gaussian` and `student_t` (product: none); checks 4/5/7 skip where not applicable; `docs/guide/extending.md` moved to `extending/index.md` to host `extending/families.md`. |
| E11-T02 | Gaussian full covariance | M | 10 | E11-T01 | done | a457275 | branch feat/E11-T02-gaussian-full-covariance. Decisions: `chol` is stored as the packed (row-major) lower triangle `(N, M(M+1)/2)` in the `cholesky` domain (softplus diagonal); `m_step` jitter is relative (`1e-10 mean(diag Sigma)`, the absolute 1e-10 of the ticket breaks the 1e-8 stationarity check on small-scale data); rows with any missing series are dropped under a full covariance (no per-series masking); scalar `ParamCfg` priors on `chol` act on the diagonal. |
| E11-T03 | Multivariate Student-t | S | 10 | E11-T02 | in-progress |  | branch feat/E11-T03-mv-student-t. Decision: the joint path is selected by `covariance: full` or the new parameter names (`mean`/`scale`/`chol`/`nu`), not by `M > 1` alone, so Part I `m`/`s` configs with `M > 1` keep the independent product family. |
| E11-T04 | Poisson, Bernoulli, Categorical | M | 10 | E11-T01 | todo |  |  |
| E11-T05 | MS-AR(p) | M | 10 | E11-T02, E10-T05 | todo |  |  |
| E11-T06 | Per-state finite mixture | M | 10 | E11-T02 | todo |  |  |
| E11-T07 | OU and Merton jump-diffusion | M | 10 | E11-T06, E10-T09 | other-repo |  | RegimeDetectionAndForecast |
| E11-T08 | Joint/product emission composition | M | 10 | E11-T02, E11-T04 | todo |  |  |
| E11-T09 | Correlated multi-primary finance configuration | S | 10 | E11-T08, E10-T09 | other-repo |  | RegimeDetectionAndForecast |
| E12-T01a | `SequenceBatch`, padding, buckets; ADR-0021 | M | 11 | E10-T10 | todo |  |  |
| E12-T01b | Batched engines and unpadding | M | 11 | E12-T01a | todo |  |  |
| E12-T02 | Batch MAP, per-group initial distributions, batched simulate | M | 11 | E12-T01b | todo |  |  |
| E12-T03 | Transition topology and absorbing states | M | 11 | E10-T10 | todo |  |  |
| E12-T04 | Left-censoring in the explicit engine | S | 11 | E10-T10 | todo |  |  |
| E12-T05 | Per-state durations, duration-dependent jumps | M | 11 | E12-T03 | todo |  |  |
| E12-T06 | Label/boundary constraints, label init; ADR-0019 | M | 11 | E12-T01b | todo |  |  |
| E13-T01a | Expanded-engine E-step accumulators | M | 12 | E12-T01b | todo |  |  |
| E13-T01b | Explicit-engine segment posteriors `η` | M | 12 | E12-T01b | todo |  |  |
| E13-T02a | `EMEstimator`: closed-form M-steps; ADR-0018 | M | 12 | E13-T01a, E11-T04 | todo |  |  |
| E13-T02b | ECM, tempering, restarts, parity | M | 12 | E13-T02a | todo |  |  |
| E13-T03a | GEM: jumps, nonparametric durations, censoring | M | 12 | E13-T01b, E12-T04 | todo |  |  |
| E13-T03b | GEM: parametric durations, emissions, parity | M | 12 | E13-T03a, E13-T02a, E12-T05 | todo |  |  |
| E13-T04 | FFBS (both engines) | M | 12 | E12-T01b | todo |  |  |
| E13-T05a | VB: conjugate updates, structured `q(s)`, ELBO | M | 12 | E13-T01a, E11-T04 | todo |  |  |
| E13-T05b | VB: pruning, VB-EM hybrid, parity | M | 12 | E13-T05a, E13-T02b | todo |  |  |
| E13-T06a | MCMC: NUTS, `bayes` extra, ArviZ; ADR-0024 | M | 12 | E12-T02, E10-T11 | todo |  |  |
| E13-T06b | MCMC: blocked Gibbs, relabelling, agreement | M | 12 | E13-T06a, E13-T04, E13-T05a | todo |  |  |
| E13-T07 | Estimator parity suite, docs, `compare --estimators` | M | 12 | E13-T02b, E13-T03b, E13-T05b, E13-T06b | todo |  |  |
| E13-T08 | Online EM spike; ADR-0026 | S | 14 | E13-T02b, E15-T01 | todo |  |  |
| E13-T09 | HDP-HSMM spike; ADR-0027 | S | 12 | E13-T06b | todo |  |  |
| E14-T01 | `Splitter` interface, `SequenceKFold`, walk-forward re-base | M | 13 | E12-T02 | todo |  |  |
| E14-T02 | Grid model selection | M | 13 | E14-T01, E13-T07 | todo |  |  |
| E14-T03 | Segmentation and clustering metrics; ADR-0020 | M | 13 | E10-T08 | todo |  |  |
| E14-T04 | Parametric bootstrap | M | 13 | E12-T02 | todo |  |  |
| E14-T05 | Pseudo-residuals, posterior predictive checks | M | 13 | E13-T04, E11-T04 | todo |  |  |
| E14-T06 | Extended recovery scenarios, `hsmmkit selftest` | M | 13 | E13-T02b, E13-T03b, E13-T05b, E12-T06, E11-T05 | todo |  |  |
| E15-T01 | `OnlineFilter` | M | 14 | E10-T08 | todo |  |  |
| E15-T02 | Fixed-lag smoothing | M | 14 | E15-T01 | todo |  |  |
| E15-T03 | Alerts and online forecast | S | 14 | E15-T02 | todo |  |  |
| E15-T04 | `hsmmkit stream` CLI, `Project.online()` | S | 14 | E15-T03 | todo |  |  |
| E15-T05 | (v1.x) Online EM | M | 14 | E13-T08 | todo |  |  |
| E16-T01 | scikit-learn wrapper | M | 14 | E12-T02, E12-T06, E10-T11 | todo |  |  |
| E16-T02 | pandas/xarray conversion, `hmmlearn` import | M | 14 | E11-T02, E11-T04, E10-T11 | todo |  |  |
| E16-T03 | Plotting | M | 14 | E16-T02 | todo |  |  |
| E16-T04 | Entry points, scaffold, `plugin` CI; ADR-0017 | M | 14 | E11-T01 | todo |  |  |
| E16-T05 | Presets completion, core tutorials | M | 14 | E10-T10, E11-T05, E12-T05, E12-T06, E13-T03b, E13-T04, E14-T01, E14-T03, E14-T04, E14-T05, E15-T03, E16-T03 | todo |  |  |
| E16-T06 | `regimekit` as entry-point plug-in, finance presets | S | 14 | E16-T04, E11-T07 | other-repo |  | RegimeDetectionAndForecast |
| E17-T01 | Documentation restructure | M | 15 | E16-T05, E13-T07 | todo |  |  |
| E17-T02 | PyPI + conda-forge distribution | S | 15 | E10-T11 | todo |  |  |
| E17-T03 | ADRs accepted, deprecation policy, benchmarks; ADR-0025 | S | 15 | E13-T07, E13-T09, E14-T02, E14-T04, E14-T05, E14-T06, E15-T04, E16-T01, E16-T03, E16-T04, E16-T05 | todo |  |  |
| E17-T04 | Release `hsmmkit` 0.5.0 | S | 15 | E17-T01, E17-T02, E17-T03 | todo |  |  |
| E17-T05 | Finance docs and release `regimekit` 0.5.0 | S | 15 | E17-T04, E16-T06, E11-T09 | other-repo |  | RegimeDetectionAndForecast |

## Handoff
Extracted from `RegimeDetectionAndForecast` at the E10-T07b merge commit `42528543b9f0a4b7856e72c809016aee5b12a3e9`
(E10-T08). Next eligible tickets: E10-T10 (complete schema with feature gates, presets registry), E10-T11 (lazy
optional imports, extras), E11-T01 (`family_contract` suite), E14-T03, E15-T01 (all depend on E10-T08 only).
Release 0.1.0 is prepared (tag `v0.1.0` is an owner action, see "Blocked / needs owner"). `RegimeDetectionAndForecast`
(E10-T09) installs `hsmmkit` 0.1.0 from that release.

## Blocked / needs owner

## Spec fixes
- E11-T03: the ticket selects the joint Student-t "when `M > 1` or `covariance: full`"; implemented as `covariance: full` or the new parameter names `mean`/`scale`/`chol`/`nu`, so Part I configs (`m`/`s`/`nu`) with `M > 1` keep the independent product family (Part I behaviour, v0.2.0 compatibility). Owner may prefer the ticket reading; changing it is a one-line dispatch change.
- E10-T10: Appendix H names metrics that exist only from E14-T03 (`boundary_f1`, `hausdorff`, `covering`, `hamming_matched`, `ari`, `nmi`) and uses scalar per-state `init` / `init_mean_sojourn_steps`, `init: labels` and a `chol` parameter; to make it validate (AC1) the schema accepts the planned metric names (`gates.PLANNED_METRICS`, gated at build, listed in the JSON-Schema enum) and `ParamCfg.init` / `init_mean_sojourn_steps` accept a scalar (broadcast to all states at build) or `"labels"`. `evaluation.backtest` and `forecast.covariate_forecaster` became optional (`ProjectConfig.backtest_cfg()` raises `ConfigError(path="evaluation.backtest")`; default forecaster `freeze`). The generated reference stayed in `docs/guide/config.md` (the ticket says `configuration.md`). The "family support vs data support" check stays where E10-T05 put it (the `Preparer` checks the data against the family support; `BuildInputs` carries no support).

## Findings for the owner

## Follow-ups
- E11-T01 gaps (review): no built-in family has `m_step` yet (the Gaussian M-step lives in `estimation/em.py`), so check 7 only exercises `m_step` on the test family; the zero-data posterior-equals-prior check needs the `PriorFamily` API of E11-T04/E13-T05a (check 7 only asserts `update` exists); check 2 passes silently without `moments` (support only); `pytest_generate_tests` parametrizes over every registered case, so a second test module importing `test_family_contract` repeats them (E11-T02 should add a per-module filter if it adds cases elsewhere).
- About 50 test modules of `RegimeDetectionAndForecast` needed the finance package and were not carried over.
  The core-relevant scenarios they covered (fit/backtest/bundle/CLI round trips, the product emission on two
  series, lattice and padding tests, strategy-free parts of the report) should be re-created with `gaussian`
  fixtures and core configs by the tickets that touch those areas (E10-T10, E11-T01, E12-*, E14-T06).
- The upstream history before E7-T02b was already squashed in `RegimeDetectionAndForecast`; `git log --follow`
  of `engine/expanded_hmm.py` therefore starts at E5-T05b (acceptance criterion 1 of E10-T08 could not name the
  E4-T03a/b commits).

## Needs CI / network / GPU
- E10-T11: the `extras` job (matrix none/plot/sklearn/bayes/xarray) and the import-time baseline (`benchmarks/import_time_baseline.json`, machine dependent) were not run on a CI runner; `uv.lock` was regenerated with `uv lock` for the new extras.
- E10-T10 AC3 (second half): `--config-class regimekit.config.FinanceConfig` validates Appendix A v2 and lists `gbm` needs `regimekit` installed (test `test_e10_t10_regimekit_finance_config_schema_when_installed` is skipped here; runs in `downstream`).
- E10-T10 follow-up for E12-T05: `per_state[i].params` get the generic param checks (already via `_named_param_cfgs`); `model.left_censored` with `engine: expanded` is accepted silently until E12-T04 decides.
- GitHub Actions has not run in this repository: `pr.yml`, `nightly.yml` (incl. `downstream`), the coverage gate
  (>= 90 % total, >= 85 % per package), the benchmark baseline comparison and the cross-machine golden check.
- The `downstream` job is expected red until E10-T09 has merged in `RegimeDetectionAndForecast`.

## Measurements
