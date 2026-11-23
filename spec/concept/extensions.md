---

# Part II — Extensions toward a general-purpose HSMM library (`hsmmkit`) and the finance application (`regimekit`)

**Addendum to "regimekit — Concept and Implementation Plan"**

| Field | Value |
|---|---|
| Version | 1.2 (extension addendum to concept baseline 1.1) |
| Date | 2026-11-22 (1.1-draft: 2026-09-11) |
| Status | Approved for ticket derivation (tickets E10–E17, overview `10-OVERVIEW-part2-build-order-and-conventions.md`) |
| Owner | Aino |
| Relationship to Part I | Sections 11–20 and Appendices F–I continue the numbering of Part I. Part I (concept v1.1 = v1.0 + the §15.7 amendments) is fully implemented in the `RegimeDetectionAndForecast` repository (github.com/AinoLampe/RegimeDetectionAndForecast; Python package, distribution and CLI `regimekit`) and released as `regimekit` 0.1.0 and 0.2.0 (0.2.0 adds E5-T06 and E9-T01; E9-T02…T04 are spikes/ADRs). Part II defines the extraction of the domain-neutral core into a separate library `hsmmkit` and that library's releases 0.1.0–0.5.0, plus the finance-side follow-ups in `regimekit`. |
| Scope | Two-repository split (library + finance application); domain-neutral core; general emission, duration and transition families; multiple sequences; alternative estimators (EM, VB, MCMC); supervision; online inference; interoperability; plug-in system |

### Changes in 1.2 (2026-11-22)

1. **Two repositories instead of one wheel.** The domain-neutral core becomes the library **`hsmmkit`** in a new repository `hsmmkit`, extracted from `RegimeDetectionAndForecast` with git history. The existing repository `RegimeDetectionAndForecast` (package `regimekit`) stays the finance project and depends on `hsmmkit` (D12, D22 revised; new D23). Section 15.8 defines the move map and the decouple-then-extract sequence.
2. **No `profile` key.** Domain packages extend `hsmmkit` through registries and config subclassing (new D24; FR-G05 rewritten). The Part I `profile: finance` key disappears with regimekit's `schema_version: 2`.
3. **Clocks are registered.** `hsmmkit` ships `steps` (default) and `calendar`; the `trading` clock moves to `regimekit` (D13 revised, FR-G02).
4. **New requirements** FR-G08 (dependency direction), FR-G09 (history- and behaviour-preserving extraction), NFR-18 (cross-repository compatibility).
5. **Part I state.** The §15.7 amendments are applied (regimekit 0.1.0); `regimekit` 0.2.0 (2026-11-01) added E5-T06 and E9-T01, so 0.2.0 is the baseline for the split and the next `regimekit` release is 0.3.0. E5-T06 (EM for GBM) is the seed of E13-T02, E9-T01 (`ProductEmission`) the seed of E11-T08, E9-T02 finalized ADR-0008 (so the former Part II ADR "0018 factorial" is dropped).
6. **ADR numbering.** ADR-0011 is safetensors (E8-T01); 0012 (core/profile split) and 0013 (steps clock) exist from E0-T04; 0022 (in-scan emissions, E9-T03) and 0023 (endogenous covariates, E9-T04) exist from the E9 spikes. Part II ADRs use the reserved 0014–0021 and then 0024–0027 (Section 16.7); 0014 supersedes 0012, 0015 amends 0013; further ADRs take numbers from 0028.
7. **Epic E10 rewritten** (decouple → extract → consume); E11, E16 and E17 gain `RegimeDetectionAndForecast`-side tickets; every ticket names its repository (Section 17.2).
8. **Correction (§14.2, FR-EM04).** Hamilton's mean-adjusted MS-AR depends on regimes at previous *time steps*; the segment-memory builder of Part I §4.5 cannot express this. It is realized by a time-step memory expansion (geometric durations only).
9. **Correction (NFR-15, Appendix G).** Fixed-lag smoothing yields `P(s_{t−L} | y_{1:t})`; it equals the offline smoother on the prefix `y_{1:t}`, not the full-data smoother.

---

## 11. Motivation and scope of the extensions

### 11.0 How to use this part

The rules of Section 0 apply unchanged. New requirements use the prefixes `FR-G` (generalization), `FR-EM` (emissions), `FR-DU` (durations/transitions), `FR-SQ` (sequences), `FR-ES` (estimation), `FR-SV` (supervision), `FR-ON` (online), `FR-IO` (interoperability), `FR-VZ` (visualization), `FR-PL` (plug-ins), `FR-EV` (evaluation) and `NFR-11…18`. Tickets continue as E10–E17. Every Part II ticket references Part II requirement IDs and names the repository it is implemented in (`hsmmkit` or `regimekit`).

### 11.1 Why a general package

No maintained Python library combines hidden *semi*-Markov models with explicit durations in continuous or discrete time, covariate-driven transitions/durations/emissions, segment memory and autodiff estimation on JAX. `hmmlearn` and `pomegranate` are HMM-only without covariates; `dynamax` covers HMMs/SSMs in JAX but not semi-Markov durations; `pyhsmm` is Bayesian but NumPy-based and unmaintained; the strongest covariate-driven implementations are R packages (`momentuHMM`, `depmixS4`, `mhsmm`). The design of Part I (structured state index, builders, parameter modes, families behind protocols) already contains most of what a general core needs. Generalization is therefore mainly

1. removing financial assumptions from names, defaults and the data layer and moving the finance-specific parts into the separate application `regimekit`, which becomes the first consumer of the library,
2. adding emission, duration and transition families that other domains need,
3. supporting many independent sequences, labelled data and streaming use,
4. offering estimators beyond MAP (EM, variational Bayes, MCMC) behind the existing `Estimator` protocol, and
5. interoperating with the ecosystem users already have (scikit-learn, pandas/xarray, ArviZ, matplotlib) and letting third parties add families without forking.

### 11.2 Decisions taken for Part II (baseline for E10–E17)

| ID | Decision |
|---|---|
| D12 | **Library/application split** *(revised 1.2)*. `hsmmkit` is domain-neutral: observations `y_t ∈ 𝒴^M`, grid `Δ_t`, covariates. Everything price-, return-, provider-, trading-calendar- or strategy-specific lives in the application package `regimekit`, which depends on `hsmmkit`. The Part I model is a `regimekit` configuration of `hsmmkit`. |
| D13 | **Registered clocks** *(revised 1.2)*. `hsmmkit` ships `steps` (`Δ_t ≡ 1`, parameters per step, no annualization; the default) and `calendar` (elapsed time in years). Other clocks are registered by domain packages; `regimekit` registers `trading` and uses it as its default. Every family MUST work under every clock. |
| D14 | **Multiple sequences are first-class.** Engines are `vmap`ped over a batch of `B` independent sequences with per-sequence length masks; a single sequence is `B = 1`. Contracts stay single-sequence; a `SequenceBatch` container holds many. |
| D15 | **Components are plug-ins.** Emission, duration, metric, source, clock and preset implementations are registered by name; the registry also reads `importlib.metadata` entry points (groups `hsmmkit.*`). Third-party implementations — `regimekit` included — run the same contract test-suites as built-in ones. |
| D16 | **Transition topology.** A boolean mask on the jump matrix (left-to-right, banded, absorbing, arbitrary) is a model-structure element. Duration-dependent jump probabilities are an explicit-engine option. |
| D17 | **Estimator peers.** MAP/L-BFGS remains the reference estimator. EM/generalized EM, variational Bayes and MCMC are peer `Estimator` implementations returning the same `FitResult`, with `estimator` recorded. |
| D18 | **Supervision as constraints.** Full or partial labels and known boundaries are expressed as masks on emissions and transitions consumed by the unchanged engines — no separate supervised code path. |
| D19 | **Online inference reuses the engine.** A stateful `OnlineFilter` provides exact filtering, fixed-lag smoothing and forecasting from the online state. Online parameter updates are v1.x. |
| D20 | **Interoperability, no GUI.** scikit-learn wrapper, pandas/xarray outputs, ArviZ export for sampled posteriors, matplotlib plotting behind an extra. Section 1.3's "no GUI" stands. |
| D21 | **Evaluation split.** Domain-neutral evaluation (scoring rules, information criteria, segmentation and clustering metrics, cross-validation including walk-forward) in `hsmmkit`; strategies and financial metrics in `regimekit`. |
| D22 | **Two distributions, two repositories** *(revised 1.2)*. `hsmmkit` (extras `plot`, `sklearn`, `bayes`, `xarray`, `gpu`) and `regimekit` (depends on `hsmmkit`; extra `providers`). `hsmmkit` never imports or depends on `regimekit`. |
| D23 | **Decouple, then extract** *(new 1.2)*. The split is performed in two phases: (1) inside the `RegimeDetectionAndForecast` repository, all finance-specific code is moved behind registries and extension points until the core packages contain no finance names (guarded by tests); (2) the core is extracted with git history into the new `hsmmkit` repository, renamed, and the `RegimeDetectionAndForecast` repository (package `regimekit`) switches to depending on it. Golden outputs of the finance model stay bitwise identical through both phases. |
| D24 | **Extension instead of profiles** *(new 1.2)*. `hsmmkit` has no `profile` configuration key. A domain package extends the configuration by subclassing section models and registers named components; it never changes the meaning of an `hsmmkit` key. |

### 11.3 Target domains (drive defaults, presets and test fixtures)

| Domain | Package | Typical data | Families | Distinguishing features |
|---|---|---|---|---|
| Finance (Part I) | `regimekit` | log-returns of one or several instruments, daily/intraday, macro covariates | GBM, Student-t, OU, Merton, MS-GARCH | `trading` clock, providers, covariate-driven durations, strategies/backtests |
| Sensor / activity segmentation | `hsmmkit` presets | multivariate real-valued windows, many labelled sequences | multivariate Gaussian, mixtures | `steps` clock, left-to-right or banded topology, partial labels, boundary metrics |
| Counts / reliability / operations | `hsmmkit` presets | event counts per interval, exposure varies | Poisson, Bernoulli with exposure `Δ_t` | hazard-shaped durations, absorbing states |
| Econometrics | `hsmmkit` presets | macro series, regime-switching autoregressions | MS-AR(p), Gaussian | Hamilton-type models, EM, MCMC, standard errors |
| Ecology / movement (plug-in example) | template plug-in | step lengths and turning angles | Gamma, von Mises (third-party) | demonstrates the plug-in contract |

---

## 12. Extended goals and non-goals

### 12.1 Goals (in addition to G1–G6)

| ID | Goal |
|---|---|
| G7 | Be usable without any financial concept: any real-valued, count, binary or categorical sequence on a discrete grid, installed as `pip install hsmmkit` without any finance dependency. |
| G8 | Ship a catalogue of emission, duration and transition structures that covers the common HMM/HSMM literature (Appendix F), each testable by one shared contract suite. |
| G9 | Fit and decode collections of sequences of different lengths, with optional labels, at batch-level performance. |
| G10 | Offer EM, variational and MCMC estimation alongside MAP, with comparable outputs and documented trade-offs. |
| G11 | Support streaming use: update the posterior one observation at a time, smooth with a fixed lag, forecast from the live state, raise regime alerts. |
| G12 | Interoperate: scikit-learn API, pandas/xarray results, ArviZ posteriors, plotting helpers. |
| G13 | Be extensible by third parties through entry points, with a published contract test-suite and a scaffold command; `regimekit` is the first such extension. |

### 12.2 Non-goals for Part II

- Continuous-time switching inside intervals (unchanged from Part I).
- Neural/learned emission densities (e.g. normalizing flows per state) — possible plug-in, not core.
- Bayesian nonparametric state counts (HDP-HSMM): design spike only (E13-T09); no core implementation.
- Full factorial/coupled inference beyond E9-T02.
- A GUI, dashboards or a web service.
- Reinforcement learning / decision policies beyond the finance reference strategies.
- A monorepo or shared release train: the two repositories release independently.

### 12.3 Success criteria per release

| Release | Criteria |
|---|---|
| `hsmmkit` 0.1.0 + `regimekit` 0.3.0 (E10-T01…T09) | `hsmmkit` repository exists with the history of every core module; its test-suite (Part I core tests re-expressed with domain-neutral families) is green; `import hsmmkit` imports nothing finance-related; `regimekit` 0.3.0 depends on `hsmmkit`, migrates every `schema_version: 1` config (Appendix A) automatically, reproduces the v0.2.0 golden outputs bitwise and loads 0.1.x/0.2.x bundles. |
| `hsmmkit` 0.2.0 (E10-T10…T11, E11, E12) | A core configuration (Appendix H) runs `fit → decode → forecast` on a synthetic batch of 50 labelled sequences; family contract suite passes for all families in Appendix F marked "0.2.0"; batch benchmark NFR-12 met. |
| `hsmmkit` 0.3.0 (E13–E14) | EM and MAP (no priors) agree within Appendix G tolerances on all conjugate families; VB reproduces MAP parameter means on large synthetic data and prunes an over-specified state; NumPyro MCMC on the two-regime Appendix E fixture (Gaussian form) yields posterior means within 2 posterior SD of truth; supervised recovery scenario passes; segmentation metrics validated against `ruptures`-style reference values. |
| `hsmmkit` 0.4.0 (E15–E16) | Online filter equals offline filter bitwise; sklearn wrapper passes the applicable subset of `check_estimator`; plotting of all `*Result` objects; `regimekit`, installed as a separate package, is discovered through entry points and passes the contract suites. |
| `hsmmkit` 0.5.0 + `regimekit` 0.5.0 (E17) | Docs restructured into core/extending guides (finance guide in `regimekit`); PyPI + conda-forge for `hsmmkit`; ADR-0014…0021 and ADR-0024…0027 accepted; deprecation policy (NFR-14) in force; `regimekit` released against `hsmmkit` 0.5.0. |

---

## 13. Requirements (Part II)

### 13.1 Functional requirements

#### Generalization (FR-G)

| ID | Requirement |
|---|---|
| FR-G01 | `hsmmkit` MUST NOT reference prices, returns, symbols, tickers, trading calendars, annualization constants, data providers or strategies in names, defaults or documentation. The observation field of `PreparedDataset` is `observations`; `regimekit` MAY expose `returns` as a read-only alias in its own API. |
| FR-G02 | A clock registry MUST exist. `hsmmkit` provides `steps` (`Δ_t ≡ 1`; no parameter is annualized; durations reported in steps; default clock) and `calendar` (`Δ_t` = elapsed years). `regimekit` registers `trading` (`Δ_t = 1/(252·H)`) including its future-grid rule. |
| FR-G03 | Observation supports `real` (`ℝ^M`), `count` (`ℕ^M`), `binary` (`{0,1}^M`) and `categorical` (`{0..K−1}`) MUST be supported. A family declares its `support`; the preparer validates data against it and raises `DataError` on mismatch. |
| FR-G04 | Generic data sources MUST be available in `hsmmkit`: in-memory `numpy` arrays, `pandas.DataFrame`, `xarray.Dataset`, CSV and parquet files. Under the `steps` clock a time index is optional. |
| FR-G05 | `hsmmkit` configuration has no `profile` key. Domain packages extend it by subclassing section models and registering named components (families, sources, clocks, init strategies, observation transforms, metrics, presets). An extension MUST NOT change the semantics of any `hsmmkit` key. |
| FR-G06 | A preset registry (`hsmmkit presets list/show/write`) MUST ship named, validated configurations with matching synthetic fixtures, namespaced per package (`core/*` from `hsmmkit`, `finance/*` from `regimekit`). |
| FR-G07 | Backward compatibility: every `regimekit` `schema_version: 1` configuration MUST migrate automatically to `regimekit` `schema_version: 2` and validate with unchanged meaning. `hsmmkit` configurations start at `hsmmkit` `schema_version: 1`. |
| FR-G08 | Dependency direction *(new 1.2)*: `hsmmkit` MUST NOT import or depend on `regimekit`; `regimekit` uses `hsmmkit` only through its public API (including `hsmmkit.testing`). |
| FR-G09 | Extraction *(new 1.2)*: the `hsmmkit` repository MUST preserve the git history of every module it takes over, and the extraction MUST NOT change numerical behaviour (Part I golden values reproduced bitwise by `hsmmkit` for core families and by `regimekit` for finance configurations). |

#### Emission families (FR-EM)

| ID | Requirement |
|---|---|
| FR-EM01 | Multivariate Gaussian with `diag` or `full` covariance (Cholesky parameterization), optional `Δ`-scaling of mean and covariance, optional Itô correction, covariate links on mean and log-scale. |
| FR-EM02 | Multivariate Student-t (`ν` free/tied/fixed) with the same options. |
| FR-EM03 | Poisson (exposure `Δ_t`, log link), Bernoulli (logit link) and Categorical (softmax link), each with covariate links on the natural parameter. |
| FR-EM04 | Markov-switching autoregression MS-AR(p) with per-state intercept, AR coefficient matrices and covariance, realized as a Gaussian family whose emission covariates are the preparer-generated own lags; the first `p` steps are conditioned on (masked). Hamilton's mean-adjusted variant is supported for `p ≤ 2` through a time-step memory expansion with geometric durations (corrected 1.2; segment memory D5 cannot express it) and documented as unsupported beyond. |
| FR-EM05 | Per-state finite mixture of any base family with static component count `K_i`. |
| FR-EM06 | `regimekit` MUST provide GBM (= `hsmmkit` Gaussian with `Δ`-scaling and Itô correction, registered as `gbm`), OU with exact discretization and Merton jump-diffusion (truncated Poisson mixture). MS-GARCH remains the E9-T03 spike in `regimekit`. |
| FR-EM07 | Families MAY implement `sufficient_stats`/`m_step` (closed-form EM) and `conjugate_prior` (VB/Gibbs); absence degrades gracefully to numerical M-steps or point estimates. |
| FR-EM08 | Custom families MUST be registrable via entry points and MUST pass `hsmmkit.testing.family_contract`. |

#### Durations and transitions (FR-DU)

| ID | Requirement |
|---|---|
| FR-DU01 | A transition topology mask `Ω ∈ {0,1}^{N×N}` (diagonal ignored) MUST be configurable, with shorthands `full`, `left_to_right`, `banded(k)`, `cyclic` and explicit matrices. Rows without any allowed target are absorbing states. |
| FR-DU02 | Absorbing states MUST be supported in both engines (infinite sojourn; the final segment is right-censored). |
| FR-DU03 | Duration-dependent jump probabilities `π_ij(d)` MUST be available in the explicit-duration engine. |
| FR-DU04 | Left-censoring of the first segment MUST be available (`initial: stationary` in the expanded engine; residual-life initial duration in the explicit engine). |
| FR-DU05 | Per-state duration families (e.g. geometric for one state, gamma for another) MUST be supported in the explicit engine; the expanded engine supports per-state `r_i` only (already in Part I). |

#### Multiple sequences (FR-SQ)

| ID | Requirement |
|---|---|
| FR-SQ01 | A `SequenceBatch` of `B` independent `PreparedDataset`s with different lengths MUST be accepted by `fit`, `decode`, `score` and `simulate`; the joint log-likelihood is the sum over sequences. |
| FR-SQ02 | The initial distribution MAY be shared across sequences or free per sequence group. |
| FR-SQ03 | Posteriors, Viterbi paths and segments MUST be returned per sequence with original lengths. |
| FR-SQ04 | Sequences MUST be padded to length buckets to bound recompilation; bucket edges are configurable. |

#### Estimation (FR-ES)

| ID | Requirement |
|---|---|
| FR-ES01 | EM (Baum-Welch on the expanded chain with phase pooling) MUST be available for the expanded engine, with closed-form M-steps where families provide them and one-step numerical M-steps otherwise (ECM). |
| FR-ES02 | Generalized EM (Guédon 2003) MUST be available for the explicit-duration engine, including the nonparametric duration M-step. |
| FR-ES03 | Forward-filtering backward-sampling (FFBS) MUST be available for posterior path sampling in both engines. |
| FR-ES04 | Variational Bayes (mean-field over parameters, exact structured posterior over states) MUST be available for conjugate families, with ELBO reporting and automatic state pruning; non-conjugate components fall back to point estimates (VB-EM hybrid) with a warning. |
| FR-ES05 | MCMC MUST be available through the `bayes` extra: NUTS on `log_joint(u)` with marginalized states, and blocked Gibbs (FFBS + conjugate updates) for conjugate families; output as ArviZ `InferenceData`. |
| FR-ES06 | All estimators MUST return a `FitResult` with `estimator`, `loglik` (at the point estimate or posterior mean) and, where applicable, `elbo`, `posterior_samples`. |
| FR-ES07 | Tempered/deterministic-annealing EM SHOULD be available as an option to reduce multimodality sensitivity. |

#### Supervision (FR-SV)

| ID | Requirement |
|---|---|
| FR-SV01 | Partial labels (allowed regime set per step, `(T, N)` bool) MUST be consumable by fit, decode and score; a missing label allows all regimes. |
| FR-SV02 | Known segment boundaries (`must_switch_at`, `must_not_switch_at`) MUST be consumable as transition constraints; infeasible constraints (e.g. NB minimum duration) raise `ConstraintError`. |
| FR-SV03 | Label-informed initialization (`init: labels`) MUST be available: emission parameters from labelled steps, jump matrix and durations from labelled segments. |

#### Online inference (FR-ON)

| ID | Requirement |
|---|---|
| FR-ON01 | An `OnlineFilter` MUST provide `update(y, dt, z) → filtered regime probabilities` in `O(S²)` per step, bitwise equal to the offline filter. |
| FR-ON02 | Fixed-lag smoothing with lag `L` MUST be available (`smoothed_at(t − L)`). |
| FR-ON03 | Alert rules (probability threshold with hysteresis, expected remaining duration below a bound) MUST be configurable and emit structured events. |
| FR-ON04 | The online state MUST be serializable and restorable (`save_state/load_state`) so a stream can be resumed. |
| FR-ON05 | Online/recursive EM (Cappé & Moulines 2009) is v1.x (E15-T05). |

#### Interoperability and data (FR-IO)

| ID | Requirement |
|---|---|
| FR-IO01 | A scikit-learn estimator `HSMM(BaseEstimator)` with `fit(X, lengths=None, y=None)`, `predict`, `predict_proba`, `score`, `decode`, `sample` MUST be available behind the `sklearn` extra (Appendix I). |
| FR-IO02 | All `*Result` objects MUST convert to `pandas` (long format) and `xarray` (labelled dims `time, state, series, sim, horizon`). |
| FR-IO03 | Sampled posteriors MUST export to ArviZ `InferenceData`. |
| FR-IO04 | `hmmlearn` `GaussianHMM`/`CategoricalHMM` parameters MUST be importable into a geometric-duration `hsmmkit` model (one-way). |
| FR-IO05 | The configuration schema MUST be exportable as JSON Schema for editor support, including the names of registered components (also those of extension packages). |

#### Visualization (FR-VZ)

| ID | Requirement |
|---|---|
| FR-VZ01 | Plot helpers (matplotlib, `plot` extra): observations with regime shading and filtered/smoothed probabilities; duration pmfs and hazards; forecast fan charts; diagnostics panel (PIT, pseudo-residuals, restart dispersion, ELBO trace). |
| FR-VZ02 | Every helper accepts an `ax` and returns it; no global state; no figure is shown implicitly. |

#### Plug-ins (FR-PL)

| ID | Requirement |
|---|---|
| FR-PL01 | Entry-point groups `hsmmkit.emission`, `hsmmkit.duration`, `hsmmkit.metric`, `hsmmkit.source`, `hsmmkit.clock`, `hsmmkit.transform`, `hsmmkit.init`, `hsmmkit.preset`, `hsmmkit.splitter`, `hsmmkit.estimator`, `hsmmkit.ppc_statistic` MUST be discovered at first registry access (lazy). |
| FR-PL02 | `hsmmkit.testing` MUST publish the contract test-suites (family, duration, metric, source, clock) as importable pytest fixtures/parametrizations. |
| FR-PL03 | `hsmmkit new-family NAME` MUST scaffold a plug-in package with tests wired to the contract suite. |

#### Evaluation (FR-EV)

| ID | Requirement |
|---|---|
| FR-EV01 | Segmentation metrics: boundary precision/recall/F1 at tolerance `τ`, Hausdorff distance, covering. |
| FR-EV02 | Clustering metrics with optimal label matching (Hungarian): Hamming accuracy, ARI, NMI, per-state confusion. |
| FR-EV03 | Cross-validation over sequences (grouped K-fold) and over time (Part I walk-forward) behind one `Splitter` interface. |
| FR-EV04 | Grid model selection over `N`, `r`, families and estimators producing one comparison table (extends `compare`). |
| FR-EV05 | Parametric bootstrap for standard errors and for nested-model comparison with bootstrap p-values (regularity caveats documented). |
| FR-EV06 | Pseudo-residuals (forecast-PIT based) and posterior predictive checks on user-selected summary statistics. |

### 13.2 Non-functional requirements (Part II)

| ID | Requirement |
|---|---|
| NFR-11 | `import hsmmkit` MUST succeed with only core dependencies; extra dependencies (`numpyro`, `arviz`, `matplotlib`, `scikit-learn`, `xarray`) are imported lazily and raise an actionable `ImportError` naming the extra. |
| NFR-12 | Batch performance guardrail (8-core x86 CPU): `B = 64`, `T = 1000`, `N = 4`, `S = 12`, order 1 — jitted batched forward-backward ≤ 300 ms; one EM iteration ≤ 400 ms. |
| NFR-13 | Estimator parity: EM vs MAP without priors agree within Appendix G; VB posterior means vs MAP on `T ≥ 20 000` within Appendix G. |
| NFR-14 | Plug-in API stability: `hsmmkit.testing`, the family/duration/metric/source/clock protocols and `PreparedDataset` follow semantic versioning from 0.3.0; breaking changes require a two-minor-version deprecation cycle. |
| NFR-15 | Online filter equals offline filter bitwise on the same backend; fixed-lag smoothed values `P(s_{t−L} | y_{1:t})` equal the offline smoother run on the prefix `y_{1:t}` (and, at `t = T`, the full offline smoother at `T − L`) *(corrected 1.2: they do not equal the full-data smoother for `t < T`)*. |
| NFR-16 | Documentation: separate guides for the core and for "extending hsmmkit" in `hsmmkit`, the finance guide in `regimekit`; one tutorial per target domain executed in CI. |
| NFR-17 | Reproducibility (NFR-03) extends to EM, VB and MCMC (seeded chains); MCMC determinism is per backend and per NumPyro version and is documented as such. |
| NFR-18 | Cross-repository compatibility *(new 1.2)*: `hsmmkit` CI runs a nightly `downstream` job that installs the current `RegimeDetectionAndForecast` main branch against the `hsmmkit` head and runs that repository's golden and contract tests; `regimekit` pins `hsmmkit` to a compatible minor range (`>=x.y,<x.(y+1)` before 1.0). |

---

## 14. Mathematical specification (additions)

### 14.1 Clocks

Under `steps`, `Δ_t = 1` for all `t`. Continuous-time families receive `Δ = 1` and their parameters are interpreted per step; duration families report `mean_sojourn` in clock units (years or steps). Nothing else changes: `a_i = 1 − exp(−λ_i)` is still the per-step exit probability, so geometric/NB semantics carry over exactly. A registered clock (e.g. `trading` in `regimekit`) only defines `Δ_t` on the grid and the rule for future grid points; no model code branches on the clock name.

### 14.2 Emission families

Notation as in Section 4.2; `m_i(t)` and `Σ_i(t)` denote covariate-linked location and scale. `s(Δ) = Δ` when `scale_by_dt: true` (continuous clocks) and `1` otherwise.

**Multivariate Gaussian.** `y_t | i ~ N(m_i(t) s(Δ_t), Σ_i(t) s(Δ_t))`, `Σ_i = L_i L_iᵀ` with `L_i` lower-triangular, `diag(L_i) = softplus(·)`; `diag` mode sets off-diagonals to zero. Itô correction (`ito_correction: true`): `m_i ← μ_i − ½ diag(Σ_i)`. GBM of Part I is exactly this family with `M = 1`, `scale_by_dt: true`, `ito_correction: true`.

**Multivariate Student-t.** `y_t | i ~ t_{ν_i}(m_i(t) s(Δ_t), Σ_i(t) s(Δ_t))`, `ν_i > 2`.

**Poisson / Bernoulli / Categorical.** `y_t | i ~ Poisson(exp(η_i + γ_iᵀ z^e_t) · s(Δ_t))` (exposure `Δ`); `y_t | i ~ Bernoulli(σ(η_i + γ_iᵀ z^e_t))`; `y_t | i ~ Cat(softmax(η_i + Γ_i z^e_t))` with the first category as reference. Sufficient statistics and closed-form M-steps exist for all three (Appendix F).

**MS-AR(p).** Regime-dependent-coefficient form: `y_t = c_i + Σ_{l=1}^{p} Φ_{i,l} y_{t−l} + ε_t`, `ε_t ~ N(0, Σ_i)`. The preparer emits `y_{t−1}, …, y_{t−p}` as emission covariates (this is the existing FR-M09 drift link with per-state coefficients and `lag ≥ 1` enforced), and masks `t ≤ p`. Hamilton's mean-adjusted form `y_t − μ_{s_t} = Σ φ_l (y_{t−l} − μ_{s_{t−l}})` depends on the regimes at `t−1..t−p`; because segment memory (Section 4.5) excludes repeated regimes, it cannot represent step memory; it is realized *(corrected 1.2)* through a time-step memory expansion of the state space (builder option `memory_mode: steps`, `N^{p+1}` tuples `(s_{t−p}, …, s_t)` with repeats allowed, geometric durations only — Hamilton's model is an HMM) for `p ≤ 2` and rejected with `ConfigError` otherwise.

**Mixture.** `f_i(y) = Σ_{k=1}^{K_i} w_{ik} g(y | θ_{ik})`, `w_i` on the simplex; `log f_i = logsumexp_k(log w_{ik} + log g)`. Ordering constraints on `θ_{ik}` within a state are supported for identifiability.

**OU (`regimekit`).** For an observed level `x_t` with regime `i`: `x_t | x_{t−1}, i ~ N(m_i + (x_{t−1} − m_i) e^{−κ_i Δ_t}, σ_i² (1 − e^{−2κ_i Δ_t}) / (2κ_i))`. Implemented as its own family (nonlinear in `Δ`), with `x_{t−1}` supplied as a lag covariate.

**Merton jump-diffusion (`regimekit`).** `y_t | i, k ~ N((μ_i − ½σ_i² − λ_i k̄_i) Δ_t + k m_{J,i}, σ_i² Δ_t + k s_{J,i}²)`, `k ~ Poisson(λ_i Δ_t)` truncated at `K_max` (default 5): a mixture family instance with Poisson weights and `k̄ = exp(m_J + ½ s_J²) − 1`.

**Extended family contract.** In addition to Section 4.2: `support ∈ {real, count, binary, categorical}`, `scale_by_dt: bool`, `sufficient_stats(y, dt, gamma, z) → stats | None`, `m_step(stats) → θ | None`, `conjugate_prior() → PriorFamily | None`, `expected_log_prob(y, dt, q_theta, z)` (VB, optional).

### 14.3 Multiple sequences and padding

For `B` independent sequences, `ℓ(θ) = Σ_b ℓ_b(θ)`. Sequences are padded to a bucket length `T_max`. For padded steps `t > T_b` the engine uses `ℓ_t(s) = 0` and `A_t = I`; both `α` and `β` are then invariant on the padding, so `logsumexp α_{T_max} = ℓ_b` and smoothed quantities for `t ≤ T_b` are unaffected. The batched recursion is `vmap` over `b` of the Section 4.7.2 scan; covariates and `Δ_t` are batched alike. Per-group initial distributions replace `ρ` by `ρ_{g(b)}`.

### 14.4 Left-censoring

If the process is already in a segment at `t = 1`:

- *Expanded engine.* `initial: stationary` (FR-M13) is the equilibrium distribution over `(memory, regime, phase)` and therefore exactly the left-censored initial distribution in the phase representation.
- *Explicit engine.* By renewal theory the remaining duration of the initial segment has the residual-life pmf `p_j^{res}(d) = S_j(d − 1) / E[d_j]`, so `α_1(j, d)` uses `ρ_j p_j^{res}(d)` in place of `ρ_j p_j(d)` for the first segment only; `ρ` is the stationary regime occupancy `ρ_j ∝ ν_j E[d_j]` with `ν` the stationary distribution of `π`, or free.

### 14.5 Transition topology and duration-dependent jumps

Let `Ω ∈ {0,1}^{N×N}`, `Ω_ii := 0`. Logits `a_ij, b_ij` exist only for `Ω_ij = 1`; the reference category of row `i` is its first allowed target; rows with exactly one allowed target carry no free parameter; `π_ij = 0` where `Ω_ij = 0` (`−inf` in log space, structural zero). A row with `Σ_j Ω_ij = 0` is absorbing: `a_i(t) ≡ 0` in the expanded engine, `p_i(d) := δ_{d, D_max}` with right-censoring in the explicit engine.

Duration-dependent jumps (explicit engine only): `π_ij(d) ∝ Ω_ij exp(a_ij + b_ijᵀ z^π_t + c_ij log d)` evaluated with the duration `d` of the segment being left. The expanded engine cannot express this (exit happens only from phase `r_i`) and rejects the option.

### 14.6 EM and generalized EM

**Expanded engine (Baum-Welch with pooling).** E-step: `γ_t(s)` and pairwise `ξ_t(s, s')` from Section 4.7.2 (computed inside the scan; never materialized as `(T, S, S)`). M-step:

- emissions: `γ_t` collapsed to regimes, `θ_i = m_step(sufficient_stats(y, Δ, γ·C))` for exponential families; otherwise one L-BFGS step on the expected complete-data log-likelihood (ECM);
- phase advance (constant `Δ`, no duration covariates): `a_i = Σ_t Σ_c ξ_t((i,c) → (i,c+1) or exit) / Σ_t Σ_c γ_t(i, c)`; with irregular `Δ_t` or covariates: ECM;
- jump matrix: `π_ij ∝ Σ_t ξ_t((i, r_i) → (j, 1))` for `Ω_ij = 1`, plus Dirichlet prior counts if `use_priors`;
- initial distribution: `γ_1` (free mode) or unchanged.

**Explicit engine (Guédon 2003).** The E-step yields segment posteriors `η_t(j, d) = P(segment of regime j ends at t with duration d | y_{1:T})` and jump posteriors `ξ_t(i, j)`. M-step: `π_ij ∝ Σ_t ξ_t(i, j)`; nonparametric durations `p_j(d) ∝ Σ_t η_t(j, d)` (with the right-censored last segment contributing survival mass); parametric durations by weighted MLE of `Σ_{t,d} η_t(j,d) log p_j(d; φ_j)` (Newton, 1–2 parameters); emissions as above with `γ_t(j) = Σ_{d} …` collapsed.

Convergence: `ℓ` MUST be non-decreasing (asserted in tests up to 1e-10); stop when `Δℓ ≤ tol_f · max(1, |ℓ|)` or `max_iter`. Tempered EM (FR-ES07): E-step with `ℓ_t^{β_k}`, `β_k ↑ 1` on a geometric schedule.

### 14.7 Posterior path sampling (FFBS)

`s_T ~ softmax(α_T)`; for `t = T−1, …, 1`: `P(s_t = s | s_{t+1}) ∝ exp(α_t(s)) A_{t+1}(s, s_{t+1})`. The explicit engine samples segments backwards using `α_t(j, d)`. FFBS provides segment-uncertainty bands for `decode` and the state block of blocked Gibbs.

### 14.8 Variational Bayes

Mean-field `q(θ) Π_b q(s^{(b)}_{1:T_b})` with `q(s)` exact (structured) via forward-backward on expected log-parameters, and `q(θ)` in the conjugate families of Appendix F: Normal-Wishart (Gaussian), Gamma (Poisson rate, exit rate), Beta (phase-advance `a_i`, Bernoulli), Dirichlet (Categorical, jump rows, nonparametric pmf). The ELBO is reported each iteration and MUST be non-decreasing. State pruning: a state with `E_q[occupancy] < ε` (default 1 %) is reported as pruned. Non-conjugate components (covariate links, parametric explicit durations, Student-t `ν`) use point estimates updated by an M-step (VB-EM hybrid); the fit report lists which components were treated how.

### 14.9 MCMC

Via the `bayes` extra. (a) NUTS on the unconstrained vector `u` with `log_joint(u) = ℓ(θ(u)) + log p(θ) + log|det ∂θ/∂u|`, states marginalized by the forward algorithm (the objective of Section 4.8 negated). (b) Blocked Gibbs for conjugate families: FFBS for states, conjugate draws for parameters. Label switching: ordering constraints as in Part I, else post-hoc relabelling by pivot matching on the MAP ordering (Stephens 2000 simplified). Output: draws of `θ`, optional path draws, converted to `InferenceData` with dims `(chain, draw, state, series, …)`; `FitResult.params` is the posterior mean on the constrained scale.

### 14.10 Supervision as constraints

Allowed-set matrix `L ∈ {0,1}^{T×N}`: `ℓ_t(s) ← ℓ_t(s) + log L_t[regime_of[s]]` (`−inf` when disallowed). Boundary constraints act on `A_t`: `must_switch_at t` keeps only exit transitions `(m, i, r_i) → (m', j, 1)`, `j ≠ i`; `must_not_switch_at t` removes exit transitions. In the explicit engine `must_switch_at t` restricts `α_t` to segments starting at `t`. Because the likelihood is unchanged in form, every estimator of Sections 4.8 and 14.6–14.9 works on constrained problems without modification. Infeasibility (`logsumexp α_T = −inf`) raises `ConstraintError` naming the first infeasible step.

### 14.11 Online filtering and fixed-lag smoothing

`OnlineFilter` holds `(log_α_t, t, carry)`. `update(y_{t+1}, Δ_{t+1}, z_{t+1})`: one step of the forward recursion, `O(S²)`. Fixed-lag smoothing with lag `L`: keep a ring buffer of the last `L` pairs `(ℓ_u, log A_u)`, run the backward recursion over the buffer from `β_{t} = 0` and return `γ_{t−L} ∝ exp(α_{t−L} + β_{t−L})`; cost `O(L S²)` per step. Alerts fire on `P(regime = j | y_{1:t})` crossing `p_on` and clear below `p_off < p_on` (hysteresis), or on expected remaining duration (Section 4.7.2 augmented posterior) below a bound. Forecasting from the online state is Section 4.9 with `α_t` taken from the filter.

### 14.12 Selection and diagnostics

- *Splitters.* Grouped K-fold over sequences (`SequenceKFold`) and time-based walk-forward (Part I) implement one `Splitter` interface; CV log score and CRPS are aggregated per fold.
- *Grid selection.* Cartesian grid over `n_states`, `r`, emission/duration family and estimator; each cell reports IC (Section 4.10) and CV scores; the table is the output of `compare --grid`.
- *Parametric bootstrap.* Simulate `R` datasets from `θ̂`, refit each from `θ̂`, SE = sample sd of `θ̂_r`; bootstrap distribution of `2(ℓ_1 − ℓ_0)` for nested comparisons (documented: the χ² approximation fails when `N` differs — Hansen 1992).
- *Pseudo-residuals.* `u_t = F(y_t | y_{1:t−1})` from the one-step predictive mixture; `r_t = Φ^{−1}(u_t)` should be i.i.d. `N(0,1)` (Zucchini, MacDonald & Langrock 2016). For discrete supports, mid-distribution pseudo-residuals.
- *Posterior predictive checks.* User-selected statistics `T(y)` (default: autocorrelation of `|y|` at lags 1–20, empirical sojourn mean per decoded regime, kurtosis) computed on simulated replicates versus observed; reported as tail probabilities.

### 14.13 Segmentation and clustering metrics

Boundaries `𝔅 = {t : q_t ≠ q_{t−1}}`. Boundary precision/recall/F1 at tolerance `τ` (a predicted boundary matches if a true boundary lies within `τ` steps; one-to-one matching), Hausdorff distance between boundary sets, covering (mean over true segments of the best Jaccard overlap with a predicted segment). Label-dependent metrics after Hungarian matching of predicted to true states: Hamming accuracy, per-state confusion; label-free: ARI, NMI. Balanced accuracy of Part I is retained.

---

## 15. Architecture (additions and amendments)

### 15.1 Repository and package layout *(rewritten 1.2)*

Two repositories, two distributions. Arrows point from dependent to dependency.

```
RegimeDetectionAndForecast (repo, existing; package regimekit)  ──depends on──▶  hsmmkit (repo, new; extracted with history)
```

```
hsmmkit/src/hsmmkit/
├── __init__.py               # version, Project re-export, jax x64 bootstrap
├── project.py                # Project facade (core operations, sources)
├── config/                   # schema (core sections, extension hooks), yaml_io, migrate, defaults, gates.py (feature gates)
├── data/
│   ├── contracts.py          # ObservationSet (generic), CovariateSeries, PrepStats, PreparedDataset, BoundaryConstraints
│   ├── clocks.py             # Clock protocol + registry; steps, calendar
│   ├── calendar.py           # grid building, alignment, lagging (clock-agnostic)
│   ├── prepare.py, transforms.py, quality.py (generic checks), batch.py (SequenceBatch)
│   └── sources/              # base.py (Source protocol), numpy.py, pandas.py, xarray.py, csv.py, parquet.py
├── model/
│   ├── emission/             # base.py, gaussian.py, student_t.py, poisson.py, bernoulli.py, categorical.py, msar.py, mixture.py, product.py
│   ├── duration/             # Part I families + per_state.py (explicit engine composite)
│   ├── topology.py           # Ω masks and shorthands
│   └── spec.py, params.py, dists.py, transition.py, covariates.py, lattice.py
├── statespace/               # + duration-dependent jump support in transitions.py (explicit); factorial.py (experimental, E9-T02)
├── engine/                   # + batch.py (vmap wrappers, bucketing), constraints.py (label/boundary masks), ffbs.py
├── estimation/
│   ├── map.py                # Estimator "map" (reference)
│   ├── em.py                 # expanded-engine EM/ECM, tempering (supersedes the E5-T06 GBM-only EM)
│   ├── gem.py                # explicit-engine generalized EM
│   ├── vb.py                 # mean-field VB, conjugate updates, pruning
│   ├── mcmc.py               # NumPyro bridge (bayes extra), relabelling
│   └── init.py, objective.py, optimizers.py, report.py, results.py
├── inference/                # decode, simulate, forecast, covariate_forecasters
├── online/                   # filter.py (OnlineFilter), alerts.py, state.py
├── selection/                # splitters.py, grid.py, bootstrap.py, ppc.py, pseudo_residuals.py
├── evaluation/               # metrics/ (+ segmentation.py, clustering.py), backtest.py (walk-forward, origin hooks), compare.py, report.py
├── interop/                  # sklearn.py, xarray.py, arviz.py, hmmlearn.py
├── plot/                     # regimes.py, durations.py, forecast.py, diagnostics.py (plot extra)
├── persistence/              # bundle v2 (provenance + extension files)
├── plugins.py                # registries, entry-point discovery
├── presets/                  # core/* YAML presets + fixture generators
├── testing/                  # contract suites: family_contract, duration_contract, metric_contract, source_contract, clock_contract
├── cli/app.py                # typer app factory build_app(); console script `hsmmkit`
└── utils/                    # jax_setup, rng, logging, errors (HsmmkitError …), typing, optional_import

RegimeDetectionAndForecast/src/regimekit/   # the finance application (flattened: no `finance/` subpackage after E10-T09)
├── __init__.py               # registers finance components (entry points from E16-T06), re-exports FinanceProject as Project
├── config.py                 # FinanceConfig(ProjectConfig): provider sources, cache, strategy, schema v1 → v2 migration
├── clocks.py                 # trading clock (+ future trading-day grid)
├── providers/                # yfinance, stooq, tiingo, fred, cboe, csv (Source kind "provider")
├── cache.py                  # parquet cache, snapshot ids, offline replay
├── transforms.py             # log_returns observation transform; returns alias
├── quality.py                # exchange-calendar gaps, non-positive prices, adjustment discontinuities
├── emission/                 # gbm.py (alias of hsmmkit gaussian), ou.py, merton.py, msgarch.py (E9-T03 spike)
├── strategies.py, metrics.py # regime_long_flat, regime_target_vol, sharpe, max_drawdown, turnover, hit_rate
├── presets/                  # finance/dax_3state_gbm_nb, finance/two_regime_sanity
├── project.py                # FinanceProject(Project): fetch(), strategy backtests
└── cli.py                    # console script `regimekit` = hsmmkit app + fetch + finance config
```

### 15.2 Core data contracts (additions)

```python
@dataclass(frozen=True)
class ObservationSet:                       # Part I 5.4, generalized (1.2)
    series_names: tuple[str, ...]           # M      — was `symbols`
    times: np.ndarray | None                # (T0,) datetime64[ns] UTC; None allowed under `steps`
    values: np.ndarray                      # (T0, M) — was `prices`
    source: str                             # was `provider`
    snapshot_id: str
    meta: dict[str, Any]                    # regimekit: interval, adjusted, field

@dataclass(frozen=True)
class PreparedDataset:                      # Part I 5.4 as amended by 15.7 (already implemented)
    observations: Array                     # (T, M)
    support: Literal["real", "count", "binary", "categorical"]
    n_categories: int | None                # categorical only
    labels: Array | None                    # (T, N) bool allowed-set, None = unlabelled
    boundaries: BoundaryConstraints | None  # must_switch_at / must_not_switch_at step indices
    group: str | None                       # initial-distribution group (FR-SQ02)
    clock: str                              # registered clock name (1.2: no longer a Literal)
    ...                                     # remaining fields unchanged

@dataclass(frozen=True)
class SequenceBatch:
    sequences: tuple[PreparedDataset, ...]  # B, heterogeneous lengths
    bucket_edges: tuple[int, ...]           # padding buckets (FR-SQ04)
    def padded(self) -> PaddedBatch: ...    # observations (B, T_max, M), mask (B, T_max), dt, covariates, labels
    def unpad(self, arr: Array) -> tuple[Array, ...]: ...

@dataclass(frozen=True)
class Topology:
    allowed: np.ndarray                     # (N, N) bool, diagonal False
    absorbing: tuple[int, ...]

class ModelSpec(eqx.Module):                # Part I, amended
    topology: Topology
    left_censored: bool
    duration_dependent_jumps: bool          # explicit engine only
    per_state_durations: tuple[DurationFamily, ...] | None

@dataclass(frozen=True)
class FitResult:                            # Part I, amended
    estimator: Literal["map", "em", "gem", "vb", "mcmc"]
    elbo: float | None
    posterior_samples: Any | None           # ArviZ InferenceData when available
    pruned_states: tuple[int, ...]

@dataclass(frozen=True)
class BacktestResult:                       # Part I, amended (1.2)
    folds: pd.DataFrame; metrics: pd.DataFrame; summary: pd.DataFrame
    extras: dict[str, pd.DataFrame]         # origin-hook outputs; regimekit stores "strategy" here

@dataclass(frozen=True)
class OnlineState:
    log_alpha: Array; t: int; carry: Any; buffer: RingBuffer | None; spec_hash: str
```

### 15.3 Protocols (additions)

```python
class EmissionFamily(Protocol):             # Part I 5.5, extended
    support: Literal["real", "count", "binary", "categorical"]
    scale_by_dt: bool
    def sufficient_stats(self, y, dt, gamma, z) -> Any | None: ...
    def m_step(self, stats: Any) -> Params | None: ...
    def conjugate_prior(self) -> PriorFamily | None: ...
    def expected_log_prob(self, y, dt, q_theta, z) -> Array | None: ...   # VB

class DurationFamily(Protocol):             # extended
    def sufficient_stats(self, eta) -> Any | None: ...       # explicit engine E-step output
    def m_step(self, stats) -> Params | None: ...
    def conjugate_prior(self) -> PriorFamily | None: ...

class Estimator(Protocol):                  # implementations: MAPEstimator, EMEstimator, GEMEstimator, VBEstimator, MCMCEstimator
    name: str
    def fit(self, spec, data: PreparedDataset | SequenceBatch, cfg, key) -> FitResult: ...

class OnlineFilter(Protocol):
    def start(self, fit: FitResult, initial: PreparedDataset | None) -> OnlineState: ...
    def update(self, state: OnlineState, y, dt, z) -> tuple[OnlineState, Array]: ...
    def smoothed_at(self, state: OnlineState, lag: int) -> Array: ...
    def forecast(self, state: OnlineState, horizon: int, ...) -> ForecastResult: ...

class Source(Protocol):                     # generic data source; replaces DataProvider in the core
    name: str
    def load(self, cfg: SourceCfg) -> tuple[ObservationSet, list[CovariateSeries]]: ...

class Clock(Protocol):                      # new 1.2
    name: str; unit: Literal["year", "step"]; options_model: type[BaseModel]
    def dt(self, times: np.ndarray | None, n: int, opts) -> np.ndarray: ...                       # (T,)
    def future_grid(self, last_time, h: int, opts, times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]: ...

class ObservationTransform(Protocol):       # new 1.2; regimekit registers "log_returns"
    name: str
    def __call__(self, values: np.ndarray, times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]: ...

class OriginHook(Protocol):                 # new 1.2; walk-forward extension point used by regimekit strategies
    name: str
    def on_origin(self, fold: int, origin: int, posterior, forecast) -> dict[str, Any]: ...
    def finalize(self, rows: list[dict[str, Any]]) -> pd.DataFrame: ...

class Splitter(Protocol):
    def split(self, batch: SequenceBatch) -> Iterator[tuple[SequenceBatch, SequenceBatch]]: ...

class Plotter(Protocol):
    def __call__(self, result: Any, ax: "matplotlib.axes.Axes | None" = None, **kw) -> "matplotlib.axes.Axes": ...
```

`DataProvider` (Part I) becomes a `Source` specialization (kind `provider`) in `regimekit.providers`; the contract test-suite of E2-T01 is generalized to `hsmmkit.testing.source_contract`.

### 15.4 Configuration schemas *(rewritten 1.2)*

**`hsmmkit` `schema_version: 1`** (no `profile` key):

| Section / key | Detail |
|---|---|
| `data.source` | `{kind: numpy \| pandas \| xarray \| csv \| parquet \| <registered>, path, observation_columns, time_column, sequence_column, covariate_columns}`; per-kind options validated by the source's options model |
| `data.clock`, `data.clock_options` | registered clock name (`steps` default, `calendar`); options validated by the clock's options model |
| `data.covariates` | `[{name, source: null \| SourceCfg, column, transform, lag ≥ 1, missing, max_staleness, use_in}]`; `source: null` = column of the primary source |
| `data.preparation` | `{observation_transform: null \| <registered>, standardize_observations, standardize_covariates, missing_observations: mask \| drop \| error}` |
| `data.sequences` | `{group_by: column \| null, bucket_edges: [256, 512, 1024, 2048, 4096]}` |
| `data.labels` | `{source: column \| file, allowed_sets: bool, boundaries: {must_switch_at: [...], must_not_switch_at: [...]}}` |
| `model.emission.family` | registered name (Appendix F); `params` per family; `scale_by_dt`, `ito_correction`, `covariance: diag \| full`, `mixture: {components: K, base: …}` |
| `model.topology` | `full \| left_to_right \| banded: k \| cyclic \| matrix: [[…]]` |
| `model.left_censored` | bool |
| `model.initial_groups` | `shared \| per_group` (FR-SQ02) |
| `model.duration` | `family` or `per_state: [{family, params}, …]` (explicit engine); `duration_dependent_jumps: bool` |
| `estimation.method` | `map \| em \| gem \| vb \| mcmc`; per-method sub-sections (`em.tempering`, `vb.prune_below`, `mcmc.{sampler, chains, warmup, draws}`) |
| `estimation.init` | registered init strategies: `rolling_kmeans`, `lattice_jitter`, `random`, `labels` |
| `online` | `{lag: L, alerts: [{regime, p_on, p_off}, {remaining_duration_below: …}]}` |
| `evaluation.metrics` | registered names incl. segmentation/clustering; `evaluation.cv: {splitter: sequence_kfold \| walk_forward, …}`; `evaluation.grid: {...}`; `evaluation.backtest` with windows in clock units (`initial_train`, `rolling_window`) |

Validation additions: family `support` matches data support; `topology` rows non-empty unless `absorbing` intended (warning); `duration_dependent_jumps` requires `engine: explicit`; `method: vb|mcmc` warns on non-conjugate components; `labels` shape `(T, N)`; unknown registered names → `ConfigError` listing the registered names and their providing package.

**`regimekit` `schema_version: 2`** = `FinanceConfig(ProjectConfig)`: adds `data.source.kind: provider` (`provider, symbols, interval, start, end, adjusted, field`), covariate sources of kind `provider` (`provider, series`), `data.cache`, `evaluation.strategy`, `output.write: strategy`; defaults `data.clock: trading`, `data.preparation.observation_transform: log_returns`, `missing_observations: drop`. Migration 1 → 2: drop `profile`; `data.primary` → `data.source {kind: provider, …}`; `covariates[*].provider/series` → `covariates[*].source`; `bars_per_day` → `clock_options.bars_per_day`; `preparation.returns: log` → `observation_transform: log_returns`; `missing_returns` → `missing_observations`; `init_mean_sojourn_days` kept (finance key) ; `rollvol_kmeans` → `rolling_kmeans`; `backtest.initial_train_years/rolling_years` → `initial_train/rolling_window` (years under `trading`); emission `student_t` gains explicit `scale_by_dt: true`.

### 15.5 Extras and dependencies *(rewritten 1.2)*

| Package | Extra | Dependencies | Enables |
|---|---|---|---|
| `hsmmkit` | — | Part I runtime set minus provider packages | core |
| `hsmmkit` | `plot` | `matplotlib>=3.9` | `hsmmkit.plot` |
| `hsmmkit` | `sklearn` | `scikit-learn>=1.6` | `hsmmkit.interop.sklearn` |
| `hsmmkit` | `bayes` | `numpyro>=0.20`, `arviz>=0.21` | MCMC, `InferenceData` export (`mcmc` kept as deprecated alias until 0.5.0) |
| `hsmmkit` | `xarray` | `xarray>=2025.1` | xarray source, `to_xarray()` |
| `hsmmkit` | `gpu` | unchanged | |
| `regimekit` | — | `hsmmkit` (pinned minor range), `pandas-datareader`, `fredapi` | finance application |
| `regimekit` | `providers` | `yfinance`, `curl_cffi` | network providers (unchanged from Part I) |

### 15.6 Presets

`hsmmkit.presets` ships validated YAML plus a synthetic fixture generator per preset: `core/segmentation_gaussian_ltr` (Appendix H), `core/counts_poisson_hazard`, `core/msar2_hamilton`, `core/two_regime_gaussian` (Appendix E in Gaussian form). `regimekit` registers `finance/dax_3state_gbm_nb` (Appendix A) and `finance/two_regime_sanity` (Appendix E) through the `hsmmkit.preset` entry point. Presets are the fixtures of the tutorials and of the recovery suite extensions.

### 15.7 Amendments to Part I *(status 1.2: applied 2026-09-14, released in regimekit 0.1.0)*

| Part I section | Amendment | Affected tickets |
|---|---|---|
| 5.4 `PreparedDataset.returns` | renamed to `observations`; `support`, `clock`, `labels`, `boundaries`, `group` with defaults | E1-T01, E2-T07 |
| 4.1 / 5.6 clocks | `steps` clock (`Δ_t ≡ 1`) | E2-T06 and others |
| 5.3 layout | `regimekit/finance/` subpackage; providers there | E0-T01, E2-T01…T09 |
| 5.5 `EmissionFamily` | `support`, `scale_by_dt` | E3-T01 |
| 5.5 `Estimator` | `estimation/map.py`, `FitResult.estimator` | E1-T05, E5-T03, E5-T04 |
| 5.7 config | `profile` key, JSON-Schema export | E1-T02a/b |
| 5.8 bundle | manifest records `profile`, `schema_version` | E8-T01 |
| 4.3 `mean_sojourn_years` | `mean_sojourn` (clock units) | E3-T03, E3-T04, E5-T04 |

The `profile` key and the `finance/` subpackage are transitional: Section 15.8 removes both.

### 15.8 Repository split *(new 1.2)*

**Sequence (D23).** Phase 1 (*decouple*, in the `RegimeDetectionAndForecast` repository, tickets E10-T01…T07): registries and extension points are introduced in the core packages, every finance-specific element is moved under `regimekit/finance/`, a neutral-name scanner shrinks its allow-list to zero and a `core-only` CI job proves the core runs with `regimekit.finance` blocked. Phase 2 (*extract*, E10-T08…T09): the core paths are extracted with `git filter-repo` into the new `hsmmkit` repository and renamed; `RegimeDetectionAndForecast` deletes its copy of the core, flattens `regimekit.finance` into `regimekit` and depends on `hsmmkit`.

**Move map.**

| Part I element (regimekit 0.2.0) | Destination |
|---|---|
| `config/` generic sections, `yaml_io`, `migrate`, JSON-Schema export | `hsmmkit.config` |
| `PriceSource`, `CacheCfg`, `StrategyCfg`, `bars_per_day`, `init_mean_sojourn_days`, `profile` | `regimekit.config` (`profile` dropped in schema v2) |
| `data/contracts.py` | `hsmmkit.data.contracts` (`ObservationSet` generalized) |
| `data/calendar.py` — grid, alignment, lagging, `steps`, `calendar` | `hsmmkit.data.calendar`, `hsmmkit.data.clocks` |
| `trading` clock, weekday `future_grid` | `regimekit.clocks` |
| `data/cache.py` (snapshots, offline replay) | `regimekit.cache` |
| `data/prepare.py`, `transforms.py` (log, diff, pct, standardization) | `hsmmkit.data` |
| `transforms.log_returns` | `regimekit.transforms` (observation transform `log_returns`) |
| `data/quality.py` generic checks / exchange-calendar, price and adjustment checks | `hsmmkit.data.quality` / `regimekit.quality` |
| `finance/providers/*` | `regimekit.providers` |
| `model/emission/gbm.py` | `regimekit.emission.gbm` (alias of `hsmmkit` `gaussian`) |
| `model/emission/student_t.py`, `product.py`; `model/*` (rest), `statespace`, `engine`, `estimation` (`em.py` from E5-T06 included), `inference` | `hsmmkit` |
| `estimation/init.py` `rollvol_kmeans` | `hsmmkit` `rolling_kmeans` |
| `evaluation/metrics` (predictive, model, labels), `backtest.py` (walk-forward), `compare.py`, `report.py` (core sections) | `hsmmkit.evaluation` |
| `evaluation/strategies.py`, `metrics/strategy.py`, strategy report section | `regimekit.strategies`, `regimekit.metrics` |
| `persistence/` | `hsmmkit.persistence` (bundle v2; regimekit writes an extension file) |
| `cli/app.py`, `project.py` | `hsmmkit` (app factory, `Project`) + `regimekit` (`regimekit` CLI, `FinanceProject`) |
| `statespace/factorial.py` (E9-T02 prototype) | `hsmmkit` (experimental) |
| MS-GARCH spike (E9-T03) | `regimekit.emission.msgarch` (carry hooks stay in `hsmmkit`) |
| `tests/reference`, core unit/property/golden/recovery/benchmark tests | `hsmmkit` (GBM uses re-expressed as `gaussian`) |
| finance tests, cassettes, Appendix A fixtures, e2e example, tutorial | `regimekit` |
| `docs/adr/0001…0013`, `0022`, `0023` | both repositories (shared history); new ADRs 0014–0021, 0024–0027 (then 0028+), stored where they apply |
| `spec/tickets/`, `spec/concept/`, `spec/STATUS.md`, `CLAUDE.md` | both repositories; `CLAUDE.md` and `STATUS.md` rewritten for `hsmmkit` |

---

## 16. Engineering guideline deltas

### 16.1 Stack additions (each behind an extra, each with an ADR)

| Area | Choice | Notes |
|---|---|---|
| Plotting | `matplotlib ≥ 3.9` | `plot` extra; no seaborn/plotly in core |
| sklearn interop | `scikit-learn ≥ 1.6` | `sklearn` extra; only `BaseEstimator`, `check_estimator` subset in tests |
| Bayesian | `numpyro ≥ 0.20`, `arviz ≥ 0.21` | `bayes` extra; NumPyro model built from `log_joint(u)` |
| Labelled arrays | `xarray ≥ 2025.1` | `xarray` extra |
| Assignment (Hungarian) | `scipy.optimize.linear_sum_assignment` | promote scipy to a runtime dependency for `evaluation.metrics.clustering` (ADR-0020) — or vendor a 60-line implementation; decision in E14-T03 |
| Scaffold | `copier` | dev group only; template under `templates/family-plugin/` |
| Extraction | `git-filter-repo` | one-off tooling for E10-T08; not a project dependency |

### 16.2 Coding standards (additions)

- Lazy imports for every extra (`hsmmkit.utils.optional_import("matplotlib", extra="plot")`); importing the core MUST NOT import them (NFR-11, tested with `python -X importtime` allow-list).
- Every family, duration, metric, source and clock implementation — built-in or plug-in, `regimekit`'s included — is parametrized into the contract suites of `hsmmkit.testing`; a new implementation without a contract-suite entry fails CI.
- Estimators MUST NOT special-case families; capability discovery goes through the optional protocol methods (`m_step is not None`, `conjugate_prior() is not None`).
- Batched code is written for `B` sequences and called with `B = 1` for single sequences; no single-sequence fast paths.
- Deprecations use `warnings.warn(..., DeprecationWarning)` with the removal version and are listed in `docs/guide/deprecations.md` (NFR-14).
- No `hsmmkit` module may contain a finance term (FR-G01); a neutral-name scanner test enforces this (E10-T01).

### 16.3 JAX conventions (additions)

9. Batch dimension `B` is handled by `jax.vmap` over the per-sequence engine functions; static shapes are `(T_bucket, S)`; bucket edges are static config.
10. EM E-step accumulators (`γ`, pooled `ξ` counts, sufficient statistics) are computed inside the scan and reduced on the fly; `(T, S, S)` is never materialized (Section 6.3 rule 8 extended).
11. Label and boundary masks are dynamic inputs (arrays of `−inf/0`), not static configuration, so a compiled engine serves labelled and unlabelled sequences alike.
12. `OnlineFilter.update` is a jitted single step; the ring buffer for fixed-lag smoothing is a fixed-size array with a write pointer (no Python lists in the hot path).

### 16.4 Testing strategy (additions)

| Layer | Tests | TDD |
|---|---|---|
| new families (`gaussian`, `student_t`, `poisson`, `bernoulli`, `categorical`, `msar`, `mixture`, `ou`, `merton`) | contract suite + scipy agreement + closed-form M-step vs L-BFGS optimum | mandatory |
| `topology`, `constraints`, `batch` | property tests (row sums, structural zeros, padding invariance, constraint feasibility) | mandatory |
| `em`, `gem`, `vb` | monotonicity of `ℓ`/ELBO, parity with MAP (Appendix G), pruning on over-specified `N` | mandatory |
| `ffbs`, `mcmc` | path-sample marginals vs smoothed posteriors within MC error; Gibbs vs NUTS posterior means on the Appendix E fixture | recommended |
| `online` | bitwise equality with offline filter; fixed-lag smoothing equality; state round-trip | mandatory |
| `interop`, `plot` | sklearn `check_estimator` subset; `to_xarray/to_pandas` round-trip; plots return `Axes` and render headless (`Agg`) | recommended |
| `plugins` | a fixture plug-in package installed in a temp venv is discovered and passes the contract suite; `regimekit` passes it as a real plug-in | recommended |
| split (E10) | neutral-name scanner; `core-only` job with `regimekit.finance` blocked; golden values of v0.2.0 reproduced bitwise | mandatory |
| recovery (extended) | scenarios: Gaussian full-cov 3-state batch of 50; Poisson 2-state hazard; MS-AR(2) Hamilton; left-to-right topology with labels; VB on 4 configured/3 true states | n/a (nightly) |

Golden references (`tests/reference/`) gain: batched forward-backward, EM M-steps for Gaussian/Poisson/Categorical, Guédon E-step segment posteriors, FFBS sampling on a 2-state/4-step enumerable example, Hungarian matching.

### 16.5 CI/CD (additions)

`hsmmkit`: jobs `extras` (installs each extra in isolation and runs its tests; verifies NFR-11), `plugin` (builds and installs the template plug-in), `bayes` (nightly, short chains), `downstream` (nightly; NFR-18). `regimekit`: CI installs `hsmmkit` from the pinned range (and, in a nightly job, from `hsmmkit` main). Coverage gate unchanged (≥ 90 % overall, ≥ 85 % per package, now including `online`, `selection`, `interop`).

### 16.6 Documentation (additions)

`hsmmkit`: `docs/guide/core/` (quickstart with a synthetic labelled batch, families, topology, supervision, online), `docs/guide/extending/` (writing a family/source/clock, running the contract suites, publishing a plug-in; `regimekit` as worked example), `docs/math/` pages for Sections 14.x, `docs/guide/estimators.md` (decision table MAP/EM/VB/MCMC with cost and assumptions), `docs/guide/deprecations.md`; tutorials segmentation and counts. `regimekit`: the Part I guide (finance) and the finance tutorial, executed in CI from presets.

### 16.7 Architecture Decision Records *(renumbered 1.2; drafted in the ticket named, accepted per epic)*

| ADR | Repo | Decision (proposed default) | Ticket |
|---|---|---|---|
| 0014 | both | Two-repository split: `hsmmkit` library + `regimekit` application; decouple-then-extract (D12, D22, D23); supersedes ADR-0012 | E10-T01 |
| 0015 | both | Clock registry; `steps` and `calendar` in `hsmmkit`, `trading` in `regimekit` (D13); amends ADR-0013 | E10-T03 |
| 0016 | hsmmkit | Registries and config extension by subclassing instead of a `profile` key (D24) | E10-T02 |
| 0017 | hsmmkit | Plug-in discovery via entry points; `hsmmkit.testing` as public contract (D15, NFR-14) | E16-T04 |
| 0018 | hsmmkit | EM/GEM/VB/MCMC as peer estimators behind `Estimator`; MAP remains reference (D17) | E13-T02 |
| 0019 | hsmmkit | Supervision as engine-level masks, no supervised code path (D18) | E12-T06 |
| 0020 | hsmmkit | scipy as runtime dependency (or vendored assignment) for clustering metrics | E14-T03 |
| 0021 | hsmmkit | Multi-sequence support via `vmap` and length buckets; contracts stay single-sequence (D14) | E12-T01 |
| 0024 | hsmmkit | ArviZ as the Bayesian output format; NumPyro as the only sampler backend in 0.x | E13-T06 |
| 0025 | hsmmkit | Deprecation policy and semver scope from 0.3.0 (NFR-14) | E17-T03 |
| 0026 | hsmmkit | Online EM algorithm choice (spike outcome) | E13-T08 |
| 0027 | hsmmkit | HDP-HSMM go/no-go and interface diff (spike outcome) | E13-T09 |

The former draft entry "0018 approximate inference for factorial models" is dropped: ADR-0008 was finalized by E9-T02. ADR-0022 (in-scan emissions, E9-T03) and ADR-0023 (endogenous covariates, E9-T04) already exist from Part I and are not Part II ADRs; Part II therefore skips those two numbers.

### 16.8 Definition of Done (additions)

7. New implementations are registered in the appropriate contract suite parametrization.
8. Extras remain lazily imported (import-time test passes).
9. Public protocol changes list the deprecation path (from 0.3.0).
10. `hsmmkit` changes keep the nightly `downstream` job green or ship with a matching `regimekit` ticket.

---

## 17. Implementation plan (Part II)

### 17.1 Milestones and release mapping

| Milestone | Content | Release | Exit criteria |
|---|---|---|---|
| M7a Decouple | E10-T01…T07 (in `RegimeDetectionAndForecast`) | — | neutral-name allow-list empty; `core-only` job green; finance golden outputs unchanged |
| M7b Extract | E10-T08, E10-T09 | `hsmmkit` 0.1.0, `regimekit` 0.3.0 | `hsmmkit` repo with history; `regimekit` depends on it; v0.2.0 golden values and bundles reproduced |
| M8 Family catalogue | E10-T10, E10-T11, E11 | `hsmmkit` 0.2.0 (with M9) | all Appendix F families marked 0.2.0 pass the contract suite; MS-AR(2) recovery passes |
| M9 Sequences & structure | E12 | `hsmmkit` 0.2.0 | batch of 50 sequences fits under NFR-12; topology/left-censoring/labels golden tests pass |
| M10 Estimators | E13 | `hsmmkit` 0.3.0 (with M11) | EM/GEM/VB/MCMC parity per Appendix G; estimator decision table documented |
| M11 Selection & supervision | E14 | `hsmmkit` 0.3.0 | grid selection and CV tables; segmentation metrics; supervised recovery scenario passes |
| M12 Online & interop | E15, E16 | `hsmmkit` 0.4.0 | online filter bitwise equality; sklearn wrapper; plots; `regimekit` discovered as plug-in |
| M13 General release | E17 | `hsmmkit` 0.5.0, `regimekit` 0.5.0 | docs restructured; PyPI + conda-forge; ADRs 0014–0021 and 0024–0027 accepted |

`regimekit` 0.4.0 ships OU, Merton and correlated multi-primary configurations (E11-T07, E11-T09) on `hsmmkit` 0.2.0.

### 17.2 Epics and tickets

Sizes and rules as in Section 7.2. Every ticket includes tests, docs and — where a decision is taken — an ADR. Column *Repo*: where the ticket is implemented (`RegimeDetectionAndForecast*` = in the `RegimeDetectionAndForecast` repository on code that moves to `hsmmkit` in E10-T08; `RegimeDetectionAndForecast` = that repository after the split).

#### E10 — Decouple, extract and generalization baseline

| ID | Repo | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|---|
| E10-T01 | RegimeDetectionAndForecast* | Split guardrails: neutral-name scanner with shrinking allow-list, `core-only` CI job (finance import blocked), `finance` test marker; ADR-0014 | S | Part I | FR-G01, FR-G08, D23 | scanner fails on a new finance term in a core module; `core-only` job runs and reports the baseline failures as expected-xfail list |
| E10-T02 | RegimeDetectionAndForecast* | Component registries (emission, duration, metric, source, clock, transform, init, covariate forecaster, preset) and config extension by subclassing; `FinanceConfig` with schema v2 and v1 → v2 migration; ADR-0016 | L | E10-T01 | FR-G05, FR-G07, D24 | Appendix A (v1) migrates to v2 and builds the identical `ModelSpec`; core `ProjectConfig` has no finance field; unknown names list registered names |
| E10-T03 | RegimeDetectionAndForecast* | Clock registry; `steps` (default) and `calendar` in core; `trading` and its future grid in finance; ADR-0015 | M | E10-T02 | FR-G02, D13 | Appendix E numbers unchanged under `trading`; no literal `252` in core; clock contract suite |
| E10-T04 | RegimeDetectionAndForecast* | Generic `ObservationSet`, `Source` protocol and numpy/pandas/xarray/csv/parquet sources, `source_contract`; `DataProvider` re-based as source kind `provider`; cache/snapshots moved to finance | L | E10-T02 | FR-G04, FR-PL02 | contract suite parametrized over all sources and providers; `DataFrame` with/without time index loads under `steps`/`calendar`; provider tests unchanged |
| E10-T05 | RegimeDetectionAndForecast* | Generic `Preparer` with observation-transform hook (`log_returns` registered by finance), observation supports and validation, quality report split | M | E10-T03, E10-T04 | FR-G03 | each support has an accepting and a rejecting fixture; finance `PreparedDataset` bitwise unchanged |
| E10-T06 | RegimeDetectionAndForecast* | Core `gaussian` emission (`diag`, `M ≥ 1`, `scale_by_dt`, `ito_correction`); `gbm` registered by finance as alias; core tests re-expressed with `gaussian` | M | E10-T02 | FR-EM01 (diag), FR-EM06 | GBM golden outputs bitwise unchanged; `gaussian` with GBM options equals GBM ≤ 1e-14 |
| E10-T07 | RegimeDetectionAndForecast* | Neutral init (`rolling_kmeans`), walk-forward origin hooks, strategies to finance, bundle v2 (provenance + extension files), CLI app factory, `Project`/`FinanceProject` | L | E10-T03…T06 | FR-G01, FR-G08 | allow-list empty; `core-only` job green; CLI/Project parity tests of E8-T02 unchanged for finance |
| E10-T08 | hsmmkit | Extract `hsmmkit` with `git filter-repo`, rename package/CLI/errors/env vars, CI incl. `downstream` job, docs; release `hsmmkit` 0.1.0 | M | E10-T07 | FR-G09, NFR-18 | `git log --follow` shows Part I history; full suite green; `import hsmmkit` imports no finance module |
| E10-T09 | RegimeDetectionAndForecast | `regimekit` on top of `hsmmkit`: delete core copy, flatten `finance/`, pin dependency, register components, bundle v1 compatibility; release `regimekit` 0.3.0 | M | E10-T08 | FR-G07, FR-G08, FR-G09 | v0.2.0 golden outputs bitwise; 0.1.x/0.2.x bundles load; all Appendix A flows pass |
| E10-T10 | hsmmkit | Complete `hsmmkit` schema (all §15.4 keys) with feature gates; JSON-Schema export with registry enums; presets registry and `core/*` entries | M | E10-T08 | FR-G05, FR-G06, FR-IO05 | `hsmmkit presets show core/segmentation_gaussian_ltr` prints Appendix H; JSON Schema validates Appendix H and regimekit's Appendix A (v2) |
| E10-T11 | hsmmkit | Lazy optional imports, extras in `pyproject.toml` (`mcmc` → `bayes` alias), import-time test, `extras` CI job | S | E10-T08 | NFR-11 | `python -c "import hsmmkit"` imports none of the extras' packages; each extra's tests run in isolation |

#### E11 — Emission family catalogue

| ID | Repo | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|---|
| E11-T01 | hsmmkit | `family_contract` suite in `hsmmkit.testing`: log-prob vs scipy, sampler moments, `Δ`-scaling, support validation, covariate links, param-spec round-trip, optional-method consistency | M | E10-T08 | FR-EM08, FR-PL02 | `gaussian`, `student_t`, `product` pass; suite importable by external packages |
| E11-T02 | hsmmkit | Gaussian `full` covariance (Cholesky bijector) | M | E11-T01, E10-T06 | FR-EM01 | log-pdf vs `scipy.stats.multivariate_normal` ≤ 1e-10; Cholesky Jacobian vs autodiff ≤ 1e-8; `diag` outputs bitwise unchanged |
| E11-T03 | hsmmkit | Multivariate Student-t | S | E11-T02 | FR-EM02 | vs `scipy.stats.multivariate_t`; `ν → ∞` limit; univariate Part I outputs unchanged |
| E11-T04 | hsmmkit | Poisson, Bernoulli, Categorical with covariate links and closed-form sufficient statistics / M-steps / conjugate priors | M | E11-T01 | FR-EM03, FR-EM07 | contract suite; M-step closed form equals numerical optimum ≤ 1e-8; exposure `Δ_t` tested under `calendar` |
| E11-T05 | hsmmkit | MS-AR(p): preparer lag generation, per-state coefficient matrices, first-`p` masking; Hamilton mean-adjusted variant for `p ≤ 2` | M | E11-T02, E10-T05 | FR-EM04 | 2-state MS-AR(2) recovery; `p = 3` mean-adjusted → `ConfigError`; 1-state AR(p) matches `statsmodels` ≤ 1e-6 |
| E11-T06 | hsmmkit | Per-state finite mixture of a base family with in-state ordering constraints | M | E11-T02 | FR-EM05 | contract suite; 1-component mixture equals base family bitwise |
| E11-T07 | RegimeDetectionAndForecast | OU (exact discretization) and Merton jump-diffusion | M | E11-T06, E10-T09 | FR-EM06 | OU density vs closed form ≤ 1e-12; Merton with `λ = 0` equals GBM; both pass `family_contract` |
| E11-T08 | hsmmkit | Joint/product emission composition: `ProductEmission` with per-block families of any support and full-covariance Gaussian blocks | M | E11-T02, E11-T04 | FR-M12, FR-EM01 | two-series recovery of a 2×2 covariance per state; E9-T01 regression values unchanged |
| E11-T09 | RegimeDetectionAndForecast | Correlated multi-primary finance configuration (`gbm` with `covariance: full`) | S | E11-T08, E10-T09 | FR-M12, FR-EM06 | two-index synthetic recovery; `M = 1` golden outputs unchanged |

#### E12 — Sequences, topology, censoring, constraints (all `hsmmkit`)

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E12-T01 | `SequenceBatch`, padding to buckets, `A_t = I`/`ℓ_t = 0` padding semantics, `vmap`ped engine wrappers for both engines, per-sequence unpadding; ADR-0021 | L | E10-T10 | FR-SQ01, FR-SQ03, FR-SQ04 | batched loglik equals sum of single-sequence logliks ≤ 1e-10; padding invariance property test; recompilations ≤ number of buckets |
| E12-T02 | Batch estimation (MAP) with per-group initial distributions; batched simulate | M | E12-T01 | FR-SQ01, FR-SQ02 | recovery on 50 × `T = 200` matches single-sequence recovery on `T = 10 000` within Appendix G; NFR-12 benchmark recorded |
| E12-T03 | Transition topology `Ω`: shorthands, masked logits with per-row reference category, absorbing states in both engines | M | E10-T10 | FR-DU01, FR-DU02 | left-to-right model never decodes a backward jump; absorbing state has `a_i = 0`; parameter count matches Appendix F; `full` bitwise equals Part I |
| E12-T04 | Left-censoring: residual-life initial duration in the explicit engine | S | E10-T10 | FR-DU04 | explicit residual-life loglik equals expanded `stationary` on shifted-NB within Appendix D tolerance |
| E12-T05 | Per-state duration families and duration-dependent jumps in the explicit engine | M | E12-T03 | FR-DU03, FR-DU05 | mixed geometric/gamma model recovers both families; `c_ij` sign recovered; option rejected under `engine: expanded` |
| E12-T06 | Label and boundary constraints as engine masks, `ConstraintError`, label-informed initialization; ADR-0019 | M | E12-T01 | FR-SV01, FR-SV02, FR-SV03 | fully labelled data reproduces labels under Viterbi; infeasible boundary raises with step index; `init: labels` within 15 % |

#### E13 — Estimators (all `hsmmkit`)

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E13-T01 | E-step accumulators inside the scan (expanded); explicit-engine segment posteriors `η_t(j,d)` | L | E12-T01 | FR-ES01, FR-ES02 | accumulators vs NumPy reference ≤ 1e-8; no `(T, S, S)` allocation |
| E13-T02 | `EMEstimator` (expanded): closed-form M-steps, ECM, tempering, restarts, convergence; supersedes E5-T06; ADR-0018 | L | E13-T01, E11-T04 | FR-ES01, FR-ES07, NFR-13 | monotone `ℓ`; parity with MAP per Appendix G on Gaussian/Poisson/Categorical; tempered EM beats plain EM on a multimodal fixture (≥ 4/5 seeds); E5-T06 tests pass |
| E13-T03 | `GEMEstimator` (explicit, Guédon): jump, nonparametric and parametric duration M-steps, right-censoring | L | E13-T01, E12-T04, E12-T05 | FR-ES02 | monotone `ℓ`; nonparametric pmf TV ≤ 0.05 at `T = 20 000`; parity with MAP for gamma durations |
| E13-T04 | FFBS for both engines; `decode --samples R` | M | E12-T01 | FR-ES03 | path frequencies vs smoothed marginals within 3 MC SE; enumerable golden test |
| E13-T05 | `VBEstimator`: conjugate updates, structured `q(s)`, ELBO, pruning, VB-EM hybrid | L | E13-T01, E11-T04, E13-T02 | FR-ES04, NFR-13 | ELBO monotone; means vs MAP within Appendix G; 4-state fit on 3-state data prunes one state; hybrid components listed |
| E13-T06 | `MCMCEstimator` (`bayes` extra): NUTS on `log_joint(u)`, blocked Gibbs, relabelling, `InferenceData`; ADR-0024 | L | E13-T04, E13-T05, E10-T11, E12-T02 | FR-ES05, FR-IO03, NFR-17 | Appendix E fixture: means within 2 posterior SD, `R̂ ≤ 1.05`; Gibbs and NUTS agree within 2 MC SE; seeded chains reproducible |
| E13-T07 | Estimator parity suite, `docs/guide/estimators.md`, `compare --estimators` | M | E13-T02, E13-T03, E13-T05, E13-T06 | FR-ES06, FR-EV04 | parity table generated in CI; documentation lists assumptions, cost and failure modes |
| E13-T08 | Online/recursive EM spike; ADR-0026 | S | E13-T02, E15-T01 | FR-ON05 | ADR with algorithm choice; prototype tracks drifting `σ`; no merge beyond ADR |
| E13-T09 | HDP-HSMM design spike; ADR-0027 | S | E13-T06 | 12.2 | ADR with go/no-go and interface diff; no code merged |

#### E14 — Selection, diagnostics and evaluation (all `hsmmkit`)

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E14-T01 | `Splitter` interface: `SequenceKFold`, walk-forward re-based on it; CV log score / CRPS aggregation | M | E12-T02 | FR-EV03 | walk-forward results bitwise unchanged; grouped folds never split a group; look-ahead test retained |
| E14-T02 | Grid model selection (`compare --grid`) | M | E14-T01, E13-T07 | FR-EV04 | grid on the Appendix E fixture selects `N = 2` by BIC and CV; table schema documented |
| E14-T03 | Segmentation and clustering metrics; Hungarian matching; ADR-0020 | M | E10-T08 | FR-EV01, FR-EV02 | closed-form cases; ARI/NMI vs `sklearn.metrics` ≤ 1e-12 (dev-only comparison) |
| E14-T04 | Parametric bootstrap SE and nested-model bootstrap tests | M | E12-T02 | FR-EV05 | bootstrap SE within 20 % of simulation SE; documented caveat |
| E14-T05 | Pseudo-residuals and posterior predictive checks; fit-report integration | M | E13-T04, E11-T04 | FR-EV06 | correct model passes Shapiro–Wilk on ≥ 90 % of seeds; PPC flags Gaussian on Student-t data |
| E14-T06 | Extended recovery scenarios (16.4) and `hsmmkit selftest` | M | E13-T02, E13-T03, E13-T05, E12-T06, E11-T05 | NFR-04 | five new scenarios pass ≥ 4/5 seeds nightly; selftest < 2 min |

#### E15 — Online inference (all `hsmmkit`)

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E15-T01 | `OnlineFilter`: jitted single-step update, `OnlineState`, save/load, `spec_hash` | M | E10-T08 | FR-ON01, FR-ON04, NFR-15 | bitwise equality over a 5000-step replay; restored state continues bitwise |
| E15-T02 | Fixed-lag smoothing with ring buffer | M | E15-T01 | FR-ON02, NFR-15 | equals the offline smoother on the prefix `y_{1:t}` at `t − L` ≤ 1e-12; cost linear in `L` |
| E15-T03 | Alert rules; structured events; online forecast | S | E15-T02 | FR-ON03 | exactly one `on` and one `off` event; online forecast equals offline forecast |
| E15-T04 | `hsmmkit stream` CLI and `Project.online()` | S | E15-T03 | FR-C02, FR-C03 | piped fixture integration test; exit codes |
| E15-T05 | (v1.x) Online EM implementation | M | E13-T08 | FR-ON05 | tracks drift with lag < 500 steps; frozen-schedule equivalence |

#### E16 — Interoperability, visualization, plug-ins

| ID | Repo | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|---|
| E16-T01 | hsmmkit | scikit-learn `HSMM` estimator (Appendix I) | M | E12-T02, E12-T06, E10-T11 | FR-IO01 | applicable `check_estimator` checks pass; `lengths` semantics match `hmmlearn` |
| E16-T02 | hsmmkit | `to_pandas()`/`to_xarray()` for all `*Result` objects; `hmmlearn` import | M | E11-T02, E11-T04, E10-T11 | FR-IO02, FR-IO04 | xarray round-trip bitwise; imported `GaussianHMM` loglik ≤ 1e-8 |
| E16-T03 | hsmmkit | `hsmmkit.plot` | M | E16-T02 | FR-VZ01, FR-VZ02 | every helper renders headless and returns `Axes` |
| E16-T04 | hsmmkit | Entry-point discovery, `hsmmkit new-family` scaffold, `plugin` CI job; ADR-0017 | M | E11-T01, E10-T04 | FR-PL01…PL03, NFR-14 | template plug-in discovered and passes `family_contract`; duplicate names rejected with provenance |
| E16-T05 | hsmmkit | Presets completion and core tutorials (segmentation, counts) | M | E10-T10, E11-T05, E12-T05, E12-T06, E13-T03, E13-T04, E14-T01, E14-T03…T05, E15-T03, E16-T03 | FR-G06, NFR-16 | tutorials run in CI from presets |
| E16-T06 | RegimeDetectionAndForecast | `regimekit` as entry-point plug-in: registration via entry points, all finance components in the `hsmmkit` contract suites, `finance/*` presets | S | E16-T04, E11-T07 | FR-PL01, FR-G06 | `hsmmkit presets list` in an env with `regimekit` shows `finance/*`; no import-time registration left |

#### E17 — Documentation, distribution, releases

| ID | Repo | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|---|
| E17-T01 | hsmmkit | Docs restructure: core and extending guides, math pages 14.x, estimator table, deprecations, API reference | M | E16-T05, E13-T07 | NFR-16, NFR-07 | `mkdocs build --strict`; every Part II FR/NFR referenced |
| E17-T02 | hsmmkit | PyPI trusted publishing, conda-forge feedstock, extras smoke installs | S | E10-T11 | D22, NFR-06 | fresh-env installs succeed |
| E17-T03 | hsmmkit | ADRs 0014–0021 and 0024–0027 accepted, deprecation policy, benchmark baselines; ADR-0025 | S | all `hsmmkit` E10–E16 | NFR-14, NFR-12 | all ADRs accepted; benchmark JSON committed |
| E17-T04 | hsmmkit | Release `hsmmkit` 0.5.0; roadmap page | S | E17-T01…T03 | NFR-10 | tag `v0.5.0` published |
| E17-T05 | RegimeDetectionAndForecast | Finance docs on `hsmmkit` 0.5, finance tutorial from presets; release `regimekit` 0.5.0 | S | E17-T04, E16-T06, E11-T09 | NFR-16, NFR-18 | tutorial runs in CI; `regimekit` 0.5.0 installs with `hsmmkit` 0.5.0 from PyPI |

### 17.3 Dependency graph (critical path, Part II)

```
Part I ─▶ E10-T01 ─▶ E10-T02 ─▶ E10-T03 ─▶ E10-T05 ─▶ E10-T07 ─▶ E10-T08 ─▶ E10-T10 ─▶ E12-T01 ─▶ E13-T01 ─▶ E13-T02 ─▶ E13-T05 ─▶ E13-T06 ─▶ E13-T07 ─▶ E14-T02 ─▶ E17-T01 ─▶ E17-T04 ─▶ E17-T05
                       └▶ E10-T04 ↗         E10-T06 ↗          └▶ E10-T09 (regimekit 0.3.0)
                                                               └▶ E11-T01 ─▶ E11-T02 ─▶ E11-T04 ↗ (E13-T02)
                                                               └▶ E15-T01 ─▶ E15-T02 ─▶ E15-T03 ─▶ E15-T04
```

Parallel tracks after M7b: (a) families E11, (b) batch/structure E12, (c) online E15, (d) `regimekit` follow-ups (E11-T07, E11-T09). E13 needs (a) and (b). E16 and E17 close the releases.

### 17.4 Traceability (Part II requirement → tickets)

| Requirement | Tickets |
|---|---|
| FR-G01…G09 | E10-T01…T11 |
| FR-EM01…EM08 | E10-T06, E11-T01…T09 |
| FR-DU01…DU05 | E12-T03, E12-T04, E12-T05 |
| FR-SQ01…SQ04 | E12-T01, E12-T02 |
| FR-ES01…ES07 | E13-T01…T09 |
| FR-SV01…SV03 | E12-T06 |
| FR-ON01…ON05 | E15-T01…T05, E13-T08 |
| FR-IO01…IO05 | E16-T01, E16-T02, E13-T06, E10-T10 |
| FR-VZ01, VZ02 | E16-T03 |
| FR-PL01…PL03 | E16-T04, E16-T06, E11-T01, E10-T04 |
| FR-EV01…EV06 | E14-T01…T06 |
| NFR-11…NFR-18 | E10-T11, E12-T02, E13-T02/T05, E16-T04, E15-T01, E17-T01, E13-T06, E10-T08 |

---

## 18. Risks and mitigations (Part II)

| Risk | Impact | Mitigation |
|---|---|---|
| Extraction breaks finance behaviour | Silent numerical drift in `regimekit` | Decouple first in one repo (D23); v0.2.0 golden values stored and asserted bitwise in both repos; `downstream` nightly job (NFR-18) |
| Hidden finance coupling in the core (names, defaults, constants such as `252`) | Domain assumptions in "neutral" code | Neutral-name scanner with shrinking allow-list; `core-only` CI job with `regimekit.finance` blocked |
| Lost history or blame after extraction | Harder maintenance | `git filter-repo` with path list and renames recorded in E10-T08; `git log --follow` acceptance check |
| Version skew between the repos | `regimekit` breaks on an `hsmmkit` release | Minor-range pin; `downstream` job; NFR-14 deprecation cycle from 0.3.0 |
| Scope creep of a "general" package | Never-ending catalogue, diluted quality | Appendix F is the closed catalogue for 0.2.0–0.5.0; anything else is a plug-in; contract suite is the entry bar |
| EM for HSMM (Guédon) is intricate; explicit engine memory | Wrong M-steps, OOM | NumPy references first (16.4); monotonicity asserted; chunked scan of E4-T06 reused; MAP remains the fallback and the parity oracle |
| VB conjugacy limits | Users expect VB for covariate models | Hybrid documented per component in the fit report; estimator decision table |
| MCMC cost on long series | Hours per fit | Marginalized NUTS on `u` only; Gibbs for conjugate families; `bayes` extra opt-in; nightly job with short chains |
| Variable sequence lengths cause recompilation | Slow batch fits | Bucketing (FR-SQ04); benchmark NFR-12; bucket edges exposed |
| sklearn API impedance (2-D `X` vs sequences, covariates) | Confusing wrapper | `lengths` convention of `hmmlearn`; covariates as extra columns declared by index; wrapper documented as convenience, `Project` as the full API |
| Plug-in API instability | Third-party breakage | NFR-14 deprecation policy; `hsmmkit.testing` versioned with the package |
| Optional-dependency sprawl | Heavy installs, import-time regressions | NFR-11 import-time test; each extra tested in isolation |
| Label switching in EM/VB/MCMC | Unusable posteriors | Ordering constraints reused; pivot relabelling; restart dispersion diagnostics extended to ELBO/chains |

---

## 19. Open decisions with proposed defaults (Part II)

| # | Question | Proposed default / resolution |
|---|---|---|
| 10 | Where does GBM live? | *Resolved 1.2:* `regimekit.emission.gbm` as an alias of `hsmmkit` `gaussian`; core default emission is `gaussian` |
| 11 | Is `B` a dimension of `PreparedDataset`? | No; `SequenceBatch` wraps single-sequence datasets (D14) |
| 12 | Hungarian matching dependency | scipy as runtime dependency (ADR-0020), unless import-time cost > 50 ms — then vendored |
| 13 | VB scope | Conjugate families only; hybrid elsewhere; no VB for explicit parametric durations |
| 14 | MS-AR sequence start | Condition on the first `p` observations (mask); no stationary-start likelihood |
| 15 | Second sampler backend (BlackJAX) | Not in 0.x; `log_joint(u)` is backend-agnostic |
| 16 | Plotting library | matplotlib only; plotly as a plug-in example if demanded |
| 17 | Package name | *Resolved 1.2:* library `hsmmkit`; finance application keeps `regimekit` |
| 18 | `polars` support in sources | Via `pandas` conversion in 0.x; native source as plug-in |
| 19 | Deprecation start | 0.3.0 (first release with public protocols in `hsmmkit.testing`) |
| 20 | `calendar` clock unit | Years only in 0.x (bitwise Part I semantics); a `unit` option is a backlog item |
| 21 | `regimekit` package layout after the split | Flatten `regimekit.finance.*` into `regimekit.*` (no compatibility shim in 0.x) |
| 22 | First `hsmmkit` version | 0.1.0 = extraction release (behaviour of the `regimekit` 0.2.0 core); feature releases 0.2.0–0.5.0 as Section 17.1 |
| 23 | Bundle directory suffix | `hsmmkit`: `.hsmm/`; `regimekit` keeps `.rkmodel/` (suffix is a `ModelStore` option) |

---

## 20. References (additional)

- Zucchini, W., MacDonald, I. L. & Langrock, R. (2016). *Hidden Markov Models for Time Series: An Introduction Using R* (2nd ed.). CRC Press. — pseudo-residuals, covariates, model checking.
- Langrock, R. & Zucchini, W. (2011). Hidden Markov models with arbitrary state dwell-time distributions. *CSDA* 55(1). — phase-expansion approximation of HSMMs.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proc. IEEE* 77(2). — topologies, multiple sequences, Baum-Welch.
- Beal, M. J. (2003). *Variational Algorithms for Approximate Bayesian Inference*. PhD thesis, UCL. — VB for HMMs.
- Scott, S. L. (2002). Bayesian methods for hidden Markov models: recursive computing in the 21st century. *JASA* 97(457). — FFBS.
- Cappé, O. & Moulines, E. (2009). On-line expectation–maximization algorithm for latent data models. *JRSS B* 71(3).
- Hansen, B. E. (1992). The likelihood ratio test under nonstandard conditions: testing the Markov switching model of GNP. *J. Applied Econometrics* 7(S1).
- Stephens, M. (2000). Dealing with label switching in mixture models. *JRSS B* 62(4).
- Ueda, N. & Nakano, R. (1998). Deterministic annealing EM algorithm. *Neural Networks* 11(2).
- Truong, C., Oudre, L. & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing* 167. — segmentation metrics.
- Hubert, L. & Arabie, P. (1985). Comparing partitions. *J. Classification* 2. — ARI.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly* 2.
- McClintock, B. T. & Michelot, T. (2018). momentuHMM: R package for generalized hidden Markov models of animal movement. *Methods in Ecology and Evolution* 9(6). — covariate-driven HMM design reference.
- Visser, I. & Speekenbrink, M. (2010). depmixS4: an R package for hidden Markov models. *JSS* 36(7).
- O'Connell, J. & Højsgaard, S. (2011). Hidden semi Markov models for multiple observation sequences: the mhsmm package for R. *JSS* 39(4).
- Phan, D., Pradhan, N. & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. *arXiv:1912.11554*.
- Kumar, R. et al. (2019). ArviZ: a unified library for exploratory analysis of Bayesian models in Python. *JOSS* 4(33).
- Hoyer, S. & Hamman, J. (2017). xarray: N-D labeled arrays and datasets in Python. *J. Open Research Software* 5(1).
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *JMLR* 12.

---

## Appendix F — Family catalogue (closed for 0.2.0–0.5.0)

`p_e` = free parameters per state without covariates (`M`-dimensional where relevant); "M-step" = closed-form M-step available; "Conj." = conjugate prior for VB/Gibbs.

| Family | Package | Support | `p_e` | M-step | Conj. | Δ-scaling | Release |
|---|---|---|---|---|---|---|---|
| `gaussian` (diag) | hsmmkit | real | `2M` | yes | Normal-Gamma | optional | hsmmkit 0.1.0 (E10-T06) |
| `gaussian` (full) | hsmmkit | real | `M + M(M+1)/2` | yes | Normal-Wishart | optional | 0.2.0 |
| `student_t` (uv) | hsmmkit | real | `3` | no | no | optional | hsmmkit 0.1.0 (from Part I) |
| `student_t` (mv) | hsmmkit | real | `M + M(M+1)/2 + 1` | no (ECM) | no | optional | 0.2.0 |
| `poisson` | hsmmkit | count | `M` | yes | Gamma | exposure | 0.2.0 |
| `bernoulli` | hsmmkit | binary | `M` | yes | Beta | no | 0.2.0 |
| `categorical` | hsmmkit | categorical | `K − 1` | yes | Dirichlet | no | 0.2.0 |
| `msar` (p) | hsmmkit | real | `M + pM² + M(M+1)/2` | yes (weighted LS) | Normal-Wishart (coeffs) | no | 0.2.0 |
| `mixture(base, K)` | hsmmkit | as base | `K·p_e(base) + K − 1` | if base has | if base has | as base | 0.2.0 |
| `product` (blocks) | hsmmkit | per block | sum over blocks | if all blocks have | if all blocks have | per block | hsmmkit 0.1.0 (E9-T01), generalized 0.2.0 |
| `gbm` | regimekit | real | `2` | yes | Normal-Gamma | yes + Itô | regimekit 0.1.0 |
| `ou` | regimekit | real | `3` | no | no | exact | regimekit 0.4.0 |
| `merton` | regimekit | real | `5` | no | no | yes | regimekit 0.4.0 |
| `msgarch` | regimekit | real | `4` | no | no | discrete | spike (E9-T03) |

Duration families (Part I) gain conjugates: `geometric` exit rate — Gamma (on `λ`) or Beta (on per-step `a`, constant `Δ`); `negbin` phase advance — Beta; `nonparametric` pmf — Dirichlet; `poisson/gamma/weibull/lognormal` — none (point estimate in VB).

Parameter counts with topology `Ω` (order 1, no covariates): jump matrix `Σ_i max(0, Σ_j Ω_ij − 1)`; each transition covariate adds the same number; order-2 counts of Appendix B apply with `(N−1)` replaced by the admissible memory tuples of `Ω`.

## Appendix G — Additional numerical tolerances

| Check | Tolerance |
|---|---|
| Batched vs summed single-sequence log-likelihood | relative ≤ 1e-10 |
| Padding invariance of posteriors for `t ≤ T_b` | max-abs ≤ 1e-12 |
| Closed-form M-step vs numerical optimum of expected complete-data objective | relative ≤ 1e-8 |
| EM log-likelihood monotonicity | violations ≤ 1e-10 absolute |
| EM vs MAP (no priors), same restart start, conjugate families | `θ` relative ≤ 1e-4; loglik relative ≤ 1e-6 |
| GEM nonparametric pmf recovery (`T = 20 000`) | total variation ≤ 0.05 |
| VB posterior mean vs MAP (`T ≥ 20 000`, conjugate families, flat priors) | relative ≤ 2 % |
| VB ELBO monotonicity | violations ≤ 1e-8 |
| FFBS path frequencies vs smoothed marginals | within 3 MC SE |
| MCMC (Appendix E fixture, 4 chains × 1000 draws) | means within 2 posterior SD of truth; `R̂ ≤ 1.05`; ESS ≥ 200 |
| Online vs offline filter | bitwise |
| Fixed-lag smoothing vs offline smoothing on the same prefix `y_{1:t}` | max-abs ≤ 1e-12 |
| Segmentation metrics on closed-form cases | exact |
| ARI/NMI vs scikit-learn (dev-only) | ≤ 1e-12 |
| `hmmlearn` import log-likelihood | relative ≤ 1e-8 |
| Extended recovery scenarios | as Appendix D; covariance entries ≤ 15 % relative; AR coefficients ≤ 2 SE |
| Split: `regimekit` 0.3.0 vs v0.2.0 golden values; `hsmmkit` core vs Part I golden values (*new 1.2*) | bitwise |
| `gaussian` (GBM options) vs Part I GBM log-density (*new 1.2*) | ≤ 1e-14 absolute (bitwise where the operation order is kept) |

## Appendix H — Reference configuration, core (validates against `hsmmkit` schema 1, Section 15.4)

```yaml
schema_version: 1
project:
  name: segmentation_gaussian_ltr
  seed: 7
  output_dir: runs/segmentation_gaussian_ltr
  log_level: INFO

data:
  source:
    kind: parquet
    path: data/sensor_windows.parquet
    observation_columns: [acc_x, acc_y, acc_z]
    sequence_column: session_id
    time_column: null
  clock: steps
  sequences:
    group_by: null
    bucket_edges: [256, 512, 1024, 2048]
  labels:
    source: column
    column: activity
    allowed_sets: false          # exact labels where present, NaN = unlabelled
    boundaries: null
  preparation:
    observation_transform: null
    standardize_observations: true
    missing_observations: mask

model:
  n_states: 4
  engine: explicit
  d_max: 300
  topology: left_to_right
  left_censored: true
  initial_state: free
  emission:
    family: gaussian
    covariance: full
    scale_by_dt: false
    params:
      mean:  {mode: free, init: labels}
      chol:  {mode: free, init: labels, prior: {type: lognormal, sd: 1.0}}
  duration:
    per_state:
      - {family: gamma,     params: {shape: {mode: free, init: 2.0}, scale: {mode: free, init: 20}}}
      - {family: gamma,     params: {shape: {mode: free, init: 2.0}, scale: {mode: free, init: 40}}}
      - {family: geometric, params: {rate:  {mode: free, init_mean_sojourn_steps: 15}}}
      - {family: nonparametric, params: {pmf: {mode: free, prior: {type: dirichlet, alpha: 0.5}}}}
    duration_dependent_jumps: false
  transition:
    order: 1
    covariates: []
    prior: {type: dirichlet, alpha: 1.0}

estimation:
  method: gem
  max_iter: 300
  tol_f: 1.0e-7
  restarts: 4
  init: [labels, random]
  use_priors: true
  gem:
    tempering: {enabled: true, beta0: 0.3, steps: 10}

forecast:
  horizon: 10
  n_sim: 2000
  quantiles: [0.1, 0.5, 0.9]

online:
  lag: 20
  alerts:
    - {regime: 3, p_on: 0.8, p_off: 0.5}

evaluation:
  metrics: [log_score, bic, icl, boundary_f1, hausdorff, covering, hamming_matched, ari, nmi]
  boundary_tolerance: 5
  cv:
    splitter: sequence_kfold
    folds: 5
    seed: 7

output:
  write: [posterior, forecast, cv, report]
```

## Appendix I — scikit-learn wrapper mapping

| sklearn call | hsmmkit operation | Notes |
|---|---|---|
| `HSMM(n_states=3, emission="gaussian", duration="negbin", r=3, topology="full", estimator="em", **kw)` | builds `ProjectConfig` with `clock: steps` | `kw` maps 1:1 to config paths; `get_params()` returns the flattened config |
| `fit(X, lengths=None, y=None, sample_weight=None)` | `SequenceBatch` from `X` split by `lengths`; `y` (int labels, `-1` = unknown) → allowed-set constraints; `sample_weight` unsupported (raises) | `X` is `(ΣT_b, M)`; covariates via `covariate_columns=[...]` in the constructor |
| `predict(X, lengths=None)` | Viterbi regimes, concatenated | |
| `predict_proba(X, lengths=None)` | smoothed regime probabilities `(ΣT_b, N)` | `filtered=True` switches to filtered |
| `score(X, lengths=None)` | total log-likelihood | `score_samples` gives per-step predictive log-density |
| `decode(X, lengths=None)` | `(loglik, regimes)` | hmmlearn-compatible signature |
| `sample(n_samples, random_state)` | `simulate` | returns `(X, regimes)` |
| `forecast(X, horizon, lengths=None)` | `ForecastResult` for the last sequence | extension beyond the sklearn API |
| `result_` | the underlying `FitResult` | for everything the wrapper does not expose |
