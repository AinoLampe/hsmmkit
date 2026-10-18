---

# Part II — Extensions toward a general-purpose HSMM package

**Addendum to "regimekit — Concept and Implementation Plan"**

| Field | Value |
|---|---|
| Version | 1.1-draft (extension addendum to concept baseline 1.0) |
| Date | 2026-09-11 |
| Status | Proposed — for review; becomes binding on acceptance of ADR-0012 |
| Owner | Aino |
| Relationship to Part I | Sections 11–20 and Appendices F–I continue the numbering of Part I. Part I remains authoritative for release 0.1.0. Part II defines releases 0.2.0–0.5.0 and a small set of amendments to Part I (Section 15.7) that SHOULD be applied *before* the affected v1 tickets start, because they are cheap now and expensive later. |
| Amendment status | §15.7 was applied to the Part I ticket set on 2026-09-14 (Part I → v1.1; ADR-0012/0013 recorded by E0-T04). E10-T01 is reduced to the residual listed in §17.2. |
| Scope | Domain-neutral core; general emission, duration and transition families; multiple sequences; alternative estimators (EM, VB, MCMC); supervision; online inference; interoperability; plug-in system |

---

## 11. Motivation and scope of the extensions

### 11.0 How to use this part

The rules of Section 0 apply unchanged. New requirements use the prefixes `FR-G` (generalization), `FR-EM` (emissions), `FR-DU` (durations/transitions), `FR-SQ` (sequences), `FR-ES` (estimation), `FR-SV` (supervision), `FR-ON` (online), `FR-IO` (interoperability), `FR-VZ` (visualization), `FR-PL` (plug-ins), `FR-EV` (evaluation) and `NFR-11…17`. Tickets continue as E10–E17. Every Part II ticket references Part II requirement IDs; where a ticket amends Part I it lists the amended section and the cost of applying the amendment now versus after 0.1.0.

### 11.1 Why a general package

No maintained Python library combines hidden *semi*-Markov models with explicit durations in continuous or discrete time, covariate-driven transitions/durations/emissions, segment memory and autodiff estimation on JAX. `hmmlearn` and `pomegranate` are HMM-only without covariates; `dynamax` covers HMMs/SSMs in JAX but not semi-Markov durations; `pyhsmm` is Bayesian but NumPy-based and unmaintained; the strongest covariate-driven implementations are R packages (`momentuHMM`, `depmixS4`, `mhsmm`). The design of Part I (structured state index, builders, parameter modes, families behind protocols) already contains most of what a general core needs. Generalization is therefore mainly

1. removing financial assumptions from names, defaults and the data layer (the finance-specific parts become one *profile*),
2. adding emission, duration and transition families that other domains need,
3. supporting many independent sequences, labelled data and streaming use,
4. offering estimators beyond MAP (EM, variational Bayes, MCMC) behind the existing `Estimator` protocol, and
5. interoperating with the ecosystem users already have (scikit-learn, pandas/xarray, ArviZ, matplotlib) and letting third parties add families without forking.

### 11.2 Decisions taken for Part II (baseline for E10–E17)

| ID | Decision |
|---|---|
| D12 | **Core/profile split.** `regimekit` core is domain-neutral: observations `y_t ∈ 𝒴^M`, grid `Δ_t`, covariates. Everything price-, return-, provider- or strategy-specific lives in the profile subpackage `regimekit.finance`. The Part I model is the finance profile using the core with finance defaults. |
| D13 | **Three clocks.** `trading` and `calendar` (Part I) plus `steps` (`Δ_t ≡ 1`, parameters per step, no annualization). Every family MUST work under every clock. `steps` is the core default, `trading` the finance default. |
| D14 | **Multiple sequences are first-class.** Engines are `vmap`ped over a batch of `B` independent sequences with per-sequence length masks; a single sequence is `B = 1`. Contracts stay single-sequence; a `SequenceBatch` container holds many. |
| D15 | **Families are plug-ins.** Emission, duration, metric and data-source implementations are registered by name; the registry also reads `importlib.metadata` entry points. Third-party families run the same contract test-suite as built-in ones. |
| D16 | **Transition topology.** A boolean mask on the jump matrix (left-to-right, banded, absorbing, arbitrary) is a model-structure element. Duration-dependent jump probabilities are an explicit-engine option. |
| D17 | **Estimator peers.** MAP/L-BFGS remains the reference estimator. EM/generalized EM, variational Bayes and MCMC are peer `Estimator` implementations returning the same `FitResult`, with `estimator` recorded. |
| D18 | **Supervision as constraints.** Full or partial labels and known boundaries are expressed as masks on emissions and transitions consumed by the unchanged engines — no separate supervised code path. |
| D19 | **Online inference reuses the engine.** A stateful `OnlineFilter` provides exact filtering, fixed-lag smoothing and forecasting from the online state. Online parameter updates are v1.x. |
| D20 | **Interoperability, no GUI.** scikit-learn wrapper, pandas/xarray outputs, ArviZ export for sampled posteriors, matplotlib plotting behind an extra. Section 1.3's "no GUI" stands. |
| D21 | **Evaluation split.** Domain-neutral evaluation (scoring rules, information criteria, segmentation and clustering metrics, cross-validation) in core; strategies and financial metrics in `regimekit.finance`. |
| D22 | **One distribution.** A single wheel `regimekit` with extras `finance`, `providers`, `plot`, `sklearn`, `bayes`, `gpu`. No second distribution until a second profile exists. |

### 11.3 Target profiles (drive defaults, presets and test fixtures)

| Profile | Typical data | Families | Distinguishing features |
|---|---|---|---|
| Finance (Part I) | log-returns of one or several instruments, daily/intraday, macro covariates | GBM, Student-t, OU, Merton, MS-GARCH | continuous-time clocks, covariate-driven durations, strategies/backtests |
| Sensor / activity segmentation | multivariate real-valued windows, many labelled sequences | multivariate Gaussian, mixtures | `steps` clock, left-to-right or banded topology, partial labels, boundary metrics |
| Counts / reliability / operations | event counts per interval, exposure varies | Poisson, Bernoulli with exposure `Δ_t` | hazard-shaped durations, absorbing states |
| Econometrics | macro series, regime-switching autoregressions | MS-AR(p), Gaussian | Hamilton-type models, EM, MCMC, standard errors |
| Ecology / movement (plug-in example, not core) | step lengths and turning angles | Gamma, von Mises (third-party) | demonstrates the plug-in contract |

---

## 12. Extended goals and non-goals

### 12.1 Goals (in addition to G1–G6)

| ID | Goal |
|---|---|
| G7 | Be usable without any financial concept: any real-valued, count, binary or categorical sequence on a discrete grid. |
| G8 | Ship a catalogue of emission, duration and transition structures that covers the common HMM/HSMM literature (Appendix F), each testable by one shared contract suite. |
| G9 | Fit and decode collections of sequences of different lengths, with optional labels, at batch-level performance. |
| G10 | Offer EM, variational and MCMC estimation alongside MAP, with comparable outputs and documented trade-offs. |
| G11 | Support streaming use: update the posterior one observation at a time, smooth with a fixed lag, forecast from the live state, raise regime alerts. |
| G12 | Interoperate: scikit-learn API, pandas/xarray results, ArviZ posteriors, plotting helpers. |
| G13 | Be extensible by third parties through entry points, with a published contract test-suite and a scaffold command. |

### 12.2 Non-goals for Part II

- Continuous-time switching inside intervals (unchanged from Part I).
- Neural/learned emission densities (e.g. normalizing flows per state) — possible plug-in, not core.
- Bayesian nonparametric state counts (HDP-HSMM): design spike only (E13-T09); no core implementation.
- Full factorial/coupled inference beyond E9-T02.
- A GUI, dashboards or a web service.
- Reinforcement learning / decision policies beyond the finance reference strategies.

### 12.3 Success criteria per release

| Release | Criteria |
|---|---|
| 0.2.0 (E10–E12) | Every Part I Appendix A config validates unchanged under `profile: finance`; a core-profile configuration (Appendix H) runs `fit → decode → forecast` on a synthetic batch of 50 labelled sequences; family contract suite passes for all families in Appendix F marked "0.2.0"; batch benchmark NFR-12 met. |
| 0.3.0 (E13–E14) | EM and MAP (no priors) agree within Appendix G tolerances on all conjugate families; VB reproduces MAP parameter means on large synthetic data and prunes an over-specified state; NumPyro MCMC on the two-regime Appendix E fixture yields posterior means within 2 posterior SD of truth; supervised recovery scenario passes; segmentation metrics validated against `ruptures`-style reference values. |
| 0.4.0 (E15–E16) | Online filter equals offline filter bitwise; sklearn wrapper passes the applicable subset of `check_estimator`; plotting of all `*Result` objects; a third-party family installed from a separate package is discovered and passes the contract suite. |
| 0.5.0 (E17) | Docs restructured into core/finance/extending guides; PyPI + conda-forge; ADR-0012…0019 accepted; deprecation policy (NFR-14) in force. |

---

## 13. Requirements (Part II)

### 13.1 Functional requirements

#### Generalization (FR-G)

| ID | Requirement |
|---|---|
| FR-G01 | The core MUST NOT reference prices, returns, symbols, trading calendars or data providers in names, defaults or documentation. The observation field of `PreparedDataset` is `observations`; the finance profile MAY expose `returns` as a read-only alias. |
| FR-G02 | A clock `steps` with `Δ_t ≡ 1` MUST be available; under it, no parameter is annualized and durations are reported in steps. |
| FR-G03 | Observation supports `real` (`ℝ^M`), `count` (`ℕ^M`), `binary` (`{0,1}^M`) and `categorical` (`{0..K−1}`) MUST be supported. A family declares its `support`; the preparer validates data against it and raises `DataError` on mismatch. |
| FR-G04 | Generic data sources MUST be available: in-memory `numpy` arrays, `pandas.DataFrame`, `xarray.Dataset`, CSV and parquet files. Under the `steps` clock a time index is optional. |
| FR-G05 | The configuration MUST carry `profile: core \| finance`. A profile selects defaults, available families and evaluation modules; it MUST NOT change the semantics of any shared key. |
| FR-G06 | A preset registry (`regimekit presets list/show`) MUST ship named, validated configurations per profile with matching synthetic fixtures. |
| FR-G07 | Backward compatibility: every `schema_version: 1` configuration MUST migrate automatically to `schema_version: 2` (`profile: finance`, `returns → observations`) and validate unchanged. |

#### Emission families (FR-EM)

| ID | Requirement |
|---|---|
| FR-EM01 | Multivariate Gaussian with `diag` or `full` covariance (Cholesky parameterization), optional `Δ`-scaling of mean and covariance, optional Itô correction, covariate links on mean and log-scale. |
| FR-EM02 | Multivariate Student-t (`ν` free/tied/fixed) with the same options. |
| FR-EM03 | Poisson (exposure `Δ_t`, log link), Bernoulli (logit link) and Categorical (softmax link), each with covariate links on the natural parameter. |
| FR-EM04 | Markov-switching autoregression MS-AR(p) with per-state intercept, AR coefficient matrices and covariance, realized as a Gaussian family whose emission covariates are the preparer-generated own lags; the first `p` steps are conditioned on (masked). Hamilton's mean-adjusted variant is supported for `p ≤ 2` through the memory builder (D5) and documented as unsupported beyond. |
| FR-EM05 | Per-state finite mixture of any base family with static component count `K_i`. |
| FR-EM06 | The finance profile MUST provide GBM (= Gaussian with `Δ`-scaling and Itô correction), OU with exact discretization and Merton jump-diffusion (truncated Poisson mixture). MS-GARCH remains E9-T03. |
| FR-EM07 | Families MAY implement `sufficient_stats`/`m_step` (closed-form EM) and `conjugate_prior` (VB/Gibbs); absence degrades gracefully to numerical M-steps or point estimates. |
| FR-EM08 | Custom families MUST be registrable via entry points and MUST pass `regimekit.testing.family_contract`. |

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
| FR-IO04 | `hmmlearn` `GaussianHMM`/`CategoricalHMM` parameters MUST be importable into a geometric-duration `regimekit` model (one-way). |
| FR-IO05 | The configuration schema MUST be exportable as JSON Schema for editor support. |

#### Visualization (FR-VZ)

| ID | Requirement |
|---|---|
| FR-VZ01 | Plot helpers (matplotlib, `plot` extra): observations with regime shading and filtered/smoothed probabilities; duration pmfs and hazards; forecast fan charts; diagnostics panel (PIT, pseudo-residuals, restart dispersion, ELBO trace). |
| FR-VZ02 | Every helper accepts an `ax` and returns it; no global state; no figure is shown implicitly. |

#### Plug-ins (FR-PL)

| ID | Requirement |
|---|---|
| FR-PL01 | Entry-point groups `regimekit.emission`, `regimekit.duration`, `regimekit.metric`, `regimekit.source` MUST be discovered at first registry access (lazy). |
| FR-PL02 | `regimekit.testing` MUST publish the contract test-suites (family, duration, metric, source) as importable pytest fixtures/parametrizations. |
| FR-PL03 | `regimekit new-family NAME` MUST scaffold a plug-in package with tests wired to the contract suite. |

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
| NFR-11 | `import regimekit` MUST succeed with only core dependencies; profile and extra dependencies (`yfinance`, `numpyro`, `arviz`, `matplotlib`, `scikit-learn`, `xarray`) are imported lazily and raise an actionable `ImportError` naming the extra. |
| NFR-12 | Batch performance guardrail (8-core x86 CPU): `B = 64`, `T = 1000`, `N = 4`, `S = 12`, order 1 — jitted batched forward-backward ≤ 300 ms; one EM iteration ≤ 400 ms. |
| NFR-13 | Estimator parity: EM vs MAP without priors agree within Appendix G; VB posterior means vs MAP on `T ≥ 20 000` within Appendix G. |
| NFR-14 | Plug-in API stability: `regimekit.testing`, the family/duration/metric/source protocols and `PreparedDataset` follow semantic versioning from 0.3.0; breaking changes require a two-minor-version deprecation cycle. |
| NFR-15 | Online filter equals offline filter bitwise on the same backend; fixed-lag smoothed values equal offline smoothed values for `t ≤ T − L`. |
| NFR-16 | Documentation: separate guides for core, finance profile and "extending regimekit"; one tutorial per profile executed in CI. |
| NFR-17 | Reproducibility (NFR-03) extends to EM, VB and MCMC (seeded chains); MCMC determinism is per backend and per NumPyro version and is documented as such. |

---

## 14. Mathematical specification (additions)

### 14.1 Clocks

Under `steps`, `Δ_t = 1` for all `t`. Continuous-time families receive `Δ = 1` and their parameters are interpreted per step; duration families report `mean_sojourn` in clock units (years or steps). Nothing else changes: `a_i = 1 − exp(−λ_i)` is still the per-step exit probability, so geometric/NB semantics carry over exactly.

### 14.2 Emission families

Notation as in Section 4.2; `m_i(t)` and `Σ_i(t)` denote covariate-linked location and scale. `s(Δ) = Δ` when `scale_by_dt: true` (continuous clocks) and `1` otherwise.

**Multivariate Gaussian.** `y_t | i ~ N(m_i(t) s(Δ_t), Σ_i(t) s(Δ_t))`, `Σ_i = L_i L_iᵀ` with `L_i` lower-triangular, `diag(L_i) = softplus(·)`; `diag` mode sets off-diagonals to zero. Itô correction (`ito_correction: true`): `m_i ← μ_i − ½ diag(Σ_i)`. GBM of Part I is exactly this family with `M = 1`, `scale_by_dt: true`, `ito_correction: true`.

**Multivariate Student-t.** `y_t | i ~ t_{ν_i}(m_i(t) s(Δ_t), Σ_i(t) s(Δ_t))`, `ν_i > 2`.

**Poisson / Bernoulli / Categorical.** `y_t | i ~ Poisson(exp(η_i + γ_iᵀ z^e_t) · s(Δ_t))` (exposure `Δ`); `y_t | i ~ Bernoulli(σ(η_i + γ_iᵀ z^e_t))`; `y_t | i ~ Cat(softmax(η_i + Γ_i z^e_t))` with the first category as reference. Sufficient statistics and closed-form M-steps exist for all three (Appendix F).

**MS-AR(p).** Regime-dependent-coefficient form: `y_t = c_i + Σ_{l=1}^{p} Φ_{i,l} y_{t−l} + ε_t`, `ε_t ~ N(0, Σ_i)`. The preparer emits `y_{t−1}, …, y_{t−p}` as emission covariates (this is the existing FR-M09 drift link with per-state coefficients and `lag ≥ 1` enforced), and masks `t ≤ p`. Hamilton's mean-adjusted form `y_t − μ_{s_t} = Σ φ_l (y_{t−l} − μ_{s_{t−l}})` depends on the regimes at `t−1..t−p`; it is realized through the order-`(p+1)` memory builder of Section 4.5 for `p ≤ 2` and rejected with `ConfigError` otherwise.

**Mixture.** `f_i(y) = Σ_{k=1}^{K_i} w_{ik} g(y | θ_{ik})`, `w_i` on the simplex; `log f_i = logsumexp_k(log w_{ik} + log g)`. Ordering constraints on `θ_{ik}` within a state are supported for identifiability.

**OU (finance).** For an observed level `x_t` with regime `i`: `x_t | x_{t−1}, i ~ N(m_i + (x_{t−1} − m_i) e^{−κ_i Δ_t}, σ_i² (1 − e^{−2κ_i Δ_t}) / (2κ_i))`. Implemented as its own family (nonlinear in `Δ`), with `x_{t−1}` supplied as a lag covariate.

**Merton jump-diffusion (finance).** `y_t | i, k ~ N((μ_i − ½σ_i² − λ_i k̄_i) Δ_t + k m_{J,i}, σ_i² Δ_t + k s_{J,i}²)`, `k ~ Poisson(λ_i Δ_t)` truncated at `K_max` (default 5): a mixture family instance with Poisson weights and `k̄ = exp(m_J + ½ s_J²) − 1`.

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

### 15.1 Package layout delta

```
src/regimekit/
├── data/                     # generic: contracts, calendar (three clocks), prepare, quality, batch.py (SequenceBatch)
│   └── sources/              # base.py (Source Protocol + registry), numpy.py, pandas.py, xarray.py, csv.py, parquet.py
├── model/
│   ├── emission/             # base.py, gaussian.py, student_t.py, poisson.py, bernoulli.py, categorical.py, msar.py, mixture.py
│   ├── duration/             # + per_state.py (explicit engine composite)
│   ├── topology.py           # Ω masks and shorthands
│   └── ...
├── statespace/               # + duration-dependent jump support in transitions.py (explicit)
├── engine/                   # + batch.py (vmap wrappers, bucketing), constraints.py (label/boundary masks), ffbs.py
├── estimation/
│   ├── map.py                # Part I fit.py renamed (Estimator "map")
│   ├── em.py                 # expanded-engine EM/ECM, tempering
│   ├── gem.py                # explicit-engine generalized EM
│   ├── vb.py                 # mean-field VB, conjugate updates, pruning
│   └── mcmc.py               # NumPyro bridge (bayes extra), relabelling
├── online/                   # filter.py (OnlineFilter), alerts.py, state.py
├── selection/                # splitters.py, grid.py, bootstrap.py, ppc.py, pseudo_residuals.py
├── evaluation/
│   ├── metrics/              # + segmentation.py, clustering.py
│   └── ...
├── interop/                  # sklearn.py, xarray.py, arviz.py, hmmlearn.py
├── plot/                     # regimes.py, durations.py, forecast.py, diagnostics.py (plot extra)
├── plugins.py                # entry-point discovery, registries
├── presets/                  # YAML presets per profile + fixtures manifest
├── testing/                  # contract suites: family_contract, duration_contract, metric_contract, source_contract
└── finance/                  # profile: emission/{gbm,ou,merton}.py, returns.py (Preparer subclass), providers/ (moved from data/),
                              # strategies.py, metrics_finance.py, presets/
```

Part I modules keep their names; `data/providers/` moves to `finance/providers/` (E10-T03), `estimation/fit.py` becomes `estimation/map.py` with a re-export.

### 15.2 Core data contracts (additions)

```python
@dataclass(frozen=True)
class PreparedDataset:                      # Part I 5.4, amended
    observations: Array                     # (T, M)  — was `returns`
    support: Literal["real", "count", "binary", "categorical"]
    n_categories: int | None                # categorical only
    labels: Array | None                    # (T, N) bool allowed-set, None = unlabelled
    boundaries: BoundaryConstraints | None  # must_switch_at / must_not_switch_at step indices
    group: str | None                       # initial-distribution group (FR-SQ02)
    clock: Literal["trading", "calendar", "steps"]
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

class Source(Protocol):                     # generic data source, replaces DataProvider in core
    name: str
    def load(self, cfg: SourceCfg) -> tuple[ObservationSet, list[CovariateSeries]]: ...

class Splitter(Protocol):
    def split(self, batch: SequenceBatch) -> Iterator[tuple[SequenceBatch, SequenceBatch]]: ...

class Plotter(Protocol):
    def __call__(self, result: Any, ax: "matplotlib.axes.Axes | None" = None, **kw) -> "matplotlib.axes.Axes": ...
```

`DataProvider` (Part I) becomes a `Source` specialization in `regimekit.finance.providers`; the contract test-suite of E2-T01 is generalized to `source_contract`.

### 15.4 Configuration schema (schema_version 2)

| Change | Detail |
|---|---|
| `profile` | top-level, `core \| finance`; selects defaults and family registries; `schema_version: 1` files migrate to `profile: finance` |
| `data.source` | `{kind: numpy \| pandas \| xarray \| csv \| parquet \| provider, …}`; `provider` valid only under `profile: finance` |
| `data.clock` | `trading \| calendar \| steps` |
| `data.sequences` | `{group_by: column \| null, bucket_edges: [256, 512, 1024, 2048, 4096]}` |
| `data.labels` | `{source: column \| file, allowed_sets: bool, boundaries: {must_switch_at: [...], must_not_switch_at: [...]}}` |
| `model.emission.family` | names from Appendix F; `params` per family; `scale_by_dt`, `ito_correction`, `covariance: diag \| full`, `mixture: {components: K, base: …}` |
| `model.topology` | `full \| left_to_right \| banded: k \| cyclic \| matrix: [[…]]` |
| `model.left_censored` | bool |
| `model.duration` | `family` or `per_state: [{family, params}, …]` (explicit engine); `duration_dependent_jumps: bool` |
| `estimation.method` | `map \| em \| gem \| vb \| mcmc`; per-method sub-sections (`em.tempering`, `vb.prune_below`, `mcmc.{sampler, chains, warmup, draws}`) |
| `estimation.init` | adds `labels` |
| `online` | `{lag: L, alerts: [{regime, p_on, p_off}, {remaining_duration_below: …}]}` |
| `evaluation.metrics` | adds segmentation/clustering names; `evaluation.cv: {splitter: sequence_kfold \| walk_forward, …}`; `evaluation.grid: {...}` |
| `evaluation.strategy` | valid only under `profile: finance` |

Validation additions: family `support` matches data support; `topology` rows non-empty unless `absorbing` intended (warning); `duration_dependent_jumps` requires `engine: explicit`; `method: vb|mcmc` warns on non-conjugate components; `labels` shape `(T, N)`; `profile: core` forbids `provider` sources and strategies.

### 15.5 Extras

| Extra | Dependencies | Enables |
|---|---|---|
| `finance` | `pandas-datareader`, `fredapi` | profile modules, providers (`yfinance`, `curl_cffi` in `providers`) |
| `providers` | `yfinance`, `curl_cffi` | network providers (unchanged from Part I) |
| `plot` | `matplotlib>=3.9` | `regimekit.plot` |
| `sklearn` | `scikit-learn>=1.6` | `regimekit.interop.sklearn` |
| `bayes` | `numpyro>=0.20`, `arviz>=0.21` | MCMC, `InferenceData` export |
| `xarray` | `xarray>=2025.1` | `to_xarray()` (pandas conversion needs no extra) |
| `gpu` | unchanged | |

### 15.6 Presets

`regimekit.presets` ships validated YAML plus a synthetic fixture generator per preset: `finance/dax_3state_gbm_nb` (Appendix A), `finance/two_regime_sanity` (Appendix E), `core/segmentation_gaussian_ltr` (Appendix H), `core/counts_poisson_hazard`, `core/msar2_hamilton`. Presets are the fixtures of the tutorials and of the recovery suite extensions.

### 15.7 Amendments to Part I (apply before the affected v1 tickets)

> **Status 2026-09-14: applied.** All nine amendments were folded into the existing Part I tickets (E0-T01, E0-T04, E1-T01, E1-T02a/b, E1-T05, E2-T01…T04, E2-T09, E2-T06, E2-T07, E3-T01, E3-T03, E3-T04, E5-T03, E5-T04, E6-T04a, E8-T01; see ticket overview §8). Part I is now v1.1. The table is kept for traceability.

| Part I section | Amendment | Cost now | Cost after 0.1.0 | Affected tickets |
|---|---|---|---|---|
| 5.4 `PreparedDataset.returns` | rename to `observations`; add `support`, `clock`, `labels=None`, `boundaries=None`, `group=None` with defaults | S | L (rename across all layers, bundle migration) | E1-T01 |
| 4.1 / 5.6 clocks | add `steps` clock (`Δ_t ≡ 1`) to `calendar.py` | S | S | E2-T06 |
| 5.3 layout | create `regimekit/finance/` from the start; place `data/providers/` there; keep `data/` generic | S | M (import-path churn, bundle manifests) | E2-T01…T09 |
| 5.5 `EmissionFamily` | add `support`, `scale_by_dt` attributes; keep `sufficient_stats` optional | S | S | E3-T01 |
| 5.5 `Estimator` | rename `estimation/fit.py` → `estimation/map.py`, add `name = "map"`; `FitResult.estimator` | S | S | E5-T03 |
| 5.7 config | add `profile` (default `finance` for `schema_version: 1`), `data.clock: steps`; JSON-Schema export hook | S | M (migration path for user configs) | E1-T02 |
| 5.8 bundle | `manifest.json` records `profile` and `schema_version` | S | S | E8-T01 |
| 4.3 `mean_sojourn_years` | rename to `mean_sojourn` (clock units) | S | S | E3-T03 |
| 6.1 stack | no new runtime dependency for the amendments | — | — | — |

Applying the amendments bumps Part I to version 1.1; the amended tickets add "(Part II amendment 15.7)" to their acceptance criteria. None of the amendments changes behaviour of the finance profile.

---

## 16. Engineering guideline deltas

### 16.1 Stack additions (each behind an extra, each with an ADR)

| Area | Choice | Notes |
|---|---|---|
| Plotting | `matplotlib ≥ 3.9` | `plot` extra; no seaborn/plotly in core |
| sklearn interop | `scikit-learn ≥ 1.6` | `sklearn` extra; only `BaseEstimator`, `check_estimator` subset in tests |
| Bayesian | `numpyro ≥ 0.20`, `arviz ≥ 0.21` | `bayes` extra; NumPyro model built from `log_joint(u)` |
| Labelled arrays | `xarray ≥ 2025.1` | `xarray` extra |
| Assignment (Hungarian) | `scipy.optimize.linear_sum_assignment` | scipy is already a dev dependency; promote to runtime dependency for `evaluation.metrics.clustering` (ADR-0017) — or vendor a 60-line implementation; decision in E14-T03 |
| Scaffold | `copier` | dev group only; template under `templates/family-plugin/` |

### 16.2 Coding standards (additions)

- Lazy imports for every extra (`regimekit.utils.optional_import("matplotlib", extra="plot")`); importing the core MUST NOT import them (NFR-11, tested with `python -X importtime` allow-list).
- Every family, duration, metric and source implementation — built-in or plug-in — is parametrized into the contract suites of `regimekit.testing`; a new implementation without a contract-suite entry fails CI.
- Estimators MUST NOT special-case families; capability discovery goes through the optional protocol methods (`m_step is not None`, `conjugate_prior() is not None`).
- Batched code is written for `B` sequences and called with `B = 1` for single sequences; no single-sequence fast paths.
- Deprecations use `warnings.warn(..., DeprecationWarning)` with the removal version and are listed in `docs/guide/deprecations.md` (NFR-14).

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
| `plugins` | a fixture plug-in package installed in a temp venv is discovered and passes the contract suite | recommended |
| recovery (extended) | scenarios: Gaussian full-cov 3-state batch of 50; Poisson 2-state hazard; MS-AR(2) Hamilton; left-to-right topology with labels; VB on 4 configured/3 true states | n/a (nightly) |

Golden references (`tests/reference/`) gain: batched forward-backward, EM M-steps for Gaussian/Poisson/Categorical, Guédon E-step segment posteriors, FFBS sampling on a 2-state/4-step enumerable example, Hungarian matching.

### 16.5 CI/CD (additions)

Jobs `extras` (installs each extra in isolation and runs its tests; verifies NFR-11), `plugin` (builds and installs the template plug-in), `bayes` (nightly, short chains). Coverage gate unchanged (≥ 90 % overall, ≥ 85 % per package, now including `online`, `selection`, `interop`, `finance`).

### 16.6 Documentation (additions)

`docs/guide/core/` (quickstart with a synthetic labelled batch, families, topology, supervision, online), `docs/guide/finance/` (Part I guide moved), `docs/guide/extending/` (writing a family, running the contract suite, publishing a plug-in), `docs/math/` pages for Sections 14.x, `docs/guide/estimators.md` (decision table MAP/EM/VB/MCMC with cost and assumptions), `docs/guide/deprecations.md`. Tutorials: finance (Part I), segmentation, counts — all executed in CI from presets.

### 16.7 Architecture Decision Records (additions; ADR-0011 is the safetensors decision of Part I/E8-T01, hence numbering starts at 0012; 0012 and 0013 are already accepted via E0-T04, the rest are drafted in their epics)

| ADR | Decision (proposed default) |
|---|---|
| 0012 | Core/profile split; `regimekit.finance` as first profile; single distribution with extras (D12, D22) |
| 0013 | `steps` clock and clock-neutral family contract (D13) |
| 0014 | Plug-in discovery via entry points; `regimekit.testing` as public contract (D15, NFR-14) |
| 0015 | EM/GEM/VB/MCMC as peer estimators behind `Estimator`; MAP remains reference (D17) |
| 0016 | Supervision as engine-level masks, no supervised code path (D18) |
| 0017 | scipy as runtime dependency (or vendored assignment) for clustering metrics |
| 0018 | Multi-sequence support via `vmap` and length buckets; contracts stay single-sequence (D14) |
| 0019 | Approximate inference for factorial models (carried over from Part I; finalized in E9-T02) |
| 0020 | ArviZ as the Bayesian output format; NumPyro as the only sampler backend in 0.x |
| 0021 | Deprecation policy and semver scope from 0.3.0 (NFR-14) |

### 16.8 Definition of Done (additions)

7. New implementations are registered in the appropriate contract suite parametrization.
8. Extras remain lazily imported (import-time test passes).
9. Public protocol changes list the deprecation path (from 0.3.0).

---

## 17. Implementation plan (Part II)

### 17.1 Milestones and release mapping

| Milestone | Content | Release | Exit criteria |
|---|---|---|---|
| M7 Generalization baseline | E10 | 0.2.0 (with M8, M9) | Amendments 15.7 applied; core profile config validates and runs on numpy input under `steps`; finance profile unchanged (Part I tests green) |
| M8 Family catalogue | E11 | 0.2.0 | All Appendix F families marked 0.2.0 pass the contract suite; MS-AR(2) recovery passes |
| M9 Sequences & structure | E12 | 0.2.0 | Batch of 50 sequences fits under NFR-12; topology/left-censoring/labels golden tests pass |
| M10 Estimators | E13 | 0.3.0 | EM/GEM/VB/MCMC parity per Appendix G; estimator decision table documented |
| M11 Selection & supervision | E14 | 0.3.0 | Grid selection and CV tables; segmentation metrics; supervised recovery scenario passes |
| M12 Online & interop | E15, E16 | 0.4.0 | Online filter bitwise equality; sklearn wrapper; plots; template plug-in discovered |
| M13 General release | E17 | 0.5.0 | Docs restructured; PyPI + conda-forge; ADRs 0012–0021 accepted |

M7 SHOULD run before or in parallel with Part I M1/M2 (its content is the amendment list); M8–M13 follow 0.1.0.

### 17.2 Epics and tickets

Sizes and rules as in Section 7.2. Every ticket includes tests, docs and — where a decision is taken — an ADR.

#### E10 — Core/profile split and generalization baseline

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E10-T01 | Residual of amendments 15.7 (the rename/layout items were absorbed into Part I v1.1 tickets on 2026-09-14): register the `1 → 2` migration step (`profile` explicit, `schema_version: 2`), lift the 0.1.0 validation rule that rejects `profile: core`, draft ADR-0018 | S | E1-T02b (v1.1) | FR-G01, FR-G07 | `schema_version: 1` config migrates to 2 with `profile: finance` and validates unchanged; `profile: core` validates; golden outputs of the finance profile identical |
| E10-T02 | `steps` clock in `calendar.py`; clock-neutral duration reporting; families tested under all three clocks | S | E2-T06, E3-T03 | FR-G02, D13 | `Δ_t ≡ 1` path; geometric/NB sojourn means reported in steps; Appendix E numbers unchanged under `trading` |
| E10-T03 | Generic `Source` protocol + registry + `numpy`, `pandas`, `xarray`, `csv`, `parquet` sources; generalized `source_contract`; `DataProvider` re-based on `Source` in `finance` | M | E2-T01, E10-T01 | FR-G04, FR-PL02 | contract suite parametrized over all sources; `DataFrame` with/without time index loads under `steps`/`calendar`; provider tests of E2 unchanged |
| E10-T04 | Observation supports and validation (`real/count/binary/categorical`), `n_categories`, `DataError` on mismatch; preparer bypass of log-return transform under `profile: core` | S | E10-T03 | FR-G03 | each support has an accepting and a rejecting fixture; finance profile still computes log-returns |
| E10-T05 | `profile` in config; profile-specific defaults, registries and validation rules (15.4); JSON-Schema export; `presets` registry with `finance/*` and `core/*` entries and fixture generators | M | E10-T01 | FR-G05, FR-G06, FR-IO05 | `profile: core` rejects providers/strategies with path-qualified error; `regimekit presets show core/segmentation_gaussian_ltr` prints Appendix H; JSON Schema validates Appendix A and H |
| E10-T06 | Lazy optional imports, extras in `pyproject.toml`, import-time test (NFR-11), `extras` CI job | S | E10-T01 | NFR-11 | `python -c "import regimekit"` imports none of the extras' packages; each extra's tests run in isolation |

#### E11 — Emission family catalogue

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E11-T01 | `family_contract` suite in `regimekit.testing`: log-prob vs scipy, sampler moments, `Δ`-scaling, support validation, covariate links, param-spec round-trip, optional-method consistency (`m_step` optimum equals L-BFGS optimum on the expected complete-data objective) | M | E3-T01, E10-T04 | FR-EM08, FR-PL02 | GBM and Student-t of Part I pass unchanged; suite importable by external packages |
| E11-T02 | Multivariate Gaussian (`diag`/`full`, Cholesky bijector, `scale_by_dt`, `ito_correction`); GBM re-expressed as a configured Gaussian in `finance` | M | E11-T01, E1-T03 | FR-EM01, FR-EM06 | log-pdf vs `scipy.stats.multivariate_normal` ≤ 1e-10; Cholesky Jacobian vs autodiff ≤ 1e-8; GBM golden outputs bitwise unchanged |
| E11-T03 | Multivariate Student-t | S | E11-T02 | FR-EM02 | vs `scipy.stats.multivariate_t`; `ν → ∞` limit |
| E11-T04 | Poisson, Bernoulli, Categorical with covariate links and closed-form sufficient statistics / M-steps / conjugate priors | M | E11-T01 | FR-EM03, FR-EM07 | contract suite; M-step closed form equals numerical optimum ≤ 1e-8; exposure `Δ_t` tested under `calendar` |
| E11-T05 | MS-AR(p): preparer lag generation as emission covariates, per-state coefficient matrices, first-`p` masking; Hamilton mean-adjusted variant for `p ≤ 2` via memory builder | M | E11-T02, E4-T02 | FR-EM04 | recovery of a 2-state MS-AR(2) (Hamilton 1989 magnitudes); `p = 3` mean-adjusted → `ConfigError`; likelihood of 1-state AR(p) matches `statsmodels` ≤ 1e-6 |
| E11-T06 | Per-state finite mixture of a base family with in-state ordering constraints | M | E11-T02 | FR-EM05 | contract suite; 1-component mixture equals base family bitwise; sufficient statistics via responsibilities |
| E11-T07 | Finance: OU (exact discretization) and Merton jump-diffusion (truncated Poisson mixture) | M | E11-T06 | FR-EM06 | OU transition density vs closed form ≤ 1e-12; Merton with `λ = 0` equals GBM; `K_max` truncation mass reported |
| E11-T08 | Correlated multi-series emissions (`M > 1`, full covariance) as the general case of E9-T01; finance data layer for multiple primaries | M | E11-T02, E9-T01 | FR-M12 | two-series synthetic recovery of a 2×2 covariance per state; `M = 1` golden outputs unchanged |

#### E12 — Sequences, topology, censoring, constraints

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E12-T01 | `SequenceBatch`, padding to buckets, `A_t = I`/`ℓ_t = 0` padding semantics, `vmap`ped engine wrappers for both engines, per-sequence unpadding of posteriors | L | E4-T03, E4-T06, E10-T01 | FR-SQ01, FR-SQ03, FR-SQ04 | batched loglik equals sum of single-sequence logliks ≤ 1e-10; padding invariance property test; bucket recompilation count ≤ number of buckets |
| E12-T02 | Batch estimation (MAP) with per-group initial distributions; batched simulate | M | E12-T01, E5-T03 | FR-SQ01, FR-SQ02 | recovery on 50 sequences × `T = 200` matches single-sequence recovery on `T = 10 000` within Appendix G; NFR-12 benchmark recorded |
| E12-T03 | Transition topology `Ω`: shorthands, masked logits with per-row reference category, absorbing states in both engines | M | E3-T05, E4-T02 | FR-DU01, FR-DU02 | left-to-right model never decodes a backward jump; absorbing state has `a_i = 0`; parameter count matches Appendix F formula; `full` topology bitwise equals Part I |
| E12-T04 | Left-censoring: residual-life initial duration in the explicit engine; documentation that `initial: stationary` covers the expanded engine | S | E4-T06 | FR-DU04 | explicit-engine log-likelihood with residual-life initial equals expanded-engine `stationary` on shifted-NB within Appendix D tolerance |
| E12-T05 | Per-state duration families and duration-dependent jumps in the explicit engine | M | E3-T04, E12-T03 | FR-DU03, FR-DU05 | mixed geometric/gamma model recovers both families; `c_ij` sign recovered on synthetic data; option rejected with `ConfigError` under `engine: expanded` |
| E12-T06 | Label and boundary constraints as engine masks (`engine/constraints.py`), `ConstraintError`, label-informed initialization | M | E12-T01 | FR-SV01, FR-SV02, FR-SV03 | fully labelled data reproduces labels under Viterbi; infeasible boundary raises with step index; `init: labels` recovers parameters without optimization within 15 % on synthetic data |

#### E13 — Estimators

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E13-T01 | E-step accumulators inside the scan (regime `γ`, pooled phase-advance and jump counts, family sufficient statistics) for the expanded engine; explicit-engine segment posteriors `η_t(j,d)` | L | E12-T01 | FR-ES01, FR-ES02 | accumulators vs NumPy reference ≤ 1e-8; no `(T, S, S)` allocation (memory test) |
| E13-T02 | `EMEstimator` (expanded engine): closed-form M-steps where available, ECM otherwise, tempering option, restarts, convergence | L | E13-T01, E11-T04 | FR-ES01, FR-ES07, NFR-13 | monotone `ℓ` (≤ 1e-10 violations); parity with MAP (no priors) per Appendix G on Gaussian/Poisson/Categorical; tempered EM reaches the MAP optimum on a multimodal fixture where plain EM does not (≥ 4/5 seeds) |
| E13-T03 | `GEMEstimator` (explicit engine, Guédon): jump, nonparametric and parametric duration M-steps, right-censoring handling | L | E13-T01, E12-T04 | FR-ES02 | monotone `ℓ`; nonparametric pmf recovery on synthetic data (TV distance ≤ 0.05 at `T = 20 000`); parity with MAP for gamma durations |
| E13-T04 | FFBS for both engines; path-sample export in `decode --samples R` | M | E13-T01 | FR-ES03 | path-sample regime frequencies vs smoothed marginals within 3 MC SE; enumerable 2-state/4-step golden test |
| E13-T05 | `VBEstimator`: conjugate updates (Normal-Wishart, Gamma, Beta, Dirichlet), structured `q(s)`, ELBO, pruning, VB-EM hybrid for non-conjugate components | L | E13-T01, E11-T04 | FR-ES04, NFR-13 | ELBO monotone; posterior means vs MAP on `T = 20 000` per Appendix G; a 4-state VB fit on 3-state data prunes one state; hybrid components listed in the fit report |
| E13-T06 | `MCMCEstimator` (`bayes` extra): NumPyro NUTS on `log_joint(u)`; blocked Gibbs (FFBS + conjugate draws); relabelling; ArviZ `InferenceData`; `FitResult.posterior_samples` | L | E13-T04, E13-T05 | FR-ES05, FR-IO03, NFR-17 | Appendix E fixture: posterior means within 2 posterior SD of truth, `R̂ ≤ 1.05`; Gibbs and NUTS agree on means within 2 MC SE; seeded chains reproducible |
| E13-T07 | Estimator parity suite and decision documentation (`docs/guide/estimators.md`); `compare --estimators` | M | E13-T02, E13-T03, E13-T05, E13-T06 | FR-ES06, FR-EV04 | parity table generated in CI; documentation lists assumptions, cost and failure modes per estimator |
| E13-T08 | Online/recursive EM spike (Cappé & Moulines) — design note and prototype on the expanded engine | S | E13-T02, E15-T01 | FR-ON05 | ADR with algorithm choice; prototype tracks a drifting `σ` on synthetic data; no merge to `main` beyond ADR |
| E13-T09 | HDP-HSMM design spike (Johnson & Willsky): what a weak-limit implementation would need from the engine | S | E13-T06 | 12.2 | ADR with go/no-go and interface diff; no code merged |

#### E14 — Selection, diagnostics and evaluation

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E14-T01 | `Splitter` interface: `SequenceKFold` (grouped), walk-forward re-based on it; CV log score / CRPS aggregation | M | E12-T02, E7-T02 | FR-EV03 | walk-forward results bitwise unchanged after re-basing; grouped folds never split a group; look-ahead test retained |
| E14-T02 | Grid model selection (`compare --grid`) over `N`, `r`, families, estimators; IC and CV columns; parallel cells (sequential default) | M | E14-T01, E13-T07 | FR-EV04 | grid on the Appendix E fixture selects `N = 2` by BIC and CV; table schema documented |
| E14-T03 | Segmentation and clustering metrics (boundary F1@τ, Hausdorff, covering, Hungarian-matched Hamming, ARI, NMI); ADR-0017 (scipy vs vendored assignment) | M | E7-T01 | FR-EV01, FR-EV02 | closed-form cases (perfect, shifted-by-τ, random); ARI/NMI vs `sklearn.metrics` ≤ 1e-12 on fixtures (dev-only comparison) |
| E14-T04 | Parametric bootstrap SE and nested-model bootstrap tests; Hessian SE comparison | M | E12-T02 | FR-EV05 | bootstrap SE within 20 % of simulation SE on a large synthetic set; documented caveat on `N` comparisons |
| E14-T05 | Pseudo-residuals (continuous and mid-distribution) and posterior predictive checks with configurable statistics; diagnostics integrated into the fit report | M | E6-T04, E13-T04 | FR-EV06 | pseudo-residuals of a correctly specified model pass Shapiro–Wilk at 5 % on ≥ 90 % of seeds; PPC flags a misspecified model (Gaussian on Student-t data) |
| E14-T06 | Extended recovery scenarios (16.4) and `selftest --profile core` | M | E13-T02, E12-T06 | NFR-04 | five new scenarios pass ≥ 4/5 seeds nightly; core selftest < 2 min |

#### E15 — Online inference

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E15-T01 | `OnlineFilter`: jitted single-step update, `OnlineState`, save/load, `spec_hash` check | M | E4-T03, E8-T01 | FR-ON01, FR-ON04, NFR-15 | bitwise equality with offline filtered posteriors over a 5000-step replay; restored state continues bitwise identically |
| E15-T02 | Fixed-lag smoothing with ring buffer | M | E15-T01 | FR-ON02, NFR-15 | equals offline smoothed `γ_{t−L}` ≤ 1e-12; cost linear in `L` (benchmark) |
| E15-T03 | Alert rules with hysteresis and expected-remaining-duration triggers; structured events; online forecast | S | E15-T02, E6-T04 | FR-ON03 | synthetic regime switch produces exactly one `on` and one `off` event; forecast from online state equals offline forecast for the same seed |
| E15-T04 | `regimekit stream` CLI (stdin/CSV tail → JSONL events) and `Project.online()` | S | E15-T03 | FR-C02, FR-C03 | integration test on a piped fixture; exit codes |
| E15-T05 | (v1.x) Online EM implementation following E13-T08 ADR | M | E13-T08 | FR-ON05 | tracks a synthetic parameter drift with lag < 500 steps; offline EM equivalence when the step size schedule is frozen |

#### E16 — Interoperability, visualization, plug-ins

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E16-T01 | scikit-learn `HSMM` estimator (`sklearn` extra): `fit(X, lengths, y=None)`, `predict`, `predict_proba`, `score`, `decode`, `sample`, `get_params/set_params` mapped to config (Appendix I) | M | E12-T02, E12-T06 | FR-IO01 | applicable `check_estimator` checks pass; `lengths` semantics match `hmmlearn`; labels via `y` reach the constraint layer |
| E16-T02 | `to_pandas()`/`to_xarray()` for all `*Result` objects; `hmmlearn` parameter import | M | E6-T04, E11-T02, E11-T04 | FR-IO02, FR-IO04 | round-trip `to_xarray → from_xarray` bitwise; imported `GaussianHMM` reproduces `hmmlearn` log-likelihood ≤ 1e-8 |
| E16-T03 | `regimekit.plot` (`plot` extra): regimes, durations/hazards, forecast fan, diagnostics panel (incl. ELBO/MCMC traces) | M | E16-T02 | FR-VZ01, FR-VZ02 | every helper renders headless and returns `Axes`; image-diff smoke tests with tolerance |
| E16-T04 | Plug-in discovery via entry points (lazy), registries for emission/duration/metric/source, `regimekit new-family` scaffold (copier template), `plugin` CI job | M | E11-T01, E10-T03 | FR-PL01, FR-PL02, FR-PL03, NFR-14 | template plug-in (von Mises emission) installs into a temp venv, is discovered, passes `family_contract`; duplicate names rejected with provenance |
| E16-T05 | Presets completion (`core/counts_poisson_hazard`, `core/msar2_hamilton`, `finance/two_regime_sanity`) with fixture generators; tutorials for segmentation and counts | M | E10-T05, E11-T05 | FR-G06, NFR-16 | tutorials run in CI from presets; presets validate against schema 2 |

#### E17 — Documentation, distribution, release 0.5.0

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E17-T01 | Docs restructure: core / finance / extending guides, math pages 14.x, estimator decision table, deprecation page; API reference regenerated | M | E16-T05, E13-T07 | NFR-16, NFR-07 | `mkdocs build --strict`; every FR/NFR of Part II referenced from at least one page |
| E17-T02 | Distribution: PyPI trusted publishing, conda-forge feedstock, wheel/extras smoke installs in CI (Linux, WSL2, macOS CPU) | S | E10-T06 | D22, NFR-06 | `pip install regimekit[finance,plot]` and `conda install regimekit` succeed in fresh environments |
| E17-T03 | ADRs 0012–0021 accepted; deprecation policy in force; benchmark baselines refreshed for batch and EM | S | all E10–E16 | NFR-14, NFR-12 | all ADRs status "accepted"; benchmark JSON committed |
| E17-T04 | Release 0.5.0 with changelog; roadmap page for factorial (E9-T02), MS-GARCH (E9-T03), online EM (E15-T05) | S | E17-T01…T03 | NFR-10 | tag `v0.5.0` published |

### 17.3 Dependency graph (critical path, Part II)

```
Part I E1-T01/E1-T02 ─▶ E10-T01 ─▶ E10-T03 ─▶ E10-T04 ─▶ E11-T01 ─▶ E11-T02 ─▶ E11-T04 ─▶ E13-T01 ─▶ E13-T02 ─▶ E13-T05 ─▶ E13-T06 ─▶ E13-T07 ─▶ E14-T02 ─▶ E17-T01 ─▶ E17-T04
                          └▶ E10-T05 ─▶ E16-T05                    Part I E4-T03/E4-T06 ─▶ E12-T01 ─▶ E12-T02 ─▶ E14-T01 ↗
                                                                                            └▶ E12-T03 ─▶ E12-T05
                                                                                            └▶ E12-T06 ─▶ E16-T01
Part I E4-T03 ─▶ E15-T01 ─▶ E15-T02 ─▶ E15-T03 ─▶ E15-T04          E11-T01 ─▶ E16-T04
```

Parallel tracks after M7: (a) families E11, (b) batch/structure E12, (c) online E15 (depends only on Part I engine and bundle). E13 needs both (a) and (b). E16 and E17 close the release.

### 17.4 Traceability (Part II requirement → tickets)

| Requirement | Tickets |
|---|---|
| FR-G01…G07 | E10-T01…T06 |
| FR-EM01…EM08 | E11-T01…T08 |
| FR-DU01…DU05 | E12-T03, E12-T04, E12-T05 |
| FR-SQ01…SQ04 | E12-T01, E12-T02 |
| FR-ES01…ES07 | E13-T01…T09 |
| FR-SV01…SV03 | E12-T06 |
| FR-ON01…ON05 | E15-T01…T05, E13-T08 |
| FR-IO01…IO05 | E16-T01, E16-T02, E13-T06, E10-T05 |
| FR-VZ01, VZ02 | E16-T03 |
| FR-PL01…PL03 | E16-T04, E11-T01, E10-T03 |
| FR-EV01…EV06 | E14-T01…T06 |
| NFR-11…NFR-17 | E10-T06, E12-T02, E13-T02/T05, E16-T04, E15-T01, E17-T01, E13-T06 |

---

## 18. Risks and mitigations (Part II)

| Risk | Impact | Mitigation |
|---|---|---|
| Amendments 15.7 applied late | Rename churn across layers, bundle migrations, user-config breakage | Apply in E10-T01 before E2/E3 land; all amendments are additive with defaults; migration hook already in FR-C01 |
| Scope creep of a "general" package | Never-ending catalogue, diluted quality | Appendix F is the closed catalogue for 0.2.0–0.5.0; anything else is a plug-in; contract suite is the entry bar |
| EM for HSMM (Guédon) is intricate; explicit engine memory | Wrong M-steps, OOM | NumPy references first (16.4); monotonicity asserted; chunked scan of E4-T06 reused; MAP remains the fallback and the parity oracle |
| VB conjugacy limits | Users expect VB for covariate models | Hybrid documented per component in the fit report; estimator decision table |
| MCMC cost on long series | Hours per fit | Marginalized NUTS on `u` only; Gibbs for conjugate families; `bayes` extra opt-in; nightly job with short chains |
| Variable sequence lengths cause recompilation | Slow batch fits | Bucketing (FR-SQ04); benchmark NFR-12; bucket edges exposed |
| sklearn API impedance (2-D `X` vs sequences, covariates) | Confusing wrapper | `lengths` convention of `hmmlearn`; covariates as extra columns declared by index; wrapper documented as convenience, `Project` as the full API |
| Plug-in API instability | Third-party breakage | NFR-14 deprecation policy; `regimekit.testing` versioned with the package |
| Optional-dependency sprawl | Heavy installs, import-time regressions | NFR-11 import-time test; each extra tested in isolation |
| Label switching in EM/VB/MCMC | Unusable posteriors | Ordering constraints reused; pivot relabelling; restart dispersion diagnostics extended to ELBO/chains |
| Finance defaults leaking into the core | Domain assumptions in "neutral" code | `profile: core` CI job runs the full core suite with the finance package uninstalled |

---

## 19. Open decisions with proposed defaults (Part II)

| # | Question | Proposed default |
|---|---|---|
| 10 | Where does GBM live? | `regimekit.finance.emission.gbm` as a configured Gaussian; core default emission is `gaussian` |
| 11 | Is `B` a dimension of `PreparedDataset`? | No; `SequenceBatch` wraps single-sequence datasets (D14) |
| 12 | Hungarian matching dependency | scipy as runtime dependency (ADR-0017), unless import-time cost > 50 ms — then vendored |
| 13 | VB scope | Conjugate families only; hybrid elsewhere; no VB for explicit parametric durations |
| 14 | MS-AR sequence start | Condition on the first `p` observations (mask); no stationary-start likelihood |
| 15 | Second sampler backend (BlackJAX) | Not in 0.x; `log_joint(u)` is backend-agnostic |
| 16 | Plotting library | matplotlib only; plotly as a plug-in example if demanded |
| 17 | Package rename to a domain-neutral name | Keep `regimekit`; "regime" is domain-neutral enough; revisit at 1.0 |
| 18 | `polars` support in sources | Via `pandas` conversion in 0.x; native source as plug-in |
| 19 | Deprecation start | 0.3.0 (first release with public protocols in `regimekit.testing`) |

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

| Family | Profile | Support | `p_e` | M-step | Conj. | Δ-scaling | Release |
|---|---|---|---|---|---|---|---|
| `gaussian` (diag) | core | real | `2M` | yes | Normal-Gamma | optional | 0.2.0 |
| `gaussian` (full) | core | real | `M + M(M+1)/2` | yes | Normal-Wishart | optional | 0.2.0 |
| `student_t` (mv) | core | real | `M + M(M+1)/2 + 1` | no (ECM) | no | optional | 0.2.0 |
| `poisson` | core | count | `M` | yes | Gamma | exposure | 0.2.0 |
| `bernoulli` | core | binary | `M` | yes | Beta | no | 0.2.0 |
| `categorical` | core | categorical | `K − 1` | yes | Dirichlet | no | 0.2.0 |
| `msar` (p) | core | real | `M + pM² + M(M+1)/2` | yes (weighted LS) | Normal-Wishart (coeffs) | no | 0.2.0 |
| `mixture(base, K)` | core | as base | `K·p_e(base) + K − 1` | if base has | if base has | as base | 0.2.0 |
| `gbm` | finance | real | `2` | yes | Normal-Gamma | yes + Itô | 0.1.0 |
| `student_t` (uv) | finance/core | real | `3` | no | no | yes | 0.1.0 |
| `ou` | finance | real | `3` | no | no | exact | 0.2.0 |
| `merton` | finance | real | `5` | no | no | yes | 0.2.0 |
| `msgarch` | finance | real | `4` | no | no | discrete | spike (E9-T03) |

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
| Fixed-lag smoothing vs offline smoothing (`t ≤ T − L`) | max-abs ≤ 1e-12 |
| Segmentation metrics on closed-form cases | exact |
| ARI/NMI vs scikit-learn (dev-only) | ≤ 1e-12 |
| `hmmlearn` import log-likelihood | relative ≤ 1e-8 |
| Extended recovery scenarios | as Appendix D; covariance entries ≤ 15 % relative; AR coefficients ≤ 2 SE |

## Appendix H — Reference configuration, core profile (validates against Section 15.4)

```yaml
schema_version: 2
profile: core
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
      - {family: geometric, params: {rate:  {mode: free, init_mean_sojourn: 15}}}
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

| sklearn call | regimekit operation | Notes |
|---|---|---|
| `HSMM(n_states=3, emission="gaussian", duration="negbin", r=3, topology="full", estimator="em", **kw)` | builds `ProjectConfig` under `profile: core`, `clock: steps` | `kw` maps 1:1 to config paths; `get_params()` returns the flattened config |
| `fit(X, lengths=None, y=None, sample_weight=None)` | `SequenceBatch` from `X` split by `lengths`; `y` (int labels, `-1` = unknown) → allowed-set constraints; `sample_weight` unsupported (raises) | `X` is `(ΣT_b, M)`; covariates via `covariate_columns=[...]` in the constructor |
| `predict(X, lengths=None)` | Viterbi regimes, concatenated | |
| `predict_proba(X, lengths=None)` | smoothed regime probabilities `(ΣT_b, N)` | `filtered=True` switches to filtered |
| `score(X, lengths=None)` | total log-likelihood | `score_samples` gives per-step predictive log-density |
| `decode(X, lengths=None)` | `(loglik, regimes)` | hmmlearn-compatible signature |
| `sample(n_samples, random_state)` | `simulate` | returns `(X, regimes)` |
| `forecast(X, horizon, lengths=None)` | `ForecastResult` for the last sequence | extension beyond the sklearn API |
| `result_` | the underlying `FitResult` | for everything the wrapper does not expose |
