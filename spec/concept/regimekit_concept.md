# regimekit — Concept and Implementation Plan

**Hidden Semi-Markov framework for market-regime detection and probabilistic return forecasting**

| Field | Value |
|---|---|
| Version | 1.0 (concept baseline) |
| Date | 2026-09-04 |
| Status | Approved for ticket derivation |
| Amended | 2026-10-18 -- parameter-recovery suite sizing and pass criterion (sections 6.4.4, 6.5 CI/CD, 7.1 milestones, 7.2 tickets, Appendix D); see the ticket overview, section 9 |
| Owner | Aino |
| Working package name | `regimekit` (rename is a one-line change; nothing depends on it) |
| Scope | v1 = single series, full HSMM feature set; interfaces multivariate-ready |

---

## 0. How to use this document

**Purpose.** This document is the single source of truth for the project. Every requirement carries an ID (`FR-…`, `NFR-…`), every design element references the requirements it satisfies, and every ticket in Section 7 references requirement IDs and acceptance criteria. Tickets MUST NOT introduce behaviour that is not traceable to this document; if implementation reveals a needed deviation, the deviation is recorded as an ADR (Section 6.7) and this document is updated with a version bump.

**Keywords.** MUST / SHOULD / MAY as in RFC 2119.

**Ticket derivation rules.**
1. One ticket = one deliverable that can be reviewed and merged on its own, sized S (≤ 1 day), M (1–3 days) or L (3–5 days). Anything larger is split.
2. Each ticket lists: goal, scope (in/out), deliverables (files/modules), acceptance criteria (testable), dependencies, referenced requirement IDs.
3. Tests are part of the ticket, never a separate ticket. TDD applies to all pure-function/numerical code (Section 6.5).
4. Documentation updates (docstrings, user guide section, ADR if applicable) are part of the ticket.

### 0.1 Decisions taken in the brainstorm (baseline, not re-opened)

| ID | Decision |
|---|---|
| D1 | Parameters are defined in continuous time (annualized, calendar/trading clock); inference runs in discrete time on the observation grid. Regime switches inside an interval are out of scope. |
| D2 | One generic inference engine over a *structured state index*. Higher-order memory, negative-binomial durations and (later) factorial structure are **state-space builders** that produce a first-order model for that engine. |
| D3 | Core built from scratch in JAX; estimation by autodiff MAP/MLE (Optax L-BFGS); GPU optional; EM optional later. |
| D4 | Every emission/duration/transition parameter has a mode: `free`, `fixed`, `tied`, `bounded`, `prior`. A state lattice can be generated from parameter grids. |
| D5 | Higher order = memory over the previous *k regimes (segments)*, not time steps. Default order 1; order 2 supported; k > 2 allowed with warning. |
| D6 | Covariates may enter transitions, durations and emissions; each entry point is independently configurable; emissions are the first to drop if scope pressure arises. |
| D7 | Multi-step forecasts are Monte Carlo with pluggable covariate forecasters: `freeze`, `scenario`, `bootstrap`, `ar`. |
| D8 | Data layer fully decoupled; yfinance default, Stooq/Tiingo fallback for prices; FRED/CBOE for covariates; local parquet cache with snapshot IDs. |
| D9 | Outputs: regime probabilities *and* h-step return distributions; backtest and reference strategies in scope; metric set configurable. |
| D10 | Single series in v1; all array contracts carry a series dimension `M` so the universe extension does not change interfaces. |
| D11 | Tooling: uv, pytest with ≥ 90 % coverage, ruff, mypy strict, pydantic v2 config from YAML. |

---

## 1. Vision, goals and non-goals

### 1.1 Problem statement

Financial return series alternate between persistent regimes (calm/trending, turbulent/mean-reverting, crisis) that differ mainly in volatility and secondarily in drift. Plain HMMs impose geometric regime durations (the most likely exit time is always "next step"), ignore what drives switches, and forget which regime preceded the current one. `regimekit` provides a configurable Hidden Semi-Markov Model (HSMM) framework that (a) models regime durations explicitly in calendar/trading time, (b) lets external covariates drive transitions, durations and emissions, (c) supports segment-level memory, (d) uses per-state diffusion dynamics (GBM by default) as emissions, and (e) delivers decoded regimes and probabilistic multi-step forecasts with a reproducible data layer and a backtest harness.

### 1.2 Goals

| ID | Goal |
|---|---|
| G1 | Detect the current regime: filtered and smoothed regime probabilities, decoded segments, posterior over elapsed duration. |
| G2 | Forecast the h-step return distribution (samples, quantiles, moments) and regime probabilities. |
| G3 | Quantify regime duration structure (sojourn distributions, hazards) in calendar/trading time. |
| G4 | Explain regime dynamics via covariates with interpretable coefficients. |
| G5 | Make every run reproducible from (config, data snapshot, seed) and evaluable through walk-forward backtests with configurable metrics. |
| G6 | Be extensible: new emission/duration families, multi-series (factorial/coupled), endogenous covariates, without touching the engine. |

### 1.3 Non-goals (v1)

- Continuous-time switching inside observation intervals (Markov-modulated diffusions with Wonham/particle filtering).
- Live trading, order execution, broker connectivity.
- Portfolio optimization beyond the reference strategies in Section 5.
- Full factorial/coupled inference (design readiness and a spike only).
- Markov-switching GARCH (spike only; interface prepared).
- Bayesian posterior sampling (MCMC) as an acceptance criterion (optional extra via NumPyro, not v1).
- Any GUI.

### 1.4 Success criteria for release 0.1.0

1. Parameter-recovery suite (Section 6.5.4) passes on synthetic data for the default 3-state GBM/NB configuration and for order-2 and covariate variants.
2. End-to-end CLI flow `validate-config → fetch → fit → decode → forecast → backtest` runs on a real index (`^GDAXI` and `^GSPC`, daily) and is bitwise reproducible given (config, snapshot, seed) on the same hardware/backend.
3. Coverage ≥ 90 %, CI green (lint, types, tests, docs).
4. The three engines/paths agree numerically where they must (Appendix D tolerances).

---

## 2. Glossary

| Term | Meaning |
|---|---|
| Regime / state | One of N latent market states `i ∈ {0..N−1}`. |
| Segment | A maximal run of consecutive time steps in the same regime; `(q_n, d_n)` = regime and duration of the n-th segment. |
| Sojourn / duration | Length `d_n` of a segment in grid steps; parameterized via a distribution in years. |
| Jump matrix `π` | Transition probabilities between *different* regimes at segment boundaries (`π_ii = 0`). |
| Phase | Sub-state used by the negative-binomial expansion; regime `i` has `r_i` phases. |
| Expanded state `s` | Element of the structured state index `S` = (memory, regime, phase) the engine iterates over. |
| Memory | The previous `k−1` regimes for an order-`k` model. |
| Hazard | Probability of leaving a regime at step `t` given it has not been left before. |
| Covariate `z_t` | Exogenous explanatory variable known at the *end of t−1*. |
| Link | Function mapping covariates and base parameters to a time-varying parameter (softmax, exp, linear). |
| `Δ_t` | Length of interval `(τ_{t−1}, τ_t]` in years under the configured clock. |
| Clock | `trading` (Δ = 1/252 per trading day, 1/(252·H) per intraday bar) or `calendar` (elapsed time / 365.25). |
| Emission family | Distribution of the observation `y_t` given regime, Δ_t and emission covariates (e.g. GBM ⇒ Gaussian log-returns). |
| Engine | Implementation of forward-backward/Viterbi over an expanded state space. |
| Filtered / smoothed | `P(state_t | y_{1:t})` / `P(state_t | y_{1:T})`. |
| Augmented posterior | Posterior over the expanded state (regime + phase/elapsed duration + memory). |
| Lattice | Set of states generated from the Cartesian product of parameter grids. |
| Snapshot | Immutable, hashed copy of raw provider data used for a run. |
| Walk-forward | Backtest scheme with sequential refits on expanding/rolling windows. |

---

## 3. Requirements

### 3.1 Functional requirements

#### Model (FR-M)

| ID | Requirement |
|---|---|
| FR-M01 | The number of regimes `N ≥ 2` MUST be configurable. |
| FR-M02 | The emission family MUST be configurable per model (one family for all states in v1). GBM is the default. A family exposes: log-density `log f_i(y_t | Δ_t, θ_i, z^e_t)`, sampler, parameter specifications, native time convention, optional recurrent carry (for future GARCH-type families). |
| FR-M03 | Every parameter of emission, duration and transition components MUST support the modes `free`, `fixed`, `tied` (shared across a named group of states), `bounded` (free within `[lo, hi]`), `prior` (free with a log-prior term). Modes are set per parameter *and* per state. |
| FR-M04 | A state lattice MUST be generable from Cartesian products of parameter grids (e.g. drifts × volatilities), with all lattice parameters `fixed`. |
| FR-M05 | Ordering constraints (e.g. `σ_0 < σ_1 < …`) MUST be available for identifiability. |
| FR-M06 | Duration families: `geometric` (⇒ HMM special case), `negbin` (phase expansion), `poisson`, `gamma`, `weibull`, `lognormal` (all discretized), `nonparametric` (free pmf up to `D_max`). |
| FR-M07 | Duration distributions are parameterized in years and mapped to the grid via `Δ` (Section 4.3). |
| FR-M08 | Transitions are a jump matrix without self-transitions (HSMM). In HMM mode (`geometric`, `r = 1`) the model MAY alternatively be parameterized by a generator `Q` with `P_t = expm(Q Δ_t)`. |
| FR-M09 | Covariates MAY enter (a) transitions via multinomial logit, (b) durations via a log-rate (hazard) link in the expanded engine or a scale link evaluated at segment entry in the explicit-duration engine, (c) emissions via linear drift and log-volatility links. Each covariate declares which entry points it feeds. |
| FR-M10 | Higher order: memory over the previous `k−1` regimes (`k ≥ 1`), implemented by state-space expansion. Duration conditioning is `current` (default) or `pair`. `k > 2` emits a warning about parameter count. |
| FR-M11 | Priors/penalties MUST be available: Gaussian (ridge) on covariate coefficients, log-normal on volatilities and rates, Dirichlet on jump-matrix rows. MAP with priors is the default; `use_priors: false` yields MLE. |
| FR-M12 | All observation contracts carry a series dimension `M`. v1 implements `M = 1` fully; `M > 1` with independent per-series emissions is an extension ticket (E9-T01). |
| FR-M13 | Initial distribution over expanded states: `uniform` (default), `stationary`, or `free`. |

#### Inference and forecasting (FR-I)

| ID | Requirement |
|---|---|
| FR-I01 | Filtering, smoothing, Viterbi decoding (expanded space) and segmental decoding (list of segments) MUST be provided. |
| FR-I02 | The augmented posterior over (regime, phase or elapsed duration, memory) MUST be exposed, plus collapse operators to regime marginals. |
| FR-I03 | Log-likelihood, AIC, BIC and ICL MUST be computed for any fitted model. |
| FR-I04 | Forecasts for horizon `h` (grid steps) MUST return per-step regime probabilities, return samples `(n_sim, h, M)`, cumulative-return quantiles and moments. |
| FR-I05 | Covariate forecasters `freeze`, `scenario`, `bootstrap`, `ar` MUST be available and selectable; `h = 1` never requires one. |
| FR-I06 | Simulation from any (fitted or hand-specified) model MUST be available, including covariate-dependent components given a covariate path. |

#### Data (FR-D)

| ID | Requirement |
|---|---|
| FR-D01 | Providers behind one interface: `yfinance` (default), `stooq`, `csv`, `fred`, `cboe`; `tiingo` optional. Adding a provider MUST NOT require changes outside `regimekit.data.providers`. |
| FR-D02 | Observation span (`start`, `end`) and frequency (`1h`, `1d`, `1wk`, `1mo`) MUST be configurable; intraday support is best-effort within provider limits. |
| FR-D03 | Raw data MUST be cached locally (parquet) and versioned by a snapshot ID; a run MUST be reproducible offline from its snapshot. |
| FR-D04 | Series MUST be aligned to a master calendar (default: trading days of the primary instrument); `Δ_t` MUST be computed per configured clock. |
| FR-D05 | Preparation is a fit/transform object: log-returns, covariate lagging (lag ≥ 1 enforced), transforms (`log`, `diff`, `pct`, `zscore`), standardization statistics fitted on the training window only, missing-value policy per covariate (`ffill` with `max_staleness`, `drop`, `error`). |
| FR-D06 | A data-quality report (gaps, duplicates, outliers, adjustment discontinuities, staleness) MUST be produced by `fetch`. |
| FR-D07 | The model layer MUST NOT import the data layer. The only contract between them is `PreparedDataset` (Section 5.4). |

#### Evaluation and backtesting (FR-E)

| ID | Requirement |
|---|---|
| FR-E01 | A metric registry with configurable selection: `log_score`, `crps`, `pit_coverage`, `rmse`, `mae`, `aic`, `bic`, `icl`, `regime_occupancy`, `regime_persistence`, `balanced_accuracy` (if labels are supplied), strategy metrics `sharpe`, `max_drawdown`, `turnover`, `hit_rate`. |
| FR-E02 | Walk-forward backtesting with `expanding` or `rolling` windows, refit cadence, embargo, and per-horizon metric tables. |
| FR-E03 | A strategy interface mapping posterior/forecast to positions, with reference strategies `regime_long_flat` and `regime_target_vol`, and proportional transaction costs. |
| FR-E04 | Model comparison across multiple configs (same data snapshot) producing one table. |

#### Configuration, CLI, persistence (FR-C / FR-P)

| ID | Requirement |
|---|---|
| FR-C01 | One YAML file validated by pydantic v2 models with `schema_version`; unknown keys are errors; every defaulted field is echoed into the resolved config stored with results. |
| FR-C02 | CLI commands: `validate-config`, `fetch`, `fit`, `decode`, `forecast`, `backtest`, `simulate`, `compare`, `selftest`. |
| FR-C03 | A Python API (`Project`) MUST offer the same operations as the CLI. |
| FR-P01 | Fitted models persist as a bundle (Section 5.8) with lossless round-trip. |
| FR-P02 | Results (posteriors, forecasts, backtest tables) persist as parquet/JSON. |

#### Extensibility (FR-X)

| ID | Requirement |
|---|---|
| FR-X01 | Factorial/coupled readiness: structured state index with collapse operators, composable transition builders, and an approximate-inference plug-in point in the engine interface. |
| FR-X02 | Endogenous-covariate readiness: emission interface accepts `M`-dimensional observations so covariates can later become emission channels. |

### 3.2 Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | Numerics: all recursions in log space; float64 by default (`jax_enable_x64`); no NaN/Inf may leave a public function (guarded, raise `NumericalError`). |
| NFR-02 | Performance guardrails (8-core x86 CPU, T = 5000, N = 4, NB r = 5, order 1, 2 covariates): jitted forward-backward ≤ 100 ms; one L-BFGS fit ≤ 500 iterations ≤ 60 s; explicit-duration forward-backward with `D_max = 250` ≤ 2 s. Benchmarks are recorded in CI; regressions > 30 % fail the benchmark job (not the test job). |
| NFR-03 | Reproducibility: identical (config, snapshot, seed, backend) ⇒ bitwise identical parameters and forecasts. |
| NFR-04 | Tests: ≥ 90 % line coverage enforced; property tests for invariants; golden tests against NumPy reference; parameter-recovery tests. |
| NFR-05 | Typing: mypy strict on `src/`; runtime shape/dtype checks (jaxtyping + beartype) on public numerical functions in tests and optionally at runtime. |
| NFR-06 | Portability: CPU-only install is the default; GPU via optional extra `gpu`. Linux and WSL2 are supported targets; macOS CPU best-effort. |
| NFR-07 | Documentation: user guide, math appendix, API reference, ADRs, one end-to-end tutorial. |
| NFR-08 | Licensing/ToS: raw provider data is never redistributed or committed; caches are local; provider ToS notes are surfaced in docs and in `fetch` logs. Library license: Apache-2.0 (ADR-0002). |
| NFR-09 | Observability: structured logging (`structlog`), fit progress every k iterations, timings in reports. |
| NFR-10 | Stability: semantic versioning; config `schema_version` with migration hooks. |

---

## 4. Mathematical specification

### 4.1 Time conventions

- Observation times `τ_0 < τ_1 < … < τ_T`; observations `y_t` for `t = 1..T` refer to interval `(τ_{t−1}, τ_t]`.
- `Δ_t` in years: `clock = trading` ⇒ `Δ_t = 1/(252·H)` with `H` bars per trading day (`H = 1` daily); `clock = calendar` ⇒ `Δ_t = (τ_t − τ_{t−1}) / 365.25`.
- All model parameters are annualized. Emission and duration components receive `Δ_t` explicitly; nothing inside the model assumes a constant grid except the explicit-duration engine (Section 4.7.3), which uses the nominal `Δ = median(Δ_t)` and warns if `max Δ_t / min Δ_t > 1.5`.

### 4.2 Observations and emission families

`y_t ∈ ℝ^M` are log-returns of adjusted prices, `y_t = log(P_t / P_{t−1})`. `M = 1` in v1.

**GBM (default).** In regime `i`: `dS = μ_i S dt + σ_i S dW` ⇒ exactly `y_t | i ~ N((μ_i − σ_i²/2) Δ_t, σ_i² Δ_t)`. With emission covariates `z^e_t ∈ ℝ^{K_e}`:

```
μ_i(t)      = μ_i + γ_iᵀ z^e_t
log σ_i(t)  = log σ_i + δ_iᵀ z^e_t
```

**Student-t.** `y_t | i ~ m_i Δ_t + s_i √Δ_t · t_{ν_i}` with `ν_i > 2` (ν may be tied across states). Same covariate links on `m_i`, `log s_i`.

**Interface contract of a family** (mathematical): `logpdf(y, Δ, θ_i, z^e) → ℝ`, `sample(key, Δ, θ_i, z^e) → ℝ^M`, `param_specs() → list[ParamSpec]`, `native_time ∈ {continuous, discrete}`, `init_carry() / step_carry()` (identity for GBM; reserved for GARCH-type families).

Families planned but not in v1: OU (mean reversion, exact discretization), Merton jump-diffusion, MS-GARCH (Haas et al. 2004).

### 4.3 Regime process

**HSMM definition.** The latent process is a sequence of segments `(q_1, d_1), (q_2, d_2), …` with `q_{n+1} | q_n = i ~ π_{i·}`, `π_ii = 0`, and `d_n | q_n = i ~ p_i(d)`, `d ∈ {1, …, D_max}`.

**Continuous-time sojourn → grid pmf.** For a sojourn cdf `F_i` in years (Gamma, Weibull, log-normal), `p_i(d) = [F_i(dΔ) − F_i((d−1)Δ)] / F_i(D_max Δ)`. Shifted Poisson: `d = 1 + Poisson(κ_i Δ⁻¹)` truncated at `D_max`.

**Geometric (HMM special case).** Exit rate `λ_i` (per year): per-step exit probability `a_i(t) = 1 − exp(−λ_i Δ_t)`; expected sojourn `1/λ_i` years. Agrees to first order in `Δ` (error `O((λΔ)²)`, i.e. multiple jumps within one interval are neglected) with the CTMC generator `Q_ij = λ_i π_ij`, `Q_ii = −λ_i` observed on the grid; exact CTMC semantics are available via generator mode below.

**Negative-binomial via phase expansion (default HSMM engine path).** Regime `i` has `r_i ≥ 1` phases, each with advance probability `a_i(t) = 1 − exp(−λ_i(t) Δ_t)`. The chain enters phase 1, advances phase by phase, and exits from phase `r_i` to `j ≠ i` with probability `a_i(t) π_ij(t)`. On a constant grid the sojourn satisfies `d − r_i ~ NegBin(r_i, a_i)` (failures before `r_i` successes), so `E[d] = r_i / a_i`, `Var[d] = r_i (1 − a_i) / a_i²`. `r_i` is an integer hyperparameter (config), selected by information criteria, not by gradient. Minimum duration is `r_i` steps (documented; an entry-distribution variant that removes this floor is a backlog item).

**Hazard form.** For the explicit-duration engine, `h_i(d) = p_i(d) / S_i(d−1)` with survival `S_i(d) = 1 − Σ_{u≤d} p_i(u)`.

**Generator mode (HMM only).** `P_t = expm(Q Δ_t)` with `Q` parameterized by `log λ_i` and `π`. Provided for users who want exact CTMC semantics across frequencies; not used by the HSMM paths.

### 4.4 Covariates

**Alignment.** A covariate used at step `t` must be known at the end of `t−1`. The preparation layer enforces `lag ≥ 1` (in grid steps) per covariate and stores `z_t` already shifted. Standardization uses statistics fitted on the training window only; the statistics are persisted with the model.

**Transitions (multinomial logit).** For current regime `i` (or memory tuple `m`), target `j ≠ i`:

```
π_ij(t) = exp(a_ij + b_ijᵀ z^π_t) / Σ_{l≠i} exp(a_il + b_ilᵀ z^π_t)
```

with one reference target per row fixed to `a_i,ref = 0`, `b_i,ref = 0` for identifiability. No covariates ⇒ `π` constant.

**Durations.** Expanded engine: `log λ_i(t) = log λ_i + c_iᵀ z^d_t` (proportional-hazard-like, evaluated every step). Explicit-duration engine: `log(scale_i) = log(scale_i^0) + c_iᵀ z^d_{t_entry}` evaluated once at segment entry (the pmf is fixed for the segment).

**Emissions.** As in Section 4.2.

**Regularization.** Default priors `b, c, γ, δ ~ N(0, τ²)` with `τ = 1` on standardized covariates (config `coef_prior_sd`).

### 4.5 Higher-order (segment memory)

Order `k` means the jump distribution out of the current regime depends on the previous `k−1` regimes: `P(q_{n+1} = l | q_n = j, q_{n−1} = i, …)`. Implementation for `k = 2`: memory-augmented regime index `(i, j)` with `i ≠ j`, transitions `(i, j) → (j, l)` with `l ≠ j`, `π_{(i,j) l}(t)` multinomial-logit as in 4.4. Duration attached to `j` (`duration_conditioning: current`) or to `(i, j)` (`pair`). General `k` uses the last `k−1` regimes as memory; the builder is generic, parameter counts grow as `N (N−1)^{k−1} (N−2)` (Appendix B). Initial memory at `t = 1`: uniform over admissible tuples (default) or free.

### 4.6 Parameters, modes and constraints

Every parameter is declared by a `ParamSpec(name, shape, domain, mode, value|init, bounds, prior, tie_group, constraint)`. The `ParamManager` maps between a flat unconstrained vector `u ∈ ℝ^p` (what the optimizer sees) and the constrained pytree `θ` (what the model sees) via bijectors:

| Domain | Bijector `u → θ` |
|---|---|
| real | identity |
| positive (σ, λ, rates, Gamma shape/scale) | `softplus` (default) or `exp` |
| interval `[lo, hi]` (`bounded`) | `lo + (hi − lo)·sigmoid(u)` |
| simplex row with reference category | `softmax([0, u])` |
| `ν > 2` | `2 + softplus(u)` |
| nonparametric pmf over `D_max` | `softmax(u)` |
| increasing vector (ordering constraint) | `θ_0 = g(u_0)`, `θ_i = θ_{i−1} + softplus(u_i)` |

Modes: `fixed` ⇒ not in `u`; `tied` ⇒ one entry in `u` broadcast to the tie group; `prior` ⇒ adds `log p(θ)` to the objective (evaluated on `θ`, with the log-det-Jacobian of the bijector included so the MAP is defined on the constrained scale).

### 4.7 Likelihood and inference

#### 4.7.1 Structured state index

The builder produces `S` expanded states with index arrays `regime_of[s]`, `phase_of[s]`, `memory_of[s]` and a collapse matrix `C ∈ {0,1}^{S×N}` (`C[s, regime_of[s]] = 1`). `S = |memory tuples| · Σ_i r_i`; for order 1, `S = Σ_i r_i`.

#### 4.7.2 Expanded-HMM engine (default)

Per step, the transition matrix `A_t ∈ ℝ^{S×S}` is constructed on the fly inside `lax.scan` (never materialized for all `t`):

```
stay:     A_t[(m,i,c) → (m,i,c)]     = 1 − a_i(t)
advance:  A_t[(m,i,c) → (m,i,c+1)]   = a_i(t)                 for c < r_i
exit:     A_t[(m,i,r_i) → (m',j,1)]  = a_i(t) · π_{(m,i) j}(t)  for j ≠ i, m' = shift(m, i)
```

Emission log-density `log f_{regime_of[s]}(y_t)` is shared across phases/memory of one regime (computed once per regime, gathered by `C`).

Forward (log space): `α_1(s) = log ρ(s) + ℓ_1(s)`, `α_t(s) = ℓ_t(s) + logsumexp_{s'}[α_{t−1}(s') + log A_t(s', s)]`; log-likelihood `= logsumexp_s α_T(s)`. Backward: `β_T = 0`, `β_t(s) = logsumexp_{s'}[log A_{t+1}(s, s') + ℓ_{t+1}(s') + β_{t+1}(s')]`. Smoothed `γ_t(s) = softmax(α_t + β_t)`; filtered `softmax(α_t)`. Regime marginals `= γ_t C`. Complexity `O(T S²)`; `S` typically 10–60.

Irregular `Δ_t` is handled natively (`a_i(t)` uses `Δ_t`).

#### 4.7.3 Explicit-duration engine

For arbitrary duration pmfs (Poisson, Gamma, Weibull, log-normal, nonparametric) the Guédon (2003)/Yu (2010) explicit-duration forward-backward is implemented with right-censoring of the last segment and `D_max` truncation:

```
α_t(j, d) = P(segment of regime j ends at t with duration d, y_{1:t})
          = [Σ_i Σ_{d'} α_{t−d}(i, d') π_ij] · p_j(d) · Π_{u=t−d+1}^{t} f_j(y_u)
```

implemented with cumulative log-emission sums and a `(T, N, D_max)` scan. Complexity `O(T N² + T N D_max)`. Requires a regular grid (Section 4.1). Duration covariates enter at segment entry. Outputs identical in shape to 4.7.2 (augmented state = (regime, elapsed duration)).

Equivalence test: for a shifted-NB pmf both engines MUST agree on the log-likelihood (Appendix D).

#### 4.7.4 Decoding

Viterbi on the expanded space (log-space max-product) followed by collapse to regimes and run-length encoding to segments `[(start, end, regime)]`. The explicit-duration engine uses the segmental Viterbi variant.

### 4.8 Estimation

**Objective.** `J(u) = −ℓ(θ(u)) − Σ log p(θ) − Σ log|det ∂θ/∂u|` (MAP), or `J = −ℓ` if `use_priors: false`. Gradients via `jax.grad`; the scan is differentiable end-to-end.

**Optimizers.** Default `optax.lbfgs` with zoom line search; fallback `adam` (lr 1e-2 with cosine decay) for badly conditioned starts. Convergence: `|ΔJ| ≤ tol_f · max(1, |J|)` (default 1e-9) or `‖∇J‖_∞ ≤ tol_g` (default 1e-6), `max_iter` (default 1000). Non-finite `J` or gradient aborts the restart with a diagnostic, never the whole fit.

**Initialization strategies** (`init` list, tried in order, each producing one or more starts):
- `rollvol_kmeans`: 21-step realized volatility → k-means with `N` clusters → `σ_i` sorted ascending, `μ_i` = cluster mean return (annualized), `λ_i` from mean cluster run length, `π` from empirical cluster transitions.
- `lattice_jitter`: fixed values from lattice/config `init`, free parameters jittered by ±10 % log-normal.
- `random`: uniform in bounds/log-space plausible ranges (σ ∈ [0.05, 0.8], λ ∈ [1, 50]).

**Multi-start.** `restarts` (default 8) starts spread over strategies; each restart gets an independent PRNG key derived from the seed; the best objective wins; all restarts are reported (objective, iterations, converged flag, wall time). Label switching is prevented by ordering constraints or fixed parameters; if neither is configured, states are post-sorted by `σ` and a warning is issued.

**Standard errors (optional).** Observed information via `jax.hessian` at the optimum on `u`, delta method to `θ`; reported with a caveat when the Hessian is not positive definite.

**EM.** A generalized Baum-Welch for GBM emissions without covariates is a v1.x option (E5-T06); the MAP path is authoritative.

### 4.9 Forecasting

Inputs: fitted model, `PreparedDataset` up to `t`, horizon `h`, `n_sim`, covariate forecaster, seed.

```
1. α_t ← filtered augmented posterior at t (expanded states)
2. for each of n_sim paths:
     s ← sample(softmax(α_t))
     for k = 1..h:
         z_{t+k} ← known if k = 1 else CovariateForecaster.step(...)
         build A_{t+k}(z_{t+k}); s ← sample(A_{t+k}[s, ·])
         y_{t+k} ← EmissionFamily.sample(regime_of[s], Δ_{t+k}, z^e_{t+k})
3. aggregate: regime_probs (h, N), samples (n_sim, h, M), cumulative-return quantiles, mean, variance
```

`Δ_{t+k}` follows the calendar (next trading days) — forecasts are indexed by future grid dates.

**Covariate forecasters.**

| Type | Rule |
|---|---|
| `freeze` | `z_{t+k} = z_t` |
| `scenario` | user-supplied path `(h, K)`; missing columns fall back to `freeze` |
| `bootstrap` | stationary block bootstrap of historical first differences (all covariates jointly), mean block length configurable, re-standardized with training stats |
| `ar` | VAR(1) fitted on the training window by OLS with Gaussian innovations; per-covariate AR(1) if `diagonal: true` |

### 4.10 Model selection and diagnostics

- Information criteria: `AIC = 2p − 2ℓ`, `BIC = p log T − 2ℓ`, `ICL = BIC + 2·H(γ)` with `H` the smoothed-posterior entropy; `p` counts free entries of `u`.
- Out-of-sample: per-horizon log score and CRPS from the walk-forward backtest are the primary selection criteria.
- Diagnostics emitted with every fit: regime occupancy (warn if any regime < 2 % of steps — indicates misspecified fixed parameters or too many states), posterior mean duration vs. configured/estimated duration, PIT histogram of one-step forecasts, Ljung-Box on standardized residuals, ordering/label-switching check, restart dispersion (warn if best two objectives differ by > 1 % — likely multimodality).

### 4.11 Worked default

Three-state GBM on a daily index, drifts fixed at `{−0.30, 0.00, 0.20}`, volatilities free with increasing constraint, NB durations `r = {3, 3, 2}`, VIX (log, lag 1) in transitions and durations, term spread in transitions, order 1. Full YAML in Appendix A.

---

## 5. Architecture

### 5.1 Principles

1. **Two layers, one contract.** Data layer and model layer are independent packages inside one distribution; the only shared type is `PreparedDataset`. The model layer has no network access and no provider knowledge.
2. **One engine, many builders.** Model structure (durations, memory, later factorial) is compiled into a `StateSpace` (index arrays + transition-construction function). Engines operate on `StateSpace`, emissions and data only.
3. **Everything is a pytree.** Parameters, specs and results are `equinox.Module`s or frozen dataclasses registered as pytrees; JIT boundaries are explicit and few (`loglik`, `forward_backward`, `viterbi`, `simulate_paths`).
4. **Config-driven, code-first.** The YAML config is a serialization of pydantic models; the Python API constructs the same objects. Nothing is reachable from YAML that is not reachable from Python.
5. **Reproducibility is a contract.** Every artifact stores the resolved config, snapshot ID, seed, library version and backend.
6. **Fail loudly on numerics.** Guards convert NaN/Inf, empty states and degenerate grids into typed exceptions with actionable messages.

### 5.2 Component overview

```
┌──────────────────────────────── regimekit ─────────────────────────────────┐
│  cli / Project API                                                           │
│  ┌───────────────┐   PreparedDataset   ┌──────────────────────────────────┐ │
│  │  data          │ ─────────────────▶ │  model                            │ │
│  │  providers     │                    │  spec · families · params         │ │
│  │  cache/snapshot│                    │  statespace builders              │ │
│  │  calendar      │                    │  engine (expanded | explicit)     │ │
│  │  prepare       │                    │  estimation · inference · forecast│ │
│  │  quality       │                    └──────────────┬───────────────────┘ │
│  └───────────────┘                                   │ Posterior/Forecast   │
│                       ┌──────────────┐   ┌───────────▼─────────┐            │
│                       │ persistence  │◀──│ evaluation           │            │
│                       │ bundle/store │   │ metrics·backtest·strategy         │
│                       └──────────────┘   └─────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Package layout (src layout)

```
regimekit/
├── pyproject.toml            # uv-managed; extras: gpu, mcmc, dev groups
├── uv.lock
├── src/regimekit/
│   ├── __init__.py           # version, Project re-export, jax x64 bootstrap
│   ├── project.py            # Project facade (Python API mirroring CLI)
│   ├── config/               # pydantic schema (schema.py), yaml_io.py, defaults.py, migrate.py
│   ├── data/
│   │   ├── contracts.py      # ObservationSet, CovariateSeries, PreparedDataset, PrepStats
│   │   ├── providers/        # base.py (Protocol+registry), yfinance.py, stooq.py, tiingo.py,
│   │   │                     # fred.py, cboe.py, csv.py
│   │   ├── cache.py          # parquet cache, snapshot ids
│   │   ├── calendar.py       # master grid, Δ computation, clocks
│   │   ├── prepare.py        # Preparer (fit/transform), transforms, missing policies
│   │   └── quality.py        # DataQualityReport
│   ├── model/
│   │   ├── spec.py           # ModelSpec (structure only, no data)
│   │   ├── params.py         # ParamSpec, ParamManager, bijectors, priors
│   │   ├── dists.py          # log-pdfs/pmfs + samplers (normal, student_t, poisson, negbin,
│   │   │                     # discretized gamma/weibull/lognormal, nonparametric)
│   │   ├── emission/         # base.py, gbm.py, student_t.py
│   │   ├── duration/         # base.py, geometric.py, negbin.py, poisson.py, gamma.py, weibull.py,
│   │   │                     # lognormal.py, nonparametric.py
│   │   ├── transition.py     # jump matrix, generator mode, multinomial-logit link
│   │   ├── covariates.py     # CovariateSet, link helpers
│   │   └── lattice.py        # state lattice generator, ordering constraints
│   ├── statespace/
│   │   ├── index.py          # StateIndex (regime_of, phase_of, memory_of, collapse C)
│   │   ├── builders.py       # build_expanded(spec) → StateSpace; memory expansion; (factorial later)
│   │   └── transitions.py    # per-step A_t construction (jit-able)
│   ├── engine/
│   │   ├── base.py           # InferenceEngine Protocol, Posterior dataclass
│   │   ├── expanded_hmm.py   # log-space forward/backward/viterbi via lax.scan
│   │   ├── explicit_duration.py
│   │   └── viterbi.py
│   ├── estimation/
│   │   ├── objective.py      # NLL + priors + log-det-J
│   │   ├── optimizers.py     # lbfgs / adam wrappers, convergence
│   │   ├── init.py           # rollvol_kmeans, lattice_jitter, random
│   │   ├── fit.py            # multi-start orchestration → FitResult
│   │   └── report.py         # diagnostics, std errors
│   ├── inference/
│   │   ├── decode.py         # filter/smooth/viterbi/segments → PosteriorResult
│   │   ├── simulate.py       # simulate regimes + returns
│   │   ├── forecast.py       # MonteCarloForecaster → ForecastResult
│   │   └── covariate_forecasters.py
│   ├── evaluation/
│   │   ├── metrics/          # registry.py + one module per metric family
│   │   ├── backtest.py       # walk-forward
│   │   ├── strategies.py     # Strategy Protocol + reference strategies
│   │   └── compare.py
│   ├── persistence/
│   │   ├── store.py          # ModelBundle save/load
│   │   └── results.py        # parquet/json writers
│   ├── cli/app.py            # typer application
│   └── utils/                # jax_setup.py, rng.py, logging.py, errors.py, typing.py
├── tests/
│   ├── reference/            # pure-NumPy reference implementations (small, readable)
│   ├── unit/ property/ golden/ recovery/ integration/ benchmark/
│   └── fixtures/             # synthetic datasets, recorded provider responses
├── docs/                     # mkdocs: guide/, math/, api/, adr/
└── examples/                 # configs + tutorial notebook
```

### 5.4 Core data contracts

All arrays are `jax.Array` float64 unless stated; shapes in comments. Contracts are frozen dataclasses (pytrees) with `validate()` methods.

```python
@dataclass(frozen=True)
class ObservationSet:                       # raw, per provider, before alignment
    symbols: tuple[str, ...]                # M
    times: np.ndarray                       # (T0,) datetime64[ns, UTC]
    prices: np.ndarray                      # (T0, M) adjusted close (or configured field)
    interval: str                           # "1h" | "1d" | "1wk" | "1mo"
    adjusted: bool
    provider: str
    snapshot_id: str                        # sha256 of (provider, symbols, interval, raw bytes, fetched_at)
    meta: dict[str, Any]

@dataclass(frozen=True)
class CovariateSeries:
    name: str; times: np.ndarray; values: np.ndarray  # (T1,)
    provider: str; series_id: str; snapshot_id: str; frequency: str

@dataclass(frozen=True)
class PrepStats:                            # fitted on the training window only
    cov_mean: np.ndarray; cov_std: np.ndarray            # (K,)
    train_window: tuple[str, str]; cov_names: tuple[str, ...]
    transforms: dict[str, list[str]]; lags: dict[str, int]

@dataclass(frozen=True)
class PreparedDataset:                      # the ONLY object the model layer consumes
    times: np.ndarray                       # (T,) grid end times
    returns: Array                          # (T, M) log-returns
    dt: Array                               # (T,) Δ_t in years
    mask: Array                             # (T, M) bool, True = observed
    cov_transition: Array | None            # (T, K_pi)  already lagged & standardized
    cov_duration: Array | None              # (T, K_d)
    cov_emission: Array | None              # (T, K_e)
    cov_names: dict[str, tuple[str, ...]]   # per entry point
    clock: str; nominal_dt: float
    prep_stats: PrepStats; snapshot_ids: tuple[str, ...]
    def window(self, start: int, end: int) -> "PreparedDataset": ...   # slicing keeps contracts
```

Model-side contracts:

```python
class ModelSpec(eqx.Module):                # structure only
    n_states: int; n_series: int
    emission: EmissionFamily; duration: DurationFamily; transition: TransitionSpec
    order: int; duration_conditioning: Literal["current", "pair"]
    engine: Literal["expanded", "explicit"]; d_max: int
    initial: Literal["uniform", "stationary", "free"]
    param_specs: tuple[ParamSpec, ...]      # union of all component specs

class StateSpace(eqx.Module):
    index: StateIndex                       # regime_of (S,), phase_of (S,), memory_of (S, k-1), C (S, N)
    n_expanded: int
    transition_fn: Callable[[Params, Array, Array], Array]   # (θ, z_pi_t, z_d_t, Δ_t) → A_t (S, S)

@dataclass(frozen=True)
class Posterior:
    log_alpha: Array                        # (T, S) filtered, log space
    log_beta: Array                         # (T, S)
    smoothed: Array                         # (T, S)
    loglik: float
    index: StateIndex
    def regime_filtered(self) -> Array: ... # (T, N)
    def regime_smoothed(self) -> Array: ...
    def elapsed_posterior(self) -> Array: ... # (T, N, D_max) for explicit engine; phase marginals for expanded

@dataclass(frozen=True)
class PosteriorResult:                      # user-facing
    times: np.ndarray; filtered: Array; smoothed: Array          # (T, N)
    viterbi: np.ndarray; segments: list[tuple[int, int, int]]    # (start, end, regime)
    augmented: Posterior; loglik: float; aic: float; bic: float; icl: float

@dataclass(frozen=True)
class FitResult:
    params: Params; uparams: Array; objective: float; loglik: float
    restarts: pd.DataFrame                  # objective, iters, converged, wall_s, init_strategy
    converged: bool; n_iter: int; std_errors: dict[str, Array] | None
    diagnostics: dict[str, Any]; spec: ModelSpec; prep_stats: PrepStats; config_resolved: dict

@dataclass(frozen=True)
class ForecastResult:
    origin_time: np.datetime64; horizon_times: np.ndarray       # (h,)
    regime_probs: Array                     # (h, N)
    samples: Array                          # (n_sim, h, M) per-step log-returns
    cum_quantiles: Array                    # (h, Q, M)
    mean: Array; var: Array                 # (h, M)
    covariate_forecaster: str; seed: int

@dataclass(frozen=True)
class BacktestResult:
    folds: pd.DataFrame                     # fold, train_start, train_end, test_start, test_end, fit_time
    metrics: pd.DataFrame                   # fold × horizon × metric
    strategy: pd.DataFrame | None           # equity curve, positions, turnover
    summary: pd.DataFrame                   # aggregated per horizon/metric
```

### 5.5 Interfaces (Protocols)

```python
class EmissionFamily(Protocol):
    name: str; native_time: Literal["continuous", "discrete"]
    def param_specs(self, n_states: int, n_series: int, cov_names: tuple[str, ...]) -> tuple[ParamSpec, ...]: ...
    def log_prob(self, y: Array, dt: Array, theta: Params, z: Array | None) -> Array:
        """y (T, M), dt (T,), z (T, K_e) → (T, N) log-densities per regime."""
    def sample(self, key: Array, regime: Array, dt: Array, theta: Params, z: Array | None) -> Array: ...
    def init_carry(self) -> Any: ...       # None for GBM/Student-t
    def sufficient_stats(self, y, dt, gamma) -> Any | None: ...   # optional, enables EM

class DurationFamily(Protocol):
    name: str; representation: Literal["phase", "explicit"]
    def param_specs(self, n_states: int, cov_names) -> tuple[ParamSpec, ...]: ...
    # phase representation
    def n_phases(self, i: int) -> int: ...
    def advance_prob(self, theta: Params, dt: Array, z_d: Array | None) -> Array:  # (T, N)
    # explicit representation
    def log_pmf(self, theta: Params, d_max: int, nominal_dt: float, z_entry: Array | None) -> Array:  # (N, D_max)
    def mean_sojourn_years(self, theta: Params) -> Array: ...

class TransitionModel(Protocol):
    def param_specs(self, n_states, order, cov_names) -> tuple[ParamSpec, ...]: ...
    def jump_probs(self, theta: Params, z_pi: Array | None, memory_index: StateIndex) -> Array:
        """→ (T, |memory tuples|, N) rows sum to 1 with zero on the current regime."""

class StateSpaceBuilder(Protocol):
    def build(self, spec: ModelSpec) -> StateSpace: ...

class InferenceEngine(Protocol):
    def loglik(self, params, spec, space, data: PreparedDataset) -> float: ...
    def forward_backward(self, params, spec, space, data) -> Posterior: ...
    def viterbi(self, params, spec, space, data) -> Array: ...          # (T,) expanded states
    def approx(self) -> ApproxInference | None: ...                       # FR-X01 plug-in point

class Estimator(Protocol):
    def fit(self, spec, data, cfg: EstimationConfig, key) -> FitResult: ...

class DataProvider(Protocol):
    name: str; supports_intraday: bool; tos_note: str
    def fetch_prices(self, symbols, start, end, interval, adjusted) -> ObservationSet: ...
    def fetch_series(self, series_id, start, end) -> CovariateSeries: ...   # macro providers

class Preparer:                              # fit/transform, sklearn-style
    def fit(self, obs: ObservationSet, covs: list[CovariateSeries], train_end: str) -> "Preparer": ...
    def transform(self, obs, covs) -> PreparedDataset: ...

class CovariateForecaster(Protocol):
    def fit(self, data: PreparedDataset) -> "CovariateForecaster": ...
    def sample_paths(self, key, h: int, n_sim: int, last: Array) -> Array: ...  # (n_sim, h, K)

class Metric(Protocol):
    name: str; needs: set[Literal["forecast", "posterior", "fit", "labels", "strategy"]]
    def __call__(self, **ctx) -> float | dict[str, float]: ...

class Strategy(Protocol):
    def positions(self, posterior: PosteriorResult, forecast: ForecastResult | None) -> Array: ...  # (T, M) in [-1, 1]

class ModelStore(Protocol):
    def save(self, fit: FitResult, path: Path) -> None: ...
    def load(self, path: Path) -> FitResult: ...
```

### 5.6 Data layer design

**Providers.** Registry keyed by name; each adapter converts provider output into `ObservationSet`/`CovariateSeries` in UTC with float64 values, applying retries with exponential backoff (max 5) and provider-specific throttling. Adapters:

| Adapter | Use | Notes |
|---|---|---|
| `yfinance` | prices, indices, FX, ETFs (default) | `auto_adjust=True`; `curl_cffi` session; intraday windows capped (1m ≈ 7 d, 1h ≈ 730 d); logs the personal-use ToS note on each fetch |
| `stooq` | EOD fallback, EU coverage | API key from env `STOOQ_API_KEY`; daily quota — cache aggressively |
| `tiingo` (optional extra) | clean US EOD | key `TIINGO_API_KEY`; free-tier limits enforced client-side |
| `fred` | macro/covariates | key `FRED_API_KEY`; 120 req/min; `"."` → NaN |
| `cboe` | VIX/VVIX daily CSV | no key; URL patterns in config |
| `csv` | local files / tests | reference adapter; must pass the same contract tests as network adapters |

**Cache and snapshots.** `~/.cache/regimekit/<provider>/<symbol>_<interval>_<adjusted>.parquet` plus `.meta.json` (fetched_at, provider version, request params). `snapshot_id = sha256(provider, symbols, interval, adjusted, raw parquet bytes)`; a run records all snapshot IDs and can be replayed with `--offline`. `snapshot: latest` re-fetches if the cache is older than `max_age` (default 1 day); `snapshot: <id>` pins.

**Calendar.** Master grid = timestamps of the primary instrument after dropping rows with missing price. `Δ_t` per clock (Section 4.1). Covariates are aligned by `merge_asof(direction="backward")` onto the grid, then lagged by `lag` grid steps, then forward-filled up to `max_staleness` steps (beyond → NaN → policy `drop|error`). Mixed-frequency covariates (monthly macro) are therefore step functions with a publication-lag-aware `lag`.

**Preparation.** `Preparer.fit` computes `PrepStats` on `[start, train_end]`; `transform` applies to any window. Transforms are applied in declared order (`log → diff → zscore`). `PreparedDataset.window()` re-slices without re-fitting — used by the backtester.

**Quality report.** Gaps vs. exchange calendar, duplicated timestamps, zero/negative prices, |return| > 8σ (flag only), adjustment discontinuities between cache generations, covariate staleness histogram. Written as JSON + Markdown next to the cache.

### 5.7 Configuration schema (pydantic v2)

Top-level `ProjectConfig` with sections `project`, `data`, `model`, `estimation`, `forecast`, `evaluation`, `output`. Key models:

| Model | Fields (abridged) |
|---|---|
| `ProjectSection` | `name`, `seed: int`, `output_dir`, `log_level` |
| `DataSection` | `primary: PriceSource`, `clock`, `bars_per_day`, `cache: CacheCfg`, `covariates: list[CovariateCfg]`, `preparation: PrepCfg` |
| `PriceSource` | `provider`, `symbols: list[str]`, `interval`, `start`, `end`, `adjusted`, `field` |
| `CovariateCfg` | `name`, `provider`, `series`, `transform: list`, `lag: int ≥ 1`, `missing`, `max_staleness`, `use_in: list[transition|duration|emission]` |
| `ModelSection` | `n_states`, `engine`, `d_max`, `emission: EmissionCfg`, `duration: DurationCfg`, `transition: TransitionCfg`, `initial_state`, `lattice: LatticeCfg | None` |
| `ParamCfg` | `mode`, `values`, `init`, `bounds`, `constraint: none|increasing`, `prior: PriorCfg | None`, `tie: str | None` |
| `EmissionCfg` | `family`, `params: dict[str, ParamCfg]`, `covariates: {drift: [...], log_vol: [...]}` |
| `DurationCfg` | `family`, `r: list[int]` (negbin), `params`, `covariates: list[str]` |
| `TransitionCfg` | `order`, `duration_conditioning`, `covariates`, `prior`, `coef_prior_sd`, `mode: jump|generator` |
| `EstimationSection` | `optimizer`, `max_iter`, `tol_f`, `tol_g`, `restarts`, `init: list`, `use_priors`, `std_errors`, `device: cpu|gpu|auto` |
| `ForecastSection` | `horizon`, `n_sim`, `covariate_forecaster: {type, …}`, `quantiles` |
| `EvaluationSection` | `metrics: list`, `backtest: BacktestCfg`, `strategy: StrategyCfg | None`, `labels: LabelSource | None` |

Validation rules (all raise `ConfigError` with path and reason): `len(values) == n_states` for per-state lists; a covariate listed under `use_in` must exist; `order ≥ 1`; `engine: explicit` incompatible with `duration.family: negbin` with covariates evaluated per step (warn and switch to entry-time evaluation); `d_max ≥ 2·max expected sojourn in steps` (warn); `lag ≥ 1`. Unknown keys → error (`extra="forbid"`). The resolved config (all defaults filled) is written to `output_dir/config.resolved.yaml`.

### 5.8 Persistence

Model bundle directory `<name>.rkmodel/`:

| File | Content |
|---|---|
| `manifest.json` | bundle version, regimekit version, jax version, backend, created_at, seed |
| `config.resolved.yaml` | full configuration |
| `spec.json` | `ModelSpec` (structure) |
| `params.safetensors` | constrained `θ` and unconstrained `u` |
| `prep_stats.json` | standardization stats, transforms, lags |
| `snapshots.json` | snapshot IDs and provider metadata |
| `fit_report.json` | restarts table, convergence, diagnostics, std errors |

Round-trip invariant: `load(save(fit)).loglik(data) == fit.loglik` bitwise on the same backend. Results: `posterior.parquet` (times × regime probs + viterbi), `forecast.parquet` (long format), `backtest_metrics.parquet`, `strategy.parquet`, plus a `report.md` summary.

### 5.9 CLI and Python API

CLI (`typer`): `regimekit <command> CONFIG [--out DIR] [--offline] [--device cpu|gpu] [--seed N] [--log-level L]`. Commands: `validate-config`, `fetch`, `fit`, `decode` (`--model`), `forecast` (`--model --horizon --n-sim`), `backtest`, `simulate` (`--params from model|config --steps`), `compare` (`CONFIG...`), `selftest` (parameter-recovery smoke test on synthetic data). Exit codes: 0 ok, 2 config error, 3 data error, 4 numerical error, 5 provider error.

Python API:

```python
from regimekit import Project
p = Project.from_yaml("dax_3state.yaml")
data = p.fetch()                     # PreparedDataset (+ quality report)
fit = p.fit(data)                    # FitResult
post = p.decode(fit, data)           # PosteriorResult
fc = p.forecast(fit, data, horizon=20)
bt = p.backtest(data)                # BacktestResult
p.save(fit, "runs/dax_3state")
```

### 5.10 Extension readiness

| Extension | Prepared in v1 by |
|---|---|
| Multi-series universe (`M > 1`, independent emissions) | `M` dimension in all contracts; `EmissionFamily.log_prob` returns `(T, N)` after summing over series; `ProductEmission` wrapper ticket E9-T01 |
| Factorial/coupled HMM | `StateIndex` generalizes to tuples; `builders.py` accepts a list of chains and a coupling spec producing Kronecker-structured `A_t`; `InferenceEngine.approx()` returns a structured-variational/Gibbs plug-in (ADR-0008) |
| Endogenous covariates | covariates become extra emission channels (`M` grows) with a joint emission family; data contract unchanged |
| MS-GARCH | `EmissionFamily.init_carry/step_carry` are threaded through the engine's scan from day one (identity for GBM) |
| OU / jump-diffusion | new emission modules; exact/approximate transition densities given `Δ_t` |
| Bayesian sampling | objective exposed as `log_joint(u)` compatible with NumPyro/BlackJAX potentials (`mcmc` extra) |
| EM | `sufficient_stats` hook on families; E5-T06 |

---

## 6. Engineering guidelines

### 6.1 Stack and versions (verified 2026-09-03; pin in `uv.lock`, review quarterly)

| Area | Choice | Version floor | Notes |
|---|---|---|---|
| Python | CPython | 3.13 | JAX supports 3.12–3.15; 3.13 is the stable baseline; no free-threaded builds |
| Arrays / autodiff | `jax`, `jaxlib` | 0.11 | `jax_enable_x64=True` at import; CPU wheel default, `jax[cuda12]` via extra `gpu` |
| Pytrees / modules | `equinox` | 0.13 | Model/params as `eqx.Module`; `eqx.filter_jit` at API boundaries |
| Optimization | `optax` | 0.2.8 | `optax.lbfgs` + zoom line search; `adam` fallback |
| Optional MCMC | `numpyro` | 0.20 | extra `mcmc` only |
| Distributions | in-house `regimekit.model.dists` on `jax.scipy` | — | avoids a TFP dependency; validated against `scipy.stats` in tests |
| Data frames | `pandas` 2.x, `pyarrow` 25 | — | pandas for provider compatibility; parquet via pyarrow |
| Providers | `yfinance` 1.7 (+`curl_cffi ≥ 0.15`), `pandas-datareader` 0.10 (stooq/fred/famafrench), `fredapi` | — | all behind adapters |
| Config | `pydantic` 2.13, `pydantic-settings`, `ruamel.yaml` | — | `extra="forbid"` |
| CLI | `typer` | — | |
| Logging | `structlog` | — | JSON logs in CI, console locally |
| Tests | `pytest` 9, `pytest-cov`, `hypothesis` 6.163, `pytest-xdist`, `pytest-benchmark` | — | `fail_under = 90` |
| Quality | `ruff` 0.16, `mypy` 2.1, `jaxtyping` 0.3 + `beartype`, `pre-commit`, `nox` | — | `ty` is pre-1.0: not used in gates |
| Docs | `mkdocs-material` 9.7 + `mkdocstrings[python]` | — | maintenance mode acknowledged; migrate to Zensical when stable (ADR) |
| Release | `towncrier`, `python-semantic-release` | — | conventional commits |

`pyproject.toml` skeleton:

```toml
[project]
name = "regimekit"; requires-python = ">=3.13"
dependencies = ["jax>=0.11", "equinox>=0.13", "optax>=0.2.8", "numpy>=2.1", "pandas>=2.2",
  "pyarrow>=25", "pydantic>=2.13", "ruamel.yaml>=0.18", "typer>=0.15", "structlog>=25", "jaxtyping>=0.3"]
[project.optional-dependencies]
gpu = ["jax[cuda12]>=0.11"]
providers = ["yfinance>=1.7", "curl_cffi>=0.15", "pandas-datareader>=0.10", "fredapi>=0.5"]
mcmc = ["numpyro>=0.20"]
[dependency-groups]
dev = ["pytest>=9", "pytest-cov", "pytest-xdist", "pytest-benchmark", "hypothesis>=6.163", "beartype",
  "ruff>=0.16", "mypy>=2.1", "pre-commit", "nox", "scipy>=1.15", "mkdocs-material>=9.7", "mkdocstrings[python]"]
[tool.coverage.report]
fail_under = 90; show_missing = true; exclude_also = ["if TYPE_CHECKING:", "@overload"]
[tool.mypy]
strict = true; plugins = ["pydantic.mypy"]
[tool.ruff]
line-length = 110; target-version = "py313"
[tool.ruff.lint]
select = ["E","F","W","I","N","UP","B","SIM","PT","RUF","D","ANN","NPY"]; pydocstyle.convention = "google"
```

### 6.2 Coding standards

- **Typing.** `mypy --strict` on `src/`; arrays annotated with `jaxtyping` (`Float[Array, "T N"]`). Public numerical functions are decorated with `@jaxtyped(typechecker=beartype)`; the check is active in tests (env `REGIMEKIT_TYPECHECK=1`) and off by default at runtime.
- **Purity.** Everything under `model/`, `statespace/`, `engine/`, `estimation/objective.py`, `inference/` is pure and side-effect free (no I/O, no logging inside jitted code). I/O lives in `data/`, `persistence/`, `cli/`, `project.py`.
- **Errors.** Custom hierarchy in `utils/errors.py`: `RegimekitError → ConfigError, DataError, ProviderError, NumericalError, PersistenceError`. Messages include the config path or array name involved.
- **Docstrings.** Google style, with the math symbol mapping (`sigma ↔ σ_i`) in every family/engine docstring.
- **Naming.** Math-faithful (`log_alpha`, `jump_probs`, `advance_prob`); shapes in docstrings; `n_states`, `n_expanded`, `d_max`, `n_sim`.
- **Dependencies.** No new runtime dependency without an ADR.

### 6.3 JAX conventions

1. `regimekit.utils.jax_setup` enables x64 and selects the device at import; importing `regimekit` after any other JAX init that disabled x64 raises `NumericalError`.
2. JIT boundaries: `engine.loglik`, `engine.forward_backward`, `engine.viterbi`, `estimation.objective.value_and_grad`, `inference.simulate.simulate_paths`. Everything else is plain Python that composes these.
3. Static vs. dynamic: `N`, `S`, `r_i`, `order`, `d_max`, `T` are static (shape-defining); parameters, data and covariates are dynamic. Rebuilding the `StateSpace` recompiles — done once per fit.
4. Sequential recursions use `jax.lax.scan`; `A_t` is computed inside the scan body. `associative_scan` parallel smoothing is a benchmark-gated option (E4-T07).
5. All probabilities live in log space; normalizations via `jax.scipy.special.logsumexp`; `−inf` allowed for structural zeros (masked with `jnp.where` before `exp`).
6. Randomness: a single `seed` → `jax.random.key(seed)` → keys split per restart/path with `jax.random.fold_in(key, idx)`; no global RNG; NumPy RNG only in data-layer bootstrap with `np.random.default_rng(seed)`.
7. No Python control flow on traced values; branch on config, not on data. Masked observations contribute zero log-density (`jnp.where(mask, logf, 0)`).
8. Memory: never materialize `(T, S, S)`; `(T, N, D_max)` for the explicit engine is acceptable up to `T·N·D_max ≤ 5e7` floats, else chunked scan (E4-T06 acceptance).

### 6.4 Testing strategy

**Test pyramid and TDD scope.**

| Layer | Tests | TDD |
|---|---|---|
| `dists`, `params` (bijectors), `statespace`, `engine`, `estimation.objective`, `inference.forecast` aggregation, `evaluation.metrics` | unit + property + golden | mandatory: tests first |
| `data/prepare`, `data/calendar`, `data/cache` | unit with fixtures | mandatory |
| providers | contract tests against recorded responses (`fixtures/cassettes`) + opt-in live tests (`-m live`) | recommended |
| CLI, `Project` | integration on synthetic CSV | recommended |
| recovery, benchmark | slow suites (`-m slow`, `-m benchmark`) | n/a |

**6.4.1 Reference implementations.** `tests/reference/` contains small pure-NumPy implementations (HMM forward-backward, explicit-duration forward-backward, Viterbi, NB pmf, GBM log-density) written for readability, not speed. Golden tests compare JAX engines against them on random instances (hypothesis strategies: `N ∈ [2,4]`, `T ∈ [5,200]`, `r_i ∈ [1,3]`, `order ∈ {1,2}`, random covariates).

**6.4.2 Property tests (hypothesis).** Filtered/smoothed rows sum to 1; `α`/`β` consistency (`logsumexp(α_t + β_t)` constant in `t`); log-likelihood invariant to permuting equivalent phases; `ParamManager` round-trip `θ = θ(u(θ))`; bijector Jacobians vs. autodiff; duration pmfs sum to 1 and have the analytic mean; simulation → empirical sojourn mean within 3 SE of analytic; `expm(QΔ)` vs. geometric-family `A_t` in HMM mode agree to `O(Δ²)` and converge as `Δ → 0`.

**6.4.3 Gradient checks.** `jax.grad` of the objective vs. central finite differences on `u` (step 1e-6, relative error ≤ 1e-5) for every family/engine combination in a small instance.

**6.4.4 Parameter-recovery suite** (`tests/recovery/`; a nightly smoke run, a weekly full run and the on-demand `regimekit selftest`). Simulate with known `θ*`, fit with `restarts = 4`, assert (Appendix D): volatilities within 10 % relative, drifts within 2 SE, mean sojourns within 25 %, jump probabilities within 0.15 absolute, covariate coefficients sign-correct and within 2 SE. Scenarios: `3-state GBM/NB order 1`; `+ transition covariates`; `+ duration covariates` (coefficients `(0.6, −0.6, 0.4)`); `order 2`; `explicit gamma durations`; `fixed-drift lattice`. Default size `T = 20000` (`order 2`: `T = 50000`); `REGIMEKIT_RECOVERY_T` scales it. Pass criterion (amended 2026-10-18): each tolerance *category* (σ, μ, mean sojourn, jump probabilities, coefficients) holds in at least 4 of the 5 seeds, each category counted on its own; the mean-sojourn category of the explicit-gamma scenario is reported but not gated. Rationale: the tolerances are unchanged, but at `T = 5000` the default truth produces only about 64 regime visits per path, so even an oracle that reads the true regime path meets the sojourn and jump tolerances together in only half of the seeds (99 % at `T = 20000`); the 2-SE checks miss about 5 % of the time by design, so requiring every check in the same seed is not attainable; a duration-covariate effect of 0.2–0.3 is as small as its standard error at `T = 20000`; and with both gamma shape and scale free the mean sojourn is weakly identified (the fit's likelihood exceeds the truth's by about 12 nats). The nightly smoke run uses `T = 5000` on one scenario and asserts the pipeline and the σ category only.

**6.4.5 Engine equivalence.** Expanded (NB) vs. explicit (shifted-NB pmf, `D_max` with truncated mass < 1e-10): log-likelihood relative difference ≤ 1e-6; smoothed regime marginals max-abs ≤ 1e-6.

**6.4.6 Determinism.** Same config/snapshot/seed on CPU ⇒ bitwise identical `u` and forecast samples (test runs the fit twice).

**6.4.7 Coverage.** ≥ 90 % overall and ≥ 85 % per package (`model`, `engine`, `estimation`, `inference`, `data`, `evaluation`); providers' live paths excluded via `# pragma: no cover` only on the network call line.

### 6.5 CI/CD

- GitHub Actions: `lint` (ruff, mypy), `test` (Linux, Python 3.13, CPU, `pytest -n auto -m "not slow and not live and not benchmark"`), `docs` (mkdocs build strict), `recovery_smoke` (nightly, one scenario at `T = 5000`), `recovery` (weekly, `-m slow`, the full suite), `benchmark` (nightly, compares against stored baseline, fails on > 30 % regression), `gpu` (self-hosted runner if available, otherwise manual).
- `pre-commit`: ruff format/lint, mypy (staged), end-of-file, yaml check, `uv lock --check`.
- `nox` sessions: `lint`, `types`, `tests`, `recovery_smoke`, `recovery`, `docs`, `bench`.
- Branching: trunk-based; short-lived branches `feat/E4-T03-expanded-engine`; squash merges; PR title = ticket ID + title; conventional commit prefix.
- Release: semantic-release on tags; changelog fragments via towncrier; `0.y.z` until the factorial extension lands.

### 6.6 Documentation

`docs/guide` (install, quickstart, config reference generated from pydantic schema, data providers and ToS, forecasting, backtesting), `docs/math` (Sections 4.x rendered with MathJax, one page per component), `docs/api` (mkdocstrings), `docs/adr` (Section 6.7), `examples/` (DAX and S&P configs, tutorial notebook executed in CI with `--offline` fixtures).

### 6.7 Architecture Decision Records

Format: `docs/adr/NNNN-title.md` with Context / Decision / Consequences / Status. Initial set to be written in E0-T04:

| ADR | Decision (proposed default) |
|---|---|
| 0001 | CT parameterization, DT inference (D1) |
| 0002 | Library license Apache-2.0; provider data never redistributed |
| 0003 | JAX + Equinox + Optax; no TFP dependency; in-house distributions |
| 0004 | One engine over structured state index; builders for memory/NB/factorial (D2) |
| 0005 | MAP via L-BFGS as primary estimator; EM optional |
| 0006 | Shifted-NB semantics (minimum duration `r_i`) for the phase expansion |
| 0007 | Explicit-duration engine requires a regular grid |
| 0008 | Approximate-inference plug-in interface for factorial models |
| 0009 | pydantic + YAML (not Hydra) for configuration |
| 0010 | `typer` CLI; `pandas` in the data layer |

### 6.8 Definition of Done (per ticket)

1. Acceptance criteria demonstrably met by tests named after the ticket (`test_e4_t03_*`).
2. Coverage of touched modules not decreased; global gate passes.
3. `ruff`, `mypy --strict`, `jaxtyped` checks pass.
4. Docstrings and, where applicable, guide/math docs updated; ADR added if a decision was taken.
5. No new runtime dependency without ADR.
6. PR description references requirement IDs and lists deviations from this document (must be none, or an ADR).

### 6.9 Ticket template

```
ID: E4-T03    Title: Expanded-HMM engine: log-space forward/backward via lax.scan
Requirements: FR-I01, FR-I02, NFR-01, NFR-02
Goal: <one sentence>
Scope in / out: ...
Deliverables: src/regimekit/engine/expanded_hmm.py, tests/golden/test_e4_t03_*.py, docs/math/engine.md
Acceptance criteria: <copied from Section 7, made concrete>
Dependencies: E4-T01, E4-T02
Size: M
```

---

## 7. Implementation plan

### 7.1 Milestones

| Milestone | Content | Exit criteria |
|---|---|---|
| M0 Bootstrap | E0, E1 | `uv sync` + `nox -s lint types tests` green on an empty-but-structured package; config schema validates Appendix A; `ParamManager` and `dists` fully tested |
| M1 Data | E2 | `regimekit fetch` produces a `PreparedDataset` + quality report for `^GDAXI` with VIX/T10Y2Y covariates; offline replay from snapshot |
| M2 Core engine | E3, E4-T01…T05 | GBM + geometric/NB + jump matrix + covariates + order 2 all pass golden tests vs. reference; forward-backward ≤ 100 ms |
| M3 Estimation + explicit engine | E5, E4-T06 | Recovery suite passes for all six scenarios (6.4.4 criterion, weekly full run); explicit engine agrees with expanded engine on NB |
| M4 Inference & forecasting | E6 | `decode` and `forecast` CLI on real data; four covariate forecasters; simulation round-trip tests |
| M5 Evaluation | E7 | Walk-forward backtest with metrics and reference strategies; `compare` table |
| M6 Release 0.1.0 | E8, E9 spikes | Bundle round-trip, docs, tutorial, ADRs, extension spikes with ADR-0008 |

### 7.2 Epics and tickets

Sizes: S ≤ 1 d, M 1–3 d, L 3–5 d. Every ticket includes its tests and docs (Section 6.8).

#### E0 — Project bootstrap

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E0-T01 | Repository skeleton: uv project, src layout, `pyproject.toml` with extras/groups (6.1), `uv.lock`, package `__init__` with version | S | — | D11, NFR-06 | `uv sync --all-groups` succeeds on Linux/WSL2; `python -c "import regimekit"` works; `uv sync --extra gpu` resolves (install not required in CI) |
| E0-T02 | Quality tooling: ruff config, mypy strict, coverage `fail_under=90`, pytest markers (`slow`, `live`, `benchmark`, `gpu`), pre-commit, nox sessions | S | E0-T01 | NFR-04, NFR-05 | `nox -s lint types tests` green; a deliberately failing type annotation is caught in CI |
| E0-T03 | CI workflows: lint/test/docs on PR; nightly recovery smoke + benchmark; weekly full recovery; artifact upload of coverage, benchmark and recovery JSON | S | E0-T02 | NFR-04, NFR-02 | PR pipeline < 10 min; nightly and weekly jobs scheduled; benchmark baseline file committed |
| E0-T04 | Docs skeleton (mkdocs, guide/math/api/adr), ADR-0001…0010 drafted with Status "accepted", CONTRIBUTING with ticket template and DoD | M | E0-T01 | NFR-07 | `mkdocs build --strict` passes; all ADRs present |
| E0-T05 | JAX runtime bootstrap: x64 enabling, device selection (`cpu|gpu|auto`), PRNG utilities (`key_for(seed, *ids)`), error hierarchy, structlog setup | S | E0-T01 | NFR-01, NFR-03, NFR-09 | importing after x64-disabled init raises `NumericalError`; keys are deterministic; logs are JSON in CI |

#### E1 — Contracts, configuration, parameters, distributions

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E1-T01 | Data contracts (5.4): `ObservationSet`, `CovariateSeries`, `PrepStats`, `PreparedDataset` with `validate()` and `window()` | M | E0-T05 | FR-D07, FR-M12 | shape/dtype validation errors are typed; `window()` preserves `dt`, masks, covariates; pytree registration tested |
| E1-T02 | Config schema (5.7) + YAML I/O + `validate-config` + resolved-config dump + `schema_version` migration hook | L | E1-T01 | FR-C01 | Appendix A validates; each rule in 5.7 has a failing test; unknown key fails; resolved YAML round-trips |
| E1-T03 | `ParamSpec`/`ParamManager` with bijectors, modes (`free/fixed/tied/bounded/prior`), ordering constraint, log-det-Jacobian, priors (normal, lognormal, dirichlet) | L | E0-T05 | FR-M03, FR-M05, FR-M11 | round-trip property tests; Jacobian vs autodiff ≤ 1e-8; tied params share one `u` entry; `fixed` absent from `u` |
| E1-T04 | `dists`: log-pdf/pmf + samplers for normal, student-t, poisson, negbin, discretized gamma/weibull/lognormal, nonparametric; NB mean/variance helpers | M | E0-T05 | FR-M06 | agreement with `scipy.stats` ≤ 1e-10; pmfs sum to 1 over `1..D_max`; samplers' moments within 3 SE |
| E1-T05 | Model-side contracts: `ModelSpec`, `StateIndex`, `Posterior`, `PosteriorResult`, `FitResult`, `ForecastResult`, `BacktestResult` (5.4) | S | E1-T03 | FR-I01…I04 | dataclasses frozen, pytree-registered where needed; collapse helpers unit-tested |

#### E2 — Data layer

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E2-T01 | `DataProvider` protocol, registry, `csv` adapter, provider contract test-suite (shared for all adapters) | M | E1-T01 | FR-D01 | contract suite parameterized over adapters; csv adapter passes; registry rejects duplicate names |
| E2-T02 | `yfinance` adapter (prices; intervals; adjusted; curl_cffi session; retry/backoff; intraday window capping; ToS log line) | M | E2-T01 | FR-D01, FR-D02, NFR-08 | recorded-response tests for `^GDAXI` daily and hourly; live test (`-m live`) fetches 30 days; window capping warns |
| E2-T03 | `stooq` adapter with API key + quota-aware throttling | S | E2-T01 | FR-D01 | recorded tests; missing key → `ProviderError` with instructions |
| E2-T04 | `fred` + `cboe` adapters (series IDs from config; `"."` → NaN; 120 req/min limiter) | M | E2-T01 | FR-D01 | recorded tests for VIXCLS, T10Y2Y, VIX_History.csv |
| E2-T05 | Parquet cache + snapshot IDs + `--offline` replay + `max_age` | M | E2-T01 | FR-D03 | second fetch hits cache (no network, asserted via mock); snapshot ID stable across runs; offline replay reproduces identical `PreparedDataset` |
| E2-T06 | Calendar & alignment: master grid, clocks, `Δ_t`, `merge_asof` alignment, lag ≥ 1 enforcement, staleness | M | E1-T01 | FR-D04, FR-D05 | trading vs calendar `Δ` tested on a known holiday span; monthly covariate becomes step function with correct lag; lag 0 → `ConfigError` |
| E2-T07 | `Preparer` fit/transform: log-returns, transforms, standardization on train window, missing policies, `PrepStats` persistence | M | E2-T06 | FR-D05 | transform on test window uses train stats; policies `ffill/drop/error` tested; `PrepStats` JSON round-trip |
| E2-T08 | `fetch` command + data-quality report | S | E2-T05, E2-T07 | FR-D06, FR-C02 | report lists gaps/outliers on a fixture with injected defects; exit code 3 on `error` policy |
| E2-T09 | `tiingo` adapter (optional extra) | S | E2-T01 | FR-D01 | recorded tests; client-side limit enforcement |

#### E3 — Families and structure

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E3-T01 | `EmissionFamily` protocol + GBM (Δ-scaled Gaussian) with drift/log-vol covariate links, carry hooks (identity) | M | E1-T03, E1-T04 | FR-M02, FR-M09 | log-density vs reference ≤ 1e-12; annualization checked at daily/weekly Δ; covariate links tested with zero/non-zero coefficients |
| E3-T02 | Student-t emission family (ν free/tied/fixed) | S | E3-T01 | FR-M02 | matches scipy; `ν → ∞` approaches GBM within 1e-6 |
| E3-T03 | `DurationFamily` protocol + geometric + negbin (phase representation, `advance_prob` with irregular `Δ_t`, log-rate covariate link) | M | E1-T03, E1-T04 | FR-M06, FR-M07, FR-M09 | `E[d] = r/a` verified by simulation; `r=1` equals geometric; irregular `Δ_t` handled |
| E3-T04 | Explicit duration families: poisson, gamma, weibull, lognormal, nonparametric (`log_pmf`, entry-time covariate scale link) | M | E3-T03 | FR-M06, FR-M07 | pmfs sum to 1; means match analytic within truncation error; `D_max` warning rule |
| E3-T05 | `TransitionModel`: jump matrix with reference-category logits, multinomial-logit covariate link, generator mode (`expm`) for HMM | M | E1-T03 | FR-M08, FR-M09 | rows sum to 1 with zero diagonal; no-covariate case constant; `expm(QΔ)` and geometric `A_t` differ by ≤ `(λ_max Δ)²` per entry and converge as `Δ → 0`; Appendix E weekly numbers reproduced |
| E3-T06 | Lattice generator + ordering constraints + label-order post-sorting | S | E1-T03, E3-T01 | FR-M04, FR-M05 | 3×2 grid yields 6 fixed states; increasing σ enforced through bijector |

#### E4 — State space and engines

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E4-T01 | NumPy reference implementations (`tests/reference/`): HMM FB, explicit-duration FB, Viterbi (both), NB/geometric expansion by brute force | M | E1-T04 | NFR-04 | readable, < 300 lines each; self-tests on hand-computed 2-state/3-step examples |
| E4-T02 | `StateIndex` + `build_expanded(spec)`: phases × memory (order k), `regime_of/phase_of/memory_of`, collapse `C`, per-step `A_t` builder (5.4, 4.7.2) | L | E1-T05, E3-T03, E3-T05 | FR-M10, FR-X01 | `S` formula verified; `A_t` rows sum to 1; order-2 admissible tuples exclude `i=j`; structural zeros preserved |
| E4-T03 | `ExpandedHMMEngine`: log-space forward/backward via `lax.scan`, masked observations, initial distribution modes, `loglik` and `Posterior` | L | E4-T01, E4-T02, E3-T01 | FR-I01, FR-I02, NFR-01, NFR-02 | golden tests vs reference (Appendix D); property tests 6.4.2; ≤ 100 ms jitted at NFR-02 size; gradient check 6.4.3 |
| E4-T04 | Augmented posterior utilities: phase/elapsed marginals, regime collapse, expected remaining/elapsed duration | S | E4-T03 | FR-I02 | marginals sum to 1; expected elapsed duration matches simulation on synthetic data |
| E4-T05 | Viterbi (expanded) + segmental output; AIC/BIC/ICL | S | E4-T03 | FR-I01, FR-I03 | Viterbi path log-prob ≥ any sampled path; segments RLE correct; ICL ≥ BIC |
| E4-T06 | `ExplicitDurationEngine` (4.7.3) with right-censoring, `D_max`, chunked scan for memory bound, segmental Viterbi | L | E4-T01, E3-T04 | FR-M06, FR-I01, FR-I02 | golden tests vs reference; equivalence with expanded engine on shifted-NB (6.4.5); ≤ 2 s at NFR-02 size; irregular grid warning |
| E4-T07 | Benchmarks (`pytest-benchmark`), GPU test markers, optional `associative_scan` smoothing behind a flag | M | E4-T03, E4-T06 | NFR-02, NFR-06 | baseline JSON committed; GPU tests skipped cleanly when absent; `associative_scan` result equals `scan` within 1e-10 |

#### E5 — Estimation

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E5-T01 | Objective: NLL + priors + log-det-J on `u`; `value_and_grad` jitted; non-finite guards | M | E4-T03, E1-T03 | FR-M11, NFR-01 | gradient check 6.4.3 for all family/engine combos; `use_priors=false` equals NLL |
| E5-T02 | Optimizers: `optax.lbfgs` (zoom line search) and `adam`; convergence criteria; per-restart result record | M | E5-T01 | D3 | converges on a 2-state synthetic problem within 200 iterations; non-finite step aborts restart only |
| E5-T03 | Initialization strategies (`rollvol_kmeans`, `lattice_jitter`, `random`) + multi-start orchestration with per-restart keys | M | E5-T02 | NFR-03 | strategies produce valid `u`; best-of-restarts selection; determinism test 6.4.6 |
| E5-T04 | Fit report and diagnostics (4.10): occupancy warning, restart dispersion, ordering check, optional Hessian std errors, PIT/Ljung-Box hooks | M | E5-T03, E4-T05 | FR-I03 | each diagnostic has a triggering synthetic case; std errors within 20 % of simulation-based SE on a large synthetic sample |
| E5-T05 | Parameter-recovery suite (6.4.4) + `regimekit selftest` | L | E5-T03, E6-T01 | NFR-04, §1.4 criterion 1 | six scenarios meet the 6.4.4 criterion (every tolerance category in ≥ 4/5 seeds) in the weekly full run; nightly smoke run; `selftest` runs a 1-scenario smoke version in < 2 min |
| E5-T06 | (v1.x) EM for GBM without covariates using `sufficient_stats` | M | E5-T03 | D3 | monotone log-likelihood; agrees with MAP (no priors) within 1e-4 relative |

#### E6 — Inference, simulation, forecasting

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E6-T01 | Simulator: regime/phase paths and returns from any model given `Δ_t` and covariate paths; segment bookkeeping | M | E4-T02, E3-T01 | FR-I06 | empirical sojourn moments match analytic (property test); deterministic per key |
| E6-T02 | `decode` API and CLI: filtered/smoothed/viterbi/segments, augmented posterior export, `posterior.parquet` | S | E4-T05, E1-T02 | FR-I01, FR-I02, FR-C02, FR-P02 | CLI on synthetic CSV produces parquet with `T×N` probabilities; exit codes |
| E6-T03 | Covariate forecasters `freeze`, `scenario`, `bootstrap` (stationary block), `ar` (VAR(1)/AR(1)) | M | E1-T01 | FR-I05 | `h=1` bypass; shapes `(n_sim, h, K)`; bootstrap preserves cross-correlation within tolerance; VAR(1) recovers synthetic coefficients |
| E6-T04 | `MonteCarloForecaster` (4.9): sampling from filtered augmented posterior, path simulation, aggregation, quantiles, future grid dates | L | E6-T01, E6-T03, E4-T04 | FR-I04 | for `h=1` analytic mixture mean/variance matched within MC error; regime_probs rows sum to 1; determinism |
| E6-T05 | `forecast` and `simulate` CLI + result writers (`forecast.parquet`) | S | E6-T04 | FR-C02, FR-P02 | CLI round-trip on synthetic data; long-format parquet schema documented |

#### E7 — Evaluation and backtesting

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E7-T01 | Metric registry + metrics: log score, CRPS (sample-based), PIT coverage, RMSE/MAE, IC wrappers, regime occupancy/persistence, balanced accuracy (labels) | M | E6-T04 | FR-E01 | each metric tested on closed-form cases (e.g. Gaussian CRPS); registry resolves config names; unknown metric → `ConfigError` |
| E7-T02 | Walk-forward backtester: expanding/rolling, refit cadence, embargo, per-fold `Preparer.fit` on train only, per-horizon forecasts, parallel folds | L | E5-T03, E6-T04, E7-T01 | FR-E02 | no look-ahead (test injects a future-only signal and asserts no gain); fold table correct; metrics table `fold × horizon × metric` |
| E7-T03 | Strategy protocol + `regime_long_flat`, `regime_target_vol`, transaction costs, strategy metrics (Sharpe, MDD, turnover, hit rate) | M | E7-T02 | FR-E03 | equity curve reproducible; costs reduce return by expected amount on a synthetic path |
| E7-T04 | `backtest` and `compare` CLI; comparison table across configs on one snapshot; `report.md` | M | E7-T03 | FR-E04, FR-C02 | compare refuses differing snapshot IDs unless `--allow-mixed`; markdown report generated |

#### E8 — Persistence, API, documentation, release

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E8-T01 | `ModelStore` bundle (5.8): safetensors + JSON, manifest, version checks, load/save round-trip | M | E5-T03 | FR-P01 | bitwise `loglik` equality after round-trip; loading a newer bundle version fails with a clear error |
| E8-T02 | `Project` facade + CLI completion (`fit`, global options, exit codes, logging) | M | E8-T01, E7-T04 | FR-C02, FR-C03 | every CLI command has a `Project` equivalent tested for parity on synthetic data |
| E8-T03 | End-to-end example: `^GDAXI` and `^GSPC` daily, Appendix A config; tutorial notebook executed offline in CI from fixtures | M | E8-T02 | §1.4 criterion 2 | notebook runs in CI (`--offline`); results directory contains all artifacts of 5.8 |
| E8-T04 | Documentation completion (guide, math pages, config reference from schema, provider ToS page), changelog, release 0.1.0 | M | E8-T03 | NFR-07, NFR-10 | `mkdocs build --strict`; tag `v0.1.0` published with `uv build` artifacts |

#### E9 — Extension readiness (spikes; after 0.1.0 or in parallel with E8)

| ID | Title | Size | Deps | Requirements | Acceptance criteria |
|---|---|---|---|---|---|
| E9-T01 | `ProductEmission` for `M > 1` with independent per-series GBM/Student-t; data layer for multiple primaries | M | E3-T01, E2-T06 | FR-M12 | two-series synthetic recovery; `M=1` path unchanged (regression tests) |
| E9-T02 | Factorial/coupled builder design spike: tuple `StateIndex`, Kronecker `A_t`, `ApproxInference` protocol; ADR-0008 finalized with complexity analysis and chosen algorithm (structured mean-field or blocked Gibbs) | M | E4-T02 | FR-X01 | ADR accepted; prototype builder passes exact-inference equality for two independent chains with `N ≤ 3` |
| E9-T03 | MS-GARCH emission spike using carry hooks (Haas et al. 2004 parallel recursions) | M | E3-T01, E4-T03 | 5.10 | carry threading works in scan; likelihood of a 1-state GARCH matches `arch` package within 1e-6 |
| E9-T04 | Endogenous covariates design spike (covariates as emission channels) | S | E9-T01 | FR-X02 | ADR with interface diff; no code merged to `main` beyond ADR |

### 7.3 Dependency graph (critical path)

```
E0-T01 → E0-T02 → E0-T03
E0-T05 → E1-T03 → E3-T01 → E4-T02 → E4-T03 → E5-T01 → E5-T02 → E5-T03 → E6-T04 → E7-T02 → E7-T04 → E8-T02 → E8-T04
          E1-T04 ↗           E3-T03 ↗   E4-T01 ↗                       E6-T01 ↗   E7-T01 ↗
E1-T01 → E2-T01 → E2-T02/03/04/09 → E2-T05 → E2-T08
       → E2-T06 → E2-T07 ↗
E3-T04 → E4-T06 (parallel to E5)
```

Parallel tracks after M0: (a) data layer E2, (b) families E3, (c) reference implementations E4-T01. The critical path is E1-T03 → E3 → E4-T02/T03 → E5 → E6-T04 → E7-T02.

### 7.4 Traceability (requirement → tickets)

| Requirement | Tickets |
|---|---|
| FR-M01…M05 | E1-T02, E1-T03, E3-T01, E3-T06 |
| FR-M06, M07 | E1-T04, E3-T03, E3-T04, E4-T06 |
| FR-M08, M09 | E3-T01, E3-T03, E3-T05 |
| FR-M10 | E4-T02 |
| FR-M11 | E1-T03, E5-T01 |
| FR-M12, FR-X01, FR-X02 | E1-T01, E9-T01…T04 |
| FR-I01…I03 | E4-T03…T06, E6-T02 |
| FR-I04…I06 | E6-T01, E6-T03, E6-T04 |
| FR-D01…D07 | E2-T01…T09, E1-T01 |
| FR-E01…E04 | E7-T01…T04 |
| FR-C01…C03, FR-P01, P02 | E1-T02, E6-T02, E6-T05, E8-T01, E8-T02 |
| NFR-01…NFR-10 | E0-T02…T05, E4-T03, E4-T07, E5-T03, E5-T05, E8-T04 |

---

## 8. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Too many parameters for the number of regime segments (durations, order 2, covariates) | Overfitting, unstable decoding | Priors on by default; order 1 and no emission covariates as defaults; IC + walk-forward selection; occupancy and restart-dispersion diagnostics; fixed/tied parameter modes |
| Multimodal likelihood / label switching | Wrong optimum, non-reproducible states | Ordering constraints or fixed lattice; multi-start with reported dispersion; k-means init |
| Yahoo throttling or API change | Data fetch failures | Adapter isolation, cache/snapshots, Stooq/Tiingo fallback, offline replay; live tests separated |
| Explicit-duration engine memory `(T, N, D_max)` | OOM on long intraday series | Chunked scan (E4-T06); NB expansion as default; `D_max` validation |
| JAX recompilation across configs/backtest folds | Slow backtests | Static shapes fixed per fit; folds reuse compiled functions when `T` is padded to a bucket size (E7-T02 note) |
| Drift not identifiable from daily data | Meaningless drift estimates | Documented; defaults tie or fix drifts; report SE |
| Covariate leakage | Optimistic backtests | Lag ≥ 1 enforced in the data layer; standardization on train only; look-ahead test in E7-T02 |
| mkdocs-material maintenance mode | Docs tooling stagnation | Low coupling; migration ADR when Zensical stabilizes |
| GPU CUDA wheel resolution with uv | Install friction | GPU behind extra; documented `[tool.uv.sources]`; CPU is the tested default |

---

## 9. Open decisions with proposed defaults

| # | Question | Proposed default (used unless overridden before the affected ticket starts) |
|---|---|---|
| 1 | Package name | `regimekit` |
| 2 | Library license | Apache-2.0 (ADR-0002) |
| 3 | NB phase count `r_i` selection | Config integer per state (default 3); `compare` over `r ∈ {1,2,3,5}` documented as the selection recipe |
| 4 | `D_max` default | `min(500, 10 × expected sojourn in steps)`; validated at config time |
| 5 | Intraday clocks with overnight gaps | `trading` clock with `bars_per_day` from config; `calendar` allowed with a warning |
| 6 | `associative_scan` | Off by default; enabled after E4-T07 benchmark shows gain on GPU |
| 7 | Bayesian sampling | `mcmc` extra exposes `log_joint`; no NumPyro code in core |
| 8 | Backtest fold parallelism | Sequential by default; `joblib` process pool optional (CPU) — ADR if adopted |
| 9 | Data frames in results | pandas; polars evaluated later |

---

## 10. References

- Yu, S.-Z. (2010). Hidden semi-Markov models. *Artificial Intelligence* 174(2), 215–243.
- Guédon, Y. (2003). Estimating hidden semi-Markov chains from discrete sequences. *JCGS* 12(3), 604–639.
- Johnson, M. J. & Willsky, A. S. (2013). Bayesian nonparametric hidden semi-Markov models. *JMLR* 14, 673–701.
- Dewar, M., Wiggins, C. & Wood, F. (2012). Inference in hidden Markov models with explicit state duration distributions. *IEEE SPL* 19(4).
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica* 57(2).
- Diebold, F. X., Lee, J.-H. & Weinbach, G. C. (1994). Regime switching with time-varying transition probabilities. In *Nonstationary Time Series Analysis and Cointegration*.
- Hughes, J. P. & Guttorp, P. (1994). A class of stochastic models for relating synoptic atmospheric patterns to regional hydrologic phenomena. *Water Resources Research* 30(5).
- Filardo, A. J. (1994). Business-cycle phases and their transitional dynamics. *JBES* 12(3).
- Ang, A. & Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics* 4.
- Bulla, J. & Bulla, I. (2006). Stylized facts of financial time series and hidden semi-Markov models. *CSDA* 51(4).
- Nystrup, P., Madsen, H. & Lindström, E. (2015). Stylised facts of financial time series and hidden Markov models in continuous time. *Quantitative Finance* 15(9); Nystrup et al. (2017–2020) on regime-based asset allocation.
- Haas, M., Mittnik, S. & Paolella, M. S. (2004). A new approach to Markov-switching GARCH models. *J. Financial Econometrics* 2(4).
- Ardia, D. et al. (2019). Markov-switching GARCH models in R: the MSGARCH package. *JSS* 91(4).
- Ghahramani, Z. & Jordan, M. I. (1997). Factorial hidden Markov models. *Machine Learning* 29.
- Brand, M., Oliver, N. & Pentland, A. (1997). Coupled hidden Markov models for complex action recognition. *CVPR*.
- Israel, R. B., Rosenthal, J. S. & Wei, J. Z. (2001). Finding generators for Markov chains via empirical transition matrices. *Mathematical Finance* 11(2).
- Wonham, W. M. (1965). Some applications of stochastic differential equations to optimal nonlinear filtering. *SIAM J. Control* 2(3).
- Frühwirth-Schnatter, S. (2006). *Finite Mixture and Markov Switching Models*. Springer.
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477).
- Linderman, S. et al. (2025). Dynamax: a Python package for probabilistic state space modeling with JAX. *JOSS* 10(108) — reference for scan-based HMM inference.

---

## Appendix A — Reference configuration (validates against Section 5.7)

```yaml
schema_version: 1
project:
  name: dax_3state_gbm_nb
  seed: 42
  output_dir: runs/dax_3state_gbm_nb
  log_level: INFO

data:
  primary:
    provider: yfinance
    symbols: ["^GDAXI"]
    interval: 1d
    start: "2000-01-03"
    end: null
    adjusted: true
    field: close
  clock: trading
  bars_per_day: 1
  cache:
    dir: ~/.cache/regimekit
    snapshot: latest
    max_age_days: 1
  covariates:
    - name: log_vix
      provider: fred
      series: VIXCLS
      transform: [log]
      lag: 1
      missing: ffill
      max_staleness: 5
      use_in: [transition, duration]
    - name: term_spread
      provider: fred
      series: T10Y2Y
      transform: []
      lag: 1
      missing: ffill
      max_staleness: 5
      use_in: [transition]
  preparation:
    returns: log
    standardize_covariates: true
    missing_returns: drop

model:
  n_states: 3
  engine: expanded
  d_max: 500
  initial_state: uniform
  emission:
    family: gbm
    params:
      mu:    {mode: fixed, values: [-0.30, 0.00, 0.20]}
      sigma: {mode: free, init: [0.12, 0.18, 0.35], constraint: increasing,
              prior: {type: lognormal, sd: 1.0}}
    covariates: {drift: [], log_vol: []}
  duration:
    family: negbin
    r: [3, 3, 2]
    params:
      rate: {mode: free, init_mean_sojourn_days: [60, 120, 30],
             prior: {type: lognormal, sd: 1.0}}
    covariates: [log_vix]
  transition:
    mode: jump
    order: 1
    duration_conditioning: current
    covariates: [log_vix, term_spread]
    prior: {type: dirichlet, alpha: 1.0}
    coef_prior_sd: 1.0

estimation:
  optimizer: lbfgs
  max_iter: 1000
  tol_f: 1.0e-9
  tol_g: 1.0e-6
  restarts: 8
  init: [rollvol_kmeans, lattice_jitter, random]
  use_priors: true
  std_errors: true
  device: auto

forecast:
  horizon: 20
  n_sim: 10000
  covariate_forecaster: {type: bootstrap, mean_block_length: 10}
  quantiles: [0.05, 0.25, 0.50, 0.75, 0.95]

evaluation:
  metrics: [log_score, crps, pit_coverage, bic, icl, regime_occupancy, regime_persistence, sharpe, max_drawdown]
  backtest:
    scheme: expanding
    initial_train_years: 8
    refit_every_steps: 63
    horizons: [1, 5, 20]
    embargo_steps: 0
  strategy:
    type: regime_long_flat
    flat_if: {regime: 0, prob_above: 0.6, use: filtered}
    cost_bps: 5

output:
  write: [posterior, forecast, backtest, strategy, report]
```

## Appendix B — Free-parameter counts (no covariates; `p_e` emission params per state, `p_d` duration params per state)

| Component | Order 1 | Order 2 |
|---|---|---|
| Jump matrix | `N (N−2)` | `N (N−1)(N−2)` |
| Durations (`current`) | `N p_d` | `N p_d` |
| Durations (`pair`) | — | `N (N−1) p_d` |
| Emissions | `N p_e` | `N p_e` |
| Expanded states `S` | `Σ r_i` | `(N−1) Σ r_i` |

Example `N=4`, GBM (`p_e=2`), NB (`p_d=1`): order 1 → 8 + 4 + 8 = 20; order 2 → 24 + 4 + 8 = 36. Each transition covariate adds `N (N−2)` (order 1) or `N (N−1)(N−2)` (order 2) coefficients; each duration covariate adds `N`; each emission covariate adds `2N`.

## Appendix C — Engine pseudo-code (expanded HMM, log space)

```
inputs: log_emis (T, N) per regime, C (S, N), build_A(θ, z_pi_t, z_d_t, Δ_t) → (S, S), log_rho (S,)
ℓ_t(s) = log_emis[t, regime_of[s]]                           # gather via C
α_1 = log_rho + ℓ_1
scan over t = 2..T:
    logA = log(build_A(θ, z_pi_t, z_d_t, Δ_t))               # −inf on structural zeros
    α_t = ℓ_t + logsumexp(α_{t−1}[:, None] + logA, axis=0)
loglik = logsumexp(α_T)
β_T = 0
reverse scan over t = T−1..1:
    logA_next = log(build_A(θ, z_pi_{t+1}, z_d_{t+1}, Δ_{t+1}))
    β_t = logsumexp(logA_next + (ℓ_{t+1} + β_{t+1})[None, :], axis=1)
γ_t = softmax(α_t + β_t); regime marginals = γ_t @ C
```

## Appendix D — Numerical tolerances

| Check | Tolerance |
|---|---|
| JAX engine vs NumPy reference, log-likelihood | relative ≤ 1e-8 |
| JAX engine vs NumPy reference, posteriors | max-abs ≤ 1e-8 |
| Posterior row normalization | ≤ 1e-10 |
| Expanded vs explicit engine on shifted-NB (`D_max` mass loss < 1e-10) | loglik relative ≤ 1e-6; smoothed max-abs ≤ 1e-6 |
| Gradient vs central finite differences (step 1e-6) | relative ≤ 1e-5 |
| Bijector log-det-Jacobian vs autodiff | ≤ 1e-8 |
| Distribution log-pdf/pmf vs scipy | ≤ 1e-10 |
| Parameter recovery (`T=20000`, `order 2`: `T=50000`; `N=3`, GBM, NB `r=3`, 5 seeds, each tolerance category in ≥ 4 seeds; explicit-gamma mean sojourn reported only) | σ: ≤ 10 % rel; μ: ≤ 2 SE; mean sojourn: ≤ 25 % rel; π: ≤ 0.15 abs; coefficients: sign-correct, ≤ 2 SE |
| Simulation moments vs analytic | within 3 SE |
| Determinism (same config/snapshot/seed/backend) | bitwise |
| Bundle round-trip | bitwise `u`, `loglik` |

## Appendix E — Worked numbers for the default (from the brainstorm)

Two-regime sanity example used in docs and as a golden fixture: calm `(μ, σ) = (0.08, 0.12)`, turbulent `(−0.10, 0.35)`, rates calm→turbulent 2/yr, turbulent→calm 6/yr. Daily `Δ = 1/252`: `a_calm = 0.0079`, `a_turb = 0.0235`; return mean/sd calm `0.029 % / 0.756 %`, turbulent `−0.064 % / 2.205 %`. Weekly `Δ = 1/52`: `expm(QΔ)` off-diagonals `0.0357 / 0.1070` versus first-order `0.0385 / 0.1154` — the golden test for generator mode.
