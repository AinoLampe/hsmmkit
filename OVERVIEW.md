# regimekit: model overview and usage guide

A broad tour of what regimekit models, where it can be extended, and how to run the main workflow end to
end. It is a map, not a reference. The detailed material lives in `docs/guide/` (usage), `docs/math/`
(formulas), `docs/adr/` (design decisions) and the generated configuration reference
`docs/guide/config.md`.

---

## Part 1: The model and its extension scope

### 1.1 What it does

regimekit fits a **Hidden Semi-Markov Model (HSMM)** to a return series (for example a stock index).
The market is assumed to sit in one of `N` hidden *regimes* (calm, turbulent, crash, ...). Each regime has its
own return distribution, and each stay in a regime has a random *duration*. From the fitted model you get:

- **Detection:** the probability of each regime at each date (filtered, using past data only; smoothed, using
  all data), the single most likely regime path (Viterbi), regime segments, and AIC/BIC/ICL.
- **Forecasting:** Monte-Carlo paths from the end of the data, giving regime probabilities, return quantiles,
  mean and variance for every horizon.
- **Evaluation:** walk-forward backtests with proper scoring rules and optional regime-driven strategies.

### 1.2 The generative model

Three components, each swappable and each able to take covariates:

```text
           transition (who comes next?)
 regime i ─────────────────────────────▶ regime j        jump matrix, rows sum to 1, no self-jumps
    │
    │ duration (how long do we stay?)
    ▼
 sojourn of D_t steps                                    distribution depends on regime i
    │
    │ emission (what do we observe?)
    ▼
 return y_t ~ f_i( · | covariates, Δ_t)                  one density per regime
```

| Component | Question it answers | Shipped families | Covariates enter via |
|---|---|---|---|
| **Emission** `f_i` | distribution of the return given the regime | `gbm` (Gaussian log-returns with drift `mu` and volatility `sigma`), `student_t` | drift and log-volatility links |
| **Duration** | how long a regime lasts | `geometric`, `negbin` (phase-expanded), `poisson`, `gamma`, `weibull`, `lognormal`, `nonparametric` (explicit) | log-rate link (and the entry-time option) |
| **Transition** | which regime follows | `jump` matrix (softmax-logit link when covariates are used), `generator` mode; order 1 or 2 | logit coefficients per target |

Three clocks control the time unit (`data.clock`): `trading` (years of 252 days), `calendar` (365.25) and
`steps` (every step is 1, parameters are per step). Parameters are always in **clock units**, never both
annualized and per step.

### 1.3 How inference works

There are two exact engines behind one interface (`InferenceEngine`: `loglik`, `forward`,
`forward_backward`, `viterbi`, `approx`).

**Expanded HMM engine (`model.engine: expanded`, default).** A semi-Markov process with negative-binomial
durations is an ordinary HMM on an *expanded* state `s = (memory, regime, phase)`. A regime `i` with `r_i`
phases becomes `r_i` chain states that advance with probability `a_i(t)` and jump on completion. So

- `S = Σ r_i` states at order 1, `(N−1) Σ r_i` at order 2.
- The forward recursion runs in log space with `A_t` rebuilt inside the `lax.scan` (the `(T, S, S)` tensor is
  never stored):
  `log α_t(j) = logsumexp_i[log α_{t−1}(i) + log A_t(i,j)] + log b_j(y_t)`.
- Covariates make `A_t` time-varying at no extra cost.
- Geometric is `r = 1`. Higher `r` gives the humped duration shape that real regimes show.

**Explicit-duration engine (`model.engine: explicit`).** Keeps the duration pmf on a grid `1..D_max` and
runs a segmental forward-backward and segmental Viterbi. It supports non-phase-type families (gamma, Weibull,
log-normal, nonparametric) and gives exact elapsed-duration posteriors. It is more expensive (`(T, N, D_max)`
memory, chunked when large). It agrees with the expanded engine on negative binomial to `1e-6`
(`tests/golden/`).

**Estimation.** MAP by multi-start L-BFGS (Adam optional) in JAX with exact gradients. Parameters are mapped
to an unconstrained vector by bijectors (positivity, ordering, simplex), with priors plus log-det-Jacobian
terms in the objective. Initialisation uses rolling-volatility k-means, lattice jitter and random starts;
`estimation.restarts` controls how many. The same config, snapshot, seed and backend give bitwise-identical
results. Covariate-free GBM models can also use EM (Baum-Welch).

**Forecasting.** Filter to the end of the data, then simulate regime-and-return paths forward. Covariates are
exogenous, so their future values come from a *covariate forecaster* (`freeze`, `bootstrap`, `ar`,
`scenario`).

### 1.4 Code layout

```text
config/      pydantic v2 schema + YAML I/O          data/        contracts, cache, calendar, Preparer
finance/     providers (yfinance, stooq, fred...)   model/       spec, params, emission/, duration/, transition
statespace/  index + per-step A_t builders          engine/      expanded, explicit, viterbi, criteria
estimation/  objective, optimizers, init, MAP, EM   inference/   decode, simulate, forecast, cov. forecasters
evaluation/  metrics, backtest, strategies          persistence/ ModelStore (.rkmodel), result I/O
project.py   Project facade        cli/app.py   thin typer wrappers over Project
```

Two import rules are enforced by a static test:

- **Layer rule:** `model/`, `statespace/`, `engine/`, `estimation/` and `inference/` import nothing from
  `regimekit.data` except `data.contracts`.
- **Profile rule:** no core package imports `regimekit.finance`. Only `cli/` and `project.py` may.

This is what keeps the core domain-neutral and makes the extensions below cheap.

### 1.5 Extension scope

#### Supported today through configuration only

| You want | Change in YAML |
|---|---|
| More or fewer regimes | `model.n_states` |
| Fat-tailed returns | `model.emission.family: student_t` |
| Humped, covariate-dependent durations | `duration.family: negbin`, `duration.r: [3, 3, 2]`, `duration.covariates: [...]` |
| Non-phase-type durations | `engine: explicit`, `duration.family: gamma / weibull / lognormal / nonparametric`, `d_max` |
| Covariates driving regime switching | `data.covariates[].use_in: [transition]`, `model.transition.covariates: [...]` |
| Covariates driving volatility or drift | `use_in: [emission]`, `model.emission.covariates: {drift: [...], log_vol: [...]}` |
| Covariates driving durations | `use_in: [duration]`, `model.duration.covariates: [...]` |
| Memory of the previous regime (A→B→C differs from C→B→C) | `model.transition.order: 2`, optionally `duration_conditioning: pair` |
| Fix, bound, tie, order or put priors on any parameter | per-parameter `mode: free / fixed / tied / bounded / prior`, `constraint: increasing`, `prior:` |
| Identify regimes by volatility ordering | `sigma: {constraint: increasing}` |
| Multiple assets | several `data.primary.symbols` (`M > 1`; product emission, independent across series given the regime) |
| Different data | any provider registered in `finance/providers` (csv, yfinance, stooq, fred, cboe, tiingo) |
| Different clock or frequency | `data.clock`, `data.primary.interval`, `bars_per_day` |
| Strategy overlays | `evaluation.strategy` (`regime_long_flat`, `regime_target_vol`, with costs) |

Covariates are always **lagged** (`lag >= 1`) so there is no look-ahead. They are standardised on the
training window only, and a staleness policy (`missing`, `max_staleness`) is applied.

#### Extension by writing code (stable protocols)

Everything a family does is behind a protocol, and all parameters go through `ParamSpec` / `ParamManager`, so
bijectors, priors, fixing and tying work for new components automatically.

| To add | Implement | Register |
|---|---|---|
| Emission family | `EmissionFamily` (`model/emission/base.py`): `param_specs`, `log_prob`, `cdf`, `sample`, plus `support` and `scale_by_dt` | `EMISSION_FAMILIES` in `model/build.py`, with a `from_config` classmethod |
| Duration family | `DurationFamily` (`model/duration/base.py`): `param_specs`, `n_phases`, `advance_prob`, `log_pmf`, `mean_sojourn`, `init_from_mean_sojourn`, `supports_explicit` | `DURATION_FAMILIES` |
| Transition structure | `TransitionModel` (`model/transition.py`) and the `MemoryLayout` row enumeration | |
| Covariate forecaster | the forecaster protocol in `inference/covariate_forecasters.py` | `forecast.covariate_forecaster.type` |
| Metric | a function of a context with declared `needs` | `evaluation/metrics/registry.py` |
| Data provider | `fetch_prices` / `fetch_series` returning contract types | provider registry |

The cache and the preparer depend on a structural `CacheableProvider` protocol, so a new provider does not
touch the core.

#### Experimental, with a design and an ADR but not wired into the config or CLI

| Direction | State | See |
|---|---|---|
| **Recurrent emissions, Markov-switching GARCH** | `model/emission/msgarch.py` works through the `init_carry` / `step_carry` hook. It has a single series, no `sample`/`cdf`, and is not registered | ADR-0022 |
| **Several series with different families** (`ProductEmission`) | implemented for `M > 1`, independent given the regime | `model/emission/product.py` |
| **Factorial / coupled regime chains** | composition by Kronecker product reproduces single-chain results exactly. Approximate inference (structured mean-field) is a reserved plug-in (`InferenceEngine.approx()` returns `None` for exact engines) | ADR-0008 |
| **Endogenous covariates** (modelling the VIX jointly instead of forecasting it separately) | design only | ADR-0023 |

#### Deliberately out of scope for 0.1.x

No batch dimension `B`, no `profile: core` execution path, no extras beyond `gpu` and `providers`. The
`profile` config key already exists (default `finance`) so the domain-neutral core can be generalised later.
Open limitations are listed in `docs/releases/0.1.0.md` (for example the first forecast step uses the last
observed covariate row, and only a weekday calendar is used for forecast dates).

---

## Part 2: How to use it

### 2.0 Setup

```bash
uv sync --all-groups              # development environment
uv sync --extra providers         # yfinance, fredapi, curl_cffi for live data
export FRED_API_KEY=...           # only for FRED covariates (VIXCLS, T10Y2Y); also STOOQ_API_KEY, TIINGO_API_KEY
```

There are two equivalent entry points. The CLI is a thin wrapper over the Python `Project` class, which is
the same with `write=True`. In Python nothing is written unless you ask.

```python
from regimekit import Project
p = Project.from_yaml("examples/configs/dax_3state_gbm_nb.yaml", overrides={"project.seed": 7})
```

Global CLI options come **before** the command: `--out DIR`, `--offline`, `--seed N`, `--device cpu|gpu|auto`,
`--log-level LEVEL`. Exit codes: 0 ok, 2 config, 3 data, 4 numerical, 5 provider, 1 anything else.

### 2.1 Describe the model in YAML

A single file drives everything. Start from `examples/configs/dax_3state_gbm_nb.yaml`. The skeleton:

```yaml
project:    {name: my_run, seed: 42, output_dir: runs/my_run}
data:
  primary:  {provider: yfinance, symbols: ["^GDAXI"], interval: 1d, start: "2005-01-03"}
  clock: trading
  covariates:
    - {name: log_vix, provider: fred, series: VIXCLS, transform: [log], lag: 1, use_in: [transition, duration]}
model:
  n_states: 3
  emission:   {family: gbm, params: {mu: {mode: fixed, values: [-0.3, 0, 0.2]},
                                     sigma: {mode: free, init: [0.12, 0.18, 0.35], constraint: increasing}}}
  duration:   {family: negbin, r: [3, 3, 2], covariates: [log_vix],
               params: {rate: {mode: free, init_mean_sojourn_days: [60, 120, 30]}}}
  transition: {mode: jump, order: 1, covariates: [log_vix]}
estimation:  {restarts: 8, init: [rollvol_kmeans, lattice_jitter, random]}
forecast:    {horizon: 20, n_sim: 10000, covariate_forecaster: {type: bootstrap, mean_block_length: 10},
              quantiles: [0.05, 0.5, 0.95]}
evaluation:  {metrics: [log_score, crps, bic], backtest: {scheme: expanding, initial_train_years: 8,
              refit_every_steps: 63, horizons: [1, 5, 20]}}
output:      {write: [posterior, forecast, backtest, report]}
```

Always validate first. It reports errors with the offending config path and writes the fully resolved config:

```bash
regimekit validate-config config.yaml
regimekit config-schema schema.json        # JSON-Schema for editor autocompletion
```

### 2.2 Fetch data

```bash
regimekit fetch config.yaml
```

Pipeline: provider → parquet cache with content-hash **snapshot ids** → calendar alignment → covariate lagging,
transforms, standardisation → `PreparedDataset` (observations, `dt`, lagged covariates for each entry point).

Writes `prepared.parquet`, `prep_stats.json`, `snapshots.json` and a data-quality report (`quality.json/md`).
The cache lives in `~/.cache/regimekit` by default.

- Re-fetching reuses the cache while it is younger than `data.cache.max_age_days`.
- `--offline` replays the cached snapshot exactly, which gives reproducible runs.
- `data.cache.snapshot: <id>` pins an exact data generation. It cannot be combined with covariates in 0.1.0.

```python
data = p.fetch()                  # PreparedDataset; quality report on p.last_quality
```

### 2.3 Train a model, with different features

```bash
regimekit fit config.yaml         # fetch + fit -> <name>.rkmodel/, fit_report.json, config.resolved.yaml
```

```python
fit = p.fit(data)                 # FitResult: params, loglik, converged, restarts table, diagnostics
fit.params; fit.loglik; fit.restarts
```

Different features are different YAML. The easiest way to try variants is `overrides`, with no file edits:

```python
# plain 3-state GBM + geometric durations, no covariates
base = {"model.duration.family": "geometric", "model.transition.covariates": [], "model.duration.covariates": []}
# fat tails
p_t = Project.from_yaml(cfg, overrides={"model.emission.family": "student_t"})
# explicit-duration engine with gamma durations
p_g = Project.from_yaml(cfg, overrides={"model.engine": "explicit", "model.duration.family": "gamma",
                                        "model.d_max": 400})
# second-order memory
p_o2 = Project.from_yaml(cfg, overrides={"model.transition.order": 2})
```

(A variant that changes a family needs matching `params` blocks, since the parameter names differ per
family. Copy the YAML for anything beyond a one-key change.)

Things to know:

- `estimation.restarts` trades time for robustness. Start low (2) while exploring.
- `estimation.std_errors: true` adds standard errors.
- `fit_report.json` lists every restart (objective, converged, iterations), so check that the best restarts
  agree before trusting a fit. The diagnostics also include mean sojourn per regime.
- Fits are deterministic for a given seed and backend.
- If you doubt your backend, `regimekit selftest` (`--full` for all six scenarios) fits simulated data twice
  and exits 4 unless the fits are finite and reproducible.

### 2.4 Backtest

```bash
regimekit backtest config.yaml    # -> backtest_metrics/folds/summary.parquet, strategy.parquet, report.md
```

```python
bt = p.backtest()                 # fetches raw inputs itself; the preparer is refit per fold
```

Walk-forward scheme (`evaluation.backtest`): fit on an expanding (or rolling) window, forecast from every
origin in the next `refit_every_steps` rows, then roll forward.

- Standardisation statistics, the fit, and the covariates all use **training data only**, per fold.
- Each fold warm-starts from the previous optimum.
- `embargo_steps` leaves a gap between training and test.
- Predictive metrics (`log_score`, `crps`, `pit_coverage`, `rmse`, `mae`) are evaluated per horizon on the
  *cumulative* return. In-sample criteria (`aic/bic/icl`), regime occupancy and persistence, and strategy
  metrics (`sharpe`, `max_drawdown`, `turnover`, `hit_rate`) are reported with `horizon == 0`.
- `evaluation.strategy` adds a regime-conditioned position (for example flat when the filtered probability
  of the crash regime exceeds 0.6) with transaction costs.

Compare model variants on the same data snapshot:

```bash
regimekit compare geometric.yaml negbin.yaml student_t.yaml           # -> compare.parquet, compare.md
regimekit compare a.yaml b.yaml --metric crps --metric bic --reuse     # reuse stored backtests
```

`compare` refuses runs on different snapshots (exit 3) unless `--allow-mixed`.

### 2.5 Save a trained model

`fit` already writes the bundle when run from the CLI. From Python:

```python
p.save(fit, "runs/my_run/my_run.rkmodel")      # needs the dataset it was fitted on (defaults to the last one)
```

A bundle is a directory:

| File | Holds |
|---|---|
| `manifest.json` | bundle/regimekit/JAX versions, backend, seed, profile, schema version |
| `config.resolved.yaml` | the full resolved configuration |
| `spec.json` | model structure, data-derived build inputs (series count, covariate names, clock, `nominal_dt`) |
| `params.safetensors` | constrained `theta` and unconstrained `u`, float64 |
| `prep_stats.json`, `snapshots.json` | standardisation statistics, data snapshot ids |
| `fit_report.json` | objective, loglik, restarts, diagnostics, standard errors |

### 2.6 Load a model, detect regimes, predict

```bash
regimekit --offline decode   config.yaml --model runs/my_run/my_run.rkmodel [--augmented]
regimekit --offline forecast config.yaml --model runs/my_run/my_run.rkmodel --horizon 20 --n-sim 10000 [--samples]
```

```python
from regimekit import Project

fit = Project.load("runs/my_run/my_run.rkmodel")       # FitResult, no data needed
p = Project.from_yaml("runs/my_run/my_run.rkmodel/config.resolved.yaml")   # same config the model was fit with
data = p.fetch(offline=True)                           # prepared data from the cached snapshot

post = p.decode(fit, data)
post.filtered      # (T, N) P(regime_t | y_1..t)   causal, usable live
post.smoothed      # (T, N) P(regime_t | y_1..T)   retrospective
post.viterbi       # (T,)   most likely regime path
post.segments      # [(start, end_inclusive, regime), ...]
post.loglik, post.aic, post.bic, post.icl

fc = p.forecast(fit, data, horizon=20, n_sim=10000)
fc.regime_probs    # (h, N)      regime probabilities at each step ahead
fc.cum_quantiles   # (h, Q, M)   quantiles of the cumulative log-return
fc.mean, fc.var    # (h, M)
fc.samples         # (n_sim, h, M) per-step simulated returns
```

Notes:

- `decode` is the **detection** step. Use `filtered` for anything that must be causal (trading signals,
  backtests) and `smoothed` / `viterbi` for historical analysis.
- `forecast` starts from the filtered state at the last row of `data`. For scenario analysis pass a CSV of
  future covariates: `--scenario path.csv`, in prepared units, and unlisted covariates stay frozen.
- Covariates are exogenous, so a model that uses them needs a `forecast.covariate_forecaster`. Forecasts are
  bitwise reproducible for the same seed.
- `regimekit simulate config.yaml --params model --model DIR --steps 500 --n-paths 100` draws synthetic
  regime-and-return paths from a fitted model (`--params config` uses the configured initial values; not
  available for the explicit engine).
- The bundle stores the data preparation statistics (`prep_stats.json`), so a model is tied to the
  standardisation it was trained with. `p.fetch()` re-estimates standardisation from the config's initial
  training window. To keep decode and forecast on exactly the training scale, reuse the `prepared.parquet`
  written by the original `fetch`, which is what the CLI does.

### 2.7 Typical session at a glance

```bash
regimekit validate-config cfg.yaml            # 1. check the config
regimekit fetch           cfg.yaml            # 2. data + quality report (online once)
regimekit --offline fit   cfg.yaml            # 3. train -> my_run.rkmodel
regimekit --offline decode   cfg.yaml --model runs/my_run/my_run.rkmodel
regimekit --offline forecast cfg.yaml --model runs/my_run/my_run.rkmodel
regimekit --offline backtest cfg.yaml         # 4. out-of-sample evaluation
regimekit --offline compare cfg.yaml variant.yaml
```

Without network access or API keys, `examples/synthetic.py` (`write_offline_project`) builds a synthetic index
and covariates as CSV files. `docs/guide/quickstart.md` and `examples/tutorial.ipynb` run on it.

### 2.8 Where to go next

| Need | Document |
|---|---|
| Every config option | `docs/guide/config.md` |
| Output file schemas | `docs/guide/outputs.md` |
| Provider terms, keys, caching | `docs/guide/providers.md` |
| Backtest metrics and folds | `docs/guide/backtesting.md` |
| Bundle format | `docs/guide/bundle.md` |
| Python API vs CLI parity | `docs/guide/project.md` |
| When something fails | `docs/guide/troubleshooting.md` |
| Formulas | `docs/math/` |
| Why it is built this way | `docs/adr/` |
