# E1-T02a — Configuration schema (pydantic v2 models + validation rules)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| M | FR-C01; Part II FR-G02, FR-G05 | M0 | §5.7, Appendix A; Part II §15.4, §15.7 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendments 2 and 6: `profile` key, `steps` clock, `init_mean_sojourn_steps`). No behavioural change for the finance profile.

## Goal
Model the whole YAML configuration as pydantic v2 classes with `extra="forbid"` and every cross-field validation rule from the concept.

## Context (self-contained)
Top level `ProjectConfig(schema_version: int, profile: Literal["core", "finance"] = "finance", project, data, model, estimation, forecast, evaluation, output)`. `profile` is a Part II hook: in 0.1.0 only `finance` is executable; `core` fails validation (see rules). `schema_version` stays `1` — Part II will introduce `2`.

| Model | Fields |
|---|---|
| `ProjectSection` | `name`, `seed: int`, `output_dir: Path`, `log_level: Literal[DEBUG,INFO,WARNING,ERROR]="INFO"` |
| `DataSection` | `primary: PriceSource`, `clock: Literal["trading","calendar","steps"]="trading"`, `bars_per_day: int=1`, `cache: CacheCfg`, `covariates: list[CovariateCfg]=[]`, `preparation: PrepCfg` |
| `PriceSource` | `provider: str`, `symbols: list[str]`, `interval: Literal["1h","1d","1wk","1mo"]`, `start: str`, `end: str|None`, `adjusted: bool=True`, `field: str="close"` |
| `CacheCfg` | `dir: Path="~/.cache/regimekit"`, `snapshot: str="latest"`, `max_age_days: int=1` |
| `CovariateCfg` | `name`, `provider`, `series`, `transform: list[Literal[log,diff,pct,zscore]]=[]`, `lag: int (≥1)`, `missing: Literal[ffill,drop,error]="ffill"`, `max_staleness: int=5`, `use_in: list[Literal[transition,duration,emission]]` |
| `PrepCfg` | `returns: Literal["log"]="log"`, `standardize_covariates: bool=True`, `missing_returns: Literal[drop,error]="drop"` |
| `ModelSection` | `n_states: int (≥2)`, `engine: Literal[expanded,explicit]="expanded"`, `d_max: int|None`, `initial_state: Literal[uniform,stationary,free]="uniform"`, `emission: EmissionCfg`, `duration: DurationCfg`, `transition: TransitionCfg`, `lattice: LatticeCfg|None` |
| `ParamCfg` | `mode: Literal[free,fixed,tied,bounded,prior]`, `values: list[float]|None`, `init: list[float]|None`, `init_mean_sojourn_days: list[float]|None`, `init_mean_sojourn_steps: list[float]|None`, `bounds: tuple[float,float]|None`, `constraint: Literal[none,increasing]="none"`, `prior: PriorCfg|None`, `tie: str|None` |
| `PriorCfg` | `type: Literal[normal,lognormal,dirichlet]`, `sd: float|None`, `alpha: float|None` |
| `EmissionCfg` | `family: Literal[gbm,student_t]`, `params: dict[str, ParamCfg]`, `covariates: {drift: list[str], log_vol: list[str]}` |
| `DurationCfg` | `family: Literal[geometric,negbin,poisson,gamma,weibull,lognormal,nonparametric]`, `r: list[int]|None`, `params: dict[str, ParamCfg]`, `covariates: list[str]=[]` |
| `TransitionCfg` | `mode: Literal[jump,generator]="jump"`, `order: int (≥1)=1`, `duration_conditioning: Literal[current,pair]="current"`, `covariates: list[str]=[]`, `prior: PriorCfg|None`, `coef_prior_sd: float=1.0` |
| `LatticeCfg` | `grids: dict[str, list[float]]` |
| `EstimationSection` | `optimizer: Literal[lbfgs,adam]="lbfgs"`, `max_iter=1000`, `tol_f=1e-9`, `tol_g=1e-6`, `restarts=8`, `init: list[Literal[rollvol_kmeans,lattice_jitter,random]]`, `use_priors=True`, `std_errors=False`, `device: Literal[cpu,gpu,auto]="auto"` |
| `ForecastSection` | `horizon: int`, `n_sim: int`, `covariate_forecaster: dict` (`type: freeze|scenario|bootstrap|ar` + type-specific keys, validated by a discriminated union), `quantiles: list[float]` |
| `EvaluationSection` | `metrics: list[str]`, `backtest: BacktestCfg`, `strategy: StrategyCfg|None`, `labels: LabelSource|None` |
| `BacktestCfg` | `scheme: Literal[expanding,rolling]`, `initial_train_years: float`, `rolling_years: float|None`, `refit_every_steps: int`, `horizons: list[int]`, `embargo_steps: int=0` |
| `StrategyCfg` | `type: Literal[regime_long_flat,regime_target_vol]`, `flat_if: dict|None`, `target_vol: float|None`, `cost_bps: float=0` |
| `OutputSection` | `write: list[Literal[posterior,forecast,backtest,strategy,report]]` |

Validation rules (all raise `ConfigError(path=...)`): `profile: core` → `ConfigError(path="profile", "profile 'core' is not executable in this release; use 'finance'")`; per-state lists (`values`, `init`, `r`, `init_mean_sojourn_days`, `init_mean_sojourn_steps`) have length `n_states`; at most one of `init_mean_sojourn_days`/`init_mean_sojourn_steps` per `ParamCfg`; under `clock: steps` the `days` variant is rejected and `bars_per_day` must be `1` (path `data.bars_per_day`); under `trading`/`calendar` a `steps` variant is converted with `nominal_dt` at model-build time; every covariate name in `use_in`/`transition.covariates`/`duration.covariates`/`emission.covariates.*` exists in `data.covariates` and lists the matching entry point in its `use_in`; `lag ≥ 1`; `order ≥ 1`, `order > 2` → warning; `engine: explicit` with `duration.family: negbin` and duration covariates → warning "switching to entry-time evaluation" (store a flag `duration_covariates_at_entry=True`); `d_max` default `min(500, 10 × expected sojourn in steps)` where expected sojourn comes from `init_mean_sojourn_days`/`init_mean_sojourn_steps`/`init` (steps clock: steps directly); if `d_max < 2 × max expected sojourn in steps` → warning; `mode: generator` only with `family: geometric` and `r` all 1; `values` required for `fixed`, `bounds` for `bounded`, `tie` for `tied`; `prior` present when `mode: prior`. `quantiles` in (0,1) strictly increasing. Unknown keys anywhere → `ConfigError`.

Warnings are collected on the model (`config.warnings: list[str]`) and also emitted via `warnings.warn`.

## Creates
- `src/regimekit/config/schema.py`, `src/regimekit/config/defaults.py`
- `tests/unit/test_e1_t02a_schema.py`; `tests/fixtures/configs/appendix_a.yaml` (verbatim Appendix A of the concept), `tests/fixtures/configs/invalid/*.yaml`

## Acceptance criteria
1. Appendix A parses into `ProjectConfig` without error and yields exactly the documented warnings (none); the parsed model has `profile == "finance"` and `data.clock == "trading"` from defaults.
5. `profile: core` fails with path `profile`; a fixture `tests/fixtures/configs/steps_clock.yaml` (Appendix A with `clock: steps`, `init_mean_sojourn_steps` instead of `_days`) validates; the same fixture with `_days` fails with the `ParamCfg` path; `bars_per_day: 2` under `steps` fails with path `data.bars_per_day`.
2. Every validation rule above has at least one failing fixture and a test asserting the `ConfigError.path`.
3. An unknown key at any nesting level fails with its path.
4. `ProjectConfig.model_dump()` round-trips through `model_validate` unchanged.

## Tests
Parametrized invalid-config fixtures; round-trip; defaults filled.

## Out of scope
YAML reading/writing, `schema_version` migration, CLI (E1-T02b). Config → `ModelSpec` construction (E3-T*/E4-T02a).

## Connections
- Depends on: E1-T01
- Blocks: E1-T02b
- Relates to: E5-T03 (estimation fields), E6-T03 (forecaster fields), E7-T02a (backtest fields)
