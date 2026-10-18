"""Configuration schema: pydantic v2 models with `extra="forbid"` and cross-field validation.

Rules (concept §5.7, overview §4.1, CLAUDE.md §3.4): see `_validate_cross_fields` for the
numbered rules. `parse_project_config` is the entry point that converts pydantic's own
`ValidationError` (unknown keys, wrong types, ...) into `ConfigError(path=...)`, matching the
"all raise ConfigError" contract; custom cross-field rules raise `ConfigError` directly from
inside `@model_validator` methods (pydantic does not intercept non-`ValueError` exceptions, so
they propagate unwrapped).
"""

from __future__ import annotations

import warnings as warnings_module
from pathlib import Path
from typing import Any, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator

from regimekit.config.defaults import (
    D_MAX_CAP,
    D_MAX_SOJOURN_MULTIPLIER,
    D_MAX_WARN_SOJOURN_MULTIPLIER,
    MAX_ORDER_WARN,
)
from regimekit.utils.clock import CALENDAR_DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR, Clock, days_to_steps
from regimekit.utils.errors import ConfigError

CURRENT_SCHEMA_VERSION = 1

Transform = Literal["log", "diff", "pct", "zscore"]
MissingPolicy = Literal["ffill", "drop", "error"]
EntryPoint = Literal["transition", "duration", "emission"]
ParamMode = Literal["free", "fixed", "tied", "bounded", "prior"]
EmissionFamily = Literal["gbm", "student_t"]
DurationFamily = Literal["geometric", "negbin", "poisson", "gamma", "weibull", "lognormal", "nonparametric"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --- project / data -------------------------------------------------------------------------


class ProjectSection(_Base):
    """`project:` — run identity, seed, output location, logging."""

    name: str
    seed: int
    output_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class PriceSource(_Base):
    """`data.primary:` — the primary price series to fetch."""

    provider: str
    symbols: list[str]
    interval: Literal["1h", "1d", "1wk", "1mo"]
    start: str
    end: str | None = None
    adjusted: bool = True
    field: str = "close"


class CacheCfg(_Base):
    """`data.cache:` — local parquet cache settings."""

    dir: Path = Path("~/.cache/regimekit")
    snapshot: str = "latest"
    max_age_days: int = 1


class CovariateCfg(_Base):
    """`data.covariates[]:` — one covariate series and how it enters the model."""

    name: str
    provider: str
    series: str
    transform: list[Transform] = Field(default_factory=list)
    lag: int = Field(ge=1)
    missing: MissingPolicy = "ffill"
    max_staleness: int = 5
    use_in: list[EntryPoint]


class PrepCfg(_Base):
    """`data.preparation:` — return/covariate preparation settings."""

    returns: Literal["log"] = "log"
    standardize_covariates: bool = True
    missing_returns: Literal["drop", "error"] = "drop"


class DataSection(_Base):
    """`data:` — provider, clock, cache, covariates, preparation."""

    primary: PriceSource
    clock: Clock = "trading"
    bars_per_day: int = 1
    cache: CacheCfg = Field(default_factory=CacheCfg)
    covariates: list[CovariateCfg] = Field(default_factory=list)
    preparation: PrepCfg = Field(default_factory=PrepCfg)


# --- model: params, priors, emission/duration/transition/lattice -----------------------------


class PriorCfg(_Base):
    """`prior:` — a named prior distribution and its parameters."""

    type: Literal["normal", "lognormal", "dirichlet"]
    sd: float | None = None
    alpha: float | None = None


class ParamCfg(_Base):
    """A single (possibly per-state) parameter's estimation mode and initialization."""

    mode: ParamMode
    values: list[float] | None = None
    init: list[float] | None = None
    init_mean_sojourn_days: list[float] | None = None
    init_mean_sojourn_steps: list[float] | None = None
    bounds: tuple[float, float] | None = None
    constraint: Literal["none", "increasing"] = "none"
    prior: PriorCfg | None = None
    tie: str | None = None


class EmissionCovariates(_Base):
    """`model.emission.covariates:` — which covariates drive drift vs. log-volatility."""

    drift: list[str] = Field(default_factory=list)
    log_vol: list[str] = Field(default_factory=list)


class EmissionCfg(_Base):
    """`model.emission:` — emission family, per-parameter configs, covariate links."""

    family: EmissionFamily
    params: dict[str, ParamCfg]
    covariates: EmissionCovariates = Field(default_factory=EmissionCovariates)


class DurationCfg(_Base):
    """`model.duration:` — duration family, minimum sojourns, per-parameter configs."""

    family: DurationFamily
    r: list[int] | None = None
    params: dict[str, ParamCfg]
    covariates: list[str] = Field(default_factory=list)
    duration_covariates_at_entry: bool = False


class TransitionCfg(_Base):
    """`model.transition:` — jump/generator mode, order, covariates, priors."""

    mode: Literal["jump", "generator"] = "jump"
    order: int = Field(default=1, ge=1)
    duration_conditioning: Literal["current", "pair"] = "current"
    covariates: list[str] = Field(default_factory=list)
    prior: PriorCfg | None = None
    coef_prior_sd: float = 1.0


class LatticeCfg(_Base):
    """`model.lattice:` — multi-start initialization grids."""

    grids: dict[str, list[float]]


class ModelSection(_Base):
    """`model:` — state count, engine, initial-distribution mode, families, lattice."""

    n_states: int = Field(ge=2)
    engine: Literal["expanded", "explicit"] = "expanded"
    d_max: int | None = None
    initial_state: Literal["uniform", "stationary", "free"] = "uniform"
    emission: EmissionCfg
    duration: DurationCfg
    transition: TransitionCfg
    lattice: LatticeCfg | None = None


# --- estimation / forecast / evaluation / output ----------------------------------------------


class EstimationSection(_Base):
    """`estimation:` — optimizer, convergence, multi-start, device."""

    optimizer: Literal["lbfgs", "adam"] = "lbfgs"
    max_iter: int = 1000
    tol_f: float = 1.0e-9
    tol_g: float = 1.0e-6
    restarts: int = 8
    init: list[Literal["rollvol_kmeans", "lattice_jitter", "random"]] = Field(default_factory=list)
    use_priors: bool = True
    std_errors: bool = False
    device: Literal["cpu", "gpu", "auto"] = "auto"
    chunk_size: int | None = Field(default=None, ge=1)  # explicit engine: steps per chunk (None = auto)
    parallel_scan: bool = False  # expanded engine: associative-scan smoothing (memory T*S^2)


class CovariateForecasterCfg(_Base):
    """`forecast.covariate_forecaster:` — a discriminated-union-ish free-form config.

    Type-specific keys (`mean_block_length` for `bootstrap`, `diagonal` for `ar`, `path` for
    `scenario`) are validated loosely as extra fields are forbidden per type; unrecognized keys
    for the chosen `type` are still rejected by `extra='forbid'` on the concrete subclass chosen
    by `type`, resolved in `_validate_cross_fields`.
    """

    type: Literal["freeze", "scenario", "bootstrap", "ar"]
    mean_block_length: int | None = None
    diagonal: bool | None = None
    path: str | None = None


class ForecastSection(_Base):
    """`forecast:` — horizon, simulation count, covariate forecaster, quantile levels."""

    horizon: int
    n_sim: int
    covariate_forecaster: CovariateForecasterCfg
    quantiles: list[float]

    @model_validator(mode="after")
    def _check_quantiles(self) -> ForecastSection:
        qs = self.quantiles
        if any(not (0.0 < q < 1.0) for q in qs):
            raise ConfigError("quantiles must be strictly between 0 and 1", path="quantiles")
        if list(qs) != sorted(qs) or len(set(qs)) != len(qs):
            raise ConfigError("quantiles must be strictly increasing", path="quantiles")
        return self


class BacktestCfg(_Base):
    """`evaluation.backtest:` — walk-forward scheme, horizons, embargo."""

    scheme: Literal["expanding", "rolling"]
    initial_train_years: float
    rolling_years: float | None = None
    refit_every_steps: int
    horizons: list[int]
    embargo_steps: int = 0


class StrategyCfg(_Base):
    """`evaluation.strategy:` — a simple regime-conditioned trading rule."""

    type: Literal["regime_long_flat", "regime_target_vol"]
    flat_if: dict[str, Any] | None = None
    target_vol: float | None = None
    cost_bps: float = 0.0


class LabelSource(_Base):
    """`evaluation.labels:` — an optional external ground-truth regime label source."""

    path: str


class EvaluationSection(_Base):
    """`evaluation:` — metrics, backtest scheme, optional strategy/labels."""

    metrics: list[str]
    backtest: BacktestCfg
    strategy: StrategyCfg | None = None
    labels: LabelSource | None = None


class OutputSection(_Base):
    """`output:` — which artifacts to write."""

    write: list[Literal["posterior", "forecast", "backtest", "strategy", "report"]]


# --- top level ---------------------------------------------------------------------------------


class ProjectConfig(_Base):
    """The whole resolved regimekit configuration."""

    schema_version: int = CURRENT_SCHEMA_VERSION
    profile: Literal["core", "finance"] = "finance"
    project: ProjectSection
    data: DataSection
    model: ModelSection
    estimation: EstimationSection
    forecast: ForecastSection
    evaluation: EvaluationSection
    output: OutputSection
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> ProjectConfig:
        _check_profile(self)
        _check_param_cfg_consistency(self)
        _check_clock_and_bars_per_day(self)
        _check_per_state_lengths(self)
        _check_covariate_cross_references(self)
        _check_transition_order(self)
        _check_generator_mode(self)
        _check_engine_duration_covariates_warning(self)
        _resolve_d_max(self)
        return self


# --- cross-field rule implementations (kept out of the class body for readability) -----------


def _warn(config: ProjectConfig, message: str) -> None:
    config.warnings.append(message)
    warnings_module.warn(message, stacklevel=3)


def _check_profile(config: ProjectConfig) -> None:
    if config.profile == "core":
        raise ConfigError("profile 'core' is not executable in this release; use 'finance'", path="profile")


def _named_param_cfgs(config: ProjectConfig) -> list[tuple[str, ParamCfg]]:
    """`(dotted path, ParamCfg)` for every `ParamCfg` in `model.emission.params`/`model.duration.params`."""
    return [(f"model.emission.params.{key}", pc) for key, pc in config.model.emission.params.items()] + [
        (f"model.duration.params.{key}", pc) for key, pc in config.model.duration.params.items()
    ]


def _check_param_cfg_consistency(config: ProjectConfig) -> None:
    for path, pc in _named_param_cfgs(config):
        if pc.init_mean_sojourn_days is not None and pc.init_mean_sojourn_steps is not None:
            raise ConfigError(
                "at most one of init_mean_sojourn_days/init_mean_sojourn_steps",
                path=f"{path}.init_mean_sojourn_days",
            )
        if pc.mode == "fixed" and pc.values is None:
            raise ConfigError("mode='fixed' requires values", path=f"{path}.values")
        if pc.mode == "bounded" and pc.bounds is None:
            raise ConfigError("mode='bounded' requires bounds", path=f"{path}.bounds")
        if pc.mode == "tied" and pc.tie is None:
            raise ConfigError("mode='tied' requires tie", path=f"{path}.tie")
        if pc.mode == "prior" and pc.prior is None:
            raise ConfigError("mode='prior' requires prior", path=f"{path}.prior")


def _check_clock_and_bars_per_day(config: ProjectConfig) -> None:
    if config.data.clock != "steps":
        return
    if config.data.bars_per_day != 1:
        raise ConfigError("bars_per_day must be 1 under clock='steps'", path="data.bars_per_day")
    for path, pc in _named_param_cfgs(config):
        if pc.init_mean_sojourn_days is not None:
            raise ConfigError(
                "init_mean_sojourn_days is not allowed under clock='steps'; use init_mean_sojourn_steps",
                path=f"{path}.init_mean_sojourn_days",
            )


def _check_per_state_lengths(config: ProjectConfig) -> None:
    n = config.model.n_states
    duration = config.model.duration
    if duration.r is not None and len(duration.r) != n:
        raise ConfigError(f"r must have length n_states={n}", path="model.duration.r")

    def _check(section_path: str, params: dict[str, ParamCfg]) -> None:
        for key, pc in params.items():
            for field_name in ("values", "init", "init_mean_sojourn_days", "init_mean_sojourn_steps"):
                value = getattr(pc, field_name)
                if value is not None and len(value) != n:
                    raise ConfigError(
                        f"{field_name} must have length n_states={n}",
                        path=f"{section_path}.params.{key}.{field_name}",
                    )

    _check("model.emission", config.model.emission.params)
    _check("model.duration", config.model.duration.params)


def _check_covariate_cross_references(config: ProjectConfig) -> None:
    by_name = {c.name: c for c in config.data.covariates}

    def _require(name: str, entry_point: EntryPoint, path: str) -> None:
        cov = by_name.get(name)
        if cov is None:
            raise ConfigError(f"covariate {name!r} not found in data.covariates", path=path)
        if entry_point not in cov.use_in:
            raise ConfigError(f"covariate {name!r} does not list {entry_point!r} in its use_in", path=path)

    for name in config.model.transition.covariates:
        _require(name, "transition", "model.transition.covariates")
    for name in config.model.duration.covariates:
        _require(name, "duration", "model.duration.covariates")
    for name in config.model.emission.covariates.drift:
        _require(name, "emission", "model.emission.covariates.drift")
    for name in config.model.emission.covariates.log_vol:
        _require(name, "emission", "model.emission.covariates.log_vol")


def _check_transition_order(config: ProjectConfig) -> None:
    order = config.model.transition.order
    if order > MAX_ORDER_WARN:
        _warn(
            config, f"transition.order={order} > {MAX_ORDER_WARN}; higher orders expand the state space fast"
        )


def _check_generator_mode(config: ProjectConfig) -> None:
    transition = config.model.transition
    if transition.mode != "generator":
        return
    duration = config.model.duration
    if duration.family != "geometric":
        raise ConfigError(
            "mode='generator' is only valid with duration.family='geometric'", path="model.transition.mode"
        )
    if duration.r is not None and any(r != 1 for r in duration.r):
        raise ConfigError("mode='generator' requires all r == 1", path="model.transition.mode")


def _check_engine_duration_covariates_warning(config: ProjectConfig) -> None:
    duration = config.model.duration
    if config.model.engine == "explicit" and duration.family == "negbin" and duration.covariates:
        duration.duration_covariates_at_entry = True
        _warn(config, "explicit engine + negbin duration covariates: switching to entry-time evaluation")


def _nominal_dt_estimate(clock: Clock) -> float:
    if clock == "steps":
        return 1.0
    days_per_year = TRADING_DAYS_PER_YEAR if clock == "trading" else CALENDAR_DAYS_PER_YEAR
    return 1.0 / days_per_year


def _expected_sojourn_steps(config: ProjectConfig) -> float | None:
    clock = config.data.clock
    nominal_dt = _nominal_dt_estimate(clock)
    bars_per_day = config.data.bars_per_day
    values: list[float] = []
    for pc in config.model.duration.params.values():
        if pc.init_mean_sojourn_steps is not None:
            values.extend(pc.init_mean_sojourn_steps)
        elif pc.init_mean_sojourn_days is not None:
            values.extend(
                days_to_steps(d, clock, nominal_dt, bars_per_day) for d in pc.init_mean_sojourn_days
            )
        elif pc.init is not None and clock == "steps":
            values.extend(pc.init)
    return max(values) if values else None


def _resolve_d_max(config: ProjectConfig) -> None:
    expected = _expected_sojourn_steps(config)
    if config.model.d_max is None:
        if expected is None:
            config.model.d_max = D_MAX_CAP
        else:
            config.model.d_max = min(D_MAX_CAP, round(D_MAX_SOJOURN_MULTIPLIER * expected))
    elif expected is not None and config.model.d_max < D_MAX_WARN_SOJOURN_MULTIPLIER * expected:
        _warn(
            config,
            f"model.d_max={config.model.d_max} is small relative to the expected sojourn "
            f"({expected:.1f} steps); consider raising it",
        )


def parse_project_config(data: dict[str, Any]) -> ProjectConfig:
    """Validate `data` into a `ProjectConfig`, converting pydantic errors into `ConfigError`."""
    try:
        return ProjectConfig.model_validate(data)
    except pydantic.ValidationError as exc:
        first = exc.errors()[0]
        path = ".".join(str(part) for part in first["loc"])
        raise ConfigError(first["msg"], path=path or None) from exc
