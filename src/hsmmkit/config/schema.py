"""Configuration schema: pydantic v2 models with `extra="forbid"` and cross-field validation.

Rules (concept §5.7, overview §4.1, CLAUDE.md §3.4): see `_validate_cross_fields` for the
numbered rules. `parse_project_config` is the entry point that converts pydantic's own
`ValidationError` (unknown keys, wrong types, ...) into `ConfigError(path=...)`, matching the
"all raise ConfigError" contract; custom cross-field rules raise `ConfigError` directly from
inside `@model_validator` methods (pydantic does not intercept non-`ValueError` exceptions, so
they propagate unwrapped).
"""

from __future__ import annotations

import importlib
import warnings as warnings_module
from pathlib import Path
from typing import Any, ClassVar, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hsmmkit.config.defaults import (
    D_MAX_CAP,
    D_MAX_SOJOURN_MULTIPLIER,
    D_MAX_WARN_SOJOURN_MULTIPLIER,
    MAX_ORDER_WARN,
)
from hsmmkit.plugins import REGISTRY, Kind
from hsmmkit.utils.clock import get_clock, parse_clock_options
from hsmmkit.utils.errors import ConfigError

CURRENT_SCHEMA_VERSION = 1

Transform = Literal["log", "diff", "pct", "zscore"]
MissingPolicy = Literal["ffill", "drop", "error"]
EntryPoint = Literal["transition", "duration", "emission"]
ParamMode = Literal["free", "fixed", "tied", "bounded", "prior"]
CORE_OUTPUTS = ("posterior", "forecast", "backtest", "cv", "report")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _registered_names(kind: Kind, *, needs: str | None = None) -> list[str]:
    """Registered names of `kind` (optionally only objects with attribute `needs`); loads the components."""
    importlib.import_module("hsmmkit.model.emission")  # the families register on import
    importlib.import_module("hsmmkit.model.duration")
    importlib.import_module("hsmmkit.inference.covariate_forecasters")
    importlib.import_module("hsmmkit.estimation.init")
    if kind == "metric":
        from hsmmkit.evaluation.metrics.registry import available

        return list(available())
    return [e.name for e in REGISTRY.entries(kind) if needs is None or hasattr(e.obj, needs)]


def _check_name(kind: Kind, name: str, path: str, *, needs: str | None = None) -> str:
    """`name` if it is registered for `kind`, else a `ConfigError(path)` listing the registered names."""
    names = _registered_names(kind, needs=needs)
    if name not in names:
        listing = (
            ", ".join(f"{e.name} ({e.provenance})" for e in REGISTRY.entries(kind) if e.name in names)
            or "<none>"
        )
        raise ConfigError(f"unknown {kind} '{name}'; registered: {listing}", path=path)
    return name


def _register_core_sources() -> None:
    """Import the core sources (lazy: they import this module) so that their kinds are registered."""
    importlib.import_module("hsmmkit.data.sources")


def _validate_source_options(
    kind: str, options: dict[str, Any], path: str, *, covariate: bool = False
) -> None:
    """Validate a core source's options with its `options_model` (lazy: the sources import this module)."""
    from hsmmkit.data.sources.base import validate_options

    validate_options(kind, options, path, covariate=covariate)


# --- project / data -------------------------------------------------------------------------


class ProjectSection(_Base):
    """`project:` — run identity, seed, output location, logging."""

    name: str
    seed: int
    output_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class SourceCfg(_Base):
    """`data.source:` — where the observations come from.

    `kind` names a registered source (`REGISTRY` kind `source`); the other keys are the options of that
    source and are validated by the options model it registers.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    kind: str

    def option(self, name: str, default: Any = None) -> Any:  # noqa: ANN401
        """The source option `name` (extra keys), or `default`."""
        extra = self.model_extra or {}
        return extra.get(name, default)

    def options(self) -> dict[str, Any]:
        """All options of the source (the extra keys)."""
        return dict(self.model_extra or {})


class CovariateCfg(_Base):
    """`data.covariates[]:` — one covariate series and how it enters the model."""

    name: str
    source: SourceCfg | None = None
    column: str | None = None
    transform: list[Transform] = Field(default_factory=list)
    lag: int = Field(ge=1)
    missing: MissingPolicy = "ffill"
    max_staleness: int = 5
    use_in: list[EntryPoint]


class PrepCfg(_Base):
    """`data.preparation:` — observation and covariate preparation settings."""

    observation_transform: str | None = None
    standardize_observations: bool = False
    standardize_covariates: bool = True
    missing_observations: Literal["mask", "drop", "error"] = "mask"

    @field_validator("observation_transform")
    @classmethod
    def _known_transform(cls, value: str | None) -> str | None:
        if value is not None and REGISTRY.names("transform"):  # the registry fills in E10-T05
            _check_name("transform", value, "data.preparation.observation_transform")
        return value


class DataSection(_Base):
    """`data:` — source, clock, covariates, preparation."""

    source: SourceCfg
    clock: str = "steps"
    clock_options: dict[str, Any] = Field(default_factory=dict)
    covariates: list[CovariateCfg] = Field(default_factory=list)
    preparation: PrepCfg = Field(default_factory=PrepCfg)

    def clock_option(self, name: str, default: Any = None) -> Any:  # noqa: ANN401
        """The clock option `name` (`clock_options`), or `default`."""
        return self.clock_options.get(name, default)

    @field_validator("source")
    @classmethod
    def _known_source(cls, value: SourceCfg) -> SourceCfg:
        _register_core_sources()
        _check_name("source", value.kind, "data.source.kind")
        _validate_source_options(
            value.kind, value.options(), "data.source"
        )  # core sources: ConfigError per key
        return value

    @model_validator(mode="after")
    def _valid_covariate_sources(self) -> DataSection:
        for i, cov in enumerate(self.covariates):
            if cov.source is not None:
                _register_core_sources()
                _check_name("source", cov.source.kind, f"data.covariates[{i}].source.kind")
                _validate_source_options(
                    cov.source.kind, cov.source.options(), f"data.covariates[{i}].source", covariate=True
                )
        return self

    @field_validator("clock")
    @classmethod
    def _known_clock(cls, value: str) -> str:
        get_clock(value)  # ConfigError(path="data.clock") listing the registered clocks
        return value

    @model_validator(mode="after")
    def _valid_clock_options(self) -> DataSection:
        parse_clock_options(self.clock, self.clock_options)  # ConfigError(path="data.clock_options.<field>")
        return self


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
    init_mean_sojourn_steps: list[float] | None = None
    bounds: tuple[float, float] | None = None
    constraint: Literal["none", "increasing"] = "none"
    prior: PriorCfg | None = None
    tie: str | None = None


class EmissionCovariates(_Base):
    """`model.emission.covariates:` — which covariates drive the location and the log-scale.

    `mean` / `log_scale` are the keys of the `gaussian` family; `drift` / `log_vol` those of an application's
    `gbm` family and of `student_t`.
    """

    mean: list[str] = Field(default_factory=list)
    log_scale: list[str] = Field(default_factory=list)
    drift: list[str] = Field(default_factory=list)
    log_vol: list[str] = Field(default_factory=list)


class EmissionCfg(_Base):
    """`model.emission:` — emission family, per-parameter configs, covariate links."""

    family: str
    params: dict[str, ParamCfg]
    covariates: EmissionCovariates = Field(default_factory=EmissionCovariates)
    scale_by_dt: bool | None = None
    """Declared time scaling of the family (`None`: the family's own); a different value is rejected."""
    ito_correction: bool | None = None
    """`gaussian`: subtract `scale**2 / 2` from the location (`None`: the family's own, `False`)."""
    covariance: Literal["diag"] = "diag"
    """`gaussian`: covariance structure; `full` is not available before E11-T02."""
    per_series: bool = True
    """`True`: own location/scale per series, `(N, M)`; `False`: one per regime, shared."""

    @field_validator("family")
    @classmethod
    def _known_family(cls, value: str) -> str:
        return _check_name("emission", value, "model.emission.family", needs="from_config")


class DurationCfg(_Base):
    """`model.duration:` — duration family, minimum sojourns, per-parameter configs."""

    family: str
    r: list[int] | None = None
    params: dict[str, ParamCfg]
    covariates: list[str] = Field(default_factory=list)
    duration_covariates_at_entry: bool = False

    @field_validator("family")
    @classmethod
    def _known_family(cls, value: str) -> str:
        return _check_name("duration", value, "model.duration.family", needs="from_config")


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
    init: list[str] = Field(default_factory=list)
    use_priors: bool = True
    std_errors: bool = False
    device: Literal["cpu", "gpu", "auto"] = "auto"
    chunk_size: int | None = Field(default=None, ge=1)  # explicit engine: steps per chunk (None = auto)
    parallel_scan: bool = False  # expanded engine: associative-scan smoothing (memory T*S^2)

    @field_validator("init")
    @classmethod
    def _known_inits(cls, value: list[str]) -> list[str]:
        for i, name in enumerate(value):
            _check_name("init", name, f"estimation.init[{i}]")
        return value


class CovariateForecasterCfg(_Base):
    """`forecast.covariate_forecaster:` — a discriminated-union-ish free-form config.

    Type-specific keys (`mean_block_length` for `bootstrap`, `diagonal` for `ar`, `path` for
    `scenario`) are validated loosely as extra fields are forbidden per type; unrecognized keys
    for the chosen `type` are still rejected by `extra='forbid'` on the concrete subclass chosen
    by `type`, resolved in `_validate_cross_fields`.
    """

    type: str
    mean_block_length: int | None = None
    diagonal: bool | None = None
    path: str | None = None

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        return _check_name("covariate_forecaster", value, "forecast.covariate_forecaster.type")


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
    """`evaluation.backtest:` — walk-forward scheme, horizons, embargo.

    `initial_train` and `rolling_window` are lengths in clock units (years under the calendar-based clocks,
    steps under `steps`); `initial_train_steps` and `rolling_window_steps` give the same lengths in rows.
    Exactly one of `initial_train` / `initial_train_steps`; at most one of the rolling pair.
    """

    scheme: Literal["expanding", "rolling"]
    initial_train: float | None = None
    initial_train_steps: int | None = Field(default=None, ge=1)
    rolling_window: float | None = None
    rolling_window_steps: int | None = Field(default=None, ge=1)
    refit_every_steps: int
    horizons: list[int]
    embargo_steps: int = 0
    n_sim: int | None = Field(default=None, ge=1)  # simulated paths per origin (None = forecast.n_sim)

    @model_validator(mode="after")
    def _one_of_each_pair(self) -> BacktestCfg:
        if (self.initial_train is None) == (self.initial_train_steps is None):
            raise ConfigError(
                "exactly one of initial_train / initial_train_steps", path="evaluation.backtest.initial_train"
            )
        if self.rolling_window is not None and self.rolling_window_steps is not None:
            raise ConfigError(
                "at most one of rolling_window / rolling_window_steps",
                path="evaluation.backtest.rolling_window",
            )
        return self


class LabelSource(_Base):
    """`evaluation.labels:` — an optional external ground-truth regime label source."""

    path: str


class EvaluationSection(_Base):
    """`evaluation:` — metrics, backtest scheme, optional labels."""

    metrics: list[str]
    backtest: BacktestCfg
    labels: LabelSource | None = None

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, value: list[str]) -> list[str]:
        for i, name in enumerate(value):
            _check_name("metric", name, f"evaluation.metrics[{i}]")
        return value


class OutputSection(_Base):
    """`output:` — which artifacts to write."""

    write: list[str]

    extra_outputs: ClassVar[tuple[str, ...]] = ()
    """Output names an extension adds (set by the subclass of the extension)."""

    @field_validator("write")
    @classmethod
    def _known_outputs(cls, value: list[str]) -> list[str]:
        allowed = (*CORE_OUTPUTS, *cls.extra_outputs)
        for i, name in enumerate(value):
            if name not in allowed:
                raise ConfigError(
                    f"unknown output '{name}'; allowed: {', '.join(allowed)}", path=f"output.write[{i}]"
                )
        return value


# --- top level ---------------------------------------------------------------------------------


class ProjectConfig(_Base):
    """The whole resolved configuration of the domain-neutral core.

    Applications extend it by subclassing (D24): they replace sections by subclasses, set `SCHEMA_ID` and
    `SCHEMA_VERSION`, and register their own migrations (`hsmmkit.config.migrate`).
    """

    SCHEMA_ID: ClassVar[str] = "core"
    SCHEMA_VERSION: ClassVar[int] = CURRENT_SCHEMA_VERSION

    schema_version: int = CURRENT_SCHEMA_VERSION
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
        _check_param_cfg_consistency(self)
        _check_per_state_lengths(self)
        _check_covariate_cross_references(self)
        _check_transition_order(self)
        _check_generator_mode(self)
        _check_engine_duration_covariates_warning(self)
        _resolve_d_max(self)
        return self

    def expected_sojourn_steps(self) -> float | None:
        """Largest initial mean sojourn in steps that the config states (`None` when it states none)."""
        values: list[float] = []
        for pc in self.model.duration.params.values():
            if pc.init_mean_sojourn_steps is not None:
                values.extend(pc.init_mean_sojourn_steps)
            elif pc.init is not None and self.data.clock == "steps":
                values.extend(pc.init)
        return max(values) if values else None


# --- cross-field rule implementations (kept out of the class body for readability) -----------


def _warn(config: ProjectConfig, message: str) -> None:
    config.warnings.append(message)
    warnings_module.warn(message, stacklevel=3)


def _named_param_cfgs(config: ProjectConfig) -> list[tuple[str, ParamCfg]]:
    """`(dotted path, ParamCfg)` for every `ParamCfg` in `model.emission.params`/`model.duration.params`."""
    return [(f"model.emission.params.{key}", pc) for key, pc in config.model.emission.params.items()] + [
        (f"model.duration.params.{key}", pc) for key, pc in config.model.duration.params.items()
    ]


def _check_param_cfg_consistency(config: ProjectConfig) -> None:
    for path, pc in _named_param_cfgs(config):
        if pc.mode == "fixed" and pc.values is None:
            raise ConfigError("mode='fixed' requires values", path=f"{path}.values")
        if pc.mode == "bounded" and pc.bounds is None:
            raise ConfigError("mode='bounded' requires bounds", path=f"{path}.bounds")
        if pc.mode == "tied" and pc.tie is None:
            raise ConfigError("mode='tied' requires tie", path=f"{path}.tie")
        if pc.mode == "prior" and pc.prior is None:
            raise ConfigError("mode='prior' requires prior", path=f"{path}.prior")


def _check_per_state_lengths(config: ProjectConfig) -> None:
    n = config.model.n_states
    duration = config.model.duration
    if duration.r is not None and len(duration.r) != n:
        raise ConfigError(f"r must have length n_states={n}", path="model.duration.r")

    def _check(section_path: str, params: dict[str, ParamCfg]) -> None:
        for key, pc in params.items():
            for field_name in ("values", "init", "init_mean_sojourn_steps"):
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
    for key in ("mean", "log_scale", "drift", "log_vol"):
        for name in getattr(config.model.emission.covariates, key):
            _require(name, "emission", f"model.emission.covariates.{key}")


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


def _resolve_d_max(config: ProjectConfig) -> None:
    expected = config.expected_sojourn_steps()
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


def parse_project_config(
    data: dict[str, Any], config_cls: type[ProjectConfig] = ProjectConfig
) -> ProjectConfig:
    """Validate `data` into a `config_cls`, converting pydantic errors into `ConfigError`."""
    try:
        return config_cls.model_validate(data)
    except pydantic.ValidationError as exc:
        first = exc.errors()[0]
        path = ".".join(str(part) for part in first["loc"])
        raise ConfigError(first["msg"], path=path or None) from exc
