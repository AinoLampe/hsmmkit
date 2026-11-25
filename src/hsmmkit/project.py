"""`Project`: the Python entry point that mirrors every CLI command (E8-T02).

A `Project` holds a resolved `FinanceConfig` and the directory its relative paths (csv files) resolve
against. Every method returns its result object and writes nothing unless called with `write=True`;
the CLI (`cli/app.py`) is a thin wrapper that passes `write=True`. Importing this module triggers no
network access and no file writes.

This module and `cli/` are the only places allowed to import the finance profile (overview section 4.3).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic

from regimekit.config.schema import BacktestCfg
from regimekit.config.yaml_io import dump_resolved, load_config
from regimekit.data.contracts import CovariateSeries, ObservationSet, PreparedDataset
from regimekit.data.prepare import Preparer, PrepDiagnostics, prep_stats_to_json
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.engine.explicit_duration import ExplicitDurationEngine
from regimekit.estimation import fit as fit_model
from regimekit.estimation.report import diagnose
from regimekit.estimation.results import FitResult
from regimekit.estimation.selftest import SelftestResult, run_selftest
from regimekit.evaluation.backtest import run_backtest
from regimekit.evaluation.compare import CompareEntry, check_snapshots, comparison_table, render_markdown
from regimekit.evaluation.report import build_report, versions
from regimekit.evaluation.results import BacktestResult
from regimekit.evaluation.strategies import make_strategy
from regimekit.finance import (
    Cache,
    DataQualityReport,
    FetchContext,
    FinanceConfig,
    PriceSource,
    SnapshotPolicy,
    build_quality_report,
)
from regimekit.inference.covariate_forecasters import CovariateForecaster, make_forecaster
from regimekit.inference.decode import decode as decode_posterior
from regimekit.inference.forecast import MonteCarloForecaster
from regimekit.inference.results import ForecastResult, PosteriorResult
from regimekit.inference.simulate import segments_of, simulate_paths
from regimekit.model.build import build_model_spec
from regimekit.model.params.manager import ParamManager
from regimekit.persistence.results import (
    BACKTEST_META,
    read_backtest,
    write_backtest,
    write_forecast,
    write_posterior,
    write_prepared,
    write_simulation,
)
from regimekit.persistence.store import ModelStore
from regimekit.plugins import REGISTRY
from regimekit.statespace.builders import build_expanded, build_explicit
from regimekit.utils.clock import CALENDAR_DAYS_PER_YEAR
from regimekit.utils.errors import ConfigError, DataError, PersistenceError, RegimekitError
from regimekit.utils.logging import get_logger
from regimekit.utils.rng import PURPOSE_SIMULATE, key_for

__all__ = [
    "FetchedInputs",
    "FinanceConfig",
    "Project",
    "SimulationOutput",
    "apply_overrides",
    "check_simulate_args",
]


class FetchedInputs(NamedTuple):
    """Raw inputs of a run: primary prices, covariate series, and the cache/provider they came through."""

    obs: ObservationSet
    covs: list[CovariateSeries]
    cache: Cache
    primary_name: str


@dataclass(frozen=True)
class SimulationOutput:
    """`Project.simulate`: `n_paths` simulated paths of `steps` steps (what `simulation.parquet` holds)."""

    times: np.ndarray
    regimes: np.ndarray
    phases: np.ndarray
    returns: np.ndarray
    segments: list[list[tuple[int, int, int]]]


def apply_overrides(config: FinanceConfig, overrides: Mapping[str, object]) -> None:
    """Set dotted config keys (`{"project.seed": 7}`) on `config` in place, with validation.

    Raises:
        ConfigError: an unknown key or a value the schema rejects (path = the dotted key).
    """
    for key, value in overrides.items():
        *parents, leaf = key.split(".")
        target: Any = config
        try:
            for part in parents:
                target = getattr(target, part)
            if leaf not in type(target).model_fields:
                raise AttributeError(leaf)
            setattr(target, leaf, value)
        except AttributeError as exc:
            raise ConfigError(f"unknown configuration key {key!r}", path=key) from exc
        except pydantic.ValidationError as exc:
            raise ConfigError(exc.errors()[0]["msg"], path=key) from exc


def summary_lines(config: FinanceConfig) -> list[str]:
    """The `validate-config` summary: states, engine, families, covariate entry points, warnings."""
    emission_covariates = config.model.emission.covariates
    entry_points = sorted(
        {ep for cov in config.data.covariates for ep in cov.use_in}
        | ({"transition"} if config.model.transition.covariates else set())
        | ({"duration"} if config.model.duration.covariates else set())
        | (
            {"emission"}
            if any(
                (
                    emission_covariates.mean,
                    emission_covariates.log_scale,
                    emission_covariates.drift,
                    emission_covariates.log_vol,
                )
            )
            else set()
        )
    )
    return [
        f"n_states: {config.model.n_states}",
        f"engine: {config.model.engine}",
        f"emission family: {config.model.emission.family}",
        f"duration family: {config.model.duration.family}",
        f"covariate entry points: {', '.join(entry_points) if entry_points else '(none)'}",
        f"warnings: {'; '.join(config.warnings) if config.warnings else '(none)'}",
    ]


def _train_end(obs: ObservationSet, backtest: BacktestCfg) -> np.datetime64:
    """End of the initial training window.

    First timestamp + `initial_train` years (or the row given by `initial_train_steps`), capped at the last.
    """
    times = obs.require_times()
    last: np.datetime64 = times[-1]
    if backtest.initial_train_steps is not None:
        return cast(np.datetime64, times[min(backtest.initial_train_steps, times.shape[0]) - 1])
    assert backtest.initial_train is not None
    seconds = round(backtest.initial_train * CALENDAR_DAYS_PER_YEAR * 86_400)
    end: np.datetime64 = times[0] + np.timedelta64(seconds, "s")
    return min(end, last)


def _two_newest_generations(
    cache: Cache, provider: str, primary: PriceSource
) -> tuple[ObservationSet, ObservationSet] | None:
    """The two newest cached generations of the primary request `(newest, previous)`, if both exist."""
    key = cache.price_key(provider, primary.symbols, primary.interval, primary.adjusted)
    newest = cache.generations(key)[:2]
    if len(newest) < 2:
        return None
    loaded = [cache.load_prices(key, g.snapshot_id) for g in newest]
    if loaded[0] is None or loaded[1] is None:  # pragma: no cover  (listed generations exist on disk)
        return None
    return loaded[0], loaded[1]


def _dropped_rows(diagnostics: PrepDiagnostics | None, completed: bool) -> dict[str, int]:
    if diagnostics is None:
        return {}
    if not completed:  # the policies did not finish: only the missing-price count is known
        return {"missing_price": diagnostics.missing_rows}
    return {
        "missing_price": diagnostics.missing_rows,
        "warmup": diagnostics.warmup_rows,
        **{f"covariate:{name}": n for name, n in diagnostics.dropped.items()},
    }


def _check_pinned(config: FinanceConfig, offline: bool) -> None:
    data = config.data
    policy = SnapshotPolicy(data.cache.snapshot, data.cache.max_age_days, offline)
    if policy.pinned is not None and data.covariates:
        raise ConfigError(
            "a pinned snapshot id identifies one cache entry and cannot pin covariates; use "
            "`snapshot: latest` with `--offline` to replay a run with covariates",
            path="data.cache.snapshot",
        )


def engine_for(config: FinanceConfig, kind: str) -> ExpandedHMMEngine | ExplicitDurationEngine:
    """The engine `kind` names, configured from `config.estimation` (`chunk_size`/`parallel_scan`)."""
    if kind == "explicit":
        return ExplicitDurationEngine(chunk_size=config.estimation.chunk_size)
    return ExpandedHMMEngine(parallel_scan=config.estimation.parallel_scan)


def _read_scenario(path: Path) -> dict[str, np.ndarray]:
    """A scenario CSV (one column per covariate, `h` rows) as `{name: values}`; `time` columns are ignored.

    Values are in the *prepared* covariate units (transformed and standardized, as in
    `prepared.parquet`).
    """
    try:
        frame = pd.read_csv(path)
        frame = frame.drop(columns=[c for c in frame.columns if str(c).lower() in ("time", "date")])
        return {str(c): frame[c].to_numpy(dtype=np.float64) for c in frame.columns}
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(f"cannot read scenario file {path}: {exc}", path="--scenario") from exc


def covariate_forecaster(
    config: FinanceConfig, config_dir: Path, dataset: PreparedDataset, scenario: Path | None
) -> CovariateForecaster | None:
    """The configured covariate forecaster (`scenario` forces one); `None` without covariates."""
    if not dataset.prep_stats.cov_names:
        if scenario is not None:
            raise ConfigError("--scenario given but the model has no covariates", path="--scenario")
        return None
    cfg = config.forecast.covariate_forecaster
    values: dict[str, np.ndarray] | None = None
    if scenario is not None:
        cfg = cfg.model_copy(update={"type": "scenario"})
        values = _read_scenario(scenario)
    elif cfg.type == "scenario":
        if cfg.path is None:
            raise ConfigError(
                "scenario forecaster needs forecast.covariate_forecaster.path or --scenario",
                path="forecast.covariate_forecaster.path",
            )
        values = _read_scenario((config_dir / cfg.path).resolve())
    return make_forecaster(cfg, values)


def check_simulate_args(params: str, steps: int | None, n_paths: int) -> None:
    """Validate `simulate`'s arguments (before any data is fetched).

    Raises:
        ConfigError: unknown `params`, or non-positive `steps` / `n_paths`.
    """
    if params not in ("config", "model"):
        raise ConfigError(f"--params must be 'config' or 'model', got {params!r}", path="--params")
    if n_paths < 1 or (steps is not None and steps < 1):
        raise ConfigError("--steps and --n-paths must be >= 1", path="--steps")


def _json_default(value: object) -> object:
    if hasattr(value, "__array__"):
        return np.asarray(value).tolist()
    return str(value)


def _write_text(path: Path, text: str, what: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise PersistenceError(f"cannot write {what} to {path.parent}: {exc}", path=str(path.parent)) from exc


class Project:
    """A regimekit project: configuration plus the operations of the CLI as methods.

    Args:
        config: The validated configuration (apply overrides before constructing, or use `from_yaml`).
        config_dir: Directory relative provider paths (csv files) and scenario paths resolve against
            (default: the current directory).
        out_dir: Where `write=True` puts files (default: `config.project.output_dir`).
        offline: Replay the cache without network access.

    Attributes:
        last_quality: The data-quality report of the last `fetch`.
        last_fetched: The raw inputs of the last `fetch` / `fetch_inputs`.
        last_data: The prepared dataset of the last `fetch` / `fit`.
    """

    def __init__(
        self,
        config: FinanceConfig,
        *,
        config_dir: Path | str | None = None,
        out_dir: Path | str | None = None,
        offline: bool = False,
    ) -> None:
        """Wrap `config`; no network access and no file writes happen here."""
        self.config = config
        self.config_dir = Path(config_dir) if config_dir is not None else Path.cwd()
        self.out_dir = Path(out_dir) if out_dir is not None else Path(config.project.output_dir)
        self.offline = offline
        self.last_quality: DataQualityReport | None = None
        self.last_fetched: FetchedInputs | None = None
        self.last_data: PreparedDataset | None = None
        self.last_backtest_meta: dict[str, Any] | None = None

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        overrides: Mapping[str, object] | None = None,
        *,
        out_dir: Path | str | None = None,
        offline: bool = False,
    ) -> Project:
        """Load `path`, apply dotted `overrides` (`{"project.seed": 7}`), and wrap the result."""
        config = cast(FinanceConfig, load_config(path, FinanceConfig))
        apply_overrides(config, overrides or {})
        return cls(config, config_dir=Path(path).resolve().parent, out_dir=out_dir, offline=offline)

    # --- validate-config ---------------------------------------------------------------------------

    def validate(self, *, write: bool = False) -> list[str]:
        """The configuration summary; `write=True` also writes `config.resolved.yaml` to `out_dir`."""
        if write:
            try:
                self.out_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise PersistenceError(
                    f"cannot create {self.out_dir}: {exc}", path=str(self.out_dir)
                ) from exc
            dump_resolved(self.config, self.out_dir / "config.resolved.yaml")
        return summary_lines(self.config)

    # --- fetch -------------------------------------------------------------------------------------

    def fetch_inputs(self, *, offline: bool | None = None) -> FetchedInputs:
        """Fetch the primary prices and covariates through the cache (no preparation, no writes).

        Raises:
            ConfigError: a pinned snapshot together with covariates.
            DataError: the primary request returned fewer than two rows.
            ProviderError: a provider failed.
        """
        offline = self.offline if offline is None else offline
        data = self.config.data
        _check_pinned(self.config, offline)
        context = FetchContext(data, self.config_dir, offline)
        source = REGISTRY.get("source", data.source.kind)
        obs, covs = source.load(data.primary, obj=context)
        self.last_fetched = FetchedInputs(obs, covs, context.cache, context.primary_name)
        return self.last_fetched

    def fetch(self, *, offline: bool | None = None, write: bool = False) -> PreparedDataset:
        """Fetch through the cache and prepare; `write=True` writes the dataset, stats, snapshots, quality.

        The quality report (`last_quality`, and its files when writing) exists even when preparation
        fails (e.g. `missing: error`); the error is then re-raised.

        Raises:
            ConfigError: a pinned `data.cache.snapshot` combined with covariates.
            DataError: an empty primary request, or preparation failed.
            ProviderError: a provider failed.
            PersistenceError: the output directory cannot be written (exit 1, CLAUDE.md section 3.8).
        """
        offline = self.offline if offline is None else offline
        config, data_cfg, out_dir = self.config, self.config.data, self.out_dir
        _check_pinned(config, offline)
        log = get_logger("regimekit.fetch")
        outputs = [out_dir / n for n in ("prepared.parquet", "prep_stats.json", "snapshots.json")]
        if write:
            try:  # a failed run must not leave an earlier run's dataset next to its own report
                for stale in outputs:
                    stale.unlink(missing_ok=True)
            except OSError as exc:
                raise PersistenceError(f"cannot clear {out_dir}: {exc}", path=str(out_dir)) from exc
        fetched = self.fetch_inputs(offline=offline)
        obs, covs, p = fetched.obs, fetched.covs, data_cfg.primary
        generations = _two_newest_generations(fetched.cache, fetched.primary_name, p)

        preparer = Preparer(data_cfg)
        failure: DataError | None = None
        try:
            train_end = _train_end(obs, config.evaluation.backtest)
            dataset = preparer.fit(obs, covs, train_end).transform(obs, covs)
        except DataError as exc:
            failure = exc
        diagnostics = preparer.diagnostics
        report = build_quality_report(
            obs,
            generations=generations,
            staleness=None if diagnostics is None else diagnostics.staleness,
            dropped_rows=_dropped_rows(diagnostics, completed=failure is None),
        )
        self.last_quality = report
        if write:
            try:
                report.write(out_dir)
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the quality report to {out_dir}: {exc}", path=str(out_dir)
                ) from exc
        if failure is not None:
            raise failure

        if write:
            snapshots = {
                "primary": {
                    "provider": p.provider,
                    "symbols": list(p.symbols),
                    "interval": p.interval,
                    "adjusted": p.adjusted,
                    "snapshot_id": obs.snapshot_id,
                },
                "covariates": [
                    {"name": c.name, "provider": c.provider, "series": c.series, "snapshot_id": s.snapshot_id}
                    for c, s in zip(data_cfg.covariates, covs, strict=True)
                ],
            }
            try:
                write_prepared(dataset, outputs[0])
                outputs[1].write_text(prep_stats_to_json(dataset.prep_stats), encoding="utf-8")
                outputs[2].write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the prepared dataset to {out_dir}: {exc}", path=str(out_dir)
                ) from exc
        self.last_data = dataset
        log.info("fetch_done", out=str(out_dir) if write else None, rows=int(dataset.times.shape[0]))
        return dataset

    # --- fit ---------------------------------------------------------------------------------------

    def fit(self, data: PreparedDataset | None = None, *, write: bool = False) -> FitResult:
        """MAP-fit the configured model to `data` (default: `fetch()`).

        `write=True` saves the bundle `<out_dir>/<name>.rkmodel` and writes `fit_report.json` and
        `config.resolved.yaml`.

        Raises:
            ConfigError, DataError, ProviderError: from the fetch and the model assembly.
            NumericalError: no restart produced a finite fit.
            PersistenceError: the outputs cannot be written.
        """
        dataset = data if data is not None else self.fetch()
        spec = build_model_spec(self.config, dataset)
        result = fit_model(spec, dataset, self.config.estimation, self.config.project.seed)
        self.last_data = dataset
        if write:
            self.validate(write=True)
            self.save(result, self.out_dir / f"{self.config.project.name}.rkmodel", dataset)
            self._write_fit_report(result, dataset)
        get_logger("regimekit.fit").info(
            "fit_done", loglik=result.loglik, converged=result.converged, n_iter=result.n_iter
        )
        return result

    def _write_fit_report(self, fit: FitResult, data: PreparedDataset) -> None:
        spec = fit.spec
        space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
        try:
            diagnostics = diagnose(fit, spec, space, engine_for(self.config, spec.engine), data)
        except RegimekitError as exc:  # the fit itself succeeded: report without diagnostics
            get_logger("regimekit.fit").warning("diagnostics_unavailable", error=str(exc))
            diagnostics = {}
        report = {
            "name": self.config.project.name,
            "seed": self.config.project.seed,
            "converged": fit.converged,
            "n_iter": fit.n_iter,
            "objective": fit.objective,
            "loglik": fit.loglik,
            "n_free": ParamManager(spec.param_specs).n_free,
            "snapshot_ids": list(data.snapshot_ids),
            "restarts": fit.restarts.to_dict(orient="records"),
            "diagnostics": diagnostics,
        }
        _write_text(
            self.out_dir / "fit_report.json",
            json.dumps(report, indent=2, default=_json_default),
            "fit_report.json",
        )

    # --- save / load -------------------------------------------------------------------------------

    def save(self, fit: FitResult, path: Path | str, data: PreparedDataset | None = None) -> Path:
        """Write `fit` as a model bundle; `data` defaults to the dataset of the last `fit` / `fetch`."""
        dataset = data if data is not None else self.last_data
        if dataset is None:
            raise ConfigError("saving a model needs the dataset it was fitted on; pass data=", path="data")
        return ModelStore.save(fit, path, self.config, dataset)

    @staticmethod
    def load(path: Path | str) -> FitResult:
        """Read a model bundle back as a `FitResult`."""
        return ModelStore.load(path, FinanceConfig)

    # --- decode / forecast -------------------------------------------------------------------------

    def decode(
        self, fit: FitResult, data: PreparedDataset, *, write: bool = False, augmented: bool = False
    ) -> PosteriorResult:
        """Filtered / smoothed posteriors, Viterbi path, segments and IC of `fit` on `data`.

        The engine is built from `config.estimation` (`chunk_size`/`parallel_scan`) so decoding matches
        how the model was fit.

        Raises:
            NumericalError: the engine's log-likelihood is non-finite.
            PersistenceError: outputs cannot be written.
        """
        result = decode_posterior(fit, data, engine=engine_for(self.config, fit.spec.engine))
        if write:
            write_posterior(result, self.out_dir, augmented=augmented)
        get_logger("regimekit.decode").info(
            "decode_done", rows=int(data.times.shape[0]), loglik=result.loglik
        )
        return result

    def forecast(
        self,
        fit: FitResult,
        data: PreparedDataset,
        *,
        horizon: int | None = None,
        n_sim: int | None = None,
        scenario: Path | str | None = None,
        samples: bool = False,
        write: bool = False,
    ) -> ForecastResult:
        """Monte-Carlo forecast from the end of `data` (defaults: `forecast.horizon`, `forecast.n_sim`).

        Raises:
            ConfigError: invalid horizon / `n_sim`, an unusable scenario, covariates without a forecaster.
            NumericalError: non-finite filter or forecast output, or `n_sim * horizon * M > 2e8`.
            PersistenceError: outputs cannot be written.
        """
        config, spec = self.config, fit.spec
        space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
        forecaster = covariate_forecaster(
            config, self.config_dir, data, None if scenario is None else Path(scenario)
        )
        mc = MonteCarloForecaster(
            spec,
            space,
            engine_for(config, spec.engine),
            forecaster,
            config.forecast,
            clock_options=config.data.clock_options,
        )
        result = mc.forecast(
            fit,
            data,
            config.forecast.horizon if horizon is None else horizon,
            config.forecast.n_sim if n_sim is None else n_sim,
            config.project.seed,
        )
        if write:
            write_forecast(result, self.out_dir, samples=samples, snapshot_ids=data.snapshot_ids)
        get_logger("regimekit.forecast").info(
            "forecast_done", horizon=int(result.horizon_times.shape[0]), seed=result.seed
        )
        return result

    # --- simulate ----------------------------------------------------------------------------------

    def simulate(
        self,
        params: str = "config",
        *,
        model: FitResult | Path | str | None = None,
        steps: int | None = None,
        n_paths: int = 1,
        data: PreparedDataset | None = None,
        write: bool = False,
    ) -> SimulationOutput:
        """Simulate `n_paths` paths of `steps` steps from the configured or a fitted (`params="model"`) model.

        `dt` and the covariates come from the prepared dataset (`data`, default `fetch()`) and cycle
        through its rows when `steps` exceeds its length; `time` extends the dataset's times with their
        median spacing.

        Raises:
            ConfigError: unknown `params`, `params="model"` without `model`, an explicit-engine model,
                or non-positive `steps` / `n_paths`.
            DataError: preparation of the data failed.
        """
        check_simulate_args(params, steps, n_paths)
        dataset = data if data is not None else self.fetch()
        if params == "model":
            if model is None:
                raise ConfigError("--params model requires --model DIR", path="--model")
            fit = model if isinstance(model, FitResult) else ModelStore.load(model, FinanceConfig)
            spec, theta = fit.spec, fit.params
        else:
            spec = build_model_spec(self.config, dataset)
            manager = ParamManager(spec.param_specs)
            theta = manager.unflatten(manager.init_u())
        if spec.engine == "explicit":
            raise ConfigError(
                "simulate supports the expanded engine only (the explicit-duration engine has no A_t)",
                path="model.engine",
            )
        space = build_expanded(spec)

        t0 = int(dataset.times.shape[0])
        n_steps = t0 if steps is None else steps
        idx = np.arange(n_steps) % t0

        def cycled(x: object) -> object:
            return None if x is None else jnp.asarray(np.asarray(x)[idx])

        times = np.asarray(dataset.times).astype("datetime64[ns]")
        if n_steps <= t0:
            sim_times = times[:n_steps]
        else:
            spacing = np.median(np.diff(times)) if t0 > 1 else np.timedelta64(1, "D")
            sim_times = times[-1] + spacing.astype("m8[ns]") * np.arange(1, n_steps - t0 + 1)
            sim_times = np.concatenate([times, sim_times])
        result = simulate_paths(
            key_for(self.config.project.seed, PURPOSE_SIMULATE),
            theta,
            spec,
            space,
            jnp.asarray(np.asarray(dataset.dt)[idx]),
            cycled(dataset.cov_transition),  # type: ignore[arg-type]
            cycled(dataset.cov_duration),  # type: ignore[arg-type]
            cycled(dataset.cov_emission),  # type: ignore[arg-type]
            n_paths,
        )
        regimes = np.asarray(result.regimes)
        output = SimulationOutput(
            times=sim_times,
            regimes=regimes,
            phases=np.asarray(space.index.phase_of)[np.asarray(result.states)],
            returns=np.asarray(result.returns),
            segments=segments_of(regimes),
        )
        if write:
            write_simulation(
                self.out_dir,
                times=output.times,
                regimes=output.regimes,
                phases=output.phases,
                returns=output.returns,
                segments=output.segments,
            )
        get_logger("regimekit.simulate").info(
            "simulate_done", steps=n_steps, n_paths=n_paths, seed=self.config.project.seed
        )
        return output

    # --- backtest / compare ------------------------------------------------------------------------

    def _run_backtest(self, fetched: FetchedInputs, name: str) -> tuple[BacktestResult, dict[str, Any], str]:
        """Run the walk-forward backtest; returns `(result, meta, report markdown)`."""
        config = self.config
        obs, covs = fetched.obs, fetched.covs
        grid = Preparer(config.data).fit(obs, covs, train_end=obs.require_times()[-1]).transform(obs, covs)
        spec = build_model_spec(config, grid)
        n_free = ParamManager(spec.param_specs).n_free
        cfg_strategy = config.evaluation.strategy
        strategy = (
            None
            if cfg_strategy is None
            else make_strategy(cfg_strategy, grid.nominal_dt, int(grid.observations.shape[1]))
        )
        last: dict[str, Any] = {}

        def on_fold(fold: object, fitted: FitResult, train: PreparedDataset) -> None:
            last["fit"], last["train"] = fitted, train

        started = time.perf_counter()
        result = run_backtest(config, obs, covs, config.project.seed, strategy, on_fold)
        wall = time.perf_counter() - started
        diagnostics = None
        if "fit" in last:
            space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
            try:
                diagnostics = diagnose(
                    last["fit"], spec, space, engine_for(config, spec.engine), last["train"]
                )
            except RegimekitError as exc:  # the backtest itself succeeded: report without diagnostics
                get_logger("regimekit.backtest").warning("diagnostics_unavailable", error=str(exc))
        meta: dict[str, Any] = {
            "config": name,
            "seed": config.project.seed,
            "snapshot_ids": [fetched.obs.snapshot_id, *(c.snapshot_id for c in fetched.covs)],
            "n_free": n_free,
            "versions": versions(),
            "wall_s": wall,
        }
        return result, meta, build_report(name, config, result, meta, diagnostics)

    def _write_backtest(
        self, result: BacktestResult, meta: dict[str, Any], report: str, out_dir: Path
    ) -> None:
        write_backtest(result, out_dir, meta)
        _write_text(out_dir / "report.md", report, "the report")

    def backtest(self, data: FetchedInputs | None = None, *, write: bool = False) -> BacktestResult:
        """Walk-forward backtest on raw inputs `data` (default: `fetch_inputs()`).

        `write=True` writes metrics, folds, strategy trace, `backtest.json` and `report.md` to `out_dir`.
        The run's metadata is kept in `last_backtest_meta`.

        Raises:
            ConfigError, DataError, ProviderError, NumericalError: from the fetch and the backtest.
            PersistenceError: outputs cannot be written.
        """
        fetched = data if data is not None else self.fetch_inputs()
        result, meta, report = self._run_backtest(fetched, self.config.project.name)
        self.last_backtest_meta = meta
        if write:
            self._write_backtest(result, meta, report, self.out_dir)
        get_logger("regimekit.backtest").info(
            "backtest_done", folds=len(result.folds), wall_s=round(meta["wall_s"], 1)
        )
        return result

    @classmethod
    def compare(
        cls,
        projects: Sequence[Project | FinanceConfig | Path | str],
        *,
        allow_mixed: bool = False,
        reuse: bool = False,
        metrics: Sequence[str] | None = None,
        offline: bool = False,
        overrides: Mapping[str, object] | None = None,
        out_dir: Path | str | None = None,
        write: bool = False,
    ) -> pd.DataFrame:
        """Backtest every configuration (or reuse its stored result) and tabulate them.

        Entries are projects, configs, or config paths (named by file stem; `overrides` apply to paths
        only). `write=True` also stores each backtest in its project's `out_dir` and writes
        `compare.parquet` / `compare.md` to `out_dir` (default `runs/compare`).

        Raises:
            DataError: the configurations use different snapshots and `allow_mixed` is false.
            ConfigError, ProviderError, NumericalError, PersistenceError: from loading, fetching, running.
        """
        items: list[tuple[str, Project]] = []
        for entry in projects:
            if isinstance(entry, Project):
                items.append((entry.config.project.name, entry))
            elif isinstance(entry, FinanceConfig):
                items.append((entry.project.name, cls(entry, offline=offline)))
            else:
                path = Path(entry)
                items.append((path.stem, cls.from_yaml(path, overrides, offline=offline)))
        names = [n for n, _ in items]
        if len(set(names)) != len(names):
            items = [(f"{n}_{i}", p) for i, (n, p) in enumerate(items)]
        metric_list = list(metrics) if metrics else None

        plan: list[tuple[str, Project, Path, FetchedInputs | None]] = []
        snapshots: dict[str, list[str]] = {}
        for name, proj in items:
            run_dir = proj.out_dir
            if any(run_dir == other[2] for other in plan):
                raise ConfigError(
                    f"{name} shares project.output_dir {run_dir} with another configuration; "
                    "give each configuration its own output directory",
                    path="project.output_dir",
                )
            fetched: FetchedInputs | None = None
            if reuse and (run_dir / BACKTEST_META).exists():
                snapshots[name] = list(read_backtest(run_dir)[1]["snapshot_ids"])
            else:
                fetched = proj.fetch_inputs(offline=offline or proj.offline)
                snapshots[name] = [fetched.obs.snapshot_id, *(c.snapshot_id for c in fetched.covs)]
            plan.append((name, proj, run_dir, fetched))
        check_snapshots(snapshots, allow_mixed)

        entries: list[CompareEntry] = []
        for name, proj, run_dir, fetched in plan:
            if fetched is None:
                result, meta = read_backtest(run_dir)
            else:
                result, meta, report = proj._run_backtest(fetched, name)
                if write:
                    proj._write_backtest(result, meta, report, run_dir)
            entries.append(CompareEntry(name, result, meta))

        table = comparison_table(entries, metric_list)
        if write:
            target = Path(out_dir) if out_dir is not None else Path("runs") / "compare"
            try:
                target.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(table, preserve_index=False), target / "compare.parquet")
                (target / "compare.md").write_text(render_markdown(table, snapshots), encoding="utf-8")
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the comparison to {target}: {exc}", path=str(target)
                ) from exc
        return table

    # --- selftest ----------------------------------------------------------------------------------

    @staticmethod
    def selftest(*, seed: int = 0, full: bool = False, device: str = "cpu") -> SelftestResult:
        """The `regimekit selftest` run (see `estimation.selftest`)."""
        return run_selftest(seed=seed, full=full, device=device)
