"""Typer CLI entry point (further commands are added by their own tickets)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import typer

from regimekit.config.schema import DataSection, PriceSource, ProjectConfig
from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config
from regimekit.data.cache import Cache, CachedProvider, SnapshotPolicy
from regimekit.data.contracts import CovariateSeries, ObservationSet, PreparedDataset
from regimekit.data.prepare import Preparer, PrepDiagnostics, prep_stats_to_json
from regimekit.data.quality import build_quality_report
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.engine.explicit_duration import ExplicitDurationEngine
from regimekit.estimation.report import diagnose
from regimekit.estimation.results import FitResult
from regimekit.evaluation.backtest import run_backtest
from regimekit.evaluation.compare import CompareEntry, check_snapshots, comparison_table, render_markdown
from regimekit.evaluation.report import build_report, versions
from regimekit.evaluation.results import BacktestResult
from regimekit.evaluation.strategies import make_strategy
from regimekit.finance.providers import get_provider
from regimekit.inference.covariate_forecasters import CovariateForecaster, make_forecaster
from regimekit.inference.decode import decode
from regimekit.inference.forecast import MonteCarloForecaster
from regimekit.inference.simulate import segments_of, simulate_paths
from regimekit.model.build import build_model_spec
from regimekit.model.params.manager import ParamManager
from regimekit.persistence.results import (
    BACKTEST_META,
    read_backtest,
    read_prepared,
    write_backtest,
    write_forecast,
    write_posterior,
    write_prepared,
    write_simulation,
)
from regimekit.persistence.temp_model import load_temp_model
from regimekit.statespace.builders import build_expanded, build_explicit
from regimekit.utils.clock import CALENDAR_DAYS_PER_YEAR
from regimekit.utils.errors import EXIT_CODES, ConfigError, DataError, PersistenceError, RegimekitError
from regimekit.utils.logging import configure_logging, get_logger
from regimekit.utils.rng import PURPOSE_SIMULATE, key_for

app = typer.Typer(name="regimekit")


@dataclass
class GlobalOptions:
    """CLI-wide options set on the `regimekit` app itself, read by subcommands."""

    out: Path | None = None
    offline: bool = False
    device: str | None = None
    seed: int | None = None
    log_level: str | None = None


_global_options = GlobalOptions()


@app.callback()
def _main_callback(
    out: Path | None = typer.Option(None, "--out", help="Output directory (overrides project.output_dir)."),
    offline: bool = typer.Option(False, "--offline", help="Do not access the network."),
    device: str | None = typer.Option(None, "--device", help="cpu|gpu|auto (overrides estimation.device)"),
    seed: int | None = typer.Option(None, "--seed", help="Overrides project.seed."),
    log_level: str | None = typer.Option(None, "--log-level", help="Overrides project.log_level."),
) -> None:
    """regimekit: JAX-based Hidden Semi-Markov Model regime detection and forecasting."""
    global _global_options
    _global_options = GlobalOptions(out=out, offline=offline, device=device, seed=seed, log_level=log_level)


def _apply_overrides(config: ProjectConfig, options: GlobalOptions) -> None:
    """Apply `--seed`/`--device` (and any other global overrides) onto `config`, in place.

    `validate_assignment=True` (E1-T02a) re-validates each field on assignment, raising
    `pydantic.ValidationError` for e.g. `--device bogus` -- converted here to `ConfigError` so
    an invalid override surfaces the same way any other invalid config value would (exit 2, path
    in the message), rather than an unhandled pydantic traceback.
    """
    try:
        if options.seed is not None:
            config.project.seed = options.seed
        if options.device is not None:
            config.estimation.device = options.device  # type: ignore[assignment]
        if options.log_level is not None:
            config.project.log_level = options.log_level  # type: ignore[assignment]
    except pydantic.ValidationError as exc:
        first = exc.errors()[0]
        path = ".".join(str(part) for part in first["loc"])
        raise ConfigError(first["msg"], path=path or None) from exc


def _summary_lines(config: ProjectConfig) -> list[str]:
    emission_covariates = config.model.emission.covariates
    entry_points = sorted(
        {ep for cov in config.data.covariates for ep in cov.use_in}
        | ({"transition"} if config.model.transition.covariates else set())
        | ({"duration"} if config.model.duration.covariates else set())
        | ({"emission"} if emission_covariates.drift or emission_covariates.log_vol else set())
    )
    return [
        f"n_states: {config.model.n_states}",
        f"engine: {config.model.engine}",
        f"emission family: {config.model.emission.family}",
        f"duration family: {config.model.duration.family}",
        f"covariate entry points: {', '.join(entry_points) if entry_points else '(none)'}",
        f"warnings: {'; '.join(config.warnings) if config.warnings else '(none)'}",
    ]


@app.command("validate-config")
def validate_config_cmd(config_path: Path) -> None:
    """Validate `config_path`, print a summary, and write the resolved config alongside it."""
    try:
        config = load_config(config_path)
        _apply_overrides(config, _global_options)
        out_dir = _global_options.out or config.project.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_resolved(config, out_dir / "config.resolved.yaml")
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc
    for line in _summary_lines(config):
        typer.echo(line)


@app.command("config-schema")
def config_schema_cmd(out_path: Path) -> None:
    """Write `ProjectConfig`'s JSON-Schema to `out_path`."""
    export_json_schema(out_path)


def _make_provider(name: str, data: DataSection, config_dir: Path) -> object:
    """Instantiate a provider adapter for `fetch`.

    `csv` files are resolved relative to the config file's directory (the schema has no provider
    options); quota-tracking providers keep their quota files in the cache directory. API keys come
    from the environment only (CLAUDE.md §9).
    """
    kwargs: dict[str, object] = {}
    if name == "csv":
        kwargs["root"] = config_dir
    elif name in ("stooq", "tiingo"):
        kwargs["quota_dir"] = Path(data.cache.dir).expanduser()
    return get_provider(name, **kwargs)


def _train_end(obs: ObservationSet, initial_train_years: float) -> np.datetime64:
    """End of the initial training window: first timestamp + `initial_train_years`, capped at the last."""
    seconds = round(initial_train_years * CALENDAR_DAYS_PER_YEAR * 86_400)
    end: np.datetime64 = obs.times[0] + np.timedelta64(seconds, "s")
    last: np.datetime64 = obs.times[-1]
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
        return {"missing_price": diagnostics.missing_price_rows}
    return {
        "missing_price": diagnostics.missing_price_rows,
        "warmup": diagnostics.warmup_rows,
        **{f"covariate:{name}": n for name, n in diagnostics.dropped.items()},
    }


class _Fetched(NamedTuple):
    """Raw inputs of a run: primary prices, covariate series, and the cache/provider they came through."""

    obs: ObservationSet
    covs: list[CovariateSeries]
    cache: Cache
    primary_name: str


def _check_pinned(config: ProjectConfig, offline: bool) -> None:
    data = config.data
    policy = SnapshotPolicy(data.cache.snapshot, data.cache.max_age_days, offline)
    if policy.pinned is not None and data.covariates:
        raise ConfigError(
            "a pinned snapshot id identifies one cache entry and cannot pin covariates; use "
            "`snapshot: latest` with `--offline` to replay a run with covariates",
            path="data.cache.snapshot",
        )


def _fetch_inputs(config: ProjectConfig, config_dir: Path, offline: bool) -> _Fetched:
    """Fetch the primary prices and covariates through the cache (shared by `fetch` and `backtest`).

    Raises:
        ConfigError: a pinned snapshot together with covariates.
        DataError: the primary request returned fewer than two rows.
        ProviderError: a provider failed.
    """
    data = config.data
    _check_pinned(config, offline)
    policy = SnapshotPolicy(data.cache.snapshot, data.cache.max_age_days, offline)
    log = get_logger("regimekit.fetch")
    cache = Cache(data.cache.dir)
    providers: dict[str, CachedProvider] = {}

    def cached(name: str) -> CachedProvider:
        if name not in providers:
            inner = _make_provider(name, data, config_dir)
            providers[name] = CachedProvider(inner, cache, policy)  # type: ignore[arg-type]
            log.info("provider_terms_of_use", provider=name, note=providers[name].tos_note)
        return providers[name]

    p = data.primary
    primary = cached(p.provider)
    obs = primary.fetch_prices(list(p.symbols), p.start, p.end, p.interval, p.adjusted)
    covs = [cached(c.provider).fetch_series(c.series, p.start, p.end) for c in data.covariates]
    if obs.times.shape[0] < 2:
        raise DataError(f"the primary request returned {obs.times.shape[0]} rows", path="data.primary")
    return _Fetched(obs, covs, cache, primary.name)


def run_fetch(config: ProjectConfig, config_dir: Path, out_dir: Path, offline: bool) -> None:
    """Fetch through the cache, prepare, and write the dataset, stats, snapshot ids and quality report.

    The quality report is written even when preparation fails (e.g. `missing: error`); the error is
    then re-raised.

    Raises:
        ConfigError: a pinned `data.cache.snapshot` combined with covariates (one id names one cache
            entry; pinning multi-source runs awaits an owner decision).
        DataError: an empty primary request, or preparation failed.
        ProviderError: a provider failed.
        PersistenceError: the output directory cannot be written (exit 1, CLAUDE.md §3.8).
    """
    data = config.data
    _check_pinned(config, offline)
    log = get_logger("regimekit.fetch")
    outputs = [out_dir / n for n in ("prepared.parquet", "prep_stats.json", "snapshots.json")]
    try:  # a failed run must not leave an earlier run's dataset next to its own report
        for stale in outputs:
            stale.unlink(missing_ok=True)
    except OSError as exc:
        raise PersistenceError(f"cannot clear {out_dir}: {exc}", path=str(out_dir)) from exc
    fetched = _fetch_inputs(config, config_dir, offline)
    obs, covs, p = fetched.obs, fetched.covs, data.primary
    generations = _two_newest_generations(fetched.cache, fetched.primary_name, p)

    preparer = Preparer(data)
    failure: DataError | None = None
    try:
        train_end = _train_end(obs, config.evaluation.backtest.initial_train_years)
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
    try:
        report.write(out_dir)
    except OSError as exc:
        raise PersistenceError(
            f"cannot write the quality report to {out_dir}: {exc}", path=str(out_dir)
        ) from exc
    if failure is not None:
        raise failure

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
            for c, s in zip(data.covariates, covs, strict=True)
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
    log.info("fetch_done", out=str(out_dir), rows=int(dataset.times.shape[0]))


@app.command("fetch")
def fetch_cmd(config_path: Path) -> None:
    """Fetch data through the cache, prepare it, and write the dataset plus a data-quality report."""
    try:
        config = load_config(config_path)
        _apply_overrides(config, _global_options)
        configure_logging(config.project.log_level)
        out_dir = _global_options.out or config.project.output_dir
        run_fetch(config, config_path.resolve().parent, out_dir, _global_options.offline)
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc


def _load_prepared(config: ProjectConfig, config_dir: Path, out_dir: Path, offline: bool) -> PreparedDataset:
    """Read `prepared.parquet` from `out_dir`, running `fetch` first when it is missing (unless offline)."""
    prepared_path = out_dir / "prepared.parquet"
    if not prepared_path.exists():
        if offline:
            raise DataError(f"no prepared dataset at {prepared_path} and --offline forbids fetching one")
        run_fetch(config, config_dir, out_dir, offline)
    return read_prepared(prepared_path)


def _engine_for(config: ProjectConfig, kind: str) -> ExpandedHMMEngine | ExplicitDurationEngine:
    """The engine `kind` names, configured from `config.estimation` (`chunk_size`/`parallel_scan`)."""
    if kind == "explicit":
        return ExplicitDurationEngine(chunk_size=config.estimation.chunk_size)
    return ExpandedHMMEngine(parallel_scan=config.estimation.parallel_scan)


def run_decode(
    config: ProjectConfig,
    config_dir: Path,
    out_dir: Path,
    model_dir: Path,
    offline: bool,
    *,
    augmented: bool = False,
) -> None:
    """Load/refetch `prepared.parquet`, load `--model DIR`, decode and write the outputs.

    Raises:
        DataError: no prepared dataset exists and `--offline` forbids fetching one.
        NumericalError: the engine's log-likelihood is non-finite (exit code 4).
        PersistenceError: `model_dir` is not a `write_temp_model` directory, or outputs can't be written.
    """
    dataset = _load_prepared(config, config_dir, out_dir, offline)
    fit = load_temp_model(model_dir, dataset)
    # `decode`'s own `engine=None` default always builds a default-configured engine; pass one built
    # from `config.estimation` (`chunk_size`/`parallel_scan`) so decoding matches how the model was
    # fit, the same way `estimation.map.fit` builds its engine (independent review finding).
    result = decode(fit, dataset, engine=_engine_for(config, fit.spec.engine))
    write_posterior(result, out_dir, augmented=augmented)
    get_logger("regimekit.decode").info(
        "decode_done", out=str(out_dir), rows=int(dataset.times.shape[0]), loglik=result.loglik
    )


@app.command("decode")
def decode_cmd(
    config_path: Path,
    model: Path = typer.Option(..., "--model", help="Directory from `write_temp_model`."),
    augmented: bool = typer.Option(False, "--augmented", help="Also write augmented.parquet."),
) -> None:
    """Decode a fitted model against `config_path`'s data: filtered/smoothed/Viterbi/segments/IC."""
    try:
        config = load_config(config_path)
        _apply_overrides(config, _global_options)
        configure_logging(config.project.log_level)
        out_dir = _global_options.out or config.project.output_dir
        run_decode(
            config,
            config_path.resolve().parent,
            out_dir,
            model,
            _global_options.offline,
            augmented=augmented,
        )
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc


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


def _covariate_forecaster(
    config: ProjectConfig, config_dir: Path, dataset: PreparedDataset, scenario: Path | None
) -> CovariateForecaster | None:
    """The configured covariate forecaster (`--scenario` forces `scenario`); `None` without covariates."""
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


def run_forecast(
    config: ProjectConfig,
    config_dir: Path,
    out_dir: Path,
    model_dir: Path,
    offline: bool,
    *,
    horizon: int | None = None,
    n_sim: int | None = None,
    scenario: Path | None = None,
    samples: bool = False,
) -> None:
    """Load the prepared data and `--model DIR`, run `MonteCarloForecaster`, write the forecast files.

    Raises:
        DataError: no prepared dataset exists and `--offline` forbids fetching one.
        ConfigError: invalid horizon / `n_sim`, an unusable scenario, covariates without a forecaster.
        NumericalError: non-finite filter or forecast output, or `n_sim * horizon * M > 2e8`.
        PersistenceError: `model_dir` is not a model directory or outputs can't be written.
    """
    dataset = _load_prepared(config, config_dir, out_dir, offline)
    fit = load_temp_model(model_dir, dataset)
    spec = fit.spec
    space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
    forecaster = _covariate_forecaster(config, config_dir, dataset, scenario)
    mc = MonteCarloForecaster(
        spec,
        space,
        _engine_for(config, spec.engine),
        forecaster,
        config.forecast,
        bars_per_day=config.data.bars_per_day,
    )
    result = mc.forecast(
        fit,
        dataset,
        config.forecast.horizon if horizon is None else horizon,
        config.forecast.n_sim if n_sim is None else n_sim,
        config.project.seed,
    )
    write_forecast(result, out_dir, samples=samples, snapshot_ids=dataset.snapshot_ids)
    get_logger("regimekit.forecast").info(
        "forecast_done", out=str(out_dir), horizon=int(result.horizon_times.shape[0]), seed=result.seed
    )


@app.command("forecast")
def forecast_cmd(
    config_path: Path,
    model: Path = typer.Option(..., "--model", help="Directory from `write_temp_model`."),
    horizon: int | None = typer.Option(None, "--horizon", help="Steps ahead (default forecast.horizon)."),
    n_sim: int | None = typer.Option(None, "--n-sim", help="Simulated paths (default forecast.n_sim)."),
    scenario: Path | None = typer.Option(None, "--scenario", help="Scenario CSV of future covariates."),
    samples: bool = typer.Option(False, "--samples", help="Also write forecast_samples.parquet."),
) -> None:
    """Monte-Carlo forecast from the end of `config_path`'s data: regime probabilities, quantiles, moments."""
    try:
        config = load_config(config_path)
        _apply_overrides(config, _global_options)
        configure_logging(config.project.log_level)
        out_dir = _global_options.out or config.project.output_dir
        run_forecast(
            config,
            config_path.resolve().parent,
            out_dir,
            model,
            _global_options.offline,
            horizon=horizon,
            n_sim=n_sim,
            scenario=scenario,
            samples=samples,
        )
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc


def run_simulate(
    config: ProjectConfig,
    config_dir: Path,
    out_dir: Path,
    offline: bool,
    *,
    params: str = "config",
    model_dir: Path | None = None,
    steps: int | None = None,
    n_paths: int = 1,
) -> None:
    """Simulate `n_paths` paths of `steps` steps and write `simulation.parquet` + segments.

    `params="config"` uses the parameters' `init`/`values` from the config (`ParamManager.init_u`);
    `params="model"` loads `model_dir`. `dt` and the covariates come from the prepared dataset and
    cycle through its rows when `steps` exceeds its length; `time` extends the dataset's times with
    their median spacing.

    Raises:
        ConfigError: unknown `params`, `params="model"` without `model_dir`, an explicit-engine model,
            or non-positive `steps` / `n_paths`.
        DataError: no prepared dataset exists and `--offline` forbids fetching one.
    """
    if params not in ("config", "model"):
        raise ConfigError(f"--params must be 'config' or 'model', got {params!r}", path="--params")
    if n_paths < 1 or (steps is not None and steps < 1):
        raise ConfigError("--steps and --n-paths must be >= 1", path="--steps")
    dataset = _load_prepared(config, config_dir, out_dir, offline)
    if params == "model":
        if model_dir is None:
            raise ConfigError("--params model requires --model DIR", path="--model")
        fit = load_temp_model(model_dir, dataset)
        spec, theta = fit.spec, fit.params
    else:
        spec = build_model_spec(config, dataset)
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
        key_for(config.project.seed, PURPOSE_SIMULATE),
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
    write_simulation(
        out_dir,
        times=sim_times,
        regimes=regimes,
        phases=np.asarray(space.index.phase_of)[np.asarray(result.states)],
        returns=np.asarray(result.returns),
        segments=segments_of(regimes),
    )
    get_logger("regimekit.simulate").info(
        "simulate_done", out=str(out_dir), steps=n_steps, n_paths=n_paths, seed=config.project.seed
    )


@app.command("simulate")
def simulate_cmd(
    config_path: Path,
    params: str = typer.Option("config", "--params", help="'config' (init values) or 'model'."),
    model: Path | None = typer.Option(None, "--model", help="Model directory for --params model."),
    steps: int | None = typer.Option(None, "--steps", help="Steps per path (default: dataset length)."),
    n_paths: int = typer.Option(1, "--n-paths", help="Number of simulated paths."),
) -> None:
    """Simulate regime/return paths from the configured (or a fitted) model."""
    try:
        config = load_config(config_path)
        _apply_overrides(config, _global_options)
        configure_logging(config.project.log_level)
        out_dir = _global_options.out or config.project.output_dir
        run_simulate(
            config,
            config_path.resolve().parent,
            out_dir,
            _global_options.offline,
            params=params,
            model_dir=model,
            steps=steps,
            n_paths=n_paths,
        )
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc


def _snapshot_ids(fetched: _Fetched) -> list[str]:
    return [fetched.obs.snapshot_id, *(c.snapshot_id for c in fetched.covs)]


def _execute_backtest(
    config: ProjectConfig, fetched: _Fetched, name: str
) -> tuple[BacktestResult, dict[str, Any], str]:
    """Run the walk-forward backtest; returns `(result, meta, report markdown)`."""
    obs, covs = fetched.obs, fetched.covs
    grid = Preparer(config.data).fit(obs, covs, train_end=obs.times[-1]).transform(obs, covs)
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
            diagnostics = diagnose(last["fit"], spec, space, _engine_for(config, spec.engine), last["train"])
        except RegimekitError as exc:  # the backtest itself succeeded: report without diagnostics
            get_logger("regimekit.backtest").warning("diagnostics_unavailable", error=str(exc))
    meta: dict[str, Any] = {
        "config": name,
        "seed": config.project.seed,
        "snapshot_ids": _snapshot_ids(fetched),
        "n_free": n_free,
        "versions": versions(),
        "wall_s": wall,
    }
    return result, meta, build_report(name, config, result, meta, diagnostics)


def run_backtest_command(config: ProjectConfig, config_dir: Path, out_dir: Path, offline: bool) -> None:
    """Fetch, run `run_backtest`, and write the backtest artifacts (metrics, folds, strategy, report).

    Also written: `backtest_summary.parquet` and `backtest.json` (metadata for `compare --reuse`).

    Raises:
        ConfigError, DataError, ProviderError, NumericalError: from the fetch and the backtest.
        PersistenceError: the output directory cannot be written.
    """
    fetched = _fetch_inputs(config, config_dir, offline)
    result, meta, report = _execute_backtest(config, fetched, config.project.name)
    write_backtest(result, out_dir, meta)
    try:
        (out_dir / "report.md").write_text(report, encoding="utf-8")
    except OSError as exc:
        raise PersistenceError(f"cannot write the report to {out_dir}: {exc}", path=str(out_dir)) from exc
    get_logger("regimekit.backtest").info(
        "backtest_done", out=str(out_dir), folds=len(result.folds), wall_s=round(meta["wall_s"], 1)
    )


@app.command("backtest")
def backtest_cmd(config_path: Path) -> None:
    """Walk-forward backtest: metrics, folds, strategy trace and `report.md` in the output directory."""
    try:
        config = load_config(config_path)
        _apply_overrides(config, _global_options)
        configure_logging(config.project.log_level)
        out_dir = _global_options.out or config.project.output_dir
        run_backtest_command(config, config_path.resolve().parent, out_dir, _global_options.offline)
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc


def run_compare(
    config_paths: list[Path],
    out_dir: Path,
    offline: bool,
    *,
    allow_mixed: bool,
    reuse: bool,
    metrics: list[str] | None,
) -> None:
    """Backtest every config (or reuse its stored result), check the snapshots, write `compare.*`.

    Raises:
        DataError: the configurations use different snapshots and `allow_mixed` is false.
        ConfigError, ProviderError, NumericalError, PersistenceError: from loading, fetching and running.
    """
    names = [p.stem for p in config_paths]
    if len(set(names)) != len(names):
        names = [f"{n}_{i}" for i, n in enumerate(names)]
    if metrics is not None and not metrics:
        metrics = None
    plan: list[tuple[str, ProjectConfig, Path, Path, _Fetched | None]] = []
    snapshots: dict[str, list[str]] = {}
    for name, path in zip(names, config_paths, strict=True):
        config = load_config(path)
        _apply_overrides(config, _global_options)
        run_dir = Path(config.project.output_dir)
        if any(run_dir == other[3] for other in plan):
            raise ConfigError(
                f"{path} shares project.output_dir {run_dir} with another configuration; "
                "give each configuration its own output directory",
                path="project.output_dir",
            )
        fetched: _Fetched | None = None
        if reuse and (run_dir / BACKTEST_META).exists():
            snapshots[name] = list(read_backtest(run_dir)[1]["snapshot_ids"])
        else:
            fetched = _fetch_inputs(config, path.resolve().parent, offline)
            snapshots[name] = _snapshot_ids(fetched)
        plan.append((name, config, path, run_dir, fetched))
    check_snapshots(snapshots, allow_mixed)

    entries: list[CompareEntry] = []
    for name, config, _path, run_dir, fetched in plan:
        if fetched is None:
            result, meta = read_backtest(run_dir)
        else:
            result, meta, report = _execute_backtest(config, fetched, name)
            write_backtest(result, run_dir, meta)
            try:
                (run_dir / "report.md").write_text(report, encoding="utf-8")
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the report to {run_dir}: {exc}", path=str(run_dir)
                ) from exc
        entries.append(CompareEntry(name, result, meta))

    table = comparison_table(entries, metrics)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(table, preserve_index=False), out_dir / "compare.parquet")
        (out_dir / "compare.md").write_text(render_markdown(table, snapshots), encoding="utf-8")
    except OSError as exc:
        raise PersistenceError(f"cannot write the comparison to {out_dir}: {exc}", path=str(out_dir)) from exc


@app.command("compare")
def compare_cmd(
    configs: list[Path] = typer.Argument(..., help="Two or more configuration files."),
    allow_mixed: bool = typer.Option(
        False, "--allow-mixed", help="Compare runs on different data snapshots."
    ),
    metric: list[str] = typer.Option([], "--metric", help="Only these metrics (repeatable)."),
    reuse: bool = typer.Option(
        False, "--reuse", help="Use a config's stored backtest instead of re-running."
    ),
) -> None:
    """Backtest several configs and tabulate them (`compare.parquet`, `compare.md` in the output dir)."""
    try:
        configure_logging(_global_options.log_level or "INFO")
        out_dir = _global_options.out or Path("runs") / "compare"
        run_compare(
            configs,
            out_dir,
            _global_options.offline,
            allow_mixed=allow_mixed,
            reuse=reuse,
            metrics=metric or None,
        )
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_CODES.get(type(exc), 1)) from exc


def main() -> None:  # pragma: no cover
    """Run the regimekit CLI, mapping `RegimekitError` subclasses to their exit codes."""
    try:
        app()
    except RegimekitError as exc:
        typer.echo(str(exc), err=True)
        sys.exit(EXIT_CODES.get(type(exc), 1))


if __name__ == "__main__":  # pragma: no cover
    main()
