"""`Project`: the domain-neutral Python entry point that mirrors every CLI command (E8-T02, E10-T07b).

A `Project` holds a resolved `ProjectConfig` and the directory its relative paths resolve against. Every
method returns its result object and writes nothing unless called with `write=True`; the CLI
(`cli/app.py`) is a thin wrapper that passes `write=True`. Data enters through `load_data`, which reads the
configured source (`data.source`). An application subclasses `Project` and overrides `config_cls`,
`_load_raw`, `_hooks`, `_report_sections` and `_extensions` (the finance application: `FinanceProject`).
Importing this module triggers no network access and no file writes.
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, Self, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic

from hsmmkit.config.schema import BacktestCfg, ProjectConfig
from hsmmkit.config.yaml_io import dump_resolved, load_config
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.data.contracts import CovariateSeries, ObservationSet, PreparedDataset
from hsmmkit.data.prepare import Preparer, prep_stats_to_json
from hsmmkit.data.sources import load_data as load_source_data
from hsmmkit.engine.batch import batched_loglik
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.estimation import fit as fit_model
from hsmmkit.estimation.compare import compare_estimators, render_estimator_table
from hsmmkit.estimation.report import diagnose
from hsmmkit.estimation.results import FitResult
from hsmmkit.estimation.selftest import SelftestResult, run_selftest
from hsmmkit.evaluation.backtest import OriginHook, run_backtest
from hsmmkit.evaluation.compare import CompareEntry, check_snapshots, comparison_table, render_markdown
from hsmmkit.evaluation.report import build_report, versions
from hsmmkit.evaluation.results import BacktestResult
from hsmmkit.inference.covariate_forecasters import CovariateForecaster, make_forecaster
from hsmmkit.inference.decode import decode as decode_posterior
from hsmmkit.inference.forecast import MonteCarloForecaster
from hsmmkit.inference.results import ForecastResult, PosteriorResult
from hsmmkit.inference.simulate import segments_of, simulate_paths
from hsmmkit.model.build import build_model_spec, required_own_lags, required_series_support
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.online.session import OnlineSession
from hsmmkit.persistence.results import (
    BACKTEST_META,
    read_backtest,
    write_backtest,
    write_forecast,
    write_posterior,
    write_prepared,
    write_simulation,
)
from hsmmkit.persistence.store import ModelStore
from hsmmkit.selection.bootstrap import (
    BootstrapResult,
    bootstrap_se,
    render_bootstrap_section,
    replace_bootstrap_section,
)
from hsmmkit.selection.cv import (
    CVResult,
    cross_validate,
    make_splitter,
    render_cv_section,
    replace_cv_section,
)
from hsmmkit.selection.grid import render_grid_markdown, run_grid
from hsmmkit.selection.ppc import ppc
from hsmmkit.selection.pseudo_residuals import pseudo_residuals
from hsmmkit.statespace.builders import build_expanded, build_explicit
from hsmmkit.utils.clock import CALENDAR_DAYS_PER_YEAR
from hsmmkit.utils.errors import ConfigError, HsmmkitError, PersistenceError
from hsmmkit.utils.logging import get_logger
from hsmmkit.utils.rng import PURPOSE_SIMULATE, key_for

__all__ = [
    "FetchedInputs",
    "Project",
    "ScoreResult",
    "SimulationOutput",
    "apply_overrides",
    "check_simulate_args",
]


class FetchedInputs(NamedTuple):
    """Raw inputs of a run: observations, covariate series and what the source loader left behind."""

    obs: ObservationSet
    covs: list[CovariateSeries]
    context: Any = None  # what the loader left behind (an application's context)


@dataclass(frozen=True)
class ScoreResult:
    """`Project.score`: the total log-likelihood and the per-sequence table `(id, T, loglik)`."""

    loglik: float
    per_sequence: pd.DataFrame


@dataclass(frozen=True)
class SimulationOutput:
    """`Project.simulate`: `n_paths` simulated paths of `steps` steps (what `simulation.parquet` holds)."""

    times: np.ndarray
    regimes: np.ndarray
    phases: np.ndarray
    values: np.ndarray
    segments: list[list[tuple[int, int, int]]]


def apply_overrides(config: ProjectConfig, overrides: Mapping[str, object]) -> None:
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


def summary_lines(config: ProjectConfig) -> list[str]:
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


def _train_end(obs: ObservationSet, backtest: BacktestCfg) -> Any:  # noqa: ANN401  (a timestamp or a step index)
    """End of the initial training window (the standardization window of `load_data`).

    First timestamp + `initial_train` years (or the row given by `initial_train_steps`), capped at the last;
    time-less data gives a step index.
    """
    if obs.times is None:
        n = int(obs.values.shape[0])
        steps = backtest.initial_train_steps
        return n - 1 if steps is None else min(steps, n) - 1
    times = obs.require_times()
    last: np.datetime64 = times[-1]
    if backtest.initial_train_steps is not None:
        return cast(np.datetime64, times[min(backtest.initial_train_steps, times.shape[0]) - 1])
    assert backtest.initial_train is not None
    seconds = round(backtest.initial_train * CALENDAR_DAYS_PER_YEAR * 86_400)
    end: np.datetime64 = times[0] + np.timedelta64(seconds, "s")
    return min(end, last)


def engine_for(config: ProjectConfig, kind: str) -> ExpandedHMMEngine | ExplicitDurationEngine:
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
    config: ProjectConfig, config_dir: Path, dataset: PreparedDataset, scenario: Path | None
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
    """A hsmmkit project: configuration plus the operations of the CLI as methods.

    Args:
        config: The validated configuration (apply overrides before constructing, or use `from_yaml`).
        config_dir: Directory relative paths (data files, scenario CSVs) resolve against
            (default: the current directory).
        out_dir: Where `write=True` puts files (default: `config.project.output_dir`).
        offline: Replay the cache without network access.

    Attributes:
        last_fetched: The raw inputs of the last `load_data` / `load_inputs`.
        last_data: The prepared dataset of the last `load_data` / `fit`.
    """

    config_cls: ClassVar[type[ProjectConfig]] = ProjectConfig

    def __init__(
        self,
        config: ProjectConfig,
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
        self.last_fetched: FetchedInputs | None = None
        self.last_data: PreparedDataset | SequenceBatch | None = None
        self.last_backtest_meta: dict[str, Any] | None = None

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        overrides: Mapping[str, object] | None = None,
        *,
        config_cls: type[ProjectConfig] | None = None,
        out_dir: Path | str | None = None,
        offline: bool = False,
    ) -> Self:
        """Load `path`, apply dotted `overrides` (`{"project.seed": 7}`), and wrap the result."""
        config = load_config(path, config_cls or cls.config_cls)
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

    # --- data --------------------------------------------------------------------------------------

    def _load_raw(self, obj: Any, offline: bool) -> FetchedInputs:  # noqa: ANN401
        """Read the observations and covariates of `data.source` (applications override this)."""
        if offline:
            get_logger("hsmmkit.data").debug("offline_ignored", source=self.config.data.source.kind)
        obs, covs = load_source_data(self.config.data, obj=obj)
        return FetchedInputs(obs, covs)

    def load_inputs(self, obj: Any = None, *, offline: bool | None = None) -> FetchedInputs:  # noqa: ANN401
        """The raw inputs of `data.source` (no preparation, no writes); `obj` for numpy/pandas/xarray."""
        offline = self.offline if offline is None else offline
        self.last_fetched = self._load_raw(obj, offline)
        return self.last_fetched

    def load_data(
        self,
        obj: object = None,
        *,
        offline: bool | None = None,
        write: bool = False,
    ) -> PreparedDataset:
        """Read `data.source` and prepare it (`Preparer`, standardization window = the initial train window).

        `write=True` writes `prepared.parquet`, `prep_stats.json` and `provenance.json` (the snapshot ids).

        Raises:
            ConfigError, DataError: from the source and the preparation.
            PersistenceError: the output directory cannot be written.
        """
        fetched = self.load_inputs(obj, offline=offline)
        obs, covs = fetched.obs, fetched.covs
        dataset = (
            Preparer(
                self.config.data,
                own_lags=required_own_lags(self.config),
                series_support=required_series_support(self.config),
                n_regimes=self.config.model.n_states,
            )
            .fit(obs, covs, _train_end(obs, self.config.backtest_cfg()))
            .transform(obs, covs)
        )
        if write:
            out = self.out_dir
            try:
                out.mkdir(parents=True, exist_ok=True)
                write_prepared(dataset, out / "prepared.parquet")
                (out / "prep_stats.json").write_text(prep_stats_to_json(dataset.prep_stats), encoding="utf-8")
                (out / "provenance.json").write_text(
                    json.dumps({"snapshot_ids": list(dataset.snapshot_ids)}, indent=2), encoding="utf-8"
                )
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the prepared dataset to {out}: {exc}", path=str(out)
                ) from exc
        self.last_data = dataset
        get_logger("hsmmkit.data").info(
            "load_data_done", out=str(self.out_dir) if write else None, rows=int(dataset.times.shape[0])
        )
        return dataset

    def is_multi_sequence(self) -> bool:
        """Whether `data` describes several independent sequences (`data.sequences` / `sequence_column`)."""
        data = self.config.data
        return data.sequences is not None or data.source.option("sequence_column") is not None

    def load_batch(self, obj: object = None, *, offline: bool | None = None) -> SequenceBatch:
        """Read `data.source` and prepare every sequence (E12-T02): rows grouped by `sequence_column`.

        The standardization statistics are pooled over all sequences; own lags, lagged covariates and
        first-step masks stay inside each sequence (`Preparer.transform_batch`). `data.sequences.group_by`
        gives the initial-distribution group of each sequence.
        """
        fetched = self.load_inputs(obj, offline=offline)
        obs, covs = fetched.obs, fetched.covs
        end = obs.values.shape[0] - 1 if obs.times is None else obs.require_times()[-1]
        batch = (
            Preparer(
                self.config.data,
                own_lags=required_own_lags(self.config),
                series_support=required_series_support(self.config),
                n_regimes=self.config.model.n_states,
            )
            .fit(obs, covs, end)
            .transform_batch(obs, covs)
        )
        self.last_data = batch
        get_logger("hsmmkit.data").info("load_batch_done", sequences=len(batch.sequences))
        return batch

    def needs_network(self) -> bool:
        """Whether loading the data may need the network (then `--offline` forbids loading it again)."""
        return False

    def _hooks(self, grid: PreparedDataset) -> list[OriginHook]:
        """The origin hooks a backtest runs by default (none in the core)."""
        del grid
        return []

    def _report_sections(self) -> list[Callable[[dict[str, Any]], str]]:
        """The report sections contributed by the default hooks (none in the core)."""
        return []

    def _extensions(self) -> dict[str, Any]:
        """Extension payloads stored in a model bundle (none in the core)."""
        return {}

    # --- fit ---------------------------------------------------------------------------------------

    def fit(self, data: PreparedDataset | SequenceBatch | None = None, *, write: bool = False) -> FitResult:
        """MAP-fit the configured model to `data`.

        The default data is `load_batch()` for a multi-sequence config, else `load_data()`.

        `write=True` saves the bundle `<out_dir>/<name>.hsmm` and writes `fit_report.json` and
        `config.resolved.yaml`.

        Raises:
            ConfigError, DataError: from the data loading and the model assembly.
            NumericalError: no restart produced a finite fit.
            PersistenceError: the outputs cannot be written.
        """
        dataset = data
        if dataset is None:
            dataset = self.load_batch() if self.is_multi_sequence() else self.load_data()
        spec = build_model_spec(self.config, dataset)
        result = fit_model(spec, dataset, self.config.estimation, self.config.project.seed)
        self.last_data = dataset
        if write:
            self.validate(write=True)
            self.save(result, self.out_dir / f"{self.config.project.name}.hsmm", dataset)
            self._write_fit_report(result, dataset)
        get_logger("hsmmkit.fit").info(
            "fit_done", loglik=result.loglik, converged=result.converged, n_iter=result.n_iter
        )
        return result

    def score(self, fit: FitResult, data: PreparedDataset | SequenceBatch) -> ScoreResult:
        """The log-likelihood of `data` under `fit`: total and per sequence (E12-T02).

        A single `PreparedDataset` is a one-sequence batch. With per-group initial distributions each
        sequence starts from its group's row; a group the model has not seen falls back (with a warning)
        to the mean of the group rows.

        Raises:
            NumericalError: a non-finite log-likelihood.
        """
        batch = data if isinstance(data, SequenceBatch) else SequenceBatch.of(data)
        spec = fit.spec
        space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
        engine = engine_for(self.config, spec.engine)
        theta = dict(fit.params)
        pb = batch.padded()
        if spec.n_groups > 1:
            known = list(fit.diagnostics.get("initial_groups") or [])
            rows = np.asarray(theta["initial.logits"])
            if len(known) != rows.shape[0]:
                raise ConfigError(
                    "the fit has per-group initial distributions but no group names "
                    "(diagnostics['initial_groups']): it did not come from a batch fit",
                    path="diagnostics.initial_groups",
                )
            theta["initial.logits"] = jnp.asarray(np.vstack([rows, rows.mean(axis=0, keepdims=True)]))
            index = np.full(pb.group_index.shape, rows.shape[0], dtype=np.int32)  # unseen -> the fallback row
            unseen = sorted({s.group for s in batch.sequences if s.group not in known}, key=str)
            if unseen:
                warnings.warn(
                    f"initial-distribution groups {unseen} were not seen in the fit: "
                    "using the mean of the group rows",
                    UserWarning,
                    stacklevel=2,
                )
            for b, seq in enumerate(batch.sequences):
                if seq.group in known:
                    index[b] = known.index(seq.group)
            pb = dataclasses.replace(pb, group_index=jnp.asarray(index))
        per = np.asarray(batched_loglik(engine, theta, spec, space, pb, batch.ids))[: len(batch.sequences)]
        return ScoreResult(
            loglik=float(per.sum()),
            per_sequence=pd.DataFrame(
                {
                    "id": list(batch.ids),
                    "T": [int(t) for t in batch.lengths],
                    "loglik": per.astype(float),
                }
            ),
        )

    def _adequacy_sections(self, fit: FitResult, data: PreparedDataset) -> dict[str, object]:
        """`estimation.diagnostics` sections of the fit report: pseudo-residual and PPC tables (E14-T05)."""
        out: dict[str, object] = {}
        wanted = self.config.estimation.diagnostics
        try:
            if "pseudo_residuals" in wanted:
                res = pseudo_residuals(fit, data, engine_for(self.config, fit.spec.engine))
                out["pseudo_residuals"] = {
                    "kind": res.kind,
                    "table": res.table.to_dict(orient="records"),
                    "notes": list(res.notes),
                }
            if "ppc" in wanted:
                chk = ppc(fit, data, self.config.estimation.ppc_replicates, self.config.project.seed)
                out["ppc"] = {"table": chk.table.to_dict(orient="records")}
        except HsmmkitError as exc:
            get_logger("hsmmkit.fit").warning("adequacy_unavailable", error=str(exc))
        return out

    def _write_fit_report(self, fit: FitResult, data: PreparedDataset | SequenceBatch) -> None:
        spec = fit.spec
        space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
        first = data.sequences[0] if isinstance(data, SequenceBatch) else data
        try:
            if isinstance(data, SequenceBatch):  # the batch fit already reports per sequence
                diagnostics = dict(fit.diagnostics)
            else:
                diagnostics = diagnose(fit, spec, space, engine_for(self.config, spec.engine), data)
        except HsmmkitError as exc:  # the fit itself succeeded: report without diagnostics
            get_logger("hsmmkit.fit").warning("diagnostics_unavailable", error=str(exc))
            diagnostics = {}
        if not isinstance(data, SequenceBatch):
            diagnostics.update(self._adequacy_sections(fit, data))
        report = {
            "name": self.config.project.name,
            "seed": self.config.project.seed,
            "converged": fit.converged,
            "n_iter": fit.n_iter,
            "objective": fit.objective,
            "loglik": fit.loglik,
            "n_free": ParamManager(spec.param_specs).n_free,
            "snapshot_ids": list(first.snapshot_ids),
            "restarts": fit.restarts.to_dict(orient="records"),
            "diagnostics": diagnostics,
        }
        _write_text(
            self.out_dir / "fit_report.json",
            json.dumps(report, indent=2, default=_json_default),
            "fit_report.json",
        )

    # --- save / load -------------------------------------------------------------------------------

    def save(
        self, fit: FitResult, path: Path | str, data: PreparedDataset | SequenceBatch | None = None
    ) -> Path:
        """Write `fit` as a model bundle; `data` defaults to the dataset of the last `fit` / `load_data`."""
        dataset = data if data is not None else self.last_data
        if dataset is None:
            raise ConfigError("saving a model needs the dataset it was fitted on; pass data=", path="data")
        return ModelStore.save(fit, path, self.config, dataset, extensions=self._extensions())

    @classmethod
    def load(cls, path: Path | str) -> FitResult:
        """Read a model bundle back as a `FitResult`."""
        return ModelStore.load(path, cls.config_cls)

    # --- online ------------------------------------------------------------------------------------

    def online(
        self,
        fit: FitResult,
        *,
        initial: PreparedDataset | None = None,
        lag: int | None = None,
        nominal_dt: float | None = None,
        group: str | None = None,
    ) -> OnlineSession:
        """A streaming session of `fit` under this project's configuration (`push`, `push_many`, `forecast`).

        Args:
            fit: the fitted model (`engine: expanded`).
            initial: prepared history for a warm start.
            lag: fixed-lag smoothing delay (default `online.lag`).
            nominal_dt: grid step for a cold start under a calendar clock.
            group: initial-distribution group of the stream.

        Raises:
            ConfigError: an unsupported model or configuration (explicit engine, own-lag emissions, ...).
        """
        return OnlineSession(fit, self.config, initial=initial, lag=lag, nominal_dt=nominal_dt, group=group)

    # --- decode / forecast -------------------------------------------------------------------------

    def decode(
        self,
        fit: FitResult,
        data: PreparedDataset,
        *,
        write: bool = False,
        augmented: bool = False,
        samples: int = 0,
    ) -> PosteriorResult:
        """Filtered / smoothed posteriors, Viterbi path, segments and IC of `fit` on `data`.

        The engine is built from `config.estimation` (`chunk_size`/`parallel_scan`) so decoding matches
        how the model was fit.

        Raises:
            NumericalError: the engine's log-likelihood is non-finite.
            PersistenceError: outputs cannot be written.
        """
        result = decode_posterior(
            fit,
            data,
            engine=engine_for(self.config, fit.spec.engine),
            samples=samples,
            seed=self.config.project.seed,
        )
        if write:
            write_posterior(result, self.out_dir, augmented=augmented)
        get_logger("hsmmkit.decode").info("decode_done", rows=int(data.times.shape[0]), loglik=result.loglik)
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
        get_logger("hsmmkit.forecast").info(
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

        `dt` and the covariates come from the prepared dataset (`data`, default `load_data()`) and cycle
        through its rows when `steps` exceeds its length; `time` extends the dataset's times with their
        median spacing.

        Raises:
            ConfigError: unknown `params`, `params="model"` without `model`, an explicit-engine model,
                or non-positive `steps` / `n_paths`.
            DataError: preparation of the data failed.
        """
        check_simulate_args(params, steps, n_paths)
        dataset = data if data is not None else self.load_data()
        if params == "model":
            if model is None:
                raise ConfigError("--params model requires --model DIR", path="--model")
            fit = model if isinstance(model, FitResult) else ModelStore.load(model, self.config_cls)
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
            values=np.asarray(result.values),
            segments=segments_of(regimes),
        )
        if write:
            write_simulation(
                self.out_dir,
                times=output.times,
                regimes=output.regimes,
                phases=output.phases,
                values=output.values,
                segments=output.segments,
            )
        get_logger("hsmmkit.simulate").info(
            "simulate_done", steps=n_steps, n_paths=n_paths, seed=self.config.project.seed
        )
        return output

    # --- backtest / compare ------------------------------------------------------------------------

    def _run_backtest(
        self, fetched: FetchedInputs, name: str, extra_hooks: Sequence[OriginHook] = ()
    ) -> tuple[BacktestResult, dict[str, Any], str]:
        """Run the walk-forward backtest; returns `(result, meta, report markdown)`."""
        config = self.config
        obs, covs = fetched.obs, fetched.covs
        grid = (
            Preparer(
                config.data,
                own_lags=required_own_lags(config),
                series_support=required_series_support(config),
                n_regimes=config.model.n_states,
            )
            .fit(obs, covs, train_end=obs.require_times()[-1])
            .transform(obs, covs)
        )
        spec = build_model_spec(config, grid)
        n_free = ParamManager(spec.param_specs).n_free
        hooks = [*self._hooks(grid), *extra_hooks]
        last: dict[str, Any] = {}

        def on_fold(fold: object, fitted: FitResult, train: PreparedDataset) -> None:
            last["fit"], last["train"] = fitted, train

        started = time.perf_counter()
        result = run_backtest(config, lambda: (obs, covs), config.project.seed, hooks, on_fold)
        wall = time.perf_counter() - started
        diagnostics = None
        if "fit" in last:
            space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
            try:
                diagnostics = diagnose(
                    last["fit"], spec, space, engine_for(config, spec.engine), last["train"]
                )
            except HsmmkitError as exc:  # the backtest itself succeeded: report without diagnostics
                get_logger("hsmmkit.backtest").warning("diagnostics_unavailable", error=str(exc))
        meta: dict[str, Any] = {
            "config": name,
            "seed": config.project.seed,
            "snapshot_ids": [fetched.obs.snapshot_id, *(c.snapshot_id for c in fetched.covs)],
            "n_free": n_free,
            "versions": versions(),
            "wall_s": wall,
        }
        return result, meta, build_report(name, config, result, meta, diagnostics, self._report_sections())

    def _write_backtest(
        self, result: BacktestResult, meta: dict[str, Any], report: str, out_dir: Path
    ) -> None:
        write_backtest(result, out_dir, meta)
        _write_text(out_dir / "report.md", report, "the report")

    def cross_validate(self, *, write: bool = False) -> CVResult:
        """Cross-validate the configured model with `evaluation.cv.splitter` (E14-T01).

        `write=True` writes `cv_scores.parquet`, `cv_summary.parquet`, `cv_folds.parquet` and a
        cross-validation section of `report.md` to the output directory.

        Raises:
            ConfigError: `evaluation.cv` is absent, or the data do not suit the splitter.
            PersistenceError: the outputs cannot be written.
        """
        cfg = self.config
        fetched = self.load_inputs(offline=self.offline)
        splitter = make_splitter(cfg)

        def preparer() -> Preparer:
            return Preparer(
                cfg.data,
                own_lags=required_own_lags(cfg),
                series_support=required_series_support(cfg),
                n_regimes=cfg.model.n_states,
            )

        if splitter.name == "walk_forward":  # the backtest itself: same grid, hooks and results
            bt, _, _ = self._run_backtest(fetched, cfg.project.name)
            scores = bt.metrics.assign(sequence="0")[["fold", "sequence", "horizon", "metric", "value"]]
            result = CVResult(bt.folds, scores, bt.summary, bt.extras)
        else:
            result = cross_validate(cfg, (fetched.obs, fetched.covs), splitter, cfg.project.seed, preparer)
        if write:
            target = self.out_dir
            try:
                target.mkdir(parents=True, exist_ok=True)
                for name, frame in (
                    ("cv_scores", result.scores),
                    ("cv_summary", result.summary),
                    ("cv_folds", result.folds),
                ):
                    pq.write_table(
                        pa.Table.from_pandas(frame, preserve_index=False), target / f"{name}.parquet"
                    )
                report = target / "report.md"
                text = report.read_text(encoding="utf-8") if report.exists() else ""
                _write_text(
                    report, replace_cv_section(text, render_cv_section(result, splitter.name)), "the report"
                )
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the CV results to {target}: {exc}", path=str(target)
                ) from exc
        return result

    def backtest(
        self, data: FetchedInputs | None = None, hooks: Sequence[OriginHook] = (), *, write: bool = False
    ) -> BacktestResult:
        """Walk-forward backtest on raw inputs `data` (default: `load_inputs()`) plus `hooks`.

        `write=True` writes metrics, folds, the hooks' frames, `backtest.json` and `report.md` to `out_dir`.
        The run's metadata is kept in `last_backtest_meta`.

        Raises:
            ConfigError, DataError, NumericalError: from the data loading and the backtest.
            PersistenceError: outputs cannot be written.
        """
        fetched = data if data is not None else self.load_inputs()
        result, meta, report = self._run_backtest(fetched, self.config.project.name, hooks)
        self.last_backtest_meta = meta
        if write:
            self._write_backtest(result, meta, report, self.out_dir)
        get_logger("hsmmkit.backtest").info(
            "backtest_done", folds=len(result.folds), wall_s=round(meta["wall_s"], 1)
        )
        return result

    @classmethod
    def compare(
        cls,
        projects: Sequence[Project | ProjectConfig | Path | str],
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
            ConfigError, NumericalError, PersistenceError: from loading the data and running.
        """
        items: list[tuple[str, Project]] = []
        for entry in projects:
            if isinstance(entry, Project):
                items.append((entry.config.project.name, entry))
            elif isinstance(entry, ProjectConfig):
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
                fetched = proj.load_inputs(offline=offline or proj.offline)
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

    def compare_estimators(
        self,
        estimators: Sequence[str],
        data: PreparedDataset | SequenceBatch | None = None,
        *,
        write: bool = False,
        out_dir: Path | str | None = None,
    ) -> pd.DataFrame:
        """Fit the configured model with each estimator of `estimators` (same data, seed); tabulate the fits.

        See `estimation.compare.compare_estimators` for the columns. `write=True` writes `compare.parquet` and
        `compare.md` to `out_dir` (default `runs/compare`).

        Raises:
            ConfigError: unknown estimator names, or an estimator that cannot fit the model.
            PersistenceError: the outputs cannot be written.
        """
        dataset = data
        if dataset is None:
            dataset = self.load_batch() if self.is_multi_sequence() else self.load_data()
        spec = build_model_spec(self.config, dataset)
        table, _ = compare_estimators(
            spec, dataset, self.config.estimation, self.config.project.seed, estimators
        )
        if write:
            target = Path(out_dir) if out_dir is not None else Path("runs") / "compare"
            try:
                target.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(table, preserve_index=False), target / "compare.parquet")
                (target / "compare.md").write_text(render_estimator_table(table), encoding="utf-8")
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the comparison to {target}: {exc}", path=str(target)
                ) from exc
        return table

    def compare_grid(self, *, write: bool = False, out_dir: Path | str | None = None) -> pd.DataFrame:
        """Grid model selection over `evaluation.grid` (E14-T02): one fit and score per cell.

        Returns the table (`selection.grid.COLUMNS`); `write=True` writes `grid.parquet` and `grid.md` to
        `out_dir` (default `runs/compare`).

        Raises:
            ConfigError: `evaluation.grid` is absent or empty.
            PersistenceError: the outputs cannot be written.
        """
        cfg = self.config
        fetched = self.load_inputs(offline=self.offline)
        table = run_grid(cfg, fetched.obs, fetched.covs, cfg.project.seed, Preparer)
        if write:
            target = Path(out_dir) if out_dir is not None else Path("runs") / "compare"
            try:
                target.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(table, preserve_index=False), target / "grid.parquet")
                (target / "grid.md").write_text(render_grid_markdown(table), encoding="utf-8")
            except OSError as exc:
                raise PersistenceError(f"cannot write the grid to {target}: {exc}", path=str(target)) from exc
        return table

    def bootstrap(
        self,
        fit: FitResult,
        data: PreparedDataset | SequenceBatch,
        *,
        n_boot: int = 200,
        refit_from: Literal["mle", "init"] = "mle",
        workers: int = 1,
        write: bool = False,
    ) -> BootstrapResult:
        """Parametric-bootstrap standard errors of `fit` (E14-T04) with the configured estimator settings.

        `write=True` writes `bootstrap.parquet` and replaces the `## Bootstrap` section of `report.md` (and
        writes `bootstrap.md`) in the output directory.

        Raises:
            ConfigError: unsupported estimator / emission, or too few accepted refits.
            PersistenceError: the outputs cannot be written.
        """
        result = bootstrap_se(
            fit,
            data,
            n_boot,
            self.config.project.seed,
            refit_from=refit_from,
            estimation=self.config.estimation.model_copy(update={"method": fit.estimator}),
            workers=workers,
        )
        if write:
            target = self.out_dir
            section = render_bootstrap_section(result)
            try:
                target.mkdir(parents=True, exist_ok=True)
                pq.write_table(
                    pa.Table.from_pandas(result.table, preserve_index=False), target / "bootstrap.parquet"
                )
                (target / "bootstrap.md").write_text(section, encoding="utf-8")
                report = target / "report.md"
                if report.exists():
                    text = report.read_text(encoding="utf-8")
                    report.write_text(replace_bootstrap_section(text, section), encoding="utf-8")
            except OSError as exc:
                raise PersistenceError(
                    f"cannot write the bootstrap to {target}: {exc}", path=str(target)
                ) from exc
        return result

    # --- selftest ----------------------------------------------------------------------------------

    @staticmethod
    def selftest(*, seed: int = 0, full: bool = False, device: str = "cpu") -> SelftestResult:
        """The `hsmmkit selftest` run (see `estimation.selftest`)."""
        return run_selftest(seed=seed, full=full, device=device)
