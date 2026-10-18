"""Walk-forward evaluation: fold scheduler (E7-T02a) and runner (E7-T02b).

Concept refs: §3.1 FR-E02, §5.4 `BacktestResult`, §5.7 `BacktestCfg`. The scheduler is pure index
arithmetic, the runner (`run_backtest`, below) executes the folds. All indices are rows of the prepared grid
(`PreparedDataset.times` / `dt`, length `T`), with **exclusive** ends (CLAUDE.md §3.7), i.e. the same
`[start, end)` convention as `PreparedDataset.window`. A forecast origin is a row `o` of
`[test_start, test_end)`: the model is fitted on rows `[train_start, train_end)` and forecasts from the
data `[0, o + 1)`; the realized `Y_h` uses rows `o + 1 .. o + h`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from regimekit.data.prepare import Preparer
from regimekit.engine.base import Posterior
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.engine.explicit_duration import ExplicitDurationEngine
from regimekit.estimation.map import fit as fit_map
from regimekit.estimation.objective import canonical_data
from regimekit.evaluation.metrics.predictive import one_step_mixture_weights, pit_histogram
from regimekit.evaluation.metrics.registry import resolve
from regimekit.evaluation.padding import pad_dataset
from regimekit.evaluation.results import BacktestResult
from regimekit.evaluation.strategies import Strategy, check_backtest_admissible, strategy_trace
from regimekit.inference.covariate_forecasters import make_forecaster
from regimekit.inference.decode import decode
from regimekit.inference.forecast import MonteCarloForecaster
from regimekit.model.build import build_model_spec
from regimekit.statespace.builders import build_expanded, build_explicit
from regimekit.utils.clock import future_grid
from regimekit.utils.errors import ConfigError, DataError
from regimekit.utils.logging import get_logger
from regimekit.utils.rng import PURPOSE_BACKTEST_FIT, PURPOSE_BACKTEST_FORECAST, derive_seed

if TYPE_CHECKING:
    from regimekit.config.schema import BacktestCfg, ProjectConfig
    from regimekit.data.contracts import CovariateSeries, ObservationSet, PreparedDataset
    from regimekit.engine.base import InferenceEngine
    from regimekit.estimation.results import FitResult
    from regimekit.evaluation.metrics.registry import Metric
    from regimekit.statespace.index import StateSpace
    from regimekit.utils.typing import Array

_LOG = get_logger("regimekit.backtest")

_YEARS_TOL = 1e-9  # elapsed-time comparisons: `504 * (1/252)` must count as 2 years


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold; all indices are grid rows, ends exclusive."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def origins(self) -> range:
        """Forecast origins of the fold: every row of `[test_start, test_end)`."""
        return range(self.test_start, self.test_end)


def schedule_folds(times: np.ndarray, dt: np.ndarray, cfg: BacktestCfg, max_horizon: int) -> list[Fold]:
    """The fold table for a grid of `T` rows.

    The initial training window is the shortest prefix whose elapsed time `cumsum(dt)` reaches
    `initial_train_years`. Fold `k` has `train_end = initial_end + k * refit_every_steps`,
    `test_start = train_end + embargo_steps` and
    `test_end = min(test_start + refit_every_steps, T - max_horizon)`, so every origin has
    `max_horizon` realized future rows. `expanding` folds start at row 0; `rolling` folds start at the
    latest row whose window still spans `rolling_years` (constant length up to one step). Folds with an
    empty test range are dropped.

    Args:
        times: `(T,)` grid end times (only their length is used here).
        dt: `(T,)` `Δ` per row in clock units (years under the trading/calendar clocks).
        cfg: the `evaluation.backtest` section.
        max_horizon: largest forecast horizon (`max(cfg.horizons)`).

    Raises:
        ConfigError: invalid backtest settings, `initial_train_years` longer than the data span, a
            rolling window longer than the initial window, or no fold left.
        DataError: `times` and `dt` disagree in length or `dt` is not positive.
    """
    dt = np.asarray(dt, dtype=np.float64)
    n_rows = int(np.asarray(times).shape[0])
    if dt.shape != (n_rows,) or n_rows == 0 or not bool(np.all(dt > 0)):
        raise DataError("times and dt must be equal-length non-empty (T,) arrays with dt > 0", path="dt")
    _check_cfg(cfg, max_horizon)

    elapsed = np.cumsum(dt)
    if elapsed[-1] < cfg.initial_train_years - _YEARS_TOL:
        raise ConfigError(
            f"initial_train_years={cfg.initial_train_years} exceeds the data span of {elapsed[-1]:.4g}",
            path="evaluation.backtest.initial_train_years",
        )
    initial_end = int(np.searchsorted(elapsed, cfg.initial_train_years - _YEARS_TOL, side="left")) + 1

    if cfg.scheme == "rolling":
        if cfg.rolling_years is None or cfg.rolling_years <= 0:
            raise ConfigError(
                "scheme 'rolling' requires rolling_years > 0", path="evaluation.backtest.rolling_years"
            )
        if cfg.rolling_years > cfg.initial_train_years + _YEARS_TOL:
            raise ConfigError(
                "rolling_years may not exceed initial_train_years (the first window could not be filled)",
                path="evaluation.backtest.rolling_years",
            )
    cum0 = np.concatenate([[0.0], elapsed])  # cum0[e] - cum0[s] = elapsed time of rows [s, e)

    last_origin_end = n_rows - max_horizon
    folds: list[Fold] = []
    k = 0
    while True:
        train_end = initial_end + k * cfg.refit_every_steps
        test_start = train_end + cfg.embargo_steps
        if test_start >= last_origin_end:
            break
        test_end = min(test_start + cfg.refit_every_steps, last_origin_end)
        if cfg.scheme == "expanding":
            train_start = 0
        else:
            assert cfg.rolling_years is not None
            # the latest start whose window [s, train_end) still spans rolling_years
            target = cum0[train_end] - cfg.rolling_years + _YEARS_TOL
            train_start = max(int(np.searchsorted(cum0, target, side="right")) - 1, 0)
        folds.append(Fold(len(folds), train_start, train_end, test_start, test_end))
        k += 1

    if not folds:
        raise ConfigError(
            f"no walk-forward fold fits: training ends at row {initial_end}, embargo {cfg.embargo_steps}, "
            f"max_horizon {max_horizon}, T={n_rows}",
            path="evaluation.backtest",
        )
    return folds


def _check_cfg(cfg: BacktestCfg, max_horizon: int) -> None:
    if cfg.refit_every_steps < 1:
        raise ConfigError("refit_every_steps must be >= 1", path="evaluation.backtest.refit_every_steps")
    if cfg.embargo_steps < 0:
        raise ConfigError("embargo_steps must be >= 0", path="evaluation.backtest.embargo_steps")
    if cfg.initial_train_years <= 0:
        raise ConfigError("initial_train_years must be > 0", path="evaluation.backtest.initial_train_years")
    if max_horizon < 1 or (cfg.horizons and max_horizon < max(cfg.horizons)):
        raise ConfigError(
            f"max_horizon={max_horizon} must be >= 1 and cover horizons {cfg.horizons}",
            path="evaluation.backtest.horizons",
        )


def fold_table(folds: list[Fold], times: np.ndarray) -> pd.DataFrame:
    """The folds as a table: row indices plus the timestamps they refer to.

    Columns: `fold`, `train_start`, `train_end`, `test_start`, `test_end` (grid rows, ends exclusive) and
    `train_start_time`, `train_end_time` (the last training row), `test_start_time`, `test_end_time` (the
    last test origin). `BacktestResult.folds` extends this with `fit_time`.
    """
    stamps = np.asarray(times).astype("datetime64[ns]")
    return pd.DataFrame(
        {
            "fold": [f.fold for f in folds],
            "train_start": [f.train_start for f in folds],
            "train_end": [f.train_end for f in folds],
            "test_start": [f.test_start for f in folds],
            "test_end": [f.test_end for f in folds],
            "train_start_time": stamps[[f.train_start for f in folds]],
            "train_end_time": stamps[[f.train_end - 1 for f in folds]],
            "test_start_time": stamps[[f.test_start for f in folds]],
            "test_end_time": stamps[[f.test_end - 1 for f in folds]],
        }
    )


# --- runner (E7-T02b) ----------------------------------------------------------------------------

_ORIGIN_COLUMNS = ["fold", "origin", "horizon", "metric", "value"]
_FOLD_COLUMNS = ["fold", "horizon", "metric", "value"]
_PIT_PREFIX = "pit_coverage.pit"

OriginRow = tuple[int, int, int, str, float]


def _flatten(name: str, value: float | dict[str, float]) -> dict[str, float]:
    """A metric result as `{row name: value}`; dict components become `<metric>.<key>`."""
    if isinstance(value, dict):
        return {(k if k.startswith(name) else f"{name}.{k}"): float(v) for k, v in value.items()}
    return {name: float(value)}


def _pool(values: np.ndarray, how: str) -> float:
    """Combine values with the metric's `aggregate` rule: mean, or root mean square."""
    return float(np.sqrt(np.mean(values**2))) if how == "rms" else float(np.mean(values))


def _filter_posterior(log_alpha: Array, loglik: Array, space: StateSpace, engine_tag: str) -> Posterior:
    """A `Posterior` carrying only the filter rows.

    `MonteCarloForecaster` reads `log_alpha[-1]` and nothing else; `log_beta` / `smoothed` are placeholders
    (the same rows) so that the dataclass is complete.
    """
    return Posterior(
        log_alpha=log_alpha,
        log_beta=log_alpha,
        smoothed=log_alpha,
        loglik=loglik,
        index=space.index,
        engine="explicit" if engine_tag == "explicit" else "expanded",
    )


def _next_dt(data: PreparedDataset, bars_per_day: int) -> float:
    """`Δ` of the first forecast step, on the same grid `MonteCarloForecaster` samples on."""
    times = np.asarray(data.times).astype("datetime64[ns]")
    step = np.median(np.diff(times)).astype("m8[ns]") if times.shape[0] > 1 else None
    _, dt = future_grid(times[-1], 1, data.clock, bars_per_day, step)
    return float(dt[0])


def _row(block: Array | None, index: int) -> Array | None:
    return None if block is None else jnp.asarray(block)[index]


def _rule(name: str, agg_of: dict[str, str]) -> str:
    """The aggregation rule of a (possibly flattened `metric.key`) row name."""
    return agg_of.get(name, agg_of.get(name.split(".", 1)[0], "mean"))


def run_backtest(
    cfg: ProjectConfig,
    obs: ObservationSet,
    covs: Sequence[CovariateSeries],
    seed: int,
    strategy: Strategy | None = None,
) -> BacktestResult:
    """Walk-forward evaluation: per fold refit on the train window, forecast from every test origin.

    Per fold (`schedule_folds` on the full grid): `Preparer.fit(train_end=<last train row>)` and `transform`
    of the whole sample (standardization statistics from the train window only, `PrepStats.train_window`);
    `fit` (E5-T03) on `window(train_start, train_end)` with the previous fold's optimum as one *extra*
    restart; one forward pass over `window(0, test_end)` gives the filtered state at every origin (the
    filter is causal, so row `o` equals the filter of `window(0, o + 1)`); for every origin `o` of the test
    range the Monte-Carlo forecast of horizon `max(horizons)` starts from that filter, with
    `derive_seed(seed, PURPOSE_BACKTEST_FORECAST, fold, o)`, and each configured metric is evaluated per
    horizon against the realized cumulative return. Windows are padded with masked rows to a multiple of 256
    (`evaluation.padding`), so the jitted engine functions compile once per padded length, not per fold.

    Metrics that need a decoded posterior (`aic`, `bic`, `icl`, `regime_*`) are evaluated once per fold on
    the train window and stored with `horizon == 0`. With a `strategy` (E7-T03), the position decided at
    every origin from its filtered probabilities and forecast is held over the next step; the resulting
    trace (`evaluation.strategies.strategy_trace`, costs from `evaluation.strategy.cost_bps`) is returned in
    `BacktestResult.strategy` and the strategy metrics (`sharpe`, `max_drawdown`, `turnover`, `hit_rate`) are
    evaluated per fold on its slice (`horizon == 0`). Without a strategy those metrics, and metrics that need
    labels, are skipped.

    Returns:
        `BacktestResult` with `folds` (`schedule_folds` table + `fit_time` seconds), `metrics` (one row per
        `(fold, horizon, metric)`: the mean over the fold's origins, root mean square for `rmse`),
        `strategy` (the trace, or `None` without a strategy) and `summary` (per `(horizon, metric)`:
        `mean`, `sd` across folds, `n` folds).
        `pit_ks` / `pit_ks_pvalue` rows pool the PIT values of all origins per horizon (`n` = pooled
        origins; the p-value ignores the overlap between horizons > 1 and is optimistic there).

    Raises:
        ConfigError: unknown metrics, a `scenario` covariate forecaster, or an infeasible fold schedule.
        DataError: the per-fold preparation changes the grid.
        NumericalError: a non-finite fit, filter or forecast.
    """
    bt = cfg.evaluation.backtest
    horizons = sorted(set(bt.horizons))
    n_sim = bt.n_sim if bt.n_sim is not None else cfg.forecast.n_sim
    metrics = resolve(list(cfg.evaluation.metrics))
    per_origin = [m for m in metrics if m.needs <= {"forecast", "fit"}]
    per_fold = [m for m in metrics if "posterior" in m.needs and m.needs <= {"posterior", "fit"}]
    per_strategy = [m for m in metrics if m.needs == {"strategy"}] if strategy is not None else []
    skipped = [m.name for m in metrics if m not in per_origin + per_fold + per_strategy]
    if skipped:
        _LOG.info("backtest_metrics_skipped", metrics=skipped, reason="need labels or a strategy trace")
    agg_of = {m.name: m.aggregate for m in per_origin + per_fold + per_strategy}
    if strategy is not None:
        check_backtest_admissible(strategy)

    # The grid does not depend on the fitted statistics: one full-sample preparation fixes the fold table.
    grid = Preparer(cfg.data).fit(obs, covs, train_end=obs.times[-1]).transform(obs, covs)
    folds = schedule_folds(np.asarray(grid.times), np.asarray(grid.dt), bt, horizons[-1])
    spec = build_model_spec(cfg, grid)
    space = build_explicit(spec) if spec.engine == "explicit" else build_expanded(spec)
    engine: InferenceEngine = (
        ExplicitDurationEngine(chunk_size=cfg.estimation.chunk_size)
        if spec.engine == "explicit"
        else ExpandedHMMEngine(parallel_scan=cfg.estimation.parallel_scan)
    )
    forecaster = None
    if grid.prep_stats.cov_names:
        if cfg.forecast.covariate_forecaster.type == "scenario":
            raise ConfigError(
                "a fixed scenario cannot be used in a walk-forward backtest (one path for every origin)",
                path="forecast.covariate_forecaster.type",
            )
        forecaster = make_forecaster(cfg.forecast.covariate_forecaster)
    mc = MonteCarloForecaster(
        spec, space, engine, forecaster, cfg.forecast, bars_per_day=cfg.data.bars_per_day
    )

    origin_rows: list[OriginRow] = []
    decisions: list[tuple[int, int, np.ndarray]] = []  # (fold, origin, position decided there)
    fold_rows: list[tuple[int, int, str, float]] = []
    fit_times: list[float] = []
    warm: Array | None = None
    for fold in folds:
        prep = Preparer(cfg.data).fit(
            obs,
            covs,
            train_end=grid.times[fold.train_end - 1],
            train_start=grid.times[fold.train_start],  # row 0 for expanding folds
        )
        data = prep.transform(obs, covs)
        if not np.array_equal(data.times, grid.times):
            raise DataError(f"fold {fold.fold}: the per-fold preparation changed the grid", path="data")

        train = data.window(fold.train_start, fold.train_end)
        started = time.perf_counter()
        fitted = fit_map(
            spec,
            pad_dataset(train),
            cfg.estimation,
            derive_seed(seed, PURPOSE_BACKTEST_FIT, fold.fold),
            warm_start=warm,
        )
        fit_times.append(time.perf_counter() - started)
        warm = fitted.uparams

        if per_fold:
            ctx = {"posterior": decode(fitted, train, engine), "fit": fitted}
            for metric in per_fold:
                for name, value in _flatten(metric.name, metric(**ctx)).items():
                    fold_rows.append((fold.fold, 0, name, value))

        if per_origin or strategy is not None:
            rows, taken = _forecast_fold(
                fold_index=fold.fold,
                origins=fold.origins(),
                data=data,
                test_end=fold.test_end,
                fitted=fitted,
                mc=mc,
                engine=engine,
                space=space,
                metrics=per_origin,
                horizons=horizons,
                n_sim=n_sim,
                seed=seed,
                bars_per_day=cfg.data.bars_per_day,
                strategy=strategy,
            )
            origin_rows += rows
            decisions += taken
        _LOG.info(
            "backtest_fold_done",
            fold=fold.fold,
            train_rows=fold.train_end - fold.train_start,
            origins=len(fold.origins()),
            fit_s=round(fit_times[-1], 3),
            converged=fitted.converged,
        )

    origin_frame = pd.DataFrame(origin_rows, columns=_ORIGIN_COLUMNS)
    fold_frame = _fold_metrics(origin_frame, fold_rows, agg_of)
    trace = None
    if strategy is not None:
        trace = _strategy_trace(decisions, grid, cfg)
        fold_frame = _with_strategy_metrics(fold_frame, trace, per_strategy)
    folds_table = fold_table(folds, grid.times)
    folds_table["fit_time"] = fit_times
    return BacktestResult(
        folds=folds_table,
        metrics=fold_frame,
        strategy=trace,
        summary=_summary(fold_frame, origin_frame, agg_of),
    )


def _forecast_fold(
    *,
    fold_index: int,
    origins: range,
    data: PreparedDataset,
    test_end: int,
    fitted: FitResult,
    mc: MonteCarloForecaster,
    engine: InferenceEngine,
    space: StateSpace,
    metrics: Sequence[Metric],
    horizons: list[int],
    n_sim: int,
    seed: int,
    bars_per_day: int,
    strategy: Strategy | None,
) -> tuple[list[OriginRow], list[tuple[int, int, np.ndarray]]]:
    """Metric rows `(fold, origin, horizon, metric, value)` and strategy decisions of one fold."""
    max_h = horizons[-1]
    engine_tag = fitted.spec.engine
    wants_mixture = any(m.name == "log_score_mixture" for m in metrics)
    padded = pad_dataset(data.window(0, test_end))
    log_alpha, loglik = engine.forward(fitted.params, fitted.spec, space, canonical_data(padded))
    obs_np = np.asarray(data.observations)
    mask_np = np.asarray(data.mask)
    rows: list[OriginRow] = []
    taken: list[tuple[int, int, np.ndarray]] = []
    for o in origins:
        post = _filter_posterior(log_alpha[: o + 1], loglik, space, engine_tag)
        data_o = data.window(0, o + 1)
        forecast = mc.forecast(
            fitted, data_o, max_h, n_sim, derive_seed(seed, PURPOSE_BACKTEST_FORECAST, fold_index, o), post
        )
        if strategy is not None and bool(mask_np[o + 1].all()):
            filtered = jax.nn.softmax(log_alpha[o]) @ space.index.collapse
            taken.append((fold_index, o, np.asarray(strategy.decide(filtered, forecast), dtype=np.float64)))
        base: dict[str, object] = {
            "forecast": forecast,
            "fit": fitted,
            "realized": obs_np[o + 1 : o + 1 + max_h],
        }
        if wants_mixture:
            dt_next = _next_dt(data_o, bars_per_day)
            base["dt_next"] = dt_next
            base["z_emission_next"] = _row(data.cov_emission, o)
            if engine_tag == "expanded":
                base["mixture_weights"] = one_step_mixture_weights(
                    log_alpha[o],
                    fitted.params,
                    space,
                    dt_next,
                    _row(data.cov_transition, o),
                    _row(data.cov_duration, o),
                )
        for h in horizons:
            if not bool(mask_np[o + 1 : o + 1 + h].all()):
                continue  # a missing realized return: no score for this (origin, horizon)
            ctx = {**base, "horizon": h}
            for metric in metrics:
                if metric.name == "log_score_mixture" and h != 1:
                    continue
                for name, value in _flatten(metric.name, metric(**ctx)).items():
                    rows.append((fold_index, o, h, name, value))
    return rows, taken


def _strategy_trace(
    decisions: list[tuple[int, int, np.ndarray]], grid: PreparedDataset, cfg: ProjectConfig
) -> pd.DataFrame:
    """The strategy table of the decisions: the position decided at origin `o` is held over row `o + 1`."""
    if not decisions:
        raise DataError(
            "the strategy recorded no position: every origin's next return is missing", path="data"
        )
    cost_bps = cfg.evaluation.strategy.cost_bps if cfg.evaluation.strategy is not None else 0.0
    rows = np.asarray([o + 1 for _, o, _ in decisions], dtype=np.int64)
    return strategy_trace(
        np.asarray(grid.times)[rows],
        np.vstack([pos for _, _, pos in decisions]),
        np.asarray(grid.observations)[rows],
        np.asarray(grid.dt)[rows],
        cost_bps,
        fold=np.asarray([f for f, _, _ in decisions], dtype=np.int64),
    )


def _with_strategy_metrics(
    fold_frame: pd.DataFrame, trace: pd.DataFrame, metrics: Sequence[Metric]
) -> pd.DataFrame:
    """`fold_frame` plus the strategy metrics of each fold's slice of `trace` (`horizon == 0`)."""
    rows: list[tuple[int, int, str, float]] = []
    for fold, part in trace.groupby("fold", sort=True):
        for metric in metrics:
            for name, value in _flatten(metric.name, metric(strategy=part.reset_index(drop=True))).items():
                rows.append((int(str(fold)), 0, name, value))
    extra = pd.DataFrame(rows, columns=_FOLD_COLUMNS).astype(
        {"fold": "int64", "horizon": "int64", "value": "float64"}
    )
    both = pd.concat([fold_frame, extra], ignore_index=True)
    return both.sort_values(["fold", "horizon", "metric"], kind="stable").reset_index(drop=True)


def _fold_metrics(
    origin_frame: pd.DataFrame, fold_rows: list[tuple[int, int, str, float]], agg_of: dict[str, str]
) -> pd.DataFrame:
    """`fold, horizon, metric, value`: origin values pooled within each fold, plus the per-fold metrics."""
    rows: list[tuple[int, int, str, float]] = list(fold_rows)
    for keys, group in origin_frame.groupby(["fold", "horizon", "metric"], sort=True):
        fold, horizon, name = (int(keys[0]), int(keys[1]), str(keys[2]))  # type: ignore[call-overload]
        rows.append((fold, horizon, name, _pool(group["value"].to_numpy(), _rule(name, agg_of))))
    frame = pd.DataFrame(rows, columns=_FOLD_COLUMNS).astype(
        {"fold": "int64", "horizon": "int64", "value": "float64"}
    )
    return frame.sort_values(["fold", "horizon", "metric"], kind="stable").reset_index(drop=True)


def _summary(fold_frame: pd.DataFrame, origin_frame: pd.DataFrame, agg_of: dict[str, str]) -> pd.DataFrame:
    """`horizon, metric, mean, sd, n`: across folds, plus the pooled-PIT Kolmogorov-Smirnov rows."""
    rows: list[dict[str, object]] = []
    for (horizon, name), group in fold_frame.groupby(["horizon", "metric"], sort=True):
        values = group["value"].to_numpy()
        rows.append(
            {
                "horizon": int(horizon),  # type: ignore[call-overload]
                "metric": str(name),
                "mean": _pool(values, _rule(str(name), agg_of)),
                "sd": float(np.std(values, ddof=1)) if values.shape[0] > 1 else float("nan"),
                "n": int(values.shape[0]),
            }
        )
    pit = origin_frame[origin_frame["metric"].str.startswith(_PIT_PREFIX)]
    for (horizon, name), group in pit.groupby(["horizon", "metric"], sort=True):
        _, d_stat, p_value = pit_histogram(group["value"].to_numpy())
        suffix = str(name)[len(_PIT_PREFIX) :]
        for label, value in ((f"pit_ks{suffix}", d_stat), (f"pit_ks_pvalue{suffix}", p_value)):
            rows.append(
                {
                    "horizon": int(horizon),  # type: ignore[call-overload]
                    "metric": label,
                    "mean": value,
                    "sd": float("nan"),
                    "n": int(group.shape[0]),
                }
            )
    summary = pd.DataFrame(rows, columns=["horizon", "metric", "mean", "sd", "n"])
    return summary.sort_values(["horizon", "metric"], kind="stable").reset_index(drop=True)
