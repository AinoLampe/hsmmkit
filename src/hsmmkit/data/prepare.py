"""`Preparer`: raw observations + covariates -> `PreparedDataset`, sklearn-style (FR-D05).

Pipeline (same for `fit` and `transform`):

1. Master grid from the primary prices (`data.calendar.build_grid`, `missing_returns` policy).
2. Observations `y_t = log(P_t / P_{t-1})` (`transforms.log_returns`); `mask = isfinite(y)`,
   masked values set to 0.
3. Per covariate, in declared order: transforms on the series' own prints (`log`, `diff`, `pct`;
   `zscore` is a deprecated alias of standardization), then alignment + lag + `max_staleness`
   forward-fill (`data.calendar.align_covariate`).
   Prints are sorted by time first; of duplicated timestamps the last (a revision) is kept.
4. Warm-up: leading rows before every covariate has its first usable value are trimmed. The lag
   shift and each `diff`/`pct` cost rows by construction; that is history that does not exist yet,
   not missing data, and is trimmed for every policy. A covariate that starts later than that
   (e.g. a series beginning after the primary) raises under `missing: error` and is trimmed with a
   warning otherwise.
5. Missing policy on the remaining NaN, per covariate: `error` -> `DataError` naming the first
   offending timestamp; `drop` -> drop those grid rows; `ffill` -> the fill up to `max_staleness`
   already happened in step 3, rows still NaN are dropped like `drop`. Drops are joint: returns,
   `dt`, `mask`, `times` and every covariate are re-sliced together.
6. Standardization (`standardize_covariates`, or `zscore` in a covariate's transforms) with mean/std
   (ddof 0) fitted by `fit` on rows whose end time is `<= train_end` only, applied by `transform`.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast, get_args

import jax.numpy as jnp
import numpy as np
import pandas as pd

from regimekit.config.schema import CovariateCfg, DataSection, EntryPoint
from regimekit.data.calendar import align_covariate, build_grid
from regimekit.data.contracts import Clock, CovariateSeries, ObservationSet, PreparedDataset, PrepStats
from regimekit.data.transforms import apply_transforms, log_returns
from regimekit.utils.errors import DataError
from regimekit.utils.logging import get_logger

MIN_STD = 1e-12
ENTRY_POINTS: tuple[EntryPoint, ...] = get_args(EntryPoint)

_log = get_logger(__name__)


def _utc_ns(t: str | np.datetime64 | pd.Timestamp) -> np.datetime64:
    """Parse a timestamp; naive means UTC, aware ones are converted to UTC."""
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.as_unit("ns").to_datetime64(), "ns")


def _canonical(series: CovariateSeries) -> CovariateSeries:
    """Sort prints by time (stable) and keep the last of duplicated timestamps (a revised print)."""
    times = np.asarray(series.times).astype("datetime64[ns]")
    order = np.argsort(times, kind="stable")
    times, values = times[order], np.asarray(series.values, dtype=np.float64)[order]
    last = np.ones(times.shape[0], dtype=bool)
    last[:-1] = times[1:] != times[:-1]
    return replace(series, times=times[last], values=values[last])


def _structural_warmup(c: CovariateCfg) -> int:
    """Leading rows a covariate cannot fill by construction: lag shift plus one per `diff`/`pct`."""
    return c.lag - 1 + sum(t in ("diff", "pct") for t in c.transform)


def _iso(t: np.datetime64) -> str:
    ts = pd.Timestamp(t)
    return str(ts.date()) if ts == ts.normalize() else ts.isoformat()


@dataclass(frozen=True, eq=False)
class _Frame:
    """Rows of the prepared grid before standardization."""

    times: np.ndarray  # (T,) observation end times
    observations: np.ndarray  # (T, M)
    mask: np.ndarray  # (T, M)
    dt: np.ndarray  # (T,)
    covariates: np.ndarray  # (T, K), raw (transformed, aligned) covariate values
    nominal_dt: float
    clock: Clock


@dataclass(frozen=True, eq=False)
class PrepDiagnostics:
    """What the last `fit`/`transform` pipeline run removed or carried forward (for the quality report).

    Attributes:
        missing_price_rows: primary rows dropped for a missing price (`missing_returns: drop`).
        warmup_rows: leading grid rows trimmed before every covariate had a usable value.
        dropped: per covariate name, grid rows its `missing` policy dropped (rows can overlap).
        staleness: per covariate name, `(T_grid,)` staleness on the full grid (-1 = no usable print).
    """

    missing_price_rows: int
    warmup_rows: int
    dropped: dict[str, int]
    staleness: dict[str, np.ndarray]


class Preparer:
    """Fit/transform preparation of a `PreparedDataset` (statistics from the training window only).

    Covariate series pair with `cfg.covariates[i]` by position; each `CovariateSeries.series_id` must
    equal `cfg.covariates[i].series`. Output covariate names are the config names.
    """

    def __init__(self, cfg: DataSection) -> None:
        """Store the `data:` config section; nothing is fitted yet."""
        self.cfg = cfg
        self._stats: PrepStats | None = None
        self.diagnostics: PrepDiagnostics | None = None

    @property
    def prep_stats(self) -> PrepStats:
        """The fitted statistics (raises `DataError` before `fit`)."""
        if self._stats is None:
            raise DataError("Preparer is not fitted: call fit() first", path="prep_stats")
        return self._stats

    def fit(
        self,
        obs: ObservationSet,
        covs: Sequence[CovariateSeries],
        train_end: str | np.datetime64 | pd.Timestamp,
        train_start: str | np.datetime64 | pd.Timestamp | None = None,
    ) -> Preparer:
        """Fit standardization statistics on rows whose end time is `<= train_end` (and `>= train_start`).

        Args:
            obs: primary prices.
            covs: raw covariate series in `data.covariates` order.
            train_end: last training timestamp (inclusive), any `pd.Timestamp`-parsable value;
                naive means UTC, timezone-aware values are converted to UTC.
            train_start: optional first training timestamp (inclusive; a rolling walk-forward window).
                `None` = from the first prepared row.

        Returns:
            `self`, fitted.

        Raises:
            DataError: no training rows, a train-window std `< 1e-12`, or any pipeline error.
        """
        frame = self._frame(obs, covs)
        train = frame.times <= _utc_ns(train_end)
        if train_start is not None:
            train &= frame.times >= _utc_ns(train_start)
        if not train.any():
            raise DataError(
                f"train_end {train_end!r} is before the first prepared observation {_iso(frame.times[0])}",
                path="train_end",
            )
        covariates = self.cfg.covariates
        mean = np.zeros(len(covariates))
        std = np.ones(len(covariates))
        for i, c in enumerate(covariates):
            if "zscore" in c.transform:
                warnings.warn(
                    f"data.covariates[{i}].transform: 'zscore' is deprecated; it is applied as "
                    "train-window standardization (use preparation.standardize_covariates)",
                    FutureWarning,
                    stacklevel=2,
                )
            if not (self.cfg.preparation.standardize_covariates or "zscore" in c.transform):
                continue
            x = frame.covariates[train, i]
            mean[i], std[i] = x.mean(), x.std()
            if not std[i] >= MIN_STD:
                raise DataError(
                    f"covariate {c.name!r} has train-window std {std[i]:.3g} < {MIN_STD} (constant?)",
                    path=f"data.covariates[{i}]",
                )
        self._stats = PrepStats(
            cov_mean=mean,
            cov_std=std,
            train_window=(_iso(frame.times[train][0]), _iso(frame.times[train][-1])),
            cov_names=tuple(c.name for c in covariates),
            transforms={c.name: list(c.transform) for c in covariates},
            lags={c.name: c.lag for c in covariates},
        )
        return self

    def transform(self, obs: ObservationSet, covs: Sequence[CovariateSeries]) -> PreparedDataset:
        """Prepare any window with the fitted statistics.

        Returns:
            A validated `PreparedDataset`; covariate blocks are `None` for unused entry points.
        """
        stats = self.prep_stats
        names = tuple(c.name for c in self.cfg.covariates)
        if stats.cov_names != names:
            raise DataError(
                f"fitted covariates {stats.cov_names} differ from the config {names}", path="prep_stats"
            )
        frame = self._frame(obs, covs)
        standardized = (frame.covariates - stats.cov_mean) / stats.cov_std

        blocks: dict[EntryPoint, Any] = {}
        cov_names: dict[str, tuple[str, ...]] = {}
        for ep in ENTRY_POINTS:
            idx = [i for i, c in enumerate(self.cfg.covariates) if ep in c.use_in]
            if idx:
                blocks[ep] = jnp.asarray(standardized[:, idx])
                cov_names[ep] = tuple(names[i] for i in idx)

        dataset = PreparedDataset(
            times=frame.times,
            observations=jnp.asarray(frame.observations),
            dt=jnp.asarray(frame.dt),
            mask=jnp.asarray(frame.mask),
            cov_transition=blocks.get("transition"),
            cov_duration=blocks.get("duration"),
            cov_emission=blocks.get("emission"),
            cov_names=cov_names,
            clock=frame.clock,
            nominal_dt=frame.nominal_dt,
            prep_stats=stats,
            snapshot_ids=(obs.snapshot_id, *(s.snapshot_id for s in covs)),
            support="real",
            n_categories=None,
            labels=None,
            boundaries=None,
            group=None,
        )
        dataset.validate()
        return dataset

    # --- pipeline ------------------------------------------------------------------------------

    def _frame(self, obs: ObservationSet, covs: Sequence[CovariateSeries]) -> _Frame:
        cfg = self.cfg
        self._check_covariates(covs)
        grid = build_grid(
            obs,
            cast(Any, cfg.preparation.missing_observations),  # "mask" arrives with E10-T05
            clock=cast(Clock, cfg.clock),
            bars_per_day=int(cfg.clock_option("bars_per_day", 1)),
        )

        y = log_returns(grid.prices)
        mask = np.isfinite(y)
        y = np.where(mask, y, 0.0)

        n_cov = len(cfg.covariates)
        x = np.empty((grid.n_steps, n_cov), dtype=np.float64)
        staleness: dict[str, np.ndarray] = {}
        for i, (c, s) in enumerate(zip(cfg.covariates, covs, strict=True)):
            s = _canonical(s)
            values = apply_transforms(s.values, c.transform, path=f"data.covariates[{i}].transform")
            x[:, i], staleness[c.name] = align_covariate(
                grid, replace(s, values=values), c.lag, c.max_staleness, index=i
            )
        x[~np.isfinite(x)] = np.nan  # defensive: nothing non-finite may reach the dataset
        missing_price_rows = int(obs.times.shape[0] - grid.times.shape[0])
        # Recorded before the policies run, so a report can still show staleness if `error` raises.
        self.diagnostics = PrepDiagnostics(missing_price_rows, 0, {}, staleness)

        start = self._warmup(x, cfg.covariates, grid.end_times)
        keep, dropped = self._policy_keep(x[start:], cfg.covariates, grid.end_times[start:])
        rows = start + np.flatnonzero(keep)
        self.diagnostics = PrepDiagnostics(
            missing_price_rows=missing_price_rows,
            warmup_rows=start,
            dropped=dropped,
            staleness=staleness,
        )
        return _Frame(
            times=grid.end_times[rows],
            observations=y[rows],
            mask=mask[rows],
            dt=grid.dt[rows],
            covariates=x[rows],
            nominal_dt=grid.nominal_dt,
            clock=grid.clock,
        )

    def _check_covariates(self, covs: Sequence[CovariateSeries]) -> None:
        expected = self.cfg.covariates
        if len(covs) != len(expected):
            raise DataError(
                f"expected {len(expected)} covariate series (data.covariates), got {len(covs)}",
                path="data.covariates",
            )
        for i, (c, s) in enumerate(zip(expected, covs, strict=True)):
            series = None if c.source is None else c.source.option("series")
            if s.series_id != series:
                raise DataError(
                    f"covariate {i} ({c.name!r}) expects series {series!r}, got {s.series_id!r}",
                    path=f"data.covariates[{i}].series",
                )

    @staticmethod
    def _warmup(x: np.ndarray, covariates: Sequence[CovariateCfg], end_times: np.ndarray) -> int:
        start = 0
        for i, c in enumerate(covariates):
            valid = np.flatnonzero(~np.isnan(x[:, i]))
            if valid.size == 0:
                raise DataError(
                    f"covariate {c.name!r} has no usable value on the grid "
                    f"({_iso(end_times[0])} .. {_iso(end_times[-1])})",
                    path=f"data.covariates[{i}]",
                )
            first = int(valid[0])
            if first > _structural_warmup(c):
                where = f"{_iso(end_times[0])} .. {_iso(end_times[first])}"
                if c.missing == "error":
                    raise DataError(
                        f"covariate {c.name!r} has no value for the observations {where} (missing='error')",
                        path=f"data.covariates[{i}].missing",
                    )
                _log.warning("covariate_starts_late", covariate=c.name, rows=first, span=where)
            start = max(start, first)
        if start:
            _log.info("warmup_rows_trimmed", count=start, first_time=_iso(end_times[start]))
        return start

    @staticmethod
    def _policy_keep(
        x: np.ndarray, covariates: Sequence[CovariateCfg], end_times: np.ndarray
    ) -> tuple[np.ndarray, dict[str, int]]:
        keep = np.ones(x.shape[0], dtype=bool)
        dropped = {c.name: 0 for c in covariates}
        for i, c in enumerate(covariates):
            missing = np.isnan(x[:, i])
            if not missing.any():
                continue
            if c.missing == "error":
                first = _iso(end_times[np.flatnonzero(missing)[0]])
                raise DataError(
                    f"covariate {c.name!r} has no value within max_staleness={c.max_staleness} "
                    f"for the observation ending {first} (missing='error')",
                    path=f"data.covariates[{i}].missing",
                )
            _log.info("covariate_rows_dropped", covariate=c.name, policy=c.missing, count=int(missing.sum()))
            dropped[c.name] = int(missing.sum())
            keep &= ~missing
        return keep, dropped


def prep_stats_to_json(stats: PrepStats) -> str:
    """Serialize `PrepStats` to JSON (floats round-trip exactly via `repr`)."""
    return json.dumps(
        {
            "cov_mean": [float(v) for v in stats.cov_mean],
            "cov_std": [float(v) for v in stats.cov_std],
            "train_window": list(stats.train_window),
            "cov_names": list(stats.cov_names),
            "transforms": {k: list(v) for k, v in stats.transforms.items()},
            "lags": dict(stats.lags),
        },
        indent=2,
    )


def prep_stats_from_json(text: str) -> PrepStats:
    """Inverse of `prep_stats_to_json`."""
    d = json.loads(text)
    return PrepStats(
        cov_mean=np.asarray(d["cov_mean"], dtype=np.float64),
        cov_std=np.asarray(d["cov_std"], dtype=np.float64),
        train_window=(d["train_window"][0], d["train_window"][1]),
        cov_names=tuple(d["cov_names"]),
        transforms={k: list(v) for k, v in d["transforms"].items()},
        lags={k: int(v) for k, v in d["lags"].items()},
    )
