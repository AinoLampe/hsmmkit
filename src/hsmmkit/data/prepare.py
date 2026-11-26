"""`Preparer`: raw observations + covariates -> `PreparedDataset`, sklearn-style (FR-D05).

Pipeline (same for `fit` and `transform`):

1. Master grid from the observations' times, or the step index (`data.calendar.build_grid`,
   `missing_observations` policy: `mask` keeps every row, `drop` removes rows where the primary series is
   missing, `error` raises).
2. Observations `y = transform(values)` with the registered observation transform (`identity` by default;
   an application registers more, e.g. a transform that drops the first row). `mask = isfinite(y)`, masked
   values are set to 0. A transform that keeps every row gets one virtual grid time before the first row,
   so that every observation has a spacing `dt`.
3. Support check (`real`, `count`, `binary`, `categorical`) against the emission family's support; counts,
   binary and categorical observations become `int64`.
4. Per covariate, in declared order: transforms on the series' own prints (`log`, `diff`, `pct`; `zscore`
   is a deprecated alias of standardization), then alignment + lag + `max_staleness` forward-fill
   (`data.calendar.align_covariate`). Prints are sorted by time first; of duplicated timestamps the last
   (a revision) is kept.
5. Warm-up: leading rows before every covariate has its first usable value are trimmed. The lag
   shift and each `diff`/`pct` cost rows by construction; that is history that does not exist yet,
   not missing data, and is trimmed for every policy. A covariate that starts later than that
   (e.g. a series beginning after the primary) raises under `missing: error` and is trimmed with a
   warning otherwise.
6. Missing policy on the remaining NaN, per covariate: `error` -> `DataError` naming the first
   offending timestamp; `drop` -> drop those grid rows; `ffill` -> the fill up to `max_staleness`
   already happened in step 4, rows still NaN are dropped like `drop`. Drops are joint: observations,
   `dt`, `mask`, `times` and every covariate are re-sliced together.
7. Standardization of the covariates (`standardize_covariates`, or `zscore` in a covariate's transforms)
   and, for real observations, of the observations (`standardize_observations`), with mean/std (ddof 0)
   fitted by `fit` on rows whose end time is `<= train_end` only, applied by `transform`.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, get_args

import jax.numpy as jnp
import numpy as np
import pandas as pd

from hsmmkit.config.schema import CovariateCfg, DataSection, EntryPoint
from hsmmkit.data.batch import SequenceBatch, split_sequences
from hsmmkit.data.calendar import Grid, align_covariate, build_grid
from hsmmkit.data.contracts import (
    BoundaryConstraints,
    Clock,
    CovariateSeries,
    ObservationSet,
    PreparedDataset,
    PrepStats,
    Support,
    lag_names,
    own_lag_columns,
)
from hsmmkit.data.labels import allowed_from_values, boundaries_of
from hsmmkit.data.transforms import apply_transforms, get_observation_transform
from hsmmkit.utils.clock import get_clock, parse_clock_options
from hsmmkit.utils.errors import ConfigError, DataError
from hsmmkit.utils.logging import get_logger

MIN_STD = 1e-12
ENTRY_POINTS: tuple[EntryPoint, ...] = get_args(EntryPoint)

_log = get_logger(__name__)


def _step(t: Any) -> int:  # noqa: ANN401
    """A step index from `t`; time-less data takes integer train bounds."""
    try:
        return int(t)
    except (TypeError, ValueError):
        raise DataError(
            f"time-less data needs an integer train bound (a step index), got {t!r}", path="train_end"
        ) from None


def _at_or_before(times: np.ndarray, t: Any) -> np.ndarray:  # noqa: ANN401
    """`times <= t` for timestamps or a step index."""
    if np.issubdtype(times.dtype, np.integer):
        return np.asarray(times <= _step(t))
    return np.asarray(times <= _utc_ns(t))


def _at_or_after(times: np.ndarray, t: Any) -> np.ndarray:  # noqa: ANN401
    """`times >= t` for timestamps or a step index."""
    if np.issubdtype(times.dtype, np.integer):
        return np.asarray(times >= _step(t))
    return np.asarray(times >= _utc_ns(t))


def _utc_ns(t: str | np.datetime64 | pd.Timestamp) -> np.datetime64:
    """Parse a timestamp; naive means UTC, aware ones are converted to UTC."""
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.as_unit("ns").to_datetime64(), "ns")


def _canonical(series: CovariateSeries) -> CovariateSeries:
    """Sort prints by time (stable) and keep the last of duplicated timestamps (a revised print)."""
    if series.times is None or np.issubdtype(np.asarray(series.times).dtype, np.integer):
        return series  # time-less: one print per grid row, already in order
    times = np.asarray(series.times).astype("datetime64[ns]")
    order = np.argsort(times, kind="stable")
    times, values = times[order], np.asarray(series.values, dtype=np.float64)[order]
    last = np.ones(times.shape[0], dtype=bool)
    last[:-1] = times[1:] != times[:-1]
    return replace(series, times=times[last], values=values[last])


def _structural_warmup(c: CovariateCfg) -> int:
    """Leading rows a covariate cannot fill by construction: lag shift plus one per `diff`/`pct`."""
    return c.lag - 1 + sum(t in ("diff", "pct") for t in c.transform)


def _iso(t: np.datetime64 | np.integer) -> str:
    if isinstance(t, np.integer):
        return str(int(t))
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
    n_categories: int | None = None


@dataclass(frozen=True, eq=False)
class PrepDiagnostics:
    """What the last `fit`/`transform` pipeline run removed or carried forward (for the quality report).

    Attributes:
        missing_rows: primary rows dropped for a missing value (`missing_observations: drop`).
        warmup_rows: leading grid rows trimmed before every covariate had a usable value.
        dropped: per covariate name, grid rows its `missing` policy dropped (rows can overlap).
        staleness: per covariate name, `(T_grid,)` staleness on the full grid (-1 = no usable print).
    """

    missing_rows: int
    warmup_rows: int
    dropped: dict[str, int]
    staleness: dict[str, np.ndarray]


def _with_virtual_start(grid: Grid, clock: str, clock_options: Any) -> Grid:  # noqa: ANN401
    """The grid with one extra time before the first row, so that every observation has a spacing.

    Used when the observation transform keeps every row: observation `t` then covers `(τ_{t-1}, τ_t]`
    with `τ_{-1}` one median spacing before the first time (for a step index: one step).
    """
    clk = get_clock(clock)
    opts = parse_clock_options(clock, clock_options)
    times = grid.times
    if np.issubdtype(times.dtype, np.integer):
        first = times[:1] - 1
    else:
        spacing = np.median(np.diff(times)) if times.shape[0] > 1 else np.timedelta64(1, "D")
        first = times[:1] - spacing
    extended = np.concatenate([first, times])
    dt = clk.dt(extended, extended.shape[0] - 1, opts)
    values = np.vstack([np.full((1, grid.values.shape[1]), np.nan), grid.values])
    return Grid(
        times=extended,
        values=values,
        dt=dt,
        nominal_dt=float(np.median(dt)),
        irregular=False if clk.unit == "step" else grid.irregular,
        clock=grid.clock,
    )


class Preparer:
    """Fit/transform preparation of a `PreparedDataset` (statistics from the training window only).

    Covariate series pair with `cfg.covariates[i]` by position; each `CovariateSeries.series_id` must
    equal `cfg.covariates[i].series`. Output covariate names are the config names.
    """

    def __init__(
        self,
        cfg: DataSection,
        *,
        support: Support = "real",
        n_categories: int | None = None,
        own_lags: int = 0,
        series_support: Sequence[Support] | None = None,
        n_regimes: int | None = None,
    ) -> None:
        """Store the `data:` config section and the observation support; nothing is fitted yet.

        Args:
            cfg: the `data:` section.
            support: the emission family's support (`real`, `count`, `binary`, `categorical`).
            n_categories: number of categories for `categorical` (`None`: `max + 1` with a warning).
            series_support: per-series supports of a mixed-family (`product` blocks) model; each column is
                validated against its own support and the observations stay float (`support` is then the first
                non-real entry). `None`: every series has `support`.
            n_regimes: the number of regimes `N`, needed to encode `data.labels` as the `(T, N)`
                allowed-regime matrix of `PreparedDataset.labels` (E12-T06).
            own_lags: `p > 0` appends the own lags `y_{t-1..t-p}` (after the observation transform and
                standardization) to the emission covariates as `__lag{l}_{series}`, and masks the first `p`
                steps and every step with an unobserved lag (autoregressive emissions, `requires_own_lags`).

        Raises:
            ConfigError: `standardize_observations` with a non-real support.
        """
        if cfg.preparation.standardize_observations and support != "real":
            raise ConfigError(
                f"standardize_observations needs real observations, not support {support!r}",
                path="data.preparation.standardize_observations",
            )
        self.series_support = None if series_support is None else tuple(series_support)
        if self.series_support is not None and support == "real":
            support = next((s for s in self.series_support if s != "real"), "real")
        self.support = support
        self.n_categories = n_categories
        if own_lags < 0:
            raise ConfigError("own_lags must be >= 0", path="model.emission.ar.p")
        self.own_lags = own_lags
        self.n_regimes = n_regimes
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
            obs: the observations.
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
        frame, train = self._train_frame(obs, covs, train_end, train_start)
        if not train.any():
            raise DataError(
                f"train_end {train_end!r} is before the first prepared observation {_iso(frame.times.min())}",
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
        obs_mean: np.ndarray | None = None
        obs_std: np.ndarray | None = None
        if self.cfg.preparation.standardize_observations:
            y, ok = frame.observations[train], frame.mask[train]
            counts = ok.sum(axis=0)
            if bool(np.any(counts < 2)):
                raise DataError(
                    "standardize_observations needs two observed train rows per series",
                    path="data.preparation.standardize_observations",
                )
            obs_mean = (np.where(ok, y, 0.0).sum(axis=0) / counts).astype(np.float64)
            centered = np.where(ok, y - obs_mean, 0.0)
            obs_std = np.sqrt((centered**2).sum(axis=0) / counts)
            if not bool(np.all(obs_std >= MIN_STD)):
                raise DataError(
                    f"an observation series has train-window std < {MIN_STD} (constant?)",
                    path="data.preparation.standardize_observations",
                )
        self._stats = PrepStats(
            cov_mean=mean,
            cov_std=std,
            train_window=(_iso(frame.times[train].min()), _iso(frame.times[train].max())),
            cov_names=tuple(c.name for c in covariates),
            transforms={c.name: list(c.transform) for c in covariates},
            lags={c.name: c.lag for c in covariates},
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        return self

    def _train_frame(
        self,
        obs: ObservationSet,
        covs: Sequence[CovariateSeries],
        train_end: str | np.datetime64 | pd.Timestamp,
        train_start: str | np.datetime64 | pd.Timestamp | None,
    ) -> tuple[_Frame, np.ndarray]:
        """The (pooled) prepared rows and the mask of the training rows.

        With `obs.sequence_ids` every sequence is prepared on its own and the rows of all sequences are
        concatenated, so the standardization statistics are those of the pooled training rows.
        """
        parts = split_sequences(obs, covs)
        frames = [self._frame(o, c) for _, o, c in parts]
        trains = []
        for f in frames:
            tr = _at_or_before(f.times, train_end)
            if train_start is not None:
                tr &= _at_or_after(f.times, train_start)
            trains.append(tr)
        if len(frames) == 1:
            return frames[0], trains[0]
        first = frames[0]
        pooled = _Frame(
            times=np.concatenate([f.times for f in frames]),
            observations=np.concatenate([f.observations for f in frames]),
            mask=np.concatenate([f.mask for f in frames]),
            dt=np.concatenate([f.dt for f in frames]),
            covariates=np.concatenate([f.covariates for f in frames]),
            nominal_dt=first.nominal_dt,
            clock=first.clock,
            n_categories=first.n_categories,
        )
        return pooled, np.concatenate(trains)

    def transform_batch(self, obs: ObservationSet, covs: Sequence[CovariateSeries]) -> SequenceBatch:
        """Prepare every sequence of `obs` (rows grouped by `sequence_ids`) with the fitted pooled statistics.

        Own lags, lagged covariates and first-step masks are generated **within** each sequence, so nothing
        crosses a sequence boundary. `data.sequences.group_by` (a column constant within a sequence) becomes
        `PreparedDataset.group`; `data.sequences.bucket_edges` are the batch's bucket edges.

        Returns:
            A `SequenceBatch` in first-appearance order of the sequence ids (`B = 1` without ids).

        Raises:
            ConfigError: a group column that varies within a sequence (`data.sequences.group_by`).
        """
        parts = split_sequences(obs, covs)
        datasets = [dataclasses.replace(self.transform(o, c), group=o.meta.get("group")) for _, o, c in parts]
        edges = () if self.cfg.sequences is None else tuple(self.cfg.sequences.bucket_edges)
        return SequenceBatch(tuple(datasets), edges, tuple(sid for sid, _, _ in parts))

    def transform(self, obs: ObservationSet, covs: Sequence[CovariateSeries]) -> PreparedDataset:
        """Prepare any window with the fitted statistics.

        Returns:
            A validated `PreparedDataset`; covariate blocks are `None` for unused entry points.
        """
        if obs.sequence_ids is not None and np.unique(np.asarray(obs.sequence_ids)).shape[0] > 1:
            raise ConfigError(
                "the observations hold several sequences: use transform_batch",
                path="data.source.sequence_column",
            )
        stats = self.prep_stats
        names = tuple(c.name for c in self.cfg.covariates)
        if stats.cov_names != names:
            raise DataError(
                f"fitted covariates {stats.cov_names} differ from the config {names}", path="prep_stats"
            )
        frame = self._frame(obs, covs)
        standardized = (frame.covariates - stats.cov_mean) / stats.cov_std
        observations = frame.observations
        if stats.obs_mean is not None and stats.obs_std is not None:
            observations = np.where(frame.mask, (observations - stats.obs_mean) / stats.obs_std, 0.0)

        blocks: dict[EntryPoint, Any] = {}
        cov_names: dict[str, tuple[str, ...]] = {}
        for ep in ENTRY_POINTS:
            idx = [i for i, c in enumerate(self.cfg.covariates) if ep in c.use_in]
            if idx:
                blocks[ep] = jnp.asarray(standardized[:, idx])
                cov_names[ep] = tuple(names[i] for i in idx)

        mask = frame.mask
        if self.own_lags:
            lags, ok = own_lag_columns(observations, mask, self.own_lags)
            lag_block = jnp.asarray(lags)
            block = blocks.get("emission")
            blocks["emission"] = lag_block if block is None else jnp.concatenate([block, lag_block], axis=1)
            cov_names["emission"] = (
                *cov_names.get("emission", ()),
                *lag_names(obs.series_names, self.own_lags),
            )
            mask = mask & ok[:, None]

        labels, boundaries = self._supervision(obs, frame)
        dataset = PreparedDataset(
            times=frame.times,
            observations=jnp.asarray(observations),
            dt=jnp.asarray(frame.dt),
            mask=jnp.asarray(mask),
            cov_transition=blocks.get("transition"),
            cov_duration=blocks.get("duration"),
            cov_emission=blocks.get("emission"),
            cov_names=cov_names,
            clock=frame.clock,
            nominal_dt=frame.nominal_dt,
            prep_stats=stats,
            snapshot_ids=(obs.snapshot_id, *(s.snapshot_id for s in covs)),
            support=self.support,
            series_support=self.series_support,
            n_categories=frame.n_categories,
            labels=labels,
            boundaries=boundaries,
            group=None,
        )
        dataset.validate()
        return dataset

    def _supervision(self, obs: ObservationSet, frame: _Frame) -> tuple[Any, BoundaryConstraints | None]:
        """`(labels (T, N) bool or None, boundaries)` of the prepared rows (`data.labels`, E12-T06)."""
        cfg = self.cfg.labels
        if cfg is None:
            return None, None
        if self.n_regimes is None:
            raise ConfigError(
                "data.labels needs the number of regimes: Preparer(..., n_regimes=N)", path="data.labels"
            )
        raw = obs.meta.get("label_values")
        if raw is None:
            raise DataError(
                "data.labels is configured but the source delivered no labels", path="data.labels"
            )
        allowed = allowed_from_values(raw, self.n_regimes, allowed_sets=cfg.allowed_sets)
        keys = (
            np.arange(len(raw), dtype=np.int64)
            if obs.times is None
            else np.asarray(obs.times).astype("datetime64[ns]")
        )
        times = np.asarray(frame.times)
        times = times.astype(np.int64) if obs.times is None else times.astype("datetime64[ns]")
        pos = np.minimum(np.searchsorted(keys, times), len(keys) - 1)
        matched = keys[pos] == times  # grid rows without a source row are unlabelled
        labels = jnp.asarray(np.where(matched[:, None], allowed[pos], True))
        bounds = cfg.boundaries
        boundaries = (
            None
            if bounds is None
            else boundaries_of(bounds.must_switch_at, bounds.must_not_switch_at, int(times.shape[0]))
        )
        return labels, boundaries

    # --- pipeline ------------------------------------------------------------------------------

    def _frame(self, obs: ObservationSet, covs: Sequence[CovariateSeries]) -> _Frame:
        cfg = self.cfg
        self._check_covariates(covs)
        grid = build_grid(
            obs,
            cfg.preparation.missing_observations,
            clock=cfg.clock,
            clock_options=cfg.clock_options,
        )

        transform = get_observation_transform(cfg.preparation.observation_transform)
        y, _ = transform(grid.values, grid.times)
        y = np.asarray(y)
        removed = grid.times.shape[0] - y.shape[0]
        if removed not in (0, 1):
            raise DataError(
                f"observation transform '{transform.name}' removed {removed} rows; only 0 or 1 are supported",
                path="data.preparation.observation_transform",
            )
        if removed == 0:
            grid = _with_virtual_start(grid, cfg.clock, cfg.clock_options)
        mask = np.isfinite(y)
        y = np.where(mask, y, 0.0)
        y, n_categories = self._apply_support(y, mask, obs.series_names)

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
        missing_rows = int(obs.values.shape[0] - (grid.times.shape[0] - (1 - removed)))
        # Recorded before the policies run, so a report can still show staleness if `error` raises.
        self.diagnostics = PrepDiagnostics(missing_rows, 0, {}, staleness)

        start = self._warmup(x, cfg.covariates, grid.end_times)
        keep, dropped = self._policy_keep(x[start:], cfg.covariates, grid.end_times[start:])
        rows = start + np.flatnonzero(keep)
        self.diagnostics = PrepDiagnostics(
            missing_rows=missing_rows,
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
            n_categories=n_categories,
        )

    def _apply_support(
        self, y: np.ndarray, mask: np.ndarray, names: Sequence[str]
    ) -> tuple[np.ndarray, int | None]:
        """Check the observations against the support; counts, binary and categorical become `int64`."""
        if self.series_support is not None:
            return self._apply_series_support(y, mask, names), None
        support = self.support
        if support == "real":
            return y, None
        observed = np.where(mask, y, 0.0)
        path = "data.source.observation_columns"

        def violated(bad: np.ndarray) -> DataError:
            r, m = (int(v) for v in np.argwhere(bad)[0])
            return DataError(f"{support} support violated at row {r}, series {names[m]}", path=path)

        integral = np.abs(observed - np.rint(observed)) < 1e-12
        if not bool(np.all(integral | ~mask)):
            raise violated(~integral & mask)
        ints = np.rint(observed).astype(np.int64)
        if support == "count":
            if bool(np.any((ints < 0) & mask)):
                raise violated((ints < 0) & mask)
            return ints, None
        if support == "binary":
            bad = ~np.isin(ints, (0, 1)) & mask
            if bool(bad.any()):
                raise violated(bad)
            return ints, None
        if y.shape[1] != 1:
            raise DataError("categorical support needs a single series", path=path)
        if bool(np.any((ints < 0) & mask)):
            raise violated((ints < 0) & mask)
        n_cat = self.n_categories
        if n_cat is None:
            n_cat = int(ints[mask].max()) + 1 if bool(mask.any()) else 1
            warnings.warn(f"n_categories not given; using max + 1 = {n_cat}", UserWarning, stacklevel=3)
        if bool(np.any((ints >= n_cat) & mask)):
            raise violated((ints >= n_cat) & mask)
        return ints, n_cat

    def _apply_series_support(self, y: np.ndarray, mask: np.ndarray, names: Sequence[str]) -> np.ndarray:
        """Mixed supports: check each column against its own support; the array stays float."""
        assert self.series_support is not None
        if len(self.series_support) != y.shape[1]:
            raise ConfigError(
                f"{len(self.series_support)} block supports for {y.shape[1]} observation series",
                path="model.emission.blocks",
            )
        for m, support in enumerate(self.series_support):
            col = y[mask[:, m], m]
            if support == "real":
                continue
            if support == "categorical":
                raise ConfigError(
                    "a categorical series cannot be part of a product", path="model.emission.blocks"
                )
            bad = np.abs(col - np.rint(col)) >= 1e-12
            bad |= col < 0 if support == "count" else ~np.isin(np.rint(col), (0, 1))
            if bool(bad.any()):
                row = int(np.flatnonzero(mask[:, m])[int(np.argmax(bad))])
                raise DataError(
                    f"{support} support violated at row {row}, series {names[m]}",
                    path="data.source.observation_columns",
                )
        return y

    def _check_covariates(self, covs: Sequence[CovariateSeries]) -> None:
        expected = self.cfg.covariates
        if len(covs) != len(expected):
            raise DataError(
                f"expected {len(expected)} covariate series (data.covariates), got {len(covs)}",
                path="data.covariates",
            )
        for i, (c, s) in enumerate(zip(expected, covs, strict=True)):
            if c.source is None:
                series: Any = c.column or c.name
            else:
                series = c.source.option("series")
            if series is not None and s.series_id != series:
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
            **(
                {}
                if stats.obs_mean is None or stats.obs_std is None
                else {
                    "obs_mean": [float(v) for v in stats.obs_mean],
                    "obs_std": [float(v) for v in stats.obs_std],
                }
            ),
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
        obs_mean=None if "obs_mean" not in d else np.asarray(d["obs_mean"], dtype=np.float64),
        obs_std=None if "obs_std" not in d else np.asarray(d["obs_std"], dtype=np.float64),
    )
