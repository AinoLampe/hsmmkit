"""Data contracts: the only shared types between the data layer and the model layer (FR-D07).

`PreparedDataset` is the sole contract the model layer (`model`, `statespace`, `engine`,
`estimation`, `inference`) may import from `hsmmkit.data` (the layer rule, overview §4.3).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import numpy as np

from hsmmkit.utils.clock import get_clock
from hsmmkit.utils.errors import DataError
from hsmmkit.utils.typing import Array

Clock = str  # the name of a registered clock (`hsmmkit.data.clocks`, `PreparedDataset.clock`)
Support = Literal["real", "count", "binary", "categorical"]


LAG_PREFIX = "__lag"
"""Prefix of the own-lag emission covariates generated for autoregressive emissions (`msar`)."""


def lag_names(series_names: tuple[str, ...], p: int) -> tuple[str, ...]:
    """Names of the own-lag covariate columns: `__lag{l}_{series}` for `l = 1..p`, lag-major order."""
    return tuple(f"{LAG_PREFIX}{lag}_{name}" for lag in range(1, p + 1) for name in series_names)


def own_lag_columns(observations: np.ndarray, mask: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """`(lags (T, p M), ok (T,))`: row `t` holds `y_{t-1}, ..., y_{t-p}` (lag-major, then series).

    `ok[t]` is true when `t >= p` and every series was observed at the `p` previous steps; a row that is not
    `ok` has zeros where a lag is unavailable and must be masked by the caller. Lags are `>= 1` by
    construction: row `t` never reads `observations[t:]`.
    """
    t, m = observations.shape
    y = np.where(mask, observations, 0.0)
    lags = np.zeros((t, p * m))
    ok = np.arange(t) >= p
    observed = mask.all(axis=1)
    for lag in range(1, p + 1):
        lags[lag:, (lag - 1) * m : lag * m] = y[: t - lag]
        ok[lag:] &= observed[: t - lag]
    return lags, ok


def _is_float_dtype(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.floating)


def _array_equal(a: np.ndarray | jax.Array | None, b: np.ndarray | jax.Array | None) -> bool:
    if a is None or b is None:
        return a is b
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


@dataclass(frozen=True, eq=False)
class ObservationSet:
    """Raw observations of a source before alignment (data layer only).

    Attributes:
        series_names: `(M,)` names of the observed series (columns of `values`).
        times: `(T0,)` `datetime64[ns]` UTC, strictly increasing; `None` only for time-less data, which
            is valid under a `step` clock (`PreparedDataset.times` is then an `int64` step index).
        values: `(T0, M)` `float64` or `int64`.
        source: name of the source that produced the data.
        snapshot_id: sha256 of the source bytes / array buffers; `""` if not computed.
        meta: source-specific extras (an application puts its own descriptors here).
        sequence_ids: `(T0,)` group ids when present; consumed from E12-T01a on, ignored before.
    """

    series_names: tuple[str, ...]
    times: np.ndarray | None
    values: np.ndarray
    source: str
    snapshot_id: str
    meta: dict[str, Any] = field(default_factory=dict)
    sequence_ids: np.ndarray | None = None

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        """Structural equality, safe for numpy array fields."""
        if not isinstance(other, ObservationSet):
            return NotImplemented
        return (
            self.series_names == other.series_names
            and _array_equal(self.times, other.times)
            and _array_equal(self.values, other.values)
            and self.source == other.source
            and self.snapshot_id == other.snapshot_id
            and self.meta == other.meta
            and _array_equal(self.sequence_ids, other.sequence_ids)
        )

    def require_times(self) -> np.ndarray:
        """The timestamps; `DataError` for time-less data (callers that cannot work without them)."""
        if self.times is None:
            raise DataError("this operation needs observations with timestamps", path="times")
        return self.times

    def validate(self) -> None:
        """Raise `DataError(path=<field>)` on any contract violation."""
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise DataError("values must be 2-D (T0, M)", path="values")
        t0, m = values.shape
        if len(self.series_names) != m:
            raise DataError(
                f"series_names has {len(self.series_names)} entries but values has {m} columns",
                path="series_names",
            )
        if not (np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.integer)):
            raise DataError(f"values must be float64 or int64, got {values.dtype}", path="values")
        if self.times is not None:
            times = np.asarray(self.times)
            if times.ndim != 1 or times.shape[0] != t0:
                raise DataError(f"times must have shape (T0={t0},)", path="times")
            if t0 > 1 and not bool(np.all(np.diff(times) > np.timedelta64(0, "ns"))):
                raise DataError("times must be strictly increasing", path="times")
        if self.sequence_ids is not None and np.asarray(self.sequence_ids).shape != (t0,):
            raise DataError(f"sequence_ids must have shape (T0={t0},)", path="sequence_ids")


@dataclass(frozen=True, eq=False)
class CovariateSeries:
    """A single raw covariate series before alignment (data layer only)."""

    name: str
    times: np.ndarray | None  # (T1,); None = one value per grid row (aligned by row index)
    values: np.ndarray  # (T1,), NaN allowed
    source: str
    series_id: str
    snapshot_id: str
    frequency: str

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        """Structural equality, safe for numpy array fields."""
        if not isinstance(other, CovariateSeries):
            return NotImplemented
        return (
            self.name == other.name
            and _array_equal(self.times, other.times)
            and _array_equal(self.values, other.values)
            and self.source == other.source
            and self.series_id == other.series_id
            and self.snapshot_id == other.snapshot_id
            and self.frequency == other.frequency
        )


@dataclass(frozen=True, eq=False)
class PrepStats:
    """Preparation statistics fitted on the training window only."""

    cov_mean: np.ndarray  # (K,)
    cov_std: np.ndarray  # (K,)
    train_window: tuple[str, str]
    cov_names: tuple[str, ...]
    transforms: dict[str, list[str]]
    lags: dict[str, int]
    obs_mean: np.ndarray | None = None  # (M,) train-window mean when `standardize_observations`
    obs_std: np.ndarray | None = None  # (M,)

    def __eq__(self, other: object) -> bool:
        """Structural equality, safe for numpy array fields."""
        if not isinstance(other, PrepStats):
            return NotImplemented
        return (
            _array_equal(self.cov_mean, other.cov_mean)
            and _array_equal(self.cov_std, other.cov_std)
            and self.train_window == other.train_window
            and self.cov_names == other.cov_names
            and self.transforms == other.transforms
            and self.lags == other.lags
            and _array_equal(self.obs_mean, other.obs_mean)
            and _array_equal(self.obs_std, other.obs_std)
        )

    def __hash__(self) -> int:
        """Content-based hash, consistent with `__eq__` (required: `PrepStats` sits in pytree aux data)."""
        return hash(
            (
                self.cov_mean.shape,
                self.cov_mean.dtype.str,
                self.cov_mean.tobytes(),
                self.cov_std.shape,
                self.cov_std.dtype.str,
                self.cov_std.tobytes(),
                self.train_window,
                self.cov_names,
                tuple(sorted((k, tuple(v)) for k, v in self.transforms.items())),
                tuple(sorted(self.lags.items())),
                None if self.obs_mean is None else self.obs_mean.tobytes(),
                None if self.obs_std is None else self.obs_std.tobytes(),
            )
        )


@dataclass(frozen=True)
class BoundaryConstraints:
    """Part II supervision hook; always `None` on `PreparedDataset` in 0.1.0."""

    must_switch_at: tuple[int, ...] = ()
    must_not_switch_at: tuple[int, ...] = ()

    def __hash__(self) -> int:
        """Value-based hash over the two index tuples."""
        return hash((self.must_switch_at, self.must_not_switch_at))


@dataclass(frozen=True, eq=False)
class PreparedDataset:
    """The only object the model layer consumes."""

    times: np.ndarray  # (T,) grid end times (datetime64[ns]) or an int64 step index; never a jit leaf
    observations: Array  # (T, M) float64
    dt: Array  # (T,) clock units, > 0
    mask: Array  # (T, M) bool, True = observed
    cov_transition: Array | None  # (T, K_pi)
    cov_duration: Array | None  # (T, K_d)
    cov_emission: Array | None  # (T, K_e)
    cov_names: dict[str, tuple[str, ...]]
    clock: Clock
    nominal_dt: float
    prep_stats: PrepStats
    snapshot_ids: tuple[str, ...]
    support: Support = "real"
    n_categories: int | None = None
    labels: Array | None = None  # (T, N) bool
    boundaries: BoundaryConstraints | None = None
    group: str | None = None
    series_support: tuple[Support, ...] | None = None
    """Per-series supports of a mixed-family model (`None`: every series has `support`). With mixed blocks
    `support` holds the first non-real support and `series_support` is authoritative; `observations` stay
    float and each column is validated against its own support."""

    def __hash__(self) -> int:
        """Identity-based hash; equality below is value-based (fine: only used for `is`-safety)."""
        return object.__hash__(self)

    def __eq__(self, other: object) -> bool:
        """Structural equality, safe for array-valued fields (used by round-trip tests)."""
        if not isinstance(other, PreparedDataset):
            return NotImplemented
        return (
            _array_equal(self.times, other.times)
            and _array_equal(self.observations, other.observations)
            and _array_equal(self.dt, other.dt)
            and _array_equal(self.mask, other.mask)
            and _array_equal(self.cov_transition, other.cov_transition)
            and _array_equal(self.cov_duration, other.cov_duration)
            and _array_equal(self.cov_emission, other.cov_emission)
            and self.cov_names == other.cov_names
            and self.clock == other.clock
            and self.nominal_dt == other.nominal_dt
            and self.prep_stats == other.prep_stats
            and self.snapshot_ids == other.snapshot_ids
            and self.support == other.support
            and self.n_categories == other.n_categories
            and _array_equal(self.labels, other.labels)
            and self.boundaries == other.boundaries
            and self.group == other.group
            and self.series_support == other.series_support
        )

    def window(self, start: int, end: int) -> PreparedDataset:
        """Slice `[start, end)` along `T`.

        Keeps `dt`, masks, covariates, `cov_names`, `prep_stats`, `snapshot_ids`, `support`,
        `group`, and `nominal_dt` (a property of the grid, not the slice).
        """

        def _slice(arr: Array | None) -> Array | None:
            return None if arr is None else arr[start:end]

        def _slice_required(arr: Array) -> Array:
            return arr[start:end]

        boundaries = None
        if self.boundaries is not None:
            boundaries = BoundaryConstraints(
                must_switch_at=tuple(i - start for i in self.boundaries.must_switch_at if start <= i < end),
                must_not_switch_at=tuple(
                    i - start for i in self.boundaries.must_not_switch_at if start <= i < end
                ),
            )

        return dataclasses.replace(
            self,
            times=self.times[start:end],
            observations=_slice_required(self.observations),
            dt=_slice_required(self.dt),
            mask=_slice_required(self.mask),
            cov_transition=_slice(self.cov_transition),
            cov_duration=_slice(self.cov_duration),
            cov_emission=_slice(self.cov_emission),
            labels=_slice(self.labels),
            boundaries=boundaries,
        )

    def validate(self) -> None:
        """Raise `DataError(path=<field>)` on any contract violation."""
        times, observations, dt, mask = self.times, self.observations, self.dt, self.mask

        if times.ndim != 1:
            raise DataError("times must be 1-D", path="times")
        t = times.shape[0]
        step_index = np.issubdtype(times.dtype, np.integer)
        if step_index and get_clock(self.clock).unit != "step":
            raise DataError(
                f"an integer step index is only valid under a step clock, not {self.clock!r}", path="times"
            )
        if t > 1 and not np.all(np.diff(times) > (0 if step_index else np.timedelta64(0, "ns"))):
            raise DataError("times must be strictly increasing", path="times")

        if observations.ndim != 2 or observations.shape[0] != t:
            raise DataError(f"observations must have shape (T={t}, M)", path="observations")
        m = observations.shape[1]

        if dt.ndim != 1 or dt.shape[0] != t:
            raise DataError(f"dt must have shape (T={t},)", path="dt")
        if not bool(np.all(np.asarray(dt) > 0)):
            raise DataError("dt must be > 0 everywhere", path="dt")
        clock = get_clock(self.clock)  # unknown name: ConfigError listing the registered clocks
        if clock.unit == "step" and not bool(np.all(np.asarray(dt) == 1)):
            raise DataError(f"dt must be exactly 1 under the step clock {self.clock!r}", path="dt")

        if mask.shape != (t, m) or np.asarray(mask).dtype != np.bool_:
            raise DataError(f"mask must be a bool array of shape (T={t}, M={m})", path="mask")

        obs_np = np.asarray(observations)
        mask_np = np.asarray(mask)
        if not np.all(np.isfinite(obs_np[mask_np])):
            raise DataError("observations must be finite where mask is True", path="observations")

        self._validate_support(obs_np, mask_np)
        self._validate_covariates(t)
        self._validate_labels(t)
        self._validate_boundaries(t)

    def _validate_support(self, obs_np: np.ndarray, mask_np: np.ndarray) -> None:
        if self.series_support is not None:
            self._validate_series_support(obs_np, mask_np)
            return
        observed = obs_np[mask_np]
        if self.support == "real":
            if not _is_float_dtype(obs_np.dtype):
                raise DataError("support='real' requires float observations", path="observations")
        elif self.support == "count":
            if not (np.issubdtype(obs_np.dtype, np.integer) and bool(np.all(observed >= 0))):
                raise DataError(
                    "support='count' requires non-negative integer observations", path="observations"
                )
        elif self.support == "binary":
            if not bool(np.all(np.isin(observed, [0, 1]))):
                raise DataError("support='binary' requires observations in {0, 1}", path="observations")
        elif self.support == "categorical":
            if self.n_categories is None:
                raise DataError("support='categorical' requires n_categories", path="n_categories")
            if not (
                np.issubdtype(obs_np.dtype, np.integer)
                and bool(np.all((observed >= 0) & (observed < self.n_categories)))
            ):
                raise DataError(
                    f"support='categorical' requires integers in [0, {self.n_categories})",
                    path="observations",
                )

    def _validate_series_support(self, obs_np: np.ndarray, mask_np: np.ndarray) -> None:
        """Mixed supports: every column against its own support (`categorical` is not allowed in a mix)."""
        assert self.series_support is not None
        if len(self.series_support) != obs_np.shape[1]:
            raise DataError(
                f"series_support has {len(self.series_support)} entries for {obs_np.shape[1]} series",
                path="series_support",
            )
        for m, support in enumerate(self.series_support):
            col = obs_np[mask_np[:, m], m]
            if support == "count":
                ok = bool(np.all((col >= 0) & (col == np.rint(col))))
            elif support == "binary":
                ok = bool(np.all(np.isin(col, [0, 1])))
            elif support == "real":
                ok = bool(np.all(np.isfinite(col)))
            else:
                raise DataError(f"series {m}: support {support!r} cannot be mixed", path="series_support")
            if not ok:
                raise DataError(
                    f"series {m}: observations violate support {support!r}", path=f"observations[:, {m}]"
                )

    def _validate_covariates(self, t: int) -> None:
        blocks = {
            "transition": self.cov_transition,
            "duration": self.cov_duration,
            "emission": self.cov_emission,
        }
        for entry_point, block in blocks.items():
            path = f"cov_{entry_point}"
            names = self.cov_names.get(entry_point, ())
            if block is None:
                if names:
                    raise DataError(f"cov_names['{entry_point}'] set but {path} is None", path="cov_names")
                continue
            if block.ndim != 2 or block.shape[0] != t:
                raise DataError(f"{path} must have shape (T={t}, K)", path=path)
            if len(names) != block.shape[1]:
                raise DataError(
                    f"cov_names['{entry_point}'] length must match {path}.shape[1]", path="cov_names"
                )

    def _validate_labels(self, t: int) -> None:
        if self.labels is None:
            return
        if self.labels.ndim != 2 or self.labels.shape[0] != t or np.asarray(self.labels).dtype != np.bool_:
            raise DataError("labels must be a bool array of shape (T, N)", path="labels")

    def _validate_boundaries(self, t: int) -> None:
        if self.boundaries is None:
            return
        for indices in (self.boundaries.must_switch_at, self.boundaries.must_not_switch_at):
            for i in indices:
                if not (1 <= i <= t - 1):
                    raise DataError(f"boundary index {i} outside [1, {t - 1}]", path="boundaries")


@dataclass(frozen=True)
class _HashableArray:
    """Wraps a numpy array so it can sit in a pytree's static aux data.

    `times` is metadata (grid end times, dtype `datetime64[ns]`): not a valid JAX array dtype, so
    it cannot be a pytree leaf (`jax.jit` would reject it), yet a raw `np.ndarray` is also not
    safely hashable/comparable as plain aux data (`array == array` returns an array, not a bool,
    breaking JAX's aux-equality check whenever `times` has more than one element — see CLAUDE.md
    §5's "NumPy arrays are unhashable" gotcha). This wrapper is used only inside flatten/unflatten
    below; `PreparedDataset.times` itself stays a plain `np.ndarray`.
    """

    array: np.ndarray

    def __eq__(self, other: object) -> bool:
        """Value equality via `np.array_equal`, always returning a plain `bool`."""
        if not isinstance(other, _HashableArray):
            return NotImplemented
        return bool(np.array_equal(self.array, other.array))

    def __hash__(self) -> int:
        """Hash the array's shape, dtype, and bytes."""
        return hash((self.array.shape, self.array.dtype.str, self.array.tobytes()))


def _flatten_prepared_dataset(
    dataset: PreparedDataset,
) -> tuple[tuple[Array | None, ...], tuple[Any, ...]]:
    children = (
        dataset.observations,
        dataset.dt,
        dataset.mask,
        dataset.cov_transition,
        dataset.cov_duration,
        dataset.cov_emission,
        dataset.labels,
    )
    aux = (
        _HashableArray(dataset.times),
        tuple(sorted(dataset.cov_names.items())),
        dataset.clock,
        dataset.nominal_dt,
        dataset.prep_stats,
        dataset.snapshot_ids,
        dataset.support,
        dataset.n_categories,
        dataset.boundaries,
        dataset.group,
        dataset.series_support,
    )
    return children, aux


def _unflatten_prepared_dataset(aux: tuple[Any, ...], children: tuple[Array | None, ...]) -> PreparedDataset:
    (
        times,
        cov_names_items,
        clock,
        nominal_dt,
        prep_stats,
        snapshot_ids,
        support,
        n_categories,
        boundaries,
        group,
        series_support,
    ) = aux
    observations, dt, mask, cov_transition, cov_duration, cov_emission, labels = children
    assert observations is not None
    assert dt is not None
    assert mask is not None
    return PreparedDataset(
        times=times.array,
        observations=observations,
        dt=dt,
        mask=mask,
        cov_transition=cov_transition,
        cov_duration=cov_duration,
        cov_emission=cov_emission,
        cov_names=dict(cov_names_items),
        clock=clock,
        nominal_dt=nominal_dt,
        prep_stats=prep_stats,
        snapshot_ids=snapshot_ids,
        support=support,
        n_categories=n_categories,
        labels=labels,
        boundaries=boundaries,
        group=group,
        series_support=series_support,
    )


jax.tree_util.register_pytree_node(PreparedDataset, _flatten_prepared_dataset, _unflatten_prepared_dataset)
