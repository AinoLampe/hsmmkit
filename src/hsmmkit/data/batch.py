"""Multi-sequence data: `SequenceBatch` and its padded, bucketed view `PaddedBatch` (E12-T01a, ADR-0021).

Many independent sequences of different lengths are kept as single-sequence `PreparedDataset`s (the contracts
have no batch dimension); `SequenceBatch.padded()` stacks them into `(B, T_max, ...)` arrays for the batched
engines of E12-T01b. Shapes are bucketed so that compiled shapes do not vary with the data:

* `T_max` is the smallest bucket edge `>=` the longest sequence (else the longest length rounded up to a
  multiple of the last edge);
* `B` is padded up to the next power of two with **empty sequences** (`valid` all `False`, length 0).

Padding semantics (engines, E12-T01b): a step with `valid == False` has emission log-density 0 and
`A_t = I`; this module only provides `valid`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.data.contracts import CovariateSeries, ObservationSet, PreparedDataset
from hsmmkit.utils.errors import ConfigError, DataError
from hsmmkit.utils.typing import Array

__all__ = ["PaddedBatch", "SequenceBatch", "split_sequences"]


def split_sequences(
    obs: ObservationSet, covs: Sequence[CovariateSeries]
) -> list[tuple[str, ObservationSet, list[CovariateSeries]]]:
    """Group the rows of `obs` by `obs.sequence_ids` (first-appearance order) into one data set per sequence.

    Without `sequence_ids` the result is the single pair `("0", obs, covs)`. Covariate series must be
    row-aligned columns of the source (one value per observation row) to be split.

    Raises:
        ConfigError: a covariate series that is not aligned with the observation rows.
    """
    if obs.sequence_ids is None:
        return [("0", obs, list(covs))]
    ids = np.asarray(obs.sequence_ids)
    uniq, first = np.unique(ids, return_index=True)
    order = uniq[np.argsort(first)]
    out: list[tuple[str, ObservationSet, list[CovariateSeries]]] = []
    for sid in order:
        rows = np.flatnonzero(ids == sid)
        sub_obs = dataclasses.replace(
            obs,
            times=None if obs.times is None else obs.times[rows],
            values=obs.values[rows],
            sequence_ids=None,
        )
        sub_covs: list[CovariateSeries] = []
        for c in covs:
            if np.asarray(c.values).shape[0] != ids.shape[0]:
                raise ConfigError(
                    f"covariate '{c.name}' has {np.asarray(c.values).shape[0]} values for {ids.shape[0]} "
                    "observation rows: covariates of a multi-sequence data set must be row-aligned columns",
                    path="data.covariates",
                )
            sub_covs.append(
                dataclasses.replace(
                    c,
                    times=None if obs.times is None else obs.times[rows],
                    values=np.asarray(c.values)[rows],
                )
            )
        group_values = obs.meta.get("group_values")
        if group_values is not None:
            groups = np.unique(np.asarray(group_values, dtype=object)[rows].astype(str))
            if groups.shape[0] != 1:
                raise ConfigError(
                    f"the group column varies within sequence {sid!r} ({groups.tolist()})",
                    path="data.sequences.group_by",
                )
            meta = {k: v for k, v in obs.meta.items() if k != "group_values"}
            sub_obs = dataclasses.replace(sub_obs, meta={**meta, "group": str(groups[0])})
        out.append((str(sid), sub_obs, sub_covs))
    return out


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PaddedBatch:
    """The padded array view of a `SequenceBatch` (a pytree; every leaf is a `jnp` array or `None`).

    Attributes:
        observations: `(B, T_max, M)`, padded with 0.
        mask: `(B, T_max, M)`, `False` on padding.
        valid: `(B, T_max)`, `True` for `t < T_b`.
        dt: `(B, T_max)`; padding steps carry `nominal_dt` (a positive dummy).
        cov_transition, cov_duration, cov_emission: `(B, T_max, K)`, 0 on padding (`K = 0` when absent).
        labels: `(B, T_max, N)` or `None`; `True` on padding (everything allowed).
        lengths: `(B,)` int32 (0 for the empty padding sequences).
        group_index: `(B,)` int32 initial-distribution group (0 if there are no groups).
    """

    observations: Array
    mask: Array
    valid: Array
    dt: Array
    cov_transition: Array
    cov_duration: Array
    cov_emission: Array
    labels: Array | None
    lengths: Array
    group_index: Array


def _bucket(max_len: int, edges: tuple[int, ...]) -> int:
    """Smallest edge `>= max_len`; else `max_len` rounded up to a multiple of the last edge."""
    if not edges:
        return max_len
    for e in edges:
        if e >= max_len:
            return e
    return -(-max_len // edges[-1]) * edges[-1]


def _next_pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


@dataclass(frozen=True)
class SequenceBatch:
    """`B >= 1` independent sequences (same `M`, covariate layout and clock) with their bucket edges.

    Attributes:
        sequences: the single-sequence datasets.
        bucket_edges: strictly increasing time-length edges (may be empty: `T_max` is the longest length).
        ids: the sequence ids, in input order.
    """

    sequences: tuple[PreparedDataset, ...]
    bucket_edges: tuple[int, ...] = ()
    ids: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        """Validate the layout shared by all sequences."""
        if not self.sequences:
            raise DataError("a SequenceBatch needs at least one sequence", path="sequences")
        if any(b <= a for a, b in zip(self.bucket_edges, self.bucket_edges[1:], strict=False)) or any(
            e < 1 for e in self.bucket_edges
        ):
            raise ConfigError(
                "bucket_edges must be strictly increasing and >= 1", path="data.sequences.bucket_edges"
            )
        ids = self.ids or tuple(str(i) for i in range(len(self.sequences)))
        if len(ids) != len(self.sequences):
            raise DataError("ids and sequences differ in length", path="ids")
        object.__setattr__(self, "ids", tuple(ids))
        first = self.sequences[0]
        for i, s in enumerate(self.sequences):
            if self._layout(s) != self._layout(first):
                raise DataError(
                    f"sequence {self.ids[i]!r} differs from sequence {self.ids[0]!r} in series count, "
                    "covariate layout, clock or labels",
                    path=f"sequences[{i}]",
                )

    @staticmethod
    def _layout(s: PreparedDataset) -> tuple[object, ...]:
        def width(a: Array | None) -> int | None:
            return None if a is None else int(a.shape[1])

        return (
            int(s.observations.shape[1]),
            width(s.cov_transition),
            width(s.cov_duration),
            width(s.cov_emission),
            width(s.labels),
            s.clock,
            s.support,
            s.series_support,
        )

    @classmethod
    def of(cls, ds: PreparedDataset, bucket_edges: tuple[int, ...] = (), seq_id: str = "0") -> SequenceBatch:
        """A single `PreparedDataset` as a `B = 1` batch."""
        return cls((ds,), tuple(bucket_edges), (seq_id,))

    @property
    def lengths(self) -> np.ndarray:
        """`(B,)` sequence lengths."""
        return np.asarray([s.observations.shape[0] for s in self.sequences], dtype=np.int64)

    @property
    def t_max(self) -> int:
        """The bucketed time length of the padded view."""
        return _bucket(int(self.lengths.max()), self.bucket_edges)

    @property
    def b_pad(self) -> int:
        """`B` padded to the next power of two."""
        return _next_pow2(len(self.sequences))

    @property
    def group_names(self) -> tuple[str | None, ...]:
        """The distinct initial-distribution groups, sorted (`group_index` indexes this tuple)."""
        names = {s.group for s in self.sequences if s.group is not None}
        return tuple(sorted(names)) if names else (None,)

    def padded(self) -> PaddedBatch:
        """Stack the sequences into `(B_pad, T_max, ...)` arrays (see `PaddedBatch`)."""
        b, t_max, first = self.b_pad, self.t_max, self.sequences[0]
        m = int(first.observations.shape[1])
        obs_dtype = np.result_type(*[np.asarray(s.observations).dtype for s in self.sequences])
        observations = np.zeros((b, t_max, m), dtype=obs_dtype)
        mask = np.zeros((b, t_max, m), dtype=bool)
        valid = np.zeros((b, t_max), dtype=bool)
        dt = np.full((b, t_max), first.nominal_dt, dtype=np.float64)
        blocks = {}
        for name in ("transition", "duration", "emission"):
            block = getattr(first, f"cov_{name}")
            blocks[name] = np.zeros((b, t_max, 0 if block is None else int(block.shape[1])))
        n_labels = None if first.labels is None else int(first.labels.shape[1])
        labels = None if n_labels is None else np.ones((b, t_max, n_labels), dtype=bool)
        groups = {g: i for i, g in enumerate(self.group_names)}
        group_index = np.zeros(b, dtype=np.int32)
        for i, s in enumerate(self.sequences):
            n = int(s.observations.shape[0])
            observations[i, :n] = np.asarray(s.observations)
            mask[i, :n] = np.asarray(s.mask)
            valid[i, :n] = True
            dt[i, :n] = np.asarray(s.dt)
            for name, arr in blocks.items():
                src = getattr(s, f"cov_{name}")
                if src is not None:
                    arr[i, :n] = np.asarray(src)
            if labels is not None and s.labels is not None:
                labels[i, :n] = np.asarray(s.labels)
            group_index[i] = groups[s.group if s.group is not None else self.group_names[0]]
        lengths = np.zeros(b, dtype=np.int32)
        lengths[: len(self.sequences)] = self.lengths
        return PaddedBatch(
            observations=jnp.asarray(observations),
            mask=jnp.asarray(mask),
            valid=jnp.asarray(valid),
            dt=jnp.asarray(dt),
            cov_transition=jnp.asarray(blocks["transition"]),
            cov_duration=jnp.asarray(blocks["duration"]),
            cov_emission=jnp.asarray(blocks["emission"]),
            labels=None if labels is None else jnp.asarray(labels),
            lengths=jnp.asarray(lengths),
            group_index=jnp.asarray(group_index),
        )

    def unpad(self, arr: Array, *, time_axis: int = 1) -> tuple[Array, ...]:
        """`(B_pad, T_max, ...)` -> per-sequence `(T_b, ...)` arrays (the padding sequences are dropped).

        `time_axis` is the axis of `arr` that carries time (default 1, right after the batch axis).
        """
        if time_axis < 1:
            raise DataError("time_axis must be >= 1 (axis 0 is the batch)", path="time_axis")
        out = []
        for i, n in enumerate(self.lengths):
            row = arr[i]
            index = [slice(None)] * row.ndim
            index[time_axis - 1] = slice(0, int(n))
            out.append(row[tuple(index)])
        return tuple(out)
