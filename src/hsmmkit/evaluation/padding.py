"""Pad a `PreparedDataset` to a multiple of 256 rows with masked dummy rows (E7-T02b).

A walk-forward run fits and filters windows of many different lengths. Every distinct `T` is a new
compilation of the jitted engine functions, so each window is padded at the *end* to the next multiple of
`PAD_MULTIPLE` rows. A padded row has `mask=False` (so it adds zero log-density), `observations=0`, the
last row's `dt` (positive, and exactly 1 under the `steps` clock), zero covariates and a time that
continues the grid's median step. The engines apply the transition kernel `A_t` to a masked row without an
emission term; every row of `A_t` sums to one, so the log-likelihood of the padded series equals the
unpadded one (tested to 1e-8), and the filter rows `0..T-1` are untouched (the filter is causal).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from regimekit.data.contracts import PreparedDataset
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

PAD_MULTIPLE = 256

_FALLBACK_STEP = np.timedelta64(1, "D")


def padded_length(n_rows: int, multiple: int = PAD_MULTIPLE) -> int:
    """The smallest multiple of `multiple` that is `>= n_rows` (and at least one multiple)."""
    if multiple < 1:
        raise ConfigError(f"padding multiple must be >= 1, got {multiple}", path="evaluation.padding")
    return max(multiple, -(-n_rows // multiple) * multiple)


def pad_dataset(data: PreparedDataset, multiple: int = PAD_MULTIPLE) -> PreparedDataset:
    """`data` with masked dummy rows appended so that `T` is a multiple of `multiple`.

    Returns `data` itself when `T` already is one. Static fields (`prep_stats`, `snapshot_ids`,
    `nominal_dt`, `boundaries`, ...) are kept; `labels` get `False` rows.
    """
    t = int(data.times.shape[0])
    extra = padded_length(t, multiple) - t
    if extra == 0:
        return data

    times = np.asarray(data.times).astype("datetime64[ns]")
    step = np.median(np.diff(times)).astype("m8[ns]") if t > 1 else _FALLBACK_STEP.astype("m8[ns]")
    new_times = np.concatenate([times, times[-1] + step * np.arange(1, extra + 1)])

    def rows(block: Array, fill: float | bool, dtype: DTypeLike) -> Array:
        arr = jnp.asarray(block)
        pad = jnp.full((extra, *arr.shape[1:]), fill, dtype=dtype)
        return jnp.concatenate([arr, pad], axis=0)

    obs = jnp.asarray(data.observations)
    dt = jnp.asarray(data.dt)

    def cov(block: Array | None) -> Array | None:
        return None if block is None else rows(block, 0.0, jnp.asarray(block).dtype)

    return dataclasses.replace(
        data,
        times=new_times,
        observations=rows(obs, 0, obs.dtype),
        dt=jnp.concatenate([dt, jnp.full((extra,), dt[-1], dtype=dt.dtype)]),
        mask=rows(data.mask, False, jnp.bool_),
        cov_transition=cov(data.cov_transition),
        cov_duration=cov(data.cov_duration),
        cov_emission=cov(data.cov_emission),
        labels=None if data.labels is None else rows(data.labels, False, jnp.bool_),
    )
