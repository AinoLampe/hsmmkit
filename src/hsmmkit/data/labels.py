"""Regime labels and boundary constraints as data (E12-T06, FR-SV01, FR-SV02).

`data.labels` gives full or partial supervision. Three encodings reach the `Preparer` as one raw value per
observation row (`ObservationSet.meta["label_values"]`): a column of regime ids (`NaN` / `-1` = unlabelled), a
column of allowed sets (`allowed_sets: true`: a bitmask int, or a list-valued column), or a label file
(`sequence_id`, `t`, then `regime` or `allowed`; `t` is the 0-based row within the sequence). `Preparer` turns
them into `PreparedDataset.labels (T, N)` bool: one `True` per labelled row, all `True` where unlabelled.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hsmmkit.data.contracts import BoundaryConstraints, ObservationSet
from hsmmkit.utils.errors import DataError

__all__ = ["allowed_from_values", "attach_label_file", "boundaries_of"]

_PATH = "data.labels"


def _is_missing(v: Any) -> bool:  # noqa: ANN401
    if v is None:
        return True
    if isinstance(v, float | np.floating):
        return math.isnan(float(v))
    return bool(v is pd.NA)


def _as_int(v: Any, what: str) -> int:  # noqa: ANN401
    f = float(v)
    if not f.is_integer():
        raise DataError(f"{what} must be an integer, got {v!r}", path=_PATH)
    return int(f)


def allowed_from_values(
    values: Sequence[Any], n_regimes: int, *, allowed_sets: bool = False, path: str = _PATH
) -> np.ndarray:
    """`(T, N)` bool allowed regimes from raw label values (an unlabelled row allows every regime).

    Raises:
        DataError: an id outside `[0, N)`, a set with a regime outside `[0, N)`, or an empty set.
    """
    out = np.ones((len(values), n_regimes), dtype=bool)
    for i, v in enumerate(values):
        if isinstance(v, str | bytes):
            raise DataError(f"label of row {i} is a string ({v!r}); expected regime ids", path=path)
        if not isinstance(v, list | tuple | set | np.ndarray) and _is_missing(v):
            continue
        if allowed_sets:
            if isinstance(v, list | tuple | set | np.ndarray):
                ids = {_as_int(x, f"label set of row {i}") for x in v}
            else:
                mask = _as_int(v, f"label set of row {i}")
                if mask == -1:
                    continue
                if mask < 0:
                    raise DataError(f"label set of row {i} is a negative bitmask ({mask})", path=path)
                ids = {k for k in range(max(mask.bit_length(), 1)) if (mask >> k) & 1}
            if not ids:
                raise DataError(f"label set of row {i} is empty", path=path)
            if min(ids) < 0 or max(ids) >= n_regimes:
                raise DataError(
                    f"label set of row {i} names regimes {sorted(ids)} outside [0, {n_regimes})", path=path
                )
            out[i] = False
            out[i, sorted(ids)] = True
        else:
            k = _as_int(v, f"label of row {i}")
            if k == -1:
                continue
            if not 0 <= k < n_regimes:
                raise DataError(f"label {k} of row {i} is outside [0, {n_regimes})", path=path)
            out[i] = False
            out[i, k] = True
    return out


def attach_label_file(obs: ObservationSet, path: Path | str) -> ObservationSet:
    """`obs` with `meta["label_values"]` read from a label file (parquet or CSV).

    The file has the columns `t` (0-based row within the sequence), `regime` or `allowed`, and `sequence_id`
    (optional for a single-sequence data set). Rows of `obs` without an entry are unlabelled.

    Raises:
        DataError: an unreadable file, missing columns, or an entry that matches no row of `obs`.
    """
    p = Path(path)
    try:
        frame = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read the label file {p}: {exc}", path="data.labels.path") from exc
    value_col = "regime" if "regime" in frame.columns else "allowed" if "allowed" in frame.columns else None
    if "t" not in frame.columns or value_col is None:
        raise DataError(
            f"the label file {p} needs the columns 't' and 'regime' (or 'allowed'), "
            f"has {list(frame.columns)}",
            path="data.labels.path",
        )
    n_rows = obs.values.shape[0]
    ids = np.asarray(obs.sequence_ids).astype(str) if obs.sequence_ids is not None else np.full(n_rows, "0")
    position = np.zeros(n_rows, dtype=np.int64)
    seen: dict[str, int] = {}
    for i, sid in enumerate(ids):
        position[i] = seen.get(sid, 0)
        seen[sid] = position[i] + 1
    file_ids = (
        frame["sequence_id"].astype(str)
        if "sequence_id" in frame.columns
        else pd.Series("0", index=frame.index)
    )
    entries = {
        (str(s), int(t)): v for s, t, v in zip(file_ids, frame["t"], frame[value_col].tolist(), strict=True)
    }
    values: list[Any] = [entries.pop((ids[i], int(position[i])), None) for i in range(n_rows)]
    if entries:
        key = next(iter(entries))
        raise DataError(
            f"{len(entries)} label entries match no observation row (first: sequence {key[0]!r}, t={key[1]})",
            path="data.labels.path",
        )
    return dataclasses.replace(obs, meta={**obs.meta, "label_values": values})


def boundaries_of(
    must_switch_at: Sequence[int], must_not_switch_at: Sequence[int], t_len: int
) -> BoundaryConstraints | None:
    """The boundary indices that fall inside a sequence of `t_len` steps (`1 <= t <= t_len - 1`), else `None`.

    Indices outside the sequence are dropped (a configured list applies to every sequence of a batch).

    Raises:
        DataError: an index in both lists, or an index below 1.
    """
    if any(i < 1 for i in (*must_switch_at, *must_not_switch_at)):
        raise DataError("boundary steps must be >= 1 (a switch is between steps t-1 and t)", path=_PATH)
    both = set(must_switch_at) & set(must_not_switch_at)
    if both:
        raise DataError(f"step(s) {sorted(both)} are both must_switch_at and must_not_switch_at", path=_PATH)
    sw = tuple(sorted(i for i in set(must_switch_at) if i < t_len))
    ns = tuple(sorted(i for i in set(must_not_switch_at) if i < t_len))
    return BoundaryConstraints(sw, ns) if sw or ns else None
