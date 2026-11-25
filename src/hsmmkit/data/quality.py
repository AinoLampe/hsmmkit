"""Domain-neutral data-quality checks (E10-T05, FR-D06).

Run on the raw observations of a source, before the master grid is built:

- duplicated timestamps,
- gaps in the time index (spacing `> 1.5 x` the median spacing; needs timestamps),
- outliers `|y - median| > 8 x MAD` per series (flag only; the MAD is raw, not rescaled),
- the staleness histogram of the covariates (from the preparer's `PrepDiagnostics`).

An application adds its own checks on top (the finance application: weekday gaps, non-positive values,
adjustment discontinuities).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from regimekit.data.contracts import ObservationSet

GAP_FACTOR = 1.5
OUTLIER_MADS = 8.0


def iso(t: np.datetime64 | pd.Timestamp) -> str:
    """A timestamp as `YYYY-MM-DD` (midnight) or ISO-8601."""
    ts = pd.Timestamp(t)
    return str(ts.date()) if ts == ts.normalize() else ts.isoformat()


@dataclass(frozen=True)
class GenericQuality:
    """Findings of the generic checks; every finding carries its index or timestamp."""

    n_rows: int
    duplicates: list[str] = field(default_factory=list)
    index_gaps: list[str] = field(default_factory=list)
    outliers: list[dict[str, Any]] = field(default_factory=list)
    staleness: dict[str, dict[str, int]] = field(default_factory=dict)


def staleness_histogram(staleness: np.ndarray) -> dict[str, int]:
    """Counts of grid steps since the print in use became visible; `no_value` where none was usable."""
    values = np.asarray(staleness)
    counts = {
        str(int(v)): int(n) for v, n in zip(*np.unique(values[values >= 0], return_counts=True), strict=True)
    }
    never = int((values < 0).sum())
    if never:
        counts["no_value"] = never
    return counts


def duplicate_times(times: np.ndarray) -> list[str]:
    """The timestamps that occur more than once (listed once each)."""
    index = pd.DatetimeIndex(times)
    return [iso(t) for t in index[index.duplicated(keep="first")].unique()]


def index_gaps(times: np.ndarray) -> list[str]:
    """Timestamps that follow a gap of more than `1.5 x` the median spacing (the first one after the gap)."""
    if times.shape[0] < 3:
        return []
    stamps = np.asarray(times).astype("datetime64[ns]")
    steps = np.diff(stamps).astype(np.int64)
    positive = steps[steps > 0]
    if positive.size == 0:
        return []
    limit = GAP_FACTOR * float(np.median(positive))
    return [iso(stamps[i + 1]) for i in np.flatnonzero(steps > limit)]


def mad_outliers(obs: ObservationSet) -> list[dict[str, Any]]:
    """Values with `|y - median| > 8 x MAD` per series; a series with MAD 0 gets no check."""
    found: list[dict[str, Any]] = []
    values = np.asarray(obs.values, dtype=np.float64)
    for c, name in enumerate(obs.series_names):
        column = values[:, c]
        finite = np.isfinite(column)
        if finite.sum() < 3:
            continue
        median = float(np.median(column[finite]))
        mad = float(np.median(np.abs(column[finite] - median)))
        if mad <= 0:
            continue
        for r in np.flatnonzero(finite & (np.abs(column - median) > OUTLIER_MADS * mad)):
            where = iso(obs.times[r]) if obs.times is not None else str(int(r))
            found.append(
                {
                    "at": where,
                    "series": name,
                    "value": float(column[r]),
                    "mads": float(abs(column[r] - median) / mad),
                }
            )
    return sorted(found, key=lambda o: (o["at"], o["series"]))


def generic_checks(obs: ObservationSet, staleness: Mapping[str, np.ndarray] | None = None) -> GenericQuality:
    """Run the generic checks on `obs`; `staleness` maps covariate names to their staleness on the grid."""
    has_times = obs.times is not None
    return GenericQuality(
        n_rows=int(obs.values.shape[0]),
        duplicates=duplicate_times(np.asarray(obs.times)) if has_times else [],
        index_gaps=index_gaps(np.asarray(obs.times)) if has_times else [],
        outliers=mad_outliers(obs),
        staleness={name: staleness_histogram(s) for name, s in (staleness or {}).items()},
    )
