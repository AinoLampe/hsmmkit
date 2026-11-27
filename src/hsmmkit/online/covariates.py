"""`OnlineCovariatePreparer`: raw covariate prints -> the lagged, standardized rows of the filter (E15-T01).

Applies, print by print, what the offline `Preparer` does to a covariate: the stored transforms (`log`,
`diff`, `pct`; `zscore` is a no-op there), the as-of alignment with lag `>= 1` and the stale limit, and the
standardization with the fitted `prep_stats` (`(x - cov_mean) / cov_std`). Prints are dated by their grid
index `j` (time-less / step alignment, `data.calendar._align_by_row`): row `t` takes the last non-NaN print
with index `<= t + 1 - lag`, carried at most `max_staleness` steps. Only `O(lag)` history is kept.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from hsmmkit.utils.errors import ConfigError, DataError

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PrepStats


class OnlineCovariatePreparer:
    """Stateful per-print covariate preparation.

    Args:
        prep_stats: the fitted `PrepStats` (names, transforms, lags, mean, std).
        max_staleness: grid steps a print may be carried forward; one value for all covariates, a sequence
            per covariate, or `None` for no limit.
        entry_points: optional `{"transition" | "duration" | "emission": [covariate indices]}` (the
            `use_in` of `data.covariates`); with it `z(t)` returns the dict the filter expects.
    """

    def __init__(
        self,
        prep_stats: PrepStats,
        *,
        max_staleness: int | Sequence[int] | None = None,
        entry_points: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        """Validate the stored lags and set up empty per-covariate histories."""
        self.names = tuple(prep_stats.cov_names)
        k = len(self.names)
        self._transforms = [list(prep_stats.transforms.get(n, [])) for n in self.names]
        self._lags = [int(prep_stats.lags.get(n, 1)) for n in self.names]
        for n, lag in zip(self.names, self._lags, strict=True):
            if lag < 1:
                raise ConfigError(f"covariate {n!r}: lag must be >= 1, got {lag}", path="prep_stats.lags")
        self._mean = np.asarray(prep_stats.cov_mean, dtype=np.float64)
        self._std = np.asarray(prep_stats.cov_std, dtype=np.float64)
        if isinstance(max_staleness, int) or max_staleness is None:
            self._stale: list[int | None] = [max_staleness] * k
        else:
            self._stale = list(max_staleness)
            if len(self._stale) != k:
                raise ConfigError("max_staleness needs one value per covariate", path="max_staleness")
        self._entry = None if entry_points is None else {e: tuple(i) for e, i in entry_points.items()}
        self._prev: list[list[float | None]] = [[None] * len(tr) for tr in self._transforms]
        # per covariate: states "as of print j" = (j, last valid transformed value, its index)
        self._hist: list[deque[tuple[int, float, int]]] = [deque(maxlen=lag) for lag in self._lags]
        self.n_pushed = 0

    def _transform(self, k: int, x: float) -> float:
        """The covariate's transforms applied to one print (NaN in = NaN out, stage states untouched)."""
        if not np.isfinite(x):
            return float("nan")
        for stage, name in enumerate(self._transforms[k]):
            prev = self._prev[k][stage]
            if name == "log":
                if x <= 0:
                    raise DataError("transform 'log' needs strictly positive values", path=self.names[k])
                out = float(np.log(x))
            elif name in ("diff", "pct"):
                if name == "pct" and prev == 0:
                    raise DataError("transform 'pct' needs non-zero values", path=self.names[k])
                out = float("nan") if prev is None else (x - prev if name == "diff" else x / prev - 1.0)
                self._prev[k][stage] = x
            elif name == "zscore":
                out = x
            else:
                raise DataError(f"unknown transform {name!r}", path=self.names[k])
            if name == "log":
                self._prev[k][stage] = x
            x = out
            if np.isnan(x):
                return x
        return x

    def push(self, raw: Sequence[float] | np.ndarray) -> None:
        """Register the prints dated at the next grid index (`(K,)`, `NaN` = no print)."""
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(self.names):
            raise ConfigError(f"expected {len(self.names)} prints, got {values.shape[0]}", path="raw")
        j = self.n_pushed
        for k, x in enumerate(values):
            value = self._transform(k, float(x))
            hist = self._hist[k]
            if np.isfinite(value):
                hist.append((j, value, j))
            elif hist:
                _, last_value, last_idx = hist[-1]
                hist.append((j, last_value, last_idx))
            else:
                hist.append((j, float("nan"), -1))
        self.n_pushed += 1

    def row(self, t: int) -> np.ndarray:
        """The standardized lagged row `(K,)` for step `t` (0-based); `NaN` where no usable print exists.

        Raises:
            ConfigError: the prints up to index `t + 1 - lag` have not all been pushed, or have been dropped.
        """
        out = np.full(len(self.names), np.nan)
        for k, lag in enumerate(self._lags):
            source = t + 1 - lag
            if source < 0:
                continue
            if source >= self.n_pushed:
                raise ConfigError(
                    f"covariate {self.names[k]!r}: step {t} needs the print with index {source}, "
                    f"only {self.n_pushed} pushed",
                    path="t",
                )
            state = next((s for s in self._hist[k] if s[0] == source), None)
            if state is None:
                raise ConfigError(f"the print state for step {t} was already dropped", path="t")
            _, value, idx = state
            stale = self._stale[k]
            if idx < 0 or (stale is not None and source - idx > stale):
                continue
            out[k] = value
        return (out - self._mean) / self._std

    def z(self, t: int) -> dict[str, np.ndarray]:
        """`row(t)` split by entry point, as `ExpandedOnlineFilter.update` expects (needs `entry_points`)."""
        if self._entry is None:
            raise ConfigError("entry_points were not given", path="entry_points")
        r = self.row(t)
        return {e: r[list(idx)] for e, idx in self._entry.items()}
