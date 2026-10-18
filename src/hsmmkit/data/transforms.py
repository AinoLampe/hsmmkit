"""Pure observation and covariate transforms used by the `Preparer` (FR-D05).

`log_returns` is the finance profile's observation transform; it is a standalone pure function so
Part II can move it into the profile without touching the `Preparer`. Covariate transforms act on a
series' own prints, before alignment onto the grid (E2-T06), in declared order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from regimekit.utils.errors import DataError


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log-returns `y_t = log(P_t / P_{t-1})` per series.

    Args:
        prices: `(T+1, M)` prices on the master grid.

    Returns:
        `(T, M)` float64 log-returns. Non-positive prices give `±inf`/NaN; the caller masks them.
    """
    p = np.asarray(prices, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p[1:]) - np.log(p[:-1])


def _apply_one(x: np.ndarray, name: str, path: str | None) -> np.ndarray:
    if name == "log":
        if np.any(x <= 0):
            raise DataError("transform 'log' needs strictly positive values", path=path)
        return np.log(x)
    if name == "diff":
        out = np.full_like(x, np.nan)
        out[1:] = x[1:] - x[:-1]
        return out
    if name == "pct":
        if np.any(x[:-1] == 0):
            raise DataError(
                "transform 'pct' needs non-zero values (division by the previous print)", path=path
            )
        out = np.full_like(x, np.nan)
        out[1:] = x[1:] / x[:-1] - 1.0
        return out
    if name == "zscore":
        # Deprecated alias of standardization: handled by the Preparer with train-window stats.
        return x
    raise DataError(f"unknown transform {name!r}", path=path)


def apply_transforms(values: np.ndarray, transforms: Sequence[str], *, path: str | None = None) -> np.ndarray:
    """Apply covariate transforms in declared order to a series' prints.

    Non-finite prints (NaN such as FRED holidays, or ±inf) are skipped: `diff`/`pct` are taken
    between consecutive finite prints, and the skipped positions become NaN. The first valid print
    becomes NaN after `diff`/`pct` (no predecessor).

    Args:
        values: `(T1,)` raw values in time order.
        transforms: any sequence of `log`, `diff`, `pct` (`x_t / x_{t-1} - 1`), `zscore` (no-op here).
        path: config path for error messages (`data.covariates[i].transform`).

    Returns:
        `(T1,)` float64 transformed values.

    Raises:
        DataError: `log` of a non-positive value, `pct` after a zero print, or an unknown transform.
    """
    x = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x)
    compact = x[valid]
    for name in transforms:
        kept = ~np.isnan(compact)
        step = np.full_like(compact, np.nan)
        step[kept] = _apply_one(compact[kept], name, path)
        compact = step
    out = np.full_like(x, np.nan)
    out[valid] = compact
    return out
