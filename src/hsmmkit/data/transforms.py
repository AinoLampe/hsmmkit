"""Observation transforms (registry kind `transform`) and covariate transforms of the `Preparer` (FR-D05).

The core registers `identity` and `diff`; an application registers more (the finance application: a
log-ratio transform). Covariate transforms act on a series' own prints, before alignment onto the grid
(E2-T06), in declared order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from hsmmkit.plugins import REGISTRY, provenance_of
from hsmmkit.utils.errors import ConfigError, DataError


class ObservationTransform(Protocol):
    """A registered observation transform (registry kind `transform`).

    Called with the `(T0, M)` values and their `(T0,)` times (`None` for time-less data); gives back the
    transformed values and the matching times. A transform that shortens the series drops leading rows only
    (the first row has no predecessor) and gives back the shortened times.
    """

    name: str

    def __call__(self, values: np.ndarray, times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        """Transform `values`; the result has `T0 - k` rows and `times[k:]` for some `k >= 0`."""
        ...


class IdentityTransform:
    """`y_t = x_t`: the observations pass through unchanged (the default)."""

    name = "identity"

    def __call__(self, values: np.ndarray, times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        """Return `values` and `times` as they are."""
        return values, times


class DiffTransform:
    """`y_t = x_t - x_{t-1}`; the first row is dropped."""

    name = "diff"

    def __call__(self, values: np.ndarray, times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        """First differences per series, in float64."""
        x = np.asarray(values, dtype=np.float64)
        return x[1:] - x[:-1], None if times is None else times[1:]


REGISTRY.register(
    "transform", IdentityTransform.name, IdentityTransform(), provenance=provenance_of(__name__)
)
REGISTRY.register("transform", DiffTransform.name, DiffTransform(), provenance=provenance_of(__name__))


def get_observation_transform(name: str | None) -> ObservationTransform:
    """The registered transform `name` (`None` = `identity`).

    Raises:
        ConfigError: unknown name (`path="data.preparation.observation_transform"`).
    """
    chosen = name or "identity"
    if not REGISTRY.has("transform", chosen):
        listing = ", ".join(f"{e.name} ({e.provenance})" for e in REGISTRY.entries("transform"))
        raise ConfigError(
            f"unknown observation transform '{chosen}'; registered: {listing}",
            path="data.preparation.observation_transform",
        )
    transform: ObservationTransform = REGISTRY.get("transform", chosen)
    return transform


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
