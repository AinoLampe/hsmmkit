"""Transition topology `Omega`: which regime-to-regime jumps are allowed (E12-T03, FR-DU01, FR-DU02, D16).

`Topology.allowed (N, N)` is the boolean jump matrix (diagonal `False`: a regime never jumps to itself, the
duration family models staying); `absorbing` lists the regimes with no allowed target. Shorthands
(`model.topology`): `full` (every off-diagonal), `left_to_right` (`j > i`, the last regime is absorbing),
`cyclic` (`j = (i + 1) mod N`), `{banded: k}` (`0 < |j - i| <= k`) and `{matrix: [[0/1]]}` as given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hsmmkit.utils.errors import ConfigError

if TYPE_CHECKING:
    from hsmmkit.config.schema import BandedCfg, MatrixCfg

__all__ = ["Topology", "from_config"]


@dataclass(frozen=True, eq=False)
class Topology:
    """The allowed jumps `allowed (N, N)` (diagonal `False`) and the absorbing regimes."""

    allowed: np.ndarray
    absorbing: tuple[int, ...]

    def __eq__(self, other: object) -> bool:
        """Structural equality (the numpy array makes the dataclass default unusable)."""
        if not isinstance(other, Topology):
            return NotImplemented
        return bool(np.array_equal(self.allowed, other.allowed))

    def __hash__(self) -> int:
        """Content-based hash (a `ModelSpec` field is a static jit argument)."""
        return hash((self.allowed.shape, self.allowed.tobytes()))

    @property
    def n_states(self) -> int:
        """`N`."""
        return int(self.allowed.shape[0])

    @property
    def is_full(self) -> bool:
        """Every off-diagonal jump is allowed (Part I behaviour)."""
        n = self.n_states
        return bool(np.array_equal(self.allowed, ~np.eye(n, dtype=bool)))

    def n_targets(self) -> np.ndarray:
        """`(N,)` number of allowed targets of each regime."""
        return self.allowed.sum(axis=1)

    def n_free_per_row(self) -> np.ndarray:
        """`(N,)` free logits of each regime's row: `max(0, |A_i| - 1)` (Appendix F)."""
        return np.asarray(np.maximum(self.n_targets() - 1, 0))


def _of(allowed: np.ndarray) -> Topology:
    absorbing = tuple(int(i) for i in np.flatnonzero(~allowed.any(axis=1)))
    return Topology(allowed=allowed, absorbing=absorbing)


def from_config(cfg: str | BandedCfg | MatrixCfg, n_states: int) -> Topology:
    """The `Topology` of `model.topology` for `n_states` regimes.

    Raises:
        ConfigError: an unknown shorthand, a matrix of the wrong shape, a non-zero diagonal or a non 0/1 entry
            (`path="model.topology"`).
    """
    n = n_states
    idx = np.arange(n)
    off = idx[None, :] != idx[:, None]
    if isinstance(cfg, str):
        if cfg == "full":
            return _of(off.copy())
        if cfg == "left_to_right":
            return _of(idx[None, :] > idx[:, None])
        if cfg == "cyclic":
            return _of(idx[None, :] == (idx[:, None] + 1) % n)
        raise ConfigError(f"unknown topology {cfg!r}", path="model.topology")
    banded = getattr(cfg, "banded", None)
    if banded is not None:
        return _of(off & (np.abs(idx[None, :] - idx[:, None]) <= banded))
    matrix = getattr(cfg, "matrix", None)
    if matrix is None:
        raise ConfigError(f"unknown topology {cfg!r}", path="model.topology")
    arr = np.asarray(matrix)
    if arr.shape != (n, n):
        raise ConfigError(f"matrix must be {n} x {n}", path="model.topology.matrix")
    if not bool(np.all((arr == 0) | (arr == 1))):
        raise ConfigError("matrix entries must be 0 or 1", path="model.topology.matrix")
    if bool(np.any(np.diag(arr) != 0)):
        raise ConfigError("matrix must have a zero diagonal", path="model.topology.matrix")
    return _of(arr.astype(bool))
