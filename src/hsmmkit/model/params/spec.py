"""`ParamSpec`: a single named parameter block's layout and estimation mode (E1-T03b)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from hsmmkit.utils.typing import Array

Mode = Literal["free", "fixed", "tied", "bounded", "prior"]
Domain = Literal["real", "positive", "interval", "simplex_ref", "nu_gt_2", "pmf", "cholesky"]


class Prior(Protocol):
    """Structural type for the prior objects (`NormalPrior`, `LogNormalPrior`, `DirichletPrior`)."""

    def log_prob(self, theta: Array) -> Array:
        """Summed log-density of `theta` under this prior."""
        ...


@dataclass(frozen=True, eq=False)
class ParamSpec:
    """A single named parameter's shape, domain, and per-state (or uniform) estimation mode.

    Args:
        name: Dotted parameter name, e.g. `"emission.sigma"`.
        shape: Full shape of `theta[name]`, e.g. `(N,)` or `(N, K)`. The leading axis (if any) is
            the "state" axis that `mode`, when given per-state, indexes.
        domain: Bijector domain for every free/tied/bounded/prior entry.
        mode: One mode for the whole spec, or one per leading-index (state).
        value: Required wherever `mode == "fixed"`; broadcast/sliced to match `shape`.
        init: Initial `theta` wherever `mode != "fixed"`; used by `ParamManager.init_u`.
        bounds: `(lo, hi)`, required when `domain == "interval"`.
        prior: Evaluated on entries where `mode == "prior"`.
        tie_group: Entries (within or across specs) sharing this string share one `u` entry.
        constraint: `"increasing"` applies the `increasing` bijector across the whole state axis
            (requires `shape = (N,)`, a single uniform `mode != "fixed"`, no `tie_group`).
    """

    name: str
    shape: tuple[int, ...]
    domain: Domain
    mode: Mode | tuple[Mode, ...]
    value: np.ndarray | None = None
    init: np.ndarray | None = None
    bounds: tuple[float, float] | None = None
    prior: Prior | None = None
    tie_group: str | None = None
    constraint: Literal["none", "increasing"] = "none"
