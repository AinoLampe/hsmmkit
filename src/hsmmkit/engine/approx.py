"""`ApproxInference`: the approximate-inference plug-in protocol for factorial models (E9-T02, ADR-0008).

Design only: no engine implements it in 0.1.x, and `InferenceEngine.approx()` of the exact engines still
returns `None`. A future factorial engine returns an `ApproxInference` from `approx()`; callers then run
`fit(...)` to obtain per-chain marginals instead of the exact joint `Posterior`. The accepted algorithm
(structured mean-field) and its complexity are recorded in `docs/adr/0008-approx-inference-plugin.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from regimekit.data.contracts import PreparedDataset
    from regimekit.engine.base import Posterior
    from regimekit.model.params import Params
    from regimekit.statespace.factorial import FactorialSpace
    from regimekit.utils.typing import Array


@dataclass(frozen=True)
class ApproxPosterior:
    """Result of an approximate inference sweep.

    Attributes:
        marginals: per chain `c`, the approximate smoothed regime marginals `(T, N_c)` (rows sum to 1).
        bound: a lower bound on `log p(y_1:T)` (the ELBO for structured mean-field), 0-d.
        n_sweeps: coordinate-ascent sweeps (or Gibbs sweeps) actually run.
    """

    marginals: tuple[Array, ...]
    bound: Array
    n_sweeps: int


class ApproxInference(Protocol):
    """Approximate posterior over a factorial model (several regime chains)."""

    def fit(
        self,
        posterior_init: Posterior | None,
        theta: Params,
        factorial: FactorialSpace,
        data: PreparedDataset,
    ) -> ApproxPosterior:
        """Run the approximation, optionally warm-started from an exact `Posterior` (e.g. of a smaller model).

        Per-chain marginals are returned in chain order; the joint is never materialised.
        """
        ...
