"""`Posterior` (pytree-registered) and the `InferenceEngine` protocol stub (E1-T05).

`Posterior.engine` is a static `Literal["expanded", "explicit"]` tag (E4-T04); the utilities in
`engine/augmented.py` operate on `Posterior`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import jax

from hsmmkit.statespace.index import StateIndex
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.contracts import PreparedDataset
    from hsmmkit.model.params import Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace


@dataclass(frozen=True)
class _StaticRef:
    """Wraps an arbitrary object as identity-hashable pytree aux data.

    `StateIndex` mixes JAX-array leaves with static ints (equinox's own default `__eq__`/`__hash__`
    would try to compare/hash the array leaves too, hitting the same "numpy/JAX arrays are
    unhashable" gotcha as CLAUDE.md §5 -- see `data/contracts.py`'s `_HashableArray` for the same
    pattern). A `StateIndex` is built once per model and reused for the fit's lifetime, so
    identity (not structural) equality is the right tradeoff here: cheap, and correct because
    no two distinct-but-equal `StateIndex` instances are expected to coexist in practice.
    """

    value: Any

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StaticRef) and self.value is other.value

    def __hash__(self) -> int:
        return id(self.value)


@dataclass(frozen=True)
class Posterior:
    """Raw forward-backward output: log-alpha/beta, smoothed posterior, loglik, and the index."""

    log_alpha: Array  # (T, S)
    log_beta: Array  # (T, S)
    smoothed: Array  # (T, S)
    loglik: Array  # 0-d Array (a pytree leaf; PosteriorResult/FitResult.loglik are plain float)
    index: StateIndex
    engine: Literal["expanded", "explicit"] = "expanded"  # static tag (E4-T04; explicit set by E4-T06a)

    def regime_filtered(self) -> Array:
        """`softmax(log_alpha) @ C`, shape `(T, N)`."""
        return jax.nn.softmax(self.log_alpha, axis=-1) @ self.index.collapse

    def regime_smoothed(self) -> Array:
        """`smoothed @ C`, shape `(T, N)`."""
        return self.smoothed @ self.index.collapse

    def elapsed_posterior(self) -> Array:
        """`(T, N, max_phase)`: smoothed mass bucketed by `(regime, phase)` (see `augmented.phase_marginals`).

        `max_phase` is `max(r_i)` for the expanded engine's phase expansion and `D_max` for the
        explicit engine (where `phase_of` holds the elapsed duration); both are "how many
        phase/elapsed buckets a regime has", so one bucketing serves both `engine` tags.
        """
        from hsmmkit.engine.augmented import phase_marginals

        return phase_marginals(self)


def _flatten_posterior(posterior: Posterior) -> tuple[tuple[Array, ...], tuple[_StaticRef, str]]:
    children = (posterior.log_alpha, posterior.log_beta, posterior.smoothed, posterior.loglik)
    return children, (_StaticRef(posterior.index), posterior.engine)


def _unflatten_posterior(aux: tuple[_StaticRef, str], children: tuple[Array, ...]) -> Posterior:
    log_alpha, log_beta, smoothed, loglik = children
    ref, engine = aux
    return Posterior(
        log_alpha=log_alpha,
        log_beta=log_beta,
        smoothed=smoothed,
        loglik=loglik,
        index=ref.value,
        engine=engine,  # type: ignore[arg-type]
    )


jax.tree_util.register_pytree_node(Posterior, _flatten_posterior, _unflatten_posterior)


class InferenceEngine(Protocol):
    """Structural type for the expanded/explicit-duration inference engines (E4-T03*/E4-T06*)."""

    def loglik(self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
        """`log p(y_1:T)`, a 0-d `Array` (a pytree leaf)."""
        ...

    def forward(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> tuple[Array, Array]:
        """`(log_alpha (T, S), loglik)`."""
        ...

    def forward_backward(
        self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset
    ) -> Posterior:
        """Full forward-backward, returning a `Posterior`."""
        ...

    def viterbi(self, theta: Params, spec: ModelSpec, space: StateSpace, data: PreparedDataset) -> Array:
        """The most likely expanded-state path, shape `(T,)`."""
        ...

    def approx(self) -> object | None:
        """Approximate-inference plug-in (FR-X01, ADR-0008); `None` for exact engines."""
        ...
