"""EXPERIMENTAL: factorial state space for several regime chains (E9-T02 design spike, ADR-0008).

`build_factorial(chains, coupling)` composes `C` single-chain models into one `StateSpace` whose states are
the Cartesian product of the chains' expanded states (chain 0 varies slowest). For independent chains
(`coupling=None`) the joint transition matrix is the Kronecker product `A_t = A_t^(0) (x) ... (x) A_t^(C-1)`
(computed in log space as a broadcast sum, so `-inf` structural zeros never produce NaN). The joint space can
be run through the existing `ExpandedHMMEngine` unchanged; that is the exact-inference baseline the spike
checks. Cost is `O(T prod_c S_c^2)` time and `prod_c S_c` states, which is why ADR-0008 reserves an
approximate plug-in (`engine/approx.py`) for larger models.

Spike limits (deliberate, nothing here is production wiring):
* only order-1, `jump`-mode chains, `initial: uniform`, no transition/duration covariates;
* `StateIndex` is **not** changed: `regime_of` of the joint space is the flattened *joint* regime id, and the
  per-chain regimes live in `FactorialSpace.regime_of_chain (S, C)` (ADR-0008 proposes folding this into
  `StateIndex` once the design is adopted);
* the coupling hook is design-only: a user-supplied log interaction factor added to `log A_t` and
  renormalised per row; it is not estimated.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from hsmmkit.model.spec import ModelSpec
from hsmmkit.model.transition import build_memory_layout
from hsmmkit.statespace.index import StateIndex, StateSpace
from hsmmkit.utils.errors import ConfigError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.model.emission.base import EmissionFamily
    from hsmmkit.model.params import Params

EXPERIMENTAL = True


def chain_prefix(c: int) -> str:
    """Prefix of chain `c`'s parameters inside the joint `theta` (`"c0."`, `"c1."`, ...)."""
    return f"c{c}."


def chain_theta(theta: Params, c: int) -> Params:
    """The parameters of chain `c` with the prefix stripped."""
    p = chain_prefix(c)
    return {k[len(p) :]: v for k, v in theta.items() if k.startswith(p)}


@dataclass(frozen=True)
class CouplingSpec:
    """Design-only coupling between chains.

    Attributes:
        log_factor: `(theta, dt_t) -> (S, S)` log interaction factor added to the joint `log A_t` before the
            rows are renormalised. `None` means independent chains.
    """

    log_factor: Callable[[Params, Array], Array] | None = None

    def __hash__(self) -> int:
        """Identity hash (the callable is the content)."""
        return id(self.log_factor)


@dataclass(frozen=True)
class FactorialSpace:
    """Joint `StateSpace` plus the per-chain bookkeeping.

    Attributes:
        space: the joint space (usable by `ExpandedHMMEngine`).
        chains: the single-chain specs, in order.
        regime_of_chain: `(S, C)` regime of each chain in each joint state.
        chain_sizes: `(S_c,)` expanded-state counts per chain.
    """

    space: StateSpace
    chains: tuple[ModelSpec, ...]
    regime_of_chain: np.ndarray
    chain_sizes: tuple[int, ...]

    def chain_marginals(self, smoothed: Array) -> tuple[Array, ...]:
        """Per-chain regime marginals `(T, N_c)` from a joint smoothed posterior `(T, S)`."""
        out = []
        for c, chain in enumerate(self.chains):
            onehot = jnp.asarray(np.eye(chain.n_states)[self.regime_of_chain[:, c]])  # (S, N_c)
            out.append(smoothed @ onehot)
        return tuple(out)


@dataclass(frozen=True)
class JointEmission:
    """Emission of the joint regimes: the sum of per-chain emission log-densities (spike convention).

    `log_prob` returns `(T, prod N_c)`; joint regime `(r_0, ..., r_{C-1})` has id `ravel_multi_index`.
    """

    families: tuple[EmissionFamily, ...]
    chain_regimes: tuple[int, ...]
    name: str = "factorial"
    native_time: str = "continuous"
    support: str = "real"
    scale_by_dt: bool = True

    def log_prob(self, y: Array, dt: Array, theta: Params, z: Array | None) -> Array:
        """`(T, prod N_c)` summed per-chain log-densities (chain 0 slowest)."""
        total = jnp.zeros((y.shape[0],) + (1,) * len(self.families))
        for c, fam in enumerate(self.families):
            lp = fam.log_prob(y, dt, chain_theta(theta, c), z)  # (T, N_c)
            shape = [y.shape[0]] + [1] * len(self.families)
            shape[1 + c] = self.chain_regimes[c]
            total = total + lp.reshape(shape)
        return total.reshape(y.shape[0], -1)

    def init_carry(self) -> object:
        """No recurrent state (the spike chains use memoryless families)."""
        return None


def _joint_log_transition(
    spaces: tuple[StateSpace, ...],
    coupling: CouplingSpec | None,
) -> Callable[..., tuple[Array, Array]]:
    def transition_fn(
        theta: Params, z_pi_t: Array | None, z_d_t: Array | None, dt_t: Array
    ) -> tuple[Array, Array]:
        log_a: Array = jnp.zeros((1, 1))
        for c, sp in enumerate(spaces):
            _, la = sp.transition_fn(chain_theta(theta, c), z_pi_t, z_d_t, dt_t)  # (S_c, S_c)
            s_prev = log_a.shape[0]
            s_c = la.shape[0]
            # kron in log space: [(a, i), (b, j)] -> log_a[a, b] + la[i, j]
            log_a = (log_a[:, None, :, None] + la[None, :, None, :]).reshape(s_prev * s_c, s_prev * s_c)
        if coupling is not None and coupling.log_factor is not None:
            log_a = log_a + coupling.log_factor(theta, dt_t)
            m = jnp.max(log_a, axis=1, keepdims=True)
            m = jnp.where(jnp.isfinite(m), m, 0.0)
            total = jnp.sum(jnp.exp(log_a - m), axis=1, keepdims=True)
            lse = m + jnp.log(jnp.where(total > 0, total, 1.0))  # an all--inf row stays -inf, no NaN
            log_a = log_a - lse
        a = jnp.exp(log_a)
        return a, log_a

    return transition_fn


def build_factorial(
    chains: Sequence[ModelSpec],
    coupling: CouplingSpec | None = None,
    *,
    experimental: bool = False,
) -> FactorialSpace:
    """Compose `chains` into one joint `StateSpace` (Kronecker product of the per-chain `A_t`).

    Args:
        chains: single-chain specs (order 1, `jump` transition, `initial: uniform`); chain `c`'s parameters
            appear in the joint `theta` under the prefix `"c{c}."`.
        coupling: `None` for independent chains, or a design-only `CouplingSpec`.
        experimental: must be `True`; the module is a spike, not a supported feature.

    Raises:
        ConfigError: `experimental` is not set, fewer than two chains, or an unsupported chain.
    """
    from hsmmkit.statespace.builders import build_expanded

    if not experimental:
        raise ConfigError("statespace.factorial is experimental: pass experimental=True", path="factorial")
    chains_t = tuple(chains)
    if len(chains_t) < 2:
        raise ConfigError("a factorial model needs at least two chains", path="factorial.chains")
    for c, ch in enumerate(chains_t):
        if ch.order != 1 or ch.transition.mode != "jump" or ch.initial != "uniform":
            raise ConfigError(
                "factorial spike supports order-1 jump chains with a uniform initial distribution",
                path=f"factorial.chains[{c}]",
            )
        if any(ch.duration.n_phases(k) != 1 for k in range(ch.n_states)):
            raise ConfigError(
                "factorial spike supports single-phase chains (r = 1)",
                path=f"factorial.chains[{c}]",
            )

    spaces = tuple(build_expanded(ch) for ch in chains_t)
    sizes = tuple(sp.n_expanded for sp in spaces)
    n_joint = int(np.prod(sizes))
    n_regimes = tuple(ch.n_states for ch in chains_t)
    n_joint_regimes = int(np.prod(n_regimes))

    combos = list(itertools.product(*[range(s) for s in sizes]))  # chain 0 slowest
    reg_chain = np.array(
        [[int(spaces[c].index.regime_of[s]) for c, s in enumerate(combo)] for combo in combos], dtype=np.int64
    )
    phase_chain = np.array(
        [[int(spaces[c].index.phase_of[s]) for c, s in enumerate(combo)] for combo in combos], dtype=np.int64
    )
    regime_of = np.ravel_multi_index(tuple(reg_chain.T), n_regimes)
    phase_of = np.where((phase_chain == 1).all(axis=1), 1, 2)  # joint phase 1 <=> every chain at phase 1

    index = StateIndex(
        regime_of=jnp.asarray(regime_of),
        phase_of=jnp.asarray(phase_of),
        memory_of=jnp.zeros((n_joint, 0), dtype=jnp.int32),
        collapse=jnp.asarray(np.eye(n_joint_regimes)[regime_of]),
        row_of_state=jnp.asarray(regime_of),
        n_expanded=n_joint,
        n_states=n_joint_regimes,
        order=1,
    )
    space = StateSpace(
        index=index,
        layout=build_memory_layout(n_joint_regimes, 1),
        n_expanded=n_joint,
        n_rates=n_joint_regimes,
        rate_index_of=jnp.asarray(regime_of),
        entry_state_of=jnp.full((n_joint_regimes, n_joint_regimes), -1),
        phase1_idx=jnp.asarray(np.flatnonzero(phase_of == 1)),
        next_phase_state=jnp.full((n_joint,), -1),
        is_last_phase=jnp.ones((n_joint,), dtype=bool),
        exit_targets=jnp.full((n_joint, n_joint_regimes), -1),
    )
    space = dataclasses.replace(space, transition_fn=_joint_log_transition(spaces, coupling))
    return FactorialSpace(space=space, chains=chains_t, regime_of_chain=reg_chain, chain_sizes=sizes)


def joint_model_spec(factorial: FactorialSpace) -> ModelSpec:
    """A `ModelSpec` the existing engine accepts for the joint space (emission = sum of the chains')."""
    chains = factorial.chains
    emission = JointEmission(
        families=tuple(ch.emission for ch in chains), chain_regimes=tuple(ch.n_states for ch in chains)
    )
    first = chains[0]
    return ModelSpec(
        n_states=int(np.prod([ch.n_states for ch in chains])),
        n_series=first.n_series,
        emission=emission,  # type: ignore[arg-type]
        duration=first.duration,
        transition=first.transition,
        order=1,
        duration_conditioning="current",
        engine="expanded",
        d_max=first.d_max,
        initial="uniform",
        param_specs=(),
    )
