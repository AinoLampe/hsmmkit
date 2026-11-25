"""State lattices from parameter grids, ordering constraints and label-order post-sorting (E3-T06).

Three identifiability helpers for label switching (concept §4.8):

* `build` / `validate_lattice`: a fixed-parameter state lattice from the Cartesian product of grids.
* `apply_ordering`: wires `constraint="increasing"` onto a named spec.
* `post_sort_states`: relabels a fitted `theta` so a chosen per-state parameter is ascending, permuting
  every per-state block (emission, duration, transition rows/columns incl. memory tuples) consistently.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from hsmmkit.config.schema import ParamCfg
from hsmmkit.model.params import Params
from hsmmkit.model.params.spec import ParamSpec
from hsmmkit.model.transition import _target_order, build_memory_layout
from hsmmkit.utils.errors import ConfigError

if TYPE_CHECKING:
    from hsmmkit.config.schema import ProjectConfig
    from hsmmkit.model.spec import ModelSpec

# Emission blocks that place a state on the observation scale; if all are `fixed` the states are
# identified by construction (a lattice). Coefficient blocks (`gamma`, `delta`) and `nu` do not count.
_SCALE_NAMES = ("emission.scale", "emission.sigma", "emission.s", "emission.chol")
_IDENTIFYING_EMISSION = frozenset(
    {"emission.mu", "emission.sigma", "emission.mean", "emission.scale", "emission.m", "emission.s"}
)


@dataclass(frozen=True, eq=False)
class LatticeStates:
    """Cartesian product of parameter grids: `values[name]` is `(n_states,)`, first grid slowest."""

    n_states: int
    values: dict[str, np.ndarray]

    def to_param_cfgs(self) -> dict[str, ParamCfg]:
        """`ParamCfg(mode="fixed", values=...)` per grid parameter, overriding `model.emission.params`."""
        return {
            name: ParamCfg(mode="fixed", values=[float(v) for v in vals])
            for name, vals in self.values.items()
        }


def _check_grids(grids: dict[str, list[float]], path: str) -> None:
    if not grids:
        raise ConfigError("lattice needs at least one parameter grid", path=path)
    for name, grid in grids.items():
        vals = np.asarray(grid, dtype=np.float64)
        if vals.size == 0:
            raise ConfigError(f"grid {name!r} is empty", path=f"{path}.{name}")
        if not np.all(np.isfinite(vals)):
            raise ConfigError(f"grid {name!r} has non-finite values", path=f"{path}.{name}")
        if np.unique(vals).size != vals.size:
            raise ConfigError(f"grid {name!r} has duplicate values (duplicate states)", path=f"{path}.{name}")


def build(grids: dict[str, list[float]]) -> LatticeStates:
    """Cartesian product of `grids` (e.g. `mu x sigma`): `n_states = prod(len(grid))`."""
    _check_grids(grids, "model.lattice.grids")
    names = list(grids)
    combos = list(itertools.product(*(grids[n] for n in names)))
    values = {n: np.asarray([c[k] for c in combos], dtype=np.float64) for k, n in enumerate(names)}
    return LatticeStates(n_states=len(combos), values=values)


def validate_lattice(cfg: ProjectConfig) -> None:
    """Raise `ConfigError` unless `model.n_states` equals the product of the lattice grid sizes."""
    lattice = cfg.model.lattice
    if lattice is None:
        return
    path = "model.lattice.grids"
    _check_grids(lattice.grids, path)
    n_expected = int(np.prod([len(g) for g in lattice.grids.values()]))
    if n_expected != cfg.model.n_states:
        raise ConfigError(
            f"model.n_states={cfg.model.n_states} but the lattice grids give {n_expected} states", path=path
        )


def apply_ordering(specs: tuple[ParamSpec, ...], name: str | None = None) -> tuple[ParamSpec, ...]:
    """Return `specs` with `constraint="increasing"` set on the spec called `name`.

    `name=None` picks the family's scale parameter (`emission.scale`, `emission.sigma` or `emission.s`).

    The `increasing` bijector acts on a 1-D `(N,)` cell, so an `(N, 1)` spec is squeezed to `(N,)`
    (single series); `(N, M > 1)` is rejected rather than silently sharing one ordering across series.
    """
    if name is None:
        name = next((n for n in _SCALE_NAMES if any(s.name == n for s in specs)), _SCALE_NAMES[0])
    out: list[ParamSpec] = []
    found = False
    for spec in specs:
        if spec.name != name:
            out.append(spec)
            continue
        found = True
        shape, value, init = spec.shape, spec.value, spec.init
        if len(shape) == 2:
            if shape[1] != 1:
                raise ConfigError(
                    f"constraint='increasing' is only supported for a single series (shape {shape})",
                    path=name,
                )
            shape = (shape[0],)
            value = None if value is None else np.asarray(value)[:, 0]
            init = None if init is None else np.asarray(init)[:, 0]
        elif len(shape) != 1:
            raise ConfigError(f"cannot order a spec of shape {shape}", path=name)
        out.append(replace(spec, shape=shape, value=value, init=init, constraint="increasing"))
    if not found:
        raise ConfigError(f"no parameter spec named {name!r}", path=name)
    return tuple(out)


def has_identifiability_guard(spec: ModelSpec) -> bool:
    """`True` if some per-state spec is `increasing`, or every location/scale emission spec is `fixed`."""
    if any(s.constraint == "increasing" for s in spec.param_specs):
        return True
    identifying = [s for s in spec.param_specs if s.name in _IDENTIFYING_EMISSION]
    if not identifying:
        return False
    return all(
        all(m == "fixed" for m in ((s.mode,) if isinstance(s.mode, str) else s.mode)) for s in identifying
    )


def _pair_rate_permutation(n_states: int, perm: np.ndarray) -> np.ndarray:
    """`(N(N-1),)`: old index of the pair `(perm[a], perm[b])` for each new pair `(a, b)` (`a != b`)."""
    pairs = [(i, j) for i in range(n_states) for j in range(n_states) if i != j]
    pair_to_idx = {p: k for k, p in enumerate(pairs)}
    return np.asarray([pair_to_idx[(int(perm[a]), int(perm[b]))] for a, b in pairs], dtype=np.int64)


def _transition_gather(n_states: int, order: int, perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(old_row (R,), pos (R, N-1))` mapping a relabelled row/target slot to its old-label slot.

    New row `r'` (memory tuple `u` in new labels) is the old row with tuple `perm[u]`; its `k`-th target
    (ascending in new labels) sits at position `pos[r', k]` of that old row's ascending target list.
    """
    layout = build_memory_layout(n_states, order)
    row_of = {tuple(int(x) for x in t): r for r, t in enumerate(layout.tuples)}
    old_row = np.asarray([row_of[tuple(int(perm[x]) for x in t)] for t in layout.tuples], dtype=np.int64)
    targets = _target_order(layout.current_of, n_states)  # (R, N-1), new labels for the new row
    old_labels = perm[targets]  # (R, N-1)
    old_targets = targets[old_row]  # (R, N-1): old row's ascending targets (old labels)
    pos = np.argmax(old_targets[:, :, None] == old_labels[:, None, :], axis=1)  # (R, N-1)
    return old_row, pos


def _relabel_transition(theta: Params, n_states: int, order: int, perm: np.ndarray) -> dict[str, jnp.ndarray]:
    old_row, pos = _transition_gather(n_states, order, perm)
    rows = jnp.asarray(old_row)[:, None]
    cols = jnp.asarray(pos)
    out: dict[str, jnp.ndarray] = {}
    if "transition.pi_rows" in theta:
        out["transition.pi_rows"] = jnp.asarray(theta["transition.pi_rows"])[rows, cols]
    if "transition.a" in theta:
        # Logits are defined relative to the reference target (smallest `j != current`), whose logit is 0.
        # Relabelling changes the reference: gather the full logit vectors, then re-subtract the new one.
        a = jnp.asarray(theta["transition.a"])  # (R, N-2)
        full_a = jnp.concatenate([jnp.zeros((a.shape[0], 1)), a], axis=1)  # (R, N-1)
        new_a = full_a[rows, cols]
        out["transition.a"] = (new_a - new_a[:, :1])[:, 1:]
        if "transition.b" in theta:
            b = jnp.asarray(theta["transition.b"])  # (R, N-2, K)
            full_b = jnp.concatenate([jnp.zeros((b.shape[0], 1, b.shape[2])), b], axis=1)  # (R, N-1, K)
            new_b = full_b[rows, cols]
            out["transition.b"] = (new_b - new_b[:, :1])[:, 1:]
    return out


def post_sort_states(theta: Params, spec: ModelSpec, by: str | None = None) -> tuple[Params, np.ndarray]:
    """Relabel states so `theta[by]` (first series if `(N, M)`) is ascending.

    Returns `(theta_sorted, perm)` with `perm[new] = old` (stable `argsort`): new state `i` is old state
    `perm[i]`. Emission and duration blocks are gathered along the state axis (pair-conditioned duration
    rates by the `(previous, current)` pair), transition blocks along memory-tuple rows and target
    columns, so the model is unchanged up to labels (`jump_probs` is invariant). The caller recomputes
    `u` from the sorted `theta` (`ParamManager.flatten`).
    """
    if by is None:
        by = next((n for n in _SCALE_NAMES if n in theta), _SCALE_NAMES[0])
    if by not in theta:
        raise ConfigError(f"cannot sort by {by!r}: not in theta", path=by)
    n = spec.n_states
    key = np.asarray(theta[by])
    key = key if key.ndim == 1 else key[:, 0]
    perm = np.argsort(key, kind="stable").astype(np.int64)

    sorted_theta: dict[str, jnp.ndarray] = {}
    transition_done = False
    for name, value in theta.items():
        arr = jnp.asarray(value)
        if name.startswith("emission."):
            sorted_theta[name] = arr[jnp.asarray(perm)]
        elif name == "duration.rate" and spec.duration_conditioning == "pair":
            sorted_theta[name] = arr[jnp.asarray(_pair_rate_permutation(n, perm))]
        elif name.startswith("duration."):
            sorted_theta[name] = arr[jnp.asarray(perm)]
        elif name.startswith("transition."):
            if not transition_done:
                sorted_theta.update(_relabel_transition(theta, n, spec.order, perm))
                transition_done = True
        elif name == "initial.logits":
            if arr.shape[0] != n:
                raise ConfigError(
                    "post-sorting a free initial distribution over expanded states is not supported "
                    f"(shape {arr.shape}, n_states={n})",
                    path=name,
                )
            sorted_theta[name] = arr[jnp.asarray(perm)]
        else:
            raise ConfigError(f"don't know how to relabel parameter {name!r}", path=name)
    return sorted_theta, perm
