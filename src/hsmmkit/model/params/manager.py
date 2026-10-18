"""`ParamManager`: maps the flat unconstrained vector `u` to the constrained pytree `theta`.

`ParamManager` is a plain (non-pytree) Python object built once from a tuple of `ParamSpec`s; it
is meant to be captured by closure inside jitted functions (`unflatten`/`flatten`/`log_det_jac`/
`log_prior` are ordinary methods that trace to JAX ops parameterized by the manager's static,
Python-level layout plan -- there is no need to register it as a pytree or make it hashable for
`jax.jit` to work, since it is never itself passed as a traced argument).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from regimekit.model.params.bijectors import (
    Bijector,
    IdentityBijector,
    IncreasingBijector,
    IntervalBijector,
    NuGt2Bijector,
    PmfBijector,
    PositiveBijector,
    SimplexRefBijector,
)
from regimekit.model.params.spec import Mode, ParamSpec
from regimekit.utils.errors import ConfigError
from regimekit.utils.typing import Array

Params = dict[str, Array]

_FIXED: Mode = "fixed"
_TIED: Mode = "tied"


def _base_bijector(spec: ParamSpec) -> Bijector:
    if spec.domain == "real":
        return IdentityBijector()
    if spec.domain == "positive":
        return PositiveBijector()
    if spec.domain == "interval":
        if spec.bounds is None:
            raise ConfigError("domain='interval' requires bounds", path=spec.name)
        lo, hi = spec.bounds
        return IntervalBijector(lo=lo, hi=hi)
    if spec.domain == "simplex_ref":
        return SimplexRefBijector()
    if spec.domain == "nu_gt_2":
        return NuGt2Bijector()
    if spec.domain == "pmf":
        return PmfBijector()
    raise ConfigError(f"unknown domain {spec.domain!r}", path=spec.name)


def _free_count(domain: str, cell_size: int) -> int:
    return cell_size - 1 if domain in ("simplex_ref", "pmf") else cell_size


@dataclass
class _Coord:
    """One allocated `u` slot group: a contiguous slice, its bijector, and its label."""

    start: int
    size: int
    bijector: Bijector
    cell_shape: tuple[int, ...]
    label: str

    def slice_of(self, u: Array) -> Array:
        return u[self.start : self.start + self.size]

    def forward(self, u: Array) -> Array:
        return self.bijector.forward(self.slice_of(u)).reshape(self.cell_shape)

    def inverse(self, theta_cell: Array) -> Array:
        return self.bijector.inverse(theta_cell.reshape(-1))


@dataclass
class _SpecPlan:
    spec: ParamSpec
    n0: int
    rest: tuple[int, ...]
    is_increasing: bool
    # non-increasing: one entry per state, in state order; increasing: a single entry
    state_modes: list[Mode] = field(default_factory=list)
    coords: dict[int, _Coord] = field(default_factory=dict)  # state_index -> its own/tied _Coord
    increasing_coord: _Coord | None = None
    prior_states: list[int] = field(default_factory=list)


class ParamManager:
    """Maps `u in R^p` to/from the constrained parameter pytree `theta` (`Params`)."""

    def __init__(self, specs: tuple[ParamSpec, ...]) -> None:
        """Build the flatten/unflatten layout plan from `specs`."""
        self._specs = specs
        self._plans: list[_SpecPlan] = []
        self._coord_order: list[_Coord] = []
        tie_coords: dict[str, _Coord] = {}
        offset = 0

        for spec in specs:
            if spec.constraint == "increasing":
                offset = self._plan_increasing(spec, offset)
            else:
                offset = self._plan_regular(spec, offset, tie_coords)

        self._n_free = offset

    # --- planning ------------------------------------------------------------------------

    def _plan_regular(self, spec: ParamSpec, offset: int, tie_coords: dict[str, _Coord]) -> int:
        n0 = spec.shape[0] if spec.shape else 1
        rest = spec.shape[1:] if spec.shape else ()
        cell_size = math.prod(rest) if rest else 1
        modes = (spec.mode,) * n0 if isinstance(spec.mode, str) else spec.mode
        if len(modes) != n0:
            raise ConfigError(f"mode tuple length {len(modes)} != leading shape {n0}", path=spec.name)
        if any(m == _FIXED for m in modes) and spec.value is None:
            raise ConfigError("mode='fixed' requires value", path=spec.name)
        if any(m == "bounded" for m in modes) and (spec.domain != "interval" or spec.bounds is None):
            raise ConfigError("mode='bounded' requires domain='interval' and bounds", path=spec.name)
        if any(m == "prior" for m in modes) and spec.prior is None:
            raise ConfigError("mode='prior' requires prior", path=spec.name)
        if any(m == "tied" for m in modes) and spec.tie_group is None:
            raise ConfigError("mode='tied' requires tie_group", path=spec.name)

        plan = _SpecPlan(spec=spec, n0=n0, rest=rest, is_increasing=False, state_modes=list(modes))
        bijector = _base_bijector(spec)
        free_count = _free_count(spec.domain, cell_size)

        for i, mode in enumerate(modes):
            if mode == _FIXED:
                continue
            if mode == "prior":
                plan.prior_states.append(i)
            if mode == _TIED:
                group = spec.tie_group
                assert group is not None
                if group not in tie_coords:
                    label = f"{spec.name}[tied:{group}]"
                    tie_coords[group] = _Coord(offset, free_count, bijector, rest, label)
                    self._coord_order.append(tie_coords[group])
                    offset += free_count
                elif tie_coords[group].cell_shape != rest or tie_coords[group].size != free_count:
                    raise ConfigError(
                        f"tie_group {group!r} members have incompatible domain/cell shape "
                        f"({tie_coords[group].cell_shape} vs {rest})",
                        path=spec.name,
                    )
                plan.coords[i] = tie_coords[group]
            else:
                label = f"{spec.name}[{i}]"
                coord = _Coord(offset, free_count, bijector, rest, label)
                self._coord_order.append(coord)
                offset += free_count
                plan.coords[i] = coord

        self._plans.append(plan)
        return offset

    def _plan_increasing(self, spec: ParamSpec, offset: int) -> int:
        if len(spec.shape) != 1:
            raise ConfigError("constraint='increasing' requires shape=(N,)", path=spec.name)
        if not isinstance(spec.mode, str):
            raise ConfigError(
                "constraint='increasing' requires one uniform mode, not per-state", path=spec.name
            )
        if spec.mode == _FIXED or spec.mode == _TIED:
            raise ConfigError(
                "constraint='increasing' cannot be combined with mode='fixed'/'tied'", path=spec.name
            )
        if spec.mode == "prior" and spec.prior is None:
            raise ConfigError("mode='prior' requires prior", path=spec.name)

        n0 = spec.shape[0]
        base = _base_bijector(spec)
        bijector = IncreasingBijector(base=base)
        coord = _Coord(offset, n0, bijector, (n0,), f"{spec.name}[increasing]")
        self._coord_order.append(coord)

        plan = _SpecPlan(
            spec=spec,
            n0=n0,
            rest=(),
            is_increasing=True,
            state_modes=[spec.mode] * n0,
            increasing_coord=coord,
            prior_states=list(range(n0)) if spec.mode == "prior" else [],
        )
        self._plans.append(plan)
        return offset + n0

    # --- public API ------------------------------------------------------------------------

    @property
    def n_free(self) -> int:
        """`p`: the number of entries `u` carries."""
        return self._n_free

    def names(self) -> list[str]:
        """One label per `u` coordinate group, in flattening order."""
        labels = []
        for coord in self._coord_order:
            if coord.size == 1:
                labels.append(coord.label)
            else:
                labels.extend(f"{coord.label}:{k}" for k in range(coord.size))
        return labels

    def unflatten(self, u: Array) -> Params:
        """`u -> theta` (full shapes, dict keyed by spec name)."""
        theta: Params = {}
        for plan in self._plans:
            if plan.is_increasing:
                assert plan.increasing_coord is not None
                theta[plan.spec.name] = plan.increasing_coord.forward(u)
                continue
            cells = []
            for i in range(plan.n0):
                if plan.state_modes[i] == _FIXED:
                    assert plan.spec.value is not None
                    cells.append(jnp.asarray(plan.spec.value[i] if plan.spec.shape else plan.spec.value))
                else:
                    cells.append(plan.coords[i].forward(u))
            theta[plan.spec.name] = jnp.stack(cells, axis=0) if plan.spec.shape else jnp.asarray(cells[0])
        return theta

    def flatten(self, theta: Params) -> Array:
        """`theta -> u` (only the free/tied/bounded/prior coordinates)."""
        u = jnp.zeros((self._n_free,))
        for plan in self._plans:
            value = theta[plan.spec.name]
            if plan.is_increasing:
                assert plan.increasing_coord is not None
                coord = plan.increasing_coord
                u = u.at[coord.start : coord.start + coord.size].set(coord.inverse(value))
                continue
            seen_tied: set[int] = set()
            for i in range(plan.n0):
                if plan.state_modes[i] == _FIXED:
                    continue
                coord = plan.coords[i]
                if id(coord) in seen_tied:
                    continue
                seen_tied.add(id(coord))
                cell = value[i] if plan.spec.shape else value
                u = u.at[coord.start : coord.start + coord.size].set(coord.inverse(cell))
        return u

    def log_det_jac(self, u: Array) -> Array:
        """`sum(log |det d(theta)/d(u)|)` over every distinct coordinate group."""
        total = jnp.zeros(())
        for coord in self._coord_order:
            total = total + coord.bijector.log_det_jac(coord.slice_of(u))
        return total

    def log_prior(self, theta: Params) -> Array:
        """`sum(prior.log_prob(theta))` over every spec/state with `mode == 'prior'`."""
        total = jnp.zeros(())
        for plan in self._plans:
            if not plan.prior_states:
                continue
            assert plan.spec.prior is not None
            value = theta[plan.spec.name]
            selected = value[jnp.asarray(plan.prior_states)] if plan.spec.shape else value
            total = total + plan.spec.prior.log_prob(selected)
        return total

    def init_u(self) -> Array:
        """`flatten` of the specs' `init` (or `value`, where fixed) values."""
        theta0: Params = {}
        for plan in self._plans:
            spec = plan.spec
            if spec.shape:
                cells = []
                for i in range(plan.n0):
                    if plan.state_modes[i] == _FIXED:
                        assert spec.value is not None
                        cells.append(np.asarray(spec.value[i]))
                    else:
                        assert spec.init is not None
                        cells.append(np.asarray(spec.init[i]))
                theta0[spec.name] = jnp.asarray(np.stack(cells, axis=0))
            else:
                source = spec.value if plan.state_modes[0] == _FIXED else spec.init
                assert source is not None
                theta0[spec.name] = jnp.asarray(source)
        return self.flatten(theta0)
