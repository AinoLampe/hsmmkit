"""The MAP/MLE objective the optimizers minimize (E5-T01).

Concept refs: §4.8 Objective, §6.3 item 2, §6.4.3. With `theta = manager.unflatten(u)`:

    J(u) = -loglik(theta) - log p(theta) - log|det d theta / d u|      (use_priors=True, MAP)
    J(u) = -loglik(theta)                                              (use_priors=False, MLE)

`log p` is `manager.log_prior(theta)` plus the transition model's `log_prior_extra` (the Dirichlet prior on
`pi` when covariate logits are configured, E3-T05). `log|det|` is the change of variables from the
constrained `theta` to the unconstrained `u`, so the MAP estimate is the mode in `theta`-space's density.

`data` is a pytree *argument* of the jitted functions (not static), so folds with equal shapes reuse one
compilation. Everything in `data` that is only static bookkeeping (`times`, `prep_stats`, `snapshot_ids`;
none of it reaches the engines) is replaced by a constant before the call, otherwise every fold's different
timestamps or standardization statistics would look like a new static value and force a retrace.
"""

from __future__ import annotations

import dataclasses
import warnings
from functools import partial
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.data.contracts import PreparedDataset, PrepStats
from hsmmkit.utils.clock import is_irregular
from hsmmkit.utils.errors import NumericalError
from hsmmkit.utils.typing import Array

if TYPE_CHECKING:
    from hsmmkit.data.batch import PaddedBatch
    from hsmmkit.engine.base import InferenceEngine
    from hsmmkit.engine.batch import BatchedEngine
    from hsmmkit.model.params import ParamManager, Params
    from hsmmkit.model.spec import ModelSpec
    from hsmmkit.statespace.index import StateSpace

    type Data = PreparedDataset | PaddedBatch  # one sequence, or a padded batch of sequences (E12-T02)

_CANONICAL_STATS = PrepStats(
    cov_mean=np.zeros(0),
    cov_std=np.ones(0),
    train_window=("", ""),
    cov_names=(),
    transforms={},
    lags={},
)


def _canonical(data: Data) -> Data:
    """`data` with the static bookkeeping fields replaced by constants (they never reach an engine)."""
    if not isinstance(data, PreparedDataset):  # a PaddedBatch carries no such bookkeeping
        return data
    times = np.zeros(data.observations.shape[0], dtype="datetime64[ns]")
    return dataclasses.replace(data, times=times, prep_stats=_CANONICAL_STATS, snapshot_ids=())


def canonical_data(data: PreparedDataset) -> PreparedDataset:
    """`data` for direct engine calls that should share a compilation across datasets of one shape.

    `times`, `prep_stats` and `snapshot_ids` sit in the dataset's pytree metadata, which jit compares by
    value: every walk-forward fold would otherwise retrace the engine functions (see `Objective`).
    """
    out = _canonical(data)
    assert isinstance(out, PreparedDataset)
    return out


def _log_prior(theta: Params, spec: ModelSpec, space: StateSpace, manager: ParamManager) -> Array:
    """`manager.log_prior` plus the transition model's extra prior (0 when it defines none)."""
    total = manager.log_prior(theta)
    extra = getattr(spec.transition, "log_prior_extra", None)
    if extra is not None:
        total = total + extra(theta, space.layout)
    for family in (spec.emission, spec.duration):  # optional joint priors of a family's own parameters
        joint = getattr(family, "log_prior_extra", None)
        if joint is not None:
            total = total + joint(theta)
    return total


def _objective(
    u: Array,
    space: StateSpace,
    data: Data,
    spec: ModelSpec,
    engine: InferenceEngine,
    manager: ParamManager,
    use_priors: bool,
) -> Array:
    theta = manager.unflatten(u)
    value = -engine.loglik(theta, spec, space, data)  # type: ignore[arg-type]  # a BatchedEngine takes a PaddedBatch
    if use_priors:
        value = value - _log_prior(theta, spec, space, manager) - manager.log_det_jac(u)
    return value


_STATIC = ("spec", "engine", "manager", "use_priors")


@partial(jax.jit, static_argnames=_STATIC)
def _value(
    u: Array,
    space: StateSpace,
    data: Data,
    spec: ModelSpec,
    engine: InferenceEngine,
    manager: ParamManager,
    use_priors: bool,
) -> Array:
    return _objective(u, space, data, spec, engine, manager, use_priors)


@partial(jax.jit, static_argnames=_STATIC)
def _value_and_grad(
    u: Array,
    space: StateSpace,
    data: Data,
    spec: ModelSpec,
    engine: InferenceEngine,
    manager: ParamManager,
    use_priors: bool,
) -> tuple[Array, Array]:
    return jax.value_and_grad(_objective)(u, space, data, spec, engine, manager, use_priors)


@partial(jax.jit, static_argnames=("spec", "engine", "manager"))
def _loglik(
    u: Array,
    space: StateSpace,
    data: Data,
    spec: ModelSpec,
    engine: InferenceEngine,
    manager: ParamManager,
) -> Array:
    theta = manager.unflatten(u)
    return engine.loglik(theta, spec, space, data)  # type: ignore[arg-type]  # a BatchedEngine takes a PaddedBatch


class Objective(eqx.Module):
    """`J(u)` bound to a model, engine, parameter layout and dataset.

    Attributes:
        spec: Model structure (static).
        space: Expanded (or explicit) state space.
        engine: Inference engine providing `loglik` (static).
        manager: `ParamManager` mapping `u` to `theta` (static).
        data: The default dataset; `value`/`value_and_grad` accept another one of the same shapes.
        use_priors: MAP (priors and log-det-Jacobian) when `True`, MLE when `False` (static).
    """

    spec: ModelSpec = eqx.field(static=True)
    space: StateSpace
    engine: InferenceEngine | BatchedEngine = eqx.field(static=True)
    manager: ParamManager = eqx.field(static=True)
    data: Data
    use_priors: bool = eqx.field(static=True)

    @property
    def n_free(self) -> int:
        """`p`, the length of `u`."""
        return self.manager.n_free

    def value(self, u: Array, data: Data | None = None) -> Array:
        """`J(u)` (0-d), jitted; `data` overrides the bound dataset (same shapes reuse the compilation)."""
        d = _canonical(self.data if data is None else data)
        out: Array = _value(u, self.space, d, self.spec, self.engine, self.manager, self.use_priors)
        return out

    def value_and_grad(self, u: Array, data: Data | None = None) -> tuple[Array, Array]:
        """`(J(u), dJ/du)`, jitted. The caller checks finiteness with `check_finite`."""
        d = _canonical(self.data if data is None else data)
        out: tuple[Array, Array] = _value_and_grad(
            u, self.space, d, self.spec, self.engine, self.manager, self.use_priors
        )
        return out

    def loglik(self, u: Array, data: Data | None = None) -> Array:
        """`ell(theta(u))` alone (no priors/log-det-Jacobian), jitted with the same cache as `value`."""
        d = _canonical(self.data if data is None else data)
        out: Array = _loglik(u, self.space, d, self.spec, self.engine, self.manager)
        return out


def make_objective(
    spec: ModelSpec,
    space: StateSpace,
    engine: InferenceEngine | BatchedEngine,
    manager: ParamManager,
    data: Data,
    use_priors: bool,
) -> Objective:
    """Bind the pieces into an `Objective`.

    Warns once when `spec.engine == "explicit"` and `data` sits on an irregular grid (the explicit engine
    would warn on every call, but under `jit` its own check cannot see concrete `dt`).
    """
    if spec.engine == "explicit" and is_irregular(data.dt):
        warnings.warn(
            "explicit-duration engine assumes a regular grid: using nominal_dt for an irregular dt",
            UserWarning,
            stacklevel=2,
        )
    return Objective(spec=spec, space=space, engine=engine, manager=manager, data=data, use_priors=use_priors)


def check_finite(value: Array, grad: Array, u: Array, names: list[str]) -> None:
    """Raise `NumericalError` unless `J` and every gradient entry are finite.

    The first non-finite gradient coordinate is named (via `names`, `ParamManager.names()`), with the
    value of `u` there; a non-finite value with a finite gradient is reported as such.

    Raises:
        NumericalError: with `path` set to the offending coordinate's name (or `"objective"`).
    """
    grad_np = np.asarray(grad)
    bad = np.flatnonzero(~np.isfinite(grad_np))
    if bad.size:
        i = int(bad[0])
        name = names[i] if i < len(names) else f"u[{i}]"
        raise NumericalError(
            f"non-finite gradient {grad_np[i]} at coordinate {name} (u = {float(np.asarray(u)[i])!r}); "
            f"{bad.size} of {grad_np.size} coordinates are non-finite",
            path=name,
        )
    if not bool(jnp.isfinite(value)):
        raise NumericalError(f"non-finite objective value {float(value)!r}", path="objective")
