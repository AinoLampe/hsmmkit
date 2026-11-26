"""`ModelSpec`: structure only, no data (E1-T05)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx

if TYPE_CHECKING:
    from hsmmkit.model.duration.base import DurationFamily
    from hsmmkit.model.emission.base import EmissionFamily
    from hsmmkit.model.params.spec import ParamSpec
    from hsmmkit.model.transition import TransitionModel


class ModelSpec(eqx.Module):
    """The full model structure: family instances, state-space shape, and the union of param specs.

    Every field is static (`ModelSpec` carries no array data of its own; all array-valued
    parameters live in `Params`/`u`, managed separately by `ParamManager`).
    """

    n_states: int = eqx.field(static=True)
    n_series: int = eqx.field(static=True)
    emission: EmissionFamily = eqx.field(static=True)
    duration: DurationFamily = eqx.field(static=True)
    transition: TransitionModel = eqx.field(static=True)
    order: int = eqx.field(static=True)
    duration_conditioning: Literal["current", "pair"] = eqx.field(static=True)
    engine: Literal["expanded", "explicit"] = eqx.field(static=True)
    d_max: int = eqx.field(static=True)
    initial: Literal["uniform", "stationary", "free"] = eqx.field(static=True)
    param_specs: tuple[ParamSpec, ...] = eqx.field(static=True)
    duration_dependent_jumps: bool = eqx.field(static=True, default=False)
    """`pi_ij` depends on the duration of the segment being left (E12-T05, explicit engine only)."""
    left_censored: bool = eqx.field(static=True, default=False)
    """The sequence starts inside a running segment (E12-T04): residual-life first segment (explicit)."""
    memory_mode: Literal["segments", "steps"] = eqx.field(static=True, default="segments")
    """`segments`: the segment memory of Part I (`transition.order`); `steps`: time-step tuples with repeats
    `(s_{t-p}, ..., s_t)` for an emission that depends on the regimes of the previous steps (E11-T05)."""
